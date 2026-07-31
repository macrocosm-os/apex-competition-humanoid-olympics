"""humanoid_parkour gym_v1 REFEREE (the scorer sandbox, run at /app/referee.py).

Owns the physics: generates the round's courses from the platform-injected
master SEED, steps MuJoCo, streams observations to the player over /act, and
applies the termination + scoring gates. The player sandbox only ever sees
observation vectors — never the seed, the generator, or the course list.

raw_score = mean instance score over all courses (see env/scoring.py).
Per-course breakdowns go in metadata: hidden while the round is active,
revealed to miners when it completes.
"""

from __future__ import annotations

import json
import time

import numpy as np

from dataclasses import asdict

from gym_v1 import GameResult, Referee, RefereeContext
from gym_v1.client import PlayerClient, PlayerError
from gym_v1.referee import RESULT_PATH

from env import DIFFICULTIES, ParkourSim, generate_course, instance_score
from env.sim import InvalidAction

# Sized per HANDOFF.md §4: N = 3 x 40 = 120 course instances, 900-step episode
# cap (13.5 s sim time). The round input (CONFIG_JSON) can override.
DEFAULT_COURSES_PER_DIFFICULTY = 40
DEFAULT_MAX_STEPS = 900
DEFAULT_DEADLINE_MS = 500

# Everything a broken player can throw at us, all of it the SUBMISSION's fault.
#
# The vendored gym_v1 PlayerClient only converts urllib.error.URLError and TimeoutError into
# PlayerError. A player process that dies or raises mid-request surfaces as
# http.client.RemoteDisconnected (a ConnectionResetError, so an OSError) and a player that
# answers with a non-JSON body surfaces as json.JSONDecodeError — neither is a URLError, so
# both escape the client. If they escaped play_game too, the platform would score a bad
# submission as a REFEREE failure, which is the one misattribution the contract forbids.
PLAYER_FAULTS = (PlayerError, OSError, json.JSONDecodeError)

# A conforming action is ACT_DIM floats. Anything vastly longer is rejected before it reaches
# numpy, so a submission cannot spend the referee's memory budget on our behalf.
MAX_ACTION_LEN = 1024


class ParkourReferee(Referee):
    def play_game(self, ctx: RefereeContext, players: list[PlayerClient]) -> GameResult:
        start = time.monotonic()
        cfg = ctx.config or {}
        per_difficulty = int(cfg.get("courses_per_difficulty", DEFAULT_COURSES_PER_DIFFICULTY))
        max_steps = int(cfg.get("max_steps_per_episode", DEFAULT_MAX_STEPS))
        deadline_ms = int(cfg.get("deadline_ms", DEFAULT_DEADLINE_MS))
        player = players[0]

        # All courses derive from the per-round master seed: every submission
        # in the round runs exactly these instances, so identical resubmissions
        # score identically (no seed-fishing).
        instances = [
            (difficulty, int(seed))
            for d_idx, difficulty in enumerate(DIFFICULTIES)
            for seed in np.random.SeedSequence([ctx.seed, d_idx]).generate_state(per_difficulty)
        ]

        courses = []
        total = 0.0
        for i, (difficulty, course_seed) in enumerate(instances):
            sim = ParkourSim(generate_course(course_seed, difficulty))
            obs = sim.reset(seed=course_seed)
            # Nothing identifying the instance crosses into the player sandbox: the course
            # generator is public (env/course.py), so course_seed would let a submission
            # regenerate the WHOLE course — every hurdle, not just the 3 in the observation —
            # and the course index would tell it which difficulty tier it is on. The ONNX
            # wrapper discards both today, but the leak must not be one player-image edit away.
            player.reset(match_id=f"{ctx.match_id}:{i}", player_index=0, seed=0, config={})

            reason = None
            while reason is None:
                # Only the player call is inside the player-fault handler. sim.step() is OUR
                # code: if it raises, that is a referee bug and must surface as a referee
                # failure, not be laundered into a zero for the submission.
                try:
                    action = player.act(observation=obs.tolist(), deadline_ms=deadline_ms)
                except PLAYER_FAULTS:
                    reason = "player_error"  # unreachable / timed out / died / garbage response
                    break
                # A submission can otherwise OOM-kill the referee (1.5Gi) with a huge action
                # list and have the failure attributed to us. Oversized is invalid, not fatal.
                if isinstance(action, (list, tuple)) and len(action) > MAX_ACTION_LEN:
                    reason = "invalid_action"
                    break
                try:
                    result = sim.step(action, max_steps=max_steps)
                except (InvalidAction, TypeError):
                    reason = "invalid_action"  # NaN / wrong shape / non-numeric
                    break
                obs, reason = result.obs, result.terminal_reason

            score = instance_score(reason, sim.progress, sim.steps, max_steps)
            total += score
            courses.append(
                {
                    "difficulty": difficulty,
                    "terminal_reason": reason,
                    "progress": round(sim.progress, 4),
                    "steps": sim.steps,
                    "sim_time_s": round(sim.steps * 0.015, 2),
                    "score": round(score, 4),
                }
            )

        completed = sum(c["terminal_reason"] == "completed" for c in courses)
        raw = total / len(courses)
        return GameResult(
            raw_scores=[raw],
            winner=0 if raw > 0 else -1,
            terminal_reason="scored",
            steps=sum(c["steps"] for c in courses),
            metadata={
                "courses": courses,
                "num_courses": len(courses),
                "num_completed": completed,
                "eval_time_in_seconds": round(time.monotonic() - start, 1),
            },
        )

    def run(self) -> None:
        """Same as the toolkit's Referee.run(), except a player that never becomes ready is
        scored as a typed SUBMISSION failure instead of a referee failure.

        Why this override exists: gym_v1's Referee.run() calls wait_until_ready() BEFORE
        play_game(), so its PlayerError escapes at a point where no /data/result.json can be
        written — and a missing result.json is attributed to the referee. But a player that
        never reports ready is exactly what a malformed ONNX artifact looks like (see
        player/launch.py: a load failure serves is_ready() False rather than dying), which is
        the submission's fault and must come back to the miner as an explained zero.

        This is NOT papering over a referee bug: the scope is one specific PlayerError from the
        readiness wait. play_game() itself is left completely unguarded, so a genuine referee
        crash still produces no result.json and is still attributed to us.
        """
        ctx = RefereeContext.from_env()
        players = [PlayerClient(url) for url in ctx.player_urls]
        try:
            for p in players:
                p.wait_until_ready(self.readiness_timeout_s)
        except PlayerError as e:
            result = GameResult(
                raw_scores=[0.0],
                winner=-1,
                terminal_reason="submission_not_ready",
                steps=0,
                metadata={
                    "error": str(e),
                    "explanation": (
                        "The submission never became ready. Usually the ONNX artifact failed to "
                        "load or does not match the required interface: exactly one float32 "
                        f"input of shape [batch, {56}] and one float32 output of shape "
                        f"[batch, {17}], single file with weights embedded, <= 25 MB."
                    ),
                },
            )
        else:
            result = self.play_game(ctx, players)  # unguarded: a crash here is OUR failure

        RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
        RESULT_PATH.write_text(json.dumps(asdict(result)))


if __name__ == "__main__":
    ParkourReferee().run()
