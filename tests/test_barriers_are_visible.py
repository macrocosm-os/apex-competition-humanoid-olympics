"""Every barrier the meet can end an attempt on must be visible before it is hit.

Through 0.4.0 it was not: hurdles are `walkable=False` so the downward scan skips them (that mask
feeds the FALL_CLEARANCE gate and must stay as it is), and the forward channels were upward rays
from `pelvis + 0.05`, above six of the ten hurdle tops. `hurdles_100` was byte-identical to
`sprint_100` for the first 60.9 m.

Asserted on the observation, not a score: a score cannot tell a barrier a policy could not see
from one it saw and failed to clear. `test_clear_ground_still_reads_2` pins the drop-in half --
reporting a terrain height there instead took a real 0.4.0 submission from 0.782833 to 0.000370.
"""
from __future__ import annotations

import pathlib
import sys

import mujoco
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from env.course import HURDLE_HEIGHTS_M, PLINTH_TOP, build_event
from env.sim import OVERHEAD_N, SCAN_CLIP, SCAN_NX, SCAN_NY, OlympicsSim, instance_spec

ROUND_SEED = 757959679
FORWARD0 = 52 + SCAN_NX * SCAN_NY
# The nearest forward bin starts under the robot, the farthest ends past 4 m. A barrier must be
# seen with enough room to do something about it: at the 4.25 m/s a 100 m finish demands, 2 m is
# most of a stride cycle.
MIN_WARNING_M = 2.0


def _sim(event: str) -> OlympicsSim:
    sim = OlympicsSim(instance_spec(event, 0, ROUND_SEED))
    sim.reset()
    return sim


def _forward_at(sim: OlympicsSim, x: float, pose: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """The seven forward channels with the robot posed identically in every event."""
    qpos, qvel = pose
    sim.data.qpos[:] = qpos
    sim.data.qvel[:] = qvel
    sim.data.qpos[0] = x
    mujoco.mj_forward(sim.model, sim.data)
    return sim._obs()[FORWARD0:]


def _pose(sim: OlympicsSim) -> tuple[np.ndarray, np.ndarray]:
    return sim.data.qpos.copy(), sim.data.qvel.copy()


def test_every_hurdle_is_seen_before_it_is_reached():
    """All ten, not just the four that happened to clear the old ray origin."""
    sim = _sim("hurdles_100")
    pose = _pose(sim)
    hurdles = [s for s in build_event("hurdles_100").surfaces if s.kind == "hurdle"]
    assert len(hurdles) == len(HURDLE_HEIGHTS_M)

    for hurdle, height in zip(hurdles, HURDLE_HEIGHTS_M, strict=True):
        seen_from = None
        for x in np.arange(hurdle.x - 6.0, hurdle.x, 0.05):
            if float(_forward_at(sim, float(x), pose).min()) < SCAN_CLIP - 1e-6:
                seen_from = float(x)
                break
        assert seen_from is not None, (
            f"the {height:.2f} m hurdle at x={hurdle.x:.1f} is invisible on every channel"
        )
        warning = hurdle.x - seen_from
        assert warning >= MIN_WARNING_M, (
            f"the {height:.2f} m hurdle at x={hurdle.x:.1f} appears only {warning:.2f} m ahead"
        )


def test_a_hurdle_stays_visible_while_it_is_approached():
    """A barrier thinner than the channel spacing must not flicker in and out."""
    sim = _sim("hurdles_100")
    pose = _pose(sim)
    first = min(s.x for s in build_event("hurdles_100").surfaces if s.kind == "hurdle")

    xs = np.arange(first - 4.0, first - 0.5, 0.05)
    seen = [float(_forward_at(sim, float(x), pose).min()) < SCAN_CLIP - 1e-6 for x in xs]
    assert all(seen), (
        f"the first hurdle drops out of the observation on {seen.count(False)} of {len(seen)} "
        "steps of the approach"
    )


def test_a_hurdles_race_does_not_look_like_a_sprint():
    """The two events share a start, a finish, and a lane. Only the barriers separate them."""
    hurdles, sprint = _sim("hurdles_100"), _sim("sprint_100")
    pose = _pose(sprint)
    first = min(s.x for s in build_event("hurdles_100").surfaces if s.kind == "hurdle")

    separated = None
    for x in np.arange(-2.0, first, 0.05):
        if float(np.abs(_forward_at(hurdles, float(x), pose)
                        - _forward_at(sprint, float(x), pose)).max()) > 1e-6:
            separated = float(x)
            break
    assert separated is not None and separated < first - MIN_WARNING_M, (
        "a hurdles race is indistinguishable from a sprint until the first barrier is upon it"
    )


def test_the_high_jump_bar_reports_its_height():
    """The bar is the one barrier a policy must pass OVER, so its top is the whole signal."""
    sim = _sim("high_jump")
    pose = _pose(sim)
    layout = build_event("high_jump", sim.params.challenge)
    bar = next(s for s in layout.surfaces if s.kind == "bar")
    top = bar.z + bar.hz

    lowest = min(float(_forward_at(sim, float(x), pose).min())
                 for x in np.arange(bar.x - 4.0, bar.x, 0.05))
    reported_height = SCAN_CLIP - lowest
    assert abs((PLINTH_TOP + reported_height) - top) < 0.05, (
        f"the forward channels imply a bar top of {PLINTH_TOP + reported_height:.3f} m, "
        f"but it is {top:.3f} m"
    )
    assert top > PLINTH_TOP, "the bar is not above the deck"


def test_clear_ground_still_reads_2():
    """No barrier ahead reads exactly what 0.4.0 reported on a miss, which a trained policy
    folded in as a bias."""
    for event in ("sprint_100", "sprint_400", "long_jump", "triple_jump"):
        sim = _sim(event)
        forward = _forward_at(sim, sim.layout.start_x, _pose(sim))
        assert len(forward) == OVERHEAD_N
        assert np.allclose(forward, SCAN_CLIP), (
            f"{event} reports {forward.tolist()} on clear ground, not {SCAN_CLIP}"
        )


def test_a_barrier_subtracts_its_height():
    """A hurdle reads 2.0 minus its height above the deck, so taller means smaller."""
    sim = _sim("hurdles_100")
    pose = _pose(sim)
    for hurdle, height in zip([s for s in build_event("hurdles_100").surfaces
                               if s.kind == "hurdle"], HURDLE_HEIGHTS_M, strict=True):
        lowest = min(float(_forward_at(sim, float(x), pose).min())
                     for x in np.arange(hurdle.x - 4.0, hurdle.x, 0.05))
        assert abs((SCAN_CLIP - lowest) - height) < 0.05, (
            f"the {height:.2f} m hurdle reports a height of {SCAN_CLIP - lowest:.3f} m"
        )
