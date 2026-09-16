"""Every barrier the meet can end an attempt on must be visible before it is hit.

Through 0.4.0 it was not. Hurdles are `walkable=False`, which keeps them out of the downward
terrain scan -- that mask is WORLD_GROUP only and has to stay that way, because `_ray_down` also
feeds the FALL_CLEARANCE gate. The seven forward channels were supposed to cover them, but they
were upward rays cast from `pelvis + 0.05` (1.643 m at reset), and six of the ten hurdles top out
between 1.35 m and 1.60 m. They returned nothing, on any of the 104 channels, for the whole first
half of the race.

Asserted at the OBSERVATION level rather than through a score, for the reason
`test_friction_reaches_contacts.py` gives about friction: a score cannot distinguish a barrier a
policy could not see from one it saw and failed to clear. Both read as a low number.
"""

from __future__ import annotations

import pathlib
import sys

import mujoco
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from env.course import HURDLE_HEIGHTS_M, PLINTH_TOP, build_event
from env.sim import OVERHEAD_N, SCAN_NX, SCAN_NY, OlympicsSim, instance_spec

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
    flat = float(_forward_at(sim, -2.0, pose).min())
    hurdles = [s for s in build_event("hurdles_100").surfaces if s.kind == "hurdle"]
    assert len(hurdles) == len(HURDLE_HEIGHTS_M)

    for hurdle, height in zip(hurdles, HURDLE_HEIGHTS_M, strict=True):
        seen_from = None
        for x in np.arange(hurdle.x - 6.0, hurdle.x, 0.05):
            if float(_forward_at(sim, float(x), pose).max()) > flat + 1e-6:
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
    """A barrier thinner than the channel spacing must not flicker in and out.

    Seven point rays over 4 m sample every 0.667 m; a hurdle is 0.24 m thick, so point sampling
    missed it about two thirds of the time and delivered a "visible" hurdle as a 0.2 m flash.
    """
    sim = _sim("hurdles_100")
    pose = _pose(sim)
    flat = float(_forward_at(sim, -2.0, pose).min())
    first = min(s.x for s in build_event("hurdles_100").surfaces if s.kind == "hurdle")

    xs = np.arange(first - 4.0, first - 0.5, 0.05)
    seen = [float(_forward_at(sim, float(x), pose).max()) > flat + 1e-6 for x in xs]
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

    best = max(float(_forward_at(sim, float(x), pose).max())
               for x in np.arange(bar.x - 4.0, bar.x, 0.05))
    pelvis = float(pose[0][2])
    assert abs((best + pelvis) - top) < 0.05, (
        f"the forward channels report {best + pelvis:.3f} m for a bar whose top is {top:.3f} m"
    )
    assert top > PLINTH_TOP, "the bar is not above the deck"


def test_clear_track_reads_like_the_ground_under_it():
    """No barrier ahead must be a value on the terrain scale, not a sentinel."""
    sim = _sim("sprint_100")
    pose = _pose(sim)
    obs = sim._obs()
    forward = _forward_at(sim, -2.0, pose)
    scan = obs[52:FORWARD0]
    assert len(forward) == OVERHEAD_N
    assert abs(float(forward.max()) - float(scan.max())) < 1e-3, (
        "an unobstructed forward channel disagrees with the terrain scan beneath it"
    )
