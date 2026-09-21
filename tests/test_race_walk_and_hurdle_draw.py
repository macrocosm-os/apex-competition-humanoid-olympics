"""The two 0.7.0 changes: a contact-bound event, and hurdles you cannot memorise.

Race walk scores like a race but fouls on flight, so pace has to come from gait. Hurdle placement
and height order are drawn per round -- the SET of heights is fixed, so a round is a rearrangement
rather than a harder or easier meet, and position no longer predicts the next height.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from env.course import (EVENTS, HURDLE_FIRST_MIN_M, HURDLE_HEIGHTS_M, HURDLE_LAST_MAX_M,
                        HURDLE_MIN_GAP_M, build_event, hurdle_layout)
from env.scoring import RACE_EVENTS, instance_score
from env.sim import MAX_FLIGHT_STEPS, OlympicsSim, instance_spec

SEEDS = (0, 1, 7, 12345, 757959679)


def walk_sim() -> OlympicsSim:
    sim = OlympicsSim(instance_spec("race_walk_200", 0, 7))
    sim.reset()
    return sim


def fly(sim: OlympicsSim, steps: int, grounded: bool = False) -> None:
    sim._foot_contacts = lambda: {"left": {"track"}} if grounded else {}
    for _ in range(steps):
        sim._observe_race_walk(0.0)


# -- race walk ----------------------------------------------------------------------------------

def test_race_walk_is_in_the_meet_and_scores_as_a_race():
    assert "race_walk_200" in EVENTS
    assert "race_walk_200" in RACE_EVENTS


def test_flight_beyond_the_tolerance_fouls():
    sim = walk_sim()
    fly(sim, MAX_FLIGHT_STEPS)
    assert sim._event_reason is None, "the tolerance must absorb a short loss of contact"
    fly(sim, 1)
    assert sim._event_reason == "lost_contact"


def test_a_contact_resets_the_tolerance():
    sim = walk_sim()
    fly(sim, MAX_FLIGHT_STEPS)
    fly(sim, 1, grounded=True)
    fly(sim, MAX_FLIGHT_STEPS)
    assert sim._event_reason is None, "airborne steps must not accumulate across a contact"


def test_a_grounded_walker_reaching_the_finish_completes():
    sim = walk_sim()
    sim._foot_contacts = lambda: {"left": {"track"}}
    sim._observe_race_walk(sim.layout.finish)
    assert sim._event_reason == "completed"


def test_a_foul_scores_below_the_finish_band():
    """Flight cannot pay: the attempt ends where it left the ground."""
    fouled = instance_score("race_walk_200", "lost_contact", 0.05, 100, 3600, {})
    finished = instance_score("race_walk_200", "completed", 1.0, 3000, 3600, {})
    assert fouled < 0.25 <= finished


# -- hurdle draw --------------------------------------------------------------------------------

@pytest.mark.parametrize("seed", SEEDS)
def test_the_draw_is_a_rearrangement_not_a_difficulty_change(seed):
    layout = hurdle_layout(seed)
    assert len(layout) == len(HURDLE_HEIGHTS_M)
    assert sorted(h for _, h in layout) == sorted(HURDLE_HEIGHTS_M)


@pytest.mark.parametrize("seed", SEEDS)
def test_hurdles_stay_in_the_lane_window_and_keep_their_spacing(seed):
    xs = [x for x, _ in hurdle_layout(seed)]
    assert xs == sorted(xs)
    assert xs[0] >= HURDLE_FIRST_MIN_M - 1e-6
    assert xs[-1] <= HURDLE_LAST_MAX_M + 1e-6
    assert min(b - a for a, b in zip(xs, xs[1:])) >= HURDLE_MIN_GAP_M - 1e-6


def test_the_draw_is_reproducible_and_moves_between_rounds():
    assert hurdle_layout(7) == hurdle_layout(7)
    assert build_event("hurdles_100", seed=7) == build_event("hurdles_100", seed=7)
    assert len({tuple(hurdle_layout(s)) for s in range(40)}) > 1


def test_position_no_longer_predicts_height():
    """The old ladder rose monotonically, so knowing x was knowing the next height."""
    non_monotonic = 0
    for seed in range(40):
        heights = [h for _, h in hurdle_layout(seed)]
        if heights != sorted(heights):
            non_monotonic += 1
    assert non_monotonic > 35, f"only {non_monotonic}/40 draws broke the rising ladder"


def test_the_sim_uses_its_own_instance_draw():
    a = OlympicsSim(instance_spec("hurdles_100", 0, 1)).layout
    b = OlympicsSim(instance_spec("hurdles_100", 0, 2)).layout
    assert a != b, "two rounds must not present the same hurdles"
