"""The drawn hurdle layout has to reach whatever rebuilds the course.

Placement is drawn per attempt from the episode seed, and the seed reaches no miner-visible
surface. So anything rebuilding the scene from a history file -- the front end, `tools/replay.py`
-- had no way to know where the barriers were, and drew a different course from the one scored.

The layout now rides on the attempt's `challenge`, which `result.json` reports and the history
record already stores, so every consumer gets it without the seed.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from env.course import (HURDLE_HEIGHTS_M, build_event, hurdle_challenge, hurdle_layout,
                        hurdle_layout_from_challenge)
from env.history import DEFAULT_STRIDE, InstanceRecorder
from env.sim import OlympicsSim, instance_spec

SEEDS = (1, 20, 757959679)


def test_the_attempt_carries_its_own_layout():
    for seed in SEEDS:
        for attempt in range(4):
            layout = hurdle_layout_from_challenge(instance_spec("hurdles_100", attempt, seed).challenge)
            assert layout is not None and len(layout) == len(HURDLE_HEIGHTS_M)


def test_only_hurdles_carry_one():
    for event in ("sprint_100", "sprint_400", "high_jump", "long_jump", "triple_jump",
                  "race_walk_200"):
        challenge = instance_spec(event, 0, 1).challenge
        assert not [k for k in challenge if k.startswith("hurdle_")], event


def test_the_recorded_layout_is_the_one_the_sim_built():
    """A recorded layout that disagreed with the scene would be worse than none."""
    for seed in SEEDS:
        params = instance_spec("hurdles_100", 0, seed)
        sim = OlympicsSim(params)
        built = sorted((round(s.x, 3), round(s.hz * 2, 2))
                       for s in sim.layout.surfaces if s.kind == "hurdle")
        recorded = sorted((round(x, 3), h)
                          for x, h in hurdle_layout_from_challenge(params.challenge))
        assert built == recorded


def test_rebuilding_from_the_challenge_alone_reproduces_the_course():
    """This is the path replay and the front end take: no seed, just the recorded conditions."""
    params = instance_spec("hurdles_100", 2, 20)
    rebuilt = build_event("hurdles_100", params.challenge)
    scored = OlympicsSim(params).layout
    assert rebuilt == scored


def test_the_history_record_carries_it():
    params = instance_spec("hurdles_100", 0, 757959679)
    sim = OlympicsSim(params)
    sim.reset()
    recorder = InstanceRecorder(0, sim, DEFAULT_STRIDE)
    for _ in range(3):
        sim.step(np.zeros(12, np.float64), max_steps=3)
        recorder.capture(sim, np.zeros(12, np.float64))
    record = recorder.record(sim, {"terminal_reason": "timeout", "score": 0.0},
                             match_id="m:0", num_instances=1)
    layout = hurdle_layout_from_challenge(record["conditions"]["challenge"])
    assert layout is not None
    assert build_event("hurdles_100", record["conditions"]["challenge"]) == sim.layout


def test_the_layout_does_not_leak_the_seed():
    """Geometry is a reported condition, like friction and wind. The seed is not."""
    params = instance_spec("hurdles_100", 0, 757959679)
    text = repr(params.challenge)
    assert str(757959679) not in text
    assert str(params.seed) not in text
    for key in params.challenge:
        assert "seed" not in key


@pytest.mark.parametrize("seed", SEEDS)
def test_a_partial_challenge_falls_back_to_drawing(seed):
    """A record written before this change has no layout; it must still build a legal course."""
    partial = hurdle_challenge(hurdle_layout(seed))
    partial.pop("hurdle_x_7")
    assert hurdle_layout_from_challenge(partial) is None
    bars = [s for s in build_event("hurdles_100", partial, seed).surfaces if s.kind == "hurdle"]
    assert len(bars) == len(HURDLE_HEIGHTS_M)
