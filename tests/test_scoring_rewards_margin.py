"""Being better must always score better.

Through 0.5.1 the three field events clamped: long jump at 12 m, triple at 18 m, and high jump on
a fixed bar ladder that gave every clearing policy exactly 0.625. Half the meet carried no
information about who was better, so the field converged and a flat takeover threshold froze it.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from env.course import HIGH_JUMP_BARS_M
from env.scoring import REFERENCE_FRACTION, instance_score, margin_fraction

# event, terminal reason, metric key, entry threshold, reference performance
FIELD_EVENTS = [
    ("long_jump", "landed", "jump_distance_m", 6.0, 12.0),
    ("triple_jump", "landed", "jump_distance_m", 13.0, 18.0),
    ("high_jump", "cleared", "best_clearance_m", HIGH_JUMP_BARS_M[0], HIGH_JUMP_BARS_M[-1]),
]


def score(event: str, reason: str, key: str, value: float) -> float:
    return instance_score(event, reason, 0.0, 0, 1, {key: value, "bar_height_m": 1.0})


@pytest.mark.parametrize("event,reason,key,floor,reference", FIELD_EVENTS)
def test_a_minimum_legal_result_earns_the_finish_band_only(event, reason, key, floor, reference):
    assert score(event, reason, key, floor) == pytest.approx(0.25)


@pytest.mark.parametrize("event,reason,key,floor,reference", FIELD_EVENTS)
def test_the_reference_is_not_the_ceiling(event, reason, key, floor, reference):
    """It used to be. A policy reaching the reference must still have somewhere to go."""
    at_reference = score(event, reason, key, reference)
    assert at_reference == pytest.approx(0.25 + 0.75 * REFERENCE_FRACTION)
    assert at_reference < 1.0


@pytest.mark.parametrize("event,reason,key,floor,reference", FIELD_EVENTS)
def test_every_further_metre_pays(event, reason, key, floor, reference):
    step = (reference - floor) / 20.0
    values = [floor + step * i for i in range(1, 60)]
    scores = [score(event, reason, key, v) for v in values]
    assert all(b > a for a, b in zip(scores, scores[1:])), event
    assert all(0.25 <= s <= 1.0 for s in scores), event


@pytest.mark.parametrize("event,reason,key,floor,reference", FIELD_EVENTS)
def test_nothing_exceeds_one(event, reason, key, floor, reference):
    assert score(event, reason, key, reference * 1000) <= 1.0


def test_high_jump_scores_the_height_reached_not_the_bar_selected():
    """The ladder is fixed, so a bar-indexed score was identical for everyone who cleared."""
    low_bar_big_leap = instance_score("high_jump", "cleared", 0.0, 0, 1,
                                      {"bar_height_m": 1.00, "best_clearance_m": 1.45})
    high_bar_tight = instance_score("high_jump", "cleared", 0.0, 0, 1,
                                    {"bar_height_m": 1.30, "best_clearance_m": 1.38})
    assert low_bar_big_leap > high_bar_tight


def test_a_failed_attempt_stays_below_the_finish_band():
    assert score("long_jump", "foul", "jump_distance_m", 30.0) < 0.25
    assert score("high_jump", "bar_missed", "best_clearance_m", 2.0) < 0.25


def test_margin_fraction_is_bounded_and_monotone():
    assert margin_fraction(1.0, 1.0, 2.0) == pytest.approx(0.0)
    assert margin_fraction(2.0, 1.0, 2.0) == pytest.approx(REFERENCE_FRACTION)
    assert margin_fraction(0.0, 1.0, 2.0) == pytest.approx(0.0)  # below the floor never negative
    assert 0.0 <= margin_fraction(1e6, 1.0, 2.0) <= 1.0
