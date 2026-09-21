"""The optional `event_type` input must be optional in both directions.

A two-input policy is the entire live leaderboard, so it has to keep loading and scoring exactly
as before. A three-input policy has to receive a correct one-hot, and the index order is a
contract between two files that cannot import each other: the referee sends the event NAME, and
`player/launch.py` turns it into a vector without `env/` in its image.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from env import EVENT_DIM, EVENTS, event_one_hot


def _load_player_module():
    """The player resolves its vendored `gym_v1` from its own directory, not the repo root."""
    sys.path.insert(0, str(ROOT / "player"))
    spec = importlib.util.spec_from_file_location("olympics_player", ROOT / "player/launch.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_player_and_the_env_agree_on_the_event_order():
    """A mismatch here is silent: every policy would read the wrong discipline."""
    player = _load_player_module()
    assert player.EVENT_ORDER == tuple(EVENTS), (
        "player/launch.py EVENT_ORDER has drifted from env.course.EVENTS"
    )
    assert player.EVENT_DIM == EVENT_DIM


def test_the_referee_sends_the_event_name_and_nothing_else():
    """Widening this config is how a round-identifying value would get in."""
    source = ROOT.joinpath("referee/referee.py").read_text()
    assert 'config={"event": sim.event}' in source, (
        "the referee no longer passes exactly the event name into player.reset"
    )


@pytest.mark.parametrize("event", EVENTS)
def test_reset_builds_the_one_hot_the_env_would(event):
    player = _load_player_module()
    instance = player.OlympicsPlayer()
    instance._event_width = EVENT_DIM          # a policy declaring the full-width input
    instance.reset(match_id="m:0", player_index=0, seed=0, config={"event": event})
    assert np.array_equal(instance._event, event_one_hot(event))
    assert instance._event.dtype == np.float32
    assert instance._event.shape == (1, EVENT_DIM)


@pytest.mark.parametrize("width", range(1, len(EVENTS) + 1))
def test_a_narrower_one_hot_keeps_the_indices_it_knows(width):
    """A policy built for a shorter meet must keep working: events are appended, never reordered."""
    player = _load_player_module()
    instance = player.OlympicsPlayer()
    instance._event_width = width
    for i, event in enumerate(EVENTS):
        instance.reset(match_id="m:0", player_index=0, seed=0, config={"event": event})
        assert instance._event.shape == (1, width)
        if i < width:
            assert instance._event[0, i] == 1.0 and instance._event.sum() == 1.0
        else:
            assert not instance._event.any(), "an event it predates must read as all-zero"


def test_an_unknown_or_absent_event_is_zeros_rather_than_a_fault():
    """A reset fault is scored against the submission, so it must not be our error to make."""
    player = _load_player_module()
    instance = player.OlympicsPlayer()
    instance._event_width = EVENT_DIM
    for config in ({}, {"event": "tug_of_war"}, {"event": None}):
        instance.reset(match_id="m:0", player_index=0, seed=0, config=config)
        assert not instance._event.any(), f"config {config!r} produced a non-zero one-hot"


def test_the_one_hot_is_a_one_hot():
    for event in EVENTS:
        vector = event_one_hot(event)
        assert vector.sum() == 1.0 and vector.max() == 1.0
    with pytest.raises(ValueError):
        event_one_hot("tug_of_war")


def test_both_arities_load_and_run():
    """The contract check is structural, so exercise it with real graphs."""
    torch = pytest.importorskip("torch", reason="ONNX export needs torch; release CI has it")
    import onnxruntime as ort

    sys.path.insert(0, str(ROOT / "tools"))
    spec = importlib.util.spec_from_file_location("make_test_policy",
                                                  ROOT / "tools/make_test_policy.py")
    maker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(maker)
    player = _load_player_module()

    for event_input in (False, True):
        out = pathlib.Path(torch.hub.get_dir()) / f"olympics_arity_{int(event_input)}.onnx"
        maker.build(out, seed=0, event_input=event_input)
        session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
        assert len(session.get_inputs()) == (3 if event_input else 2)

        instance = player.OlympicsPlayer.__new__(player.OlympicsPlayer)
        instance._session = session
        instance._names = [i.name for i in session.get_inputs()]
        instance._state = np.zeros((1, player.STATE_DIM), np.float32)
        instance._event = np.zeros((1, EVENT_DIM), np.float32)
        instance._match, instance._step = "m:0", 0
        instance.reset(match_id="m:0", player_index=0, seed=0, config={"event": "hurdles_100"})

        action = instance.act(observation=[0.0] * player.OBS_DIM, deadline_ms=500)
        assert len(action) == player.ACT_DIM
        assert all(np.isfinite(action))
