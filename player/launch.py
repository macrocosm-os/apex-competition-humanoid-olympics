"""humanoid_parkour gym_v1 PLAYER server (the image's `entrypoints.evaluate.command`).

The platform writes the miner's ONNX policy to /app/submission.onnx; this server loads it,
validates the interface, and serves /health /reset /act. There is no miner code in this sandbox —
the artifact is a pure ONNX graph (spec `artifact_type: onnx`), so validation is structural.

Contract the submission must satisfy (also in the miner README):
    inputs   obs       float32 [batch, 104]
             state_in  float32 [batch, 256]
    outputs  action    float32 [batch, 12]
             state_out float32 [batch, 256]

`state_in`/`state_out` are the policy's own per-episode memory, opaque to us: this server zeroes
it on /reset and feeds each step's `state_out` back in as the next step's `state_in`. Friction
varies per instance and is NOT observable, so adapting needs memory — but a feed-forward policy
can simply ignore `state_in` and emit zeros. The observation/action layout is in env/sim.py.

A submission that violates the contract fails readiness -> a typed failure attributed to the
submission, same as a screening rejection.
"""

from __future__ import annotations

import argparse
import os
from typing import Any

import numpy as np
import onnxruntime as ort

from gym_v1 import Player, serve

# Overridable so miners (and tools/local_eval.py) can serve a local file; in the sandbox the
# platform always writes to the spec's target_path.
SUBMISSION_PATH = os.environ.get("SUBMISSION_PATH", "/app/submission.onnx")
OBS_DIM = 104
ACT_DIM = 12
STATE_DIM = 256


def _check(tensor, want_last: int, what: str) -> None:
    # Dim 0 may be 1, None, or a symbolic batch name; the last dim is fixed.
    if tensor.type != "tensor(float)":
        raise ValueError(f"{what} must be float32, got {tensor.type}")
    if len(tensor.shape) != 2 or tensor.shape[-1] != want_last:
        raise ValueError(f"{what} must have shape [batch, {want_last}], got {tensor.shape}")


def _load_session() -> ort.InferenceSession:
    opts = ort.SessionOptions()
    # Single-threaded for determinism: same policy + same observations must produce the same
    # actions on every evaluation.
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    session = ort.InferenceSession(SUBMISSION_PATH, sess_options=opts,
                                   providers=["CPUExecutionProvider"])
    ins, outs = session.get_inputs(), session.get_outputs()
    if len(ins) != 2 or len(outs) != 2:
        raise ValueError(f"model must have exactly 2 inputs and 2 outputs, "
                         f"got {len(ins)}/{len(outs)}")
    _check(ins[0], OBS_DIM, "input 0 (obs)")
    _check(ins[1], STATE_DIM, "input 1 (state_in)")
    _check(outs[0], ACT_DIM, "output 0 (action)")
    _check(outs[1], STATE_DIM, "output 1 (state_out)")
    return session


class ParkourPlayer(Player):
    def __init__(self) -> None:
        # Load + validate WITHOUT raising out of __init__. Raising here kills the process
        # before serve() binds the port, and gym_v1's Referee.run() calls wait_until_ready()
        # OUTSIDE play_game — so a never-listening player raises PlayerError at a point where
        # no result.json can be written, and the platform attributes a bad SUBMISSION to the
        # REFEREE. Serving and reporting is_ready() False keeps it a typed submission failure,
        # which is what the runtime contract asks for.
        self._session: ort.InferenceSession | None = None
        self._names: list[str] = []
        self._state = np.zeros((1, STATE_DIM), np.float32)
        self.load_error: str | None = None
        try:
            self._session = _load_session()
            self._names = [i.name for i in self._session.get_inputs()]
        except Exception as e:  # noqa: BLE001 — every load failure is the submission's fault
            self.load_error = f"{type(e).__name__}: {e}"
            print(f"submission rejected at load: {self.load_error}", flush=True)

    def is_ready(self) -> bool:
        return self._session is not None

    def reset(self, match_id: str, player_index: int, seed: int, config: dict[str, Any]) -> None:
        # New instance, fresh memory. Carrying state across instances would let a policy count
        # episodes and infer where it sits in the (fixed, stratified) friction sweep.
        self._state = np.zeros((1, STATE_DIM), np.float32)

    def act(self, observation: Any, deadline_ms: int) -> Any:  # noqa: ARG002
        if self._session is None:
            raise RuntimeError(f"submission failed to load: {self.load_error}")
        obs = np.asarray(observation, dtype=np.float32).reshape(1, OBS_DIM)
        action, state = self._session.run(
            None, {self._names[0]: obs, self._names[1]: self._state})
        # NaN state would poison every later step of the episode, and the referee's
        # invalid_action gate only inspects the action, so sanitise it here.
        self._state = np.nan_to_num(np.asarray(state, np.float32).reshape(1, STATE_DIM),
                                    nan=0.0, posinf=0.0, neginf=0.0)
        return np.asarray(action, dtype=np.float64).ravel().tolist()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    serve(ParkourPlayer(), port=args.port, readiness_path="/health")
