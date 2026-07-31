"""humanoid_parkour gym_v1 PLAYER server (the image's `entrypoints.evaluate.command`).

The platform writes the miner's ONNX policy to /app/submission.onnx; this
server loads it, validates the interface, and serves /health /reset /act.
There is no miner code in this sandbox — the artifact is a pure ONNX graph
(spec `artifact_type: onnx`), so validation is structural, not screening.

Contract the submission must satisfy (also in the miner README):
    - exactly one input:  float32, shape [1, 56] or [batch/dynamic, 56]
    - exactly one output: float32, shape [1, 17] or [batch/dynamic, 17]
The observation/action layout is documented in env/sim.py.

A submission that violates the contract fails readiness -> a typed failure
attributed to the submission, same as a screening rejection.
"""

from __future__ import annotations

import argparse
import os
from typing import Any

import numpy as np
import onnxruntime as ort

from gym_v1 import Player, serve

# Overridable so miners (and tools/local_eval.py) can serve a local file; in
# the sandbox the platform always writes to the spec's target_path.
SUBMISSION_PATH = os.environ.get("SUBMISSION_PATH", "/app/submission.onnx")
OBS_DIM = 56
ACT_DIM = 17


def _check_dims(shape: list, want_last: int, what: str) -> None:
    # Dim 0 may be 1, None, or a symbolic batch name; the last dim is fixed.
    if len(shape) != 2 or shape[-1] != want_last:
        raise ValueError(f"{what} must have shape [batch, {want_last}], got {shape}")


def _load_session() -> ort.InferenceSession:
    opts = ort.SessionOptions()
    # Single-threaded for determinism: same policy + same observations must
    # produce the same actions on every evaluation.
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    session = ort.InferenceSession(SUBMISSION_PATH, sess_options=opts, providers=["CPUExecutionProvider"])
    inputs, outputs = session.get_inputs(), session.get_outputs()
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError(f"model must have exactly 1 input and 1 output, got {len(inputs)}/{len(outputs)}")
    if inputs[0].type != "tensor(float)":
        raise ValueError(f"input must be float32, got {inputs[0].type}")
    _check_dims(inputs[0].shape, OBS_DIM, "input")
    _check_dims(outputs[0].shape, ACT_DIM, "output")
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
        self._input_name: str | None = None
        self.load_error: str | None = None
        try:
            self._session = _load_session()
            self._input_name = self._session.get_inputs()[0].name
        except Exception as e:  # noqa: BLE001 — every load failure is the submission's fault
            self.load_error = f"{type(e).__name__}: {e}"
            print(f"submission rejected at load: {self.load_error}", flush=True)

    def is_ready(self) -> bool:
        return self._session is not None

    def reset(self, match_id: str, player_index: int, seed: int, config: dict[str, Any]) -> None:
        pass  # the policy is a stateless feed-forward graph

    def act(self, observation: Any, deadline_ms: int) -> Any:  # noqa: ARG002
        if self._session is None:
            raise RuntimeError(f"submission failed to load: {self.load_error}")
        obs = np.asarray(observation, dtype=np.float32).reshape(1, OBS_DIM)
        (action,) = self._session.run(None, {self._input_name: obs})
        return np.asarray(action, dtype=np.float64).ravel().tolist()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    serve(ParkourPlayer(), port=args.port, readiness_path="/health")
