"""The trajectory log: what makes a scored evaluation replayable afterwards.

An evaluation's scalar record (`terminal_reason`, `progress`, `score`, ...) says how an instance
ended but nothing about how it got there, so it cannot drive a visualisation — replaying from it
renders a correct but empty course. This module stores the one missing piece: the robot's `qpos`
at every control step, alongside the friction values that built the scene it ran in.

That combination is enough for `tools/replay.py` to reconstruct the run with MuJoCo alone. No
policy, no onnxruntime, and no physics: setting `data.qpos` and calling `mj_forward` recomputes
every derived quantity a renderer needs. The alternative — logging actions and re-stepping — is
smaller but depends on reproducing physics bit-for-bit on the replaying machine, and any drift
silently shows a run that never happened. Positions cannot drift, because nothing is integrated.

Cost is negligible: nq is 19 (7 free-joint + 12 leg joints) and control runs at 50 Hz, so a run
that hits the 4000-step cap measures ~289 KiB, and most instances end well before it — the stock
baseline falls around 20 s, for ~70 KiB each. The file is `savez_compressed`, which buys only ~3%
on float32 mantissas; it is used for the format, not the ratio. `--record-stride` is the real
lever if a suite ever needs to be smaller.

Deliberately NOT stored: `qvel`, and therefore anything dynamic. A replay can show the motion but
not contact forces, which need the full state to recompute. Add qvel here if that changes.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass

import numpy as np

from env.sim import FRAME_SKIP, PHYS_DT

# Bump the minor half when adding keys a reader can ignore, the major half when it cannot.
FORMAT = "humanoid_parkour_traj/1"


@dataclass
class Instance:
    """One recorded episode: its conditions, its outcome, and its motion."""

    index: int
    qpos: np.ndarray        # (frames, nq) float32
    ticks: np.ndarray       # (frames,) int32, the control step each frame was captured at
    frictions: np.ndarray   # (n_course_geoms,) sliding friction, in course emission order
    row: dict               # the scored record: terminal_reason, progress, score, ...

    @property
    def frames(self) -> int:
        return int(self.qpos.shape[0])


@dataclass
class Recording:
    meta: dict
    instances: list[Instance]

    @property
    def control_dt(self) -> float:
        """Seconds of sim time per control step, before stride."""
        return float(self.meta["control_dt"])

    @property
    def frame_dt(self) -> float:
        """Seconds of sim time between consecutive recorded frames."""
        return self.control_dt * int(self.meta["stride"])

    def instance(self, index: int) -> Instance:
        for inst in self.instances:
            if inst.index == index:
                return inst
        have = ", ".join(str(i.index) for i in self.instances)
        raise KeyError(f"instance {index} is not in this recording; it has [{have}]")


class Recorder:
    """Accumulates a suite's trajectories, then writes them as one .npz.

    Held in memory until `save`, which is fine at these sizes and keeps a partial file from ever
    looking like a complete one. Usage, per instance:

        rec.begin(i, sim)                    # after sim.reset(), captures the starting pose
        ...                                  # rec.capture(sim) after every sim.step()
        rec.end(sim, row)
    """

    def __init__(self, artifact: str, n: int, max_steps: int, stride: int = 1):
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")
        self.stride = int(stride)
        # Written up front, not on the first episode, so an empty suite still yields a readable
        # file rather than one that raises on the first attribute a reader touches.
        self._meta = {"format": FORMAT, "artifact": artifact, "n": n, "max_steps": max_steps,
                      "stride": self.stride, "control_dt": PHYS_DT * FRAME_SKIP,
                      "phys_dt": PHYS_DT, "frame_skip": FRAME_SKIP}
        self._instances: list[Instance] = []
        self._cur: Instance | None = None
        self._qpos: list[np.ndarray] = []
        self._ticks: list[int] = []

    def begin(self, index: int, sim) -> None:
        self._meta.setdefault("nq", int(sim.model.nq))
        self._qpos, self._ticks = [], []
        self._cur = Instance(index=index, qpos=np.empty(0), ticks=np.empty(0),
                             frictions=np.asarray(sim.frictions, np.float64), row={})
        self._append(sim)   # the starting pose is frame 0

    def capture(self, sim) -> None:
        """Record the current pose if this control step falls on the stride."""
        if self._cur is None:
            raise RuntimeError("capture() before begin()")
        if sim.steps % self.stride == 0:
            self._append(sim)

    def end(self, sim, row: dict) -> None:
        """Close the instance, always keeping the terminal pose — that is the frame worth seeing."""
        if self._cur is None:
            raise RuntimeError("end() before begin()")
        if not self._ticks or self._ticks[-1] != sim.steps:
            self._append(sim)
        self._cur.qpos = np.asarray(self._qpos, np.float32)
        self._cur.ticks = np.asarray(self._ticks, np.int32)
        self._cur.row = dict(row)
        self._instances.append(self._cur)
        self._cur, self._qpos, self._ticks = None, [], []

    def _append(self, sim) -> None:
        self._qpos.append(np.array(sim.data.qpos, np.float32))
        self._ticks.append(int(sim.steps))

    def save(self, path: str | pathlib.Path) -> pathlib.Path:
        if self._cur is not None:
            raise RuntimeError("save() with instance still open; call end() first")
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # `instance` and `frames` are written last so they hold whatever the recorder saw, not
        # whatever the caller's scored row happened to be keyed on — `load` indexes arrays by them.
        meta = dict(self._meta, mujoco_version=_mujoco_version(),
                    instances=[i.row | {"instance": i.index, "frames": i.frames}
                               for i in self._instances])
        arrays = {"meta": np.array(json.dumps(meta))}
        for inst in self._instances:
            arrays[f"qpos_{inst.index:03d}"] = inst.qpos
            arrays[f"ticks_{inst.index:03d}"] = inst.ticks
            arrays[f"mu_{inst.index:03d}"] = inst.frictions
        np.savez_compressed(path, **arrays)
        return path

    @property
    def frames(self) -> int:
        return sum(i.frames for i in self._instances)


def load(path: str | pathlib.Path) -> Recording:
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["meta"]))
        got = str(meta.get("format", "?"))
        if got.split("/")[0] != FORMAT.split("/")[0]:
            raise ValueError(f"{path} is not a trajectory log (format {got!r})")
        instances = []
        for row in meta["instances"]:
            i = int(row["instance"])
            instances.append(Instance(index=i, qpos=z[f"qpos_{i:03d}"], ticks=z[f"ticks_{i:03d}"],
                                      frictions=z[f"mu_{i:03d}"], row=row))
    return Recording(meta=meta, instances=instances)


def _mujoco_version() -> str:
    """Recorded for provenance only — replay reads positions, so it is not version-locked."""
    try:
        import mujoco
        return str(mujoco.__version__)
    except Exception:
        return "unknown"
