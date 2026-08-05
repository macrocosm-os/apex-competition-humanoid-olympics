"""Score an ONNX policy against the course, in-process (no player/referee containers).

Same env, same gates, same scoring function the referee uses, so the numbers move together —
but this skips HTTP, so it is for calibration and variance measurement, not for producing the
figure that goes in spec.yaml. That one has to be measured inside the referee image.

    python tools/local_eval.py baseline/baseline.onnx -n 20
    python tools/local_eval.py baseline/baseline.onnx -n 20 --json out.json
    python tools/local_eval.py baseline/baseline.onnx -n 20 --record runs/base.npz

`--record` writes a trajectory log the run can be replayed from (tools/traj.py, tools/replay.py).
It costs a `qpos` copy per control step and changes nothing about the scoring, so a recorded suite
scores identically to an unrecorded one.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import numpy as np
import onnxruntime as ort

from env import ParkourSim, instance_score, instance_spec
from env.sim import OBS_DIM, STATE_DIM
from tools.traj import Recorder


def rollout(session, sim: ParkourSim, seed: int, max_steps: int,
            rec: Recorder | None = None, index: int = 0):
    obs = sim.reset(seed)
    if rec is not None:
        rec.begin(index, sim)
    state = np.zeros((1, STATE_DIM), np.float32)
    names = [i.name for i in session.get_inputs()]
    reason = None
    while reason is None:
        action, state = session.run(None, {names[0]: obs.reshape(1, OBS_DIM), names[1]: state})
        result = sim.step(np.asarray(action).ravel(), max_steps=max_steps)
        obs, reason = result.obs, result.terminal_reason
        if rec is not None:
            rec.capture(sim)
    return reason


def evaluate(path: str, n: int, max_steps: int, verbose: bool = True,
             record: str | None = None, stride: int = 1):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = opts.inter_op_num_threads = 1
    session = ort.InferenceSession(path, sess_options=opts, providers=["CPUExecutionProvider"])
    rec = Recorder(path, n, max_steps, stride=stride) if record else None

    rows, t0 = [], time.monotonic()
    for i in range(n):
        level, seed = instance_spec(i, n)
        sim = ParkourSim(level, seed)
        reason = rollout(session, sim, seed, max_steps, rec, index=i)
        score = instance_score(reason, sim.progress, sim.steps, max_steps)
        rows.append({"instance": i, "friction_level": round(level, 4), "terminal_reason": reason,
                     "progress": round(sim.progress, 4), "steps": sim.steps,
                     "score": round(score, 4), "max_x": round(sim.max_x, 2)})
        if rec is not None:
            rec.end(sim, rows[-1])
        if verbose:
            print(f"  [{i + 1:3d}/{n}] level {level:.3f}  {reason:14s} {sim.max_x:6.2f} m  "
                  f"progress {sim.progress:.3f}  score {score:.3f}")

    scores = [r["score"] for r in rows]
    summary = {
        "artifact": path, "n": n, "max_steps": max_steps,
        "raw_score": round(statistics.fmean(scores), 4),
        "stdev": round(statistics.stdev(scores), 4) if n > 1 else 0.0,
        "sem": round(statistics.stdev(scores) / n ** 0.5, 4) if n > 1 else 0.0,
        "num_completed": sum(r["terminal_reason"] == "completed" for r in rows),
        "mean_max_x": round(statistics.fmean(r["max_x"] for r in rows), 2),
        "reasons": {r: sum(x["terminal_reason"] == r for x in rows)
                    for r in sorted({x["terminal_reason"] for x in rows})},
        "wall_time_s": round(time.monotonic() - t0, 1),
        "instances": rows,
    }
    if rec is not None:
        out = rec.save(record)
        summary["recording"] = str(out)
        if verbose:
            print(f"wrote {out} ({rec.frames} frames, {out.stat().st_size / 1e6:.1f} MB)")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("artifact")
    ap.add_argument("-n", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=4000)
    ap.add_argument("--json")
    ap.add_argument("--record", metavar="PATH",
                    help="write a replayable trajectory log (.npz) — see tools/replay.py")
    ap.add_argument("--record-stride", type=int, default=1, metavar="N",
                    help="keep every Nth control step; 1 (default) records all 50 Hz")
    ap.add_argument("-q", "--quiet", action="store_true")
    a = ap.parse_args()
    s = evaluate(a.artifact, a.n, a.max_steps, verbose=not a.quiet,
                 record=a.record, stride=a.record_stride)
    print(json.dumps({k: v for k, v in s.items() if k != "instances"}, indent=2))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(s, f, indent=2)
        print("wrote", a.json)
