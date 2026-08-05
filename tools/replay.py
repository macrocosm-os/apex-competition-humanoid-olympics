"""Play back a recorded evaluation. MuJoCo only — no policy, no onnxruntime, no physics.

    PYTHONPATH=. python tools/replay.py runs/base.npz              # list what is in the log
    PYTHONPATH=. python tools/replay.py runs/base.npz -i 7         # film instance 7 to mp4
    PYTHONPATH=. python tools/replay.py runs/base.npz --worst      # film the lowest-scoring one
    PYTHONPATH=. python tools/replay.py runs/base.npz --all        # film every instance
    PYTHONPATH=. mjpython tools/replay.py runs/base.npz -i 7 --live   # interactive viewer

Replay sets `data.qpos` from the log and calls `mj_forward`, which recomputes every derived
quantity a renderer needs. Nothing is integrated, so a replay cannot drift from the scored run the
way re-simulating from an action log could — and it needs neither the submission nor the round
seed, so a run stays viewable after both are gone.

The scene is rebuilt from the frictions stored per instance, through the same `_lit_model` the
preview uses, so it is the scored geometry with lights added and nothing else changed.

`--live` needs `mjpython` on macOS rather than `python`: the passive viewer has to own the main
thread there. Filming to mp4 is fully headless and has no such constraint. Needs ffmpeg.
"""

from __future__ import annotations

import argparse
import pathlib
import time

import mujoco

from env.course import COURSE_LENGTH
from tools.preview import OUT, _camera, _lit_model, frames_dir, mp4, png
from tools.traj import Instance, Recording, load

TARGET_FPS = 30.0


def _chase(cam, qpos) -> None:
    """The same over-the-shoulder framing `tools/preview.py --run` uses, for comparability."""
    cam.lookat[:] = [float(qpos[0]) + 1.2, 0, float(qpos[2])]
    cam.distance, cam.azimuth, cam.elevation = 4.6, 128, -12


def _stride_for(frame_dt: float, target_fps: float) -> tuple[int, float]:
    """Frames to skip so playback runs at real speed near `target_fps`, and the fps that gives."""
    step = max(1, int(round((1.0 / target_fps) / frame_dt)))
    return step, 1.0 / (frame_dt * step)


def _describe(rec: Recording, inst: Instance) -> str:
    r = inst.row
    return (f"instance {inst.index:3d}  level {r.get('friction_level'):.3f}  "
            f"{str(r.get('terminal_reason')):14s} {r.get('max_x'):6.2f} m of {COURSE_LENGTH:.1f} m  "
            f"score {r.get('score'):.4f}  {inst.frames} frames "
            f"({max(inst.frames - 1, 0) * rec.frame_dt:.1f} s)")


def _scene(inst: Instance) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Rebuild the instance's scene from its recorded frictions, refusing a mismatched log."""
    model = _lit_model(inst.frictions)
    if inst.qpos.shape[1] != model.nq:
        raise ValueError(f"recording has nq={inst.qpos.shape[1]}, this model has nq={model.nq}; "
                         "the log predates a model change and cannot be replayed against it")
    return model, mujoco.MjData(model)


def film(rec: Recording, inst: Instance, out: pathlib.Path, target_fps: float = TARGET_FPS,
         width: int = 1280, height: int = 720) -> None:
    model, data = _scene(inst)
    cam, opt = _camera()
    renderer = mujoco.Renderer(model, height=height, width=width)
    step, fps = _stride_for(rec.frame_dt, target_fps)

    fd = frames_dir(f"_replay_{inst.index:03d}")
    keep = list(range(0, inst.frames, step))
    if not keep:
        raise ValueError(f"instance {inst.index} has no recorded frames")
    if keep[-1] != inst.frames - 1:
        keep.append(inst.frames - 1)   # always end on the terminal pose
    for fi, src in enumerate(keep):
        data.qpos[:] = inst.qpos[src]
        mujoco.mj_forward(model, data)
        _chase(cam, data.qpos)
        renderer.update_scene(data, camera=cam, scene_option=opt)
        png(renderer.render(), fd / f"f{fi:05d}.png")
    mp4(fd, out, fps=max(1, round(fps)))   # a heavily strided log can round below 1 fps


def live(rec: Recording, inst: Instance, speed: float = 1.0, loop: bool = False) -> None:
    import mujoco.viewer

    model, data = _scene(inst)
    try:
        viewer = mujoco.viewer.launch_passive(model, data)
    except RuntimeError as e:
        raise SystemExit(f"{e}\n\nOn macOS the passive viewer must own the main thread — run this "
                         f"with `mjpython tools/replay.py ...` instead of `python`.") from e
    dt = rec.frame_dt / max(speed, 1e-6)
    with viewer as v:
        while True:
            # Pace against a wall-clock deadline rather than sleeping a flat dt, so the time spent
            # in mj_forward and sync does not quietly stretch playback past real time.
            deadline = time.monotonic()
            for frame in inst.qpos:
                if not v.is_running():
                    return
                data.qpos[:] = frame
                mujoco.mj_forward(model, data)
                v.sync()
                deadline += dt
                time.sleep(max(0.0, deadline - time.monotonic()))
            if not loop:
                return


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Replay a recorded evaluation with MuJoCo alone.")
    ap.add_argument("recording", help=".npz written by local_eval.py --record")
    ap.add_argument("-i", "--instance", type=int, help="which instance to play back")
    ap.add_argument("--worst", action="store_true", help="pick the lowest-scoring instance")
    ap.add_argument("--all", action="store_true", help="film every instance in the log")
    ap.add_argument("--live", action="store_true",
                    help="interactive viewer instead of mp4 (needs mjpython on macOS)")
    ap.add_argument("--speed", type=float, default=1.0, help="--live playback rate; 1.0 is real time")
    ap.add_argument("--loop", action="store_true", help="--live: repeat until the window is closed")
    ap.add_argument("--fps", type=float, default=TARGET_FPS, help="mp4 frame rate to aim for")
    ap.add_argument("--out", help="mp4 path (single instance only; default renders/replay_NNN.mp4)")
    a = ap.parse_args()

    rec = load(a.recording)
    m = rec.meta
    print(f"{a.recording}: {len(rec.instances)} instances of {m.get('n')}, "
          f"{m.get('artifact')}, recorded with mujoco {m.get('mujoco_version')} "
          f"at {1 / rec.frame_dt:.0f} Hz (stride {m.get('stride')})")

    if a.all:
        OUT.mkdir(exist_ok=True)
        for inst in rec.instances:
            print(" ", _describe(rec, inst))
            film(rec, inst, OUT / f"replay_{inst.index:03d}.mp4", target_fps=a.fps)
        raise SystemExit(0)

    if a.instance is None and not a.worst:
        for inst in rec.instances:
            print(" ", _describe(rec, inst))
        print("\nPass -i N to film one, --worst for the lowest-scoring, or --all for every one.")
        raise SystemExit(0)

    if a.worst:
        chosen = min(rec.instances, key=lambda i: i.row.get("score", 0.0))
    else:
        chosen = rec.instance(a.instance)
    print(" ", _describe(rec, chosen))

    if a.live:
        live(rec, chosen, speed=a.speed, loop=a.loop)
    else:
        OUT.mkdir(exist_ok=True)
        film(rec, chosen, pathlib.Path(a.out) if a.out
             else OUT / f"replay_{chosen.index:03d}.mp4", target_fps=a.fps)
