"""Render the v0.3.0 course, and optionally drive it with Unitree's stock G1 walking policy.

    python tools/preview_v3.py                 # stills + flythrough
    python tools/preview_v3.py --walk          # + a run by the stock walker

`--walk` needs a checkout of unitree_rl_gym (for the 12-DoF robot and the pretrained policy)
plus torch and pyyaml:

    git clone --depth 1 --filter=blob:none --sparse https://github.com/unitreerobotics/unitree_rl_gym
    cd unitree_rl_gym && git sparse-checkout set resources/robots/g1_description \\
        deploy/pre_train/g1 deploy/deploy_mujoco

Pass its path with --urlg. Note the walker is Unitree's 12-DoF legs-only G1, NOT the 29-DoF
Menagerie G1 the competition uses — it is a difficulty probe, not the competition baseline.

Everything else needs only mujoco + ffmpeg, and renders against the Menagerie G1 for scale.
"""
from __future__ import annotations

import argparse
import math
import pathlib
import re
import subprocess

import mujoco
import numpy as np

from course_v3 import PLINTH_TOP, build_course, course_xml_fragment

OUT = pathlib.Path("renders")


def scene_xml(robot_scene: pathlib.Path, lit=True):
    """Splice the course into a robot scene. Returns (path, course_length, deck_height)."""
    segs, length, top = build_course()
    xml = robot_scene.read_text()
    frag = (f'    <geom type="box" pos="-1.5 0 {PLINTH_TOP - 0.2:.3f}" size="1.5 1.2 0.2" '
            f'condim="3" friction="1 .1 .1" rgba=".45 .47 .5 1"/>\n') + course_xml_fragment(segs)
    xml = xml.replace("</worldbody>", frag +
                      f'\n    <geom type="box" pos="{length:.2f} 0 {top + 0.9:.2f}" '
                      f'size="0.04 1.2 0.02" rgba="1 .95 .2 1"/>\n  </worldbody>')
    # The offscreen framebuffer must cover the largest render. Both scenes already have a
    # <global>, and the schema permits only one, so extend it rather than adding a second.
    xml = re.sub(r"<global\b", '<global offwidth="1600" offheight="900"', xml, count=1)
    if lit:
        # Stock scenes are lit for debugging one robot, not for reading a 50 m course.
        xml = re.sub(r'<headlight[^/]*/>',
                     '<headlight diffuse="0.75 0.75 0.75" ambient="0.45 0.45 0.47" '
                     'specular="0.2 0.2 0.2"/>', xml, count=1)
        xml = xml.replace('reflectance="0.2"', 'reflectance="0.0"')
        xml = xml.replace('builtin="flat" rgb1="0 0 0" rgb2="0 0 0"',
                          'builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0.1 0.15 0.25"')
        xml = xml.replace("<worldbody>", '<worldbody>\n'
            '    <light pos="8 -6 9" dir="-0.3 0.4 -1" directional="true" diffuse="0.55 0.55 0.52"/>\n'
            '    <light pos="34 6 9" dir="0.2 -0.4 -1" directional="true" diffuse="0.35 0.36 0.40"/>', 1)
    path = robot_scene.parent / "_preview_course.xml"
    path.write_text(xml)
    return path, length, top


def png(px, path):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                    "-s", f"{px.shape[1]}x{px.shape[0]}", "-i", "-", str(path)],
                   input=px.tobytes(), check=True)


def mp4(frame_dir, out, fps=30):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
                    "-i", str(frame_dir / "f%05d.png"), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "20", str(out)], check=True)


def frames_dir(name):
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    for f in d.glob("*.png"):
        f.unlink()
    return d


def render_course(menagerie: pathlib.Path):
    path, length, _ = scene_xml(menagerie / "scene_mjx.xml")
    m = mujoco.MjModel.from_xml_path(str(path))
    d = mujoco.MjData(m)
    d.qpos[:] = m.key_qpos[0]
    d.qpos[0], d.qpos[2] = -0.6, PLINTH_TOP + 0.784
    mujoco.mj_forward(m, d)
    cam, opt = mujoco.MjvCamera(), mujoco.MjvOption()
    mujoco.mjv_defaultCamera(cam); mujoco.mjv_defaultOption(opt)

    r = mujoco.Renderer(m, height=900, width=1600)
    for name, look, dist, elev in [("01_overview_iso", length / 2, 44, -22),
                                   ("02_side_elevation", length / 2, 40, -6),
                                   ("03_onramp", 8.0, 14, -14),
                                   ("04_stairs_leap_drop", 20.5, 13, -14),
                                   ("05_vault_climb", 27.5, 13, -14),
                                   ("06_crawl_beam_slick", 38.0, 15, -14)]:
        cam.lookat[:] = [look, 0, 1.2]
        cam.distance, cam.azimuth, cam.elevation = dist, (90 if "side" in name else 132), elev
        r.update_scene(d, camera=cam, scene_option=opt)
        png(r.render(), OUT / f"{name}.png")

    fd = frames_dir("_fly")
    r = mujoco.Renderer(m, height=720, width=1280)
    n = 300
    for i in range(n):
        t = i / (n - 1)
        cam.lookat[:] = [-2.0 + t * (length + 4.0), 0, 1.0]
        cam.distance, cam.elevation = 9.0, -13 - 6 * math.sin(math.pi * t)
        cam.azimuth = 118 + 24 * math.sin(2 * math.pi * t)
        r.update_scene(d, camera=cam, scene_option=opt)
        png(r.render(), fd / f"f{i:05d}.png")
    mp4(fd, OUT / "course_flythrough.mp4")
    print("wrote", OUT / "course_flythrough.mp4")


def walk(urlg: pathlib.Path, secs=60.0):
    """Drive the course with the stock 12-DoF walking policy. Renders and reports progress."""
    import torch, yaml
    cfg = yaml.safe_load((urlg / "deploy/deploy_mujoco/configs/g1.yaml").read_text())
    kp = np.array(cfg["kps"], np.float32); kd = np.array(cfg["kds"], np.float32)
    default = np.array(cfg["default_angles"], np.float32)
    na, nobs, dt, dec = cfg["num_actions"], cfg["num_obs"], cfg["simulation_dt"], cfg["control_decimation"]
    cmd_scale = np.array(cfg["cmd_scale"], np.float32)

    path, length, _ = scene_xml(urlg / "resources/robots/g1_description/scene.xml")
    m = mujoco.MjModel.from_xml_path(str(path)); m.opt.timestep = dt
    d = mujoco.MjData(m)
    assert m.nu == na, f"expected {na} actuators, model has {m.nu}"
    d.qpos[0], d.qpos[2] = -0.8, PLINTH_TOP + 0.793
    d.qpos[7:] = default
    mujoco.mj_forward(m, d)

    policy = torch.jit.load(str(urlg / "deploy/pre_train/g1/motion.pt"))
    action, target, obs = np.zeros(na, np.float32), default.copy(), np.zeros(nobs, np.float32)
    r = mujoco.Renderer(m, height=720, width=1280)
    cam, opt = mujoco.MjvCamera(), mujoco.MjvOption()
    mujoco.mjv_defaultCamera(cam); mujoco.mjv_defaultOption(opt)
    fd = frames_dir("_walk")

    fi, max_x, fell_at, fell_i = 0, -9.0, None, None
    every = int(round((1 / 30) / dt))
    for i in range(int(secs / dt)):
        d.ctrl[:] = (target - d.qpos[7:]) * kp - d.qvel[6:] * kd
        mujoco.mj_step(m, d)
        if i % dec == 0:
            qw, qx, qy, qz = d.qpos[3:7]
            # Heading hold. motion.pt tracks BODY-FRAME velocity with no heading feedback, so
            # without this it drifts ~0.26 m sideways per metre and walks off the track long
            # before reaching any obstacle. A real deployment closes this loop too.
            yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
            y = float(d.qpos[1])
            cmd = np.array([0.8, np.clip(-0.5 * y, -0.3, 0.3),
                            np.clip(-1.5 * yaw - 0.8 * y, -0.6, 0.6)], np.float32)
            phase = ((i * dt) % 0.8) / 0.8
            obs[:3] = d.qvel[3:6] * cfg["ang_vel_scale"]
            obs[3:6] = [2 * (-qz * qx + qw * qy), -2 * (qz * qy + qw * qx), 1 - 2 * (qw * qw + qz * qz)]
            obs[6:9] = cmd * cmd_scale
            obs[9:9 + na] = (d.qpos[7:] - default) * cfg["dof_pos_scale"]
            obs[9 + na:9 + 2 * na] = d.qvel[6:] * cfg["dof_vel_scale"]
            obs[9 + 2 * na:9 + 3 * na] = action
            obs[9 + 3 * na:9 + 3 * na + 2] = [math.sin(2 * math.pi * phase), math.cos(2 * math.pi * phase)]
            action = policy(torch.from_numpy(obs).unsqueeze(0)).detach().numpy().squeeze()
            target = action * cfg["action_scale"] + default

        x, z = float(d.qpos[0]), float(d.qpos[2])
        max_x = max(max_x, x)
        if fell_at is None and z < PLINTH_TOP + 0.35:
            fell_at, fell_i = x, i
        if i % every == 0:
            cam.lookat[:] = [x + 1.2, 0, z]
            cam.distance, cam.azimuth, cam.elevation = 4.6, 128, -12
            r.update_scene(d, camera=cam, scene_option=opt)
            png(r.render(), fd / f"f{fi:05d}.png"); fi += 1
        if fell_i is not None and i - fell_i > int(2.0 / dt):
            break                                   # 2 s of aftermath is plenty
    mp4(fd, OUT / "baseline_run.mp4")
    print(f"reached {max_x:.2f} m of {length:.1f} m ({100 * max(0, max_x) / length:.0f}%), "
          f"fell at {'never' if fell_at is None else f'{fell_at:.2f} m'}")
    print("wrote", OUT / "baseline_run.mp4")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--menagerie", default="mujoco_menagerie/unitree_g1")
    ap.add_argument("--urlg", default="unitree_rl_gym")
    ap.add_argument("--walk", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    render_course(pathlib.Path(a.menagerie))
    if a.walk:
        walk(pathlib.Path(a.urlg))
