"""Write an event's docs images at the size `docs/images/` already uses.

`preview.py` renders for inspection at 1600x900 and shells out to ffmpeg; this writes the 492x410
PNGs the README grid expects, encoding with the standard library so no ffmpeg is needed.

Offscreen GL has to work, which in practice means a desktop session -- a headless shell renders a
uniformly black frame, so the blank check below refuses to write one rather than let an empty
image reach the docs.

    PYTHONPATH=. .venv/bin/python tools/make_event_images.py --event race_walk_200
"""

from __future__ import annotations

import argparse
import pathlib
import struct
import zlib

import mujoco
import numpy as np

from env import EVENTS, OlympicsSim, instance_spec
from tools.preview import _camera, _lit_model

IMAGES = pathlib.Path("docs/images")
WIDTH, HEIGHT = 492, 410
# docs/images names are the human label, hyphenated: "200 m race walk" -> "200m-race-walk".
SLUG = {"sprint_100": "100m-sprint", "sprint_400": "400m-circular-sprint",
        "hurdles_100": "100m-hurdles", "high_jump": "high-jump", "long_jump": "long-jump",
        "triple_jump": "triple-jump", "race_walk_200": "200m-race-walk"}


def write_png(pixels: np.ndarray, path: pathlib.Path) -> None:
    height, width, _ = pixels.shape
    raw = b"".join(b"\x00" + pixels[y].tobytes() for y in range(height))

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    path.write_bytes(b"\x89PNG\r\n\x1a\n"
                     + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(raw, 9))
                     + chunk(b"IEND", b""))


def render(event: str, attempt: int, seed: int) -> np.ndarray:
    sim = OlympicsSim(instance_spec(event, attempt, seed))
    model = _lit_model(event, sim.frictions, sim.params.challenge, sim.params.seed)
    data = mujoco.MjData(model)
    sim.reset()
    data.qpos[:] = sim.data.qpos
    mujoco.mj_forward(model, data)
    camera, option = _camera()
    if sim.layout.is_circular:
        camera.lookat[:] = [0, 0, 0]
        camera.distance, camera.azimuth, camera.elevation = 190, 125, -55
    else:
        camera.lookat[:] = [max(8.0, sim.layout.finish / 2), 0, 1.0]
        camera.distance = max(18.0, sim.layout.finish * 0.7)
        camera.azimuth, camera.elevation = 120, -18
    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
    renderer.update_scene(data, camera=camera, scene_option=option)
    return renderer.render()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--event", choices=EVENTS, required=True)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    IMAGES.mkdir(parents=True, exist_ok=True)
    for attempt, suffix in ((0, ""), (3, "-hard")):
        pixels = render(args.event, attempt, args.seed)
        if len(np.unique(pixels.reshape(-1, 3), axis=0)) < 8:
            raise SystemExit(
                f"render is blank ({len(np.unique(pixels.reshape(-1, 3), axis=0))} distinct "
                f"colours) -- offscreen GL is not working here. Run this from a desktop session."
            )
        out = IMAGES / f"{SLUG[args.event]}{suffix}-3d.png"
        write_png(pixels, out)
        print(f"wrote {out} ({out.stat().st_size / 1000:.0f} kB)")
