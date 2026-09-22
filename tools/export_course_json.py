"""Export the referee's exact static course geometry as renderer-friendly JSON.

Run from the repository root:

    PYTHONPATH=. python tools/export_course_json.py

The result deliberately uses the same :func:`env.course.build_event` layouts that
MuJoCo receives.  Each surface is an axis-aligned local box described by a world
centre, full dimensions, z-yaw, material kind, and whether it may support the
robot.  The 400 m event therefore contains its real 96 tangent boxes rather
than a visually similar but geometrically different circle.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from dataclasses import asdict

from env.course import (COLOR, EVENTS, EVENT_LABELS, HIGH_JUMP_BARS_M, HURDLE_FIRST_MIN_M,
                        HURDLE_HEIGHTS_M, HURDLE_LAST_MAX_M, HURDLE_MIN_GAP_M, PLINTH_TOP,
                        TRACK_HALF_W, build_event)


def surface_json(surface) -> dict:
    """One MuJoCo box, using full rather than half extents for renderers."""
    data = asdict(surface)
    data["shape"] = "box"
    data["size_m"] = [2 * data.pop("hx"), 2 * data.pop("hy"), 2 * data.pop("hz")]
    data["center_m"] = [data.pop("x"), data.pop("y"), data.pop("z")]
    data["yaw_rad"] = data.pop("yaw")
    data["rgba"] = [float(value) for value in COLOR[data["kind"]].split()]
    return data


def event_json(event: str) -> dict:
    layout = build_event(event)
    result = {
        "id": event,
        "label": EVENT_LABELS[event],
        "start": {"position_m": [layout.start_x, layout.start_y, PLINTH_TOP],
                  "yaw_rad": layout.start_yaw},
        "finish": {"route_distance_m": layout.finish},
        "surfaces": [surface_json(surface) for surface in layout.surfaces],
    }
    if layout.challenge:
        result["challenge"] = dict(layout.challenge)
    if event == "hurdles_100":
        # The surfaces above are ONE draw. A renderer showing them as the course would be showing
        # a layout no attempt is guaranteed to run -- and with four attempts drawn independently,
        # it would be wrong for three of every four.
        result["surfaces_are_a_sample"] = True
        result["draw"] = {
            "note": (
                "Hurdle placement and height order are RANDOMISED per attempt. The surfaces in "
                "this file are one sample, not the course any attempt runs. Do not render them "
                "as the course."
            ),
            "authoritative_source": (
                "each attempt reports the layout it ran as hurdle_x_0..9 / hurdle_h_0..9 on "
                "conditions.challenge in its history record, and in result.json's per-task "
                "challenge"
            ),
            "count": len(HURDLE_HEIGHTS_M),
            # What the draw holds fixed: the same ten heights every attempt, in a shuffled order,
            # inside this window, never closer together than the minimum gap.
            "heights_m": list(HURDLE_HEIGHTS_M),
            "window_m": [HURDLE_FIRST_MIN_M, HURDLE_LAST_MAX_M],
            "min_gap_m": HURDLE_MIN_GAP_M,
        }
    if event == "high_jump":
        result["variants"] = {
            "bar_height_m": list(HIGH_JUMP_BARS_M),
            "bar_center_z_m": [PLINTH_TOP + height for height in HIGH_JUMP_BARS_M],
        }
    return result


def export() -> dict:
    return {
        "schema": "humanoid-olympics.course-layout.v1",
        "units": "metres and radians",
        "coordinate_system": {
            "x": "forward on straight events",
            "y": "runner-left on straight events",
            "z": "up",
            "deck_top_z_m": PLINTH_TOP,
        },
        "lane": {"full_width_m": 2 * TRACK_HALF_W, "half_width_m": TRACK_HALF_W},
        # Static geometry only. An event carrying `surfaces_are_a_sample` draws part of its course
        # per attempt, and its real layout travels on the attempt, not in this file.
        "surfaces": "static per event, except where `surfaces_are_a_sample` is set",
        "events": [event_json(event) for event in EVENTS],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("docs/course-layouts.json"))
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(export(), indent=2) + "\n")
    print(args.out)
