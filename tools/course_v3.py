"""v0.3.0 parkour course composer (prototype — visualisation and calibration only).

Builds the segment sequence from docs/v0.3.0-design.md §3 as MJCF box geoms on a raised plinth,
sized for the Unitree G1 (1.26 m tall, pelvis 0.784 m). Gaps are real voids in the plinth.

NOT the production generator: no friction randomisation, no per-round seeding, no checkpoints,
no height-scan hooks. Those land with the real env.

    python tools/course_v3.py            # print the layout
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

PLINTH_TOP = 0.8            # top surface of the track; gaps drop through it
PLINTH_THICK = 0.4
TRACK_HALF_W = 1.2

# On-ramp shape, measured against the stock G1 walker (see docs/v0.3.0-design.md §2.4):
# it climbs 15.4 deg but stalls at 20.1 deg, so this is the steepest short climb a naive policy
# can still manage. Drop height barely matters — 0.20 m and 0.55 m both end its run — so we take
# the full drop for the spectacle.
ON_RAMP_RISE, ON_RAMP_RUN, ON_RAMP_DROP = 0.55, 2.0, 0.55

COLOR = {  # by maneuver, so a render is readable at a glance
    "flat": ".55 .57 .60 1", "ramp": ".45 .80 .55 1",
    "stairs_up": ".30 .65 .45 1", "stairs_dn": ".22 .50 .38 1",
    "climb_up": ".85 .45 .20 1", "drop_down": ".70 .35 .18 1",
    "leap": ".20 .20 .24 1", "vault": ".80 .25 .35 1",
    "crawl": ".55 .30 .75 1", "beam": ".95 .75 .20 1", "slick": ".35 .70 .95 1",
}


@dataclass
class Seg:
    kind: str
    length: float
    boxes: list = field(default_factory=list)   # (cx, cy, cz, sx, sy, sz, color[, pitch])


def _slab(x0, length, top, color, half_w=TRACK_HALF_W):
    """A walkable slab whose upper surface sits at `top`."""
    return (x0 + length / 2, 0.0, top - PLINTH_THICK / 2, length / 2, half_w, PLINTH_THICK / 2, color)


def _ramp(x0, length, top0, rise, color):
    """A slab rotated about y so its top face is an incline climbing `rise` over `length`."""
    ang = math.atan2(rise, length)
    return (x0 + length / 2, 0.0, top0 + rise / 2 - (PLINTH_THICK / 2) * math.cos(ang),
            math.hypot(length, rise) / 2, TRACK_HALF_W, PLINTH_THICK / 2, color, -ang)


def build_course():
    """Return (segments, total_length, final_deck_height)."""
    segs: list[Seg] = []
    x, top = 0.0, PLINTH_TOP

    def flat(length, kind="flat"):
        nonlocal x
        segs.append(Seg(kind, length, [_slab(x, length, top, kind)]))
        x += length

    def stairs(n, rise, run, kind):
        nonlocal x, top
        s = Seg(kind, n * run)
        for i in range(n):
            step = rise if kind == "stairs_up" else -rise
            s.boxes.append(_slab(x + i * run, run, top + (i + 1) * step, kind))
        segs.append(s); x += n * run; top += n * (rise if kind == "stairs_up" else -rise)

    # ON-RAMP: a naive walking policy should clear this and nothing beyond it. Because the course
    # is linear and scored on progress, this section IS the easy tier — no separate tiers needed.
    flat(6.0)                                              # run-up: let a walker settle into gait
    segs.append(Seg("ramp_up", ON_RAMP_RUN, [_ramp(x, ON_RAMP_RUN, top, ON_RAMP_RISE, "ramp")]))
    x += ON_RAMP_RUN; top += ON_RAMP_RISE
    flat(1.6)                                              # landing, to set up for the edge
    top -= ON_RAMP_DROP                                    # sheer drop, no ramp down
    segs.append(Seg("drop_down", 0.0))
    flat(2.4)

    # THE COURSE PROPER
    flat(4.0)
    stairs(5, 0.20, 0.32, "stairs_up")
    flat(1.6)
    segs.append(Seg("leap", 1.0)); x += 1.0                # a real void: no slab at all
    flat(2.2)
    top -= 0.6                                             # drop-down
    segs.append(Seg("drop_down", 0.0))
    flat(2.4)

    s = Seg("vault", 1.0, [_slab(x, 1.0, top, "flat"),     # waist-high barrier, landing continues
                           (x + 0.5, 0.0, top + 0.31, 0.09, TRACK_HALF_W, 0.31, "vault")])
    segs.append(s); x += 1.0
    flat(2.0)

    plat = 0.55                                            # climb-up onto a hip-height platform
    segs.append(Seg("climb_up", 2.2, [_slab(x, 2.2, top + plat, "climb_up")]))
    x += 2.2; top += plat
    flat(1.2)
    top -= plat                                            # and back down
    flat(2.0)

    s = Seg("crawl", 2.0, [_slab(x, 2.0, top, "flat"),     # overhead bar at 0.75 m, on posts
                           (x + 1.0, 0.0, top + 0.83, 0.5, TRACK_HALF_W, 0.08, "crawl")])
    for sy in (-1.0, 1.0):
        s.boxes.append((x + 1.0, sy * (TRACK_HALF_W - 0.06), top + 0.375, 0.06, 0.06, 0.375, "crawl"))
    segs.append(s); x += 2.0
    flat(1.6)

    segs.append(Seg("beam", 3.5, [_slab(x, 3.5, top, "beam", half_w=0.16)]))
    x += 3.5
    flat(1.8)
    flat(3.0, kind="slick")                                # same geometry, low friction
    stairs(6, 0.18, 0.34, "stairs_dn")
    flat(4.0)                                              # final sprint to the line
    return segs, x, top


def course_xml_fragment(segs):
    out = []
    for s in segs:
        for b in s.boxes:
            cx, cy, cz, sx, sy, sz, ck = b[:7]
            euler = f' euler="0 {b[7]:.4f} 0"' if len(b) > 7 else ""
            out.append(f'    <geom type="box" pos="{cx:.3f} {cy:.3f} {cz:.3f}" '
                       f'size="{sx:.3f} {sy:.3f} {sz:.3f}"{euler} condim="3" '
                       f'friction="1 .1 .1" rgba="{COLOR[ck]}"/>')
    return "\n".join(out)


if __name__ == "__main__":
    segs, length, top = build_course()
    x = 0.0
    print(f"{'segment':12} {'x start':>8} {'length':>7}")
    for s in segs:
        print(f"{s.kind:12} {x:>8.2f} {s.length:>7.2f}")
        x += s.length
    zs = [b[2] + b[5] for s in segs for b in s.boxes if b[6] not in ("vault", "crawl")]
    print(f"\ntotal {length:.1f} m, final deck {top:.2f} m, vertical range {max(zs) - min(zs):.2f} m")
