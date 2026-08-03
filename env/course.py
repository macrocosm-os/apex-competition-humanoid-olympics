"""The Humanoid Parkour course: one fixed layout, per-instance surface friction.

The geometry is STATIC and public — every instance in every round runs this exact course.
Randomising the layout was considered and dropped: layout noise adds far more score variance
than it removes memorisation risk, and round-to-round variance is what sets the takeover margin
(docs/design.md, "Fixed evaluation suite"). What DOES vary per instance is the sliding friction
of every surface, and it is deliberately NOT observable — a policy has to feel the slip and
adapt rather than read a number.

Built as MJCF box geoms on a raised plinth, sized for the Unitree G1 (1.26 m tall, pelvis at
0.784 m). Gaps are real voids in the plinth, so a missed leap is a fall, not a stumble.

    python -m env.course        # print the layout
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

PLINTH_TOP = 0.8            # top surface of the track; gaps drop through it
PLINTH_THICK = 0.4
TRACK_HALF_W = 1.2

# Every course geom goes in this MuJoCo render group, and nothing else does. That is what lets
# the height scan ray-cast against the world while ignoring the robot's own body: mj_ray filters
# by group, not by geom. Group 2 is used because the default visualiser shows 0-2, so the course
# still renders without special options; the robot occupies groups 0 (collision) and 1 (visual).
WORLD_GROUP = 2

# Course geoms are named "course_<i>" in emission order, so friction can be set on the compiled
# model rather than baked into the XML.
GEOM_PREFIX = "course_"

# On-ramp shape, measured against the stock G1 walker (see docs/design.md): it climbs
# 15.4 deg but stalls at 20.1 deg, so this is the steepest short climb a naive policy can still
# manage. Drop height barely matters — 0.20 m and 0.55 m both end its run — so we take the full
# drop for the spectacle.
ON_RAMP_RISE, ON_RAMP_RUN, ON_RAMP_DROP = 0.55, 2.0, 0.55

# Overhead bar height. The G1 stands 1.26 m, so this forces a ~0.2 m squat-walk. It is NOT a
# crawl: with 12 leg DoF and a welded upper body the robot cannot get its head under anything
# much lower, and a segment no embodiment can clear is just a wall.
DUCK_BAR_Z = 1.05

# Per-instance sliding friction. Normal surfaces vary enough to punish a policy that has
# memorised one contact model; the slick patch is a different regime entirely.
FRICTION_NOMINAL = (0.7, 1.1)
FRICTION_SLICK = (0.12, 0.30)

COLOR = {  # by maneuver, so a render is readable at a glance
    "flat": ".55 .57 .60 1", "ramp": ".45 .80 .55 1",
    "stairs_up": ".30 .65 .45 1", "stairs_dn": ".22 .50 .38 1",
    "climb_up": ".85 .45 .20 1", "drop_down": ".70 .35 .18 1",
    "leap": ".20 .20 .24 1", "vault": ".80 .25 .35 1",
    "duck": ".55 .30 .75 1", "beam": ".95 .75 .20 1", "slick": ".35 .70 .95 1",
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

    s = Seg("duck", 2.0, [_slab(x, 2.0, top, "flat"),      # overhead bar on posts
                          (x + 1.0, 0.0, top + DUCK_BAR_Z + 0.08, 0.5, TRACK_HALF_W, 0.08, "duck")])
    for sy in (-1.0, 1.0):
        s.boxes.append((x + 1.0, sy * (TRACK_HALF_W - 0.06), top + DUCK_BAR_Z / 2,
                        0.06, 0.06, DUCK_BAR_Z / 2, "duck"))
    segs.append(s); x += 2.0
    flat(1.6)

    segs.append(Seg("beam", 3.5, [_slab(x, 3.5, top, "beam", half_w=0.16)]))
    x += 3.5
    flat(1.8)
    flat(3.0, kind="slick")                                # same geometry, low friction
    stairs(6, 0.18, 0.34, "stairs_dn")
    flat(4.0)                                              # final sprint to the line
    return segs, x, top


SEGMENTS, COURSE_LENGTH, FINAL_DECK = build_course()


def course_xml_fragment(segs, frictions=None):
    """MJCF for the course. `frictions` is one sliding-friction value per emitted geom, in the
    same order this function walks them — see `sample_frictions`."""
    out, i = [], 0
    for s in segs:
        for b in s.boxes:
            cx, cy, cz, sx, sy, sz, ck = b[:7]
            euler = f' euler="0 {b[7]:.4f} 0"' if len(b) > 7 else ""
            mu = 1.0 if frictions is None else frictions[i]
            # Named so the sim can set friction on the compiled model instead of recompiling
            # it per instance. Emission order is the contract with `sample_frictions`.
            out.append(f'    <geom name="{GEOM_PREFIX}{i}" type="box" '
                       f'pos="{cx:.3f} {cy:.3f} {cz:.3f}" '
                       f'size="{sx:.3f} {sy:.3f} {sz:.3f}"{euler} condim="3" group="{WORLD_GROUP}" '
                       f'friction="{mu:.4f} .1 .1" rgba="{COLOR[ck]}"/>')
            i += 1
    return "\n".join(out)


def sample_frictions(segs, level: float, rng: np.random.Generator) -> list[float]:
    """One sliding friction per geom, in `course_xml_fragment` order.

    `level` in [0, 1] slides the whole course from the grippy end of its range to the slippery
    end; `rng` adds a little per-geom jitter on top so no two slabs are exactly alike. Slick
    slabs use their own, much lower range. The split matters: `level` is what the evaluation
    suite STRATIFIES over (env/sim.py), so a fixed set of instances covers the whole friction
    continuum evenly instead of sampling it at random.
    """
    out = []
    for s in segs:
        lo, hi = FRICTION_SLICK if s.kind == "slick" else FRICTION_NOMINAL
        base = hi - (hi - lo) * float(level)
        for _ in s.boxes:
            jitter = (hi - lo) * 0.08 * float(rng.uniform(-1.0, 1.0))
            # Rounded to 4 dp deliberately. These values used to reach MuJoCo through the XML,
            # which serialised them at %.4f; they are now written straight into geom_friction.
            # Quantising here makes the two paths agree exactly instead of differing by the
            # serialisation rounding, which was worth 0.15% of raw_score -- inside the 1%
            # takeover margin, so not something to leave to a format string.
            out.append(round(float(np.clip(base + jitter, lo, hi)), 4))
    return out


if __name__ == "__main__":
    x = 0.0
    print(f"{'segment':12} {'x start':>8} {'length':>7}")
    for s in SEGMENTS:
        print(f"{s.kind:12} {x:>8.2f} {s.length:>7.2f}")
        x += s.length
    zs = [b[2] + b[5] for s in SEGMENTS for b in s.boxes if b[6] not in ("vault", "duck")]
    print(f"\ntotal {COURSE_LENGTH:.1f} m, final deck {FINAL_DECK:.2f} m, "
          f"vertical range {max(zs) - min(zs):.2f} m, {sum(len(s.boxes) for s in SEGMENTS)} geoms")
