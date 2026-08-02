# Humanoid Parkour

An Apex competition (Bittensor Subnet 1) where miners submit an **ONNX policy** that drives a
Unitree G1 humanoid through a procedurally generated parkour course — stairs, gaps, vaults,
climb-ups, a crawl-under, a balance beam, and hidden low-friction patches.

**Status: design and calibration. Not implemented.**

Read [`docs/v0.3.0-design.md`](docs/v0.3.0-design.md) — it is the spec for what gets built, with
every number in it measured rather than assumed.

## The course

51.1 m, linear, on a raised plinth so gaps are real voids. Difficulty ramps along its length, so
progress-based scoring gives a continuous gradient instead of discrete tiers.

| Maneuver | Geometry |
|---|---|
| on-ramp | 6 m flat, 15.4° climb over 2 m, 0.55 m sheer drop |
| stairs up / down | rise 0.18–0.20, run 0.32–0.34 |
| leap | 1.0 m void |
| drop-down | 0.6 m |
| vault | waist-high barrier |
| climb-up | 0.55 m platform (hip height for G1) |
| crawl-under | overhead bar at 0.75 m |
| balance beam | 0.32 m wide, 3.5 m long |
| slick patch | low friction, geometry identical to flat |

```bash
python tools/course_v3.py     # print the layout
python tools/preview_v3.py    # stills + flythrough (needs mujoco + ffmpeg)
```

`preview_v3.py --walk` additionally drives the course with Unitree's stock G1 walking policy — the
probe used to calibrate difficulty. See its docstring for the extra checkouts that needs.

## Key decisions

- **Unitree G1** from `mujoco_menagerie` (29 DoF, 33.3 kg, 1.26 m). Built on the `g1_mjx.xml`
  lineage, which has tuned collision geoms, realistic PD gains, and a hardware transfer via MuJoCo
  Playground. Actions are **joint position targets**, not torques.
- **ONNX submissions, free architecture.** The tensor signature is fixed; what is inside the graph
  is not. Optional recurrent state, since friction is hidden and adaptation needs memory.
- **Perception is a height scan**, not an obstacle oracle: 11×7 downward samples plus an 11-point
  overhead clearance scan. Friction and segment identity are **not** observable.
- **Arms allowed on obstacles, not on the ground** — permits vaulting and climbing, forbids
  bear-crawling the course.

## Calibration

Difficulty was set by driving a real policy over the course, not by taste:

- The stock G1 walker **climbs 15.4° and stalls at 20.1°**, so the on-ramp sits at 15.4° — the
  steepest thing a naive policy can still do.
- **Drop height is nearly free**: 0.20 m and 0.55 m end its run at the same place, because a
  flat-ground walker has no landing controller. The on-ramp takes the full 0.55 m.
- That policy needs **heading hold** to be usable at all — it tracks body-frame velocity with no
  heading feedback and drifts 0.26 m sideways per metre travelled.
- Resulting reference score: **21% progress**, dying on the on-ramp landing.

## Open questions

Tracked in [#1](https://github.com/macrocosm-os/apex-competition-humanoid-parkour-v2/issues/1) and
[apex-competitions-builder#26](https://github.com/macrocosm-os/apex-competitions-builder/issues/26).
The two that block design:

1. **Does the platform re-score the incumbent leader each round?** Adaptive difficulty — the thing
   that stops the competition having a fixed ceiling — is only coherent if it does.
2. **Is the worker fleet homogeneous in CPU generation?** A 900-step MuJoCo rollout is chaotic
   enough that a 1-ulp difference flips a completion.

## History

An earlier, much simpler version of this competition (flat plane, step-over hurdles, Gymnasium
humanoid) was built, released and signed as **`v0.2.0`**. It is preserved at the
[`v0.2.0`](https://github.com/macrocosm-os/apex-competition-humanoid-parkour-v2/tree/v0.2.0) tag
with its images in GHCR, and is not part of this codebase. The predecessor repo
`apex-competition-humanoid-parkour` is archived.
