# Humanoid Olympics design

## Goal

A winning submission should be a single recurrent, legs-only G1 policy that can run quickly,
stay inside a lane through a sustained corner, clear obstacles, and produce controlled jumps
under changing surface and wind conditions. It should be an all-round athletic controller, not a
collection of event-specific trajectories.

## Meet shape

Every round runs all six events. Attempts are grouped by event so the referee holds only one
compiled MuJoCo G1 model at a time. A normal round has four attempts per event (24 total):

| Event | maximum control steps | attempts | maximum calls |
|---|---:|---:|---:|
| 100 m | 1,200 | 4 | 4,800 |
| 400 m | 3,600 | 4 | 14,400 |
| 100 m hurdles | 1,900 | 4 | 7,600 |
| high jump | 900 | 4 | 3,600 |
| long jump | 1,000 | 4 | 4,000 |
| triple jump | 1,400 | 4 | 5,600 |
| **total** |  |  | **40,000** |

This is below the inherited 72,000-call control budget. The referee also stops scheduling new
attempts after 840 seconds, leaving time for it to write a result before the 900-second sandbox
limit. Every unscheduled attempt is retained in the fixed denominator as a zero-scored row.

## Course-relative simulation

`OlympicsSim` owns the shared G1 PD physics and delegates route state to the current event.
Straight events use the `+x` centreline. The 400 m uses an annular lane of radius
`400 / (2π)`, a route tangent for heading feedback, radial cross-track error, and signed wrapped
angle integration at every 2 ms physics step. A lap completes only after 400 m of forward route
distance while inside the lane.

The observation stays 104 floats so the inherited ONNX interface remains valid. Its heading and
lateral fields are now route-relative, and the height/overhead scan reaches 6 m ahead rather than
the previous 1.6 m horizon.

## Event rules

- Sprint and hurdles end at their 100 m finish. The ten physical hurdles rise from 0.55 m to
  1.15 m (with the last two at 1.00 m and 1.15 m); any robot contact ends that attempt.
- High jump has a physical horizontal bar. The pelvis must cross the bar plane above the selected
  clearance height while both feet have been unsupported for 40 ms, without touching the bar, and
  then make a supported far-side landing.
- Long jump has a runway, a 6 m real void, and a sand landing. A legal attempt leaves a narrow
  one-foot take-off board, remains airborne, and first regains sustained foot support on sand.
- Triple jump uses a runway, hop pad, step pad, and final sand pit. It requires a one-foot board
  take-off, same-foot hop landing, a real flight, opposite-foot step landing, another flight, and
  sustained final sand support. Any early/wrong-surface or non-foot contact is a foul.

All narrow gates, bar/hurdle contacts, and jump-phase contacts are sampled at each 500 Hz physics
substep rather than only once per 20 ms policy action.

Top-face foot contacts are the only supports accepted by a jump state machine. The controller must
leave a 0.40 m board (0.35 m before to 0.05 m after the take-off line) on one foot; side scrapes,
non-foot contacts, premature pad contacts, and a non-foot first sand contact foul the attempt.
Landing distance is latched from that first legal sand-contact point, not from a later pelvis pose.

## Scoring and calibration

Each attempt maps to `[0, 1]`; per-event means are then macro-averaged, so a long 400 m cannot
dominate five shorter disciplines. Incomplete attempts stay below 0.25 and valid finishes score
from 0.25 upward; quality within finished races is pace, high jump is selected bar height, and
horizontal jumps are legal distance.

Each launch round repeats the same public four-stratum friction/wind lattice, with opposing wind
directions. This makes the absolute score stable enough for a 1% takeover margin while still
requiring a controller to handle the full launch envelope. High-jump attempts cycle through 1.00,
1.10, 1.20, and 1.30 m bar heights above the deck.

The course's friction must be **authoritative for foot contacts**, and that is a property of the
model rather than of the numbers written into it. MuJoCo mixes contact parameters from both geoms in
a pair, and for friction the mix is the element-wise maximum whenever the two carry equal
`geom_priority`. `g1_12dof.xml` declares no geom friction, so the robot's feet sit at MuJoCo's
default of 1.0 — above most of the band this meet draws. Setting `geom_friction` on the course is
therefore necessary but not sufficient: the course geoms also carry `geom_priority = 1` so their
parameters win outright. Without it, 18 of the 24 launch attempts solve at exactly 1.0 while the
course asks for 0.52–0.98, collapsing three of the four strata into one at full grip. This is
asserted at contact level in `tests/test_friction_reaches_contacts.py`, deliberately not through a
score: a score cannot distinguish a band that applied from one that was mixed away, which is how the
defect passed 0.1.0's 20-seed calibration with a sample standard deviation of 0.0. The platform seed is retained in the request
contract but is intentionally score-neutral in v0.1; a future version can introduce fresh
conditions only after recalibrating its baseline.

The launch configuration is fixed across rounds: four attempts per event, four public condition
strata, 8 m/s maximum wind, 500 ms action deadline, and history stride 2. Accepting a different
event count, wind range, or seed-dependent condition mix would change the score distribution and
is a future versioned release.

The geometry and height band are implementation hypotheses until calibrated. Before the first
release, run a solvability probe for high and triple jump, measure full-round standard deviation
over at least 20 seeds for the baseline and two materially different policies, and remeasure
worst-case player latency inside the production image.
