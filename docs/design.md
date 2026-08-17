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
| 100 m | 2,500 | 4 | 10,000 |
| 400 m | 5,000 | 4 | 20,000 |
| 100 m hurdles | 3,000 | 4 | 12,000 |
| high jump | 1,200 | 4 | 4,800 |
| long jump | 1,800 | 4 | 7,200 |
| triple jump | 2,400 | 4 | 9,600 |
| **total** |  |  | **63,600** |

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

- Sprint and hurdles end at their 100 m finish. Hurdles are physical 0.55 m barriers; contact
  ends that attempt.
- High jump has a physical horizontal bar. The pelvis must cross the bar plane above the selected
  clearance height without touching it.
- Long jump has a runway, a 4 m gap, and a sand landing. The first supported final landing
  determines the measured distance.
- Triple jump uses a runway, hop pad, step pad, and final sand pit. The hop and step contacts
  must occur in order, with the step on the opposite foot, before the final landing counts.

All narrow gates, bar/hurdle contacts, and jump-phase contacts are sampled at each 500 Hz physics
substep rather than only once per 20 ms policy action.

## Scoring and calibration

Each attempt maps to `[0, 1]`; per-event means are then macro-averaged, so a long 400 m cannot
dominate five shorter disciplines. Incomplete attempts stay below 0.25 and valid finishes score
from 0.25 upward; quality within finished races is pace, high jump is selected bar height, and
horizontal jumps are legal distance.

Round conditions are derived deterministically from one seed. Within each event the friction and
wind samples use a shifted lattice with opposing wind directions, reducing evaluation variance
without freezing a known suite. High-jump attempts cycle through 0.80, 0.92, 1.04, and 1.16 m
bar heights above the deck.

The geometry and height band are implementation hypotheses until calibrated. Before the first
release, run a solvability probe for high and triple jump, measure full-round standard deviation
over at least 20 seeds for the baseline and two materially different policies, and remeasure
worst-case player latency inside the production image.
