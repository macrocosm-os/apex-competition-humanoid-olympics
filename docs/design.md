# Humanoid Olympics design

## Goal

A winning submission should be a single recurrent, legs-only G1 policy that can run quickly,
stay inside a lane through a sustained corner, clear obstacles, and produce controlled jumps
under changing surface and wind conditions. It should be an all-round athletic controller, not a
collection of event-specific trajectories.

## Meet shape

Every round runs all seven events. Attempts are grouped by event so the referee holds only one
compiled MuJoCo G1 model at a time. A normal round has four attempts per event (28 total):

| Event | maximum control steps | attempts | maximum calls |
|---|---:|---:|---:|
| 100 m | 1,200 | 4 | 4,800 |
| 400 m | 3,600 | 4 | 14,400 |
| 100 m hurdles | 1,900 | 4 | 7,600 |
| high jump | 900 | 4 | 3,600 |
| long jump | 1,000 | 4 | 4,000 |
| triple jump | 1,400 | 4 | 5,600 |
| 200 m race walk | 3,600 | 4 | 14,400 |
| **total** |  |  | **54,400** |

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

### Barriers must be visible (0.5.0)

A meet that ends an attempt on contact has to let the policy see what it is about to hit. Through
0.4.0 it did not. Hurdles and the high-jump bar are `walkable=False` so they stay out of the
downward terrain scan — that mask is `WORLD_GROUP` alone and must remain so, because `_ray_down`
also feeds the `FALL_CLEARANCE` gate. The seven forward channels were meant to cover them, but
they were single upward rays cast from `pelvis + 0.05` (1.643 m at reset), and six of the ten
hurdles top out between 1.35 m and 1.60 m. Those six returned nothing on any of the 104 channels:
measured, `hurdles_100` was byte-identical to `sprint_100` for the first 60.9 m of the race, and
the four taller hurdles arrived as 0.2 m flickers because seven point rays sample every 0.667 m
against a 0.24 m obstacle.

The channels now report barriers, sampled four times per bin so the spacing (0.167 m) is finer
than the thinnest barrier in the meet. All ten hurdles are visible from 4.6 m out and stay visible
through the approach, with their height reported to 0.000 m.

**The encoding is deliberately drop-in (0.5.1).** `SCAN_CLIP` (2.0) still means nothing ahead —
byte-identical to what the 0.4.0 upward ray returned on a miss — and a barrier subtracts its height
above the ground beneath it. 0.5.0 instead reported a pelvis-relative terrain-scale height, which
moved the channel to about -0.79 on clear ground: a distribution shift on every step of every
event. Measured on a real 0.4.0 submission, that took a meet score of 0.782833 to 0.000370, falling
on all 24 attempts. Under this encoding the same submission scores 0.684274, with five of the six
events bit-identical to 0.4.0 and only hurdles moving — which is the event the new information is
in, and no encoding that reveals a hurdle can leave it untouched.

`tests/test_barriers_are_visible.py` asserts both halves at the observation level rather than
through a score, for the reason the friction test gives: a score cannot tell a barrier a policy
could not see from one it saw and failed to clear. Cost is 49 extra ray casts per observation,
about 0.04 ms per control step, or 2 s across a 40,000-call meet against the referee's 840 s
budget.

### Telling a policy its discipline (0.5.0)

Five of the six events were already identifiable from the observation's `remaining / 10` channel
at the first step (400 m reads 40.0, the sprints 10.2, triple jump 2.7, long jump 2.3, high jump
1.4). `sprint_100` and `hurdles_100` share a start, a finish and a lane, so they separate only
once a barrier comes into range — about 9 m in, even with the sensing fixed.

`event_type` closes that gap explicitly rather than leaving the identity smuggled through a
distance channel, where moving a finish line would silently shift it under every trained policy.
The referee sends the event NAME in `player.reset`'s config and the player builds the one-hot, so
the index order is a contract between the two; `player/launch.py` repeats the order because the
player image does not ship `env/`, and `tests/test_event_type_input.py` pins them together.

It is optional in the load check, not merely tolerated: a two-input policy is the entire live
leaderboard and keeps scoring unchanged. What the input deliberately does NOT carry is anything
the round seed moves — friction, wind, and the bar height still have to be sensed, which is what
`tests/test_seed_is_not_disclosed.py` now enforces on the reset config.

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
from 0.25 upward; quality within finished races is pace, high jump is the height reached, and
horizontal jumps are legal distance.

### No event caps (0.6.0)

Through 0.5.1 three of the six events saturated. Long jump paid nothing past 12 m, triple jump
nothing past 18 m, and high jump was scored on which bar was up — a fixed four-rung ladder, so
every policy that cleared all four scored exactly 0.625. Half the meet therefore carried no
information about who was better, all differentiation ran through three race events, and a field
that converged could not be displaced: taking the flag needs +1%, against a measured field spread
of 0.18% and an estimated total remaining race headroom of about 1.85%.

A reference performance now earns `REFERENCE_FRACTION` (0.8) of the 0.75 margin band and the curve
keeps rising beyond it with diminishing returns, so being better always scores better and nothing
reaches a shared ceiling. The references are the old caps — 12 m, 18 m, and the top bar — so the
scale below them is nearly unchanged; what changes is that they are no longer the end of it. High
jump moves to measured clearance because the ladder is fixed and a bar-indexed score cannot
separate two policies that both clear it.

Races are untouched: pace already scaled continuously and never capped.

Each launch round repeats the same public four-stratum friction/wind lattice, with opposing wind
directions.

0.3.0 widens the friction band from `(0.50, 1.25)` to `(0.30, 1.25)`. The intent is a bottom stratum
that is a genuine slip regime rather than a merely-imperfect surface. Two consequences are worth
stating plainly, because neither is obvious from the one-line change:

- A stratum is a fraction of the band, so **every** stratum moves, not only the last. Per event the
  four nominal values go from roughly 1.08/0.90/0.71/0.52 to 1.04/0.80/0.57/0.33, and the relative
  ±8% per-slab jitter widens from ±0.060 to ±0.076.
- Solvability at the low end is a **hypothesis, not a measurement**. This course has no
  parkour-style on-ramp imposing a hard mu floor, so nothing becomes trivially impossible — but a
  long-jump take-off converts a fast approach into vertical impulse through one foot, and at mu
  ~0.38 that is the marginal case. The 0.2.0 note that "there is no trained Olympics policy yet, so
  the band's effect on a COMPETENT controller is unmeasured" applies with more force here. Probe
  jump solvability across the new low strata before release, and expect low-stratum jump attempts to
  fall back to partial progress credit rather than scoring a legal landing. This makes the absolute score stable enough for a 1% takeover margin while still
requiring a controller to handle the full launch envelope. High-jump attempts cycle through 1.00,
1.10, 1.20, and 1.30 m bar heights above the deck.

The course's friction must be **authoritative for foot contacts**, and that is a property of the
model rather than of the numbers written into it. MuJoCo mixes contact parameters from both geoms in
a pair, and for friction the mix is the element-wise maximum whenever the two carry equal
`geom_priority`. `g1_12dof.xml` declares no geom friction, so the robot's feet sit at MuJoCo's
default of 1.0 — above most of the band this meet draws. Setting `geom_friction` on the course is
therefore necessary but not sufficient: the course geoms also carry `geom_priority = 1` so their
parameters win outright. Without it, 18 of the 24 launch attempts solve at exactly 1.0 while the
course asks for less — measured as 0.52–0.98 under 0.2.0's `(0.50, 1.25)` band, and further below
1.0 under 0.3.0's `(0.30, 1.25)` — collapsing three of the four strata into one at full grip. This is
asserted at contact level in `tests/test_friction_reaches_contacts.py`, deliberately not through a
score: a score cannot distinguish a band that applied from one that was mixed away, which is how the
defect passed 0.1.0's 20-seed calibration with a sample standard deviation of 0.0. The platform seed drives the conditions as of 0.4.0: the
friction and wind strata are phase-shifted per round and per event, so the meet's envelope is
constant while its operating points move. The seed itself is emitted on no miner-visible surface —
not `player.reset` (whose config carries the event name and nothing else), not the observation,
not `result.json`, not the history files — which
`tests/test_seed_is_not_disclosed.py` enforces by scanning each surface's serialized JSON for the
value rather than for a field name. Because the realized conditions ARE reported, a guessed seed can
still be confirmed against a closed round by re-deriving the meet from this public repo; the secrecy
therefore rests on the platform drawing seeds unpredictably rather than from a round counter — a
property this repo cannot enforce, since the seed to conditions mapping ships in a public image, and
which is worth confirming with the platform before this activates. This is deliberately paired with a `baseline_raw_score`
of 0.0 -- a measured baseline is not a well-defined quantity once conditions vary, and pinning one
would put the entry bar above the true baseline in about half of all rounds.

The meet's SHAPE is fixed across rounds: four attempts per event, four evenly spaced condition
strata covering the full band, 8 m/s maximum wind, 500 ms action deadline, and history stride 2.
`tests/test_round_conditions_vary.py` asserts the shape survives the phase shift -- even cyclic
spacing of both strata, opposed wind pairs, an unmoved high-jump bar ladder, and exact
reproducibility per seed. Changing the event count, the wind range, or the stratification itself
would change the score distribution and is a future versioned release.

The cost of per-round conditions is that absolute scores compare only within a round. On the
baseline artifact the cross-round spread is CV ~1.1%, which is the same order as the 1% takeover
threshold; the reason that is not decisive is that the incumbent is re-scored on the challenger's
own round seed, so the condition draw is common-mode in the comparison that settles the round.

The geometry and height band are implementation hypotheses until calibrated. Before the first
release, run a solvability probe for high and triple jump, measure full-round standard deviation
over at least 20 seeds for the baseline and two materially different policies, and remeasure
worst-case player latency inside the production image.


### Race walk and drawn hurdles (0.7.0)

**`race_walk_200`** is a flat 200 m under a contact rule: at every 500 Hz substep at least one foot
must be on the track, and flight beyond `MAX_FLIGHT_STEPS` (40 ms, the same tolerance the jump
gates use) ends the attempt as `lost_contact`. It scores like a race — completion plus pace, with
partial progress credit otherwise — so a foul keeps only the distance walked before it. The point
is an event where speed has to come from gait: momentum carried through the air pays nothing.

**Hurdle placement and height order are now drawn per round.** The ten heights are fixed as a set
and shuffled, and positions are drawn inside a 10-90 m window on a 6 m minimum gap, so a round is a
rearrangement rather than a harder or easier meet. Through 0.6.0 the hurdles sat on a fixed lattice
with monotonically rising heights, which meant position alone predicted the next height and the
0.5.1 barrier channels added nothing a policy could not already infer. They are now load-bearing.

Two consequences. `event_type` widens to `[batch, 7]`, so a policy that declares it must be
re-exported; two-input policies are unaffected. And the meet's macro-average now divides by seven,
so every score moves even where behaviour does not.