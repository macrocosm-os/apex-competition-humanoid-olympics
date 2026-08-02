# Design decisions

What the code does is in the code. This records the things that are *not* visible there: what was
measured, what was rejected, and what is still open.

## Robot: G1 12-DoF, and why not 29

The competition ships Unitree's `g1_12dof` — 12 actuated leg joints, with the upper body welded
to the pelvis as rigid mass and collision geometry. The arms are physically present (17.7 kg of
the 32.1 kg total, 29 geoms) and they hit things. They just do not move.

The alternative was the 29-DoF Menagerie G1, which would have allowed arm motion, active vaulting
and true climbing. It was rejected on one measurement: **there is no working policy for it.** The
stock walker transfers to 29-DoF not at all — with the upper body held at neutral across stiffness
kp ∈ {40, 150, 400, 1000, 3000}, every configuration fell within 1.6 s, against 14.5 m upright on
the 12-DoF model. The gap is embodiment (+3 kg, CoM 2.4 cm higher, 17 extra articulated masses),
not a tuning knob.

**This is a one-way door and should be understood as one.** Arms are not "unlockable" later:
`g1_12dof.xml` has no arm joints to enable, and switching model changes `nq`/`nv`/`nu`, which
changes the observation and action dimensions, which invalidates every submitted policy and
requires a new `(id, version)`. Shipping 12-DoF means legs-only until the competition is replaced.

The trade taken: a demonstrated-solvable launch with a real baseline, over an unverifiable launch
with arms.

## On-ramp calibrated against a real policy, not taste

Difficulty was set by driving the stock walker over candidate geometry:

- it **climbs 15.4° and stalls at 20.1°**, so the on-ramp sits at 15.4° — the steepest short climb
  a naive policy can still manage;
- **drop height is nearly free**: 0.20 m and 0.55 m end its run in the same place, because a
  flat-ground walker has no landing controller at all. The on-ramp takes the full 0.55 m, since
  the spectacle is free;
- it needs **heading hold** to be usable as a probe — it tracks body-frame velocity with no
  heading feedback and drifts 0.26 m sideways per metre travelled.

That last point cost a misdiagnosis worth recording: an early sweep concluded the walker "falls on
a 0.10 m step" and nearly led to replacing the stairs with a ramp. It was lateral drift off the
track. The tell was the control case — flat ground failed at the identical distance.

## Duck-under at 1.05 m, not 0.75 m

The original design put the overhead bar at 0.75 m. A legs-only G1 cannot clear that: standing
head height is 1.26 m, and a deep squat only brings it to ~0.9 m. A segment no embodiment can
pass is a wall, not an obstacle. 1.05 m forces a ~0.2 m squat-walk, which is achievable and still
reads as a duck on playback.

## Fixed evaluation suite

The single most consequential decision, and the one most likely to be questioned in review.

Instances are a pure function of `(index, count)` — not of the platform's per-round seed. The
reasoning chain:

1. The course is static and public, so a per-round seed buys **no secrecy**.
2. It does buy score noise. Measured per-instance stdev is **0.0176**.
3. The takeover margin is 1% of the baseline: **0.002**.
4. The sizing criterion σ_round ≤ margin/4 would need **~1400 instances**. At ~1.14 ms per control
   step that is ~48 minutes, against a 900 s referee timeout. It does not fit.
5. A fixed suite sets σ_round to **zero** instead. Verified in-image: `SEED=777` and `SEED=999888`
   both return `0.2004409785`, bit for bit.

The cost is that the suite is memorisable. This is accepted because the *course* is already
memorisable — it is static and public by design — so the marginal loss is small, while the gain
(takeover decided by skill rather than by which instances a round drew) is large. Friction levels
are stratified across the range rather than drawn randomly, so 24 instances cover the whole
grippy-to-slippery continuum.

If the platform later re-scores the incumbent every round against the same suite, this all holds.
If it compares scores across differently-seeded rounds, it would not have — which is why the suite
does not depend on the seed.

## Recurrence is required by the design, not a nicety

Friction varies per instance and is not observable. The only way to adapt to a slick patch is to
remember having slipped on it. So the interface carries an opaque 256-float state vector, zeroed
on reset and threaded by the player between `/act` calls.

This also fell out of the baseline: the stock walker **is** an LSTM. A feed-forward-only contract
would have made the reference policy unrepresentable.

## Rejected

- **Checkpoint scoring.** Continuous progress along a linear course already gives a smooth
  gradient; checkpoints add discontinuities and a tuning surface for no benefit.
- **Observation batching.** Considered for evaluation cost. Unnecessary — worst case is 109 s
  against a 900 s budget, 8× headroom.
- **Hands-on-obstacles rules.** Moot on a legs-only robot, and the contact-based gate it needed
  was fragile. The fall gate is now geometric: pelvis clearance above the surface below it, plus
  an uprightness check.
- **Energy budget / fatigue.** Prototyped and dropped. Robots do not tire while the battery
  lasts, and it added a scoring knob without adding difficulty.

## Open

1. **Does the platform re-score the incumbent leader each round?** With a fixed suite this is now
   a correctness question rather than a design blocker: if the incumbent's stored score was
   measured on different hardware, comparisons drift even though our suite does not.
2. **Is the worker fleet homogeneous in CPU generation?** Scores are bit-identical within one
   image on one architecture, but host-vs-image differs by 0.08% and amd64-vs-arm64 by ~0.2% —
   both comparable to the 1% takeover margin. `baseline_raw_score` is therefore measured inside
   the referee image, which handles the first case but not the second.
