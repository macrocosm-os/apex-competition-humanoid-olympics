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

**Decided: launch without arms.** If the competition stagnates — miners converge and the frontier
stops moving — arms are the escalation lever, shipped as a new competition rather than an update
to this one (the interface break makes that unavoidable). Until then, every obstacle is a leg
maneuver and the docs say so plainly.

The corollary is that obstacle sizing must be audited against *leg* capability, not against a
robot with hands. Measured, for the record:

| | |
|---|---|
| leg kinematic reach | 1.30 m (hip pitch spans ±2.88 rad) |
| knee torque limit | 139 N·m |
| hurdle | 0.62 m — 2.1x reach margin |
| step-up | 0.55 m — needs ~31-63 N·m, so 2.2-4.5x torque margin |
| duck bar | 1.05 m vs 1.26 m standing height — a ~0.2 m squat |

This audit is why the duck bar moved 0.75 m -> 1.05 m, and it is the check that was skipped on the
other two segments in v0.3.0: they shipped named "vault" and "climb-up", words that presuppose
arms. Renamed in v0.3.3; the geometry was fine, the names were not.

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
5. A fixed suite sets σ_round to **zero** instead. Verified in-image: four different `SEED` values
   all return the same score, bit for bit.

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

## Submission size cap: 25 MB

Set against measurement. Single-threaded ONNX Runtime on one CPU, MLPs over the full
`obs + state` input:

| arch | params | size | ms/step | 96,000 inferences |
|---|---|---|---|---|
| (64, 32) — the reference class | 34k | 0.13 MB | 0.008 | 1 s |
| (512, 512) | 585k | 2.2 MB | 0.040 | 4 s |
| (1024, 1024) | 1.7M | 6.5 MB | 0.106 | 10 s |
| (2048, 2048) | 5.5M | 21 MB | 0.352 | 34 s |
| (4096, 4096) | 19M | 74 MB | 1.430 | 137 s |

Re-measured on a GitHub amd64 runner (worker-class, and the number that matters): 0.012 / 0.062 /
0.156 / 0.472 / 2.581 ms per step for the same five architectures — roughly 1.5-1.8x slower. At
the 25 MB cap that is still ~45 s of inference against a ~640 s budget. `measure-baseline.yml`
runs this on every dispatch and fails if the cap ever admits a model the CPU cannot run in time.

The per-inference budget for a policy that survives every step is ~6.7 ms (900 s referee timeout,
less ~258 s of physics and HTTP on a worker-class amd64 CPU, over 24 x 4000 calls). So **compute does not bind**, even at
74 MB — an earlier draft of this spec justified a 100 MB cap on compute grounds and was simply
wrong.

25 MB is chosen because every Unitree reference locomotion policy — G1, H1, H1-2 — is
**0.13-0.14 MB**, so 25 MB is ~180x the class of policy this task needs, while still fitting a
6-layer d=256 transformer over a history window (~5M params). Capacity beyond that buys nothing
on a proprioception-plus-height-scan control problem, and it carries a specific cost here: the
evaluation suite is fixed, so spare parameters invite memorising 24 instances rather than
learning to walk, and that risk scales with parameter count.

## The scene is compiled once, not per instance

The G1's collision geometry is 27 STL meshes that MuJoCo converts to convex hulls at compile time.
Building a fresh `MjModel` per instance took the referee to **1098 MiB of a 1.5 GiB limit (71%)**,
against the skill's guidance that the baseline should sit under 50%. It did not OOM, but 438 MiB of
headroom on a hard limit is not somewhere to launch from.

Friction is the only thing that varies between instances, and `geom_friction` is a runtime field,
so the scene is now compiled once and friction written per instance. Peak memory drops to
**560 MiB (36%)**.

Two things this surfaced that are worth keeping:

- The change had to be proved, not assumed. Both paths were run over the full suite and the scores
  compared exactly — bit-identical, `0.2005765356172827` either way.
- Getting there exposed a real latent issue. Friction used to reach MuJoCo through the XML, which
  serialised it at `%.4f`; writing the field directly used full float64 precision, and that
  difference alone moved `raw_score` by **0.15%** — inside the 1% takeover margin. Friction values
  are now explicitly quantised to 4 dp in `sample_frictions` so precision is a stated property of
  the design rather than a side effect of a format string.

It did **not** save wall time, contrary to the expectation that drove the change: 31 s vs 29.8 s
for the suite. MuJoCo evidently caches mesh hull construction across compiles within a process.

## Rejected

- **Checkpoint scoring.** Continuous progress along a linear course already gives a smooth
  gradient; checkpoints add discontinuities and a tuning surface for no benefit.
- **Observation batching.** Considered for evaluation cost. Unnecessary — worst case is ~258 s on
  a native amd64 runner against a 900 s budget, 3.5× headroom.
- **Hands-on-obstacles rules.** Moot on a legs-only robot, and the contact-based gate it needed
  was fragile. The fall gate is now geometric: pelvis clearance above the surface below it, plus
  an uprightness check.
- **Energy budget / fatigue.** Prototyped and dropped. Robots do not tire while the battery
  lasts, and it added a scoring knob without adding difficulty.

## Open

1. **Does the platform re-score the incumbent leader each round?** With a fixed suite this is now
   a correctness question rather than a design blocker: if the incumbent's stored score was
   measured on different hardware, comparisons drift even though our suite does not.
2. **Is the worker fleet homogeneous in CPU generation?** The remaining risk, now narrowed.

   Across *architectures* scores move by amounts comparable to the 1% takeover margin:
   host-vs-image 0.04%, amd64-vs-arm64 0.12%. So `baseline_raw_score` is measured in the referee
   image on a native amd64 runner, which is what the platform runs.

   Across *machines of the same architecture* it appears to be exact. Two separate CI runs on
   two different GitHub amd64 runners both returned `0.20068353334086175` — bit-identical, not
   merely close. That is encouraging but not conclusive: hosted runners are likely the same CPU
   model, so this shows same-generation reproducibility, not cross-generation. If the worker
   fleet spans generations with different FMA or vector-width behaviour, the same policy could
   score differently on different workers, and nothing on our side fixes that.
