# Handoff — `humanoid_parkour` v0.5.0

For platform review. Covers the build checklist, the places this competition deliberately
deviates from the skill's guidance, and what is measured versus assumed.

## Success statement

**A winning solution is a terrain-aware locomotion policy that reads the height scan and adapts
its gait to what is coming — not one that has memorised a fixed sequence of joint targets.**

Alignment checks, to run against top submissions each round:

0. **Is it a leg maneuver?** Every obstacle must be cleared with legs — the robot has no arm
   joints. If a submission appears to clear the hurdle or step-up in a way that requires arm
   contact, that is a physics exploit, not a solution; check the playback.
1. **Does it use perception?** Zero the height-scan channels (obs `[52:97]`) and re-score. A policy
   that relies on terrain should collapse toward the baseline; one that is replaying a trajectory
   will barely change. Less load-bearing than it was in 0.3.3 — randomised conditions already make
   pure open-loop replay unprofitable — but still the cheapest read on whether the scan is used at
   all, since the geometry is still fixed and still public.
2. **Does it generalise off-suite?** Re-score at a different round seed, and outside the published
   bands (µ below 0.50, wind above 8 m/s). Real locomotion degrades gracefully; a policy fitted to
   the band edges falls off a cliff. Diagnostic only — ranking on conditions the published
   distribution excludes would not be fair.
3. **Does it look like locomotion?** Watch the playback (`tools/preview.py --run`). Gait should be
   recognisable and recoverable. Ballistic dives that bank progress before falling clear the metric
   without embodying the goal.

## Deviations from the skill's guidance

### 1. Conditions rotate per round; the course does not — and σ_round is no longer zero

**Reversed in 0.4.0.** 0.3.3 declined `reference/evaluation-design.md` Defense 1 and made the suite
a pure function of `(index, count)`, on the argument that a static public course means a per-round
seed buys no secrecy and only buys noise. The argument was wrong in a specific way: what a fixed
suite gives away is not the course but the whole evaluation. Geometry, friction, reset noise and
step counts were all computable offline from this repo, so the cheapest route to the top was
offline trajectory optimisation plus replay — 24 open-loop solutions, ~2.3 MB in fp16, inside a
25 MB cap (1.7 MB at 0.5.0's 3000-step cap). That beats a real policy on the metric while
embodying none of the success statement.

0.4.0 draws **friction and wind per instance from the round seed** (`env/sim.instance_spec`).
Geometry is unchanged: still static, still public. The competition owner's constraint was to keep
the course as-is, and it also keeps difficulty stable and the diff small.

**What this costs, stated plainly: σ_round is no longer zero, and 0.3.3's own sizing argument now
applies to us.** Taking the measured per-instance stdev of 0.0176 as a stand-in, 24 independent
draws give σ_round ≈ 0.0036 against a 1% takeover margin of ~0.002 — the margin sits *inside* the
noise, so the top slot can random-walk. Wind widens it further. Mitigations, neither of which
replaces measuring it:

- The platform re-scores the incumbent on the **same round input** as its challengers, so pairing
  cancels the round-difficulty main effect. It does not cancel the policy×condition interaction.
- Only friction and wind vary. Holding geometry fixed keeps that interaction much smaller than
  full course randomisation would.

**Two things must happen before this ships:**

1. **Measure σ_round** over ≥20 seeds with a policy that gets deep into the course, then set the
   takeover threshold (or `num_instances`) against it. The 0.0036 above is extrapolated from a
   baseline whose variance is bimodal — it either clears the on-ramp or does not — so treat it as
   an order of magnitude, not a result.
2. **Confirm the platform injects a fresh `seed` into the round input every round.** The whole
   property rests on it. `seed` is `required` in `input.schema.json` so a missing one fails loudly
   rather than silently freezing the suite, and the referee prefers the round input over the
   platform's `SEED` env — but nothing on our side detects the same seed being sent twice.

### 2. `submission_reveal_days: 5` (production default is 1, range 1–7)

Trained locomotion policies carry real R&D. Per the skill's own guidance ("4–7 days where a
winning solution embodies real IP"), 5 days sits in range.

### 3. No stage validation yet

The full loop has been exercised locally by hand and in CI, but not on stage. That is the one
checklist item this repo cannot close on its own.

## Measured

All numbers from the referee image, at `fixtures/input.json`
(`num_instances: 24`, `max_steps_per_episode: 3000`, `deadline_ms: 500`).

Score rows are v0.4.0, re-measured after randomised conditions landed. Memory rows are 0.3.3 and
unaffected in kind: wind adds no allocation and no rays, and history buffers one episode at a time.

| | |
|---|---|
| `baseline_raw_score` | **0.2150** — mean of 3 seeds, native amd64 (see below) |
| completions | 0 of 24 — furthest 10.97–11.31 m of 51.1 m |
| eval wall time | 74–80 s on a CI amd64 runner; **207 s in the PR environment** at cpu_limit 2 |
| worst case | ~750 s vs the 900 s timeout, at the 15 MB cap and 3000 steps (see below) |
| referee peak memory | **560 MiB of 1536 (36%)**, measured under `--memory 1.5g` (0.3.3) |
| player peak memory | 30 MiB of 1536 (2%) (0.3.3) |
| per-instance stdev | 0.0034–0.0164 depending on seed |
| determinism | bit-identical across two separate amd64 CI runners at a fixed seed |

### σ_round, and why `baseline_raw_score` is provisional

v0.4.0 draws conditions per round, so the baseline is a random variable. Measured in the referee
image on native amd64 (runs 31145125337 / 31145130114 / 31145134945):

| seed | raw_score |
|---|---|
| 1 | 0.21575696094883565 |
| 2 | 0.21245392681077288 |
| 3 | 0.21668923138720390 |

mean **0.21497**, σ_round **0.0022**, standard error **0.0013**. A 12-seed host-side sweep agrees
(σ_round 0.0025; host mean differs from the in-image mean by 0.00003, far inside the noise), so
σ_round is real and close to the 0.0036 this document originally extrapolated.

**This is not enough for stage or prod.** The standard error is ~62% of the ~0.0021 takeover
margin, so the bar could be set off by most of a takeover step in either direction. Three seeds
is sized for PR-environment plumbing only. Re-run across ≥20 seeds (standard error ~0.0006)
before promoting. The caveat above still stands: the baseline's variance is bimodal — it either
clears the on-ramp or does not — so a policy that gets deep into the course may vary more.

Architecture sensitivity, all inside the 1% takeover margin — which is why the spec figure is
pinned to amd64-in-image:

| measured | raw_score |
|---|---|
| referee image, native amd64 — **the spec figure** | 0.20068 |
| host, no container (arm64) | 0.20058 |
| referee image, arm64 build | 0.20044 |

## Security checklist

Walked against `reference/security-checklist.md`:

- **§1 revealed data** — post-round metadata is per-instance score, friction level and µ, wind speed
  and direction, terminal reason, progress, distance, steps. These are no longer derivable from
  public code, so they are a real disclosure — but only of a round that has closed, and miners need
  them to know what they were scored on. **The round seed is deliberately NOT reported**: rounds may
  draw seeds from a predictable sequence, so publishing one round's seed could hand out the next
  round's conditions. Nothing else about scoring internals is revealed.

  The per-instance history files (§5) reveal the same conditions plus the **per-geom** friction
  array and the miner's own trajectory and actions. The seed is excluded from them for the reason
  above. Note this does not open a new class of leak: the reported `friction_level`, `wind_speed`
  and `wind_dir` already let a candidate round seed be confirmed by re-deriving the suite and
  comparing, so the seed's secrecy rests on it being unguessable, not on withholding conditions.
  That is a property of the 0.4.0 metadata, not of history — but it is worth checking that round
  seeds are drawn unpredictably, not sequentially, before this ships.
- **§2 cross-miner** — solo competition. Referee builds fresh `MjData` per instance and holds no
  state keyed on anything a submission controls. The player zeroes policy state on every `/reset`,
  so memory cannot carry across instances (which would otherwise let a policy count episodes and
  correlate conditions across the suite).
- **§3 data into the player** — the referee passes `seed=0` and `config={}` into `player.reset`
  deliberately: nothing identifying the instance crosses the boundary. The observation carries no
  friction, no wind, no segment identity, and no obstacle oracle.
- **§4 internet** — `allow_internet: false`. `network_disabled: false` only so the referee can
  reach the player on the per-job network.
- **§5 persistence** — nothing written outside `/data`; no caches, no warm-up state. The referee
  writes `/data/result.json` and, per instance, `/data/history/instance_NN.json` (the platform's
  `FileType.HISTORY` channel, same one tron uses for `trace.jsonl`). History is best-effort: a
  write failure is logged and the round still scores. ~2 MB per round typically, ~8.5 MB if a
  policy survives every instance to the step cap; `record_history: false` disables it.
- **§6/§7 screening** — `artifact_type: onnx`, Layer-1 structural validation only. No Layer-2
  image, which is the outcome the skill steers toward: an ONNX graph cannot carry arbitrary code,
  and interface violations are a typed rejection in the player's loader.
- **§8 Goodhart** — gates zero out `physics_glitch` (NaN/Inf state, |qvel| > 100) and
  `out_of_bounds` (|y| > 1.2, so no walking around the course). `invalid_action` covers NaN and
  malformed actions; both paths are tested. Residual hole worth knowing: `progress` uses
  `max_x`, so a ballistic dive banks distance before falling. Bounded (a fall terminates, so
  it buys one dive) and it is what the alignment checks are for.
- **§9 determinism** — exact dependency pins and single-threaded ONNX Runtime. Still deterministic
  *given a seed*: `(seed, i)` fixes friction, wind and reset noise, so a round is exactly
  reproducible from its round input. What is gone is determinism *across* rounds, which is the
  point of the change. Verified bit-identical within an image; see the architecture table for
  cross-arch behaviour.

## Known follow-ups

1. **Re-measure `baseline_raw_score`** as a mean over ≥20 seeds, and record the spread. Blocks
   release; the spec field currently carries a stale 0.3.3 number and says so.
2. **Measure σ_round and set the takeover threshold against it** (see deviation 1). The one open
   design question this change introduces.
3. **Confirm the platform sends a fresh `seed` every round** (see deviation 1).
4. **Re-pin both image digests** from the 0.4.0 release run.
5. **Stage validation.**
6. **Fleet CPU homogeneity** — the open platform question. Same-generation reproducibility is
   demonstrated; cross-generation is not, and it cannot be tested from this repo.

## Evaluation wall clock — the 0.5.0 sizing (measured in the PR environment)

The referee's 900 s timeout is the binding constraint, and it binds on GOOD policies: the stock
baseline falls at ~10 m so it only runs ~28k control steps, while a policy that survives every
step runs `24 x max_steps`. Measured per control step, in-sandbox, at `cpu_limit: 2`:

| submission | size | ms/step | node |
|---|---|---|---|
| stock baseline | 0.13 MiB | 7.12 | idle |
| max-size probe | 22.48 MiB | 12.01 | idle |
| mid-size probe | 14.42 MiB | 12.43 | contended (2x text-clustering) |

Only ~1.2 ms of that is accounted work (0.45 ms physics + 10 `mj_step` + rays, 0.04 ms JSON,
0.51–2.62 ms inference). The rest is the synchronous `/act` round trip: the vendored
`gym_v1.PlayerClient` opens a fresh TCP connection per call, so a full-survival run makes 72,000
connections. Keep-alive in the SDK client is the largest single lever left and would benefit every
gym_v1 competition.

`cpu_limit: 2` came from this: at 1 CPU the per-step cost was 14.99 ms, at 2 it is 7.71 ms, while
median policy inference barely moved (0.525 → 0.510 ms). Neither sandbox is CPU-bound — 2 throttled
CFS periods out of 1690 — so this is a concurrency ceiling, not starvation.

**Caveat on precision.** These are single evaluations in a shared cluster. Node contention moves
per-step cost by ~2–5 ms, which is more than the difference between a 15 and a 25 MB cap — the
14.42 MiB run above measured *slower* than the 22.48 MiB one because it landed on a busy node. The
caps are therefore sized against the **worst observed** cost, and `max_steps` is the more reliable
lever because it bounds the worst case regardless of submission. Re-measure on a quiet node, or
across several runs, before tightening further.
