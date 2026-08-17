# Competition onboarding manifest: `humanoid_olympics`

This document is completed with the candidate image digests and native-amd64 calibration artifact
after the signed candidate workflow finishes. It is not an onboarding request by itself.

## 1. Goal statement & alignment plan

**What success looks like**

> A winning submission is one compact recurrent, legs-only Unitree G1 controller that can sprint,
> stay on a narrow lane through a sustained corner, clear progressively harder hurdles, and execute
> legal vertical and horizontal jumps under seeded surface and wind conditions. It must be an
> adaptable all-round athletic controller rather than six memorised or invalid trajectories.

**Alignment checks and review plan**

- Replays show the same controller completing or materially progressing across every event, with no
  event score carried by invalid contacts, boundary violations, or typed player faults.
- The 400 m diagnostic shows forward lap progress while inside the lane; jump diagnostics show
  legal board take-off, required flight phases, and foot-top landing contacts.
- Top policies fit the CPU/1.5 GiB budget and answer within the fixed 500 ms action deadline.
- After every round reveal, inspect top-event histories plus event-score distributions on fresh
  seeds. Watch for score spikes concentrated in one event, contact-foul patterns, long evaluator
  times, or policies that rank well without exhibiting the stated athletic behaviours.

## 2. Deliverables

| Item | Where | Status |
|---|---|---|
| Competition repo + final release tag | `macrocosm-os/apex-competition-humanoid-olympics` @ pending signed candidate/final tag | pending candidate release |
| `spec.yaml` (`apex.competition.v1`) + preflight | `spec.yaml`, `fixtures/input.json` | local stage/prod preflight passes; repeat after digest pin |
| Player image | `ghcr.io/macrocosm-os/apex-competition-humanoid-olympics-player@sha256:<candidate>` | pending signed candidate build |
| Referee image | `ghcr.io/macrocosm-os/apex-competition-humanoid-olympics-referee@sha256:<candidate>` | pending signed candidate build |
| Layer-2 screen | n/a; fixed-shape ONNX artifact plus platform structural screen and player validation | n/a |
| Round generation | n/a; one platform master seed is sufficient | complete |
| Cosign identity + issuer | `.github/workflows/release.yml`; GitHub Actions OIDC | candidate workflow verifies it |
| Input schema + fixture | `input.schema.json`, `fixtures/input.json` | complete |
| Baseline integration artifact | `baseline/baseline.onnx` | pinned; full native-amd64 evidence pending |
| Miner documentation | `README.md`, `docs/design.md` | complete |
| Full end-to-end evidence | release CI two-container job + candidate calibration artifact | candidate workflow pending |

Score-affecting pins already committed:

- player base/runtime: `python:3.12-slim`, `numpy==2.3.4`, `onnxruntime==1.28.0`;
- referee base/runtime: `python:3.12-slim`, `numpy==2.3.4`, `mujoco==3.11.0`;
- vendored protocol: `macrocosm-os/apex-competitions-builder@d063d9028dbec4bb15182794496f4aa2aac19d49`;
- Unitree stock-policy source: `unitree_rl_gym@276801e46c5d433564f24658bac64f254b7d2d4b`;
- `motion.pt`: `cf668f75b90d1abf73d2b87612a6e76bccc61ff7e083b63582d3f6aaa3c1759d`;
- baseline ONNX: `1d0a88ef2edcd13f9ad0401cc72faaea664951ec0763429ad31bbc907e2954f2`;
- datasets / held-out corpora: n/a; all course geometry is code-versioned in the referee image.

## 3. Ops parameters

| Parameter | Proposal | Reason |
|---|---|---|
| Process / kind | CPU / solo | A single 15 MB ONNX policy and deterministic MuJoCo physics fit CPU; score is absolute. |
| Round length / reveal | 2 days / 5 days | The fixed v0.1 meet can loop while trained-control breakthroughs retain meaningful IP. |
| Score direction | higher is better | Legal speed, height, and distance are monotone athletic improvements. |
| Launch meet | 4 attempts × 6 events; 40,000 maximum actions | Fixed shape makes cross-round scores comparable while sampling four condition strata. |
| Resources | 2 CPU, 1.5 GiB, 0 GPU | The compiled G1 scene is referee-owned; this is the stage ceiling and is exercised in release CI. |
| Timeouts | player 1,200 s; referee 900 s; internal scheduler 840 s | Leaves persistence time after a full recorded 24-attempt meet. |
| Action deadline | 500 ms | Caps inference latency while remaining realistic for compact recurrent CPU ONNX. |
| Baseline score | pending native-amd64 20-seed mean and stability gate | Measured before the final tag; never guessed. |
| Submission fee | propose USD 1 in TAO | Discourages low-effort repetition and approximately covers simulator cost. |
| Incentive weight | propose 0.03 | Middle of the normal 0.02–0.05 range, subject to Macrocosmos. |

## 4. Evaluation sizing

The fixed launch unit is 24 attempts: four deterministic condition strata for each of six
equal-weight events, repeated identically for every round seed. The candidate workflow runs the baseline, a stationary valid ONNX reference,
and an independently seeded untrained ONNX reference across 20 master seeds in the actual
two-container native-amd64 setup. It records per-seed raw/event scores, standard deviation,
ordering, peak container memory, and wall time in `baseline-calibration-native-amd64`.

Before onboarding, fill this section from that artifact:

- Baseline mean raw score: pending.
- Baseline sample standard deviation across 20 seeds: pending.
- 1% takeover margin / quarter-margin check: pending (`σ_round <= 0.0025 × typical score`).
- Baseline ordering above both deliberately weaker references on every seed: pending.
- Mean / worst full recorded wall time and peak memory: pending.

If the variance check fails, the fixed launch configuration is not submitted; its condition mix or
attempt count is revised in a new spec version instead.

## 5. Threat-model questionnaire

1. **Miner-visible surface.** The player gets its own public 104-float observations, reset
   identifier/index, reset seed `0`, and the fixed 500 ms deadline. Observations describe current
   robot/course state only. It receives no platform seed, future conditions, referee filesystem,
   or other submission.
2. **Seed leverage.** The platform round seed remains referee-side but is intentionally
   score-neutral for v0.1. Every round repeats the same public, balanced condition suite, so no
   seed can select an easier meet or move the absolute baseline.
3. **Degenerate submissions.** Static and seeded-untrained ONNX references are measured in the
   candidate suite. Malformed/NaN/wrong-shape responses receive typed invalid outcomes; boundary,
   invalid-contact, and player faults contribute zero.
4. **Baseline resubmission.** The published baseline is a reproducible integration reference. An
   identical artifact reproduces its fixed-round result and cannot exceed itself by the required
   1% takeover margin.
5. **Metric gaming.** Review covered malformed/non-finite actions, timeout and reset faults,
   lateral bypass, hurdle contacts, ducking, bar contact, board skipping, wrong-foot pads,
   side/underside contacts, body-first landings, and floor routing. The referee owns typed action
   validation, 500 Hz route/contact gates, physical obstacles, top-face normal/impulse support
   classification, phase state machines, and bounded event scores.
6. **Copy-plus-epsilon.** The five-day reveal delay is longer than the two-day round. Within a
   round, fixed seed/config means a trivial copy reproduces its score rather than gaining 1%.
7. **Cross-round leakage.** Per-attempt histories are released only after the round and show
   already-observed public physics. The v0.1 meet repeats by design; there is no hidden answer
   corpus, unrevealed condition stream, or other submission to leak.
8. **Error-message hygiene.** Miner-facing outcomes are stable categories such as invalid action,
   player error, hurdle hit, jump foul, or timeout. They contain no stack trace, host path,
   internal seed, secret threshold, or future task information.
9. **Referee state.** For a seed/config/submission the scorer is deterministic. It compiles only
   the current event model, resets simulator/player state per attempt, and writes data only to the
   per-job artifact mount; no submission-keyed persistent cache is used.
10. **Code execution.** The submission type is ONNX, not code/TorchScript. Declarative size and
    weight checks plus player-side exact I/O/type validation constrain it to a CPU graph; no
    custom submission code executes in the referee.
11. **Player-image hygiene.** The player contains only the public HTTP wrapper, vendored protocol,
    pinned ONNX Runtime, and mounted miner artifact. Course geometry, simulation, seed handling,
    scoring, and histories remain in the referee image.
12. **Diagnostics payload.** `result.json` and replay history record event, attempt, public
    condition values, terminal reason, score, metrics, and robot trajectory after a round. They
    do not include another submission, an unrevealed future seed, secret corpus, or hidden judge.

## 6. GPU justification

Not applicable. Both policy inference and referee physics are CPU-only; the candidate workflow
verifies them under the declared 2 CPU / 1.5 GiB limit.

## 7. Candidate-to-onboarding procedure

1. Push this source and tag `calibration-<candidate>` to run the native-amd64 calibration suite.
2. If the sizing and resource checks pass, tag `v0.1.0-candidate.1`; release CI publishes and
   keyless-signs both images, then verifies the signature against the identity/issuer in `spec.yaml`.
3. Commit the resulting candidate digests into `spec.yaml` and this handoff, repeat preflight,
   then tag `v0.1.0`. The final source build is identical in both Docker contexts, and the workflow
   signs/verifies the same pinned image digests again.
4. Attach this completed manifest, release tag, image refs/digests, and calibration artifact to a
   Macrocosmos onboarding issue. Do not request production activation before the stage round passes.
