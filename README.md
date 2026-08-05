<img width="1267" height="625" alt="01_overview_iso" src="https://github.com/user-attachments/assets/17b9f4ca-39ef-4d15-a216-b3eee1888a43" />

# Humanoid Parkour

An Apex competition (Bittensor Subnet 1). Miners submit an **ONNX policy** that drives a Unitree
G1 humanoid through a 51 m parkour course: a steep on-ramp, a sheer drop, stairs, a 1 m leap over
a real void, a hip-high hurdle, a 0.55 m step-up, a duck-under, a balance beam, a slick patch,
and a stairway down.

**Nobody has finished it.** The reference policy — Unitree's own stock G1 walker — gets 21% of
the way and falls off the first ledge.

| | |
|---|---|
| id / version | `humanoid_parkour` 0.3.0 |
| robot | Unitree G1, **12 actuated leg DoF only** — no arm joints, 32.1 kg |
| submission | ONNX graph, ≤ 25 MB, architecture free |
| interface | `obs[104]` + `state_in[256]` → `action[12]` + `state_out[256]`, float32 |
| evaluation | 24 fixed instances, ≤ 4000 control steps each (80 s sim) |
| baseline | 0.2007 — see [`baseline/PROVENANCE.md`](baseline/PROVENANCE.md) |

## The robot has no arms

All 12 actuators are legs (hip pitch/roll/yaw, knee, ankle pitch/roll, ×2). The arms are 17.7 kg
of collision geometry welded to the pelvis: they are present, they have mass, and they hit things
— but nothing can move them. **Every obstacle is a leg maneuver.** There is no vaulting, no
pulling up, and no arm swing for balance.

Both tall obstacles are sized against measured leg capability, not against a robot with hands:
the leg reaches **1.30 m** kinematically (hip pitch spans ±2.88 rad) against a 0.62 m hurdle, and
the 0.55 m step-up needs ~31-63 N.m at the knee against a **139 N.m** limit.

## The course

51.1 m, linear, on a raised plinth so gaps are real voids. Difficulty ramps along its length, so
progress-based scoring gives a continuous gradient rather than discrete tiers — a policy that
gets 3 m further scores 3 m better, all the way along.

| Maneuver | Geometry |
|---|---|
| on-ramp | 6 m flat, 15.4° climb over 2 m, then a 0.55 m sheer drop |
| stairs up / down | rise 0.18–0.20 m, run 0.32–0.34 m |
| leap | 1.0 m void |
| drop-down | 0.6 m |
| hurdle | 0.62 m barrier — stepped over, not vaulted |
| step-up | 0.55 m platform (70% of hip height) |
| duck-under | overhead bar at 1.05 m — forces a ~0.2 m squat-walk |
| balance beam | 0.32 m wide, 3.5 m long |
| slick patch | low friction, geometry identical to flat |

The geometry is **static and public**. What varies between instances is surface friction, and it
is deliberately **not observable** — a policy has to feel the slip and adapt rather than read a
number. That is why the interface carries recurrent state.

```bash
python -m env.course          # print the layout
python tools/preview.py       # stills + flythrough (needs mujoco + ffmpeg)
```

## Scoring

Per instance, higher is better:

| Outcome | Score |
|---|---|
| completed | `1.0 + (max_steps - steps) / max_steps` → (1.0, 2.0] |
| fell / timeout / out_of_bounds | `progress`, the fraction of the course covered → [0.0, 1.0) |
| physics_glitch / invalid / player error | 0.0 |

`raw_score` is the mean over the 24 instances. Any completion outranks any non-completion, faster
completions outrank slower ones, and partial progress gives non-finishers a training gradient.

## Why the evaluation suite is fixed

The 24 instances are a pure function of `(index, count)` — **not** of the platform's per-round
seed (`env/sim.instance_spec`). This is the load-bearing design decision, so it is worth stating
plainly.

The course is static and public, so a per-round seed would buy no secrecy. All it would buy is
score noise, and score noise is what sets the takeover margin. Measured per-instance stdev is
**0.0176**; against a 1% takeover margin of 0.002, resolving a genuine 1% improvement through a
randomised suite would need **~1400 instances**, which does not fit the referee's 900 s budget.

A fixed suite makes a given policy score identically every round. Verified: four different `SEED`
values all produce the same score, bit for bit. Round-to-round variance is **zero**, so takeover
is decided by skill.

Coverage comes from stratification instead of randomness — friction levels are spread evenly
across the range, so 24 instances sample the whole grippy-to-slippery continuum rather than
clustering wherever a draw landed.

## Perception

The policy sees proprioception (projected gravity, base velocities, joint angles and velocities,
its own last action, a gait clock), its pose on the track (heading, lateral offset, distance to
the finish), and terrain:

- **height scan** — a 9×5 grid of downward samples in the robot's yaw frame, from 0.4 m behind to
  1.6 m ahead, given relative to the pelvis
- **overhead clearance** — 7 upward samples ahead, which is how the duck-under is visible

It does not get an obstacle oracle. There is no "a leap starts in 1.2 m" channel, no segment
identity, and no friction. Full layout in [`env/sim.py`](env/sim.py).

## Watching an evaluation

A score says how a run ended, not how it got there. `--record` writes a trajectory log — the
robot's `qpos` at every control step, plus the friction values that built the scene — and
`tools/replay.py` plays it back:

```bash
PYTHONPATH=. python tools/local_eval.py baseline/baseline.onnx -n 24 --record runs/base.npz

PYTHONPATH=. python tools/replay.py runs/base.npz            # list the instances
PYTHONPATH=. python tools/replay.py runs/base.npz --worst     # film the run that scored lowest
PYTHONPATH=. mjpython tools/replay.py runs/base.npz -i 7 --live   # interactive viewer
```

Replay is **MuJoCo only** — no policy, no onnxruntime, no physics. It sets `data.qpos` and calls
`mj_forward`, which recomputes everything a renderer needs, so a run stays viewable after the
submission is gone and cannot drift from what was scored the way re-simulating from an action log
could. Recording costs one array copy per step and does not touch scoring: a recorded suite
produces byte-identical numbers to an unrecorded one.

Sizing: ~289 KiB for an instance that runs the full 4000 steps, ~70 KiB for a typical baseline run
that falls around 20 s. `--record-stride N` keeps every Nth step if that ever matters.

`--live` needs `mjpython` rather than `python` on macOS, because the passive viewer has to own the
main thread there. Filming to mp4 is headless and needs only ffmpeg. Only `tools/local_eval.py`
records; the referee's scoring path is untouched.

## Submitting

The tensor signature is fixed; the graph is not — recurrent nets, ensembles, transformers over a
history window. The 25 MB cap is ~180x the size of the reference policies it is measured against
(every Unitree humanoid walker is 0.13-0.14 MB) and is not what will stop you. `state_in`/`state_out` are your own opaque per-episode memory — zeroed on
reset, fed back each step. A feed-forward policy ignores `state_in` and returns zeros.

Off-the-shelf G1 locomotion policies are a reasonable starting point and that is exactly what the
baseline is, but none of them can see terrain, so none of them will get past the on-ramp without
retraining.

## Resource use

Measured under the spec's own limits (`--cpus 1 --memory 1.5g`): referee peaks at **560 MiB of
1536 (36%)**, player at 30 MiB. Referee memory does not grow with submission quality, so that is
near the true ceiling rather than a floor better solutions will grow into.

## Repo layout

```
env/            course, physics, perception, gates, scoring  (referee image)
  assets/       vendored Unitree G1 12-DoF model + collision meshes (BSD-3)
player/         ONNX serving + interface validation           (player image)
referee/        match driver, fault attribution
baseline/       the reference policy and where its number came from
tools/          baseline export, local eval, course preview, trajectory record + replay
docs/           design notes
spec.yaml       the competition manifest
HANDOFF.md      platform-review notes: deviations, measurements, security walk
```

## Running it end to end

```bash
docker build -f referee/Dockerfile -t hp-referee .
docker build -f player/Dockerfile  -t hp-player  .

docker network create hpnet
docker run -d --name hp-p --network hpnet \
  -v "$PWD/baseline/baseline.onnx:/app/submission.onnx:ro" hp-player

mkdir -p /tmp/hpdata && chmod 777 /tmp/hpdata
docker run --rm --network hpnet -v /tmp/hpdata:/data \
  -e MATCH_ID=local -e SEED=1 -e NUM_PLAYERS=1 -e PLAYER_URLS=http://hp-p:8000 \
  -e CONFIG_JSON='{"num_instances":24,"max_steps_per_episode":4000,"deadline_ms":500}' \
  hp-referee
jq '.raw_scores, .metadata.num_completed' /tmp/hpdata/result.json
```

Takes ~67 s with the baseline on a worker-class amd64 CPU, and ~258 s worst case for a policy that
survives every step, against the referee's 900 s timeout. (An arm64 build of the same images runs
it in ~30 s, and scores 0.12% differently — which is why `baseline_raw_score` is measured on
amd64; see [`baseline/PROVENANCE.md`](baseline/PROVENANCE.md).)

## History

An earlier, much simpler version (flat plane, step-over hurdles, Gymnasium humanoid) was built,
released and signed as **`v0.2.0`**. It is preserved at the
[`v0.2.0`](https://github.com/macrocosm-os/apex-competition-humanoid-parkour-v2/tree/v0.2.0) tag
with its images in GHCR, and is not part of this codebase. The predecessor repo
`apex-competition-humanoid-parkour` is archived.
