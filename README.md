# hello_world — the worked example Apex competition

The minimal end-to-end [Apex](https://macrocosmos.ai) competition: a complete, buildable
`apex.competition.v1` competition in as few moving parts as possible. **Fork this repo as the
starting point for your own competition.**

The task is deliberately trivial — sort a list of numbers — so that nothing distracts from the
*structure*: a spec, a player image, a referee image, and a release workflow that signs both.

- **Submission format:** `code` — a `submission.py` exposing `sort_numbers(numbers)`.
- **Score:** `raw_score` = fraction of tasks sorted correctly, **higher is better**.
- **Baseline:** `player/submission.py` (a one-line `sorted()`), which scores 1.0.

> This is a teaching example, not a live competition. The image digests in `spec.yaml` are
> placeholder zeros and it is not registered on the platform.

## What's in here

| Path | What it is |
|------|-----------|
| `spec.yaml` | The competition: kind, resources, submission contract, screening, entrypoints, images, cosign identity. |
| `input.schema.json` | JSON Schema for the round input, `$ref`'d from the spec. |
| `fixtures/input.json` | A round-input fixture to validate against the schema. |
| `player/Dockerfile`, `player/launch.py` | The **player** image: serves the miner's submission over the gym_v1 HTTP API. |
| `player/submission.py` | The reference (baseline) submission. Not baked into the image — the platform writes the miner's version to `/app/submission.py` at run time. |
| `referee/Dockerfile`, `referee/referee.py` | The **referee** image: holds the ground truth, drives the player, writes `/data/result.json`. |
| `player/gym_v1/`, `referee/gym_v1/` | **Vendored** copy of the SDK's `gym_v1` package (see below). |
| `.github/workflows/release.yml` | Builds, pushes by digest, and keyless-signs both images on a `v*` tag. |

## The vendored SDK — this is the pattern to copy

Both images **vendor** the SDK's `gym_v1/` package into this repo and build on
`FROM python:3.12-slim`:

```dockerfile
FROM python:3.12-slim
COPY player/gym_v1/ /app/gym_v1/     # <- the vendored SDK
COPY player/launch.py /app/launch.py
```

```python
from gym_v1.player import Player, serve                    # not apex_sdk.gym_v1
from gym_v1.referee import Referee, GameResult, RefereeContext
from gym_v1.client import PlayerClient, PlayerError
```

**Do not build `FROM apex-player-base` / `apex-referee-base`.** Those base images ship the SDK
as `apex_sdk.gym_v1`, but they are not published to any registry — the build only resolves on a
machine that has `docker build`-ed the base locally, so it **fails in release CI**. Build-FROM-base
is the intended future once the bases are published; vendoring is what works today and what every
shipped competition does.

The vendored files carry a provenance header naming the SDK version they came from. Don't
hand-edit them — to update, re-copy from
[apex-competitions-sdk](https://github.com/macrocosm-os/apex-competitions-sdk) `src/apex_sdk/gym_v1/`
and rewrite the `apex_sdk.gym_v1` import root to `gym_v1`:

```bash
SDK=../apex-competitions-sdk
for side in player referee; do
  for f in __init__ client player referee; do
    sed 's/^from apex_sdk\.gym_v1\./from gym_v1./' "$SDK/src/apex_sdk/gym_v1/$f.py" > "$side/gym_v1/$f.py"
  done
done
```

## Validate and run locally

```bash
pip install apex-competition-sdk        # or: pip install -e ../apex-competitions-sdk

# 1. Validate the spec + input fixture against apex.competition.v1. No Docker.
apex-dev preflight --spec ./spec.yaml --input fixtures/input.json

# 2. Preview the resolved execution plan (player + referee images, protocol, resources).
apex-dev run --spec ./spec.yaml --input fixtures/input.json \
             --submission ./player/submission.py --dockerfile ./player/Dockerfile
```

`apex-dev run` prints the plan and exits 3: referee-driven local execution (both sandboxes on a
shared network) is not implemented in the SDK yet. Until it is, exercise the full loop by hand —
which is also the honest test of the sandboxed leg, since it runs the player with egress blocked
and the spec's resource limits:

```bash
# Build both images (build context = this repo root).
docker build -f player/Dockerfile  -t hello-world-player  .
docker build -f referee/Dockerfile -t hello-world-referee .

docker network create hello-net

# Player: submission mounted at target_path, no egress, spec resource limits.
docker run -d --name hello-player --network hello-net \
  --cpus 1 --memory 512m \
  -v "$PWD/player/submission.py:/app/submission.py:ro" \
  hello-world-player

# Referee: the platform injects these env vars and reads /data/result.json.
docker run --rm --network hello-net \
  -e MATCH_ID=local -e SEED=0 -e NUM_PLAYERS=1 \
  -e PLAYER_URLS='http://hello-player:8000' \
  -e CONFIG_JSON="$(cat fixtures/input.json)" \
  -v "$PWD/out:/data" \
  hello-world-referee

cat out/result.json     # -> {"raw_scores": [1.0], "winner": 0, "terminal_reason": "scored", ...}

docker rm -f hello-player && docker network rm hello-net
```

## Ship it

1. Tag a release (`git tag v0.1.0 && git push --tags`) — `release.yml` builds, pushes, and
   keyless-signs both images.
2. Copy the pushed digests from the Actions log into `spec.yaml` (`image.digest` and
   `referee.image.digest`).
3. Open a [Competition onboarding issue](https://github.com/macrocosm-os/apex-competitions-sdk/issues/new?template=competition-onboarding.yml)
   with your repo URL, released tag, image refs + digests, and a filled `HANDOFF.md`. A
   Macrocosmos maintainer copies your `spec.yaml` into the private registry and activates it on
   stage, then prod.

Full authoring guide, the spec schema, and the design skill:
[macrocosm-os/apex-competitions-sdk](https://github.com/macrocosm-os/apex-competitions-sdk).
