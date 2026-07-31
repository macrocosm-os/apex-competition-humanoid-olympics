#!/usr/bin/env bash
# Evaluation-sizing evidence measured INSIDE the built images — the numbers that belong in
# spec.yaml.
#
# Why this exists as well as tools/measure_variance.py: MuJoCo and onnxruntime dispatch on CPU
# features, so a host and the referee image do not agree to the last bit. Measured on an arm64
# host vs the linux image: ~1-3% apart at N=120, which is WIDER than the 1% takeover margin. The
# platform runs the referee image, so `defaults.baseline_raw_score` — the takeover floor — has to
# come from the image, not from a laptop.
#
# Runs the real two-container loop the platform runs: player and referee containers on an
# --internal (no-egress) network, at the spec's resource limits, referee writing /data/result.json.
#
#   ./tools/measure_variance_in_image.sh baseline/baseline.onnx 20 > variance_baseline_N120_image.json
#
# Build the images first:
#   docker build -f player/Dockerfile -t hp2-player . && docker build -f referee/Dockerfile -t hp2-referee .
set -euo pipefail

ONNX="${1:?usage: $0 <policy.onnx> [seeds] }"
SEEDS="${2:-20}"
PER_DIFFICULTY="${3:-40}"        # N = 3 x this; must match the round input
MAX_STEPS="${4:-900}"
DEADLINE_MS="${5:-500}"

NET=hpmeasure-$$
PLAYER=hpmeasure-player-$$
OUTDIR=$(mktemp -d)
cleanup() { docker rm -f "$PLAYER" >/dev/null 2>&1 || true; docker network rm "$NET" >/dev/null 2>&1 || true; }
trap cleanup EXIT

docker network create --internal "$NET" >/dev/null
docker run -d --name "$PLAYER" --network "$NET" --cpus 1 --memory 1.5g \
  -v "$(cd "$(dirname "$ONNX")" && pwd)/$(basename "$ONNX"):/app/submission.onnx:ro" \
  hp2-player >/dev/null

# Wait for readiness from inside the network (the network has no egress by design).
for _ in $(seq 60); do
  docker run --rm --network "$NET" hp2-player python -c "
import urllib.request,sys
try: urllib.request.urlopen('http://$PLAYER:8000/health',timeout=2)
except Exception: sys.exit(1)" >/dev/null 2>&1 && break
  sleep 1
done

for s in $(seq 0 $((SEEDS - 1))); do
  mkdir -p "$OUTDIR/$s"
  docker run --rm --network "$NET" --cpus 1 --memory 1.5g -v "$OUTDIR/$s:/data" \
    -e MATCH_ID="measure-$s" -e SEED="$s" -e NUM_PLAYERS=1 -e PLAYER_URLS="http://$PLAYER:8000" \
    -e CONFIG_JSON="{\"courses_per_difficulty\":$PER_DIFFICULTY,\"max_steps_per_episode\":$MAX_STEPS,\"deadline_ms\":$DEADLINE_MS}" \
    hp2-referee >/dev/null 2>&1
  echo "seed $s done" >&2
done

python3 - "$OUTDIR" "$ONNX" "$SEEDS" "$PER_DIFFICULTY" "$MAX_STEPS" "$DEADLINE_MS" <<'PY'
import hashlib, json, pathlib, statistics, sys
d, onnx, seeds, per_diff, max_steps, deadline = pathlib.Path(sys.argv[1]), sys.argv[2], *map(int, sys.argv[3:6]), int(sys.argv[6])
scores = [json.load(open(d / str(s) / "result.json"))["raw_scores"][0] for s in range(seeds)]
mean, sigma = statistics.mean(scores), statistics.stdev(scores)
margin = 0.01 * mean
print(json.dumps({
    "onnx": onnx,
    "onnx_sha256": hashlib.sha256(pathlib.Path(onnx).read_bytes()).hexdigest(),
    "measured_in": "referee image (two-container run, --internal network, 1 CPU / 1.5Gi)",
    "N": 3 * per_diff, "courses_per_difficulty": per_diff,
    "max_steps_per_episode": max_steps, "deadline_ms": deadline, "seeds": seeds,
    "scores": scores, "mean": mean, "sigma_round": sigma,
    "margin_1pct": margin, "quarter_margin": margin / 4,
    "sigma_over_quarter_margin": sigma / (margin / 4),
    "pass": sigma <= margin / 4,
}, indent=2))
PY
