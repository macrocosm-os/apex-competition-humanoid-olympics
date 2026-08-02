"""Evaluation-sizing evidence for HANDOFF.md §4.

Evaluates one policy across >= 20 master seeds and reports sigma_round against the platform's
1% takeover margin (requirement: sigma_round <= margin / 4, see reference/evaluation-design.md).
Run with the trained baseline AND at least one deliberately different reference policy, and check
they rank consistently across every seed.

    python tools/measure_variance.py --onnx baseline/baseline.onnx --seeds 20 \
        --out evidence/variance_baseline_N120.json

Defaults match the round input the leaderboard uses (N = 3 x 40 courses, 900-step cap). Anything
else produces a number that is NOT the takeover floor — instance_score normalizes the time bonus
by max_steps, and a longer cap converts timeouts into completions.

`--out` writes the full per-seed array *plus the config and the artifact's sha256*, because the
declared `defaults.baseline_raw_score` is only auditable if the evidence says which artifact and
which round input produced it.

NOTE ON WHERE YOU RUN THIS: MuJoCo and onnxruntime dispatch on CPU features, so a host and the
referee image do NOT agree to the last bit (measured: ~1-3% apart at N=120, wider than the 1%
takeover margin). The platform runs the referee IMAGE, so numbers destined for spec.yaml must be
measured there --- see tools/measure_variance_in_image.sh, which drives the built images.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

from local_eval import evaluate_once

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--courses-per-difficulty", type=int, default=40)
    parser.add_argument("--max-steps", type=int, default=900)
    parser.add_argument("--deadline-ms", type=int, default=500)
    parser.add_argument("--out", help="write the evidence JSON here (HANDOFF.md §4 cites it)")
    args = parser.parse_args()

    scores = []
    for seed in range(args.seeds):
        result = evaluate_once(
            args.onnx, seed, args.courses_per_difficulty, args.max_steps, args.deadline_ms
        )
        scores.append(result.raw_scores[0])
        print(
            f"seed {seed:>3}: raw_score {scores[-1]:.4f} "
            f"({result.metadata['num_completed']}/{result.metadata['num_courses']} completed, "
            f"{result.metadata['eval_time_in_seconds']}s)",
            flush=True,
        )

    mean = statistics.mean(scores)
    sigma = statistics.stdev(scores)
    margin = 0.01 * mean  # takeover threshold is 1% of the top raw score
    evidence = {
        "onnx": args.onnx,
        "onnx_sha256": hashlib.sha256(Path(args.onnx).read_bytes()).hexdigest(),
        "measured_in": "host (not the referee image --- see the module docstring)",
        "N": 3 * args.courses_per_difficulty,
        "courses_per_difficulty": args.courses_per_difficulty,
        "max_steps_per_episode": args.max_steps,
        "deadline_ms": args.deadline_ms,
        "seeds": args.seeds,
        "scores": scores,
        "mean": mean,
        "sigma_round": sigma,
        "margin_1pct": margin,
        "quarter_margin": margin / 4,
        "sigma_over_quarter_margin": sigma / (margin / 4),
        "pass": sigma <= margin / 4,
    }
    print(f"\nmean raw_score : {mean:.4f}")
    print(f"sigma_round    : {sigma:.4f}")
    print(f"1% margin      : {margin:.4f} (margin/4 = {margin / 4:.4f})")
    print(f"sigma/(margin/4): {evidence['sigma_over_quarter_margin']:.1f}x")
    print(f"sigma_round <= margin/4: {'PASS' if evidence['pass'] else 'FAIL — raise N or reduce variance'}")
    if args.out:
        Path(args.out).write_text(json.dumps(evidence, indent=2) + "\n")
        print(f"wrote {args.out}")
