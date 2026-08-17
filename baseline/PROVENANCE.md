# Baseline status

`baseline.onnx` is the inherited Unitree G1 flat-ground walker, wrapped to the unchanged
104-observation / 256-state / 12-action ONNX interface. It remains a useful integration policy:
the course-relative heading and cross-track fields occupy the same positions that its wrapper
already consumes, so it can make a real attempt at straight and curved running.

It is not an Olympics-trained policy. It has no terrain training and no jumping controller, so it
should not be interpreted as an event baseline or as the source of a production leaderboard bar.

Before release:

1. Run it through the complete 24-attempt meet inside the native-amd64 referee image over at
   least 20 round seeds.
2. Record event-level means, round standard deviation, resource peak, and worst-case wall time.
3. Replace this note with the measured figure and keep `spec.defaults.baseline_raw_score` at zero
   unless the platform review establishes a stable non-zero entry bar.

The source architecture and exporter remain in `tools/make_baseline.py`. Its route inputs now
describe the local event tangent, so the same policy interface is reusable while a stronger
Olympics-specific reference is trained.
