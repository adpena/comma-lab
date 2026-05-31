# Codex Findings: Z8 true P19 surface + full inflate runtime benchmark

UTC: 2026-05-31T19:48:54Z

## What changed

- Converted the Z8 full-video MLX VJP shard path from scalar pose-loss proxy to
  true P19: per-axis PoseNet output VJPs over the first six pose axes.
- Preserved contest-score geometry by using the shared joint P18/P19 waterfill
  helper with explicit inverse-variance Mahalanobis pose weighting.
- Made bundle assembly reject authority when shard pose metadata is stale,
  missing, non-six-axis, inconsistent, non-finite, or not exact-reduced over the
  full video.
- Added CLI exposure for `--pose-axis-count` and
  `--pose-inverse-variance`, so true P19 artifacts declare their weighting
  contract instead of silently inheriting defaults.
- Added a reusable full `inflate.sh` benchmark module plus operator CLI. This
  exercises `inflate.sh <archive_dir> <output_dir> <file_list>`, records timing,
  archive/output manifests, stdout/stderr tails, timeout/nonzero blockers, and
  stays `[macOS-CPU advisory]` with no score or promotion authority.

## Contract

The true P19 object is:

`pose_term_i = 5 / sqrt(10 * d_pose) * ||J_pose,i||_{Sigma^-1}`

where `J_pose,i` is the per-axis PoseNet Jacobian for the first six pose outputs
through the MLX scorer adapter path, and `Sigma^-1` is explicitly supplied as
positive inverse-variance weights. Identity weights match the upstream first-six
pose MSE contract until a better contest-normalization vector is supplied.

P18 and P19 remain encoder-side allocation signals only. MLX rows, local
runtime benchmarks, and parser timings are not score authority. Byte-closed
candidate promotion still requires receiver proof and exact CPU/CUDA authority.

## Verification

- `ruff check` on touched Z8 VJP, benchmark, tests, and tool files: passed.
- Focused tests:
  `pytest test_full_video_vjp_acquisition.py test_inflate_runtime_benchmark.py -q`
  -> 13 passed.
- Wider Z8 codec/bridge regression:
  `pytest test_joint_coefficient_waterfill.py test_full_video_vjp_acquisition.py test_inflate_runtime_benchmark.py test_entropy_delta_schedule.py test_detail_coeff_entropy_headroom_report.py test_archive_candidate_bridge.py -q`
  -> 64 passed.

## Remaining next actions

1. Regenerate all-pair Z8 full-video MLX VJP shards with explicit six-axis P19
   metadata and fresh archive SHA.
2. Materialize strict Z8 coefficient variants only from full-video exact-reduced
   finite bundles.
3. Run the full `inflate.sh` benchmark on the live candidate runtime before
   exact promotion so codec decode speed is measured at the receiver boundary.
4. Route positive and negative replay outcomes into posterior demotion and the
   bounded Z8 codec runner.
