# Codex Findings: Z8 true-P19 codec runner integration

UTC: 2026-05-31T19:58:28Z

## What changed

- Made the Z8 coefficient materializer and relinearized bounded search consume
  the true-P19 pose surface contract directly, not just generic
  `budget_spend_authority`.
- Added a strict `surface_true_p19_report` requiring:
  - `pose_surface_kind = per_axis_posenet_jacobian_mahalanobis_v1`
  - `pose_surface_authority = true`
  - six PoseNet axes
  - six positive finite inverse-variance Mahalanobis weights
  - no pose-surface blockers
- Propagated `surface_true_p19_report` into bounded-search candidate rows so
  acquisition/posterior consumers can see exactly whether each row used true
  P19.
- Emitted full `inflate.sh` runtime benchmark work orders by default for
  byte-closed Z8 materializations and bounded-search outputs.
- Exposed operator CLI controls for benchmark work-order emission/execution,
  timeout, auth-window denominator, and inflate device.
- Exposed MLX provider controls for `--mlx-pose-axis-count` and
  `--mlx-pose-inverse-variance`, matching the standalone full-video VJP bundle
  builder so bounded-runner-generated surfaces do not silently inherit hidden
  P19 covariance defaults.

## Authority posture

The integration is still fail-closed:

- Legacy scalar/first-six pose-loss proxy surfaces are rejected by default
  before coefficient mutation.
- Storage-only transcode probes remain allowed without a true-P19 surface
  because they do not spend a gradient surface or alter coefficient signal.
- Full `inflate.sh` runtime benchmarks remain `[macOS-CPU advisory]` and cannot
  claim score, rank, promotion, or exact dispatch readiness.
- Receiver proof and exact CPU/CUDA auth eval remain required before any score
  authority or promotion claim.

## Verification

- `ruff check` on touched Z8 materializer/provider/tools/tests: passed.
- Focused tests:
  `pytest test_joint_coefficient_waterfill.py test_full_video_vjp_acquisition.py test_inflate_runtime_benchmark.py -q`
  -> 37 passed.
- Wider Z8 codec/bridge regression:
  `pytest test_joint_coefficient_waterfill.py test_full_video_vjp_acquisition.py test_inflate_runtime_benchmark.py test_entropy_delta_schedule.py test_detail_coeff_entropy_headroom_report.py test_archive_candidate_bridge.py -q`
  -> 65 passed.

## Next required work

Regenerate a fresh full-video true-P19 bundle on the live 600-pair Z8 archive,
run strict bounded codec search with the new default true-P19 gate, materialize
the selected byte-closed archive, then execute the full `inflate.sh` benchmark
before receiver proof/exact-axis promotion.
