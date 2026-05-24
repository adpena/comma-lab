# Codex Findings: PR95 MLX RGB+YUV6 Loss Surface

UTC: 2026-05-24T20:48:16Z
Lane: codex_pr95_mlx_rgb_yuv6_loss_surface_20260524

## Finding

The PR95 MLX source-video path was still RGB-MSE timing only. That made it
useful for throughput and queue economics, but weaker as substrate-training
signal because PR95's contest-relevant train path couples source-video RGB to
the scorer-side YUV6 preprocessing space.

## Landing

- Added `rgb_yuv6_mse` as an explicit PR95 MLX source-video loss surface.
- The new path trains on RGB MSE plus differentiable MLX RGB->YUV6 MSE and
  records `training_loss_surface`, `loss_surface_weights`,
  `target_yuv6_shape`, and `yuv6_preprocess_kind` in runtime/profile/archive
  manifests.
- Wired `--source-video-loss-surface` through the direct timing smoke CLI and
  the PR95 optimizer matrix queue builder.
- Queue plans now carry the loss-surface choice in candidate ids, matrix cell
  identity, execution commands, postconditions, representation manifests, and
  harvested candidate rows.

## Authority Boundary

This improves local MLX training signal only. It is still
`[macOS-MLX research-signal]` and still refuses score, promotion, rank/kill,
dispatch, and exact-eval authority. The blockers now state the sharper truth:
RGB+YUV6 preprocessing loss is closer to the scorer path, but it is not full
SegNet/PoseNet scorer loss and still requires export parity, byte-closed
runtime replay, full-frame parity, and exact CPU/CUDA auth eval.

## Verification

- Direct local MLX executable smoke:
  `experiments/results/codex_pr95_mlx_source_video_rgb_yuv6_loss_20260524T204916Z/`.
  The runtime profile records `training_loss_surface=rgb_yuv6_mse`,
  `loss_surface_weights={"rgb_mse": 0.5, "yuv6_mse": 0.5}`,
  `target_yuv6_shape=[1, 2, 192, 256, 6]`, and
  `seconds_per_step=0.027323624992277473`.
- Queue-owned executable smoke:
  `experiments/results/codex_pr95_mlx_source_video_rgb_yuv6_queue_20260524T204928Z/`.
  `experiment_queue.v1` validated the queue, initialized canonical state, ran
  the local MLX step, satisfied postconditions, and ended with `succeeded=1`
  and `orphaned_step_count=0`.
- `.venv/bin/python -m ruff check src/tac/local_acceleration/pr95_hnerv_mlx.py tools/run_pr95_mlx_timing_smoke.py tools/build_pr95_mlx_optimizer_matrix_queue.py src/tac/tests/test_pr95_hnerv_mlx.py src/tac/tests/test_run_pr95_mlx_timing_smoke.py src/tac/tests/test_run_pr95_mlx_timing_smoke_plan.py src/tac/tests/test_pr95_mlx_optimizer_matrix_queue.py`
- `.venv/bin/python -m pytest src/tac/tests/test_pr95_hnerv_mlx.py src/tac/tests/test_pr95_hnerv_mlx_training.py src/tac/tests/test_run_pr95_mlx_timing_smoke.py src/tac/tests/test_run_pr95_mlx_timing_smoke_plan.py src/tac/tests/test_pr95_mlx_optimizer_matrix_queue.py src/tac/tests/test_representation_training_probe_integration.py -q`
  passed with `44 passed`.
- `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/review_tracker.py policy-check src/tac/local_acceleration/pr95_hnerv_mlx.py tools/run_pr95_mlx_timing_smoke.py tools/build_pr95_mlx_optimizer_matrix_queue.py src/tac/tests/test_pr95_hnerv_mlx.py src/tac/tests/test_run_pr95_mlx_timing_smoke.py src/tac/tests/test_pr95_mlx_optimizer_matrix_queue.py`
  reported `0 violations`.
- `.venv/bin/python tools/lane_maturity.py validate`
  reported `1279 lane(s) validated cleanly`.

## Next Gap

The next PR95/HNeRV MLX gap is full scorer-loss coupling and export closure:
SegNet/PoseNet loss or a calibrated differentiable proxy, PyTorch/MLX forward
parity after training, byte-closed archive export, and same-runtime replay
before exact-eval queueing.
