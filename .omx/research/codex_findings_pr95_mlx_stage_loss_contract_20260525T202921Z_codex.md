# Codex Findings - PR95 MLX Stage-Loss Contract

UTC: 2026-05-25T20:29:21Z
Agent: Codex
Lane: `lane_pr95_hnerv_mlx_reproduction`
Evidence grade: `[macOS-MLX research-signal]`

## What Landed

- Added MLX-native PR95 stage-loss primitives for the public HNeRV recipe:
  cross-entropy, tau-softplus margin, smooth-disagreement margin,
  L7-weighted softplus, pose `sqrt(10*MSE)`, and the combined stage surrogate.
- Wired a typed `pr95_hnerv_mlx_stage_loss_contract.v1` into
  `stage_smoke_config()`, timing-smoke manifests, runtime profiles, and optimizer
  recipes.
- Kept the scorer-network boundary fail-closed: the new contract marks
  `mlx_loss_primitives_implemented=true` but
  `scorer_network_forward_gradient_wired=false`.

## Verification

- `.venv/bin/python -m pytest src/tac/tests/test_pr95_hnerv_mlx_stage_losses.py -q`
  - 8 passed
- `.venv/bin/python -m pytest src/tac/tests/test_pr95_hnerv_mlx_stage_losses.py src/tac/tests/test_pr95_hnerv_mlx.py src/tac/tests/test_pr95_hnerv_mlx_training.py src/tac/tests/test_run_pr95_mlx_timing_smoke.py src/tac/tests/test_pr95_mlx_optimizer_matrix_queue.py -q`
  - 47 passed
- `.venv/bin/python -m ruff check src/tac/local_acceleration/pr95_hnerv_mlx_stage_losses.py src/tac/local_acceleration/pr95_hnerv_mlx.py src/tac/tests/test_pr95_hnerv_mlx_stage_losses.py`
  - passed

## Remaining PR95 MLX Reproduction Blockers

- Frozen SegNet/PoseNet forward and gradient path is still not wired into MLX
  training.
- The matrix profile remains independent timing cells, not checkpointed 8-stage
  resume lineage.
- QAT/C1a/resume semantics and cosine schedule continuity remain only metadata
  in the current MLX timing spine.
- Full-frame inflate parity against the source runtime is still required.
- Exact `[contest-CPU]` / `[contest-CUDA]` auth eval remains the only score
  authority.

## Next Patch

Add a fail-closed `pr95_faithful_reproduction_readiness.v1` helper that consumes
the existing archive/runtime/parity/stage-loss contracts and refuses readiness
until full source-video coverage, chained checkpoints, scorer loss, trained
latents, full-frame parity, and exact auth anchors are present.
