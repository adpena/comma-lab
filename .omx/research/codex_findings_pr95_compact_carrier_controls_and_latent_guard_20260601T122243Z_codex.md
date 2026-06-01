# Codex Findings: PR95 Compact Carrier Controls And Latent Guard

UTC: 2026-06-01T12:22:43Z

## Scope

This landing keeps score-lowering focus on compact learned carriers rather than
explicit megabyte residual fields. It hardens the immediate PR95/HNeRV control
path, preserves completed compact-carrier subagent research memos, and prevents a
specific silent-custody failure in PR95 MLX-to-contest packaging.

## What Changed

- `experiments/train_substrate_pact_nerv_selector_v3_mlx_local.py` now exposes
  the shared MLX harness controls needed for PR95-scale runs:
  checkpoint interval, early-stopping patience, PR95 curriculum toggle, scaled
  curriculum total, gradient clipping, warmup, optimizer kind, weight decay, and
  cosine decay.
- `tools/run_pr95_mlx_long_training.py` now exposes
  `--training-loss-surface` and `--curriculum-total-epochs`.
- `src/tac/local_acceleration/pr95_hnerv_mlx_long_training.py` now records the
  training loss surface in provenance, telemetry, checkpoints, reports, and exact
  blockers. The new `rgb_yuv6_mse` control routes through PR95's YUV6
  preprocessing helper, but remains explicitly fail-closed because it is not the
  real SegNet/PoseNet contest scorer objective.
- `tools/package_pr95_mlx_pytorch_state_dict_to_contest_archive.py` now refuses
  to silently pair a trained decoder checkpoint containing `latents` with stale
  source-archive latents unless the operator passes an explicit
  `--allow-source-archive-latents` decoder-only override. The normal promotion
  path is `--latents-from-pt` or `--latents-npy`.

## Empirical State

- Full-video PR95 RGB+YUV6 control smoke completed at
  `/Volumes/VertigoDataTier/pact/pr95_hnerv_mlx_rgb_yuv6_fullvideo_smoke_20260601T121506Z`.
  It decoded all 1200 source frames and emitted fail-closed MLX-local advisory
  artifacts only.
- A 29,650-epoch RGB+YUV6 control launch was stopped after fresh-eyes review
  classified it as a control basin rather than a faithful PR95 scorer lane. No
  artifact was deleted. Checkpoints/logs remain under
  `/Volumes/VertigoDataTier/pact/pr95_hnerv_mlx_rgb_yuv6_29650ep_seed0_20260601T121539Z`.

## Negative/Control Verdict

RGB+YUV6 is better than plain RGB MSE as a local control, but it is not the
frontier lane. It lacks the true PR95 scorer-aware loop: exact source decode
path, scorer preprocessing, SegNet/PoseNet loss, QAT/C1a/EMA/Muon schedule, and
archive-in-loop section pricing. Do not spend long wall-clock on this control
unless the goal is to benchmark infrastructure.

## Next Build

The score-lowering lane is compact learned carriers under PR95-style byte
grammar:

1. PR95/HNeRV faithful Stage-8-from-public-archive scorer-aware continuation.
2. RNeRV/SRNeRV/BoostNeRV comparison under the same archive section grammar.
3. PVQ/RT-VQ-NeRV/C3-style latent-codebook carrier under the same receiver proof
   and byte accounting.

Subagents launched for those three designs:

- Confucius: PR95/HNeRV faithful stack.
- Nash: RNeRV/SRNeRV/BoostNeRV byte grammar stack.
- McClintock: PVQ/RT-VQ-NeRV/C3 latent-codebook stack.

## Verification

- `ruff check --fix` on touched code and tests: passed.
- `pytest src/tac/tests/test_pr95_mlx_long_training_infrastructure.py src/tac/tests/test_pr95_mlx_pytorch_archive_package.py -q`: 19 passed.
- `pytest src/tac/tests/test_pr95_mlx_pytorch_export_bridge.py -q`: 2 passed.

