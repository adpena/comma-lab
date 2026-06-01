# Codex findings: Torch exact-CPU P18/P19 gradient fallback

UTC: 2026-06-01T23:41:40Z
Author: Codex
Axis: `[macOS-CPU advisory]` producer + `[macOS-MLX research-signal]` consumer
Score authority: false
Promotion authority: false

## What changed

The joint P18/P19 recon-pixel-weight producer now has two backend modes behind
the same artifact schema:

- `--scorer-backend mlx`: fast MLX direct scorer VJP. This remains useful for
  diagnosis, but on the real upstream scorer it currently records nonfinite VJP
  blockers and refuses training recommendation.
- `--scorer-backend torch`: exact-CPU differentiable scorer fallback using
  `tac.scorer.load_differentiable_scorers`, including the canonical
  autograd-preserving PoseNet YUV6 preprocess patch. This backend is slower but
  finite on the real 2-pair smoke and emits a training-consumable surface.

This preserves the MLX-first strategy where it matters: MLX remains the carrier
training/export consumer. The Torch backend is only the acquisition surface
producer for exact scorer gradients when direct MLX scorer autograd is unhealthy.

## Real 2-pair Torch producer smoke

Command class:

```bash
PYTHONPATH=$PWD/src:$PWD /Users/adpena/Projects/pact/.venv/bin/python \
  tools/build_joint_recon_pixel_weight_surface.py \
  --output-dir /Volumes/VertigoDataTier/pact/experiments/results/codex_joint_recon_weight_torch_2pair_20260601T234140Z \
  --source-video-path /Users/adpena/Projects/pact/upstream/videos/0.mkv \
  --upstream-dir /Users/adpena/Projects/pact/upstream \
  --num-pairs 2 \
  --pair-chunk-size 1 \
  --scorer-device cpu \
  --scorer-backend torch \
  --overwrite
```

Artifact:

- Manifest: `/Volumes/VertigoDataTier/pact/experiments/results/codex_joint_recon_weight_torch_2pair_20260601T234140Z/joint_p18_p19_recon_pixel_weight_manifest.json`
- Weight: `/Volumes/VertigoDataTier/pact/experiments/results/codex_joint_recon_weight_torch_2pair_20260601T234140Z/joint_p18_p19_recon_pixel_weight.npz`
- Weight SHA-256: `057005547f7171c1f4dd244162cdc7c1fe08defc463ddbd3263e1d67c19d0469`
- Array SHA-256: `05d44d768bbbee862b5bde13f78447b57da234699124e86421ed1488ebace21c`
- Shape: `(2,2,384,512,1)`
- Bytes: `2833246`
- Surface backend: `torch_exact_cpu_scorer_vjp.v1`
- `training_consumption_recommended=true`
- Blockers: `[]`
- Raw/video scratch retained: none

Observed stats:

- Weight min/mean/max/std: `0.0484753922 / 1.0 / 46.2163734 / 1.6327807`
- Seg saliency nonfinite count: `0`
- Pose saliency nonfinite count: `0`
- Seg saliency nonzero fraction: `0.5`
- Pose saliency nonzero fraction: `1.0`

## HiNeRV MLX consumption smoke

Command class:

```bash
PYTHONPATH=$PWD/src:$PWD /Users/adpena/Projects/pact/.venv/bin/python \
  tools/run_compact_renderer_mlx_spine_runner.py \
  --execute-family hi_nerv \
  --output-dir /Volumes/VertigoDataTier/pact/experiments/results/codex_hi_nerv_torch_joint_recon_weight_consumption_smoke_20260601T234157Z \
  --num-pairs 2 \
  --epochs 1 \
  --batch-pairs 1 \
  --learning-rate 1e-3 \
  --source-video-path /Users/adpena/Projects/pact/upstream/videos/0.mkv \
  --upstream-dir /Users/adpena/Projects/pact/upstream \
  --segnet-distillation-weight 0.01 \
  --pose-distillation-weight 0.0001 \
  --recon-pixel-weight-path /Volumes/VertigoDataTier/pact/experiments/results/codex_joint_recon_weight_torch_2pair_20260601T234140Z/joint_p18_p19_recon_pixel_weight.npz \
  --coder-aware-qat \
  --coder-qat-quant-bits 4 \
  --coder-qat-quant-residual-weight 0.001 \
  --coder-qat-magnitude-weight 0.0001 \
  --coder-qat-delta-weight 0.0002 \
  --repo-root "$PWD" \
  --overwrite
```

Result:

- Report: `/Volumes/VertigoDataTier/pact/experiments/results/codex_hi_nerv_torch_joint_recon_weight_consumption_smoke_20260601T234157Z/compact_renderer_mlx_spine_runner_report.json`
- Mode: `executed_hi_nerv_mlx_scoreaware_and_exported`
- Consumed weight SHA-256: `057005547f7171c1f4dd244162cdc7c1fe08defc463ddbd3263e1d67c19d0469`
- Consumed shape: `(2,2,384,512,1)`
- `authority=false_macos_mlx_research_signal`
- `score_claim=false`
- `ready_for_exact_eval_dispatch=false`
- Raw/video scratch retained: none

## Verification

```bash
/Users/adpena/Projects/pact/.venv/bin/python -m ruff check \
  src/tac/substrates/_shared/mlx_score_aware/loss.py \
  src/tac/substrates/_shared/mlx_score_aware/bundle.py \
  src/tac/substrates/_shared/mlx_score_aware/tests/test_recon_pixel_weight_channel.py \
  src/tac/optimization/recon_pixel_weight_surface.py \
  src/tac/optimization/tests/test_recon_pixel_weight_surface.py \
  tools/run_compact_renderer_mlx_spine_runner.py \
  tools/build_joint_recon_pixel_weight_surface.py \
  src/tac/tests/test_compact_renderer_mlx_spine_runner.py

PYTHONPATH=$PWD/src:$PWD /Users/adpena/Projects/pact/.venv/bin/python -m pytest \
  --import-mode=importlib -q \
  src/tac/substrates/_shared/mlx_score_aware/tests/test_recon_pixel_weight_channel.py \
  src/tac/optimization/tests/test_recon_pixel_weight_surface.py \
  src/tac/tests/test_compact_renderer_mlx_spine_runner.py
```

Observed: `ruff` passed and `48 passed`.

## Next action

Scale the acquisition surface in this order:

1. Torch backend: 32-pair finite P18/P19 surface.
2. HiNeRV/SNeRV MLX smoke consuming that 32-pair surface.
3. If local MLX metrics move in the right direction, run 128-pair and then
   full-600 surface/training campaigns.
4. Keep exact promotion gated: byte-closed export, local CPU replay, exact CPU
   auth, then CUDA only after CPU clears.

Do not route `--scorer-backend mlx` surfaces into real training while they carry
nonfinite-gradient blockers. Continue hardening direct MLX scorer VJP in
parallel, but the score-lowering lane now has a finite acquisition backend.
