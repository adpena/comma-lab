# Codex findings: joint P18/P19 recon-pixel-weight surface

UTC: 2026-06-01T23:35:02Z
Author: Codex
Axis: `[macOS-MLX research-signal]`
Score authority: false
Promotion authority: false

## What landed

This landing upgrades the MLX score-aware recon-pixel-weight channel from static
maps to dynamic pair/frame maps. `RendererBundle.recon_pixel_weight` now accepts
`(N,H,W,C)` and `(N,2,H,W,C)` maps, so a carrier can receive different weights
per pair and per frame instead of one frozen global mask.

New reusable producer:

- `src/tac/optimization/recon_pixel_weight_surface.py`
- `tools/build_joint_recon_pixel_weight_surface.py`

The producer emits a file-backed `joint_p18_p19_recon_pixel_weight.npz` with
shape `(N,2,H,W,1)` and a manifest with explicit false-authority fields. The
CLI prints blocker state and `training_consumption_recommended` so blocked maps
cannot be quietly treated as training authority.

## Real 2-pair producer smoke

Command class:

```bash
PYTHONPATH=$PWD/src:$PWD /Users/adpena/Projects/pact/.venv/bin/python \
  tools/build_joint_recon_pixel_weight_surface.py \
  --output-dir /Volumes/VertigoDataTier/pact/experiments/results/codex_joint_recon_weight_2pair_20260601T233502Z \
  --source-video-path /Users/adpena/Projects/pact/upstream/videos/0.mkv \
  --upstream-dir /Users/adpena/Projects/pact/upstream \
  --num-pairs 2 \
  --pair-chunk-size 1 \
  --scorer-device cpu \
  --overwrite
```

Artifact:

- Manifest: `/Volumes/VertigoDataTier/pact/experiments/results/codex_joint_recon_weight_2pair_20260601T233502Z/joint_p18_p19_recon_pixel_weight_manifest.json`
- Weight: `/Volumes/VertigoDataTier/pact/experiments/results/codex_joint_recon_weight_2pair_20260601T233502Z/joint_p18_p19_recon_pixel_weight.npz`
- Weight SHA-256: `054d2a23230a8d9b94a8d1c6004be4f4be0d78946515406ec000eef2d77137a8`
- Array SHA-256: `3e352f15a310bf840b52a14b0673c11bd2b7be72320e57c36b163b5cb1226f59`
- Shape: `(2,2,384,512,1)`
- Bytes: `1438268`
- Raw/video scratch retained: none

## Honest blocker

The real upstream MLX scorer VJP path is not yet healthy enough for campaign
scaling. The producer encountered and sanitized nonfinite gradients in the
SegNet margin term and PoseNet axis terms, then recorded 11 typed blockers:

- `nonfinite_gradient_sanitized:seg_margin_grad_pairs_0_1`
- `nonfinite_gradient_sanitized:pose_axis_0_grad_pairs_0_1`
- `nonfinite_gradient_sanitized:pose_axis_1_grad_pairs_0_1`
- `nonfinite_gradient_sanitized:pose_axis_2_grad_pairs_0_1`
- `nonfinite_gradient_sanitized:seg_margin_grad_pairs_1_2`
- `nonfinite_gradient_sanitized:pose_axis_0_grad_pairs_1_2`
- `nonfinite_gradient_sanitized:pose_axis_1_grad_pairs_1_2`
- `nonfinite_gradient_sanitized:pose_axis_2_grad_pairs_1_2`
- `nonfinite_gradient_sanitized:pose_axis_3_grad_pairs_1_2`
- `nonfinite_gradient_sanitized:pose_axis_4_grad_pairs_1_2`
- `nonfinite_gradient_sanitized:pose_axis_5_grad_pairs_1_2`

Therefore:

- `training_consumption_recommended=false`
- `score_claim=false`
- `ready_for_exact_eval_dispatch=false`

This is not a method negative for P18/P19 waterfilling. It is a backend blocker:
finite scorer-gradient production must be hardened before 32/128/600-pair
campaigns consume the surface for score-lowering training.

## HiNeRV consumption smoke

Command class:

```bash
PYTHONPATH=$PWD/src:$PWD /Users/adpena/Projects/pact/.venv/bin/python \
  tools/run_compact_renderer_mlx_spine_runner.py \
  --execute-family hi_nerv \
  --output-dir /Volumes/VertigoDataTier/pact/experiments/results/codex_hi_nerv_joint_recon_weight_consumption_smoke_20260601T233517Z \
  --num-pairs 2 \
  --epochs 1 \
  --batch-pairs 1 \
  --learning-rate 1e-3 \
  --source-video-path /Users/adpena/Projects/pact/upstream/videos/0.mkv \
  --upstream-dir /Users/adpena/Projects/pact/upstream \
  --segnet-distillation-weight 0.01 \
  --pose-distillation-weight 0.0001 \
  --recon-pixel-weight-path /Volumes/VertigoDataTier/pact/experiments/results/codex_joint_recon_weight_2pair_20260601T233502Z/joint_p18_p19_recon_pixel_weight.npz \
  --coder-aware-qat \
  --coder-qat-quant-bits 4 \
  --coder-qat-quant-residual-weight 0.001 \
  --coder-qat-magnitude-weight 0.0001 \
  --coder-qat-delta-weight 0.0002 \
  --repo-root "$PWD" \
  --overwrite
```

Result:

- Report: `/Volumes/VertigoDataTier/pact/experiments/results/codex_hi_nerv_joint_recon_weight_consumption_smoke_20260601T233517Z/compact_renderer_mlx_spine_runner_report.json`
- Mode: `executed_hi_nerv_mlx_scoreaware_and_exported`
- Consumed weight SHA-256: `054d2a23230a8d9b94a8d1c6004be4f4be0d78946515406ec000eef2d77137a8`
- Consumed shape: `(2,2,384,512,1)`
- `authority=false_macos_mlx_research_signal`
- `score_claim=false`
- `ready_for_exact_eval_dispatch=false`
- Raw/video scratch retained: none

This proves the dynamic surface is consumable by HiNeRV. It does not promote the
blocked surface for real campaign training.

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

Observed: `ruff` passed and `47 passed`.

## Next action

Do not scale this map to 32/128/600 until finite scorer-gradient production is
fixed. The next highest-EV implementation step is a finite gradient backend:

1. Harden the MLX scorer adapter path that currently emits nonfinite VJPs, or
2. Add a PyTorch exact-CPU differentiable scorer-gradient fallback that emits the
   same manifest schema and evidence boundaries.

Once finite P18/P19 surfaces are produced, the queued campaign can scale:
`surface -> HiNeRV/SNeRV score-aware fit -> byte-closed export -> local CPU replay -> exact CPU/CUDA only for local winners`.
