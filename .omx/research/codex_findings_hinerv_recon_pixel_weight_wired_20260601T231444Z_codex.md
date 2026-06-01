# Codex Findings - HiNeRV Recon Pixel Weight Wiring

Date: 2026-06-01T23:14:44Z
Agent: Codex
Scope: HiNeRV MLX train/export/archive control arm

## Verdict

HiNeRV now has an executable `recon_pixel_weight` input path in the queue-owned
compact renderer runner. This closes the immediate orphan-signal gap between
full-video P18/P19 gradient or saliency producers and the MLX training harness:
a file-backed `.npy`/`.npz` weight map can now be supplied to
`--execute-family hi_nerv`, and an opt-in SegNet top-2 boundary map can be
generated from real upstream SegNet teacher logits for quick P18-only probes.

This is not score authority. The output remains a `[macOS-MLX research-signal]`
training surface until a byte-closed full-coverage archive passes local CPU
replay and then exact contest CPU/CUDA auth eval.

## What Landed

- `tools/run_compact_renderer_mlx_spine_runner.py`
  - Added file-backed `--recon-pixel-weight-path` support for HiNeRV.
  - Added `--auto-segnet-boundary-recon-weight` as a P18-only teacher-logit
    boundary saliency producer.
  - Added strict validation for `(384,512)`, `(384,512,1/3)`, and
    `(1,384,512,1/3)` weight maps.
  - Preserves SHA-256, path, source kind, stats, normalization mode, and
    false-authority metadata in the final runner report.
  - Fails closed if both file-backed and auto-generated maps are requested.

- `src/tac/tests/test_compact_renderer_mlx_spine_runner.py`
  - Parser coverage for the new HiNeRV controls.
  - File-custody and bad-shape tests for `recon_pixel_weight` loading.
  - Final-report propagation coverage through the HiNeRV execute wrapper.

## Smoke Evidence

Artifact:
`/Volumes/VertigoDataTier/pact/experiments/results/codex_hi_nerv_auto_seg_boundary_weight_smoke_20260601T231444Z`

Command family:
`tools/run_compact_renderer_mlx_spine_runner.py --execute-family hi_nerv`

Configuration:
- `num_pairs=2`
- `epochs=1`
- real upstream SegNet and PoseNet teachers attached
- `--auto-segnet-boundary-recon-weight`
- coder-aware QAT enabled with 4-bit proposal pressure

Observed:
- `mode=executed_hi_nerv_mlx_scoreaware_and_exported`
- archive bytes: `39537`
- archive SHA-256:
  `2f5d08822b1ffe20d94c02067abc5a8e5d726be3d98baa6339a98534c2763579`
- receiver proof:
  `/Volumes/VertigoDataTier/pact/experiments/results/codex_hi_nerv_auto_seg_boundary_weight_smoke_20260601T231444Z/hi_nerv_mlx_training/receiver_proof/hi_nerv_mlx_receiver_proof.json`
- receiver contract satisfied: `true`
- runtime consumption proof ready: `true`
- score claim: `false`
- no `.raw`, `.mp4`, or `.mkv` scratch retained in the artifact tree

Recon metadata:
- schema: `compact_recon_pixel_weight.v1`
- source kind: `auto_segnet_top2_boundary_margin`
- P18 term: `top2_margin_exp_boundary_saliency_from_real_teacher`
- P19 term: `not_included_use_recon_pixel_weight_path_for_joint_map`
- stats shape: `[384,512]`
- stats mean: `0.026966461911797523`
- stats max: `0.994162380695343`

Expected blockers remain:
- partial-pair coverage only
- no full-video MLX replay attached
- no local CPU replay for partial coverage
- no exact contest CPU/CUDA auth eval

## Tests

- `python -m ruff check tools/run_compact_renderer_mlx_spine_runner.py src/tac/tests/test_compact_renderer_mlx_spine_runner.py`
- `PYTHONPATH=$PWD/src:$PWD python -m pytest --import-mode=importlib -q src/tac/tests/test_compact_renderer_mlx_spine_runner.py`

Result: `22 passed`.

## Next Planner Action

Feed the full-video joint P18/P19 shard reducer into this file-backed
`--recon-pixel-weight-path` input, then run the staged HiNeRV ladder:

1. 32-pair MLX prefilter with joint P18/P19 weight map.
2. 128-pair MLX prefilter if 32-pair distortion moves materially below the
   previous demoted HiNeRV/QAT runs.
3. 600-pair full-video MLX replay only if sampled runs clear the hard demotion
   threshold.
4. Local CPU replay only for byte-closed full-video MLX winners.
5. Exact contest CPU, then CUDA, only after local CPU win.

The auto SegNet boundary mode is useful for fast P18-only probes, but it is not
pose-protective. The promotion path should prefer file-backed joint P18/P19 maps
with PoseNet Mahalanobis null/protection terms included.
