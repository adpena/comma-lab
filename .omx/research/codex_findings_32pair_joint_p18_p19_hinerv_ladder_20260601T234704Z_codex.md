# Codex findings: 32-pair joint P18/P19 HiNeRV ladder

UTC: 2026-06-01T23:47:04Z
Author: Codex
Axis: `[macOS-CPU advisory]` acquisition + `[macOS-MLX research-signal]` training
Score authority: false
Promotion authority: false

## 32-pair finite surface

Command class:

```bash
PYTHONPATH=$PWD/src:$PWD /Users/adpena/Projects/pact/.venv/bin/python \
  tools/build_joint_recon_pixel_weight_surface.py \
  --output-dir /Volumes/VertigoDataTier/pact/experiments/results/codex_joint_recon_weight_torch_32pair_20260601T234423Z \
  --source-video-path /Users/adpena/Projects/pact/upstream/videos/0.mkv \
  --upstream-dir /Users/adpena/Projects/pact/upstream \
  --num-pairs 32 \
  --pair-chunk-size 4 \
  --scorer-device cpu \
  --scorer-backend torch \
  --overwrite
```

Result:

- Manifest: `/Volumes/VertigoDataTier/pact/experiments/results/codex_joint_recon_weight_torch_32pair_20260601T234423Z/joint_p18_p19_recon_pixel_weight_manifest.json`
- Weight: `/Volumes/VertigoDataTier/pact/experiments/results/codex_joint_recon_weight_torch_32pair_20260601T234423Z/joint_p18_p19_recon_pixel_weight.npz`
- Weight SHA-256: `83a055193abc5b97d50779b14319776bf28d9669cf69b4b2ef45f17705ce9ae8`
- Shape: `(32,2,384,512,1)`
- Bytes: `45300784`
- Backend: `torch_exact_cpu_scorer_vjp.v1`
- Blockers: `[]`
- `training_consumption_recommended=true`
- Raw/video scratch retained: none

This is the first non-toy finite exact-scorer P18/P19 surface for the current
HiNeRV MLX carrier path.

## 32-pair 1-epoch consumption smoke

Artifact:

- Report: `/Volumes/VertigoDataTier/pact/experiments/results/codex_hi_nerv_torch_joint_recon_weight_32pair_consumption_smoke_20260601T234620Z/compact_renderer_mlx_spine_runner_report.json`
- Consumed weight SHA-256: `83a055193abc5b97d50779b14319776bf28d9669cf69b4b2ef45f17705ce9ae8`
- Shape: `(32,2,384,512,1)`
- `score_claim=false`
- `ready_for_exact_eval_dispatch=false`
- Raw/video scratch retained: none

The 1-epoch smoke proved loader/trainer consumption, but retained the expected
`hi_nerv_pr95_faithful_curriculum_requires_min_8_epochs` blocker.

## 32-pair 8-epoch gate

Command class:

```bash
PYTHONPATH=$PWD/src:$PWD /Users/adpena/Projects/pact/.venv/bin/python \
  tools/run_compact_renderer_mlx_spine_runner.py \
  --execute-family hi_nerv \
  --output-dir /Volumes/VertigoDataTier/pact/experiments/results/codex_hi_nerv_torch_joint_recon_weight_32pair_8epoch_20260601T234704Z \
  --num-pairs 32 \
  --epochs 8 \
  --batch-pairs 4 \
  --learning-rate 1e-3 \
  --source-video-path /Users/adpena/Projects/pact/upstream/videos/0.mkv \
  --upstream-dir /Users/adpena/Projects/pact/upstream \
  --segnet-distillation-weight 0.01 \
  --pose-distillation-weight 0.0001 \
  --recon-pixel-weight-path /Volumes/VertigoDataTier/pact/experiments/results/codex_joint_recon_weight_torch_32pair_20260601T234423Z/joint_p18_p19_recon_pixel_weight.npz \
  --coder-aware-qat \
  --coder-qat-quant-bits 4 \
  --coder-qat-quant-residual-weight 0.001 \
  --coder-qat-magnitude-weight 0.0001 \
  --coder-qat-delta-weight 0.0002 \
  --repo-root "$PWD" \
  --overwrite
```

Result:

- Report: `/Volumes/VertigoDataTier/pact/experiments/results/codex_hi_nerv_torch_joint_recon_weight_32pair_8epoch_20260601T234704Z/compact_renderer_mlx_spine_runner_report.json`
- Archive: `/Volumes/VertigoDataTier/pact/experiments/results/codex_hi_nerv_torch_joint_recon_weight_32pair_8epoch_20260601T234704Z/hi_nerv_mlx_training/archive.zip`
- Archive bytes: `41553`
- Archive SHA-256: `adc953258067dec7fe96d396b6fceb5c27902cc9990a7127c2519fd7025e808a`
- Receiver proof: `/Volumes/VertigoDataTier/pact/experiments/results/codex_hi_nerv_torch_joint_recon_weight_32pair_8epoch_20260601T234704Z/hi_nerv_mlx_training/receiver_proof/hi_nerv_mlx_receiver_proof.json`
- Receiver contract satisfied: `true`
- Runtime consumption proof passed: `true`
- Receiver output SHA-256: `76be8c00083a2d43bdd44e3e9622775743519e78c3d6aa8f5fe27e5531b2c8ba`
- Receiver output bytes during proof: `195328512`
- Receiver output retained: `false`
- Raw/video scratch retained after proof: none

The 8-epoch gate removed the min-8 curriculum blocker. Remaining blockers:

- `contest_cpu_cuda_exact_eval_not_executed`
- `local_cpu_replay_not_run_partial_pair_coverage`
- `full_video_mlx_scorer_replay_not_attached`
- `no_full_coverage_compact_base_candidate`
- `no_full_coverage_candidate_under_any_hard_ceiling`

This is a valid byte-closed partial-coverage smoke, not a score candidate.

## Interpretation

The useful change is structural: exact scorer-gradient acquisition now produces
finite P18/P19 surfaces at 32-pair scale, and HiNeRV MLX can consume them through
the score-aware/coder-QAT spine while preserving receiver proof and cleanup. The
remaining score-lowering work is no longer blocked on nonfinite gradient
production; it is blocked on scale, replay, and quality.

## Next action

Scale to 128 pairs with the same Torch acquisition backend and MLX consumer.
Before full-600, add/attach local MLX scorer replay to the runner output so the
planner can observe component movement instead of only archive/runtime proof.
Exact CPU/CUDA remains forbidden until a full-coverage byte-closed local winner
exists.
