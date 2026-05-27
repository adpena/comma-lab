# Codex Findings: PR95 MLX Pair Indexing Fix

UTC: 2026-05-27T00:34:43Z

## Summary

The PR95 MLX long-training pipeline was allocating one latent per decoded frame
and training only decoder output frame 0. That made every source-faithful full
run structurally incompatible with the public PR95 archive grammar: the contest
video has 1200 frames, while PR95 stores 600 pair latents shaped `(600, 28)`.

This pass changes the default long-training iterator to adjacent-frame pair
semantics. Each sampled latent index now maps to frames `(2*i, 2*i+1)`, targets
are shaped `(B, 2, H, W, 3)`, loss compares both decoded frames, and setup
allocates `frame_count // 2` trainable latents. A full 1200-frame MLX run should
therefore export a `(600, 28)` latent table suitable for the PR95 packaging
contract.

## Landed Changes

- `src/tac/local_acceleration/pr95_hnerv_mlx_long_training.py`
  - `MLXPairIterator` now exposes `pair_count`.
  - Default `next_frame_pair` sampling returns adjacent frame-pair targets.
  - `MLXLongTrainingPipeline.setup()` initializes one latent per pair, not one
    latent per frame.
  - `loss_fn()` compares both decoded frames against pair targets.

- `src/tac/tests/test_pr95_mlx_long_training_infrastructure.py`
  - Adds a regression that verifies adjacent-frame pair targets and `pair_count`.
  - Updates the training-step latent-update test to pass pair-shaped targets.

## Smoke Evidence

Command:

```bash
.venv/bin/python tools/run_pr95_mlx_long_training.py \
  --output-report experiments/results/pr95_mlx_pair_indexing_smoke_20260527T0032Z/report.json \
  --checkpoint-root experiments/results/pr95_mlx_pair_indexing_smoke_20260527T0032Z/checkpoints \
  --telemetry-path experiments/results/pr95_mlx_pair_indexing_smoke_20260527T0032Z/telemetry.jsonl \
  --smoke-mode \
  --execute-smoke \
  --smoke-epochs-per-stage 1 \
  --checkpoint-every-epochs 1 \
  --max-frames 4 \
  --operator-run-label pr95_pair_indexing_smoke_20260527T0032Z
```

Result:

- executed smoke succeeded;
- Stage 8 checkpoint:
  `experiments/results/pr95_mlx_pair_indexing_smoke_20260527T0032Z/checkpoints/stage08_converge_low_lr_epoch000008_20260527T003443Z.pt`;
- exported `latents` shape is `(2, 28)`, matching `max_frames // 2`.

Full-count shape/proof smoke:

```bash
.venv/bin/python tools/run_pr95_mlx_long_training.py \
  --output-report experiments/results/pr95_mlx_pair_indexing_fullcount_smoke_20260527T0042Z/report.json \
  --checkpoint-root experiments/results/pr95_mlx_pair_indexing_fullcount_smoke_20260527T0042Z/checkpoints \
  --telemetry-path experiments/results/pr95_mlx_pair_indexing_fullcount_smoke_20260527T0042Z/telemetry.jsonl \
  --smoke-mode \
  --execute-smoke \
  --smoke-epochs-per-stage 1 \
  --checkpoint-every-epochs 1 \
  --max-frames 1200 \
  --operator-run-label pr95_pair_indexing_fullcount_smoke_20260527T0042Z
```

Result:

- decoded `source_video_frame_count = 1200`;
- Stage 8 checkpoint:
  `experiments/results/pr95_mlx_pair_indexing_fullcount_smoke_20260527T0042Z/checkpoints/stage08_converge_low_lr_epoch000008_20260527T003808Z.pt`;
- exported `latents` shape is `(600, 28)`;
- public PR95 state-dict keys are present (`blocks.0.weight`,
  `refine.0.weight`) and MLX-internal keys are absent (`blocks.0.conv.weight`,
  `refine0.weight`);
- peak MLX memory in the smoke telemetry was about 4.54 GB.

Byte-closed package using MLX-trained latents:

```bash
.venv/bin/python tools/package_pr95_mlx_pytorch_state_dict_to_contest_archive.py \
  --input-pt experiments/results/pr95_mlx_pair_indexing_fullcount_smoke_20260527T0042Z/checkpoints/stage08_converge_low_lr_epoch000008_20260527T003808Z.pt \
  --source-archive-zip experiments/results/lightning_batch/exact_eval_public_pr95_hnerv_muon_t4_fix2_20260504T0848Z/archive.zip \
  --source-submission-root experiments/results/public_pr_intake_full/public_pr95_intake_20260505_auto/source/submissions/hnerv_muon \
  --output-submission-dir experiments/results/pr95_mlx_pair_indexing_fullcount_smoke_20260527T0042Z/stage08_latents_from_pt_package \
  --latents-from-pt \
  --report-out experiments/results/pr95_mlx_pair_indexing_fullcount_smoke_20260527T0042Z/stage08_latents_from_pt_package_report.json
```

Result:

- package succeeded;
- archive bytes: `247350`;
- archive SHA-256:
  `d186f874a836de59e0991c9168ddc6244f15e8c0bc4d15b60993ecbe854524f7`;
- parsed latent shape after roundtrip: `[600, 28]`.

Runtime consumption proof:

```bash
.venv/bin/python tools/prove_pr95_public_archive_runtime_consumption.py \
  --archive-zip experiments/results/pr95_mlx_pair_indexing_fullcount_smoke_20260527T0042Z/stage08_latents_from_pt_package/archive.zip \
  --inflate-sh experiments/results/pr95_mlx_pair_indexing_fullcount_smoke_20260527T0042Z/stage08_latents_from_pt_package/inflate.sh \
  --output-json experiments/results/pr95_mlx_pair_indexing_fullcount_smoke_20260527T0042Z/stage08_latents_from_pt_runtime_consumption.json \
  --allow-large-output
```

Result:

- `runtime_consumption_proven = true`;
- raw output bytes: `3662409600`;
- raw output SHA-256:
  `d5e54d70af6080650a85cd62616e2a525dadceb40023601e30541e547dee0352`;
- exact readiness remains false because this is not source-runtime parity or
  paired contest CPU/CUDA auth eval.

## Verification

```bash
.venv/bin/python -m ruff check src/tac/local_acceleration/pr95_hnerv_mlx_long_training.py src/tac/tests/test_pr95_mlx_long_training_infrastructure.py
.venv/bin/python -m pytest src/tac/tests/test_pr95_mlx_long_training_infrastructure.py -q
```

## Remaining Work

- Run a full-video or staged `--max-frames 1200` timing smoke to measure wall
- Run a longer full-video MLX training queue after deciding stage/epoch budget.
- Run full inflate parity against the source PR95 runtime/archive pair.
- Characterize MLX/PyTorch scorer drift on this MLX-trained package.
- Dispatch exact CPU/CUDA auth eval only after readiness gates pass.

No score claim, promotion eligibility, rank/kill eligibility, or exact dispatch
authority is produced by this local MLX smoke.
