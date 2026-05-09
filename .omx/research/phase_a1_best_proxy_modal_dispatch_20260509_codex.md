# Phase A1 Best-Proxy Modal Dispatch - 2026-05-09

## Status

Active Modal T4 dispatch, no score claim.

- Lane: `track1_phase_a1_score_gradient`
- Instance/job id: `track1_phase_a1_score_gradient_bestproxy_lr2e6_20260509_codex`
- Modal call id: `fc-01KR5MQ0WYS8TQWDN7HCYBZZ3B`
- Modal app run:
  `https://modal.com/apps/adpena/main/ap-BDwM4Ky8wliVz2LUzDWxXB`
- Claim timestamp: `2026-05-09T05:52:37Z`
- Predicted ETA: `2026-05-09T09:52:37Z`
- Estimated cost: `$2.36`

## Config

This reuses the best measured A1 hyperparameter basin and changes only the
checkpoint selection policy:

- `epochs=40`
- `steps_per_epoch=8`
- `batch_size=4`
- `lr=0.000002`
- `max_frames=1200`
- `aux_kl_weight=0.2`
- `aux_pixel_l1_weight=0.01`
- `checkpoint_selection=best_proxy`
- `continue_after_nvdec_failure=true`

## Input Custody

- PR101 archive bytes: `178258`
- PR101 archive SHA-256:
  `b83bf3488625dbd73adeddff91712994197ab53098e578e91327a0c6e49efb3e`
- PR101 source snapshot bytes: `19137`
- PR101 source snapshot SHA-256:
  `cf7853a09a08654daa5a6363eba0e36f2b5d2ac9060999f7b799d3d99f8a6a17`
- Video bytes: `37545489`
- Video SHA-256:
  `2611f5f3e186f3529777749f97bd4cce3a208d6b3559e137bd45d256980d2fa9`

## Commands

Dispatch command:

```bash
PYTHONPATH=src:upstream:$PWD .venv/bin/modal run --detach \
  experiments/modal_phase_a1_score_gradient_pr101.py \
  --label track1_phase_a1_score_gradient_bestproxy_lr2e6_20260509_codex \
  --epochs 40 \
  --steps-per-epoch 8 \
  --batch-size 4 \
  --lr 0.000002 \
  --max-frames 1200 \
  --aux-kl-weight 0.2 \
  --aux-pixel-l1-weight 0.01 \
  --checkpoint-selection best_proxy \
  --continue-after-nvdec-failure \
  --train-timeout-hours 3.5 \
  --build-timeout-minutes 10 \
  --eval-timeout-minutes 30 \
  --timeout-hours 4 \
  --cost-cap-usd 8
```

Harvest command:

```bash
.venv/bin/python experiments/modal_phase_a1_score_gradient_pr101.py recover \
  --label track1_phase_a1_score_gradient_bestproxy_lr2e6_20260509_codex
```

## Evidence Discipline

This row is `[advisory only - dispatch in flight]`. The run is not a score
claim and cannot change lane status until recovered artifacts are reviewed.

Required review packet after harvest:

- archive bytes and SHA-256;
- selected checkpoint name and manifest;
- runtime-tree SHA-256;
- `contest_auth_eval.json` components if CUDA eval runs;
- exact formula recomputation;
- dispatch claim terminal row;
- paired CPU handling before public-axis promotion.
