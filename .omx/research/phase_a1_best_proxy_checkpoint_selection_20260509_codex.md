# Phase A1 Best-Proxy Checkpoint Selection - 2026-05-09

## Purpose

The latest guarded A1 refire showed that stronger KL/L1 regularization is not
the next score-lowering lever. It regressed both the exact CUDA axis and the
macOS CPU advisory screen. The next A1 run should not blindly build the final
epoch EMA checkpoint when the training proxy has already identified a better
earlier epoch.

This patch adds an explicit checkpoint-selection surface:

- `checkpoint_ema.pt`: historical final EMA checkpoint.
- `checkpoint_best_proxy.pt`: EMA snapshot at the epoch whose last training
  step has the lowest `weighted_proxy`.
- `checkpoint_best_proxy_manifest.json`: selected epoch, selected proxy value,
  and selected training metrics.

The Modal A1 wrapper now accepts:

```bash
--checkpoint-selection final_ema
--checkpoint-selection best_proxy
```

Default remains `final_ema` for backward compatibility. `best_proxy` is opt-in.

## Why This Is Score-Lowering Work

The current best A1 exact anchor is the latent-aligned refire:

- Archive bytes: `178262`
- `[contest-CPU]`: `0.19284757743677347`
- `[contest-CUDA]`: `0.2263520234784395`

The guarded refire with `kl=0.5`, `pixel_l1=0.02` regressed to:

- Archive bytes: `178279`
- `[contest-CUDA]`: `0.22655968711150934`
- `[macOS-CPU advisory]`: `0.19309483549345535`

The result points to selection/early-stopping, not stronger auxiliary losses.
Best-proxy selection lets one GPU run produce a final checkpoint and a selected
checkpoint with separate archive/eval custody.

## Verification

```bash
.venv/bin/python -m pytest \
  tests/test_train_score_gradient_pr101_real_data.py \
  src/tac/tests/test_modal_phase_a1_score_gradient_pr101.py \
  tests/test_modal_phase_a1_recover_paths.py -q
```

Result: `18 passed`.

Plan surface check:

```bash
.venv/bin/python experiments/modal_phase_a1_score_gradient_pr101.py plan \
  --label a1_best_proxy_unit_plan \
  --epochs 3 \
  --checkpoint-selection best_proxy \
  --json-out /tmp/a1_best_proxy_plan.json
```

Result: plan is ready and the dispatch command includes
`--checkpoint-selection best_proxy`.

## Dispatch Candidate

Recommended next dispatch, after commit/push and active-lane check:

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

This reuses the best measured A1 hyperparameter basin but changes the built
checkpoint from final EMA to best proxy. It is not a score claim until exact
CUDA and paired CPU artifacts are harvested and reviewed.
