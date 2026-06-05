# HiNeRV Dynamic-Range Repair Guard

Generated: 2026-06-05T00:26:07Z

Axis: `[macOS-CPU/MLX local:false-authority]`

This memo is strictly local false-authority work. It does not claim score,
promotion eligibility, rank/kill authority, or exact replay readiness.

## Starting Artifact

- Harvest: `.omx/research/hinerv_stage_qat_smoke_harvest_20260604Tcodex.json`
- Output root: `/Volumes/VertigoDataTier/pact/experiments/results/hinerv_stage_qat_runner_smoke_20260604T2111Z_codex`
- Archive: `/Volumes/VertigoDataTier/pact/experiments/results/hinerv_stage_qat_runner_smoke_20260604T2111Z_codex/hi_nerv_mlx_training/ema_archive_selection/ema/archive.zip`
- Archive bytes/SHA-256: `106351`, `85c1c44936560f87d0cf300392bdb8cba2cfe16abad47f7a9ba239960ca80890`
- Verdict: `BYTE_ATTRACTIVE_BUT_RENDERER_DEGENERATE_NOT_A_FRONTIER_CANDIDATE`

The receiver cache-quality gate failed with
`FUNDAMENTAL_RENDERER_OUTPUT_DEGENERATE`. The sampled local scorer number is
not admissible because scorer inputs were out of distribution.

Receiver cache xray from the harvest:

- SegNet last RGB: `std=0.517579197883606`, `dynamic_range=4.9344635009765625`, `mae_vs_reference=101.44928741455078`
- PoseNet YUV6 pair: `std=0.43814340233802795`, `dynamic_range=3.3023681640625`, `mae_vs_reference=69.05482482910156`

## Landed Guardrail

The shared MLX scorer-input distribution guard now includes a differentiable
per-frame RGB dynamic-range loss in addition to mean, std, and soft saturation.
This wires dynamic-range repair into the training loss before receiver replay
or downstream materializers consume the archive.

Touched training surfaces:

- `src/tac/substrates/_shared/mlx_score_aware/loss.py`
- `experiments/train_substrate_hi_nerv_mlx_local.py`

Focused tests:

- `src/tac/substrates/_shared/mlx_score_aware/tests/test_loss_adapter_harness.py`
- `src/tac/tests/test_train_substrate_hi_nerv_mlx_local.py`

## Verification

Commands run:

```bash
uv run pytest src/tac/substrates/_shared/mlx_score_aware/tests/test_loss_adapter_harness.py::test_scorer_input_distribution_guard_includes_dynamic_range_term src/tac/substrates/_shared/mlx_score_aware/tests/test_loss_adapter_harness.py::test_adapter_train_step_emits_active_score_loss_parts src/tac/tests/test_train_substrate_hi_nerv_mlx_local.py::test_hinerv_train_time_control_config_is_explicit_and_false_authority src/tac/tests/test_train_substrate_hi_nerv_mlx_local.py::test_hinerv_full_control_contract_clears_when_pr95_controls_are_present -q
uv run pytest src/tac/substrates/_shared/mlx_score_aware/tests/test_loss_adapter_harness.py src/tac/tests/test_train_substrate_hi_nerv_mlx_local.py -q
uv run ruff check src/tac/substrates/_shared/mlx_score_aware/loss.py src/tac/substrates/_shared/mlx_score_aware/tests/test_loss_adapter_harness.py experiments/train_substrate_hi_nerv_mlx_local.py src/tac/tests/test_train_substrate_hi_nerv_mlx_local.py
git diff --check
```

Results:

- `4 passed in 0.98s`
- `69 passed in 2.78s`
- `ruff`: all checks passed
- `git diff --check`: clean

## SSD-Backed Rerun Command

This is the next local smoke target: train with the dynamic-range guard active,
export to SSD, run the receiver cache-quality stop condition, and skip local CPU
replay until the cache-quality gate stops reporting degenerate renderer output.

```bash
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
uv run python tools/run_compact_renderer_mlx_spine_runner.py \
  --execute-family hi_nerv \
  --allow-manual-compact-family-launch \
  --modelsize-candidate-id auto \
  --hard-byte-ceiling 178000 \
  --output-dir "/Volumes/VertigoDataTier/pact/experiments/results/hinerv_dynamic_range_guard_smoke_${RUN_ID}_codex" \
  --num-pairs 2 \
  --epochs 8 \
  --batch-pairs 1 \
  --learning-rate 0.001 \
  --segnet-distillation-weight 1.0 \
  --pose-distillation-weight 1.0 \
  --scorer-input-distribution-guard-weight 0.01 \
  --hi-nerv-optimizer-policy pr95_curriculum \
  --hi-nerv-pr95-muon-policy every_stage \
  --hi-nerv-pr95-curriculum-total-epochs 8 \
  --coder-aware-qat \
  --coder-qat-quant-bits 7 \
  --coder-qat-c1a-entropy-weight 0.02 \
  --coder-qat-c1a-sigma 0.1 \
  --coder-qat-c1a-sample-size 64 \
  --receiver-cache-quality-max-pairs 1 \
  --receiver-cache-quality-batch-pairs 1 \
  --receiver-cache-quality-min-segnet-dynamic-range 16.0 \
  --skip-local-cpu-replay
```

Expected pass/fail authority remains local only. A pass only means the next
artifact is no longer blocked by receiver-cache dynamic-range degeneracy before
local replay; it is still not an exact contest replay result.

## New Crux

The latest runner report records resolved controls but not the exact original
argv. That is separate from this minimal renderer/cache-quality fix, but it is
worth hardening later because smoke harvests should be directly re-runnable
from their own report without reconstructing parser flags.
