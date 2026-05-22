# Codex Findings: Scorer-Response Validation Gate

UTC: 2026-05-22T02:59:01Z

## Verdict

PROCEED_WITH_GUARDS. The LL scorer-response planner now has a first-class validation gate for the previously prose-only requirement: family-diverse held-out correlation before local response rows can influence exact-eval spend selection. The gate is non-authoritative and fail-closed.

## Code Changes

- Added `scorer_response_dataset_validation_gate.v1` via `build_scorer_response_validation_gate`.
- Added `tools/validate_scorer_response_dataset.py`.
- Added Markdown rendering for validation-gate outputs.
- Wired `build_next_probe_plan` to attach the validation gate and add `do_not_use_response_dataset_for_exact_eval_selection` when the gate is blocked.
- Kept standalone validation strict, while planner attachment converts invalid legacy/minimal datasets into blockers instead of aborting older null-byte/parity checks.

## Full-300 MLX Validation Result

Input dataset:

`experiments/results/mlx_singleton_window_harvest_fec6_20260522T0250Z_full300/windowed_scorer_response_dataset_300rows.json`

Validation gate:

`experiments/results/mlx_singleton_window_harvest_fec6_20260522T0250Z_full300/response_validation_gate_300rows.json`

Status: `blocked`

Blockers:

- `family_count_below_min:1<2`
- `families_with_required_folds_below_min:1<2`
- `no_prediction_fields_present`

Coverage that now passes:

- Rows: `300`
- Global folds present: `0,1,2,3,4`
- Fold counts: `{0:66, 1:58, 2:67, 3:53, 4:56}`

Coverage that still fails:

- Only one family is present: `mlx_scorer_response`.
- No explicit prediction field is present, so held-out correlation cannot be computed.

## Regenerated LL Plan

Plan:

`experiments/results/mlx_singleton_window_harvest_fec6_20260522T0250Z_full300/ll_next_probe_plan_windowed_300rows_v2_validation_gate.json`

The planner still prioritizes `ll_mlx_cpu_stable_response_harvest`, but now includes the validation prohibition:

`do_not_use_response_dataset_for_exact_eval_selection`

This terminalizes the current MLX corpus state: row count and fold coverage are sufficient, but family diversity and prediction-vs-observed held-out correlation are still missing.

## Verification

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  src/tac/tests/test_scorer_response_dataset.py \
  src/tac/tests/test_mlx_response_windows.py \
  src/tac/tests/test_mlx_scorer_response.py
```

Result: `51 passed`.

```bash
git diff --check
```

Result: pass.

## Next Action

Add at least one second candidate-response family with fold coverage, then attach explicit prediction fields from a simple LL surrogate or deterministic response model. The validation gate should remain blocked until prediction-vs-observed held-out correlation passes; only then should MLX local signal be allowed to influence exact-eval spend filtering.
