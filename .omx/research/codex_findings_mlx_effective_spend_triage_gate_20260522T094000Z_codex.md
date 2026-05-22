# Codex Findings: MLX Effective Spend Triage Gate

UTC: 2026-05-22T09:40:00Z
Lane: `lane_mlx_effective_spend_triage_gate_20260522`

## Summary

Added a single effective MLX spend-triage gate to the LL scorer-response
planner. This prevents downstream automation from reading a passing constituent
MLX gate, such as `mlx_production_contract_gate.mlx_spend_triage_allowed=true`,
as if the full response dataset were ready for exact-eval selection.

## Fix

Added `ll_effective_mlx_spend_triage_gate.v1`, which requires all of:

- response dataset validation passed
- strict MLX Torch parity sweep
- strict MLX score calibration
- strict MLX production contract coverage
- at least one MLX row

The gate remains non-authoritative: it does not permit score claims,
promotion, rank/kill decisions, or dispatch readiness without contest auth eval.

## Tests

- Current one-row MLX fixture with passing constituent gates remains globally
  blocked by `response_validation_gate_not_passed`.
- Synthetic family-diverse, fold-covered, prediction-validated MLX dataset
  receives `mlx_exact_eval_spend_triage_allowed=true`.

## Verification

- `.venv/bin/ruff check src/tac/optimization/scorer_response_dataset.py src/tac/tests/test_scorer_response_dataset.py tools/plan_ll_scorer_response_next.py`
- `.venv/bin/python -m pytest -q src/tac/tests/test_scorer_response_dataset.py src/tac/tests/test_plan_ll_scorer_response_next_cli.py`
- `.venv/bin/python -m pytest -q src/tac/tests/test_scorer_response_dataset.py src/tac/tests/test_plan_ll_scorer_response_next_cli.py src/tac/tests/test_build_scorer_response_dataset_cli.py src/tac/tests/test_mlx_production_contract.py src/tac/tests/test_mlx_scorer_response.py`

Result: 123 passed.
