# Codex Findings: MLX Dataset-Verified Bundle Builder

UTC: 2026-05-22T10:00:00Z
Lane: `lane_mlx_dataset_verified_contract_bundle_builder_20260522`

## Summary

Hardened `tools/build_mlx_production_contract_bundle.py` so a durable MLX
production-contract bundle can be bound to a scorer-response dataset at build
time. With `--dataset`, the tool now fails the bundle unless every MLX
scorer-response row is covered by a strict child production contract.

## Fix

- Added `--dataset` to `tools/build_mlx_production_contract_bundle.py`.
- Filters MLX rows by `family=mlx_scorer_response` or
  `source_schema=mlx_scorer_response.v1`.
- Reuses the canonical `build_mlx_production_contract_gate(..., rows=...)`
  coverage logic from the LL planner.
- Embeds `dataset_coverage_gate` into the bundle artifact.
- If coverage fails, marks the bundle itself:
  - `passed=false`
  - `verdict=FAIL_MLX_SCORER_PRODUCTION_CONTRACT_BUNDLE`
  - blocker `dataset_mlx_row_coverage_gate_not_strict_pass`

## Real PR101 Artifact

Built a dataset-verified rich-identity bundle:

`experiments/results/mlx_strict_score_calibration_pr101_pose_axis_20260522/candidate_production_contract_bundle_v2_dataset_verified_rich_identity.json`

SHA-256:

`18e1cc7d78318fed7f50cc4702ac89d4d32d9727ea716f665c25f5936dc676ae`

Result:

- `PASS_MLX_SCORER_PRODUCTION_CONTRACT_BUNDLE`
- `dataset_coverage_gate.status=strict_pass`
- `dataset_mlx_row_count=1`
- `dataset_matched_row_count=1`

## Verification

- `.venv/bin/ruff check tools/build_mlx_production_contract_bundle.py src/tac/tests/test_mlx_production_contract.py`
- `.venv/bin/python -m pytest -q src/tac/tests/test_mlx_production_contract.py`
- `.venv/bin/ruff check tools/build_mlx_production_contract_bundle.py src/tac/local_acceleration/mlx_production_contract.py src/tac/optimization/scorer_response_dataset.py src/tac/tests/test_mlx_production_contract.py src/tac/tests/test_scorer_response_dataset.py src/tac/tests/test_plan_ll_scorer_response_next_cli.py`
- `.venv/bin/python -m pytest -q src/tac/tests/test_mlx_production_contract.py src/tac/tests/test_scorer_response_dataset.py src/tac/tests/test_plan_ll_scorer_response_next_cli.py src/tac/tests/test_mlx_scorer_response.py`

Result: 124 passed.

## Authority Boundary

The bundle remains local MLX research signal only. Dataset coverage does not
create score authority, promotion eligibility, rank/kill authority, or dispatch
readiness without contest CPU/CUDA auth eval.
