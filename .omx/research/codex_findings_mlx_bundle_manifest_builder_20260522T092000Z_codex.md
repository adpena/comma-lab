# Codex Findings: MLX Bundle Manifest Builder

UTC: 2026-05-22T09:20:00Z
Lane: `lane_mlx_production_contract_bundle_manifest_builder_20260522`

## Summary

Added a reusable MLX production-contract bundle builder so strict local MLX
contract coverage can be materialized as a durable manifest instead of only
synthesized inside `tools/plan_ll_scorer_response_next.py`.

## Implementation

- Added `build_mlx_scorer_production_contract_bundle_manifest(...)`.
- Added constants:
  - `mlx_scorer_production_contract_bundle.v1`
  - `PASS_MLX_SCORER_PRODUCTION_CONTRACT_BUNDLE`
  - `FAIL_MLX_SCORER_PRODUCTION_CONTRACT_BUNDLE`
- Added `tools/build_mlx_production_contract_bundle.py`.
- Reused canonical bundle constants from the LL planner gate.
- Added unit and CLI coverage for passing and blocked bundle manifests.

## Real PR101 Artifact

Built a persisted bundle for the regenerated rich-identity PR101 pose-axis
contract:

`experiments/results/mlx_strict_score_calibration_pr101_pose_axis_20260522/candidate_production_contract_bundle_v1_rich_identity.json`

SHA-256:

`82e4b73ea007f0ad666fe309e0dc8c8d9ff9560b22074c95907c5e81b6522811`

Bundle verdict:

- `PASS_MLX_SCORER_PRODUCTION_CONTRACT_BUNDLE`
- `contract_count=1`
- `strict_contract_count=1`

The LL planner accepts the persisted bundle with:

- `mlx_production_contract_gate.status=strict_pass`
- `source_schema=mlx_scorer_production_contract_bundle.v1`
- `matched_row_count=1`

## Verification

- `.venv/bin/ruff check src/tac/local_acceleration/mlx_production_contract.py src/tac/optimization/scorer_response_dataset.py tools/build_mlx_production_contract_bundle.py tools/plan_ll_scorer_response_next.py src/tac/tests/test_mlx_production_contract.py src/tac/tests/test_scorer_response_dataset.py src/tac/tests/test_plan_ll_scorer_response_next_cli.py`
- `.venv/bin/python -m pytest -q src/tac/tests/test_mlx_production_contract.py src/tac/tests/test_scorer_response_dataset.py src/tac/tests/test_plan_ll_scorer_response_next_cli.py src/tac/tests/test_build_scorer_response_dataset_cli.py`

Result: 111 passed.

## Authority Boundary

The bundle remains explicitly non-authoritative: `score_claim=false`,
`promotion_eligible=false`, `rank_or_kill_eligible=false`, and
`ready_for_exact_eval_dispatch=false`. It permits only local MLX signal use for
candidate generation and exact-eval spend triage after strict row coverage.
