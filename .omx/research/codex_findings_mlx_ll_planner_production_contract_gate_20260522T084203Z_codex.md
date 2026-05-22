# Codex Findings: MLX LL Planner Production-Contract Gate

timestamp_utc: 2026-05-22T08:42:03Z
agent: codex
lane: lane_mlx_production_contract_ll_planner_gate_20260522
evidence_grade: [macOS-MLX research-signal] calibrated against [contest-CPU]
score_claim: false
promotion_eligible: false
rank_or_kill_eligible: false
ready_for_exact_eval_dispatch: false

## Summary

Closed the LL planner gap where MLX scorer-response rows could be planned with
PyTorch parity and score calibration but without the strict production contract
that binds cache identity, reference and candidate parity, profile stability,
and score calibration to the same response identity.

The planner now carries a normalized
`ll_mlx_production_contract_gate.v1`. MLX rows remain non-authoritative local
signal; the gate only permits local exact-eval spend triage after a strict
`PASS_MLX_SCORER_PRODUCTION_CONTRACT`.

## Structural Fixes

- Added `build_mlx_production_contract_gate(...)` with fail-closed checks for:
  schema, gate-set version, pass verdict, explicit false authority,
  MLX evidence axis, strict gate policy, required constituent gates, and empty
  blockers.
- Bound the production contract to the rows it unlocks by archive SHA-256,
  inflated-output aggregate SHA-256, pair window, batch size, and sample count.
- Fixed a bypass where explicit MLX `response_family` values such as
  `pr101_pose_axis_strict_calibration` avoided the old
  `family == mlx_scorer_response` planner check. Source schema
  `mlx_scorer_response.v1` now gates those rows.
- Added `--mlx-production-contract` to `tools/plan_ll_scorer_response_next.py`.
- Rendered the production-contract gate into planner Markdown.

## Empirical Artifacts

- Dataset:
  `experiments/results/mlx_strict_score_calibration_pr101_pose_axis_20260522/candidate_scorer_response_dataset_production_contract_gate.json`
- Passing planner output:
  `experiments/results/mlx_strict_score_calibration_pr101_pose_axis_20260522/ll_next_probe_plan_production_contract_gate.json`
- The passing plan has `mlx_production_contract_gate.status=strict_pass` and no
  `do_not_use_mlx_rows_for_exact_eval_spend_triage_without_production_contract`
  prohibition.
- Negative control without `--mlx-production-contract` emits
  `do_not_use_mlx_rows_for_exact_eval_spend_triage_without_production_contract`
  even when parity and score calibration pass.

## Verification

- `.venv/bin/python -m pytest -q src/tac/tests/test_scorer_response_dataset.py src/tac/tests/test_plan_ll_scorer_response_next_cli.py`
- `.venv/bin/python -m pytest -q src/tac/tests/test_scorer_response_dataset.py src/tac/tests/test_plan_ll_scorer_response_next_cli.py src/tac/tests/test_build_scorer_response_dataset_cli.py src/tac/tests/test_mlx_production_contract.py`
- `.venv/bin/ruff check src/tac/optimization/scorer_response_dataset.py tools/plan_ll_scorer_response_next.py src/tac/tests/test_scorer_response_dataset.py src/tac/tests/test_plan_ll_scorer_response_next_cli.py`

All passed.

## Next Action

Use `--mlx-production-contract` on all LL scorer-response planner invocations
that include MLX scorer-response rows. Treat any plan missing the production
contract prohibition as eligible only for local training-data harvest, not
exact-eval spend filtering, unless the strict production gate passes for the
same row identity.
