# Codex Findings: MLX Effective Gate Coverage Summary

Date: 2026-05-22T09:37:05Z
Lane: lane_mlx_effective_gate_coverage_summary_20260522
Author: Codex

## Verdict

LANDED. The LL scorer-response planner now exposes MLX production-contract row
coverage directly in the effective MLX spend-triage gate JSON and markdown.
This is a fail-closed observability fix only; it creates no score, rank,
promotion, or exact-eval dispatch authority.

## Why This Was Needed

The current strongest local MLX response dataset can pass response validation,
Torch parity, and score calibration while still failing the production-contract
coverage requirement. Before this patch, the effective gate surfaced only the
generic blocker `mlx_production_contract_gate_not_strict_pass`, requiring a
manual drill-down into the nested gate to see whether the issue was coverage,
identity, or a failed child contract.

## Live Check

Command output artifact:
`experiments/results/mlx_effective_spend_triage_real_gate_20260522T0939Z/ll_next_probe_plan_effective_gate.json`

Input dataset:
`experiments/results/mlx_same_axis_decoderq_d1f1e56e042692f2_20260522T031205Z/mlx_fec6_decoderq_same_axis_600row_baseline_structural_oof_predicted_dataset.json`

Attached strict artifacts:

- `experiments/results/mlx_strict_score_calibration_pr101_pose_axis_20260522/candidate_torch_parity_sweep_cpu_singleton_full600.json`
- `experiments/results/mlx_strict_score_calibration_pr101_pose_axis_20260522/score_calibration_cpu_v2.json`
- `experiments/results/mlx_strict_score_calibration_pr101_pose_axis_20260522/candidate_production_contract_bundle_v2_dataset_verified_rich_identity.json`

Observed effective gate:

- `status`: `blocked`
- `blockers`: `["mlx_production_contract_gate_not_strict_pass"]`
- `response_validation_status`: `passed`
- `torch_parity_status`: `strict_pass`
- `score_calibration_status`: `strict_pass`
- `production_contract_status`: `blocked`
- `mlx_row_count`: `600`
- `production_contract_row_count`: `600`
- `production_contract_matched_row_count`: `0`
- `production_contract_unmatched_row_count`: `600`

Interpretation: the current 600-row FEC6/decoder-q MLX response dataset is not
eligible for local exact-eval spend triage because no rows are covered by a
matching strict production contract. The sampled blockers classify the current
rows as identity-missing at the production-contract gate. This is a concrete
next blocker for the MLX acceleration path, not a model-quality result.

## Code Changes

- `src/tac/optimization/scorer_response_dataset.py`
  - Added production-contract row-count, matched-row, unmatched-row, row-sample,
    and blocker-sample fields to `ll_effective_mlx_spend_triage_gate.v1`.
  - Added full unmatched-row count to the production-contract bundle summary.
  - Rendered the same coverage fields in the next-probe markdown report.
- `src/tac/tests/test_scorer_response_dataset.py`
  - Added coverage-summary assertions for strict-pass and blocked effective
    MLX gates.

## Verification

- `.venv/bin/ruff check src/tac/optimization/scorer_response_dataset.py src/tac/tests/test_scorer_response_dataset.py src/tac/tests/test_plan_ll_scorer_response_next_cli.py`
- `.venv/bin/python -m pytest -q src/tac/tests/test_scorer_response_dataset.py src/tac/tests/test_plan_ll_scorer_response_next_cli.py`
- `.venv/bin/python -m pytest -q src/tac/tests/test_mlx_production_contract.py src/tac/tests/test_plan_ll_scorer_response_next_cli.py`
- Real planner rerun with `--require-effective-mlx-spend-triage`; expected exit
  code `2`, because the effective MLX spend gate is still correctly blocked.

## Next Action

Generate strict production-contract identity coverage for the exact FEC6
singleton-window MLX response rows, or regenerate the response harvest with
complete cache/inflated-output identity fields. Do not use the 600-row dataset
for exact-eval spend triage until `production_contract_matched_row_count == 600`
and the effective gate reaches `strict_pass`.
