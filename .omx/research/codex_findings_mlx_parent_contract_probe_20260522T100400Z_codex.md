# Codex Findings: MLX Parent Contract Probe

Date: 2026-05-22T10:04:00Z

## Authority

- Score claim: `False`
- Promotion eligible: `False`
- Ready for exact-eval dispatch: `False`
- Rank/kill eligible: `False`
- Spend triage authority: `False`

## Change

Extended `build_mlx_parent_production_contract_plan` so discovered parent
response candidates also carry a dry-run strict production-contract probe. This
keeps the parent-contract queue actionable: the plan now reports not only which
parent responses could cover the 600 singleton rows, but also the exact blockers
that prevent each parent response from becoming a strict MLX production
contract.

## Live result

Command:

```bash
.venv/bin/python tools/plan_mlx_parent_production_contracts.py \
  --dataset experiments/results/mlx_source_identity_refresh_20260522T0945Z/mlx_fec6_decoderq_same_axis_600row_baseline_structural_oof_predicted_dataset_identity_refreshed.json \
  --production-contract experiments/results/mlx_strict_score_calibration_pr101_pose_axis_20260522/candidate_production_contract_bundle_v2_dataset_verified_rich_identity.json \
  --json-out experiments/results/mlx_parent_contract_plan_20260522T100352Z/parent_production_contract_plan.json \
  --md-out experiments/results/mlx_parent_contract_plan_20260522T100352Z/parent_production_contract_plan.md \
  --allow-blocked-output
```

Result remains blocked:

- MLX rows: `600`
- Required parent contracts: `2`
- Covered parent groups: `0`
- Missing parent groups: `2`

FEC6 parent probe:

- Parent response:
  `experiments/results/mlx_singleton_window_harvest_fec6_20260522T0250Z_full300/candidate_parent_0000_0300.json`
- Verdict: `FAIL_MLX_SCORER_PRODUCTION_CONTRACT`
- Blockers:
  - `response_promotable_not_false`
  - `response_cache_identity_candidate_audit_missing`
  - `cache_auth_audit_manifest_not_supplied`
  - `torch_parity_manifest_not_supplied`
  - `reference_torch_parity_manifest_not_supplied`
  - `profile_stability_manifest_not_supplied`
  - `score_calibration_manifest_not_supplied`

Decoder-q parent probe:

- Parent response:
  `experiments/results/mlx_same_axis_decoderq_d1f1e56e042692f2_20260522T031205Z/candidate_parent_0000_0300.json`
- Verdict: `FAIL_MLX_SCORER_PRODUCTION_CONTRACT`
- Blockers are identical to FEC6.

## Interpretation

The gate is now precise and correctly conservative. The immediate next work is
not another MLX planner change; it is to build or regenerate the two parent
responses with explicit `promotable: false`, attach a passing auth-axis cache
identity audit, generate candidate/reference torch parity sweeps, generate
profile stability, generate strict auth-axis score calibration for these exact
identities, then build a two-contract bundle with dataset coverage.

Do not treat the existing PR101 pose-axis strict bundle as transferable. It is
strict for its own identity and still covers `0/600` rows for the FEC6 +
decoder-q dataset.

## Verification

- `.venv/bin/python -m ruff check src/tac/optimization/scorer_response_dataset.py src/tac/tests/test_scorer_response_dataset.py tools/plan_mlx_parent_production_contracts.py`
- `.venv/bin/python -m pytest src/tac/tests/test_scorer_response_dataset.py src/tac/tests/test_mlx_production_contract.py -q`

Both passed.
