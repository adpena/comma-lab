# Codex Findings: MLX Rich-Identity Contract Regeneration

UTC: 2026-05-22T09:10:00Z
Parent commit: `6a7ab4bda`

## Summary

Regenerated the PR101 pose-axis strict MLX production contract using the
post-bundle-gate rich identity schema. The previous strict v2 contract remained
a valid pass under its original schema, but it did not carry the
candidate/reference scorer-input cache-array hashes or PosNet/SegNet component
hashes that the new bundle-aware LL planner can now require.

## Artifacts

- Contract:
  `experiments/results/mlx_strict_score_calibration_pr101_pose_axis_20260522/candidate_production_contract_strict_v3_rich_identity.json`
  - SHA-256:
    `638616eedfc7ea727f3c1176dc8f411069498cd832354df9052a02f3ccdbce75`
  - Verdict: `PASS_MLX_SCORER_PRODUCTION_CONTRACT`
- Dataset:
  `experiments/results/mlx_strict_score_calibration_pr101_pose_axis_20260522/candidate_scorer_response_dataset_production_contract_gate_v3_rich_identity.json`
  - SHA-256:
    `4dbbd914dd23a7c5d865329cb2bab915aba269ad43349d23ba5e88168e1e6264`
- Planner:
  `experiments/results/mlx_strict_score_calibration_pr101_pose_axis_20260522/ll_next_probe_plan_production_contract_gate_v3_rich_identity.json`
  - SHA-256:
    `801ef7674a26347d45744f72acea2f3d148169f9d9b6ba75460a0961895f70b9`
  - `mlx_production_contract_gate.status`: `strict_pass`

## Rich Identity Present

The regenerated contract summary now carries:

- `candidate_cache_array_sha256`
- `reference_cache_array_sha256`
- `posenet_sha256`
- `segnet_sha256`
- archive SHA-256, inflated-output aggregate SHA-256, batch pairs, sample count,
  and pair window

## Remaining Planner Guards

The v3 planner no longer emits a production-contract prohibition. It still
correctly blocks exact-eval selection authority because the response dataset is
only one row / one family:

- `do_not_use_response_dataset_for_exact_eval_selection`
- `do_not_widen_coordinate_sparse_residual_sidecar`

This remains a non-authoritative local MLX research signal for candidate
generation and exact-eval spend triage only; contest CPU/CUDA auth eval remains
required for score claims and promotion.
