# Codex Findings: FEC6 Auth MLX Parent Contract

- UTC: 20260522T110007Z
- Lane: mlx_fec6_auth_parent_production_contract
- Evidence grade: macOS-MLX research-signal only
- Score authority: false
- Promotion eligible: false

## Result

FEC6 now has a strict local MLX parent production contract for the `[0, 300]`
CPU singleton parent window:

- Contract: `experiments/results/mlx_parent_contract_prereqs_20260522T1030Z/fec6_auth_parent_contract_strict_v1.json`
- Verdict: `PASS_MLX_SCORER_PRODUCTION_CONTRACT`
- Blockers: none
- Required gates passed: cache identity, cache/auth audit, candidate Torch parity, reference Torch parity, profile stability, score calibration
- Warning retained: `batch_invariance_not_required_for_singleton_response`

This contract is still non-authoritative for contest score/rank/promotion. It is
a production-safe local MLX acceleration signal only.

## Inputs

- Auth tensor cache: `modal_fec6_pr101_cpu_auth_tensors_20260522T060605Z/fec6_pr101_cpu_auth_tensors_20260522T060605Z/scorer_input_cache_tensors`
- Cache/auth audit: `experiments/results/mlx_fec6_auth_tensor_cache_local_20260522T1022Z/cache_vs_modal_cpu_auth_tensor_cache_audit.json`
- Parent response: `experiments/results/mlx_fec6_auth_parent_response_20260522T1023Z/candidate_parent_0000_0300.json`
- Candidate Torch parity: `experiments/results/mlx_parent_contract_prereqs_20260522T1030Z/fec6_auth_candidate_torch_parity_sweep_cpu_singleton_pairs0_300.json`
- Reference Torch parity: `experiments/results/mlx_parent_contract_prereqs_20260522T1030Z/reference_torch_parity_sweep_cpu_singleton_pairs0_300.json`
- Profile stability: `experiments/results/mlx_parent_contract_prereqs_20260522T1030Z/fec6_auth_profile_stability_cpu_singleton_pairs0_300_repeat2.json`
- Score calibration: `experiments/results/mlx_parent_contract_prereqs_20260522T1030Z/fec6_plus_pr101_pose_candidate_score_calibration_cpu.json`

## Empirical Details

Auth cache identity:

- Archive SHA-256: `6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf`
- Inflated aggregate SHA-256: `10c68e4266e79fc3e878fd20136e8aaa56262b3a2ff45eed7b8d5a4b1e1ee66d`
- Raw SHA-256: `fef02ccd53ad4355f2dbb8e0b9cd4efb847daa243bd35a8411c5260d584fda8b`
- `posenet_yuv6_pair` hash: `04687540fa97209157b2ab9bcb200d098169f826db54aa4ced00c48c312bca91`
- `segnet_last_rgb` hash: `59ea5240178801774d59314a2de98764e3dba2d33c7ce3acc995bd2e87e6806d`

Parent MLX response:

- `canonical_score`: `0.19224369340250186`
- `avg_posenet_dist`: `3.968034641692005e-05`
- `avg_segnet_dist`: `0.0005345662435865961`
- `n_samples`: `300`
- `batch_pairs`: `1`
- `pair_window`: `[0, 300]`

Parity:

- Candidate parity: `PASS_MLX_TORCH_SCORER_PARITY_SWEEP`, 300 windows, 0 failed
- Reference parity: `PASS_MLX_TORCH_SCORER_PARITY_SWEEP`, 300 windows, 0 failed
- Candidate max SegNet argmax diff: 1 pixel
- Reference max SegNet argmax diff: 1 pixel

Profile:

- Profile stability: `PASS_MLX_PROFILE_STABILITY`
- Rows: 2
- Recommended row: CPU singleton, `pairs_per_second=1.197969982777083`

Score calibration:

- Rows: FEC6 auth parent plus PR101 pose candidate
- `mlx_spend_triage_pairwise_certified_count`: 1
- `mlx_spend_triage_pairwise_uncertain_count`: 0
- `recommended_min_mlx_gap_for_spend_triage`: `0.000961882606981268`

## Combined Dataset Plan

Updated combined FEC6-auth plus decoder-q dataset:

- Dataset: `experiments/results/mlx_fec6_auth_decoderq_same_axis_600row_dataset_20260522T1028Z/mlx_fec6_auth_decoderq_same_axis_600row_dataset.json`
- Parent plan: `experiments/results/mlx_parent_contract_plan_20260522T1108Z_fec6_strict_contract/parent_production_contract_plan.json`
- MLX rows: 600
- Required parent groups: 2
- Covered parent groups: 1
- Missing parent groups: 1
- Remaining blocker: `mlx_parent_contract_group_uncovered:mlx_parent_contract_f5391bf78f60224c`

FEC6 group `mlx_parent_contract_b04a89b260e7715a` is covered by the strict
contract. Decoder-q group `mlx_parent_contract_f5391bf78f60224c` still lacks an
auth-axis cache audit and strict parent production contract.

## Next Action

The MLX production hardening lane should now move to decoder-q:

1. Run a contest-CPU auth eval/cache hash or tensor export for decoder-q archive `022ac0f391bc9408c357575496c3b680fc5cf9da6ca85d23c3ff994c370a1347`.
2. Materialize or download the decoder-q auth tensor cache locally.
3. Regenerate the decoder-q parent response from auth-faithful tensors.
4. Re-run candidate/reference parity, profile stability, score calibration, and strict parent contract.

