# L5 v2 TT5L side-info variant packets

- schema: `tt5l_sideinfo_variant_packets_v1`
- source_archive_path: `experiments/results/lane_substrate_time_traveler_l5_autonomy_modal_a100_dispatch_20260514T100758Z__smoke__25ep_modal/lane_substrate_time_traveler_l5_autonomy_results/output/archive.zip`
- source_archive_sha256: `2b05b7351b690b0b2251ddc620d80dd9a1833051cfa07e679106d00fbc70024a`
- source_archive_bytes: `34603`
- source_archive_member: `0.bin`
- source_sideinfo_liveness: `{'checked': True, 'dtype': 'int8', 'shape': [600, 45], 'total_values': 27000, 'nonzero_values': 0, 'nonzero_fraction': 0.0, 'total_pairs': 600, 'nonzero_pair_count': 0, 'all_zero_pair_count': 600, 'nonzero_pair_fraction': 0.0, 'min_nonzero_values_per_pair': 0, 'max_nonzero_values_per_pair': 0, 'mean_nonzero_values_per_pair': 0.0, 'section_liveness': {'checked': True, 'per_pair_bytes': 45, 'sections': {'se3_lie': {'offset': 0, 'width': 12, 'total_values': 7200, 'nonzero_values': 0, 'nonzero_fraction': 0.0}, 'seg_boundary': {'offset': 12, 'width': 18, 'total_values': 10800, 'nonzero_values': 0, 'nonzero_fraction': 0.0}, 'hf_residual': {'offset': 30, 'width': 6, 'total_values': 3600, 'nonzero_values': 0, 'nonzero_fraction': 0.0}, 'predict_residual': {'offset': 36, 'width': 9, 'total_values': 5400, 'nonzero_values': 0, 'nonzero_fraction': 0.0}}}, 'liveness_warnings': ['tt5l_side_info_some_pairs_all_zero'], 'min': 0, 'max': 0}`
- runtime: `{'available': True, 'submission_dir': 'experiments/results/time_traveler_recovered_exact_eval_20260514_codex/runtime', 'runtime_tree_sha256': '1d19f8b48943bf99f475cd39fd65f65083087536ee2480ecd71241866e1f2012', 'runtime_content_tree_sha256': 'b9bd9ffecaac4d926b4c9174cdd7e23cb872f3139e1abf617cd4423e3a65e9c0', 'runtime_file_count': 10, 'blockers': []}`
- score_claim: `false`
- promotion_eligible: `false`
- ready_for_exact_eval_dispatch: `false`
- blockers: `['requires_paired_cpu_cuda_exact_eval_for_sideinfo_effect_curve', 'requires_dispatch_lane_claim_before_auth_eval', 'score_claim_forbidden_until_effect_curve_artifact_passes', 'tt5l_source_trained_sideinfo_all_zero']`

## Variant archives

| Variant | Archive bytes | Nonzero side-info values | Member changed | Side-section changed | Seed | SHA-256 | Blockers |
| --- | ---: | ---: | --- | --- | ---: | --- | --- |
| `zero` | 34603 | 0 | False | False | 20260517 | `71eef9bc7314dba145d2014ab4908216caf34bee4d388321ac1bd992cb1c6409` | `['requires_paired_cpu_cuda_exact_eval_before_score_claim', 'source_trained_sideinfo_all_zero', 'zero_matches_trained_control']` |
| `random_lsb` | 38911 | 27000 | True | True | 20260517 | `b6a5b63c0ea8acd582d8f273a1ee9e00f74becc9d1993a2f3085f2f89d64b1c7` | `['requires_paired_cpu_cuda_exact_eval_before_score_claim', 'source_trained_sideinfo_all_zero']` |
| `shuffled` | 34603 | 0 | False | False | 20260517 | `71eef9bc7314dba145d2014ab4908216caf34bee4d388321ac1bd992cb1c6409` | `['requires_paired_cpu_cuda_exact_eval_before_score_claim', 'source_trained_sideinfo_all_zero', 'shuffled_variant_degenerate_from_zero_source', 'shuffled_sideinfo_nonzero_missing', 'shuffled_matches_zero_control', 'shuffled_matches_trained_control', 'shuffled_expected_sideinfo_change_missing']` |
| `trained` | 34603 | 0 | False | False | 20260517 | `71eef9bc7314dba145d2014ab4908216caf34bee4d388321ac1bd992cb1c6409` | `['requires_paired_cpu_cuda_exact_eval_before_score_claim', 'source_trained_sideinfo_all_zero', 'trained_variant_degenerate_from_zero_source', 'trained_sideinfo_nonzero_missing']` |
| `ablated` | 34603 | 0 | False | False | 20260517 | `71eef9bc7314dba145d2014ab4908216caf34bee4d388321ac1bd992cb1c6409` | `['requires_paired_cpu_cuda_exact_eval_before_score_claim', 'source_trained_sideinfo_all_zero', 'ablated_variant_degenerate_from_zero_source', 'ablated_matches_zero_control', 'ablated_matches_trained_control', 'ablated_expected_sideinfo_change_missing']` |

## Variant generation rules

- `zero`: seed=`20260517`; rule=`np.zeros_like(source_sideinfo)`
- `random_lsb`: seed=`20260517`; rule=`default_rng(20260517).choice([-1, 1], size=source_shape)`
- `shuffled`: seed=`20260517`; rule=`default_rng(20260517).permutation(source_rows)`
- `trained`: seed=`20260517`; rule=`source_sideinfo.copy()`
- `ablated`: seed=`20260517`; rule=`source_sideinfo with predict_residual slice zeroed`

## Classification

These archives are packet controls for the TT5L side-info effect curve. They are not score claims and are not dispatch-ready until a lane claim and paired CPU/CUDA exact-eval cells exist for each variant.
