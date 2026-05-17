# L5 v2 TT5L side-info variant packets

- schema: `tt5l_sideinfo_variant_packets_v1`
- source_archive_path: `experiments/results/time_traveler_l5_v2/tt5l_current_code_fullshape_sideinfo_cpu_advisory_20260517T052719Z/archive.zip`
- source_archive_sha256: `2a8cf3389aaeb217a96058bf7b47b43e4fc364193795d66fe7ab75479102d11f`
- source_archive_bytes: `43323`
- source_archive_member: `0.bin`
- source_sideinfo_liveness: `{'checked': True, 'dtype': 'int8', 'shape': [600, 45], 'total_values': 27000, 'nonzero_values': 16627, 'nonzero_fraction': 0.6158148148148148, 'total_pairs': 600, 'nonzero_pair_count': 600, 'all_zero_pair_count': 0, 'nonzero_pair_fraction': 1.0, 'min_nonzero_values_per_pair': 18, 'max_nonzero_values_per_pair': 38, 'mean_nonzero_values_per_pair': 27.711666666666666, 'section_liveness': {'checked': True, 'per_pair_bytes': 45, 'sections': {'se3_lie': {'offset': 0, 'width': 12, 'total_values': 7200, 'nonzero_values': 4436, 'nonzero_fraction': 0.6161111111111112}, 'seg_boundary': {'offset': 12, 'width': 18, 'total_values': 10800, 'nonzero_values': 6636, 'nonzero_fraction': 0.6144444444444445}, 'hf_residual': {'offset': 30, 'width': 6, 'total_values': 3600, 'nonzero_values': 2234, 'nonzero_fraction': 0.6205555555555555}, 'predict_residual': {'offset': 36, 'width': 9, 'total_values': 5400, 'nonzero_values': 3321, 'nonzero_fraction': 0.615}}}, 'liveness_warnings': [], 'min': -4, 'max': 4}`
- runtime: `{'available': True, 'submission_dir': 'experiments/results/time_traveler_l5_v2/tt5l_current_code_fullshape_sideinfo_cpu_advisory_20260517T052719Z/submission_dir', 'runtime_tree_sha256': '708cf594cb8dfb31e6a4c86e40f4b904919a486c9fb35d7049f322d07805511d', 'runtime_content_tree_sha256': 'bf857cb79055e914b64e5cd5788a5a15a37824e01b32c871b84715c153c0ab32', 'runtime_file_count': 10, 'blockers': []}`
- score_claim: `false`
- promotion_eligible: `false`
- ready_for_exact_eval_dispatch: `false`
- blockers: `['requires_paired_cpu_cuda_exact_eval_for_sideinfo_effect_curve', 'requires_dispatch_lane_claim_before_auth_eval', 'score_claim_forbidden_until_effect_curve_artifact_passes']`

## Variant archives

| Variant | Archive bytes | Nonzero side-info values | Member changed | Side-section changed | Seed | SHA-256 | Blockers |
| --- | ---: | ---: | --- | --- | ---: | --- | --- |
| `zero` | 34373 | 0 | True | True | 20260517 | `b444cc91f102c9807a865ed59f182ca5c83f3239a49ec2aa400b497d7dea37a3` | `['requires_paired_cpu_cuda_exact_eval_before_score_claim']` |
| `random_lsb` | 38681 | 27000 | True | True | 20260517 | `ccce77aaf1907d6e70d8cba498261708b241ac7d78a9bf22978aa459cb6b7fd1` | `['requires_paired_cpu_cuda_exact_eval_before_score_claim']` |
| `shuffled` | 43284 | 16627 | True | True | 20260517 | `c235e5cb91f4122c3bb642354b424dd21f8b37cc0d2ac7b3e03c0b84dcc49bc3` | `['requires_paired_cpu_cuda_exact_eval_before_score_claim']` |
| `trained` | 43323 | 16627 | False | False | 20260517 | `f08299c5e77908eeeb82cc9948e530fd5a790894902a726ebd8a258596b4bf1a` | `['requires_paired_cpu_cuda_exact_eval_before_score_claim']` |
| `ablated` | 42419 | 13306 | True | True | 20260517 | `ec343265899859495fca1d2874c1b85d211210ea13a2de5bb7e52c9816ba6b39` | `['requires_paired_cpu_cuda_exact_eval_before_score_claim']` |

## Variant generation rules

- `zero`: seed=`20260517`; rule=`np.zeros_like(source_sideinfo)`
- `random_lsb`: seed=`20260517`; rule=`default_rng(20260517).choice([-1, 1], size=source_shape)`
- `shuffled`: seed=`20260517`; rule=`default_rng(20260517).permutation(source_rows)`
- `trained`: seed=`20260517`; rule=`source_sideinfo.copy()`
- `ablated`: seed=`20260517`; rule=`source_sideinfo with predict_residual slice zeroed`

## Classification

These archives are packet controls for the TT5L side-info effect curve. They are not score claims and are not dispatch-ready until a lane claim and paired CPU/CUDA exact-eval cells exist for each variant.
