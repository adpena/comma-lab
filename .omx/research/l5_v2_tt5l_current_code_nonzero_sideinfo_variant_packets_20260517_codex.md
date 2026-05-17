# L5 v2 TT5L side-info variant packets

- schema: `tt5l_sideinfo_variant_packets_v1`
- source_archive_path: `experiments/results/time_traveler_l5_v2/tt5l_current_code_nonzero_sideinfo_cpu_advisory_20260517T025048Z/archive.zip`
- source_archive_sha256: `33f27f82649b08af0bb1ea987911a96f943a397603f42eab8d3cca83d700d6b4`
- source_archive_bytes: `27147`
- source_archive_member: `0.bin`
- source_sideinfo_liveness: `{'checked': True, 'dtype': 'int8', 'shape': [2, 45], 'total_values': 90, 'nonzero_values': 62, 'nonzero_fraction': 0.6888888888888889, 'total_pairs': 2, 'nonzero_pair_count': 2, 'all_zero_pair_count': 0, 'nonzero_pair_fraction': 1.0, 'min_nonzero_values_per_pair': 31, 'max_nonzero_values_per_pair': 31, 'mean_nonzero_values_per_pair': 31.0, 'section_liveness': {'checked': True, 'per_pair_bytes': 45, 'sections': {'se3_lie': {'offset': 0, 'width': 12, 'total_values': 24, 'nonzero_values': 17, 'nonzero_fraction': 0.7083333333333334}, 'seg_boundary': {'offset': 12, 'width': 18, 'total_values': 36, 'nonzero_values': 25, 'nonzero_fraction': 0.6944444444444444}, 'hf_residual': {'offset': 30, 'width': 6, 'total_values': 12, 'nonzero_values': 7, 'nonzero_fraction': 0.5833333333333334}, 'predict_residual': {'offset': 36, 'width': 9, 'total_values': 18, 'nonzero_values': 13, 'nonzero_fraction': 0.7222222222222222}}}, 'liveness_warnings': [], 'min': -3, 'max': 3}`
- runtime: `{'available': True, 'submission_dir': 'experiments/results/time_traveler_l5_v2/tt5l_current_code_nonzero_sideinfo_cpu_advisory_20260517T025048Z/submission_dir', 'runtime_tree_sha256': '42eee25c81c81056ab5c9bf5e6c50a0494ec4e95e92a28479096b7ba81cc9894', 'runtime_content_tree_sha256': 'bf857cb79055e914b64e5cd5788a5a15a37824e01b32c871b84715c153c0ab32', 'runtime_file_count': 10, 'blockers': []}`
- score_claim: `false`
- promotion_eligible: `false`
- ready_for_exact_eval_dispatch: `false`
- blockers: `['requires_paired_cpu_cuda_exact_eval_for_sideinfo_effect_curve', 'requires_dispatch_lane_claim_before_auth_eval', 'score_claim_forbidden_until_effect_curve_artifact_passes', 'tt5l_source_num_pairs_not_full_contest:2_expected_600']`

## Variant archives

| Variant | Archive bytes | Nonzero side-info values | SHA-256 | Blockers |
| --- | ---: | ---: | --- | --- |
| `zero` | 27101 | 0 | `9eb24ed76fe107c5c73b6f00e6c33826d0f8163430a9f2f5d2a18d87fa9cb162` | `['requires_paired_cpu_cuda_exact_eval_before_score_claim']` |
| `random_lsb` | 27127 | 90 | `1b6dac0668f80a4095bf93dca88fe78b6bc7202e7de251d7443bd52fe864d882` | `['requires_paired_cpu_cuda_exact_eval_before_score_claim']` |
| `shuffled` | 27147 | 62 | `f6c7cdfe2846a7147ed9fddb40f9b7d5eedb6c27bc645d24989736ec954adbe0` | `['requires_paired_cpu_cuda_exact_eval_before_score_claim']` |
| `trained` | 27147 | 62 | `6a0af64e5a30d77e44369e2265f2d4023f4d989fae9dc331a3b9457c5ebe2baf` | `['requires_paired_cpu_cuda_exact_eval_before_score_claim']` |
| `ablated` | 27140 | 49 | `b0463b8be3864d8d354df85c9bf80d65da9fb534b5e1102fc7963d9932339bff` | `['requires_paired_cpu_cuda_exact_eval_before_score_claim']` |

## Classification

These archives are packet controls for the TT5L side-info effect curve. They are not score claims and are not dispatch-ready until a lane claim and paired CPU/CUDA exact-eval cells exist for each variant.
