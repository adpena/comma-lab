# L5 v2 Z6 identity-predictor disambiguator

- schema: `z6_predictive_coding_vs_identity_disambiguator_v1`
- probe_id: `z6_predictive_coding_vs_identity_disambiguator`
- lane_id: `lane_time_traveler_l5_z6_l1_scaffold_substrate_build_20260516`
- evidence_grade: `byte_closed_archive_pair_no_score`
- verdict: `pending_paired_exact_eval_json`
- paired_control_initialization: `shared_modules_seed_order_matched_v2`
- score_claim: `false`
- promotion_eligible: `false`
- rank_or_kill_eligible: `false`
- ready_for_exact_eval_dispatch: `false`
- ready_for_paid_dispatch: `false`
- paradigm_claim_allowed: `false`

This report is a Z6-specific probe surface. It can route the next engineering action, but it is not contest score evidence.

## Source Archives

### full_film_predictor

- path: `experiments/results/z6_candidate4c_paired_auth_stats_paircap2_codex_20260518T0256Z/0.bin`
- bytes: `194747`
- sha256: `4fd2e6cc2801bcb5ee9532b56219eb5468670b185b555d114b23ae30b59ae198`
- zip_path: `experiments/results/z6_candidate4c_paired_auth_stats_paircap2_codex_20260518T0256Z/archive.zip`
- zip_bytes: `194119`
- zip_sha256: `47a776cd36577f2cd6026b71c52187618839e9fae70e30f5d8c7eaf8b1047855`
- zip_members: `['0.bin']`
- zip_member_matches_path_bytes: `True`
- contest_archive_bytes_basis: `194119`
- identity_predictor: `False`
- identity_predictor_disambiguator: `False`
- predictor_state_dict_key_count: `8`
- num_pairs: `2`
- ego_motion_dim: `8`
- predictor_architecture: `single_layer_film_75k`

### identity_predictor

- path: `experiments/results/z6_candidate4c_paired_auth_stats_paircap2_codex_20260518T0256Z/0_identity_predictor_disambiguator.bin`
- bytes: `195066`
- sha256: `062fcdd033dfc9cbb1f55b0139b132ee8976b175500b971223293ead0344658a`
- zip_path: `experiments/results/z6_candidate4c_paired_auth_stats_paircap2_codex_20260518T0256Z/archive_identity_predictor_disambiguator.zip`
- zip_bytes: `194295`
- zip_sha256: `a2c47579f2d7e1e0c7b950e7ca7deca8fb0d8d414fd17a49d4b7946e19479d64`
- zip_members: `['0.bin']`
- zip_member_matches_path_bytes: `True`
- contest_archive_bytes_basis: `194295`
- identity_predictor: `True`
- identity_predictor_disambiguator: `True`
- predictor_state_dict_key_count: `8`
- num_pairs: `2`
- ego_motion_dim: `8`
- predictor_architecture: `single_layer_film_75k`

## Paired Archive Checks
- encoder_state_dict_equal: `True`
- decoder_state_dict_equal: `True`
- predictor_state_dict_equal: `True`
- predictor_keysets_equal: `True`
- latent_init_equal: `True`
- residuals_equal: `True`
- ego_motion_equal: `True`

## Runtime Custody
- schema: `z6_inflate_runtime_closure_v1`
- runtime_root: `experiments/results/z6_candidate4c_paired_auth_stats_paircap2_codex_20260518T0256Z/submission_dir`
- entrypoint: `experiments/results/z6_candidate4c_paired_auth_stats_paircap2_codex_20260518T0256Z/submission_dir/inflate.sh`
- aggregate_sha256: `384938f3b6a14acb5944938c45c127ccc411f00983aff7589c7c9b98cfc56073`
- file_count: `10`
- total_bytes: `62261`
- archive_payloads_excluded_from_runtime_hash: `['0.bin', '0_identity_predictor_disambiguator.bin', 'archive.zip', 'archive_identity_predictor_disambiguator.zip']`
- python_cache_files_excluded_from_runtime_hash: `13`

## Inflate Output Comparison
- evidence_axis: `[local-inflate-output advisory]`
- evidence_grade: `local_inflate_output_comparison_no_score`
- runtime_output_changed: `True`
- same_output_file_set: `True`
- same_output_aggregate_sha256: `False`
- total_byte_differences: `22253`
- inflate_sh_path: `experiments/results/z6_candidate4c_paired_auth_stats_paircap2_codex_20260518T0256Z/submission_dir/inflate.sh`
- file_list_path: `upstream/public_test_video_names.txt`
- output_root: `experiments/results/z6_candidate4c_paired_auth_stats_paircap2_codex_20260518T0256Z/z6_identity_inflate_output_comparison_runtime_source_custody_compact_20260518_codex`
- full_output_aggregate_sha256: `2b0e3345c5eb8f00beb71de62e0bdda60cc17933acbbe775ef451b2791aacf73`
- identity_output_aggregate_sha256: `856f15ea620ff704a3bebcdfef75f289e7af11d0d1e7b45f0fee7262505c6409`
- full_output_total_bytes: `12208032`
- identity_output_total_bytes: `12208032`

## Deltas
- identity_minus_full_archive_bytes: `319`
- identity_minus_full_parsed_member_bytes: `319`
- identity_minus_full_zip_bytes: `176`
- identity_minus_full_contest_archive_bytes_basis: `176`
- identity_minus_full_rate_term_basis: `0.00011719117574950216`

## Blockers
- `no_paired_exact_eval_json`
- `no_contest_cpu_cuda_pair`
- `not_score_authority`

## Reactivation Criteria
- provide both paired contest_auth_eval JSON files for the exact same ZIP sidecars
- keep axis labels adjacent to all score language
- treat full_minus_identity_score <= -decision_delta_s as a full-FiLM win
- do not promote, rank, or kill from this probe without operator review
