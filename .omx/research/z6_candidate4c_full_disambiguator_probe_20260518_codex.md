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

- path: `experiments/results/z6_candidate4c_full_disambiguator_paircap2_codex_20260518T0234Z/0.bin`
- bytes: `194747`
- sha256: `e83f8019690a9a87c1066a5791299dcc018bded9ab97f261db6444ba89b2f05b`
- zip_path: `experiments/results/z6_candidate4c_full_disambiguator_paircap2_codex_20260518T0234Z/archive.zip`
- zip_bytes: `194119`
- zip_sha256: `28b12f7cc136eb369e28df04135cca802fbbdec6eb73b07d22a796f570e11dce`
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

- path: `experiments/results/z6_candidate4c_full_disambiguator_paircap2_codex_20260518T0234Z/0_identity_predictor_disambiguator.bin`
- bytes: `195066`
- sha256: `854a009ae8e9ab2e08c4faa1afcadc92615a40bad8d7586afc84366ba7ea1f11`
- zip_path: `experiments/results/z6_candidate4c_full_disambiguator_paircap2_codex_20260518T0234Z/archive_identity_predictor_disambiguator.zip`
- zip_bytes: `194294`
- zip_sha256: `4298f449c7477704d979f2c6e46467665f0cb4ff4fc09bfa9a59de2a2b77228d`
- zip_members: `['0.bin']`
- zip_member_matches_path_bytes: `True`
- contest_archive_bytes_basis: `194294`
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

## Deltas
- identity_minus_full_archive_bytes: `319`
- identity_minus_full_parsed_member_bytes: `319`
- identity_minus_full_zip_bytes: `175`
- identity_minus_full_contest_archive_bytes_basis: `175`
- identity_minus_full_rate_term_basis: `0.00011652531679637998`

## Blockers
- `no_paired_exact_eval_json`
- `no_contest_cpu_cuda_pair`
- `not_score_authority`

## Reactivation Criteria
- provide both paired contest_auth_eval JSON files for the exact same ZIP sidecars
- keep axis labels adjacent to all score language
- treat full_minus_identity_score <= -decision_delta_s as a full-FiLM win
- do not promote, rank, or kill from this probe without operator review
