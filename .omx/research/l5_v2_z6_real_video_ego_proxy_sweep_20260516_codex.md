# L5 v2 Z6 real-video ego-proxy sweep

- schema: `z6_real_video_ego_proxy_sweep_v1`
- probe_id: `z6_real_video_ego_proxy_sweep`
- lane_id: `lane_time_traveler_l5_z6_l1_scaffold_substrate_build_20260516`
- evidence_grade: `real_video_smoke_proxy_no_scorer`
- verdict: `full_film_proxy_found_real_video_smoke`
- paired_control_initialization: `shared_modules_seed_order_matched_v2`
- best_proxy_id: `random_control`
- semantic_ego_proxy_supported: `False`
- best_identity_minus_full_loss_proxy: `5.304813385009766e-06`
- score_claim: `false`
- promotion_eligible: `false`
- rank_or_kill_eligible: `false`
- ready_for_exact_eval_dispatch: `false`
- ready_for_paid_dispatch: `false`
- paradigm_claim_allowed: `false`

This sweep is a no-scorer real-video smoke proxy. It tests whether any cheap ego proxy makes the Z6 FiLM predictor beat identity before spending on scorer-bearing training or exact eval.

## Rows

### zero

- full_film_proxy_wins: `True`
- identity_minus_full_loss_proxy: `4.678964614868164e-06`
- identity_minus_full_recon: `4.678964614868164e-06`
- identity_minus_full_residual: `-2.9103830456733704e-11`
- full_minus_identity_archive_bytes: `3509`
- full_film_loss_proxy: `0.3175433278083801`
- identity_loss_proxy: `0.317548006772995`
- full_paired_control_initialization: `shared_modules_seed_order_matched_v2`
- identity_paired_control_initialization: `shared_modules_seed_order_matched_v2`

### ramp

- full_film_proxy_wins: `True`
- identity_minus_full_loss_proxy: `4.559755325317383e-06`
- identity_minus_full_recon: `4.559755325317383e-06`
- identity_minus_full_residual: `0.0`
- full_minus_identity_archive_bytes: `3562`
- full_film_loss_proxy: `0.3175434470176697`
- identity_loss_proxy: `0.317548006772995`
- full_paired_control_initialization: `shared_modules_seed_order_matched_v2`
- identity_paired_control_initialization: `shared_modules_seed_order_matched_v2`

### frame_delta

- full_film_proxy_wins: `True`
- identity_minus_full_loss_proxy: `4.76837158203125e-06`
- identity_minus_full_recon: `4.76837158203125e-06`
- identity_minus_full_residual: `0.0`
- full_minus_identity_archive_bytes: `3457`
- full_film_loss_proxy: `0.31754323840141296`
- identity_loss_proxy: `0.317548006772995`
- full_paired_control_initialization: `shared_modules_seed_order_matched_v2`
- identity_paired_control_initialization: `shared_modules_seed_order_matched_v2`

### moment_proxy

- full_film_proxy_wins: `True`
- identity_minus_full_loss_proxy: `4.947185516357422e-06`
- identity_minus_full_recon: `4.947185516357422e-06`
- identity_minus_full_residual: `0.0`
- full_minus_identity_archive_bytes: `3487`
- full_film_loss_proxy: `0.31754305958747864`
- identity_loss_proxy: `0.317548006772995`
- full_paired_control_initialization: `shared_modules_seed_order_matched_v2`
- identity_paired_control_initialization: `shared_modules_seed_order_matched_v2`

### quadrant_delta

- full_film_proxy_wins: `True`
- identity_minus_full_loss_proxy: `4.470348358154297e-06`
- identity_minus_full_recon: `4.470348358154297e-06`
- identity_minus_full_residual: `-2.9103830456733704e-11`
- full_minus_identity_archive_bytes: `3508`
- full_film_loss_proxy: `0.31754353642463684`
- identity_loss_proxy: `0.317548006772995`
- full_paired_control_initialization: `shared_modules_seed_order_matched_v2`
- identity_paired_control_initialization: `shared_modules_seed_order_matched_v2`

### random_control

- full_film_proxy_wins: `True`
- identity_minus_full_loss_proxy: `5.304813385009766e-06`
- identity_minus_full_recon: `5.304813385009766e-06`
- identity_minus_full_residual: `0.0`
- full_minus_identity_archive_bytes: `3505`
- full_film_loss_proxy: `0.31754270195961`
- identity_loss_proxy: `0.317548006772995`
- full_paired_control_initialization: `shared_modules_seed_order_matched_v2`
- identity_paired_control_initialization: `shared_modules_seed_order_matched_v2`

## Blockers
- real_video_smoke_proxy_no_scorer
- no_contest_cpu_cuda_pair
- no_byte_closed_score_anchor
- not_paradigm_claim_authority
- ego_proxy_semantics_not_hard_earned

## Recommended Next Actions
- treat the full-FiLM win as predictor-capacity liveness, not a score or paradigm claim
- run a scorer-bearing or PoseNet-derived ego proxy probe before paid dispatch
- if the best proxy remains zero/random, redesign the ego-conditioning objective before full_main
