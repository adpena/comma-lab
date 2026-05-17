# L5 v2 Z6 identity-predictor disambiguator

- schema: `z6_predictive_coding_vs_identity_disambiguator_v1`
- probe_id: `z6_predictive_coding_vs_identity_disambiguator`
- lane_id: `lane_time_traveler_l5_z6_l1_scaffold_substrate_build_20260516`
- evidence_grade: `smoke_proxy_real_video_pair_no_scorer`
- verdict: `full_film_predictor_proxy_lower_loss`
- paired_control_initialization: `shared_modules_seed_order_matched_v2`
- score_claim: `false`
- promotion_eligible: `false`
- rank_or_kill_eligible: `false`
- ready_for_exact_eval_dispatch: `false`
- ready_for_paid_dispatch: `false`
- paradigm_claim_allowed: `false`

This report is a Z6-specific probe surface. It can route the next engineering action, but it is not contest score evidence.

## Source Stats

### full_film_predictor

- path: `experiments/results/time_traveler_l5_z6/disambiguator_full_film_real_video_smoke_20260516_codex/stats.json`
- sha256: `59ddb7ec061ff75d5f6aade49de5e8fca641b45a6531a7112ac7964f76f0cb26`
- final_loss_proxy: `0.31754323840141296`
- final_recon: `0.31718677282333374`
- final_residual: `0.00035645384923554957`
- archive_bytes: `43025`
- paired_control_initialization: `shared_modules_seed_order_matched_v2`
- smoke_target_mode: `real-video`
- smoke_ego_motion_mode: `real-video`
- ego_motion_nonzero_fraction: `1.0`

### identity_predictor

- path: `experiments/results/time_traveler_l5_z6/disambiguator_identity_real_video_smoke_20260516_codex/stats.json`
- sha256: `424b14341dc94ec2345fc6a7aa196fd3c2cdb290ccb86ca2f968bf05bd57ef8d`
- final_loss_proxy: `0.317548006772995`
- final_recon: `0.31719154119491577`
- final_residual: `0.00035645384923554957`
- archive_bytes: `39568`
- paired_control_initialization: `shared_modules_seed_order_matched_v2`
- smoke_target_mode: `real-video`
- smoke_ego_motion_mode: `real-video`
- ego_motion_nonzero_fraction: `1.0`

## Deltas
- identity_minus_full_loss_proxy: `4.76837158203125e-06`
- identity_minus_full_recon: `4.76837158203125e-06`
- identity_minus_full_residual: `0.0`
- full_minus_identity_archive_bytes: `3457`

## Blockers
- `smoke_proxy_real_video_no_scorer`
- `no_contest_cpu_cuda_pair`
- `no_byte_closed_score_anchor`
- `not_paradigm_claim_authority`

## Reactivation Criteria
- if full-FiLM wins proxy, implement real-video full_main and run paired smoke on contest video
- if identity wins proxy, keep Z6 predictive-coding claim blocked and diagnose predictor/curriculum
- only promote/rank/kill after byte-closed paired contest CPU/CUDA exact eval
