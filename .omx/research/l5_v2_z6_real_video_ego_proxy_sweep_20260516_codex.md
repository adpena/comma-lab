# L5 v2 Z6 real-video ego-proxy sweep

- schema: `z6_real_video_ego_proxy_sweep_v1`
- probe_id: `z6_real_video_ego_proxy_sweep`
- lane_id: `lane_time_traveler_l5_z6_l1_scaffold_substrate_build_20260516`
- evidence_grade: `real_video_smoke_proxy_no_scorer`
- verdict: `identity_dominates_all_tested_ego_proxies_real_video_smoke`
- best_proxy_id: `random_control`
- best_identity_minus_full_loss_proxy: `-5.4001808166503906e-05`
- score_claim: `false`
- promotion_eligible: `false`
- rank_or_kill_eligible: `false`
- ready_for_exact_eval_dispatch: `false`
- ready_for_paid_dispatch: `false`
- paradigm_claim_allowed: `false`

This sweep is a no-scorer real-video smoke proxy. It tests whether any cheap ego proxy makes the Z6 FiLM predictor beat identity before spending on scorer-bearing training or exact eval.

## Rows

### zero

- full_film_proxy_wins: `False`
- identity_minus_full_loss_proxy: `-5.510449409484863e-05`
- identity_minus_full_recon: `4.827976226806641e-06`
- identity_minus_full_residual: `-5.993008380755782e-05`
- full_minus_identity_archive_bytes: `3226`
- full_film_loss_proxy: `0.31760311126708984`
- identity_loss_proxy: `0.317548006772995`

### ramp

- full_film_proxy_wins: `False`
- identity_minus_full_loss_proxy: `-5.4329633712768555e-05`
- identity_minus_full_recon: `5.602836608886719e-06`
- identity_minus_full_residual: `-5.993017111904919e-05`
- full_minus_identity_archive_bytes: `3152`
- full_film_loss_proxy: `0.31760233640670776`
- identity_loss_proxy: `0.317548006772995`

### frame_delta

- full_film_proxy_wins: `False`
- identity_minus_full_loss_proxy: `-5.53131103515625e-05`
- identity_minus_full_recon: `4.6193599700927734e-06`
- identity_minus_full_residual: `-5.9930142015218735e-05`
- full_minus_identity_archive_bytes: `3155`
- full_film_loss_proxy: `0.31760331988334656`
- identity_loss_proxy: `0.317548006772995`

### moment_proxy

- full_film_proxy_wins: `False`
- identity_minus_full_loss_proxy: `-5.474686622619629e-05`
- identity_minus_full_recon: `5.185604095458984e-06`
- identity_minus_full_residual: `-5.993008380755782e-05`
- full_minus_identity_archive_bytes: `3165`
- full_film_loss_proxy: `0.3176027536392212`
- identity_loss_proxy: `0.317548006772995`

### quadrant_delta

- full_film_proxy_wins: `False`
- identity_minus_full_loss_proxy: `-5.4776668548583984e-05`
- identity_minus_full_recon: `5.155801773071289e-06`
- identity_minus_full_residual: `-5.9930142015218735e-05`
- full_minus_identity_archive_bytes: `3161`
- full_film_loss_proxy: `0.3176027834415436`
- identity_loss_proxy: `0.317548006772995`

### random_control

- full_film_proxy_wins: `False`
- identity_minus_full_loss_proxy: `-5.4001808166503906e-05`
- identity_minus_full_recon: `5.930662155151367e-06`
- identity_minus_full_residual: `-5.993008380755782e-05`
- full_minus_identity_archive_bytes: `3171`
- full_film_loss_proxy: `0.3176020085811615`
- identity_loss_proxy: `0.317548006772995`

## Blockers
- real_video_smoke_proxy_no_scorer
- no_contest_cpu_cuda_pair
- no_byte_closed_score_anchor
- not_paradigm_claim_authority

## Recommended Next Actions
- do not spend on Z6 full-FiLM until a scorer-bearing or PoseNet-derived ego proxy beats identity in smoke
- try a PoseNet/SegNet-derived ego proxy or redesign predictor objective before paid dispatch
- if no proxy beats identity, retire Z6-v1 FiLM as measured configuration only, not the whole L5 staircase
