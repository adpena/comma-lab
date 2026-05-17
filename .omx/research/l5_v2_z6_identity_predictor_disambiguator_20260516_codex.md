# L5 v2 Z6 identity-predictor disambiguator

- schema: `z6_predictive_coding_vs_identity_disambiguator_v1`
- probe_id: `z6_predictive_coding_vs_identity_disambiguator`
- lane_id: `lane_time_traveler_l5_z6_l1_scaffold_substrate_build_20260516`
- evidence_grade: `smoke_proxy_synthetic_pair`
- verdict: `identity_predictor_proxy_lower_loss`
- score_claim: `false`
- promotion_eligible: `false`
- rank_or_kill_eligible: `false`
- ready_for_exact_eval_dispatch: `false`
- ready_for_paid_dispatch: `false`
- paradigm_claim_allowed: `false`

This report is a Z6-specific probe surface. It can route the next engineering action, but it is not contest score evidence.

## Source Stats

### full_film_predictor

- path: `experiments/results/time_traveler_l5_z6/disambiguator_full_film_smoke_20260516_codex/stats.json`
- sha256: `ca76380008a3256582ce3ee8f58c56cc7f6a82df3c7d8b2c4818ccd36ce10518`
- final_loss_proxy: `0.16798947751522064`
- final_recon: `0.16757309436798096`
- final_residual: `0.000416384544223547`
- archive_bytes: `42643`

### identity_predictor

- path: `experiments/results/time_traveler_l5_z6/disambiguator_identity_smoke_20260516_codex/stats.json`
- sha256: `235eed2236032ecfabad395b236e58def06d7195b6e41120a1556ee3014ce11b`
- final_loss_proxy: `0.16791336238384247`
- final_recon: `0.16755691170692444`
- final_residual: `0.0003564539656508714`
- archive_bytes: `39480`

## Deltas
- identity_minus_full_loss_proxy: `-7.611513137817383e-05`
- identity_minus_full_recon: `-1.6182661056518555e-05`
- identity_minus_full_residual: `-5.9930578572675586e-05`
- full_minus_identity_archive_bytes: `3163`

## Blockers
- `smoke_proxy_synthetic_no_scorer`
- `no_contest_cpu_cuda_pair`
- `no_byte_closed_score_anchor`
- `not_paradigm_claim_authority`

## Reactivation Criteria
- if full-FiLM wins proxy, implement real-video full_main and run paired smoke on contest video
- if identity wins proxy, keep Z6 predictive-coding claim blocked and diagnose predictor/curriculum
- only promote/rank/kill after byte-closed paired contest CPU/CUDA exact eval
