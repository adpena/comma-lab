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

- path: `experiments/results/time_traveler_l5_z6/disambiguator_full_film_ramp_smoke_20260516_codex/stats.json`
- sha256: `e57701256967d108c1d387c7dd61222d20f14ae5c75ff51858f78b1b3bb4f58f`
- final_loss_proxy: `0.1679897904396057`
- final_recon: `0.16757340729236603`
- final_residual: `0.0004163846024312079`
- archive_bytes: `42682`

### identity_predictor

- path: `experiments/results/time_traveler_l5_z6/disambiguator_identity_ramp_smoke_20260516_codex/stats.json`
- sha256: `3b7c306b142595e241da84a9a8ede4cd011d2868b893fbf29e903d099751b340`
- final_loss_proxy: `0.16791336238384247`
- final_recon: `0.16755691170692444`
- final_residual: `0.0003564539656508714`
- archive_bytes: `39529`

## Deltas
- identity_minus_full_loss_proxy: `-7.642805576324463e-05`
- identity_minus_full_recon: `-1.6495585441589355e-05`
- identity_minus_full_residual: `-5.99306367803365e-05`
- full_minus_identity_archive_bytes: `3153`

## Blockers
- `smoke_proxy_synthetic_no_scorer`
- `no_contest_cpu_cuda_pair`
- `no_byte_closed_score_anchor`
- `not_paradigm_claim_authority`

## Reactivation Criteria
- if full-FiLM wins proxy, implement real-video full_main and run paired smoke on contest video
- if identity wins proxy, keep Z6 predictive-coding claim blocked and diagnose predictor/curriculum
- only promote/rank/kill after byte-closed paired contest CPU/CUDA exact eval
