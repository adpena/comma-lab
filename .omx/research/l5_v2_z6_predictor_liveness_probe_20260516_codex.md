# L5 v2 Z6 predictor liveness probe

- schema: `z6_predictor_liveness_probe_v1`
- probe_id: `z6_predictor_liveness_and_ego_motion_cargo_cult_probe`
- lane_id: `lane_time_traveler_l5_z6_l1_scaffold_substrate_build_20260516`
- evidence_grade: `synthetic_liveness_probe_no_scorer`
- verdict: `z6_predictor_live_smoke_film_conditioning_cargo_cult`
- score_claim: `false`
- promotion_eligible: `false`
- rank_or_kill_eligible: `false`
- ready_for_exact_eval_dispatch: `false`
- ready_for_paid_dispatch: `false`
- paradigm_claim_allowed: `false`

This is an engineering liveness/cargo-cult probe. It explains whether the Z6 identity-vs-full smoke result exercised the predictor path; it does not claim score movement.

## Rows

### full_film_zero_ego

- verdict: `predictor_gradient_live_but_film_conditioning_unexercised`
- predictor_param_count: `1632`
- predictor_gradient_l2: `6.337604718334034e-06`
- ego_motion_nonzero_fraction: `0.0`
- predictor_output_delta_vs_zero_ego_l2: `0.0`
- film_conditioning_exercised: `False`

### full_film_ramp_ego

- verdict: `predictor_and_film_conditioning_live_proxy_only`
- predictor_param_count: `1632`
- predictor_gradient_l2: `6.656936262657043e-06`
- ego_motion_nonzero_fraction: `1.0`
- predictor_output_delta_vs_zero_ego_l2: `0.09052672982215881`
- film_conditioning_exercised: `True`

### identity_predictor_control

- verdict: `identity_control_no_predictor_params`
- predictor_param_count: `0`
- predictor_gradient_l2: `0.0`
- ego_motion_nonzero_fraction: `0.0`
- predictor_output_delta_vs_zero_ego_l2: `None`
- film_conditioning_exercised: `False`

## Findings
- full FiLM predictor receives nonzero gradients in the smoke graph
- current zero-ego smoke does not exercise FiLM conditioning
- ramp ego-motion control changes predictor output and confirms the conditioning path is live
- identity-vs-full smoke result is not a fair falsification of ego-motion predictive coding

## Blockers
- current_z6_smoke_uses_zero_ego_motion_buffer
- current_z6_smoke_uses_synthetic_targets_without_real_temporal_pose_signal
- no_scorer_load
- no_contest_cpu_cuda_pair
- not_score_or_paradigm_authority

## Recommended Next Actions
- populate ego_motion_buffer from real-video PoseNet/proxy features before rerunning identity disambiguator
- rerun Z6 identity disambiguator on real-video paired smoke after ego-motion population
- keep full_main council-gated until predictor-vs-identity wins on a real signal axis
