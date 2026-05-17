# L5 v2 asymptotic candidate surface

- schema: `l5_v2_asymptotic_pursuit_candidates_v1`
- campaign_id: `campaign_time_traveler_l5_v2_staircase_20260516`
- candidate_count: `3`
- score_claim: `false`
- promotion_eligible: `false`
- rank_or_kill_eligible: `false`
- ready_for_exact_eval_dispatch: `false`
- ready_for_paid_dispatch: `false`

This is a planning and no-signal-loss surface. It records whether asymptotic L5-v2 candidates have their first executable artifacts, but it does not authorize score claims, rank changes, paid dispatch, or promotion.

## Candidates

### z6_z7_z8_predictive_coding_world_models

- title: Z6/Z7/Z8 predictive-coding world-model staircase
- lane_id: `lane_time_traveler_l5_z6_z7_z8_predictive_coding_world_models_scoping_design_20260516`
- local_ledger_present: `True`
- lane_registry_registered: `True`
- expected_first_artifacts_all_present: `True`
- l1_scaffold_present: `True`
- recommended_next_action_status: `completed_or_superseded`
- effective_recommended_next_action_id: `completed_or_superseded:build_z6_l1_scaffold_first`
- ready_for_l1_build: `False`
- ready_for_l1_scaffold_dispatch: `False`
- blockers: `['requires_z6_l1_scaffold_before_paid_dispatch', 'requires_identity_predictor_disambiguator_result_before_paradigm_claim', 'requires_paired_cpu_cuda_anchor_before_score_or_rank_authority']`
- l1_build_blockers: `['l1_scaffold_present_next_action_completed_or_superseded']`

Expected first artifacts:
- `src/tac/substrates/time_traveler_l5_z6/` present=`True`
- `experiments/train_substrate_time_traveler_l5_z6.py` present=`True`
- `tools/probe_z6_predictive_coding_vs_identity_disambiguator.py` present=`True`
- `.omx/operator_authorize_recipes/substrate_time_traveler_l5_z6_modal_t4_dispatch.yaml` present=`True`

### rudin_floor_interpretable_ml_substrate

- title: Rudin floor interpretable-ML compositional decoder
- lane_id: `lane_rudin_floor_interpretable_ml_substrate_scoping_design_20260516`
- local_ledger_present: `True`
- lane_registry_registered: `True`
- expected_first_artifacts_all_present: `True`
- l1_scaffold_present: `True`
- recommended_next_action_status: `completed_or_superseded`
- effective_recommended_next_action_id: `completed_or_superseded:ratify_and_build_rudin_k8_l1_scaffold`
- ready_for_l1_build: `False`
- ready_for_l1_scaffold_dispatch: `False`
- blockers: `['requires_t3_ratification_before_l1_scaffold_dispatch', 'requires_dykstra_feasibility_intersection_before_paid_smoke', 'requires_byte_mutation_proof_before_score_or_rank_authority']`
- l1_build_blockers: `['l1_scaffold_present_next_action_completed_or_superseded']`

Expected first artifacts:
- `src/tac/substrates/rudin_floor_interpretable_ml/` present=`True`
- `experiments/train_substrate_rudin_floor_interpretable_ml.py` present=`True`
- `.omx/operator_authorize_recipes/substrate_rudin_floor_interpretable_ml_modal_t4_dispatch.yaml` present=`True`

### tishby_ib_pure_substrate

- title: Tishby IB-pure primary Lagrangian substrate
- lane_id: `lane_tishby_ib_pure_substrate_scoping_design_20260516`
- local_ledger_present: `True`
- lane_registry_registered: `True`
- expected_first_artifacts_all_present: `True`
- l1_scaffold_present: `True`
- recommended_next_action_status: `completed_or_superseded`
- effective_recommended_next_action_id: `completed_or_superseded:run_d4_probe_and_build_variational_ib_tractability_tool`
- ready_for_l1_build: `False`
- ready_for_l1_scaffold_dispatch: `False`
- blockers: `['requires_d4_probe_verdict_before_tishby_scaffold', 'requires_variational_ib_tractability_before_path_vib_or_mine', 'requires_paired_smoke_vs_atw_v2_before_asymptotic_claim']`
- l1_build_blockers: `['l1_scaffold_present_next_action_completed_or_superseded']`

Expected first artifacts:
- `.omx/state/h_latent_given_scorer_class_tishby_ib_pure.json` present=`True`
- `tools/check_variational_ib_tractability.py` present=`True`
- `src/tac/substrates/tishby_ib_pure/` present=`True`
