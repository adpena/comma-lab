# L5 v2 lattice measurement schedule

- schema: `l5_v2_lattice_measurement_schedule_v1`
- active_rule_id: `fill_missing_c1_z5_tt5l_probe_observations`
- active_measurement_ids: `['measure_c1_world_model_foveation_paired_exact', 'measure_z5_predictive_coding_paired_exact', 'measure_tt5l_autonomy_paired_exact']`
- score_claim: `false`
- promotion_eligible: `false`
- ready_for_exact_eval_dispatch: `false`

## Measurements

### measure_c1_world_model_foveation_paired_exact

- candidate_id: `c1_world_model_foveation`
- purpose: fill C1 paired CPU/CUDA exact probe observation
- estimated_cost_usd: `2.0`
- evidence authority: planning-only until paired exact artifacts land

### measure_z5_predictive_coding_paired_exact

- candidate_id: `z5_predictive_coding_world_model`
- purpose: fill Z5 paired CPU/CUDA exact probe observation
- estimated_cost_usd: `5.0`
- evidence authority: planning-only until paired exact artifacts land

### measure_tt5l_autonomy_paired_exact

- candidate_id: `time_traveler_l5_autonomy`
- purpose: fill TT5L paired CPU/CUDA exact probe observation
- estimated_cost_usd: `7.5`
- evidence authority: planning-only until paired exact artifacts land

### measure_tt5l_sideinfo_effect_curve

- candidate_id: `time_traveler_l5_autonomy`
- purpose: separate side-info consumption from causal usefulness via zero, random-LSB, shuffled, trained, and ablated side-info
- estimated_cost_usd: `1.0`
- evidence authority: planning-only until paired exact artifacts land

### prepare_l5_v2_paired_anchor_packet

- candidate_id: `time_traveler_l5_autonomy`
- purpose: materialize paired-axis anchor packet only after C1/Z5/TT5L probe observations and side-info effect curve are present
- estimated_cost_usd: `0.0`
- evidence authority: planning-only until paired exact artifacts land
