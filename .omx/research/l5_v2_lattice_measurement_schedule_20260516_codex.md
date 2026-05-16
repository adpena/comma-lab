# L5 v2 lattice measurement schedule

- schema: `l5_v2_lattice_measurement_schedule_v1`
- active_rule_id: `fill_missing_c1_z5_tt5l_probe_observations`
- active_measurement_ids: `['measure_c1_world_model_foveation_paired_exact', 'measure_z5_predictive_coding_paired_exact', 'measure_tt5l_autonomy_paired_exact']`
- sideinfo_effect_curve_valid: `False`
- sideinfo_effect_curve_blockers: `['tt5l_sideinfo_effect_curve_predicate_not_passed', 'tt5l_sideinfo_effect_curve_effect_blocked:trained_not_best_or_tied:contest_cpu', 'tt5l_sideinfo_effect_curve_effect_blocked:trained_not_best_or_tied:contest_cuda', 'tt5l_sideinfo_effect_curve_trained_not_best_or_tied:contest_cpu', 'tt5l_sideinfo_effect_curve_trained_not_best_or_tied:contest_cuda', 'tt5l_sideinfo_effect_curve_cells_missing:contest_cpu/ablated,contest_cpu/random_lsb,contest_cpu/shuffled,contest_cpu/trained,contest_cpu/zero,contest_cuda/ablated,contest_cuda/random_lsb,contest_cuda/shuffled,contest_cuda/zero']`
- score_claim: `false`
- promotion_eligible: `false`
- ready_for_exact_eval_dispatch: `false`

## Measurements

### measure_c1_world_model_foveation_paired_exact

- candidate_id: `c1_world_model_foveation`
- purpose: fill C1 paired CPU/CUDA exact probe observation
- required_axes: `['contest_cpu', 'contest_cuda']`
- estimated_cost_usd: `2.0`
- evidence authority: planning-only until paired exact artifacts land

### measure_z5_predictive_coding_paired_exact

- candidate_id: `z5_predictive_coding_world_model`
- purpose: fill Z5 paired CPU/CUDA exact probe observation
- required_axes: `['contest_cpu', 'contest_cuda']`
- estimated_cost_usd: `5.0`
- evidence authority: planning-only until paired exact artifacts land

### measure_tt5l_autonomy_paired_exact

- candidate_id: `time_traveler_l5_autonomy`
- purpose: fill TT5L paired CPU/CUDA exact probe observation
- required_axes: `['contest_cpu', 'contest_cuda']`
- estimated_cost_usd: `7.5`
- evidence authority: planning-only until paired exact artifacts land

### measure_tt5l_sideinfo_effect_curve

- candidate_id: `time_traveler_l5_autonomy`
- purpose: separate side-info consumption from causal usefulness with paired CPU/CUDA zero, random-LSB, shuffled, trained, and ablated side-info
- required_axes: `['contest_cpu', 'contest_cuda']`
- estimated_cost_usd: `1.0`
- evidence authority: planning-only until paired exact artifacts land
- sideinfo_effect_curve_dispatch_variants: `['zero', 'random_lsb', 'shuffled', 'trained', 'ablated']`
- sideinfo_effect_curve_required_cell_count: `10`

### prepare_l5_v2_paired_anchor_packet

- candidate_id: `time_traveler_l5_autonomy`
- purpose: materialize paired-axis anchor packet only after C1/Z5/TT5L probe observations and side-info effect curve are present
- required_axes: `['contest_cpu', 'contest_cuda']`
- estimated_cost_usd: `0.0`
- evidence authority: planning-only until paired exact artifacts land
