# L5 v2 architecture lock packet

- schema: `l5_v2_architecture_lock_packet_v1`
- subject_id: `time_traveler_l5_autonomy`
- lane_id: `lane_time_traveler_l5_autonomy_substrate_20260513`
- architecture_lock_allowed: `False`
- readiness_architecture_lock_allowed: `False`
- next_action: `resolve_l5_v2_tt5l_modal_provider_blocker_or_dispatch_alternate_provider`
- score_claim: `false`
- promotion_eligible: `false`
- ready_for_exact_eval_dispatch: `false`

## Required Checks
- `all_gate_evidence_valid`: `False`
- `dykstra_score_axis_sanity_valid`: `True`
- `move_level_feasibility_artifact_valid`: `True`
- `sideinfo_gate_evidence_valid`: `True`
- `probe_gate_evidence_valid`: `False`
- `paired_axis_plan_evidence_valid`: `True`
- `sideinfo_effect_curve_artifact_valid`: `False`
- `first_anchor_timing_smoke_artifact_valid`: `True`
- `anchor_pair_evidence_valid`: `True`

## Prediction Band

- rank_reward_allowed: `False`
- dispatch_planning_allowed: `True`
- verdict_blockers: `['prediction_band_baseline_missing', 'prediction_band_baseline_custody_missing', 'prediction_band_baseline_artifact_missing', 'prediction_band_empirical_anchor_missing']`
- diagnostic_anchor_pair_exists: `True`
- diagnostic_anchor_pair_valid: `True`
- diagnostic_anchor_classification: `paired_exact_measured_config_failure_non_promotional_anchor`
- diagnostic_anchor_axes: `['contest_cpu', 'contest_cuda']`
- diagnostic_anchor_scores: `{'contest_cpu': 3.8987840060549908, 'contest_cuda': 3.9007398365396795}`
- diagnostic_anchor_preserved_but_not_rankable: `True`

## Lightning Paired-Axis Dry-Run Plan

- artifact_path: `.omx/research/l5_v2_tt5l_sideinfo_effect_curve_lightning_paired_axis_plan_20260517_codex.json`
- artifact_valid: `True`
- source_commit: `5447f99ff56617946aaf711c365642163b05671a`
- source_relevant_paths_match: `True`
- source_relevant_diff_paths: `[]`
- source_custody_current_for_execution: `True`
- cells: `10`/`10`
- axes: `['contest_cpu', 'contest_cuda']`
- all_cells_dry_run_ready: `True`
- all_cells_dry_run_structurally_valid: `True`
- execution_ready: `False`
- execution_blockers: `['l5_v2_tt5l_lightning_paired_axis_plan_dry_run_only_no_provider_job_launched', 'l5_v2_tt5l_lightning_paired_axis_plan_blocked:dry_run_only_no_provider_job_launched', 'l5_v2_tt5l_lightning_paired_axis_plan_blocked:requires_lightning_identity_and_workspace_preflight_before_submit', 'l5_v2_tt5l_lightning_paired_axis_plan_blocked:requires_source_manifest_staged_to_lightning_workspace_before_submit', 'l5_v2_tt5l_lightning_paired_axis_plan_blocked:requires_per_axis_lane_claim_before_non_dry_run_submit', 'l5_v2_tt5l_lightning_paired_axis_plan_blocked:requires_harvested_contest_cpu_and_contest_cuda_cells_before_sideinfo_effect_claim', 'l5_v2_tt5l_lightning_paired_axis_plan_blocked:score_claim_forbidden_until_effect_curve_artifact_passes']`
- score_claim: `false`
- promotion_eligible: `false`

## Sideinfo Effect Curve

- artifact_path: `.omx/research/l5_v2_tt5l_sideinfo_effect_curve_20260516_codex.json`
- artifact_valid: `False`
- measurement_id: `measure_tt5l_sideinfo_effect_curve`
- predicate_passed: `False`
- observed_cell_count: `1`
- missing_cells: `['contest_cpu/ablated', 'contest_cpu/random_lsb', 'contest_cpu/shuffled', 'contest_cpu/trained', 'contest_cpu/zero', 'contest_cuda/ablated', 'contest_cuda/random_lsb', 'contest_cuda/shuffled', 'contest_cuda/zero']`
- effect_blockers: `['trained_not_best_or_tied:contest_cpu', 'trained_not_best_or_tied:contest_cuda']`
- axis_effects: `{'contest_cpu': {'best_control_score': None, 'best_control_variant': '', 'delta_vs_best_control': None, 'trained_beats_or_ties_best_control': False, 'trained_score': None}, 'contest_cuda': {'best_control_score': None, 'best_control_variant': '', 'delta_vs_best_control': None, 'trained_beats_or_ties_best_control': False, 'trained_score': 3.9007398365396795}}`
- observed_cell `contest_cuda/trained`: score=`3.9007398365396795`, sideinfo_nonzero_fraction=`0.0`, sideinfo_nonzero_values=`0`/`27000`, archive_sha256=`2b05b7351b690b0b2251ddc620d80dd9a1833051cfa07e679106d00fbc70024a`, runtime_content_tree_sha256=`105fc0834cfb8a54b8f46edb81a030d076369c3062f3066c1800602f9d6035f5`
- score_claim: `false`
- promotion_eligible: `false`

## TT5L Sideinfo Dispatch Plan

- artifact_path: `.omx/research/l5_v2_tt5l_sideinfo_effect_curve_dispatch_plan_20260517_codex.json`
- artifact_valid: `True`
- plan_id: `l5_v2_tt5l_sideinfo_effect_curve_dispatch_0902d4c91d23e972`
- work_units: `5`/`5`
- required_variants: `['zero', 'random_lsb', 'shuffled', 'trained', 'ablated']`
- ready_for_operator_dispatch: `True`
- ready_for_provider_dispatch: `false`
- dispatch_attempted: `false`
- blockers: `[]`
- work_unit `zero`: ready=`True`, archive_sha256=`b444cc91f102c9807a865ed59f182ca5c83f3239a49ec2aa400b497d7dea37a3`, archive_bytes=`34373`, pair_group_id=`pair_l5_v2_tt5l_sideinfo_effect_curve_zero_b444cc91f102`
- work_unit `random_lsb`: ready=`True`, archive_sha256=`ccce77aaf1907d6e70d8cba498261708b241ac7d78a9bf22978aa459cb6b7fd1`, archive_bytes=`38681`, pair_group_id=`pair_l5_v2_tt5l_sideinfo_effect_curve_random_lsb_ccce77aaf190`
- work_unit `shuffled`: ready=`True`, archive_sha256=`c235e5cb91f4122c3bb642354b424dd21f8b37cc0d2ac7b3e03c0b84dcc49bc3`, archive_bytes=`43284`, pair_group_id=`pair_l5_v2_tt5l_sideinfo_effect_curve_shuffled_c235e5cb91f4`
- work_unit `trained`: ready=`True`, archive_sha256=`f08299c5e77908eeeb82cc9948e530fd5a790894902a726ebd8a258596b4bf1a`, archive_bytes=`43323`, pair_group_id=`pair_l5_v2_tt5l_sideinfo_effect_curve_trained_f08299c5e779`
- work_unit `ablated`: ready=`True`, archive_sha256=`ec343265899859495fca1d2874c1b85d211210ea13a2de5bb7e52c9816ba6b39`, archive_bytes=`42419`, pair_group_id=`pair_l5_v2_tt5l_sideinfo_effect_curve_ablated_ec3432658998`
- score_claim: `false`
- promotion_eligible: `false`

## Probe Gate Artifact

- artifact_path: `.omx/research/l5_v2_probe_gate_artifact_20260516_codex.json`
- artifact_exists: `True`
- artifact_valid: `False`
- architecture_lock_allowed: `False`
- selected_candidate_id: `None`
- verdict_blockers: `['l5_v2_probe_required_candidate_ineligible:c1_world_model_foveation', 'l5_v2_probe_required_candidate_ineligible:z5_predictive_coding_world_model', 'l5_v2_probe_required_candidate_ineligible:time_traveler_l5_autonomy', 'l5_v2_probe_no_eligible_candidate']`
- candidate `c1_world_model_foveation`: eligible=`False`, axes=`[]`, blockers=`['l5_v2_probe_artifact_path_missing', 'l5_v2_probe_artifact_sha_invalid', 'l5_v2_probe_predicate_failed', 'l5_v2_probe_paired_exact_axes_missing', 'l5_v2_probe_byte_closed_archive_missing', 'l5_v2_probe_sideinfo_consumption_missing', 'l5_v2_probe_archive_sha_invalid', 'l5_v2_probe_runtime_tree_sha_invalid', 'l5_v2_probe_axis_evidence_missing:contest_cpu', 'l5_v2_probe_axis_evidence_missing:contest_cuda']`
- candidate `z5_predictive_coding_world_model`: eligible=`False`, axes=`[]`, blockers=`['l5_v2_probe_artifact_path_missing', 'l5_v2_probe_artifact_sha_invalid', 'l5_v2_probe_predicate_failed', 'l5_v2_probe_paired_exact_axes_missing', 'l5_v2_probe_byte_closed_archive_missing', 'l5_v2_probe_sideinfo_consumption_missing', 'l5_v2_probe_archive_sha_invalid', 'l5_v2_probe_runtime_tree_sha_invalid', 'l5_v2_probe_axis_evidence_missing:contest_cpu', 'l5_v2_probe_axis_evidence_missing:contest_cuda']`
- candidate `time_traveler_l5_autonomy`: eligible=`False`, axes=`['contest_cpu', 'contest_cuda']`, blockers=`['l5_v2_probe_predicate_failed', 'l5_v2_probe_sideinfo_consumption_missing', 'l5_v2_probe_axis_score_delta_missing:contest_cpu', 'l5_v2_probe_axis_score_delta_missing:contest_cuda']`

## First Anchor Timing Smoke

- artifact_path: `.omx/research/l5_v2_tt5l_first_anchor_timing_smoke_20260517_codex.json`
- artifact_valid: `True`
- provider: `modal`
- hardware: `linux_x86_64_cpu + Tesla T4 CUDA paired exact auth eval`
- elapsed_seconds: `296.600486271`
- seconds_per_candidate: `296.600486271`
- axis_timings: `{'contest_cpu': {'contest_auth_eval_artifact_path': 'experiments/results/l5_v2_probe/measure_tt5l_autonomy_paired_exact/modal_auth_eval_cpu/l5_v2_measure_tt5l_autonomy_paired_exact_paired_measurement_cpu/contest_auth_eval.json', 'contest_auth_eval_artifact_sha256': '2cba8aff95751754dc378888b8fc1cdbd6a0cf4d1e7b5c91e527a2efbd5b01af', 'contest_auth_eval_elapsed_seconds': 293.79739159300004, 'evaluate_elapsed_seconds': 194.805340313, 'inflate_elapsed_seconds': 95.58252872700001, 'modal_elapsed_seconds': 296.600486271, 'score_axis': 'contest_cpu'}, 'contest_cuda': {'contest_auth_eval_artifact_path': 'experiments/results/l5_v2_probe/measure_tt5l_autonomy_paired_exact/modal_auth_eval/l5_v2_measure_tt5l_autonomy_paired_exact_paired_measurement_cuda/contest_auth_eval.json', 'contest_auth_eval_artifact_sha256': '300fedc535226513497596e5e89b9eb18f8ccd48815a256fee307b618d8d82be', 'contest_auth_eval_elapsed_seconds': 53.279486186, 'evaluate_elapsed_seconds': 28.000436932999996, 'inflate_elapsed_seconds': 14.870239315000001, 'modal_elapsed_seconds': 56.412989613, 'score_axis': 'contest_cuda'}}`
- score_claim: `false`
- promotion_eligible: `false`

## Materialized TT5L Provider Routing

- work_unit_artifact_valid: `True`
- archive_sha256: `ccce77aaf1907d6e70d8cba498261708b241ac7d78a9bf22978aa459cb6b7fd1`
- provider_blocker_active: `True`
- provider_blocker_failure_class: `modal_workspace_billing_cycle_spend_limit_reached`
- modal_execute_suppressed_until_blocker_resolved: `True`
- alternate_provider: `lightning`
- alternate_artifact_valid: `True`
- lightning_source_manifest_probe_current: `True`
- lightning_execution_ready: `False`
- lightning_execution_blockers: `['missing_lightning_ssh_target', 'missing_lightning_teamspace', 'machine_inventory_not_checked', 'source_manifest_not_staged', 'remote_cuda_runtime_not_probed']`

## Blockers
- `requires_all_l5_v2_gate_evidence_valid`
- `requires_c1_z5_tt5l_probe_gate_evidence`
- `requires_paired_cpu_cuda_sideinfo_effect_curve`

## Authority

lock/no-lock planning packet only; no score, rank, promotion, or exact-dispatch authority
