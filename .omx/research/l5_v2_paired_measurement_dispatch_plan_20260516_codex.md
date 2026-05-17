# L5 v2 paired measurement dispatch plan

- schema: `l5_v2_paired_measurement_dispatch_plan_v1`
- plan_id: `l5_v2_paired_measurement_dispatch_d245f341e39a5c9b`
- active_rule_id: `fill_missing_c1_z5_tt5l_probe_observations`
- work_unit_count: `3`
- ready_work_unit_count: `0`
- planning_only: `true`
- score_claim: `false`
- score_claim_valid: `false`
- promotion_eligible: `false`
- ready_for_exact_eval_dispatch: `false`
- rank_or_kill_eligible: `false`
- dispatch_attempted: `false`
- adjudication_required: `true`
- blockers: `['l5_v2_probe_artifact_path_missing', 'l5_v2_probe_artifact_sha_invalid', 'l5_v2_probe_predicate_failed', 'l5_v2_probe_paired_exact_axes_missing', 'l5_v2_probe_byte_closed_archive_missing', 'l5_v2_probe_sideinfo_consumption_missing', 'l5_v2_probe_archive_sha_invalid', 'l5_v2_probe_runtime_tree_sha_invalid', 'l5_v2_probe_axis_evidence_missing:contest_cpu', 'l5_v2_probe_axis_evidence_missing:contest_cuda', 'requires_byte_closed_archive_path', 'requires_archive_sha256', 'requires_submission_dir_or_inflate_runtime', 'requires_operator_execute_flag', 'l5_v2_probe_axis_score_delta_missing:contest_cpu', 'l5_v2_probe_axis_score_delta_missing:contest_cuda']`

## Work Units

### measure_c1_world_model_foveation_paired_exact

- work_unit_id: `measure_c1_world_model_foveation_paired_exact`
- source_measurement_id: `measure_c1_world_model_foveation_paired_exact`
- candidate_id: `c1_world_model_foveation`
- sideinfo_variant: ``
- sideinfo_required_cells: `[]`
- lane_id: `lane_l5_v2_measure_c1_world_model_foveation_paired_exact`
- lanes: `{'contest_cuda': 'lane_l5_v2_measure_c1_world_model_foveation_paired_exact_contest_cuda', 'contest_cpu': 'lane_l5_v2_measure_c1_world_model_foveation_paired_exact_contest_cpu'}`
- pair_group_id: `pair_l5_v2_measure_c1_world_model_foveation_paired_exact_cpu_cuda`
- required_axes: `['contest_cpu', 'contest_cuda']`
- paired_dispatch_tool: `tools/dispatch_modal_paired_auth_eval.py`
- ready_for_operator_dispatch: `False`
- ready_for_provider_dispatch: `False`
- dispatch_command_executable: `False`
- claim_lifecycle_owner: `tools/dispatch_modal_paired_auth_eval.py and the per-axis Modal auth-eval wrappers`
- measurement_blockers_to_close: `['l5_v2_probe_artifact_path_missing', 'l5_v2_probe_artifact_sha_invalid', 'l5_v2_probe_predicate_failed', 'l5_v2_probe_paired_exact_axes_missing', 'l5_v2_probe_byte_closed_archive_missing', 'l5_v2_probe_sideinfo_consumption_missing', 'l5_v2_probe_archive_sha_invalid', 'l5_v2_probe_runtime_tree_sha_invalid', 'l5_v2_probe_axis_evidence_missing:contest_cpu', 'l5_v2_probe_axis_evidence_missing:contest_cuda']`
- dispatch_blockers: `['requires_byte_closed_archive_path', 'requires_archive_sha256', 'requires_submission_dir_or_inflate_runtime', 'requires_operator_execute_flag']`
- readiness_blockers: `['l5_v2_probe_artifact_path_missing', 'l5_v2_probe_artifact_sha_invalid', 'l5_v2_probe_predicate_failed', 'l5_v2_probe_paired_exact_axes_missing', 'l5_v2_probe_byte_closed_archive_missing', 'l5_v2_probe_sideinfo_consumption_missing', 'l5_v2_probe_archive_sha_invalid', 'l5_v2_probe_runtime_tree_sha_invalid', 'l5_v2_probe_axis_evidence_missing:contest_cpu', 'l5_v2_probe_axis_evidence_missing:contest_cuda', 'requires_byte_closed_archive_path', 'requires_archive_sha256', 'requires_submission_dir_or_inflate_runtime', 'requires_operator_execute_flag']`
- dispatch_command: `.venv/bin/python tools/dispatch_modal_paired_auth_eval.py --archive FILL_ARCHIVE_ZIP --submission-dir FILL_SUBMISSION_DIR --inflate-sh inflate.sh --label l5_v2_c1_world_model_foveation --expected-archive-sha256 FILL_ARCHIVE_SHA256 --run-id l5_v2_measure_c1_world_model_foveation_paired_exact_paired_measurement --pair-group-id pair_l5_v2_measure_c1_world_model_foveation_paired_exact_cpu_cuda --lane-id-base lane_l5_v2_measure_c1_world_model_foveation_paired_exact --output-root experiments/results/l5_v2_probe/measure_c1_world_model_foveation_paired_exact --modal-bin .venv/bin/modal --gpu T4 --claim-agent codex:l5_v2_paired_measurement_dispatch --claim-notes l5_v2_paired_measurement:pair_l5_v2_measure_c1_world_model_foveation_paired_exact_cpu_cuda --expected-runtime-tree-sha256 auto --skip-axis-if-promotable-anchor-exists`

### measure_z5_predictive_coding_paired_exact

- work_unit_id: `measure_z5_predictive_coding_paired_exact`
- source_measurement_id: `measure_z5_predictive_coding_paired_exact`
- candidate_id: `z5_predictive_coding_world_model`
- sideinfo_variant: ``
- sideinfo_required_cells: `[]`
- lane_id: `lane_l5_v2_measure_z5_predictive_coding_paired_exact`
- lanes: `{'contest_cuda': 'lane_l5_v2_measure_z5_predictive_coding_paired_exact_contest_cuda', 'contest_cpu': 'lane_l5_v2_measure_z5_predictive_coding_paired_exact_contest_cpu'}`
- pair_group_id: `pair_l5_v2_measure_z5_predictive_coding_paired_exact_cpu_cuda`
- required_axes: `['contest_cpu', 'contest_cuda']`
- paired_dispatch_tool: `tools/dispatch_modal_paired_auth_eval.py`
- ready_for_operator_dispatch: `False`
- ready_for_provider_dispatch: `False`
- dispatch_command_executable: `False`
- claim_lifecycle_owner: `tools/dispatch_modal_paired_auth_eval.py and the per-axis Modal auth-eval wrappers`
- measurement_blockers_to_close: `['l5_v2_probe_artifact_path_missing', 'l5_v2_probe_artifact_sha_invalid', 'l5_v2_probe_predicate_failed', 'l5_v2_probe_paired_exact_axes_missing', 'l5_v2_probe_byte_closed_archive_missing', 'l5_v2_probe_sideinfo_consumption_missing', 'l5_v2_probe_archive_sha_invalid', 'l5_v2_probe_runtime_tree_sha_invalid', 'l5_v2_probe_axis_evidence_missing:contest_cpu', 'l5_v2_probe_axis_evidence_missing:contest_cuda']`
- dispatch_blockers: `['requires_byte_closed_archive_path', 'requires_archive_sha256', 'requires_submission_dir_or_inflate_runtime', 'requires_operator_execute_flag']`
- readiness_blockers: `['l5_v2_probe_artifact_path_missing', 'l5_v2_probe_artifact_sha_invalid', 'l5_v2_probe_predicate_failed', 'l5_v2_probe_paired_exact_axes_missing', 'l5_v2_probe_byte_closed_archive_missing', 'l5_v2_probe_sideinfo_consumption_missing', 'l5_v2_probe_archive_sha_invalid', 'l5_v2_probe_runtime_tree_sha_invalid', 'l5_v2_probe_axis_evidence_missing:contest_cpu', 'l5_v2_probe_axis_evidence_missing:contest_cuda', 'requires_byte_closed_archive_path', 'requires_archive_sha256', 'requires_submission_dir_or_inflate_runtime', 'requires_operator_execute_flag']`
- dispatch_command: `.venv/bin/python tools/dispatch_modal_paired_auth_eval.py --archive FILL_ARCHIVE_ZIP --submission-dir FILL_SUBMISSION_DIR --inflate-sh inflate.sh --label l5_v2_z5_predictive_coding_world_model --expected-archive-sha256 FILL_ARCHIVE_SHA256 --run-id l5_v2_measure_z5_predictive_coding_paired_exact_paired_measurement --pair-group-id pair_l5_v2_measure_z5_predictive_coding_paired_exact_cpu_cuda --lane-id-base lane_l5_v2_measure_z5_predictive_coding_paired_exact --output-root experiments/results/l5_v2_probe/measure_z5_predictive_coding_paired_exact --modal-bin .venv/bin/modal --gpu T4 --claim-agent codex:l5_v2_paired_measurement_dispatch --claim-notes l5_v2_paired_measurement:pair_l5_v2_measure_z5_predictive_coding_paired_exact_cpu_cuda --expected-runtime-tree-sha256 auto --skip-axis-if-promotable-anchor-exists`

### measure_tt5l_autonomy_paired_exact

- work_unit_id: `measure_tt5l_autonomy_paired_exact`
- source_measurement_id: `measure_tt5l_autonomy_paired_exact`
- candidate_id: `time_traveler_l5_autonomy`
- sideinfo_variant: ``
- sideinfo_required_cells: `[]`
- lane_id: `lane_l5_v2_measure_tt5l_autonomy_paired_exact`
- lanes: `{'contest_cuda': 'lane_l5_v2_measure_tt5l_autonomy_paired_exact_contest_cuda', 'contest_cpu': 'lane_l5_v2_measure_tt5l_autonomy_paired_exact_contest_cpu'}`
- pair_group_id: `pair_l5_v2_measure_tt5l_autonomy_paired_exact_cpu_cuda`
- required_axes: `['contest_cpu', 'contest_cuda']`
- paired_dispatch_tool: `tools/dispatch_modal_paired_auth_eval.py`
- ready_for_operator_dispatch: `False`
- ready_for_provider_dispatch: `False`
- dispatch_command_executable: `False`
- claim_lifecycle_owner: `tools/dispatch_modal_paired_auth_eval.py and the per-axis Modal auth-eval wrappers`
- measurement_blockers_to_close: `['l5_v2_probe_predicate_failed', 'l5_v2_probe_sideinfo_consumption_missing', 'l5_v2_probe_axis_score_delta_missing:contest_cpu', 'l5_v2_probe_axis_score_delta_missing:contest_cuda']`
- dispatch_blockers: `['requires_byte_closed_archive_path', 'requires_archive_sha256', 'requires_submission_dir_or_inflate_runtime', 'requires_operator_execute_flag']`
- readiness_blockers: `['l5_v2_probe_predicate_failed', 'l5_v2_probe_sideinfo_consumption_missing', 'l5_v2_probe_axis_score_delta_missing:contest_cpu', 'l5_v2_probe_axis_score_delta_missing:contest_cuda', 'requires_byte_closed_archive_path', 'requires_archive_sha256', 'requires_submission_dir_or_inflate_runtime', 'requires_operator_execute_flag']`
- dispatch_command: `.venv/bin/python tools/dispatch_modal_paired_auth_eval.py --archive FILL_ARCHIVE_ZIP --submission-dir FILL_SUBMISSION_DIR --inflate-sh inflate.sh --label l5_v2_time_traveler_l5_autonomy --expected-archive-sha256 FILL_ARCHIVE_SHA256 --run-id l5_v2_measure_tt5l_autonomy_paired_exact_paired_measurement --pair-group-id pair_l5_v2_measure_tt5l_autonomy_paired_exact_cpu_cuda --lane-id-base lane_l5_v2_measure_tt5l_autonomy_paired_exact --output-root experiments/results/l5_v2_probe/measure_tt5l_autonomy_paired_exact --modal-bin .venv/bin/modal --gpu T4 --claim-agent codex:l5_v2_paired_measurement_dispatch --claim-notes l5_v2_paired_measurement:pair_l5_v2_measure_tt5l_autonomy_paired_exact_cpu_cuda --expected-runtime-tree-sha256 auto --skip-axis-if-promotable-anchor-exists`
