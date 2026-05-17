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
- `paired_axis_plan_evidence_valid`: `False`
- `sideinfo_effect_curve_artifact_valid`: `False`
- `first_anchor_timing_smoke_artifact_valid`: `False`
- `anchor_pair_evidence_valid`: `True`

## Lightning Paired-Axis Dry-Run Plan

- artifact_path: `.omx/research/l5_v2_tt5l_sideinfo_effect_curve_lightning_paired_axis_plan_20260517_codex.json`
- artifact_valid: `True`
- source_commit: `a23606c9b1a81aacf154d6c2c12aca5408203d46`
- source_relevant_paths_match: `True`
- source_relevant_diff_paths: `[]`
- source_custody_current_for_execution: `True`
- cells: `10`/`10`
- axes: `['contest_cpu', 'contest_cuda']`
- all_cells_dry_run_ready: `True`
- execution_ready: `False`
- execution_blockers: `['l5_v2_tt5l_lightning_paired_axis_plan_dry_run_only_no_provider_job_launched', 'l5_v2_tt5l_lightning_paired_axis_plan_blocked:dry_run_only_no_provider_job_launched', 'l5_v2_tt5l_lightning_paired_axis_plan_blocked:requires_lightning_identity_and_workspace_preflight_before_submit', 'l5_v2_tt5l_lightning_paired_axis_plan_blocked:requires_source_manifest_staged_to_lightning_workspace_before_submit', 'l5_v2_tt5l_lightning_paired_axis_plan_blocked:requires_per_axis_lane_claim_before_non_dry_run_submit', 'l5_v2_tt5l_lightning_paired_axis_plan_blocked:requires_harvested_contest_cpu_and_contest_cuda_cells_before_sideinfo_effect_claim', 'l5_v2_tt5l_lightning_paired_axis_plan_blocked:score_claim_forbidden_until_effect_curve_artifact_passes']`
- score_claim: `false`
- promotion_eligible: `false`

## Blockers
- `requires_all_l5_v2_gate_evidence_valid`
- `requires_c1_z5_tt5l_probe_gate_evidence`
- `requires_paired_cpu_cuda_axis_plan`
- `requires_paired_cpu_cuda_sideinfo_effect_curve`
- `requires_tt5l_first_anchor_timing_smoke_artifact`

## Authority

lock/no-lock planning packet only; no score, rank, promotion, or exact-dispatch authority
