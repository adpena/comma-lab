# L5 v2 architecture lock packet

- schema: `l5_v2_architecture_lock_packet_v1`
- subject_id: `time_traveler_l5_autonomy`
- lane_id: `lane_time_traveler_l5_autonomy_substrate_20260513`
- architecture_lock_allowed: `False`
- readiness_architecture_lock_allowed: `False`
- next_action: `review_and_execute_l5_v2_tt5l_materialized_paired_measurement`
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
- `anchor_pair_evidence_valid`: `False`

## Lightning Paired-Axis Dry-Run Plan

- artifact_path: `.omx/research/l5_v2_tt5l_sideinfo_effect_curve_lightning_paired_axis_plan_20260517_codex.json`
- artifact_valid: `True`
- cells: `10`/`10`
- axes: `['contest_cpu', 'contest_cuda']`
- all_cells_dry_run_ready: `True`
- execution_ready: `False`
- score_claim: `false`
- promotion_eligible: `false`

## Blockers
- `requires_all_l5_v2_gate_evidence_valid`
- `requires_c1_z5_tt5l_probe_gate_evidence`
- `requires_paired_cpu_cuda_axis_plan`
- `requires_paired_cpu_cuda_sideinfo_effect_curve`
- `requires_tt5l_first_anchor_timing_smoke_artifact`
- `requires_exact_or_diagnostic_anchor_pair`

## Authority

lock/no-lock planning packet only; no score, rank, promotion, or exact-dispatch authority
