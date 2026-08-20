# SNeRV LF/HF Replacement Queue

- schema: `snerv_lf_hf_replacement_queue.v1`
- lane: `lane_snerv_lf_hf_replacement_queue_20260605`
- axis: `[planning/control:false-authority]`
- queue rows: `1`
- runnable local rows: `0`
- current reroute rows: `None`
- current SNAR2 no-LF-overrun: `None`
- LF dominance launch signal active: `False`
- receiver payload frame replay proven: `True`
- official replacement authority ready: `False`
- scorer domain tether proof passed: `False`
- value-domain noncollapse proof passed: `False`
- selected LF evidence bytes: `879633`

## Roadmap DAG

### `measured_lf_payload_reports`
- blocked: `False`
- depends on: ``
- blockers:

### `current_snar2_lf_overrun_handoff`
- blocked: `True`
- depends on: `measured_lf_payload_reports`
- blockers:
  - `current_snar2_lf_overrun_handoff_not_proven`

### `official_checkpoint_export_binding`
- blocked: `False`
- depends on: `current_snar2_lf_overrun_handoff`
- blockers:

### `receiver_output2_frame_replay`
- blocked: `False`
- depends on: `official_checkpoint_export_binding`
- blockers:

### `scorer_domain_guard`
- blocked: `True`
- depends on: `receiver_output2_frame_replay`
- blockers:
  - `snerv_scorer_input_distribution_guard_missing`

### `official_tub_lf_hf_decoder_replacement`
- blocked: `True`
- depends on: `scorer_domain_guard`
- blockers:
  - `snerv_lf_hf_replacement_family_rows_missing`

### `lf_conditioned_hf_residual_generator`
- blocked: `True`
- depends on: `official_tub_lf_hf_decoder_replacement`
- blockers:
  - `snerv_lf_hf_replacement_family_rows_missing`

### `remaining_lf_hf_family_implementations`
- blocked: `True`
- depends on: `lf_conditioned_hf_residual_generator`
- blockers:
  - `snerv_lf_hf_replacement_family_rows_missing`

## Candidate Rows

### `snerv_lf_hf_replace_global_blocker`
- family: `lf_hf_replacement_queue_bootstrap`
- action: `attach_current_snerv_campaign_plan_before_candidate_emission`
- blocked: `True`
- command: ``
- blockers:
  - `snerv_lf_hf_replacement_no_snerv_campaign_rows`
