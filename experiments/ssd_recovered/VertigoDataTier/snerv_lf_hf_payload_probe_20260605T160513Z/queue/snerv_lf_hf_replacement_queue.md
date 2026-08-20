# SNeRV LF/HF Replacement Queue

- schema: `snerv_lf_hf_replacement_queue.v1`
- lane: `lane_snerv_lf_hf_replacement_queue_20260605`
- axis: `[planning/control:false-authority]`
- queue rows: `7`
- runnable local rows: `0`
- current reroute rows: `1`
- current SNAR2 no-LF-overrun: `False`
- LF dominance launch signal active: `True`
- receiver payload frame replay proven: `False`
- official replacement authority ready: `False`
- scorer domain tether proof passed: `False`
- value-domain noncollapse proof passed: `False`
- selected LF evidence bytes: `465`

## Roadmap DAG

### `measured_lf_payload_reports`
- blocked: `False`
- depends on: ``
- blockers:

### `current_snar2_lf_overrun_handoff`
- blocked: `False`
- depends on: `measured_lf_payload_reports`
- blockers:

### `official_checkpoint_export_binding`
- blocked: `True`
- depends on: `current_snar2_lf_overrun_handoff`
- blockers:
  - `snerv_official_mfu_hfr_tub_export_not_bound`

### `receiver_output2_frame_replay`
- blocked: `True`
- depends on: `official_checkpoint_export_binding`
- blockers:
  - `receiver_output2_frame_replay_not_proven`

### `scorer_domain_guard`
- blocked: `True`
- depends on: `receiver_output2_frame_replay`
- blockers:
  - `snerv_scorer_input_distribution_guard_missing`

### `official_tub_lf_hf_decoder_replacement`
- blocked: `True`
- depends on: `scorer_domain_guard`
- blockers:
  - `snerv_lf_hf_source_forward_artifact_missing`
  - `snerv_official_tub_lf_hf_decoder_replacement_authority_gate_missing`

### `lf_conditioned_hf_residual_generator`
- blocked: `True`
- depends on: `official_tub_lf_hf_decoder_replacement`
- blockers:
  - `snerv_lf_conditioned_hf_value_domain_xray_missing`

### `remaining_lf_hf_family_implementations`
- blocked: `True`
- depends on: `lf_conditioned_hf_residual_generator`
- blockers:
  - `snerv_temporal_lf_predictor_gate_not_implemented`
  - `snerv_temporal_lf_predictor_correction_stream_not_byte_charged`
  - `snerv_lf_hf_source_forward_artifact_missing`
  - `snerv_lf_super_resolution_receiver_payload_not_implemented`
  - `snerv_lf_downsampled_anchor_component_deltas_missing`
  - `snerv_score_tethered_lf_hf_band_allocator_not_implemented`
  - `snerv_mfu_hfr_section_native_byte_telemetry_missing`
  - `snerv_lf_latent_hyperprior_not_implemented`
  - `snerv_lf_latent_hyperprior_numpy_decoder_missing`
  - `snerv_lf_latent_hyperprior_receiver_replay_missing`

## Candidate Rows

### `snerv_lf_hf_replace_snerv_lf_hf_payload_probe_4pair_official_tub_lf_hf_decoder_replacement_ab2f8b86a1ee`
- family: `official_tub_lf_hf_decoder_replacement`
- action: `run_bounded_source_faithful_lf_hf_decoder_smoke`
- blocked: `True`
- command: ``
- blockers:
  - `snerv_lf_hf_source_forward_artifact_missing`
  - `snerv_official_tub_lf_hf_decoder_replacement_authority_gate_missing`

### `snerv_lf_hf_replace_snerv_lf_hf_payload_probe_4pair_lf_conditioned_hf_residual_generator_4c15e5576ce8`
- family: `lf_conditioned_hf_residual_generator`
- action: `probe_non_scalar_hf_generation_without_skip_high_collapse`
- blocked: `True`
- command: ``
- blockers:
  - `snerv_lf_conditioned_hf_value_domain_xray_missing`

### `snerv_lf_hf_replace_snerv_lf_hf_payload_probe_4pair_joint_lf_hf_factorized_codebook_28e3c06a4f25`
- family: `joint_lf_hf_factorized_codebook`
- action: `build_score_tethered_joint_lf_hf_codebook_export`
- blocked: `False`
- command: ``
- blockers:

### `snerv_lf_hf_replace_snerv_lf_hf_payload_probe_4pair_temporal_lf_predictor_gate_62325bcb26da`
- family: `temporal_lf_predictor_gate`
- action: `learn_temporal_lf_delta_predictor_with_receiver_gate`
- blocked: `True`
- command: ``
- blockers:
  - `snerv_temporal_lf_predictor_gate_not_implemented`
  - `snerv_temporal_lf_predictor_correction_stream_not_byte_charged`
  - `snerv_lf_hf_source_forward_artifact_missing`

### `snerv_lf_hf_replace_snerv_lf_hf_payload_probe_4pair_lf_super_resolution_from_tiny_anchor_1c36241d8bc2`
- family: `lf_super_resolution_from_tiny_anchor`
- action: `store_tiny_lf_anchor_then_learn_receiver_super_resolution`
- blocked: `True`
- command: ``
- blockers:
  - `snerv_lf_super_resolution_receiver_payload_not_implemented`
  - `snerv_lf_downsampled_anchor_component_deltas_missing`

### `snerv_lf_hf_replace_snerv_lf_hf_payload_probe_4pair_score_tethered_spectral_band_allocator_304843bdaf14`
- family: `score_tethered_spectral_band_allocator`
- action: `learn_mfu_hfr_lf_hf_band_budget_from_scorer_telemetry`
- blocked: `True`
- command: ``
- blockers:
  - `snerv_score_tethered_lf_hf_band_allocator_not_implemented`
  - `snerv_mfu_hfr_section_native_byte_telemetry_missing`

### `snerv_lf_hf_replace_snerv_lf_hf_payload_probe_4pair_entropy_modeled_lf_latent_hyperprior_22c30b1ebf25`
- family: `entropy_modeled_lf_latent_hyperprior`
- action: `replace_i64_lzma_lf_planes_with_learned_entropy_model`
- blocked: `True`
- command: ``
- blockers:
  - `snerv_lf_latent_hyperprior_not_implemented`
  - `snerv_lf_latent_hyperprior_numpy_decoder_missing`
  - `snerv_lf_latent_hyperprior_receiver_replay_missing`
