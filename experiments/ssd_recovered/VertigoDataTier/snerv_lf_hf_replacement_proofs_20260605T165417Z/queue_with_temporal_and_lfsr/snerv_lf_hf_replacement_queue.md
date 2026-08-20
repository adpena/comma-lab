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
  - `snerv_official_mfu_hfr_tub_export_not_bound`
  - `snerv_official_mfu_hfr_tub_receiver_payload_not_bound`
  - `snerv_official_mfu_hfr_tub_frame_producing_export_missing`
  - `snerv_official_mfu_hfr_tub_receiver_payload_not_source_forward_authority`
  - `snerv_official_mfu_hfr_tub_full_stack_source_forward_replay_missing`
  - `snerv_scorer_input_distribution_guard_missing`
  - `snerv_lf_hf_source_forward_artifact_missing`
  - `snerv_official_tub_lf_hf_decoder_replacement_authority_gate_missing`

### `lf_conditioned_hf_residual_generator`
- blocked: `True`
- depends on: `official_tub_lf_hf_decoder_replacement`
- blockers:
  - `snerv_hf_residual_generator_receiver_payload_not_implemented`
  - `snerv_scorer_input_distribution_guard_missing`
  - `snerv_lf_conditioned_hf_value_domain_xray_missing`

### `remaining_lf_hf_family_implementations`
- blocked: `True`
- depends on: `lf_conditioned_hf_residual_generator`
- blockers:
  - `snerv_joint_lf_hf_factorized_codebook_not_implemented`
  - `snerv_joint_lf_hf_codebook_numpy_receiver_missing`
  - `snerv_joint_lf_hf_codebook_section_byte_telemetry_missing`
  - `snerv_scorer_input_distribution_guard_missing`
  - `snerv_temporal_lf_predictor_receiver_runtime_binding_missing`
  - `snerv_score_tethered_lf_hf_band_allocator_not_implemented`
  - `snerv_mfu_hfr_section_native_byte_telemetry_missing`
  - `snerv_lf_latent_hyperprior_not_implemented`
  - `snerv_lf_latent_hyperprior_numpy_decoder_missing`
  - `snerv_lf_latent_hyperprior_receiver_replay_missing`

## Candidate Rows

### `snerv_lf_hf_replace_snerv_snerv_np600_haar_lv1_lfb1p5_stepb0p5_fc11e0_p1_mfu1_2_4_hfr0_t0_tmhaar1_adofficial_oms0p05_skchannelm_95bef8d6f21f`
- family: `official_tub_lf_hf_decoder_replacement`
- action: `run_bounded_source_faithful_lf_hf_decoder_smoke`
- blocked: `True`
- command: ``
- blockers:
  - `snerv_official_mfu_hfr_tub_export_not_bound`
  - `snerv_official_mfu_hfr_tub_receiver_payload_not_bound`
  - `snerv_official_mfu_hfr_tub_frame_producing_export_missing`
  - `snerv_official_mfu_hfr_tub_receiver_payload_not_source_forward_authority`
  - `snerv_official_mfu_hfr_tub_full_stack_source_forward_replay_missing`
  - `snerv_scorer_input_distribution_guard_missing`
  - `snerv_lf_hf_source_forward_artifact_missing`
  - `snerv_official_tub_lf_hf_decoder_replacement_authority_gate_missing`

### `snerv_lf_hf_replace_snerv_snerv_np600_haar_lv1_lfb1p5_stepb0p5_fc11e0_p1_mfu1_2_4_hfr0_t0_tmhaar1_adofficial_oms0p05_skchannelm_4fe70ff52e64`
- family: `lf_conditioned_hf_residual_generator`
- action: `probe_non_scalar_hf_generation_without_skip_high_collapse`
- blocked: `True`
- command: ``
- blockers:
  - `snerv_hf_residual_generator_receiver_payload_not_implemented`
  - `snerv_scorer_input_distribution_guard_missing`
  - `snerv_lf_conditioned_hf_value_domain_xray_missing`

### `snerv_lf_hf_replace_snerv_snerv_np600_haar_lv1_lfb1p5_stepb0p5_fc11e0_p1_mfu1_2_4_hfr0_t0_tmhaar1_adofficial_oms0p05_skchannelm_1edec970b5db`
- family: `joint_lf_hf_factorized_codebook`
- action: `build_score_tethered_joint_lf_hf_codebook_export`
- blocked: `True`
- command: ``
- blockers:
  - `snerv_joint_lf_hf_factorized_codebook_not_implemented`
  - `snerv_joint_lf_hf_codebook_numpy_receiver_missing`
  - `snerv_joint_lf_hf_codebook_section_byte_telemetry_missing`
  - `snerv_scorer_input_distribution_guard_missing`

### `snerv_lf_hf_replace_snerv_snerv_np600_haar_lv1_lfb1p5_stepb0p5_fc11e0_p1_mfu1_2_4_hfr0_t0_tmhaar1_adofficial_oms0p05_skchannelm_e8a291cbe31f`
- family: `temporal_lf_predictor_gate`
- action: `learn_temporal_lf_delta_predictor_with_receiver_gate`
- blocked: `True`
- command: ``
- blockers:
  - `snerv_temporal_lf_predictor_receiver_runtime_binding_missing`

### `snerv_lf_hf_replace_snerv_snerv_np600_haar_lv1_lfb1p5_stepb0p5_fc11e0_p1_mfu1_2_4_hfr0_t0_tmhaar1_adofficial_oms0p05_skchannelm_bbda2946227f`
- family: `lf_super_resolution_from_tiny_anchor`
- action: `store_tiny_lf_anchor_then_learn_receiver_super_resolution`
- blocked: `True`
- command: ``
- blockers:
  - `snerv_scorer_input_distribution_guard_missing`

### `snerv_lf_hf_replace_snerv_snerv_np600_haar_lv1_lfb1p5_stepb0p5_fc11e0_p1_mfu1_2_4_hfr0_t0_tmhaar1_adofficial_oms0p05_skchannelm_6673eeba6ca2`
- family: `score_tethered_spectral_band_allocator`
- action: `learn_mfu_hfr_lf_hf_band_budget_from_scorer_telemetry`
- blocked: `True`
- command: ``
- blockers:
  - `snerv_score_tethered_lf_hf_band_allocator_not_implemented`
  - `snerv_mfu_hfr_section_native_byte_telemetry_missing`
  - `snerv_scorer_input_distribution_guard_missing`

### `snerv_lf_hf_replace_snerv_snerv_np600_haar_lv1_lfb1p5_stepb0p5_fc11e0_p1_mfu1_2_4_hfr0_t0_tmhaar1_adofficial_oms0p05_skchannelm_9b9e62ae49ff`
- family: `entropy_modeled_lf_latent_hyperprior`
- action: `replace_i64_lzma_lf_planes_with_learned_entropy_model`
- blocked: `True`
- command: ``
- blockers:
  - `snerv_lf_latent_hyperprior_not_implemented`
  - `snerv_lf_latent_hyperprior_numpy_decoder_missing`
  - `snerv_lf_latent_hyperprior_receiver_replay_missing`
  - `snerv_scorer_input_distribution_guard_missing`
