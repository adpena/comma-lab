# SNeRV LF/HF Replacement Queue

- schema: `snerv_lf_hf_replacement_queue.v1`
- lane: `lane_snerv_lf_hf_replacement_queue_20260605`
- axis: `[planning/control:false-authority]`
- queue rows: `21`
- runnable local rows: `0`
- current reroute rows: `9`
- current SNAR2 no-LF-overrun: `False`
- LF dominance launch signal active: `True`
- receiver payload frame replay proven: `True`
- official replacement authority ready: `False`
- scorer domain tether proof passed: `True`
- value-domain noncollapse proof passed: `False`
- selected LF evidence bytes: `879633`

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
- blocked: `False`
- depends on: `current_snar2_lf_overrun_handoff`
- blockers:

### `receiver_output2_frame_replay`
- blocked: `False`
- depends on: `official_checkpoint_export_binding`
- blockers:

### `scorer_domain_guard`
- blocked: `False`
- depends on: `receiver_output2_frame_replay`
- blockers:

### `official_tub_lf_hf_decoder_replacement`
- blocked: `True`
- depends on: `scorer_domain_guard`
- blockers:
  - `snerv_official_mfu_hfr_tub_receiver_payload_not_source_forward_authority`
  - `snerv_official_mfu_hfr_tub_full_stack_source_forward_replay_missing`
  - `snerv_official_mfu_hfr_tub_weight_mapping_missing`
  - `snerv_official_trained_checkpoint_state_dict_mapping_missing`
  - `snerv_official_trained_checkpoint_source_forward_replay_missing`
  - `snerv_official_tub_trained_temporal_encoder_decoder_weights_not_loaded`
  - `snerv_official_tub_portable_temporal_encoder_weight_mapping_missing`
  - `snerv_official_trained_checkpoint_state_dict_not_loaded`
  - `snerv_official_tub_encoder_decoder_weights_not_loaded`
  - `snerv_official_tub_normalized_lf_graph_inputs_not_full_source_forward_parity`
  - `snerv_official_mfu_hfr_tub_source_forward_replay_missing`
  - `snerv_official_tub_portable_output2_decoder_weight_mapping_missing`
  - `snerv_official_snerv_t_full_tub_source_forward_replay_missing`
  - `official_weight_tensor_mapping_not_loaded`
  - `full_official_mfu_forward_artifact_not_emitted`
  - `official_hfr_weight_tensor_mapping_not_loaded`
  - `full_official_hfr_forward_artifact_not_emitted`
  - `snerv_official_snerv_t_trained_full_tub_source_forward_parity_missing`
  - `snerv_official_pytorch_wavelets_runtime_dependency_missing`

### `lf_conditioned_hf_residual_generator`
- blocked: `True`
- depends on: `official_tub_lf_hf_decoder_replacement`
- blockers:
  - `snerv_hf_residual_generator_receiver_payload_not_implemented`
  - `snerv_lf_conditioned_hf_receiver_value_domain_sample_decode_missing`
  - `snerv_lf_conditioned_hf_value_domain_noncollapse_proof_missing`
  - `snerv_official_payload_selected_pair_value_xray_unavailable`
  - `snerv_official_skip_high_not_lossless_relative_to_source`
  - `snerv_official_skip_high_receiver_expands_compact_state`
  - `snerv_official_skip_high_scalar_mean_receiver_expand_collapse_risk`

### `remaining_lf_hf_family_implementations`
- blocked: `True`
- depends on: `lf_conditioned_hf_residual_generator`
- blockers:
  - `snerv_joint_lf_hf_factorized_codebook_not_implemented`
  - `snerv_joint_lf_hf_codebook_numpy_receiver_missing`
  - `snerv_joint_lf_hf_codebook_section_byte_telemetry_missing`
  - `snerv_temporal_lf_predictor_gate_not_implemented`
  - `snerv_temporal_lf_predictor_correction_stream_not_byte_charged`
  - `snerv_official_mfu_hfr_tub_receiver_payload_not_source_forward_authority`
  - `snerv_official_mfu_hfr_tub_full_stack_source_forward_replay_missing`
  - `snerv_official_mfu_hfr_tub_weight_mapping_missing`
  - `snerv_official_trained_checkpoint_state_dict_mapping_missing`
  - `snerv_official_trained_checkpoint_source_forward_replay_missing`
  - `snerv_official_tub_trained_temporal_encoder_decoder_weights_not_loaded`
  - `snerv_official_tub_portable_temporal_encoder_weight_mapping_missing`
  - `snerv_official_trained_checkpoint_state_dict_not_loaded`
  - `snerv_official_tub_encoder_decoder_weights_not_loaded`
  - `snerv_official_tub_normalized_lf_graph_inputs_not_full_source_forward_parity`
  - `snerv_official_mfu_hfr_tub_source_forward_replay_missing`
  - `snerv_official_tub_portable_output2_decoder_weight_mapping_missing`
  - `snerv_lf_super_resolution_receiver_payload_not_implemented`
  - `snerv_lf_downsampled_anchor_component_deltas_missing`
  - `snerv_score_tethered_lf_hf_band_allocator_not_implemented`
  - `snerv_mfu_hfr_section_native_byte_telemetry_missing`
  - `snerv_lf_latent_hyperprior_not_implemented`
  - `snerv_lf_latent_hyperprior_numpy_decoder_missing`
  - `snerv_lf_latent_hyperprior_receiver_replay_missing`

## Candidate Rows

### `snerv_lf_hf_replace_snerv_auto_bytecap_snerv_np600_haar_lv1_lfb1p5_stepb4_fc11e0_p1_mfu1_2_4_hfr0_t0_tmhaar1_adofficial_oms0p05_fdb592a769ad`
- family: `official_tub_lf_hf_decoder_replacement`
- action: `run_bounded_source_faithful_lf_hf_decoder_smoke`
- blocked: `True`
- command: ``
- blockers:
  - `snerv_official_mfu_hfr_tub_receiver_payload_not_source_forward_authority`
  - `snerv_official_mfu_hfr_tub_full_stack_source_forward_replay_missing`
  - `snerv_official_mfu_hfr_tub_weight_mapping_missing`
  - `snerv_official_trained_checkpoint_state_dict_mapping_missing`
  - `snerv_official_trained_checkpoint_source_forward_replay_missing`
  - `snerv_official_tub_trained_temporal_encoder_decoder_weights_not_loaded`
  - `snerv_official_tub_portable_temporal_encoder_weight_mapping_missing`
  - `snerv_official_trained_checkpoint_state_dict_not_loaded`
  - `snerv_official_tub_encoder_decoder_weights_not_loaded`
  - `snerv_official_tub_normalized_lf_graph_inputs_not_full_source_forward_parity`
  - `snerv_official_mfu_hfr_tub_source_forward_replay_missing`
  - `snerv_official_tub_portable_output2_decoder_weight_mapping_missing`
  - `snerv_official_snerv_t_full_tub_source_forward_replay_missing`
  - `official_weight_tensor_mapping_not_loaded`
  - `full_official_mfu_forward_artifact_not_emitted`
  - `official_hfr_weight_tensor_mapping_not_loaded`
  - `full_official_hfr_forward_artifact_not_emitted`
  - `snerv_official_snerv_t_trained_full_tub_source_forward_parity_missing`
  - `snerv_official_pytorch_wavelets_runtime_dependency_missing`

### `snerv_lf_hf_replace_snerv_auto_bytecap_snerv_np600_haar_lv1_lfb2_stepb4_fc11e0_p1_mfu1_2_4_hfr0_t0_tmhaar1_adofficial_oms0p05_i_8ded65dbc60a`
- family: `official_tub_lf_hf_decoder_replacement`
- action: `run_bounded_source_faithful_lf_hf_decoder_smoke`
- blocked: `True`
- command: ``
- blockers:
  - `snerv_official_mfu_hfr_tub_receiver_payload_not_source_forward_authority`
  - `snerv_official_mfu_hfr_tub_full_stack_source_forward_replay_missing`
  - `snerv_official_mfu_hfr_tub_weight_mapping_missing`
  - `snerv_official_trained_checkpoint_state_dict_mapping_missing`
  - `snerv_official_trained_checkpoint_source_forward_replay_missing`
  - `snerv_official_tub_trained_temporal_encoder_decoder_weights_not_loaded`
  - `snerv_official_tub_portable_temporal_encoder_weight_mapping_missing`
  - `snerv_official_trained_checkpoint_state_dict_not_loaded`
  - `snerv_official_tub_encoder_decoder_weights_not_loaded`
  - `snerv_official_tub_normalized_lf_graph_inputs_not_full_source_forward_parity`
  - `snerv_official_mfu_hfr_tub_source_forward_replay_missing`
  - `snerv_official_tub_portable_output2_decoder_weight_mapping_missing`
  - `snerv_official_snerv_t_full_tub_source_forward_replay_missing`
  - `official_weight_tensor_mapping_not_loaded`
  - `full_official_mfu_forward_artifact_not_emitted`
  - `official_hfr_weight_tensor_mapping_not_loaded`
  - `full_official_hfr_forward_artifact_not_emitted`
  - `snerv_official_snerv_t_trained_full_tub_source_forward_parity_missing`
  - `snerv_official_pytorch_wavelets_runtime_dependency_missing`

### `snerv_lf_hf_replace_snerv_auto_bytecap_snerv_np600_haar_lv1_lfb2p5_stepb4_fc11e0_p1_mfu1_2_4_hfr0_t0_tmhaar1_adofficial_oms0p05_2b86a49fdad0`
- family: `official_tub_lf_hf_decoder_replacement`
- action: `run_bounded_source_faithful_lf_hf_decoder_smoke`
- blocked: `True`
- command: ``
- blockers:
  - `snerv_official_mfu_hfr_tub_receiver_payload_not_source_forward_authority`
  - `snerv_official_mfu_hfr_tub_full_stack_source_forward_replay_missing`
  - `snerv_official_mfu_hfr_tub_weight_mapping_missing`
  - `snerv_official_trained_checkpoint_state_dict_mapping_missing`
  - `snerv_official_trained_checkpoint_source_forward_replay_missing`
  - `snerv_official_tub_trained_temporal_encoder_decoder_weights_not_loaded`
  - `snerv_official_tub_portable_temporal_encoder_weight_mapping_missing`
  - `snerv_official_trained_checkpoint_state_dict_not_loaded`
  - `snerv_official_tub_encoder_decoder_weights_not_loaded`
  - `snerv_official_tub_normalized_lf_graph_inputs_not_full_source_forward_parity`
  - `snerv_official_mfu_hfr_tub_source_forward_replay_missing`
  - `snerv_official_tub_portable_output2_decoder_weight_mapping_missing`
  - `snerv_official_snerv_t_full_tub_source_forward_replay_missing`
  - `official_weight_tensor_mapping_not_loaded`
  - `full_official_mfu_forward_artifact_not_emitted`
  - `official_hfr_weight_tensor_mapping_not_loaded`
  - `full_official_hfr_forward_artifact_not_emitted`
  - `snerv_official_snerv_t_trained_full_tub_source_forward_parity_missing`
  - `snerv_official_pytorch_wavelets_runtime_dependency_missing`

### `snerv_lf_hf_replace_snerv_auto_bytecap_snerv_np600_haar_lv1_lfb1p5_stepb4_fc11e0_p1_mfu1_2_4_hfr0_t0_tmhaar1_adofficial_oms0p05_13b914e45733`
- family: `lf_conditioned_hf_residual_generator`
- action: `probe_non_scalar_hf_generation_without_skip_high_collapse`
- blocked: `True`
- command: ``
- blockers:
  - `snerv_hf_residual_generator_receiver_payload_not_implemented`
  - `snerv_lf_conditioned_hf_receiver_value_domain_sample_decode_missing`
  - `snerv_lf_conditioned_hf_value_domain_noncollapse_proof_missing`
  - `snerv_official_payload_selected_pair_value_xray_unavailable`
  - `snerv_official_skip_high_not_lossless_relative_to_source`
  - `snerv_official_skip_high_receiver_expands_compact_state`
  - `snerv_official_skip_high_scalar_mean_receiver_expand_collapse_risk`

### `snerv_lf_hf_replace_snerv_auto_bytecap_snerv_np600_haar_lv1_lfb2_stepb4_fc11e0_p1_mfu1_2_4_hfr0_t0_tmhaar1_adofficial_oms0p05_i_b7f1f4101633`
- family: `lf_conditioned_hf_residual_generator`
- action: `probe_non_scalar_hf_generation_without_skip_high_collapse`
- blocked: `True`
- command: ``
- blockers:
  - `snerv_hf_residual_generator_receiver_payload_not_implemented`
  - `snerv_lf_conditioned_hf_receiver_value_domain_sample_decode_missing`
  - `snerv_lf_conditioned_hf_value_domain_noncollapse_proof_missing`
  - `snerv_official_payload_selected_pair_value_xray_unavailable`
  - `snerv_official_skip_high_not_lossless_relative_to_source`
  - `snerv_official_skip_high_receiver_expands_compact_state`
  - `snerv_official_skip_high_scalar_mean_receiver_expand_collapse_risk`

### `snerv_lf_hf_replace_snerv_auto_bytecap_snerv_np600_haar_lv1_lfb2p5_stepb4_fc11e0_p1_mfu1_2_4_hfr0_t0_tmhaar1_adofficial_oms0p05_f310537a843f`
- family: `lf_conditioned_hf_residual_generator`
- action: `probe_non_scalar_hf_generation_without_skip_high_collapse`
- blocked: `True`
- command: ``
- blockers:
  - `snerv_hf_residual_generator_receiver_payload_not_implemented`
  - `snerv_lf_conditioned_hf_receiver_value_domain_sample_decode_missing`
  - `snerv_lf_conditioned_hf_value_domain_noncollapse_proof_missing`
  - `snerv_official_payload_selected_pair_value_xray_unavailable`
  - `snerv_official_skip_high_not_lossless_relative_to_source`
  - `snerv_official_skip_high_receiver_expands_compact_state`
  - `snerv_official_skip_high_scalar_mean_receiver_expand_collapse_risk`

### `snerv_lf_hf_replace_snerv_auto_bytecap_snerv_np600_haar_lv1_lfb1p5_stepb4_fc11e0_p1_mfu1_2_4_hfr0_t0_tmhaar1_adofficial_oms0p05_bc8782734ca7`
- family: `joint_lf_hf_factorized_codebook`
- action: `build_score_tethered_joint_lf_hf_codebook_export`
- blocked: `True`
- command: ``
- blockers:
  - `snerv_joint_lf_hf_factorized_codebook_not_implemented`
  - `snerv_joint_lf_hf_codebook_numpy_receiver_missing`
  - `snerv_joint_lf_hf_codebook_section_byte_telemetry_missing`

### `snerv_lf_hf_replace_snerv_auto_bytecap_snerv_np600_haar_lv1_lfb2_stepb4_fc11e0_p1_mfu1_2_4_hfr0_t0_tmhaar1_adofficial_oms0p05_i_7b1208325950`
- family: `joint_lf_hf_factorized_codebook`
- action: `build_score_tethered_joint_lf_hf_codebook_export`
- blocked: `True`
- command: ``
- blockers:
  - `snerv_joint_lf_hf_factorized_codebook_not_implemented`
  - `snerv_joint_lf_hf_codebook_numpy_receiver_missing`
  - `snerv_joint_lf_hf_codebook_section_byte_telemetry_missing`

### `snerv_lf_hf_replace_snerv_auto_bytecap_snerv_np600_haar_lv1_lfb2p5_stepb4_fc11e0_p1_mfu1_2_4_hfr0_t0_tmhaar1_adofficial_oms0p05_a691433189a4`
- family: `joint_lf_hf_factorized_codebook`
- action: `build_score_tethered_joint_lf_hf_codebook_export`
- blocked: `True`
- command: ``
- blockers:
  - `snerv_joint_lf_hf_factorized_codebook_not_implemented`
  - `snerv_joint_lf_hf_codebook_numpy_receiver_missing`
  - `snerv_joint_lf_hf_codebook_section_byte_telemetry_missing`

### `snerv_lf_hf_replace_snerv_auto_bytecap_snerv_np600_haar_lv1_lfb1p5_stepb4_fc11e0_p1_mfu1_2_4_hfr0_t0_tmhaar1_adofficial_oms0p05_97fad0b4f30b`
- family: `temporal_lf_predictor_gate`
- action: `learn_temporal_lf_delta_predictor_with_receiver_gate`
- blocked: `True`
- command: ``
- blockers:
  - `snerv_temporal_lf_predictor_gate_not_implemented`
  - `snerv_temporal_lf_predictor_correction_stream_not_byte_charged`
  - `snerv_official_mfu_hfr_tub_receiver_payload_not_source_forward_authority`
  - `snerv_official_mfu_hfr_tub_full_stack_source_forward_replay_missing`
  - `snerv_official_mfu_hfr_tub_weight_mapping_missing`
  - `snerv_official_trained_checkpoint_state_dict_mapping_missing`
  - `snerv_official_trained_checkpoint_source_forward_replay_missing`
  - `snerv_official_tub_trained_temporal_encoder_decoder_weights_not_loaded`
  - `snerv_official_tub_portable_temporal_encoder_weight_mapping_missing`
  - `snerv_official_trained_checkpoint_state_dict_not_loaded`
  - `snerv_official_tub_encoder_decoder_weights_not_loaded`
  - `snerv_official_tub_normalized_lf_graph_inputs_not_full_source_forward_parity`
  - `snerv_official_mfu_hfr_tub_source_forward_replay_missing`
  - `snerv_official_tub_portable_output2_decoder_weight_mapping_missing`

### `snerv_lf_hf_replace_snerv_auto_bytecap_snerv_np600_haar_lv1_lfb2_stepb4_fc11e0_p1_mfu1_2_4_hfr0_t0_tmhaar1_adofficial_oms0p05_i_1d9a2f5b4afd`
- family: `temporal_lf_predictor_gate`
- action: `learn_temporal_lf_delta_predictor_with_receiver_gate`
- blocked: `True`
- command: ``
- blockers:
  - `snerv_temporal_lf_predictor_gate_not_implemented`
  - `snerv_temporal_lf_predictor_correction_stream_not_byte_charged`
  - `snerv_official_mfu_hfr_tub_receiver_payload_not_source_forward_authority`
  - `snerv_official_mfu_hfr_tub_full_stack_source_forward_replay_missing`
  - `snerv_official_mfu_hfr_tub_weight_mapping_missing`
  - `snerv_official_trained_checkpoint_state_dict_mapping_missing`
  - `snerv_official_trained_checkpoint_source_forward_replay_missing`
  - `snerv_official_tub_trained_temporal_encoder_decoder_weights_not_loaded`
  - `snerv_official_tub_portable_temporal_encoder_weight_mapping_missing`
  - `snerv_official_trained_checkpoint_state_dict_not_loaded`
  - `snerv_official_tub_encoder_decoder_weights_not_loaded`
  - `snerv_official_tub_normalized_lf_graph_inputs_not_full_source_forward_parity`
  - `snerv_official_mfu_hfr_tub_source_forward_replay_missing`
  - `snerv_official_tub_portable_output2_decoder_weight_mapping_missing`

### `snerv_lf_hf_replace_snerv_auto_bytecap_snerv_np600_haar_lv1_lfb2p5_stepb4_fc11e0_p1_mfu1_2_4_hfr0_t0_tmhaar1_adofficial_oms0p05_3ea1f352febc`
- family: `temporal_lf_predictor_gate`
- action: `learn_temporal_lf_delta_predictor_with_receiver_gate`
- blocked: `True`
- command: ``
- blockers:
  - `snerv_temporal_lf_predictor_gate_not_implemented`
  - `snerv_temporal_lf_predictor_correction_stream_not_byte_charged`
  - `snerv_official_mfu_hfr_tub_receiver_payload_not_source_forward_authority`
  - `snerv_official_mfu_hfr_tub_full_stack_source_forward_replay_missing`
  - `snerv_official_mfu_hfr_tub_weight_mapping_missing`
  - `snerv_official_trained_checkpoint_state_dict_mapping_missing`
  - `snerv_official_trained_checkpoint_source_forward_replay_missing`
  - `snerv_official_tub_trained_temporal_encoder_decoder_weights_not_loaded`
  - `snerv_official_tub_portable_temporal_encoder_weight_mapping_missing`
  - `snerv_official_trained_checkpoint_state_dict_not_loaded`
  - `snerv_official_tub_encoder_decoder_weights_not_loaded`
  - `snerv_official_tub_normalized_lf_graph_inputs_not_full_source_forward_parity`
  - `snerv_official_mfu_hfr_tub_source_forward_replay_missing`
  - `snerv_official_tub_portable_output2_decoder_weight_mapping_missing`

### `snerv_lf_hf_replace_snerv_auto_bytecap_snerv_np600_haar_lv1_lfb1p5_stepb4_fc11e0_p1_mfu1_2_4_hfr0_t0_tmhaar1_adofficial_oms0p05_9ac7355db283`
- family: `lf_super_resolution_from_tiny_anchor`
- action: `store_tiny_lf_anchor_then_learn_receiver_super_resolution`
- blocked: `True`
- command: ``
- blockers:
  - `snerv_lf_super_resolution_receiver_payload_not_implemented`
  - `snerv_lf_downsampled_anchor_component_deltas_missing`

### `snerv_lf_hf_replace_snerv_auto_bytecap_snerv_np600_haar_lv1_lfb2_stepb4_fc11e0_p1_mfu1_2_4_hfr0_t0_tmhaar1_adofficial_oms0p05_i_4ac4fc47a13a`
- family: `lf_super_resolution_from_tiny_anchor`
- action: `store_tiny_lf_anchor_then_learn_receiver_super_resolution`
- blocked: `True`
- command: ``
- blockers:
  - `snerv_lf_super_resolution_receiver_payload_not_implemented`
  - `snerv_lf_downsampled_anchor_component_deltas_missing`

### `snerv_lf_hf_replace_snerv_auto_bytecap_snerv_np600_haar_lv1_lfb2p5_stepb4_fc11e0_p1_mfu1_2_4_hfr0_t0_tmhaar1_adofficial_oms0p05_18cea68cfe3e`
- family: `lf_super_resolution_from_tiny_anchor`
- action: `store_tiny_lf_anchor_then_learn_receiver_super_resolution`
- blocked: `True`
- command: ``
- blockers:
  - `snerv_lf_super_resolution_receiver_payload_not_implemented`
  - `snerv_lf_downsampled_anchor_component_deltas_missing`

### `snerv_lf_hf_replace_snerv_auto_bytecap_snerv_np600_haar_lv1_lfb1p5_stepb4_fc11e0_p1_mfu1_2_4_hfr0_t0_tmhaar1_adofficial_oms0p05_16d5973968aa`
- family: `score_tethered_spectral_band_allocator`
- action: `learn_mfu_hfr_lf_hf_band_budget_from_scorer_telemetry`
- blocked: `True`
- command: ``
- blockers:
  - `snerv_score_tethered_lf_hf_band_allocator_not_implemented`
  - `snerv_mfu_hfr_section_native_byte_telemetry_missing`

### `snerv_lf_hf_replace_snerv_auto_bytecap_snerv_np600_haar_lv1_lfb2_stepb4_fc11e0_p1_mfu1_2_4_hfr0_t0_tmhaar1_adofficial_oms0p05_i_56c6e8f7fa5f`
- family: `score_tethered_spectral_band_allocator`
- action: `learn_mfu_hfr_lf_hf_band_budget_from_scorer_telemetry`
- blocked: `True`
- command: ``
- blockers:
  - `snerv_score_tethered_lf_hf_band_allocator_not_implemented`
  - `snerv_mfu_hfr_section_native_byte_telemetry_missing`

### `snerv_lf_hf_replace_snerv_auto_bytecap_snerv_np600_haar_lv1_lfb2p5_stepb4_fc11e0_p1_mfu1_2_4_hfr0_t0_tmhaar1_adofficial_oms0p05_5845a8ea8869`
- family: `score_tethered_spectral_band_allocator`
- action: `learn_mfu_hfr_lf_hf_band_budget_from_scorer_telemetry`
- blocked: `True`
- command: ``
- blockers:
  - `snerv_score_tethered_lf_hf_band_allocator_not_implemented`
  - `snerv_mfu_hfr_section_native_byte_telemetry_missing`

### `snerv_lf_hf_replace_snerv_auto_bytecap_snerv_np600_haar_lv1_lfb1p5_stepb4_fc11e0_p1_mfu1_2_4_hfr0_t0_tmhaar1_adofficial_oms0p05_6a7c2e7119cd`
- family: `entropy_modeled_lf_latent_hyperprior`
- action: `replace_i64_lzma_lf_planes_with_learned_entropy_model`
- blocked: `True`
- command: ``
- blockers:
  - `snerv_lf_latent_hyperprior_not_implemented`
  - `snerv_lf_latent_hyperprior_numpy_decoder_missing`
  - `snerv_lf_latent_hyperprior_receiver_replay_missing`

### `snerv_lf_hf_replace_snerv_auto_bytecap_snerv_np600_haar_lv1_lfb2_stepb4_fc11e0_p1_mfu1_2_4_hfr0_t0_tmhaar1_adofficial_oms0p05_i_1430c1b8af14`
- family: `entropy_modeled_lf_latent_hyperprior`
- action: `replace_i64_lzma_lf_planes_with_learned_entropy_model`
- blocked: `True`
- command: ``
- blockers:
  - `snerv_lf_latent_hyperprior_not_implemented`
  - `snerv_lf_latent_hyperprior_numpy_decoder_missing`
  - `snerv_lf_latent_hyperprior_receiver_replay_missing`

### `snerv_lf_hf_replace_snerv_auto_bytecap_snerv_np600_haar_lv1_lfb2p5_stepb4_fc11e0_p1_mfu1_2_4_hfr0_t0_tmhaar1_adofficial_oms0p05_676132dcb626`
- family: `entropy_modeled_lf_latent_hyperprior`
- action: `replace_i64_lzma_lf_planes_with_learned_entropy_model`
- blocked: `True`
- command: ``
- blockers:
  - `snerv_lf_latent_hyperprior_not_implemented`
  - `snerv_lf_latent_hyperprior_numpy_decoder_missing`
  - `snerv_lf_latent_hyperprior_receiver_replay_missing`
