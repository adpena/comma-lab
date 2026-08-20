# SNeRV Official TUB LF/HF Replacement Authority Gate

- schema: `snerv_official_tub_lf_hf_decoder_replacement_authority_gate.v1`
- lane: `lane_snerv_official_tub_lf_hf_decoder_replacement_20260605`
- replacement ready: `False`
- checkpoint export binding ready: `True`
- receiver output2 frame replay ready: `True`
- trained checkpoint mapping ready: `False`
- full TUB source-forward replay ready: `False`
- score claim: `False`

## Gates

### `official_checkpoint_export_binding`
- blocked: `False`
- depends on: ``
- blockers:

### `receiver_output2_frame_replay`
- blocked: `False`
- depends on: `official_checkpoint_export_binding`
- blockers:

### `trained_checkpoint_state_dict_mapping`
- blocked: `True`
- depends on: `receiver_output2_frame_replay`
- blockers:
  - `snerv_official_trained_checkpoint_state_dict_not_loaded`
  - `snerv_official_trained_checkpoint_hfr_weight_mapping_incomplete`
  - `snerv_official_trained_checkpoint_mfu_weight_mapping_incomplete`
  - `snerv_official_trained_checkpoint_state_dict_mapping_missing`

### `tub_temporal_output2_weight_mapping`
- blocked: `True`
- depends on: `trained_checkpoint_state_dict_mapping`
- blockers:
  - `snerv_official_tub_trained_temporal_encoder_decoder_weights_not_loaded`
  - `snerv_official_tub_portable_temporal_encoder_weight_mapping_missing`
  - `snerv_official_tub_portable_output2_decoder_weight_mapping_missing`

### `full_tub_source_forward_replay`
- blocked: `True`
- depends on: `tub_temporal_output2_weight_mapping`
- blockers:
  - `snerv_official_mfu_hfr_tub_receiver_payload_not_source_forward_authority`
  - `snerv_official_mfu_hfr_tub_full_stack_source_forward_replay_missing`

### `official_tub_lf_hf_decoder_replacement`
- blocked: `True`
- depends on: `official_checkpoint_export_binding, receiver_output2_frame_replay, trained_checkpoint_state_dict_mapping, tub_temporal_output2_weight_mapping, full_tub_source_forward_replay`
- blockers:
  - `snerv_official_trained_checkpoint_state_dict_not_loaded`
  - `snerv_official_trained_checkpoint_hfr_weight_mapping_incomplete`
  - `snerv_official_trained_checkpoint_mfu_weight_mapping_incomplete`
  - `snerv_official_trained_checkpoint_state_dict_mapping_missing`
  - `snerv_official_tub_trained_temporal_encoder_decoder_weights_not_loaded`
  - `snerv_official_tub_portable_temporal_encoder_weight_mapping_missing`
  - `snerv_official_tub_portable_output2_decoder_weight_mapping_missing`
  - `snerv_official_mfu_hfr_tub_receiver_payload_not_source_forward_authority`
  - `snerv_official_mfu_hfr_tub_full_stack_source_forward_replay_missing`

## Commands

- rebuild gate:
  - `uv run python tools/build_snerv_official_tub_lf_hf_replacement_authority_gate.py --source-forward-artifact /Volumes/VertigoDataTier/pact/experiments/results/snerv_official_source_forward_strict_output2_20260605Tcodex/snerv_official_mfu_hfr_tub_forward_parity.json --checkpoint-export-report /Volumes/VertigoDataTier/pact/nerv_long_training_campaigns/snerv_auto_bytecap_native_rate_aware_training/checkpoint_exports/epoch003999_20260604T051853Z_official_ema_export_trained_state_verify_20260604Tcodex/snerv_checkpoint_archive_export.json --output-root /Volumes/VertigoDataTier/pact/snerv_official_tub_lf_hf_replacement_authority_gate_strict_output2_20260605Tcodex --output-json /Volumes/VertigoDataTier/pact/snerv_official_tub_lf_hf_replacement_authority_gate_strict_output2_20260605Tcodex/snerv_official_tub_lf_hf_replacement_authority_gate.json --output-md /Volumes/VertigoDataTier/pact/snerv_official_tub_lf_hf_replacement_authority_gate_strict_output2_20260605Tcodex/snerv_official_tub_lf_hf_replacement_authority_gate.md`
- next unblock:
  - `uv run python tools/audit_snerv_official_source_parity.py --official-repo-dir /Volumes/VertigoDataTier/pact/experiments/results/oss_nerv_source_audit_20260602T113720Z/repos/SNeRV --checkpoint-export-report /Volumes/VertigoDataTier/pact/nerv_long_training_campaigns/snerv_auto_bytecap_native_rate_aware_training/checkpoint_exports/epoch003999_20260604T051853Z_official_ema_export_trained_state_verify_20260604Tcodex/snerv_checkpoint_archive_export.json --output-forward-parity-artifact /Volumes/VertigoDataTier/pact/snerv_official_tub_lf_hf_replacement_authority_gate_strict_output2_20260605Tcodex/snerv_official_mfu_hfr_tub_forward_parity_after_mapping.json --output-json /Volumes/VertigoDataTier/pact/snerv_official_tub_lf_hf_replacement_authority_gate_strict_output2_20260605Tcodex/snerv_official_source_parity_audit_after_mapping.json`
