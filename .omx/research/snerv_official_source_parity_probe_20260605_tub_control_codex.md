# SNeRV Official Source-Parity Audit

Schema: `snerv_official_source_parity_audit.v1`
Authority: `false_authority_source_audit_no_score_claim`
Official repo: `/Volumes/VertigoDataTier/pact/experiments/results/oss_nerv_source_audit_20260602T113720Z/repos/SNeRV`
Official head SHA: `0844a08f9591eea9625f8b961ed91d08030e06d1`

## Verdict

- official source markers present: `True`
- local receiver-safe adapter present: `True`
- official MFU/HFR/TUB receiver primitive replay proven: `True`
- official MFU/HFR/TUB numeric graph replay proven: `True`
- official receiver runtime decode proven: `True`
- official receiver source-forward replay bound: `False`
- official MFU/HFR/TUB parity proven: `False`
- official MFU/HFR/TUB parity falsified: `False`
- score claim: `False`

## Component States

| component | classification | receiver analogue | official forward parity | blockers |
|---|---|---:|---:|---|
| `mfu` | `official_source_fixture_mfu_state_dict_mapping_proven` | `True` | `True` | `` |
| `hfr` | `official_source_fixture_hfr_state_dict_mapping_proven` | `True` | `True` | `` |
| `tub` | `official_tub_graph_input_and_output2_fusion_source_fixture_proven_full_tub_blocked` | `True` | `False` | `snerv_tub_local_source_forward_markers_missing, snerv_official_mfu_hfr_tub_parity_marker_missing, snerv_official_pytorch_wavelets_runtime_dependency_missing, snerv_official_tub_portable_temporal_encoder_weight_mapping_missing, snerv_official_tub_portable_output2_decoder_weight_mapping_missing, snerv_official_snerv_t_trained_full_tub_source_forward_parity_missing` |
## Marker Groups

| group | present | missing |
|---|---:|---|
| `official_haar_dwt_lf_hf_split` | `True` | `` |
| `official_mfu_multi_resolution_fusion` | `True` | `` |
| `official_hfr_high_frequency_restoration` | `True` | `` |
| `official_tub_temporal_extension` | `True` | `` |
| `official_modelsize_fc_dim_solver` | `True` | `` |
| `official_quantized_payload_controls` | `True` | `` |

## Blockers

- `snerv_official_mfu_hfr_tub_parity_missing`

## Next Actions

- implement source-forward official MFU/HFR/TUB behavior proof or explicitly supersede it with same-axis receiver evidence
