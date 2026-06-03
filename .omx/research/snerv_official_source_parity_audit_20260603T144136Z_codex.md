# SNeRV Official Source-Parity Audit

Schema: `snerv_official_source_parity_audit.v1`
Authority: `false_authority_source_audit_no_score_claim`
Official repo: `/Volumes/VertigoDataTier/pact/experiments/results/oss_nerv_source_audit_20260602T113720Z/repos/SNeRV`
Official head SHA: `0844a08f9591eea9625f8b961ed91d08030e06d1`

## Verdict

- official source markers present: `True`
- local receiver-safe adapter present: `True`
- official MFU/HFR/TUB parity proven: `False`
- official MFU/HFR/TUB parity falsified: `True`
- score claim: `False`

## Component States

| component | classification | receiver analogue | official forward parity | blockers |
|---|---|---:|---:|---|
| `mfu` | `source_forward_parity_falsified_receiver_safe_analogue_only` | `True` | `False` | `snerv_mfu_local_source_forward_markers_missing, snerv_official_mfu_hfr_tub_parity_marker_missing, snerv_official_forward_parity_artifact_falsifies_parity, snerv_mfu_source_forward_parity_falsified_receiver_safe_analogue_only` |
| `hfr` | `source_forward_parity_falsified_receiver_safe_analogue_only` | `True` | `False` | `snerv_hfr_local_source_forward_markers_missing, snerv_official_mfu_hfr_tub_parity_marker_missing, snerv_official_forward_parity_artifact_falsifies_parity, snerv_hfr_source_forward_parity_falsified_receiver_safe_analogue_only` |
| `tub` | `official_haar_temporal_lowpass_primitive_proven_full_tub_falsified` | `True` | `False` | `snerv_tub_local_source_forward_markers_missing, snerv_official_mfu_hfr_tub_parity_marker_missing, snerv_official_forward_parity_artifact_falsifies_parity, snerv_tub_full_source_forward_parity_falsified_primitive_only` |
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
