# NeRV Source-Parity Contract

Schema: `nerv_source_parity_contract.v1`
Authority: `false_authority_source_parity_no_score_claim`

## Family Status

| family | long-training ready | blockers |
|---|---:|---:|
| snerv | no | 4 |
| hi_nerv | no | 2 |

## Blocking Gaps

- `hi_nerv_official_feature_grid_convnext_trilinear_missing`
- `hi_nerv_prune_quantnoise_torchac_pipeline_missing`
- `snerv_mfu_hfr_stride_stack_missing`
- `snerv_scorer_loop_decoder_qat_missing`
- `snerv_qat_receiver_codec_pricing_missing`
- `snerv_fc_dim_modelsize_control_missing`

## Next Actions

- close HiNeRV official config/modelsize ladder and receiver bitstream replay before long run
- close SNeRV fc_dim/MFU/HFR or explicitly block source-faithful SNeRV before long run
