# NeRV Source-Parity Contract

Schema: `nerv_source_parity_contract.v1`
Authority: `false_authority_source_parity_no_score_claim`

## Family Status

| family | long-training ready | blockers |
|---|---:|---:|
| hi_nerv | no | 1 |
| snerv | no | 1 |

## Blocking Gaps

- `hi_nerv_prune_quantnoise_torchac_pipeline_missing`
- `snerv_official_mfu_hfr_tub_parity_missing`

## Next Actions

- close HiNeRV official config/modelsize ladder and receiver bitstream replay before long run
- close SNeRV fc_dim/MFU/HFR or explicitly block source-faithful SNeRV before long run
