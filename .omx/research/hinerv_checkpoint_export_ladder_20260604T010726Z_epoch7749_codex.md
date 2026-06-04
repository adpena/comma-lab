# HiNeRV archive-size ladder

Schema: `hinerv_archive_size_ladder.v1`
Authority: `false_authority_trained_checkpoint_archive_ladder_no_score_claim`
Axis: `[planning/control]`
Decoder codec: `None`
Decoder codec policy: `trained_checkpoint_export_row_decoder_codec`
Modelsize budget schema: `None`
Modelsize receiver contract: `hinerv_archive_size_ladder_modelsize_receiver_contract.v1`

| row | params | nominal bytes | archive bytes | measured-minus-nominal | rate score [planning/control] | proof ready |
|---|---:|---:|---:|---:|---:|---:|
| hinerv_np600_ld16_ed8_dc16_mi1fi4_hfg_cnx_lg2c4_cx2k7_int7_mixed_ceil178000 | 159718 | 177554 | 121572 | -55982 | 0.080950 | True |

## Marginal Gates


## Blockers

- `hinerv_archive_size_ladder_false_authority_no_nonrate_score`
- `contest_cpu_cuda_exact_eval_not_executed`
