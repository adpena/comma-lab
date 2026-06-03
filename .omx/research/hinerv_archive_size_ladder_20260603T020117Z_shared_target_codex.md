# HiNeRV archive-size ladder

Schema: `hinerv_archive_size_ladder.v1`
Authority: `false_authority_archive_size_ladder_no_score_claim`
Axis: `[planning/control]`
Decoder codec: `int8_mixed`
Decoder codec policy: `modelsize_budget_candidate_decoder_codec_overrides_top_level_default`
Modelsize budget schema: `nerv_modelsize_budget.v1`

| row | params | nominal bytes | archive bytes | measured-minus-nominal | rate score [planning/control] | proof ready |
|---|---:|---:|---:|---:|---:|---:|
| hinerv_np600_ld8_ed16_dc6_int2_mixed_ceil178000_tgtmp0p03 | 29646 | 36812 | 61929 | 25117 | 0.041236 | None |

## Marginal Gates


## Blockers

- `hinerv_archive_size_ladder_false_authority_no_nonrate_score`
- `contest_cpu_cuda_exact_eval_not_executed`
- `receiver_proof_not_executed_for_archive_size_ladder`
