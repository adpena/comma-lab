# HiNeRV archive-size ladder

Schema: `hinerv_archive_size_ladder.v1`
Authority: `false_authority_archive_size_ladder_no_score_claim`
Decoder codec: `int8_mixed`

| row | params | archive bytes | rate score | proof ready |
|---|---:|---:|---:|---:|
| hi_nerv_local_tiny | 94764 | 134938 | 0.089850 | None |
| hi_nerv_local_small | 198873 | 247566 | 0.164844 | None |
| hi_nerv_local_base | 340802 | 398034 | 0.265035 | None |
| hi_nerv_local_wide | 738120 | 812388 | 0.540936 | None |

## Marginal Gates

- `hi_nerv_local_tiny` -> `hi_nerv_local_small` adds `112628` B; requires non-rate drop >= `0.07499436217224392`
- `hi_nerv_local_small` -> `hi_nerv_local_base` adds `150468` B; requires non-rate drop >= `0.10019046495838689`
- `hi_nerv_local_base` -> `hi_nerv_local_wide` adds `414354` B; requires non-rate drop >= `0.2759013206619842`

## Blockers

- `hinerv_archive_size_ladder_false_authority_no_nonrate_score`
- `contest_cpu_cuda_exact_eval_not_executed`
- `receiver_proof_not_executed_for_archive_size_ladder`
