# HiNeRV archive-size ladder

Schema: `hinerv_archive_size_ladder.v1`
Authority: `false_authority_archive_size_ladder_no_score_claim`
Decoder codec: `int8_mixed`

| row | params | archive bytes | rate score | proof ready |
|---|---:|---:|---:|---:|
| hi_nerv_local_tiny | 94764 | 138417 | 0.092166 | None |
| hi_nerv_local_small | 198873 | 250984 | 0.167120 | None |
| hi_nerv_local_base | 340802 | 401453 | 0.267311 | None |
| hi_nerv_local_wide | 738120 | 815573 | 0.543057 | None |

## Marginal Gates

- `hi_nerv_local_tiny` -> `hi_nerv_local_small` adds `112567` B; requires non-rate drop >= `0.07495374477610346`
- `hi_nerv_local_small` -> `hi_nerv_local_base` adds `150469` B; requires non-rate drop >= `0.10019113081734`
- `hi_nerv_local_base` -> `hi_nerv_local_wide` adds `414120` B; requires non-rate drop >= `0.2757455096669536`

## Blockers

- `hinerv_archive_size_ladder_false_authority_no_nonrate_score`
- `contest_cpu_cuda_exact_eval_not_executed`
- `receiver_proof_not_executed_for_archive_size_ladder`
