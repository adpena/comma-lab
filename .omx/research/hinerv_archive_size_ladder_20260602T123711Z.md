# HiNeRV archive-size ladder

Schema: `hinerv_archive_size_ladder.v1`
Authority: `false_authority_archive_size_ladder_no_score_claim`
Decoder codec: `int8_mixed`

| row | params | archive bytes | rate score | proof ready |
|---|---:|---:|---:|---:|
| hi_nerv_local_tiny | 94764 | 135056 | 0.089928 | None |
| hi_nerv_local_small | 198873 | 247785 | 0.164990 | None |
| hi_nerv_local_base | 340802 | 397988 | 0.265004 | None |
| hi_nerv_local_wide | 738120 | 812302 | 0.540879 | None |

## Marginal Gates

- `hi_nerv_local_tiny` -> `hi_nerv_local_small` adds `112729` B; requires non-rate drop >= `0.07506161392650926`
- `hi_nerv_local_small` -> `hi_nerv_local_base` adds `150203` B; requires non-rate drop >= `0.1000140123358095`
- `hi_nerv_local_base` -> `hi_nerv_local_wide` adds `414314` B; requires non-rate drop >= `0.2758746863038593`

## Blockers

- `hinerv_archive_size_ladder_false_authority_no_nonrate_score`
- `contest_cpu_cuda_exact_eval_not_executed`
- `receiver_proof_not_executed_for_archive_size_ladder`
