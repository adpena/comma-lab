# HiNeRV archive-size ladder

Schema: `hinerv_archive_size_ladder.v1`
Authority: `false_authority_archive_size_ladder_no_score_claim`
Decoder codec: `int8_mixed`

| row | params | archive bytes | rate score | proof ready |
|---|---:|---:|---:|---:|
| hi_nerv_local_tiny | 94764 | 134924 | 0.089840 | None |
| hi_nerv_local_small | 198873 | 247654 | 0.164903 | None |
| hi_nerv_local_base | 340802 | 398074 | 0.265061 | None |
| hi_nerv_local_wide | 738120 | 812325 | 0.540894 | None |

## Marginal Gates

- `hi_nerv_local_tiny` -> `hi_nerv_local_small` adds `112730` B; requires non-rate drop >= `0.07506227978546239`
- `hi_nerv_local_small` -> `hi_nerv_local_base` adds `150420` B; requires non-rate drop >= `0.10015850372863702`
- `hi_nerv_local_base` -> `hi_nerv_local_wide` adds `414251` B; requires non-rate drop >= `0.2758327371898126`

## Blockers

- `hinerv_archive_size_ladder_false_authority_no_nonrate_score`
- `contest_cpu_cuda_exact_eval_not_executed`
- `receiver_proof_not_executed_for_archive_size_ladder`
