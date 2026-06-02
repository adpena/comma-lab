# HiNeRV archive-size ladder

Schema: `hinerv_archive_size_ladder.v1`
Authority: `false_authority_archive_size_ladder_no_score_claim`
Decoder codec: `int8_mixed`

| row | params | archive bytes | rate score | proof ready |
|---|---:|---:|---:|---:|
| hi_nerv_local_tiny | 94764 | 135011 | 0.089898 | None |
| hi_nerv_local_small | 198873 | 247773 | 0.164982 | None |
| hi_nerv_local_base | 340802 | 397964 | 0.264988 | None |
| hi_nerv_local_wide | 738120 | 812161 | 0.540785 | None |

## Marginal Gates

- `hi_nerv_local_tiny` -> `hi_nerv_local_small` adds `112762` B; requires non-rate drop >= `0.0750835872719623`
- `hi_nerv_local_small` -> `hi_nerv_local_base` adds `150191` B; requires non-rate drop >= `0.10000602202837204`
- `hi_nerv_local_base` -> `hi_nerv_local_wide` adds `414197` B; requires non-rate drop >= `0.275796780806344`

## Blockers

- `hinerv_archive_size_ladder_false_authority_no_nonrate_score`
- `contest_cpu_cuda_exact_eval_not_executed`
- `receiver_proof_not_executed_for_archive_size_ladder`
