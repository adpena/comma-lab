# HiNeRV archive backend drift

Schema: `hinerv_archive_backend_drift.v1`
Authority: `false_authority_local_backend_byte_drift_no_scorer_claim`
Reference: `pytorch_portable_fallback`
Candidate: `mlx_metal`
Local dev velocity ready: `True`
Max abs byte drift: `81`
Sum rate-score drift: `-5.59321520622624e-05`

| row | ref bytes | cand bytes | delta | abs delta | rate-score delta | cand proof |
|---|---:|---:|---:|---:|---:|---:|
| hi_nerv_local_base | 398071 | 398074 | 3 | 3 | 1.9975768593665143e-06 | True |
| hi_nerv_local_small | 247755 | 247815 | 60 | 60 | 3.995153718733028e-05 | True |
| hi_nerv_local_tiny | 134908 | 134842 | -66 | 66 | -4.3946690906063314e-05 | True |
| hi_nerv_local_wide | 812333 | 812252 | -81 | 81 | -5.393457520289588e-05 | True |

## Blockers

- `contest_cpu_cuda_exact_eval_not_executed`
- `hinerv_archive_backend_drift_false_authority_no_nonrate_score`
- `hinerv_archive_backend_drift_not_promotion_or_rank_authority`
- `hinerv_archive_backend_drift_local_dev_velocity_only`
