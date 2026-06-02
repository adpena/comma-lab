# SNeRV LF payload codec sweep

Schema: `snerv_lf_payload_codec_sweep.v1`
Authority: `false_authority_lf_payload_codec_rate_only`
Baseline mode: `int64_lzma`
Baseline payload bytes: `98510`

| mode | payload bytes | byte delta vs baseline | economic decision | final decision |
|---|---:|---:|---|---|
| int64_lzma | 98510 | 0 | protect | demote |
| portfolio_auto | 101063 | 2553 | protect | demote |
| int8_escape | 101165 | 2655 | protect | demote |
| int4_escape | 103108 | 4598 | protect | demote |
| int2_escape | 103389 | 4879 | protect | demote |
| zero_run | 106673 | 8163 | protect | demote |
| delta_varint | 107171 | 8661 | protect | demote |
| int2 | 0 | n/a | demote | demote |
| int4 | 0 | n/a | demote | demote |
| int8 | 0 | n/a | demote | demote |

## Blockers

- `snerv_lf_payload_codec_sweep_false_authority_no_scorer_replay`
- `contest_cpu_cuda_exact_eval_not_executed`
- `snerv_lf_payload_codec_row_has_no_scorer_replay`
- `snerv_lf_payload_codec_row_not_exact_eval_authority`
- `snerv_lf_payload_codec_mode_failed`
