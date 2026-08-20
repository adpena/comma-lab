# SNeRV LF payload codec sweep

Schema: `snerv_lf_payload_codec_sweep.v1`
Authority: `false_authority_lf_payload_codec_rate_only`
Baseline mode: `int64_lzma`
Baseline payload bytes: `551`

| mode | payload bytes | byte delta vs baseline | economic decision | final decision |
|---|---:|---:|---|---|
| int64_lzma | 551 | 0 | protect | demote |
| delta_varint | 1472 | 921 | protect | demote |
| portfolio_auto | 1472 | 921 | protect | demote |
| uint8 | 1648 | 1097 | protect | demote |
| int8 | 1696 | 1145 | protect | demote |
| int4 | 1704 | 1153 | protect | demote |
| uint4 | 1704 | 1153 | protect | demote |
| uint8_escape | 1728 | 1177 | protect | demote |
| int8_escape | 1800 | 1249 | protect | demote |
| zero_run | 1824 | 1273 | protect | demote |
| uint4_escape | 1968 | 1417 | protect | demote |
| int4_escape | 1972 | 1421 | protect | demote |
| int2_escape | 2144 | 1593 | protect | demote |
| uint2_escape | 2188 | 1637 | protect | demote |
| int2 | 0 | n/a | demote | demote |
| uint2 | 0 | n/a | demote | demote |

## Blockers

- `snerv_lf_payload_codec_sweep_false_authority_no_scorer_replay`
- `contest_cpu_cuda_exact_eval_not_executed`
