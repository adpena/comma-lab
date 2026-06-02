# SNeRV LF payload codec sweep

Schema: `snerv_lf_payload_codec_sweep.v1`
Authority: `false_authority_lf_payload_codec_rate_only`
Baseline mode: `int64_lzma`
Baseline payload bytes: `666556`

| mode | payload bytes | byte delta vs baseline | economic decision | final decision |
|---|---:|---:|---|---|
| int64_lzma | 666556 | 0 | protect | demote |
| portfolio_auto | 787120 | 120564 | protect | demote |
| int8_escape | 787638 | 121082 | protect | demote |
| int4_escape | 803384 | 136828 | protect | demote |
| int2_escape | 806417 | 139861 | protect | demote |
| zero_run | 830901 | 164345 | protect | demote |
| delta_varint | 838534 | 171978 | protect | demote |
| int2 | 0 | n/a | demote | demote |
| int4 | 0 | n/a | demote | demote |
| int8 | 0 | n/a | demote | demote |

## Blockers

- `snerv_lf_payload_codec_sweep_false_authority_no_scorer_replay`
- `contest_cpu_cuda_exact_eval_not_executed`
