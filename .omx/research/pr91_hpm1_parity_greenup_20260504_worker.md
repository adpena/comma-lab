# PR91 HPM1 Parity Greenup - 2026-05-04

Scope: PR91 HPM1 local decode/replay parity for archive `x` only. No remote GPU
dispatch, no scorer load, no exact-eval submission, and no score claim were
performed.

## Input Custody

- Archive: `experiments/results/public_pr91_intake_20260504_codex/archive.zip`
- Archive bytes: `222404`
- Archive SHA-256:
  `4c16d04c746c981feb902e4dd508ffadaf3615e532d351993c3d2f6eccda1b4f`
- Single member: `x`
- Member bytes: `222304`
- Member SHA-256:
  `5c213c61cc4d29b62286063bfdcb97e812af6b06c0021aeaecc8bc46644e17bf`
- HPM1 mask bytes: `145087`
- HPM1 token stream bytes: `116796`
- HPM1 token stream SHA-256:
  `541016d83852a5bb3e0738caa3b44d7b2b0f7372f1841085cf9554f039c6cf6b`
- HPM1 HPAC model bytes: `28243`
- HPM1 HPAC model SHA-256:
  `de7638c531c9dafa06148602cf784bf3ae9997f326f85cc25b9f3646b536abdd`

## Greenup Attempt

Added a local-only PR91 stream-transform probe:

- `tac.pr91_hpm1_codec.run_pr91_hpm1_stream_transform_probe(...)`
- Focused test:
  `test_real_pr91_stream_transform_probe_rules_out_byte_word_order_if_available`

The probe keeps dispatch locked and tests whether the blocker is a simple
range-coder byte/word contract issue rather than a deeper HPAC probability or
model mismatch.

## Result

Status: fail-closed. PR91 archive `x` is still not locally replayable.

Direct command:

```text
.venv/bin/python - <<'PY'
from tac.pr91_hpm1_codec import DEFAULT_PR91_ARCHIVE, run_pr91_hpm1_stream_transform_probe
report = run_pr91_hpm1_stream_transform_probe(DEFAULT_PR91_ARCHIVE, max_frames=1)
print('status', report['status'])
print('failure_reason', report.get('failure_reason'))
for row in report['transform_results']:
    ctx = row.get('failure_context', {})
    print(row['variant'], row['status'], row.get('failure_stage'), row.get('failure_reason'), ctx.get('failed_at'), ctx.get('decoded_symbol_count_before_failure'))
PY
```

Observed transform evidence:

| Variant | Status | First failure | Decoded symbols before failure |
| --- | --- | --- | ---: |
| `raw_le_u32` | `failed_closed` | `frame=0`, `group=10`, `symbol_in_group=191` | `5951` |
| `word_byteswap` | `failed_closed` | `frame=0`, `group=8`, `symbol_in_group=483` | `4323` |
| `word_reverse` | `failed_closed` | `frame=0`, `group=9`, `symbol_in_group=479` | `5279` |
| `byte_reverse` | `failed_closed` | `frame=0`, `group=15`, `symbol_in_group=135` | `12423` |

Interpretation: the raw source contract remains blocked at
`frame=0/group=10/symbol_in_group=191`, and the simple byte-order or word-order
alternatives do not decode frame 0 either. This narrows the remaining blocker
to an unrecovered HPAC entropy/probability/model contract or invalid submitted
compressed data, not a local HPM1 slice/header parse or uint32 byte-order
mistake.

## Verification

Commands:

```text
.venv/bin/python -m py_compile src/tac/pr91_hpm1_codec.py src/tac/tests/test_pr91_hpm1_codec.py experiments/replay_pr91_hpm1_mask.py
.venv/bin/python -m pytest src/tac/tests/test_pr91_hpm1_codec.py -q
```

Results:

- Py compile: passed.
- Focused PR91 pytest: `11 passed in 48.99s`.
