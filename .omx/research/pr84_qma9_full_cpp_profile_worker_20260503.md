# PR84 QMA9 Full C++ Profile Worker - 2026-05-03

## Scope

Local-only PR84 QMA9 profiling and profiler hardening. No remote GPU, exact
eval, training, scorer invocation, or lane dispatch was performed.

## Source

- PR84 archive:
  `experiments/results/top_submission_reverse_engineering_20260503_pr84/archive.zip`
- Archive bytes/SHA-256:
  `215735`,
  `a607a6c3ae9b610e6edfb546c3206004ae40fc348ecaef2446b7134a19b8e07f`
- Public inflater constants source:
  `experiments/results/top_submission_reverse_engineering_20260503_pr84/sources/inflate.py`
- Range mask payload bytes/SHA-256:
  `159011`,
  `4b9d93fedb37a9d6fd435054cc33e216d703818b3ac94f4616c89969a4e0d179`

## Tooling Fix

`experiments/profile_qma9_range_mask_bitstream.py` previously rejected the
PR84 mirrored `inflate.py` because PR84 records reordered model split constants
but omits explicit `POSE_STREAM_BYTES` and `PACKED_PAYLOAD_BYTES`. The parser
now accepts only the known public PR84/QMA9 fixed-slice shape:

- `RANGE_MASK_BYTES = 159011`
- `SPLIT_MODEL_REORDERED_BYTES = 55725`
- inferred `POSE_STREAM_BYTES = 899`
- `PACKED_PAYLOAD_BYTES` computed with the declared router bytes.

This prevents ad hoc shim JSON from becoming hidden source-of-truth for PR84
full-stream profiling.

## Full-Stream Profile

Command:

```bash
.venv/bin/python experiments/profile_qma9_range_mask_bitstream.py \
  --archive experiments/results/top_submission_reverse_engineering_20260503_pr84/archive.zip \
  --split-constants-py experiments/results/top_submission_reverse_engineering_20260503_pr84/sources/inflate.py \
  --output-dir experiments/results/qma9_pr84_full_cpp_profile_20260503_worker \
  --output-json experiments/results/qma9_pr84_full_cpp_profile_20260503_worker/qma9_pr84_full_cpp_profile.json \
  --pure-python-max-pixels 1024 \
  --checkpoint-pixels 0,1,1023 \
  --cpp-timeout-seconds 300
```

Artifacts:

- `experiments/results/qma9_pr84_full_cpp_profile_20260503_worker/qma9_pr84_full_cpp_profile.json`
  SHA-256 `989a7e4e265e10fe96084062df79e10c188cb6d4651fc93aeaefb9a4fa9f5717`
- `experiments/results/qma9_pr84_full_cpp_profile_20260503_worker/qma9_range_mask_cpp_full_profile.json`
  SHA-256 `6d05adc72d0cff20d2eae39814b5c99575967064ad842bb9a766ac0d5c2c8527`

Key profile facts:

- Actual bitstream bits: `1271928`; estimated model bits: `1271926.81158`.
- Predictor counts: up `117394995`, left `409373`, prev `82850`,
  class fallback `77582`.
- Top frame indices by estimated bytes: `517`, `522`, `519`, `70`, `518`.
- Top row indices by estimated bytes: `0`, `288`, `320`, `296`, `324`.
- Horizontal runs length >=16: `1046026`; long-run tail lower bound:
  `57999862` pixels.

## Candidate Interpretation

The full profile keeps `qma9_horizontal_run_escape_len16` as the highest
planning signal, but it is not dispatchable yet. It needs a concrete encoder
and runtime, full raw-mask parity, archive closure, and exact CUDA auth eval.

The extra fallback-predictor gates are negative planning signals as currently
formulated: every one has rough negative net bytes after a binary gate. Do not
spend eval queue on those without a redesigned coding scheme.

## Verification

```bash
.venv/bin/python -m py_compile \
  experiments/profile_qma9_range_mask_bitstream.py \
  src/tac/tests/test_profile_qma9_range_mask_bitstream.py

.venv/bin/python -m pytest \
  src/tac/tests/test_profile_qma9_range_mask_bitstream.py -q
```

Result: `2 passed`.
