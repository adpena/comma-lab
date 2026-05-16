# PacketIR Exact Closure Axis-Semantics Hardening - 2026-05-16

## Scope

Narrow PR106/R2 PacketIR custody hardening. No score claim, no dispatch, and no
promotion change.

## Finding

`tac.packetir_exact_closure.build_packetir_exact_closure()` already joined
PacketIR byte custody, runtime-consumption proofs, full-frame same-runtime
parity, and exact auth-eval rows. The auth-eval claim parser enforced
score-axis labels and formula closure, but the PacketIR closure boundary did
not surface a dedicated blocker when a CUDA/CPU row lacked explicit device
semantics or full 600-sample contest coverage.

Failure class: a malformed closure input could look axis-labelled while missing
the concrete sample/device semantics needed for apples-to-apples CPU/CUDA
evidence.

## Fix

`src/tac/packetir_exact_closure.py` now records `eval_device`, `hardware`, and
`axis_semantics_blockers` in each eval summary. The closure adds a
`cuda_eval_axis_semantics_are_contest_cuda` check and folds CPU axis-semantics
blockers into `cpu_eval_is_axis_labeled_diagnostic_not_cuda_claim`.

Contest CUDA rows must carry:

- `score_axis=contest_cuda`;
- `n_samples=600`;
- CUDA eval/scorer device semantics;
- CUDA/GPU/T4-class hardware semantics;
- CUDA evidence grade.

Contest CPU diagnostic rows must carry:

- `score_axis=contest_cpu`;
- `n_samples=600`;
- CPU eval/scorer device semantics;
- CPU evidence grade.

## Verification

```bash
.venv/bin/python -m pytest src/tac/tests/test_packetir_exact_closure.py -q
# 27 passed

.venv/bin/python -m ruff check \
  src/tac/packetir_exact_closure.py \
  src/tac/tests/test_packetir_exact_closure.py
# All checks passed

.venv/bin/python -m py_compile \
  src/tac/packetir_exact_closure.py \
  src/tac/tests/test_packetir_exact_closure.py
```

Regression tests cover:

- CUDA claim with CPU device/hardware semantics;
- partial-sample CUDA row;
- CPU-axis diagnostic row carrying CUDA device semantics.

## Evidence Boundary

This hardens closure review only. It does not create a new PR106/R2 candidate
or change the measured format `0x04` CUDA canary. Future PacketIR candidates
still need byte-closed archive/runtime custody, consumed-byte proof,
same-runtime full-frame parity, and exact `[contest-CUDA]` eval before score or
promotion language.
