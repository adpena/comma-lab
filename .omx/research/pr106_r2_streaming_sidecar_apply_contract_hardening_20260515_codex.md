# PR106/R2 Streaming Sidecar Apply-Contract Hardening

Date: 2026-05-15

## Scope

This ledger records a narrowly scoped PR106/R2 PacketIR hardening pass. It does
not claim a contest score, promotion, or public-frontier movement.

## Finding

`runtime_sidecar_correction_digest()` already treated
`inflate.py::apply_sidecar_corrections()` as value-producing by hashing the
returned corrected latent tensor. `runtime_full_frame_streaming_digest()` did
not: it called `apply_sidecar_corrections(latents, ...)` and then rendered the
original `latents` binding. That is correct only for runtimes whose correction
function mutates in place. A runtime that returns a corrected clone could pass
decode-level consumption checks while the streaming full-frame digest renders
uncorrected latents.

Classification: no-op/consumption-proof harness bug class.

## Fix

The runtime proof module now uses one helper for both paths:

- `src/tac/packet_compiler/pr106_runtime_consumption.py`
  - `_apply_runtime_sidecar_corrections(...)`
  - honors both in-place and returned-tensor correction contracts;
  - exposes `source_latents_sha256`, `corrected_latents_sha256`, and
    `latents_changed_by_sidecar` from the streaming digest.

The regression test adds a fake submission runtime whose
`apply_sidecar_corrections()` returns a corrected clone without mutating the
input. The streaming digest must now bind and render the returned corrected
latents.

## Verification

Commands run locally:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q \
  src/tac/tests/test_packet_compiler_pr106_runtime_consumption.py \
  src/tac/tests/test_packet_compiler_pr106_sidecar_packet.py

.venv/bin/python -m ruff check \
  src/tac/packet_compiler/pr106_runtime_consumption.py \
  src/tac/tests/test_packet_compiler_pr106_runtime_consumption.py

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile \
  src/tac/packet_compiler/pr106_runtime_consumption.py \
  src/tac/tests/test_packet_compiler_pr106_runtime_consumption.py

git diff --check -- \
  src/tac/packet_compiler/pr106_runtime_consumption.py \
  src/tac/tests/test_packet_compiler_pr106_runtime_consumption.py
```

Observed: 63 pytest cases passed; ruff, py_compile, and diff whitespace checks
passed.

## Evidence Boundary

This is a stronger local proof surface for runtime sidecar consumption. It is
not an auth-eval result. PR106/R2 remains blocked from any score/promotion claim
until byte-closed same-runtime exact eval artifacts exist with explicit
`[contest-CPU]` and/or `[contest-CUDA]` axis labels, archive/runtime custody,
and component recomputation.
