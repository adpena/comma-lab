# PR106 Context-View Identity Hardening - 2026-05-16

## Scope

Closed the PacketIR custody gap in `tac.packet_compiler.pr106_context_recode`.
The context-recode profiler unwraps PR106/R2 sidecar packets to inspect inner
HNeRV sections; it now has an explicit fail-closed identity proof that the
context view re-emits both:

- the inner PR106 payload byte-for-byte; and
- the original wrapper payload / single-member ZIP byte-for-byte.

This is parser-custody evidence only. It does not claim runtime consumption,
full-frame inflate parity, contest-axis score movement, promotion eligibility,
or dispatch readiness.

## Landed

- `emit_pr106_context_source_payload(...)`
- `prove_pr106_context_source_identity(...)`
- `prove_pr106_context_archive_identity(...)`
- release-archive tests for:
  - `submissions/pr106_latent_sidecar_r2/archive.zip`
  - `submissions/pr106_latent_sidecar_r2_pr101_grammar/archive.zip`
- expected archive SHA mismatch fail-closed regression.

## Evidence Axis

`packet-ir-context-parser-local-no-score`

## Verification

```bash
.venv/bin/python -m pytest tests/test_pr106_context_recode.py -q
.venv/bin/python -m pytest src/tac/tests/test_packet_compiler_pr106_sidecar_packet.py::test_pr106_sidecar_packet_ir_identity_proof_is_operator_facing_and_nonpromotable -q
.venv/bin/ruff check src/tac/packet_compiler/pr106_context_recode.py src/tac/packet_compiler/__init__.py tests/test_pr106_context_recode.py
```

Results:

- `8 passed`
- `2 passed`
- `All checks passed`

## Remaining Blockers

- Context-recode prototype sections remain non-promotional until a runtime
  decoder consumes them.
- Same-runtime full-frame parity and exact `[contest-CPU]` / `[contest-CUDA]`
  eval remain required before score language.
