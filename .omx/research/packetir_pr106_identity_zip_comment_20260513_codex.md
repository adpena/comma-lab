# PR106 PacketIR Identity ZIP Comment Contract - 2026-05-13

## Scope

Lane: Compiler stack / PacketIR score-lowering lane.

Objective: make PR106/R2 PacketIR identity parse/re-emit more byte-closed before
HDM5-style sidecar, arithmetic/range/ANS, or low-level repack work depends on
the identity proof.

Remote dispatch: none. Score claim: none. Trainer files: untouched.

## Finding

`tac.packet_compiler.pr106_sidecar_packet.StoredZipMember` preserved the single
ZIP member name, payload, timestamp, external attributes, create system, flags,
member comment, and extra bytes, but did not preserve the archive-level ZIP
comment. A valid single-member archive with a central-directory comment could
therefore fail byte-for-byte identity re-emit even though the PacketIR payload
parse was correct.

Failure class: exact packet contract under-specified central-directory metadata.

## Landing

- Added `archive_comment` to `StoredZipMember`.
- `read_single_stored_member_archive()` now records `ZipFile.comment`.
- `emit_single_stored_member_archive()` now restores the archive-level comment.
- `prove_pr106_sidecar_packet_ir_identity()` now reports comment byte count and
  SHA-256 in the archive manifest.
- Added regression tests for archive comment preservation and identity proof on
  a commented single-member PR106-style packet.

## Verification

```bash
.venv/bin/python -m pytest \
  src/tac/tests/test_packet_compiler_pr106_sidecar_packet.py \
  src/tac/tests/test_prove_pr106_packetir_identity_tool.py
```

Result: 20 passed in 0.24s.

## Wire-In

Sensitivity-map contribution: N/A; byte-custody primitive.

Pareto constraint: non-binding; no candidate bytes or score movement claimed.

Bit-allocator hook: N/A; no allocation decision.

Cathedral autopilot dispatch hook: `ready_for_exact_eval_dispatch=false` remains
part of the identity manifest.

Continual-learning posterior update: N/A; no empirical score anchor.

Probe-disambiguator: N/A; single exact metadata preservation bug.

## Next Proof Before Promotion Language

Any sidecar or entropy transform candidate still requires runtime sidecar
decode/apply proof, full-frame same-runtime parity, and exact contest auth eval
with explicit `[contest-CUDA]` / `[contest-CPU]` axis labels before score,
promotion, or dispatch-readiness language.
