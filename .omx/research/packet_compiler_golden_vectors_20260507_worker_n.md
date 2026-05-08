# Packet Compiler Golden Vectors - Worker N - 2026-05-07

## Scope

Worker N added a deterministic Python-reference vector generator for
`src/tac/submission_packet_compiler.py` outputs. The goal is cross-language
parity scaffolding for future Rust, Zig, C, or assembly byte-level ports, not a
native rewrite and not a contest score claim.

## Artifacts

- `tools/build_packet_compiler_golden_vectors.py`
  - Emits built-in fixture packets plus JSON vectors under a caller-selected
    output directory.
  - Passes `--target-profile` through to the Python packet compiler for
    target-specific policy metadata while keeping score/dispatch gates false.
  - Normalizes packet paths inside JSON so the same vectors are byte-identical
    across output locations.
  - Pins a canonical stored single-member ZIP fixture to SHA-256
    `7cae837c71aa1abbc55b52dcdb51487a847725bb97cb507d5761ac23c344bf86`.
  - Emits a duplicate-member negative fixture that must fail closed with
    `archive:duplicate_archive_member:x`.
- `src/tac/tests/test_packet_compiler_golden_vectors.py`
  - Verifies deterministic index/vector JSON across separate output dirs.
  - Verifies ZIP header spans, archive SHA-256s, member payload SHA-256s,
    central-directory and EOCD offsets, and charged byte accounting.
  - Verifies duplicate-member fail-closed semantics remain visible to native
    ports.

No generated vector output is checked in by default; callers choose an output
directory and can commit only intentionally promoted vectors.

## Native-Port Contract

Ports should compare these surfaces byte-for-byte:

- `compiler_manifest.archive`
- `compiler_manifest.golden_vectors.member_vectors`
- `zip_header_manifest`
- `zip_header_manifest.charged_byte_accounting`

The reference vectors require matching archive bytes and SHA-256s, member order,
local and central names, offsets, CRCs, compressed/uncompressed byte counts,
payload SHA-256s, duplicate-member blockers, and charged archive/payload byte
accounting.

## Compliance Status

- `score_claim=false`
- `promotion_eligible=false`
- `dispatchable=false`
- `ready_for_exact_eval_dispatch=false`
- Evidence grade: `byte_custody_only`

These vectors are reusable conformance assets. They do not promote, rank, kill,
or dispatch a contest lane.

## Verification

Focused checks run locally:

```text
.venv/bin/python -m pytest src/tac/tests/test_packet_compiler_golden_vectors.py -q
.... 4 passed in 0.15s

.venv/bin/python -m ruff check tools/build_packet_compiler_golden_vectors.py src/tac/tests/test_packet_compiler_golden_vectors.py
All checks passed!
```
