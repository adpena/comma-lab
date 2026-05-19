# Codex Session Summary - TAC Terminology And Weight-Derived Authority

Timestamp: 2026-05-19T12:13:42Z
Author: Codex
Status: implemented locally, pending serializer commit/push at write time

## Decision

`tac` remains **Task-Aware Compression**, not Task-Aware Codec.
That matches the active repo contract and the external jargon: task-aware /
task-oriented compression in the research literature, and coding for machines,
video coding for machines, and feature coding for machines in standards and
industry language. `codec` remains the narrower word for concrete
encoder/decoder, entropy-coder, archive-grammar, or wire-format pieces inside
the broader compression stack.

Procedural generation from a seed or from weights is a first-class contest path
when the information is byte-closed. The canonical promotion modes are:

- `archive_seeded`: seed/table/transducer bytes live in `archive.zip`.
- `weight_derived`: seed/codebook is derived from an existing charged archive
  member such as renderer weights, with source-member SHA and no-new-bytes
  proof.
- `runtime_constant`: only generic decoder logic or negligible implementation
  constants live in `inflate.py`; per-video literal seeds stay research/probe
  unless a compliance ruling proves they are code rather than relocated
  payload.

## Implementation

- Hardened `docs/contest_compliance_authority.md` from a two-mode seed framing
  to the three-mode `archive_seeded` / `weight_derived` / `runtime_constant`
  authority ladder.
- Updated `docs/terminology_and_boundaries.md`, `src/tac/README.md`, and the
  terminology guard so public docs must preserve weight-derived procedural
  authority.
- Updated `tac.procedural_codebook_generator` packet wording and added a
  regression test proving default authority packets expose the
  `weight_derived` promotion path when proofs and exact eval are present.
- Fixed native metadata drift in `runtime-rs`: the packet compiler crate now
  points to `https://github.com/adpena/comma-lab`, and README text states that
  Rust is a speed layer for `tac` Task-Aware Compression primitives with Python
  golden vectors as oracle.

## Verification

- `.venv/bin/python tools/check_tac_terminology.py --strict`
- `.venv/bin/pytest src/tac/tests/test_tac_terminology_guard.py src/tac/tests/test_procedural_codebook_generator.py`
- `.venv/bin/ruff check src/tac/procedural_codebook_generator/authority.py tools/check_tac_terminology.py src/tac/tests/test_tac_terminology_guard.py src/tac/tests/test_procedural_codebook_generator.py`
- `git diff --check`
- `cargo metadata --manifest-path runtime-rs/crates/tac-packet-compiler/Cargo.toml --no-deps --format-version 1`

## Authority Notes

External terminology references used for the naming decision:

- MPEG-AI Part 2: Video coding for machines.
- MPEG-AI Part 4: Feature coding for machines.
- CVPR 2023 AccelIR: Task-aware Image Compression.
- NeurIPS 2023: Task-aware Distributed Source Coding.
- NVIDIA Research CVPR 2025: Task-Aware Video Compression.
