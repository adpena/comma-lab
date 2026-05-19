# Codex Session Summary - TAC Naming Stale Repo Link Guard

Timestamp: 2026-05-19T11:30:59Z
Author: Codex
Status: implemented locally, pending serializer commit/push at write time

## Decision

`tac` remains canonically expanded as **Task-Aware Compression**. This is the
right field-level name because the package owns the full downstream-task
compression program: scorer-aware objectives, packet compilation, procedural
byte derivation, archive custody, master-gradient consumers, and exact-eval
contracts. `codec` remains a narrower implementation word for concrete
encoders, decoders, entropy coders, packet grammars, and inflate/archive pairs.

External terminology maps this surface to task-aware/task-oriented compression,
video coding for machines, feature coding for machines, and coding for
machines. `TAC` is repository-local shorthand, not a standards-body acronym.

## Implementation

- Preserved the existing canonical README/package framing in `README.md`,
  `src/tac/README.md`, `src/comma_lab/README.md`, and
  `docs/terminology_and_boundaries.md`.
- Replaced stale public paper references to `https://github.com/adpena/tac`
  with relative links to the canonical `src/tac/README.md` package docs.
- Hardened `tools/check_tac_terminology.py` so public docs fail if the stale
  tac-only repository coordinate is reintroduced.
- Extended `src/tac/tests/test_tac_terminology_guard.py` with a regression case
  for the stale-link pattern.

## Verification

- `.venv/bin/python tools/check_tac_terminology.py --strict --json`
- `.venv/bin/python -m pytest src/tac/tests/test_tac_terminology_guard.py -q`

