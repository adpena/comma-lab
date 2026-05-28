# Codex Findings - FP11 Selector Adapter Expansion

UTC: 2026-05-28T12:22Z

## Result

The repair archive adapter stack no longer treats FEC6 as the only semantic
selector payload. I expanded the queue-owned byte-transform executor to mutate
and re-encode FP11 selector payloads for FES1, FEC3, FEC5, FEC6, and FEC8 while
keeping all local rows advisory and exact-axis fail-closed.

## Implemented

- FES1 bit-packed selector decode/mutate/re-encode, preserving archive-declared
  palette size.
- FEC3 compact selector decode/mutate/re-encode, preserving static/dynamic
  compact palette tables.
- FEC5 fixed-Huffman K8 decode/mutate/re-encode.
- FEC8 static/adaptive/static-second-order decode/mutate/re-encode through the
  existing Markov codec implementation.
- Candidate ranking now prioritizes non-authority semantic payload-change
  evidence instead of the false-authority `score_affecting_payload_changed`
  field, which is intentionally forced false.
- Archive-family coverage reports now emit implemented adapter rows plus an
  automation-ready unsupported-family gap queue with entropy position and
  optimization scopes.

## Live Evidence

- Coverage artifact:
  `.omx/research/repair_archive_family_coverage_20260528T121454Z.json`
- Live adapter proof:
  `.omx/research/fp11_selector_adapter_live_proof_20260528T1221Z/summary.json`

Live selector proof emitted semantic byte-closed candidates with receiver proof
for FEC3, FEC5, FEC8, and FES1. All rows preserve `score_claim=false`,
`promotion_eligible=false`, and `ready_for_exact_eval_dispatch=false`.

Remaining unsupported score-affecting families in the representative census:
PACT-NeRV PSV4 packet, HNeRV HDM latent sidecar, renderer ASYM, renderer DFL1,
and renderer RPK1.

## Verification

- `.venv/bin/python -m pytest src/tac/tests/test_archive_family_fingerprint.py src/tac/tests/test_repair_family_materializers.py::test_byte_transform_executor_mutates_fec6_selector_payload_when_detected src/tac/tests/test_repair_family_materializers.py::test_byte_transform_executor_mutates_non_fec6_fp11_selector_payloads src/tac/tests/test_repair_family_materializers.py::test_byte_transform_executor_mutates_fec8_fp11_selector_payload -q`
  - 8 passed
- `.venv/bin/python -m ruff check src/tac/optimization/archive_family_fingerprint.py src/tac/optimization/repair_family_byte_transform_executor.py tools/inspect_repair_archive_family_coverage.py src/tac/tests/test_archive_family_fingerprint.py src/tac/tests/test_repair_family_materializers.py`
  - clean
