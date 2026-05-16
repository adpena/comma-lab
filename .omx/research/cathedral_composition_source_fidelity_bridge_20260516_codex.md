# Cathedral Composition Source-Fidelity Bridge - 2026-05-16

## Summary

Read-only L5/Cathedral audit found that `SubstrateRow` already carries
literature source-fidelity fields, but the composition-ranking JSON bridge and
Cathedral autopilot candidate loader mostly reduced them to prose or dropped
them entirely. That weakens the paper-fidelity boundary: a `literature_anchor`
can influence ranking while the actual scope limitations and Pact proof
obligations are not structured fields on the candidate.

## Fix

- `tools/build_composition_ranking_json.py` now emits:
  - `source_supports`
  - `paper_claim_scope`
  - `pact_must_prove`
  - `decode_complexity_evidence`
  - `source_fidelity_metadata`
- `tools/cathedral_autopilot_autonomous_loop.py` now loads those fields into
  `CandidateRow` and renders them into candidate notes for operator review.

This preserves planning-only behavior. It does not make literature-backed rows
score claims, promotion candidates, or exact-dispatch-ready packets.

## Verification

- `src/tac/tests/test_build_composition_ranking_json.py` asserts Z3 bridge rows
  carry the four source-fidelity fields structurally and in notes.
- `src/tac/tests/test_cathedral_autopilot_substrate_composition_wire.py` asserts
  the Cathedral loader preserves the same fields through `CandidateRow`.

## Reactivation Criteria

Reopen if any future ranking bridge, autopilot queue loader, or campaign ledger
lets a `literature_anchor` influence class-shift ranking without carrying
structured source scope, proof obligations, and decode-complexity evidence.
