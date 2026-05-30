# Codex Findings: Source-Contract Candidate Queue Closure

- UTC: 2026-05-30T23:35:44Z
- Commit target: main
- Scope: `archive_bound_candidate_contract`, candidate queue promotion views, exact dispatch authority, exact-ready audit, and parallel dispatch fanout.

## Finding

Exact-ready promoted rows can legitimately carry raw
`ready_for_exact_eval_dispatch=true` after the exact-readiness gate, while their
upstream archive-bound materializer contract remains false-authority by design.
Before this pass, candidate queue views did not consume
`source_archive_bound_candidate_contract` as the contract surface, so a promoted
row with only the source snapshot could fall back to raw readiness/custody
interpretation.

## Fix

The shared TAC archive-bound contract module now owns source snapshot helpers:

- source contract extraction;
- contract candidate-archive custody extraction;
- snapshot blocker validation for schema, archive-bound readiness, exact-handoff
  readiness, custody completeness, SHA mismatch, and byte mismatch.

Candidate queue verification and dispatch-ready counting now use the source
snapshot before raw fields. Exact dispatch authority, exact-ready audit, and
parallel dispatch fanout all read candidate archive custody through the same
contract helper instead of carrying duplicate local readers.

## Review

Three adversarial review passes found no remaining issue in this slice:

1. False-authority pass: source snapshots do not grant score or dispatch
   authority by themselves; raw exact-ready bits are accepted only when the
   source contract proves archive-bound readiness and custody.
2. Custody pass: exact-ready rows missing duplicate archive fields can still be
   consumed from the source contract; mismatched SHA/bytes fail closed.
3. Duplication pass: duplicate source-contract readers were removed from
   candidate queue consumers, exact-ready audit, exact dispatch authority, and
   parallel dispatch fanout.

## Verification

- `.venv/bin/python -m ruff check src/tac/optimization/archive_bound_candidate_contract.py src/tac/optimizer/candidate_queue.py src/tac/optimizer/exact_dispatch_authority.py src/tac/optimizer/exact_ready_audit.py tools/parallel_dispatch_top_k.py src/tac/tests/test_optimizer_candidate_queue.py`
- `.venv/bin/python -m pytest src/tac/tests/test_optimizer_candidate_queue.py src/tac/tests/test_exact_dispatch_authority.py src/tac/tests/test_optimizer_exact_ready_audit.py src/tac/tests/test_dispatch_command_builder_shapes.py src/tac/tests/test_optimizer_exact_readiness.py -q`
- Result: 203 passed.

