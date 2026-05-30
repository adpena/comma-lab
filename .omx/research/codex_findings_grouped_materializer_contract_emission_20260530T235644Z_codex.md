# Codex Findings: Grouped Materializer Contract Emission

- UTC: 2026-05-30T23:56:44Z
- Commit target: main
- Scope: `tools/run_grouped_family_agnostic_materializer.py` grouped archive-state chain manifests.

## Finding

The grouped family-agnostic materializer emitted a final byte-closed archive
manifest with flat `candidate_archive`, `candidate_archive_sha256`, and
`candidate_archive_bytes` fields, but no shared
`archive_bound_candidate_contract`. That kept grouped archive-state chains as a
family-specific readiness shape even after the candidate queue and field-meta
selector migrated to contract-first intake.

## Fix

Grouped chain manifests now attach the canonical archive-bound candidate
contract. The contract is derived from:

- final candidate archive custody;
- source archive custody;
- final step runtime proof path;
- grouped receiver/materialization blockers;
- false-authority dispatch and score fields.

The resulting contract is ready for acquisition and exact-handoff planning but
still has `ready_for_exact_eval_dispatch=false`.

## Verification

- `.venv/bin/python -m ruff check tools/run_grouped_family_agnostic_materializer.py src/tac/tests/test_byte_shaving_campaign_queue.py`
- `.venv/bin/python -m pytest src/tac/tests/test_byte_shaving_campaign_queue.py::test_grouped_family_agnostic_materializer_chains_archive_state -q`
- Result: 1 passed.

## Existing Broader Failure

A full `src/tac/tests/test_byte_shaving_campaign_queue.py -q` run currently
fails on unrelated registry/postcondition expectations in current `main`. This
slice did not alter those surfaces; the grouped-chain test covering this emitter
passes.
