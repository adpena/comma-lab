# Codex Findings: JSONL Custody Observer False-Authority Closure

UTC: 2026-05-27T17:43:57Z

## Summary

Closed one remaining false-authority propagation path in the experiment queue
observer. JSON object artifacts already revalidated archive/runtime/receiver
custody before allowing a postcondition to remain passed, but
`jsonl_false_authority` rows only checked row schema and false-authority fields.
That left room for a JSONL row to carry `proof_present=true`,
`receiver_contract_satisfied=true`, and a candidate archive pointer without an
actual proof file, while the enclosing artifact still reported
`postcondition_passed=true`.

The observer now revalidates generic custody claims inside JSONL rows. If a row
contains archive/runtime/receiver custody fields and is not already a
materializer-contract row, it must pass the same archive file, proof file,
receiver proof, runtime identity, and false-authority checks as JSON object
artifacts.

## Landed Integration

- `src/comma_lab/scheduler/experiment_queue_observer.py`
  - `_jsonl_false_authority_revalidation(...)` now routes non-materializer
    custody rows through `_generic_custody_payload_revalidation(...)`.
  - Any blocker is prefixed by row index and forces the artifact
    `postcondition_passed=false`.
- `src/tac/tests/test_experiment_queue_observer.py`
  - Added a regression where a JSONL custody row carries only
    `proof_present=true` plus candidate archive custody and no proof path.
  - The observer rejects the artifact with
    `row_1:jsonl_custody_runtime_or_receiver_proof_path_missing`.

## Verification

- `.venv/bin/python -m ruff check src/comma_lab/scheduler/experiment_queue_observer.py src/tac/tests/test_experiment_queue_observer.py`
- `.venv/bin/python -m py_compile src/comma_lab/scheduler/experiment_queue_observer.py src/tac/tests/test_experiment_queue_observer.py`
- `.venv/bin/python -m pytest src/tac/tests/test_experiment_queue_observer.py -q`
  - `28 passed`
- `.venv/bin/python -m pytest src/tac/tests/test_materializer_chain_harvest_scheduler.py::test_harvest_family_agnostic_declaration_only_proof_is_not_receiver_ready src/tac/tests/test_materializer_chain_harvest_scheduler.py::test_harvest_rejects_chain_manifest_stale_runtime_tree_identity -q`
  - `2 passed`
- `.venv/bin/python tools/lane_maturity.py validate`
  - `1440 lane(s) validated cleanly`
- `.venv/bin/python tools/review_gate_hook.py`
  - passed

## Remaining Scope

This closes the JSONL observer leak for Week 1. The broader goal remains active:
repair waterfill must become executable default scoring, repair families must be
expanded and stacked, and autopilot must close byte-closed exact-readiness and
continual-learning loops without manual queue babysitting.
