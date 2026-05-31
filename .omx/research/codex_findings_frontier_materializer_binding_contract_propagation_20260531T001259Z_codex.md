# Codex Findings: Frontier Materializer Binding Contract Propagation

UTC: 2026-05-31T00:12:59Z

## Verdict

The repair-budget materializer binding path now carries the shared
`tac_archive_bound_candidate_contract.v1` surface from materializer manifests
into binding rows and materialization execution rows.

This closes the flat-field orphan identified in
`_materializer_manifest_record()`: verified archive bytes and receiver proof no
longer have to be reinterpreted from duplicate `candidate_archive_*` fields by
downstream consumers.

## Landed Surfaces

- `src/comma_lab/scheduler/frontier_rate_attack_feedback.py`
  - `_materializer_manifest_record()` emits the shared archive-bound candidate
    contract from validated archive path/SHA/bytes plus receiver proof
    revalidation.
  - `_repair_budget_materializer_binding_row()` propagates that contract into
    queue-owned binding rows.
  - `_repair_budget_materialization_execution_row()` carries the same contract
    into execution audit rows so the archive-bound surface remains available
    after binding.

- `src/tac/tests/test_frontier_rate_attack_feedback.py`
  - Adds a byte-present archive and receiver-proof-backed materializer manifest
    regression.
  - Asserts the binding and execution rows preserve candidate archive SHA
    custody through the shared contract and keep exact/score authority false.

## Validation

- `.venv/bin/python -m ruff check src/comma_lab/scheduler/frontier_rate_attack_feedback.py src/tac/tests/test_frontier_rate_attack_feedback.py`
  - Passed after import ordering fix.

- `.venv/bin/python -m pytest src/tac/tests/test_frontier_rate_attack_feedback.py::test_repair_materializer_binding_propagates_archive_bound_contract -q`
  - Passed.

- `.venv/bin/python -m pytest src/tac/tests/test_frontier_rate_attack_feedback.py -q`
  - Passed: 58 tests.

- `git diff --check -- src/comma_lab/scheduler/frontier_rate_attack_feedback.py src/tac/tests/test_frontier_rate_attack_feedback.py`
  - Passed.

- `tools/review_tracker.py policy-check ...`
  - Passed: 0 violations on both touched files.

## Authority Boundary

The propagated contracts remain false-authority:

- `score_claim=false`
- `promotion_eligible=false`
- `rank_or_kill_eligible=false`
- `ready_for_exact_eval_dispatch=false`

This is usable for acquisition, materializer binding, execution audit, and
exact-handoff planning only.

## Next Remaining Orphan

`src/comma_lab/scheduler/byte_shaving_campaign_queue.py::_materialize_row()`
still records source-unit archive SHA/bytes without the shared contract. The
next narrow patch should either propagate existing unit contracts or emit an
explicit missing-contract blocker when source-unit path/SHA/byte custody is
incomplete.
