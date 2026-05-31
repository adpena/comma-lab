# Codex Findings: Byte-Shaving Source-Unit Contract Custody

UTC: 2026-05-31T00:18:26Z

## Verdict

Byte-shaving source units no longer drop archive custody signal on the floor.
The planner now preserves candidate archive path/SHA/bytes and any existing
archive-bound contract fields, and the DQS1 materialization compiler turns
fully custodied source-unit archive signal into the shared
`tac_archive_bound_candidate_contract.v1` surface.

When a source unit carries only partial archive signal, the compiler emits an
explicit `archive_bound_candidate_contract_blockers` list on that source-unit
metadata instead of leaving orphaned `candidate_archive_*` facts as silent
authority.

## Landed Surfaces

- `src/tac/optimization/byte_shaving_campaign.py`
  - Preserves `candidate_archive_path`, nested `candidate_archive` custody, and
    existing `archive_bound_candidate_contract*` fields through normalized
    ranked units.
  - Applies the same preservation to candidate-queue intake rows before they
    become byte-shaving signal units.

- `src/comma_lab/scheduler/byte_shaving_campaign_queue.py`
  - Adds source-unit contract construction for full path/SHA/byte custody.
  - Propagates existing source-unit contracts unchanged.
  - Emits explicit source-unit contract blockers when archive signal is partial.
  - Carries the source-unit contract through portfolio `source_metadata`.

- `src/tac/tests/test_byte_shaving_campaign_queue.py`
  - Adds a DQS1 pairset selector regression proving source-unit archive custody
    survives planner normalization, compiler materialization, and portfolio
    metadata.

## Validation

- `.venv/bin/python -m ruff check src/comma_lab/scheduler/byte_shaving_campaign_queue.py src/tac/optimization/byte_shaving_campaign.py src/tac/tests/test_byte_shaving_campaign_queue.py`
  - Passed.

- `.venv/bin/python -m pytest src/tac/tests/test_byte_shaving_campaign_queue.py::test_compile_dqs1_byte_shaving_plan_materializes_pairset_selector_units -q`
  - Passed.

- `git diff --check -- src/comma_lab/scheduler/byte_shaving_campaign_queue.py src/tac/optimization/byte_shaving_campaign.py src/tac/tests/test_byte_shaving_campaign_queue.py`
  - Passed.

- `tools/review_tracker.py policy-check ...`
  - Passed: 0 violations on all three touched files.

## Known Unrelated Full-File State

The full `src/tac/tests/test_byte_shaving_campaign_queue.py -q` run still has
the current broader dirty-main failures outside this slice:

- registry expected-order drift after newer target kinds
- materializer completion postcondition fixtures missing newer readiness fields
- DFL1 exact-readiness follow-up path-vs-workload-root fixture mismatch
- renderer/tensor postcondition fixture gaps

The focused source-unit contract regression passes, and this patch does not
touch those registry/postcondition surfaces.

## Authority Boundary

Generated source-unit contracts remain non-promotional:

- `score_claim=false`
- `promotion_eligible=false`
- `rank_or_kill_eligible=false`
- `ready_for_exact_eval_dispatch=false`

Source-unit archive contracts are acquisition/materialization metadata only;
receiver proof and exact CPU/CUDA authority are still required elsewhere.
