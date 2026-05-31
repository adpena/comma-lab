# Codex Findings: Runtime Proof Contract Emission

UTC: 2026-05-31T00:07:50Z

## Verdict

The monolithic runtime-consumption proof and PR101 A5 frame-conditional runtime
packet proof now emit the shared `tac_archive_bound_candidate_contract.v1`
surface instead of forcing consumers to re-interpret flat
`candidate_archive_*` readiness fields.

This is a contract-first custody migration only. It does not grant score
authority, promotion authority, rank/kill authority, budget spend authority, or
exact-eval dispatch authority.

## Landed Surfaces

- `tools/build_monolithic_runtime_consumption_proof.py`
  - Adds `archive_bound_candidate_contract_fields_for_row(...)` to canonical
    runtime proof payloads.
  - Binds the candidate archive object, runtime proof path, receiver contract
    kind, and false-authority fields into one archive-bound contract.

- `tools/build_pr101_frame_conditional_runtime_packet.py`
  - Adds the same contract to the packet root manifest and the
    `runtime_consumption_proof.json` payload.
  - Keeps A5 frame-conditional runtime packet rows byte-closed and
    receiver-proof-backed while still blocked from exact CPU/CUDA dispatch until
    contest authority signs the handoff.

- `src/tac/tests/test_prove_monolithic_runtime_consumption.py`
  - Asserts contract schema, candidate archive SHA binding, ready archive-bound
    custody, and false exact/score authority.

- `src/tac/tests/test_pr101_frame_conditional_runtime_packet.py`
  - Asserts the root packet manifest and proof artifact both carry the shared
    contract, bind the candidate archive SHA, and remain false-authority.

## Validation

- `.venv/bin/python -m ruff check tools/build_monolithic_runtime_consumption_proof.py src/tac/tests/test_prove_monolithic_runtime_consumption.py tools/build_pr101_frame_conditional_runtime_packet.py src/tac/tests/test_pr101_frame_conditional_runtime_packet.py`
  - Passed.

- `.venv/bin/python -m pytest src/tac/tests/test_prove_monolithic_runtime_consumption.py src/tac/tests/test_pr101_frame_conditional_runtime_packet.py -q`
  - Passed: 6 tests.

- `git diff --check -- tools/build_monolithic_runtime_consumption_proof.py src/tac/tests/test_prove_monolithic_runtime_consumption.py tools/build_pr101_frame_conditional_runtime_packet.py src/tac/tests/test_pr101_frame_conditional_runtime_packet.py`
  - Passed.

- `tools/review_tracker.py mark-file ... --status reviewed`
  - Marked all four touched files reviewed.

- `tools/review_tracker.py policy-check ...`
  - Passed: 0 violations on all four touched files.

## Remaining Contract-First Targets

Read-only subagent audit found the next two smallest non-overlapping surfaces:

- `src/comma_lab/scheduler/frontier_rate_attack_feedback.py`
  - `_materializer_manifest_record()` still returns flat
    `candidate_archive_*` fields after archive/runtime validation. Next patch
    should emit and propagate the shared contract into materializer binding rows.

- `src/comma_lab/scheduler/byte_shaving_campaign_queue.py`
  - `_materialize_row()` records source units with archive SHA/bytes only. Next
    patch should propagate existing unit contracts or add an explicit
    missing-contract blocker when path/SHA/byte custody is incomplete.

## Authority Boundary

All emitted contracts retain:

- `score_claim=false`
- `promotion_eligible=false`
- `rank_or_kill_eligible=false`
- `ready_for_exact_eval_dispatch=false`

The contracts are usable for acquisition, custody, and exact-handoff planning
only.
