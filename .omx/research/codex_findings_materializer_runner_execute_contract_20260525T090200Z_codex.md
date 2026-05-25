# Codex Findings: Materializer Runner Execute Contract

UTC: 2026-05-25T09:02:00Z
Lane: `codex_materializer_runner_context_autowire_20260525`
Commit status at write time: pending

## Finding

The runner-level context autowire was sufficient to build executable local
materializer queue rows, but real `--execute` surfaced two remaining integration
issues before the materializer could become useful signal.

First, `tools/run_byte_shaving_materializer_campaign.py` used a per-run SQLite
state path by default while `tools/experiment_queue.py run-worker --execute`
requires an explicit noncanonical-state rationale. The runner now supplies that
rationale only for runner-owned per-run state. User-provided custom state still
requires a caller-provided `--queue-state-rationale`.

Second, inline compiler plans can carry a stale
`runtime_consumption_proof` path. The final-byte context compiler now treats a
missing proof path as a missing input, not as an output destination. Generated
runtime proofs are emitted beside the current materializer manifest through the
queue command's normal `--runtime-consumption-proof-out` path.

## Execution Proof

Durable run:
`.omx/research/codex_materializer_runner_execute_inline_contexts_20260525T090127Z/`

Proof summary:
`.omx/research/codex_materializer_runner_execute_inline_contexts_20260525T090127Z/proof_summary.json`

The worker launched the local materializer with an explicit noncanonical-state
rationale. The materializer returned zero and emitted:

- byte-closed candidate archive;
- candidate manifest;
- runtime-consumption proof beside the current materializer output;
- false score/promotion/rank/dispatch authority fields.

## Current Blocker

The queue step is intentionally still not successful because the synthetic
archive-section entropy recode changed section length without a satisfying
runtime adapter/receiver proof:

- `section_length_changed_requires_runtime_consumption_proof`
- `runtime_consumption_proof_not_passed`
- `archive_section_entropy_recode_receiver_contract_not_satisfied`

That is now a real receiver/materializer validity blocker, not a scheduler,
state, context-generation, or missing-manifest failure.

## Next Engineering Target

The next non-orphaned step is to make final-byte operation execution produce
actionable negative/positive signal automatically:

- either generate length-preserving/archive-safe entropy-recode candidates for
  this receiver contract;
- or add a true runtime-adapter receiver path for length-changing section
  recodes;
- and ensure queue performance/replan treats receiver-negative byte-closed
  materializations as solver signal without granting exact-eval authority.

## Verification

- `.venv/bin/python -m ruff check src/comma_lab/scheduler/final_byte_operation_contexts.py tools/run_byte_shaving_materializer_campaign.py src/tac/tests/test_final_byte_operation_contexts.py src/tac/tests/test_byte_shaving_materializer_campaign_runner.py`
- `.venv/bin/python -m pytest src/tac/tests/test_final_byte_operation_contexts.py::test_final_byte_context_compiler_emits_consumable_materializer_contexts src/tac/tests/test_final_byte_operation_contexts.py::test_final_byte_context_compiler_uses_existing_runtime_proof_as_input src/tac/tests/test_final_byte_operation_contexts.py::test_final_byte_context_compiler_covers_packet_member_and_tensor_families src/tac/tests/test_byte_shaving_materializer_campaign_runner.py::test_materializer_campaign_runner_default_state_gets_execution_rationale src/tac/tests/test_byte_shaving_materializer_campaign_runner.py::test_materializer_campaign_runner_custom_state_requires_explicit_rationale`

Latest focused result: 5 passed.
