# Codex Findings: Inverse-Steganalysis Receiver Compiler Bridge

UTC: 2026-05-25T08:13:54Z
Lane: `codex_inverse_steg_receiver_compiler_20260525`
Agent: Codex
Sidecar closed: `019e5e1a-10f1-7690-baf1-10337f6875e8`

## What Landed

- Added a reusable TAC helper:
  `src/tac/optimization/inverse_steganalysis_operation_set_compiler.py`
  - Accepts `inverse_action_operation_set_compiler_hint.v1`.
  - Emits deterministic `packet_ir_operation_set_v1`.
  - Preserves false-authority fields.
  - Carries source backlog key, source paths, operation sequence hash, and compiler bridge contract.

- Wired final-byte context generation:
  `src/comma_lab/scheduler/final_byte_operation_contexts.py`
  - High-level inverse-action rows now become blocked contexts when no compiler hint exists.
  - With `operation_set_compiler` or `packet_ir_operation_set`, the context compiler emits the high-level receiver context plus concrete lowered materializer contexts.
  - Lowered concrete contexts retain access to global sibling artifact-map hints, so archive/member/tensor materializers receive their real custody fields instead of orphaned inline fragments.

- Wired materializer work queue lowering:
  `src/comma_lab/scheduler/byte_shaving_campaign_queue.py`
  - High-level rows remain non-executable receiver/compiler handoffs.
  - Context-carried PacketIR or compiler hints lower through existing `lower_packetir_operation_set_to_backlog_rows(...)`.
  - Concrete lowered rows then use the existing family materializer command paths.
  - Invalid high-level compiler hints now fail loudly as `inverse_action_operation_set_compiler_invalid:*` instead of being silently dropped.
  - Work rows preserve lowered PacketIR provenance and `source_backlog_key`.

## Proof Artifacts

Append-only proof root:
`.omx/research/codex_inverse_steg_receiver_compiler_bridge_20260525T081354Z/`

Current widened-plan recovery proof:
- `materializer_work_queue.feedback.widened.current.json`
- The old adapter-missing blocker is gone.
- The remaining blocker is correctly classified as:
  `inverse_action_high_level_context_requires_operation_set_compiler`.

Compiler-hint bridge proof:
- `materializer_contexts.compiler_fixture.json`
  - `row_count=2`
  - `blocked_context_count=0`
  - high-level target carries `packet_ir_operation_set_v1`
  - concrete target is `archive_section_entropy_recode_v1`
- `materializer_work_queue.compiler_fixture.json`
  - `row_count=2`
  - `executable_row_count=1`
  - high-level row blocked only by `inverse_action_high_level_context_lowered_to_packet_ir_materializer_rows`
  - concrete row executable via `tools/run_family_agnostic_materializer.py`
  - concrete row preserves source high-level backlog key and `packetir_compiled_high_level_section_fixture`

## Verification

- `.venv/bin/python -m ruff check src/comma_lab/scheduler/byte_shaving_campaign_queue.py src/comma_lab/scheduler/final_byte_operation_contexts.py src/tac/optimization/inverse_steganalysis_operation_set_compiler.py src/tac/tests/test_final_byte_operation_contexts.py src/tac/tests/test_byte_shaving_campaign_queue.py`
- `.venv/bin/python -m pytest src/tac/tests/test_final_byte_operation_contexts.py src/tac/tests/test_byte_shaving_campaign_queue.py -q`
  - `93 passed`
- `.venv/bin/python -m pytest src/tac/tests/test_deterministic_compiler.py src/tac/tests/test_packet_compiler_sparse_packet_ir.py src/tac/tests/test_packetir_exact_closure.py src/tac/tests/test_optimizer_exact_readiness.py -q`
  - `203 passed`
- `.venv/bin/python -m pytest src/tac/tests/test_byte_shaving_materializer_campaign_runner.py -q`
  - `58 passed`

## Remaining Gap

The bridge is implemented and queue-wired. The remaining non-fixture production gap is upstream emission of real `operation_set_compiler` hints from the inverse-action/action-functional producer or runner artifact-map generator. Until that lands, current high-level rows fail closed at the correct gate instead of orphaning behind an adapter-missing placeholder.

