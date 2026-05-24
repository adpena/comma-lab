# Codex Findings: PacketIR Inline Context Params

Timestamp UTC: 2026-05-24T22:09:49Z

## Finding

The inverse-action compiler and PacketIR path could already carry deterministic
operation selectors such as archive section, packet member, and tensor rank, but
the downstream final-byte context path still required those selectors to be
duplicated in the external artifact map. That made PacketIR less executable
than it should be: custody paths and runtime proofs properly belonged in the
artifact map, while operation-local selector knobs were already known at the
compiled operation layer.

## Landing

- `comma_lab.scheduler.byte_shaving_campaign_queue` now preserves operation
  parameters from selected operations and PacketIR operations into materializer
  backlog rows.
- Materializer backlog rows now expose aggregate `operation_params` plus
  `source_operation_params_by_unit` for traceability.
- `comma_lab.scheduler.final_byte_operation_contexts` now treats backlog-row
  `operation_params` as inline hints before applying external artifact-map
  overrides.
- Archive-section aliases (`archive_section`, `section_name`, `target_section`)
  and packet-member alias (`packet_member`) are normalized into the context
  fields consumed by final materializer work queues.

## Safeguards

- External artifact-map hints still override inline operation params, so custody
  paths, manifests, output roots, and runtime-consumption proofs remain under
  explicit operator/artifact control.
- Inline params do not create score, promotion, rank/kill, or exact-dispatch
  authority.
- The landing only makes local materializer work rows more executable; exact
  auth eval remains required before any score claim.

## Verification

- `.venv/bin/python -m ruff check src/comma_lab/scheduler/byte_shaving_campaign_queue.py src/comma_lab/scheduler/final_byte_operation_contexts.py src/tac/tests/test_byte_shaving_campaign_queue.py src/tac/tests/test_final_byte_operation_contexts.py`
- `.venv/bin/python -m pytest src/tac/tests/test_byte_shaving_campaign_queue.py::test_direct_mlx_compiler_hint_reaches_materializer_work_queue src/tac/tests/test_final_byte_operation_contexts.py::test_final_byte_context_compiler_uses_inline_operation_params -q`
- `.venv/bin/python -m pytest src/tac/tests/test_final_byte_operation_contexts.py src/tac/tests/test_byte_shaving_campaign_queue.py -q`
- `.venv/bin/python -m pytest src/tac/tests/test_byte_shaving_materializer_campaign_runner.py -q`
- `tools/review_tracker.py policy-check` clean for the touched scheduler/context/test files.

## Remaining Gap

This does not yet prove frontier movement. It removes one execution-wiring gap:
inverse-surface and MLX-selected operations can now carry deterministic selector
knobs through PacketIR/backlog/context/work-queue without restating them in
external hint maps. The next frontier-moving step is to run the generated local
materializer work queue against real champion artifacts, harvest candidate
archives, then exact-eval only byte-closed candidates that pass custody and
runtime-consumption proofs.
