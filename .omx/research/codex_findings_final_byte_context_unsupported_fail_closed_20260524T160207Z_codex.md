# Codex Findings: Final-Byte Context Unsupported Rows Fail Closed

Timestamp UTC: 2026-05-24T16:02:07Z
Lane: `lane_codex_materializer_context_file_queue_bridge_20260523`

## Finding

The final-byte context compiler existed and could generate executable
`byte_shaving_materializer_contexts.v1` rows for archive-section, packet-member,
and tensor-family materializers. The remaining failure mode was subtler:
unsupported backlog rows were only listed in `unsupported_backlog_keys`, so
`--fail-if-blocked` could pass with `blocked_context_count=0` even when the
compiler produced no executable context for a selected high-level operation.

That is the same leaf/local-minimum risk in another form: the planner selected a
portfolio-level row, but context generation could look successful while leaving
actuation unresolved.

## Landed

- Unsupported backlog rows now emit first-class blocked context rows with
  `final_byte_context_compiler_unsupported_backlog_row`.
- `blocked_context_count` includes unsupported rows, so CLI
  `--fail-if-blocked` fails closed instead of silently succeeding.
- `byte_shaving_materializer_work_queue.v1` now carries context-level blockers
  forward generically, even when the row is not one of the currently supported
  concrete materializer families.
- Unsupported rows carry a PacketIR bridge hint that wraps the compiler-owned
  `packetir_operation_set_bridge_contract()` rather than duplicating compiler
  order or proof vocabulary in the scheduler.
- Regression tests cover high-level inverse-action compiler rows, work-queue
  blocker propagation, and CLI `--fail-if-blocked` behavior.

## Verification

- `.venv/bin/python -m pytest src/tac/tests/test_final_byte_operation_contexts.py src/tac/tests/test_byte_shaving_campaign_queue.py::test_byte_shaving_campaign_queue_cli_generates_materializer_contexts_from_artifact_map src/tac/tests/test_byte_shaving_campaign_queue.py::test_materializer_context_payload_maps_rows_to_multiple_lookup_keys src/tac/tests/test_byte_shaving_campaign_queue.py::test_materializer_work_queue_blocks_ambiguous_multi_context_backlog_row -q` (10 passed)
- `.venv/bin/python -m pytest src/tac/tests/test_final_byte_operation_contexts.py src/tac/tests/test_byte_shaving_campaign_queue.py src/tac/tests/test_byte_shaving_materializer_campaign_runner.py -q` (78 passed)
- `.venv/bin/ruff check src/comma_lab/scheduler/final_byte_operation_contexts.py src/comma_lab/scheduler/byte_shaving_campaign_queue.py src/tac/tests/test_final_byte_operation_contexts.py`
- `git diff --check`

## Remaining Gap

The compiler is now fail-closed for unsupported operations, but the next
frontier-moving step is still to implement real operation-set compilers that map
inverse-steganalysis water-bucket rows to concrete archive-section,
packet-member, tensor, byte-range, or substrate-specific transforms with
manifest/custody paths and runtime-consumption proof.
