# Codex Findings: Inverse-Scorer Context Generation

Date: 2026-05-24T16:41:38Z
Agent: Codex
Lane: codex_inverse_scorer_context_generation_20260524

## Finding

The PacketIR operation-set lowering path now feeds the materializer backlog and
work queue, but inverse-scorer cell candidates still required hand-authored
materializer contexts. That kept the scorer-inverse/action-surface path closer
to a manual leaf operation than to an automated final-byte work loop.

## Landed Changes

- `comma_lab.scheduler.final_byte_operation_contexts` now generates consumable
  contexts for `inverse_scorer_cell_candidate_v1` rows.
- The generated inverse-cell context supports both one-shot candidate archive
  materialization and chain output mode.
- Chain output mode fails closed when full-frame inflate parity context is
  missing, matching the existing materializer work-queue guard.
- The byte-shaving queue CLI can now generate inverse-scorer contexts from an
  artifact map and drive a smoke materialization into a byte-closed candidate
  manifest without hand-authored contexts.
- PacketIR provenance now flows through generated context rows:
  `source_packet_ir_*` IDs and `packet_ir_blocker_counts` are preserved both at
  the row level and inside the context payload consumed by the work queue.
- `packet_ir_blocker_counts` are normalized to positive integer counts only, so
  invalid or boolean telemetry cannot leak into queue-owned materializer
  contexts.
- The operator-facing build-queue CLI is covered for inverse-scorer artifact
  maps: it generates contexts, emits an executable
  `tools/materialize_inverse_scorer_cell_candidate.py` work row, and a bounded
  smoke executes that row against a tiny local template/action functional to
  produce a byte-closed candidate archive.

## Verification

- `.venv/bin/python -m ruff check src/comma_lab/scheduler/final_byte_operation_contexts.py src/comma_lab/scheduler/byte_shaving_campaign_queue.py src/tac/tests/test_final_byte_operation_contexts.py src/tac/tests/test_byte_shaving_campaign_queue.py src/tac/tests/test_byte_shaving_materializer_campaign_runner.py`
  passed.
- `.venv/bin/python -m pytest src/tac/tests/test_final_byte_operation_contexts.py src/tac/tests/test_byte_shaving_campaign_queue.py src/tac/tests/test_byte_shaving_materializer_campaign_runner.py -q`
  passed: 87 tests before the CLI smoke regression.
- `.venv/bin/python -m pytest src/tac/tests/test_final_byte_operation_contexts.py src/tac/tests/test_byte_shaving_campaign_queue.py src/tac/tests/test_byte_shaving_materializer_campaign_runner.py src/tac/tests/test_inverse_scorer_cell_materializer.py -q`
  passed: 126 tests, 1 expected duplicate-ZIP-member warning.
- `.venv/bin/python -m pytest src/tac/tests/test_byte_shaving_campaign.py src/tac/tests/test_byte_shaving_campaign_queue.py src/tac/tests/test_final_byte_operation_contexts.py src/tac/tests/test_byte_shaving_materializer_campaign_runner.py src/tac/tests/test_deterministic_compiler.py src/tac/tests/test_packet_compiler_sparse_packet_ir.py src/tac/tests/test_packetir_exact_closure.py src/tac/tests/test_optimizer_exact_readiness.py tests/test_pr106_context_recode.py -q`
  passed: 307 tests before the CLI smoke regression.
- `.venv/bin/python -m pytest src/tac/tests/test_byte_shaving_campaign.py src/tac/tests/test_byte_shaving_campaign_queue.py src/tac/tests/test_final_byte_operation_contexts.py src/tac/tests/test_byte_shaving_materializer_campaign_runner.py src/tac/tests/test_deterministic_compiler.py src/tac/tests/test_packet_compiler_sparse_packet_ir.py src/tac/tests/test_packetir_exact_closure.py src/tac/tests/test_optimizer_exact_readiness.py src/tac/tests/test_inverse_scorer_cell_materializer.py tests/test_pr106_context_recode.py -q`
  passed: 346 tests, 1 expected duplicate-ZIP-member warning.
- `.venv/bin/python -m pytest src/tac/tests/test_inverse_scorer_cell_materializer.py -q`
  passed: 38 tests, 1 expected duplicate-ZIP-member warning.
- `git diff --check` passed.
- `.venv/bin/python tools/lane_maturity.py validate` passed: 1250 lanes.

## Remaining Work

This closes the missing context-generation branch for inverse-scorer cell
candidates and proves the generated one-shot materializer command emits a
byte-closed candidate archive locally. The next execution gate is the chain
mode: feed real MLX/action-functional artifact maps with inflate parity context
and run a bounded local inverse-scorer candidate-chain smoke that either passes
parity or records the exact typed parity blocker.
