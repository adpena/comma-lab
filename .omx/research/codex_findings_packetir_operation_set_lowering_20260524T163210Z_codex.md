# Codex Findings: PacketIR Operation-Set Lowering And Queue Authority

Date: 2026-05-24T16:32:10Z
Agent: Codex
Lane: codex_packetir_operation_set_lowering_20260524

## Finding

The inverse-steganalysis/action-surface planner had started emitting typed
`packet_ir_operation_set_v1` rows, but those rows were still effectively
side-channel signal in the campaign compiler. They were visible in
`packet_ir_materializer_backlog_rows`, yet `materializer_work_queue` was built
from the pre-PacketIR backlog. That meant a high-level inverse-scorer operation
set could be represented and reviewed without becoming authoritative scheduler
work.

Volta also found a false-authority smell in the high-level inverse-action
adapter: it was non-executable but still advertised `emits_candidate_archive`.
That made the contract look closer to a candidate materializer than it was.

## Landed Changes

- PacketIR operation-set rows are now merged into the authoritative
  `materializer_backlog` before `materializer_backlog_summary` and
  `materializer_work_queue` are built.
- PacketIR-only plans now still produce materializer backlog/work-queue rows,
  so compiler output can drive the queue even when the legacy operation ladders
  are absent.
- DQS1 direct pair-drop PacketIR rows no longer create redundant context-only
  materializer backlog rows when the existing DQS1 local-first path is already
  executable.
- The high-level inverse-action adapter is explicitly planning-only and
  non-candidate-emitting until lowered to concrete family materializers with
  runtime-consumption proof.
- `gpu_launched=False` is now part of the canonical proxy false-authority
  contract, and exact-readiness tests reject runtime proofs that set it true.

## Verification

- `.venv/bin/python -m ruff check ...` passed for the touched queue,
  compiler, readiness, and tests.
- `.venv/bin/python -m pytest src/tac/tests/test_byte_shaving_campaign.py src/tac/tests/test_byte_shaving_campaign_queue.py src/tac/tests/test_deterministic_compiler.py src/tac/tests/test_optimizer_exact_readiness.py -q`
  passed: 162 tests.
- `.venv/bin/python -m pytest src/tac/tests/test_byte_shaving_campaign.py src/tac/tests/test_byte_shaving_campaign_queue.py src/tac/tests/test_final_byte_operation_contexts.py src/tac/tests/test_byte_shaving_materializer_campaign_runner.py src/tac/tests/test_deterministic_compiler.py src/tac/tests/test_packet_compiler_sparse_packet_ir.py src/tac/tests/test_packetir_exact_closure.py src/tac/tests/test_optimizer_exact_readiness.py tests/test_pr106_context_recode.py -q`
  passed: 305 tests.
- `git diff --check` passed.

## Remaining Work

This does not complete the theoretical-floor system. It removes one orphaned
signal boundary: inverse-steg/action-surface operation sets now enter the same
materializer queue path as legacy byte-shaving gaps.

Next engineering gates:

1. Provide materializer context payloads for archive-section, packet-member,
   tensor, and inverse-scorer-cell PacketIR backlog keys so merged rows become
   executable local proof-chain work.
2. Feed MLX/Metal/Accelerate scorer-response surfaces into the action
   functional as calibrated research signal, while keeping contest CPU/CUDA
   exact authority separate.
3. Add queue telemetry that measures PacketIR lowering row counts, blocked
   context classes, and executable conversion rate so the scheduler learns
   where inverse-surface signal is stalling.
4. Profile actual materializer execution hot paths before lowering more code to
   Rust; this tranche changed schema/backlog authority propagation, not a
   measured compute kernel.
