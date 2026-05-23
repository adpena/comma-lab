# Codex Findings: Materializer Context File Queue Bridge

**UTC**: 2026-05-23T17:01:50Z
**Lane**: `lane_codex_materializer_context_file_queue_bridge_20260523`

## Finding

The previous `byte_shaving_materializer_work_queue.v1` surface could consume
contexts through Python, but the operator CLI had no durable context-file input.
That meant byte-range entropy backlog rows still required hand-written Python
glue to become executable proof-chain commands.

## Landed

- Added `byte_shaving_materializer_contexts.v1`.
- Added `materializer_contexts_from_payload(...)` with lookup keys by
  `backlog_key`, `materializer_id`, `target_kind`, and `source_unit_ids`.
- Added `--materializer-contexts` to
  `tools/build_byte_shaving_campaign_queue.py`.
- Context files are fail-closed against truthy score/promotion/dispatch
  authority fields.
- CLI tests prove a context file can emit an executable
  `tools/run_byte_range_entropy_recode_chain.py` command into
  `byte_shaving_materializer_work_queue.v1`.

## Authority

This is a local proof-chain queue surface only. It does not grant score,
promotion, rank/kill, or exact-eval authority; exact contest CPU/CUDA auth eval
is still required before any score claim.
