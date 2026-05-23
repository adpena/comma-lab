# Codex Findings - Materializer Scheduler Storage Preflight

UTC: 2026-05-23T19:51:29Z
Lane: `lane_materializer_scheduler_storage_preflight_20260523`

## Finding

The byte-shaving materializer execution queue could already run multiple
candidate rows, but storage planning and proactive cleanup were still coupled
to the DQS1 local-first queue path. That left the materializer DAG able to
consume internal disk directly unless the operator manually wrapped it with the
same storage waterfall and rebuildable-bulk cleanup policy.

## Fix

- Added `comma_lab.scheduler.storage_preflight` as the shared queue experiment
  builder for storage-tier planning plus proactive cleanup.
- Refactored the DQS1 local-first scheduler preflight to use the shared helper.
- Added optional materializer execution queue gating via
  `materializer_scheduler_preflight.proactive_cleanup`.
- Exposed materializer preflight controls in
  `tools/build_byte_shaving_campaign_queue.py` so generated queues can target
  the external storage waterfall first, cold-store or delete only when
  explicitly requested, and keep local disk as a last resort.

## Verification

- `src/tac/tests/test_byte_shaving_campaign_queue.py`
  `src/tac/tests/test_dqs1_local_first_queue_builder.py`
  `src/tac/tests/test_experiment_queue.py`: 99 passed.
- `ruff check` passed for the touched scheduler, CLI, and test files.
- `compileall` passed for the touched scheduler, CLI, and test files.

## Authority

The storage preflight is scheduler plumbing only. It does not claim score,
promote candidates, rank or kill candidates, launch GPU work, or bypass exact
contest auth eval.
