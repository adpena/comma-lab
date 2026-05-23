# Codex Findings - Scheduler Storage Preflight Materializer

UTC: 2026-05-23T20:00:10Z
Lane: `lane_materializer_scheduler_preflight_20260523`

## Finding

DQS1 local-first queues already had storage-tier planning and proactive cleanup
as a scheduler-owned preflight, but the byte-shaving materializer execution
queue could still fan out work without the same storage-first guard. That was a
throughput risk: aggressive materialization can saturate local IO, fill the
wrong tier, or skip certified cleanup before useful local CPU work starts.

## Fix

- Added `comma_lab.scheduler.storage_preflight` as the shared storage planning
  and proactive cleanup queue experiment builder.
- Rewired DQS1 local-first queue construction to use the shared helper.
- Added optional materializer execution gating via
  `materializer_scheduler_preflight.proactive_cleanup`.
- Added CLI flags to `tools/build_byte_shaving_campaign_queue.py` for both DQS1
  queue preflight and materializer execution preflight.
- Added a workload-root custody guard: when materializer storage preflight is
  enabled, executable materializer rows must report artifact paths under the
  expected workload root.

## Verification

- `src/tac/tests/test_byte_shaving_campaign_queue.py`
  `src/tac/tests/test_dqs1_local_first_queue_builder.py`
  `src/tac/tests/test_experiment_queue.py`: 99 passed.
- Combined queue/readiness regression:
  `src/tac/tests/test_inverse_scorer_exact_eval_queue.py`
  `src/tac/tests/test_optimizer_exact_readiness.py`
  `src/tac/tests/test_byte_shaving_campaign_queue.py`
  `src/tac/tests/test_dqs1_local_first_queue_builder.py`
  `src/tac/tests/test_experiment_queue.py`: 144 passed.
- `ruff check` passed on touched Python files.
- `compileall` passed on touched modules and tools.

## Authority

This is local queue plumbing only. It plans storage and cleanup before materializer
work, preserves false-authority postconditions, and does not claim score,
promote, rank, kill, launch GPU jobs, or bypass exact auth eval.
