# Codex Findings: Scheduler Execution Safeguards

UTC: 2026-05-23T21:02:33Z
Agent: Codex
Lane: codex_scheduler_execution_safeguards_20260523

## Summary

The scheduler/DAG execution path needed a few last fail-closed guards so high
parallelism does not outrun storage, cleanup, or runtime-consumption evidence.

## Change

- Materializer execution queues that include scheduler storage preflight now
  require proactive cleanup to execute, not dry-run.
- Relative scheduler results roots now require an explicit
  `scheduler_storage_expected_workload_root` when preflight gates execution.
- Staircase DAG dispatch now carries queue `max_concurrency` and applies it
  across resource pools, so a large local pool cannot silently exceed queue
  concurrency policy.
- Exact readiness now requires runtime-consumption proof by default for changed
  byte-closed candidates, unless strict inverse-scorer full-frame parity already
  backs the row.
- Promoted exact-ready rows now preserve runtime-consumption proof status so
  downstream dispatch reloaders do not misclassify proof-backed rows as missing.

## Verification

- `.venv/bin/python -m pytest src/tac/tests/test_byte_shaving_campaign_queue.py src/tac/tests/test_staircase_dag.py src/tac/tests/test_optimizer_exact_readiness.py -q`
- `.venv/bin/python -m ruff check src/comma_lab/scheduler/byte_shaving_campaign_queue.py src/comma_lab/scheduler/staircase_dag.py src/tac/optimizer/exact_readiness.py src/tac/tests/test_byte_shaving_campaign_queue.py src/tac/tests/test_staircase_dag.py src/tac/tests/test_optimizer_exact_readiness.py`
