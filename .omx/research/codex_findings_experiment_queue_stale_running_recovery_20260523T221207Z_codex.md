# Codex Findings: Experiment Queue Stale Running Recovery

- timestamp_utc: 2026-05-23T22:12:07Z
- agent: codex
- lane_id: codex_experiment_queue_stale_running_recovery_20260523
- research_only: false

## Finding

The queue worker can timeout and finalize children while it is alive, but a
worker crash could leave local steps stuck in `running`. Those stale rows still
consumed `controls.max_concurrency` resource slots, so a local CPU/MLX queue
could look saturated while no process was actually alive.

## Fix Landed

Added `reconcile_stale_running_steps` to the experiment queue runtime and wired
it into `run_queue_worker` before launching new work. The reconciler only acts
on local resource kinds when the recorded worker/child process is gone:

- if declared postconditions already pass, the stale row is recovered as
  `succeeded`;
- otherwise it is marked `failed` with a stale-process audit event;
- live child or worker-parent processes are skipped;
- cloud/provider resources are skipped.

The operator CLI now also exposes:

```bash
.venv/bin/python tools/experiment_queue.py --queue <queue.json> reconcile-stale-running
```

## Verification

- `PYTHONPATH=. .venv/bin/pytest src/tac/tests/test_experiment_queue.py -q`
  - 45 passed
- `.venv/bin/ruff check src/comma_lab/scheduler/experiment_queue.py src/tac/tests/test_experiment_queue.py tools/experiment_queue.py`
  - passed

## Authority Boundary

This is scheduler-state recovery only. It creates no score authority, promotion
eligibility, rank/kill authority, or exact-eval readiness.
