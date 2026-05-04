# Lightning Dispatch State Atomicity Hardening - 2026-05-04

## Bug Class

Concurrent Lightning state updates could interleave against
`.omx/state/lightning_batch_jobs.json` when multiple `refresh-status`, harvest,
stop, or submit-control processes performed read/modify/write cycles at the
same time. The observed failure mode was invalid JSON with trailing/truncated
garbage after parallel refreshes against the same state file.

Evidence grade: engineering guardrail. This note records a harness hardening
fix, not a score claim.

## Guard

- `src/tac/deploy/lightning/batch_jobs.py` now serializes state-file access
  with a sibling POSIX lock file (`<state>.lock`).
- State persistence now writes JSON to a same-directory temporary file, fsyncs
  it, and atomically replaces the target with `os.replace`.
- `LightningBatchJobsClient.record()` performs append read/modify/write inside
  one critical section so concurrent submit/record appenders preserve all
  rows.
- `LightningBatchJobsClient.refresh_status_from_job()` applies the SDK
  snapshot through an atomic latest-record update rather than a stale
  load/replace pair.
- The launcher-side SSH harvest failure refinement now reuses the locked client
  update path instead of writing `lightning_batch_jobs.json` directly.

## Regression Tests

Added focused coverage in `src/tac/tests/test_lightning_batch_jobs.py`:

- a child writer blocks behind an already-held cross-process lock and writes
  only after the lock is released;
- multiple subprocess writers append concurrently and the final state remains
  valid JSON with every row preserved.

## Verification

Commands run:

```bash
.venv/bin/python -m py_compile scripts/launch_lightning_batch_job.py src/tac/deploy/lightning/batch_jobs.py src/tac/tests/test_lightning_batch_jobs.py
.venv/bin/python -m pytest src/tac/tests/test_lightning_batch_jobs.py -q
```

Result: `120 passed in 1.94s` on the final focused rerun.

## Operational Rule

Do not run ad hoc parallel processes that write the same Lightning state file
through unreviewed helper code. New state mutators should call
`LightningBatchJobsClient.record()` or
`LightningBatchJobsClient.replace_latest_record_for_job()` so they inherit the
lock and atomic replace guard.
