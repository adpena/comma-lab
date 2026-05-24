---
schema: codex_findings_v1
author: codex
created_at_utc: 2026-05-24T00:26:28Z
lane_id: codex_staircase_ssh_executor_20260524
score_claim: false
promotion_eligible: false
rank_or_kill_eligible: false
ready_for_exact_eval_dispatch: false
---

# Staircase SSH Executor Bridge

## Finding

The materializer campaign queue now emits enough staircase/DAG structure to
place work on peer machines, but the missing production surface was a
queue-owned executor bridge. Dask or SSH placement must not become a second
source of truth: `experiment_queue.v1` and its SQLite state remain the authority
for readiness, claims, dependencies, terminal status, pause/freeze, and
postcondition validation.

This tranche adds an SSH executor that consumes `staircase_dispatch_plan.v1`
`dask_task_specs` as placement hints only. It re-loads the canonical queue,
checks source queue hash and task step hashes, requires queue-state writeback
metadata, verifies machine SSH metadata, probes remote git state, atomically
claims the local `ReadyStep`, runs the queue-owned argv over SSH, and only marks
the local state succeeded when local postconditions and false-authority checks
pass.

Remote success is deliberately not score authority or even artifact authority.
If artifacts are not visible locally after the remote run, the step fails until
artifact sync/shared-storage semantics are explicit.

## Landed Surfaces

- `comma_lab.scheduler.ssh_experiment_queue_executor`: queue-owned SSH executor
  bridge for staircase-selected tasks.
- `comma_lab.scheduler.experiment_queue.finalize_claimed_step_execution`:
  public terminal-state helper for external executors that need to execute
  remotely while preserving local queue authority and postcondition validation.
- `tools/run_staircase_ssh_executor.py`: operator CLI with dry-run default,
  canonical-state default, remote repo root mapping, and staged future-executor
  override.
- `comma_lab.scheduler.staircase_dag`: resource pools now preserve
  `remote_repo_root`; `tertiary` is upgraded from future placeholder to
  `ssh_experiment_queue` CPU-only placement, still blocked unless a remote repo
  root is supplied and remote git preflight passes.
- `.gitignore`: SSH executor logs remain local state.

## Verification

- `PYTHONPATH=. .venv/bin/python -m ruff check ...` on touched executor,
  staircase, CLI, and tests.
- `PYTHONPATH=. .venv/bin/python -m pytest -q src/tac/tests/test_ssh_experiment_queue_executor.py`
  passed with guards for plan-command mutation, stale hashes, remote-success
  non-authority, future-executor gating, and CLI dry-run.
- Expanded queue/DAG regression passed:
  `73 passed` across SSH executor, staircase DAG, and experiment queue tests.
- `tools/lane_maturity.py validate` passed with `1216 lane(s) validated cleanly`.
- `git diff --check` passed.

## Frontier Status

No exact auth eval was dispatched in this tranche. `reports/latest.md` still
lists `[contest-CPU Linux x86_64]` best at `0.1920282830` and
`[contest-CUDA T4]` best at `0.2053300290`.

## Next Tranche

Next is artifact mobility and scheduler saturation: add an explicit artifact
sync/shared-storage contract so SSH workers can return materialized candidate
trees, then let `run_byte_shaving_materializer_campaign.py` generate a
staircase plan and feed it to the SSH executor/Dask layer. Once that is in
place, the M5 Max can keep MLX/local-heavy tasks while `tertiary` and other
peers consume bounded CPU-only materializer/harvest work without duplicating
queue truth.
