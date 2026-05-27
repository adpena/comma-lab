# Codex Findings: Experiment Queue Frozen Status Contract

Generated: 2026-05-27T20:02:32Z

## Finding

The final-rate autoloop had a false-progress ambiguity: experiments frozen in the
queue definition were initialized into SQLite `step_state` as `queued`, and the
observer reported those stored rows directly. `ready_steps()` correctly refused
to start the frozen work, but parent automation saw `status_counts.queued > 0`
and classified the child queue as stalled queued work.

This was a scheduler contract bug, not a materializer result.

## Fix

- `queue_summary()` now reports effective step status from the active queue
  definition while preserving the stored SQLite status for audit.
- `experiment_queue_observer` surfaces `paused_steps`, `frozen_steps`, and
  `disabled_steps` instead of dropping non-queued definition states.
- `frontier_final_rate_attack_autoloop` distinguishes real queued stalls from
  frozen/paused/disabled no-op child queues.
- Runtime-identity postcondition tests now include the expected runtime tree
  hash required by the stricter fail-closed identity contract.

## Verification

- `.venv/bin/ruff check src/comma_lab/scheduler/experiment_queue.py src/comma_lab/scheduler/experiment_queue_observer.py src/comma_lab/scheduler/frontier_final_rate_attack_autoloop.py src/tac/tests/test_experiment_queue.py src/tac/tests/test_experiment_queue_observer.py src/tac/tests/test_frontier_rate_attack_bootstrap.py`
- `.venv/bin/python -m py_compile src/comma_lab/scheduler/experiment_queue.py src/comma_lab/scheduler/experiment_queue_observer.py src/comma_lab/scheduler/frontier_final_rate_attack_autoloop.py`
- `PYTHONPATH=. .venv/bin/pytest src/tac/tests/test_experiment_queue.py src/tac/tests/test_experiment_queue_observer.py src/tac/tests/test_frontier_rate_attack_bootstrap.py -q`
  - Result: 133 passed in 20.30s.

Bounded final-rate smoke:

- Output: `.omx/research/frontier_final_rate_attack_frozen_status_smoke_20260527Tlocal/`
- Results root: `/Volumes/VertigoDataTier/pact/frontier_final_rate_attack_frozen_status_smoke_20260527Tlocal/`
- Disk footprint: 2.5M local `.omx`, 1.3M external SSD.
- Main queue: `failed_command_count=0`, `feedback_refresh_returncode=0`.
- Child queues: `selected_queue_count=2`, `executed_queue_count=2`,
  `observer_revalidation_failed_count=0`, `stalled_queue_count=0`,
  `frozen_noop_queue_count=1`.
- `operation_chain_compiler_queue`: `steps_started=1`, `progress_made=true`,
  `queue_status_counts={"queued": 1, "succeeded": 1}`.
- `autonomous_chain_optimization_queue`: `steps_started=0`,
  `progress_made=false`,
  `progress_blockers=["child_queue_remaining_work_frozen_by_definition"]`,
  `queue_status_counts={"frozen": 15}`.

## Continual-Learning Hook

Queue observers are now authoritative for definition-effective work state, not
only stored SQLite state. Parent loops should treat `queued` as runnable
remaining work, and `frozen`/`paused`/`disabled` as explicit policy states that
need activation criteria or upstream artifact repair.

The next automation step is to turn frozen child queues into activation plans:
for each frozen experiment, emit the exact missing receiver/readiness/score
evidence that would thaw it, then route that evidence request into the same
queue-owned materializer and repair-budget loops.
