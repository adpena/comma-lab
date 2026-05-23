# Codex Findings: Experiment Queue Parallel Worker

Date: 2026-05-23T06:33:55Z
Agent: Codex
Lane: `lane_codex_experiment_queue_parallel_worker_20260523`

## Summary

This pass converted the experiment queue from resource-aware but serial
execution into a parent-owned, bounded parallel worker suitable for local
CPU/MLX sweeps.

Landed fixes:

- Split queue execution into claim/start/finalize phases for subprocesses.
- Added parent-owned polling, timeout, stop-policy, and postcondition
  finalization so child processes never mutate SQLite state.
- Added `run-worker --max-parallel`, `--poll-interval-seconds`,
  `--stop-policy`, and `--shutdown-grace-seconds`.
- Added observer telemetry for worker run id, child PID, timeout, and deadline.
- Extended the DQS1 local-first queue builder to emit multiple independent safe
  candidates and explicit local CPU concurrency when requested.
- Preserved false-authority and eureka gates; cloud resources remain excluded
  unless `--allow-cloud` is explicit.

## Evidence

- `.venv/bin/python -m pytest -q src/tac/tests/test_experiment_queue.py src/tac/tests/test_experiment_queue_observer.py src/tac/tests/test_staircase_dag.py src/tac/tests/test_dqs1_local_first_queue_builder.py`
  returned `45 passed in 5.88s`.
- `.venv/bin/python -m ruff check src/comma_lab/scheduler/experiment_queue.py src/comma_lab/scheduler/experiment_queue_observer.py src/comma_lab/scheduler/dqs1_local_first_queue.py tools/experiment_queue.py tools/build_dqs1_local_first_queue.py src/tac/tests/test_experiment_queue.py src/tac/tests/test_experiment_queue_observer.py src/tac/tests/test_dqs1_local_first_queue_builder.py`
  returned `All checks passed!`.
- DQS1 no-write CLI smoke
  `.venv/bin/python tools/build_dqs1_local_first_queue.py --action-summary latest --candidate-limit 2 --local-cpu-concurrency 2`
  emitted two local-first experiments with `controls.max_concurrency.local_cpu=2`:
  `pairset_drop_two_r029_020_p0259_0430` and
  `pairset_drop_two_r028_020_p0257_0430`.
- `git diff --check` passed.

## Remaining Work

- Generate a real DQS1 multi-candidate queue from the latest action summary with
  local CPU/MLX concurrency set for the M5 Max memory budget, then run it under
  the new worker.
- Add learned acquisition/eureka pause policy around local MLX and local CPU
  results so positive frontier-proxy signals stop the worker before burying the
  result under lower-priority tasks.
- Extend the same multi-candidate queue builder pattern to PR95/HNeRV,
  NeRV-family, non-NeRV, and other substrate sweeps.

## Authority

`score_claim=false`; `promotion_eligible=false`;
`ready_for_exact_eval_dispatch=false`; `rank_or_kill_eligible=false`.
