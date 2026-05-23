# Codex Findings: Experiment Queue Resource Auto Parallelism

Timestamp UTC: 2026-05-23T09:08:00Z

## Scope

Reviewed and hardened the local-first experiment queue worker contract for the
DQS1/MLX autopilot path. The immediate bug class was a control-plane mismatch:
queue definitions already carried `controls.max_concurrency`, but the worker
defaulted to `--max-parallel 1`, so resource declarations did not automatically
drive actual process concurrency.

## Findings

- The SQLite queue worker had the right per-resource concurrency model, but the
  global worker budget was a second hand-tuned knob with a serial default.
- This created a silent under-saturation failure mode for local CPU plus MLX
  queues: adding `local_mlx: 1` to queue controls was not enough to make the
  worker run CPU and MLX jobs concurrently.
- The fix makes `0` or `None` mean automatic worker sizing from positive
  resource caps. Cloud resource kinds remain excluded unless `allow_cloud` is
  explicit.
- `tools/experiment_queue.py validate` now reports the derived local-only and
  with-cloud auto-parallelism plan before the operator starts a worker.

## Current DQS1 Queue Observation

`configs/experiment_queues/dqs1_pairset_local_first.yaml` validates with
`local_only.max_parallel = 1` because all current executable steps are
`local_cpu`. The queue declares `local_mlx: 1`, but no step consumes it yet.
That is now visible from the normal operator CLI instead of being an implicit
absence in the YAML.

## Integration

- Reusable helper: `worker_resource_limits(queue, allow_cloud=False)`.
- Reusable resolver: `resolve_worker_max_parallel(queue, requested_max_parallel, allow_cloud=False)`.
- Worker telemetry now records `requested_max_parallel`, resolved
  `max_parallel`, and `resource_limits`.
- CLI validation emits `auto_parallelism.local_only` and
  `auto_parallelism.with_cloud`.
- Queue observations emit the same auto-parallelism surface plus
  `idle_declared_resources`, so declared-but-unused capacity becomes visible in
  live operator telemetry.
- Focused regression coverage verifies both API behavior and CLI output.

## Verification

- `.venv/bin/python -m pytest -q src/tac/tests/test_experiment_queue.py src/tac/tests/test_experiment_queue_observer.py src/tac/tests/test_dqs1_local_first_queue_builder.py`
- `.venv/bin/ruff check src/comma_lab/scheduler/experiment_queue.py src/comma_lab/scheduler/experiment_queue_observer.py tools/experiment_queue.py src/tac/tests/test_experiment_queue.py src/tac/tests/test_experiment_queue_observer.py src/tac/tests/test_dqs1_local_first_queue_builder.py`
- `git diff --check`
- `.venv/bin/python tools/experiment_queue.py --queue configs/experiment_queues/dqs1_pairset_local_first.yaml validate`
- `.venv/bin/python tools/experiment_queue.py --queue configs/experiment_queues/dqs1_pairset_local_first.yaml observe --tail-lines 0`

## State Hygiene Observation

The current default DQS1 queue state was inspected read-only. It reports
`missing_step_count = 5` for the current YAML and `orphaned_step_count = 123`,
so the next execution action should be an explicit state reconciliation pass,
not an implicit worker launch. The observer now exposes this alongside the
idle `local_mlx` resource so both the stale-state and unused-resource issues are
visible from one command.

## Next Hooks

- Add MLX scorer/advisory nodes to the DQS1 local-first queue builder so local
  CPU and local MLX can run as distinct resources under the same queue worker.
- Promote queued candidate generation from one candidate YAML to a multi-candidate
  DAG fed by byte-shaving campaign units, X-ray/atom/master-gradient signals,
  and scorer-response calibration.
- Reconcile the stale DQS1 SQLite queue state with append-only rationale before
  running the worker against the default state path.
