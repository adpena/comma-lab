# Codex Findings: Scheduler Runtime Policy

Date: 2026-05-24T17:04:09Z
Agent: Codex
Lane: codex_scheduler_runtime_policy_20260524

## Finding

The experiment queue already recorded elapsed time, timeout, resource kind,
log-size, and artifact telemetry, but execution controls still relied mostly
on static `controls.max_concurrency` and per-step timeout values. That left
local CPU/MLX/IO-heavy work under-sized for a large local machine or brittle
when a queue hit timeouts.

## Landed Changes

- Added `scheduler_runtime_policy.v1` derivation from canonical queue telemetry.
- The policy recommends local CPU concurrency from detected machine capacity,
  caps `local_io_heavy` backpressure, preserves cloud-resource limits, and
  backs off resource lanes with timeout/failure pressure.
- Timeout recommendations use observed P95 runtime plus a multiplier while never
  lowering below the queue's current declared timeout envelope.
- Added an explicit `apply_scheduler_runtime_policy(...)` helper to return a
  normalized queue definition with local advisory concurrency/timeouts applied.
- Exposed the policy in `observe_experiment_queue(...)` and Markdown rendering.
- The policy is telemetry/control-plane only and carries false authority fields:
  no score claim, no promotion, no rank/kill, no exact-dispatch authorization.

## Verification

- `.venv/bin/python -m pytest src/tac/tests/test_experiment_queue.py::test_scheduler_runtime_policy_derives_advisory_limits_and_timeouts src/tac/tests/test_experiment_queue_observer.py::test_observer_surfaces_read_only_performance_telemetry -q`
  passed.
- `.venv/bin/python -m pytest src/tac/tests/test_experiment_queue.py src/tac/tests/test_experiment_queue_observer.py src/tac/tests/test_staircase_dag.py -q`
  passed: 86 tests.
- `.venv/bin/python -m ruff check src/comma_lab/scheduler/experiment_queue.py src/comma_lab/scheduler/experiment_queue_observer.py src/tac/tests/test_experiment_queue.py src/tac/tests/test_experiment_queue_observer.py`
  passed.
- `git diff --check` passed.

## Remaining Work

The next scheduler step is to make the materializer campaign runner optionally
emit and apply this policy before no-paid-dispatch local campaign execution.
The next frontier step remains a real MLX/scorer-response plus template-archive
campaign smoke that auto-generates inverse-scorer artifact maps, materializes
chain candidates, harvests exact-readiness blockers, and only then requests
contest-axis auth evaluation.
