# Codex Findings: Experiment Queue Runtime Policy CLI

Date: 2026-05-24T17:13:27Z
Agent: Codex
Lane: codex_experiment_queue_runtime_policy_cli_20260524

## Finding

`scheduler_runtime_policy.v1` was visible through observer output, but normal
operator queue flows still lacked a direct way to derive the policy, persist it,
and generate a policy-applied queue definition without hand-editing JSON. That
kept the local throughput loop too manual for large inverse-scorer/materializer
campaigns.

## Landed Changes

- Added `tools/experiment_queue.py runtime-policy`.
- The command opens SQLite state read-only, derives the advisory scheduler
  runtime policy, and can write a guarded policy artifact.
- It can also write a separate policy-applied queue definition that raises local
  CPU concurrency, caps IO-heavy pressure, preserves local accelerator defaults,
  and leaves cloud resource limits untouched.
- Output writes use guarded artifact semantics: no overwrite unless the caller
  supplies the expected existing SHA.
- Exported `SCHEDULER_RUNTIME_POLICY_SCHEMA`,
  `derive_scheduler_runtime_policy`, and `apply_scheduler_runtime_policy` from
  `comma_lab.scheduler`.

## Verification

- `.venv/bin/python -m ruff check tools/experiment_queue.py src/comma_lab/scheduler/__init__.py src/tac/tests/test_experiment_queue.py`
  passed.
- `.venv/bin/python -m pytest src/tac/tests/test_experiment_queue.py::test_experiment_queue_cli_runtime_policy_writes_guarded_artifacts -q`
  passed.
- `.venv/bin/python -m pytest src/tac/tests/test_experiment_queue.py src/tac/tests/test_experiment_queue_observer.py src/tac/tests/test_staircase_dag.py src/tac/tests/test_scheduler_cli.py src/tac/tests/test_ssh_experiment_queue_executor.py src/tac/tests/test_byte_shaving_campaign.py src/tac/tests/test_byte_shaving_campaign_queue.py src/tac/tests/test_byte_shaving_materializer_campaign_runner.py src/tac/tests/test_materializer_chain_harvest_scheduler.py -q`
  passed: 260 tests.

## Remaining Work

The next step is to have the materializer campaign runner optionally call this
policy path when it emits an `experiment_queue.v1` for local execution, so the
real no-paid-dispatch inverse-scorer campaign can saturate local CPU/MLX/IO
capacity from telemetry instead of fixed knobs.
