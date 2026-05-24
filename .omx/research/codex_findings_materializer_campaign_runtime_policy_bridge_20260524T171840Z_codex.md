# Codex Findings: Materializer Campaign Runtime Policy Bridge

Date: 2026-05-24T17:18:40Z
Agent: Codex
Lane: codex_materializer_campaign_e2e_smoke_20260524

## Finding

The materializer campaign runner could create and run a local
`experiment_queue.v1`, but it still treated queue state and local runtime
policy as external/manual concerns. That left real no-paid-dispatch inverse
scorer campaigns vulnerable to duplicated default state paths and static
concurrency knobs.

## Landed Changes

- Added `--queue-state` to `tools/run_byte_shaving_materializer_campaign.py` and
  threaded it through validate/init/run-worker/observe/performance.
- Added `--derive-runtime-policy` and `--apply-runtime-policy` bridge support.
- The runner can now call `tools/experiment_queue.py runtime-policy`, persist
  `scheduler_runtime_policy.v1`, optionally run against a policy-applied queue,
  and record both artifacts in `materializer_campaign_run.json`.
- Applying runtime policy defaults to concurrency-only so existing queue state
  does not churn step definition hashes through timeout changes unless the
  operator explicitly passes `--runtime-policy-apply-timeouts`.
- `apply_scheduler_runtime_policy(...)` now recursively rejects nested truthy
  authority fields inside runtime-policy payloads before applying queue changes.
- Fixed the runner e2e test to assert the current
  `byte_shaving_materializer_contexts.v1` field `blocked_context_count`.

## Verification

- `.venv/bin/python -m ruff check tools/run_byte_shaving_materializer_campaign.py src/tac/tests/test_byte_shaving_materializer_campaign_runner.py src/comma_lab/scheduler/experiment_queue.py src/tac/tests/test_experiment_queue.py`
  passed.
- `.venv/bin/python -m pytest src/tac/tests/test_byte_shaving_materializer_campaign_runner.py -q`
  passed: 28 tests.
- `.venv/bin/python -m pytest src/tac/tests/test_byte_shaving_materializer_campaign_runner.py src/tac/tests/test_experiment_queue.py src/tac/tests/test_experiment_queue_observer.py src/tac/tests/test_staircase_dag.py src/tac/tests/test_byte_shaving_campaign_queue.py src/tac/tests/test_materializer_chain_harvest_scheduler.py -q`
  passed: 199 tests.
- `.venv/bin/python -m pytest src/tac/tests/test_experiment_queue.py src/tac/tests/test_experiment_queue_observer.py src/tac/tests/test_staircase_dag.py src/tac/tests/test_scheduler_cli.py src/tac/tests/test_ssh_experiment_queue_executor.py src/tac/tests/test_byte_shaving_campaign.py src/tac/tests/test_byte_shaving_campaign_queue.py src/tac/tests/test_byte_shaving_materializer_campaign_runner.py src/tac/tests/test_materializer_chain_harvest_scheduler.py -q`
  passed: 264 tests.

## Remaining Work

Run the first real local campaign smoke using MLX/scorer-response artifacts plus
a real template archive, with `--apply-runtime-policy` enabled and exact auth
still blocked until local chain and readiness proofs clear.
