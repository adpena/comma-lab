# Codex Findings: Scheduler Runtime Policy Rerun Addendum

Date: 2026-05-24T17:04:39Z
Agent: Codex
Lane: codex_scheduler_runtime_policy_20260524

## Finding

This addendum preserves the later lane-maturity memory pointer created during
pre-commit hardening. The primary finding is recorded in
`codex_findings_scheduler_runtime_policy_20260524T170409Z_codex.md`; this memo
adds the current rerun evidence and keeps the append-only audit trail
resolvable.

## Landed Changes

- No additional implementation beyond the primary scheduler runtime policy memo.
- Confirms the policy remains advisory-only and keeps false score, promotion,
  rank/kill, and exact-dispatch authority.

## Verification

- `.venv/bin/python -m ruff check src/comma_lab/scheduler/experiment_queue.py src/comma_lab/scheduler/experiment_queue_observer.py src/tac/tests/test_experiment_queue.py src/tac/tests/test_experiment_queue_observer.py`
  passed.
- `.venv/bin/python -m pytest src/tac/tests/test_experiment_queue.py src/tac/tests/test_experiment_queue_observer.py -q`
  passed: 65 tests.
- `.venv/bin/python -m pytest src/tac/tests/test_experiment_queue.py src/tac/tests/test_experiment_queue_observer.py src/tac/tests/test_staircase_dag.py src/tac/tests/test_scheduler_cli.py src/tac/tests/test_ssh_experiment_queue_executor.py src/tac/tests/test_byte_shaving_campaign.py src/tac/tests/test_byte_shaving_campaign_queue.py src/tac/tests/test_byte_shaving_materializer_campaign_runner.py src/tac/tests/test_materializer_chain_harvest_scheduler.py -q`
  passed: 259 tests.
- `git diff --check` passed.

## Remaining Work

The next integration step is to make local executors consume the advisory policy
under an explicit flag or generated queue rewrite, then record before/after
throughput telemetry for inverse-scorer materializer campaigns. This policy is
execution-control signal only; it is not score authority and does not bypass
contest-axis exact eval gates.
