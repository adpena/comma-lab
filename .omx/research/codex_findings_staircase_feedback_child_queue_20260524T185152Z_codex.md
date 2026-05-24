# Codex Findings: Staircase Feedback Child Queue

Date: 2026-05-24T18:51:52Z

## Verdict

The materializer campaign runner now preserves queue-feedback replans as a typed
staircase child queue edge instead of leaving the follow-up queue as a loose
artifact. The child queue remains paused and planning-only; it is not score,
promotion, rank/kill, or paid-dispatch authority.

## Landed Integration

- `src/comma_lab/scheduler/staircase_dag.py` accepts
  `staircase_dependent_queue_ref.v1` entries and carries them through normalized
  DAG and dispatch-plan artifacts.
- `tools/run_byte_shaving_materializer_campaign.py` emits a child staircase DAG,
  child dispatch plan, and dependent-queue reference when a queue-feedback
  replan follow-up queue is emitted.
- The parent staircase DAG can include that dependent queue reference without
  scheduling the paused child queue as work.

## Safeguards

- Dependent queue refs run through `require_no_truthy_authority_fields`.
- Normalized dependent queue refs are wrapped with the proxy evidence boundary.
- Child feedback queues keep `controls.mode=paused`.
- The bridge artifact declares planning-only allowed use and forbids score,
  promotion, rank/kill, and paid-dispatch authority.

## Verification

- `.venv/bin/python -m pytest src/tac/tests/test_staircase_dag.py::test_staircase_dag_carries_dependent_queue_refs_without_authority src/tac/tests/test_staircase_dag.py::test_staircase_dag_rejects_dependent_queue_ref_authority_leak src/tac/tests/test_byte_shaving_materializer_campaign_runner.py::test_materializer_campaign_runner_executes_no_paid_inverse_scorer_chain_and_handoff -q`
- `.venv/bin/python -m ruff check src/comma_lab/scheduler/staircase_dag.py tools/run_byte_shaving_materializer_campaign.py src/tac/tests/test_staircase_dag.py src/tac/tests/test_byte_shaving_materializer_campaign_runner.py`

## Remaining Work

This closes the artifact-orphan gap for feedback replan queues. The next layer is
actuation policy: an explicit operator/autopilot resume path that can transition
the child queue from paused to runnable after exact queue-control and authority
checks pass.
