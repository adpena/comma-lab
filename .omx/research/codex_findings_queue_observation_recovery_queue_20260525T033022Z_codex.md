# Codex Findings: Queue Observation Recovery Queue

Date: 2026-05-25T03:30:22Z
Agent: Codex
Lane: `codex_queue_observation_recovery_feedback_20260525`
Research-only: false
Score authority: false

## Finding

The prior queue-observation feedback loop correctly detected when materializer
queue health must be recovered before another action-functional feedback
iteration, but the required recovery actions were still only report fields. That
left a manual gap: a required rewind/retire/observe operation could be visible
in the run artifact without becoming owned executable work.

## Landing

- Added `build_queue_observation_recovery_queue(...)` in
  `src/comma_lab/scheduler/queue_feedback_replan_policy.py`.
- The recovery queue is an `experiment_queue.v1` child queue with:
  - paused controls by default;
  - local CPU only;
  - one command step per required recovery action;
  - strict command validation limited to `tools/experiment_queue.py` actions
    `init`, `observe`, `retire-orphans`, and `rewind`;
  - false-authority metadata and exact-eval dispatch blockers.
- Wired `tools/run_byte_shaving_materializer_campaign.py` to emit
  `queue_observation_recovery_queue.json`, state path metadata, blockers, and
  staircase child artifacts when the replan policy selects
  `recover_queue_health`.
- Wired `tools/operator_briefing.py` to count recovery queues, display queue
  recovery queue readiness, and make the recovery queue initialization the next
  operator command while recovery is outstanding.

## Guardrail

The recovery queue refuses to materialize unless all of these are true:

- policy schema is `queue_feedback_replan_policy.v1`;
- policy decision is `recover_queue_health`;
- policy is `ready_for_queue_health_recovery`;
- the embedded recovery plan is required and command-backed;
- no forbidden true authority fields are present;
- every recovery command targets the exact queue/state paths from the recovery
  plan.

## Verification

Commands run:

```bash
.venv/bin/python -m ruff check src/comma_lab/scheduler/queue_feedback_replan_policy.py src/comma_lab/scheduler/__init__.py tools/run_byte_shaving_materializer_campaign.py src/tac/optimization/inverse_steganalysis_acquisition.py src/tac/tests/test_queue_feedback_replan_policy.py src/tac/tests/test_inverse_steganalysis_acquisition.py
PYTHONPATH=. .venv/bin/pytest src/tac/tests/test_queue_feedback_replan_policy.py src/tac/tests/test_inverse_steganalysis_acquisition.py -q
PYTHONPATH=. .venv/bin/pytest src/tac/tests/test_operator_briefing.py src/tac/tests/test_byte_shaving_materializer_campaign_runner.py::test_materializer_campaign_runner_executes_no_paid_inverse_scorer_chain_and_handoff src/tac/tests/test_byte_shaving_materializer_campaign_runner.py::test_materializer_campaign_feedback_followup_queue_blocks_required_queue_recovery -q
```

Observed results:

- ruff: all checks passed
- policy/acquisition tests: 61 passed
- operator/runner tests: 34 passed

## Next Integration

This closes the manual recovery-action gap, but it intentionally does not
auto-run queue-state mutation. The next useful integration is an autopilot
resume policy that can initialize, resume, run, and re-observe this recovery
queue under explicit local-only custody, then feed the resulting healthy
observation back into the continuation queue gate.
