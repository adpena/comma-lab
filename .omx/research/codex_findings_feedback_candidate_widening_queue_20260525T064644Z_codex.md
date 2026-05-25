# Codex Findings - Feedback Candidate Widening Queue

UTC: 2026-05-25T06:46:44Z
Lane: `codex_feedback_dry_candidate_widening_20260525`

## Finding

The first dry-feedback guard correctly stopped a no-selected-cell action
functional from looping into another materializer iteration, but the widening
signal was still only advisory. That left a small orchestration gap: operator
briefing could say "widen candidate generation", but the queue layer had no
paused child queue ready to perform the local widening step.

## Landing

- Added `build_queue_feedback_candidate_widening_queue(...)` as a paused,
  local-only queue builder for `widen_inverse_candidate_generation`.
- Derived a deterministic widened action-functional command from the original
  feedback request command: distinct `.widened.json` / `.widened.md` outputs,
  doubled/increased `--inverse-scorer-max-units`, larger `--max-cells`, and no
  stale expected-output hashes.
- Wired `tools/run_byte_shaving_materializer_campaign.py` to emit
  `queue_feedback_candidate_widening_queue.json` when the policy is ready.
- Surfaced emitted widening queues in `tools/operator_briefing.py` aggregates
  and latest-row output.

## Verification

- `.venv/bin/python -m ruff check src/comma_lab/scheduler/queue_feedback_replan_policy.py tools/run_byte_shaving_materializer_campaign.py tools/operator_briefing.py src/tac/tests/test_queue_feedback_replan_policy.py src/tac/tests/test_operator_briefing.py`
- `.venv/bin/python -m py_compile src/comma_lab/scheduler/queue_feedback_replan_policy.py tools/run_byte_shaving_materializer_campaign.py tools/operator_briefing.py`
- `.venv/bin/python -m pytest src/tac/tests/test_queue_feedback_replan_policy.py -q`
- `.venv/bin/python -m pytest src/tac/tests/test_operator_briefing.py::test_byte_shaving_acquisition_summary_surfaces_feedback_candidate_widening src/tac/tests/test_operator_briefing.py::test_byte_shaving_acquisition_summary_surfaces_latest_local_queue -q`
- `.venv/bin/python -m pytest src/tac/tests/test_byte_shaving_materializer_campaign_runner.py::test_materializer_campaign_runner_executes_no_paid_inverse_scorer_chain_and_handoff src/tac/tests/test_byte_shaving_materializer_campaign_runner.py::test_materializer_campaign_feedback_followup_queue_blocks_non_action_command src/tac/tests/test_byte_shaving_materializer_campaign_runner.py::test_post_recovery_feedback_replan_emits_fresh_followup_and_continuation -q`

Result: 27 focused tests passed.

## Next Step

Run a real dry-feedback materializer campaign artifact through the new queue
builder and execute the paused widening queue under local CPU/MLX policy once a
fresh dry feedback action surface is available.
