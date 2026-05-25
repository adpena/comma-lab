# Codex Findings - Feedback Dry Candidate Widening

UTC: 2026-05-25T06:36:18Z
Lane: `codex_feedback_dry_candidate_widening_20260525`

## Finding

The queue feedback replan policy could treat a successful feedback follow-up
action functional as authority to run the next materializer iteration even when
the action surface contained cells but the water bucket selected zero cells.
That state is not a materializer success frontier. It means the current inverse
candidate generation/materializer family is dry for the measured feedback and
should widen the candidate space instead of continuing the same leaf loop.

## Landing

- Added `widen_inverse_candidate_generation` as an explicit non-continuing
  policy decision.
- Loaded and summarized the feedback action functional under the queue feedback
  policy, including selected cell count, blocked cell count, archive-delta
  blocked count, and false-authority fields.
- Routed dry `cell_count > 0 && selected_count == 0` feedback to candidate
  widening with hints for inverse-scorer refresh, larger unit search, and
  compiled receiver/materializer transforms.
- Surfaced the same state in `tools/operator_briefing.py` as
  `NEEDS_CANDIDATE_WIDENING`, with aggregate and latest-row counters.

## Verification

- `.venv/bin/python -m ruff check src/comma_lab/scheduler/queue_feedback_replan_policy.py tools/operator_briefing.py src/tac/tests/test_queue_feedback_replan_policy.py src/tac/tests/test_operator_briefing.py`
- `.venv/bin/python -m py_compile src/comma_lab/scheduler/queue_feedback_replan_policy.py tools/operator_briefing.py`
- `.venv/bin/python -m pytest src/tac/tests/test_queue_feedback_replan_policy.py src/tac/tests/test_operator_briefing.py -q`

Result: 62 tests passed.

## Next Step

Wire the `widen_inverse_candidate_generation` recommended action into the next
inverse-scorer/acquisition candidate generator so dry materializer feedback
automatically expands the receiver/compiler transform space instead of waiting
for manual campaign redesign.
