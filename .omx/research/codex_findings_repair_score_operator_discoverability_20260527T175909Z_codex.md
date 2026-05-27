# Codex Findings: Repair Score Operator Discoverability

UTC: 2026-05-27T17:59:09Z

## Finding

The repair-waterfill action-functional queue was visible in operator briefing,
but the downstream repair campaign score and stackability-learning queue was not.
That made the loop harder to see from the normal dashboard even after the queue
itself was automated.

## Landing

`tools/operator_briefing.py` now surfaces `repair_campaign_score_queue` alongside
`repair_budget_waterfill_queue`:

- queue path;
- queue status;
- queued experiment count;
- frozen experiment count;
- aggregate queue count in the frontier feedback cycle summary;
- formatted text line for the latest refresh.

The autonomous queue action-order regression is also covered by a focused test:
if the repair score child appears before the waterfill child in scheduler action
input, the generated run step still depends on the waterfill run step.

## Authority

This is dashboard and queue-control visibility only. It grants no score,
promotion, rank/kill, budget-spend, or dispatch authority.

## Verification

- `ruff check tools/operator_briefing.py src/tac/tests/test_operator_briefing.py src/tac/tests/test_frontier_rate_attack_feedback.py`
- `py_compile` on the same three files
- `pytest src/tac/tests/test_operator_briefing.py::test_operator_briefing_surfaces_repair_waterfill_action_functional_queue src/tac/tests/test_frontier_rate_attack_feedback.py::test_autonomous_queue_repair_score_waits_for_waterfill_independent_of_action_order -q`
- `tools/operator_briefing.py --json`
