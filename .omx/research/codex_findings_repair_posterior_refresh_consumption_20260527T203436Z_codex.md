# Codex Findings: Repair Posterior Refresh Consumption

UTC: 2026-05-27T20:34:36Z
Agent: Codex

## Verdict

The repair campaign posterior is now consumed by the normal feedback-refresh
surfaces, not only by the repair scorer worker. Both the standalone refresh CLI
and the reusable feedback-cycle writer expose the current posterior prior
summary, including acquisition follow-up routes, inside the
`repair_campaign_score_queue_summary`.

## What Changed

- Added a public `build_repair_campaign_posterior_prior_summary(...)` helper so
  the same posterior projection used by `score_repair_campaign(...)` can be
  reused by refresh/autoloop code without duplicating policy logic.
- Wired `frontier_rate_attack_feedback_cycle.write_frontier_refresh_artifacts`
  to pass the default repair stackability posterior into
  `build_repair_campaign_score_queue(...)`.
- Added posterior summary fields to repair score queue refresh summaries:
  `campaign_scorer_uses_posterior_priors`,
  `posterior_prior_summary`,
  `posterior_acquisition_followup_route_count`, and
  `posterior_acquisition_followup_routes`.
- Kept all posterior-derived rows false-authority and planning-only.

## Mathematical Role

This moves the posterior from a passive JSONL ledger toward an action-functional
state term. The feedback refresh now carries a deterministic projection from
blocked or measured repair observations into acquisition pressure over evidence
surfaces: targeted component response harvest, receiver-closed byte credit,
local MLX custody repair, exact-axis replay, and missing repair artifacts.

The projection is still local-planning signal. It does not claim score, promote,
dispatch, spend, or rank/kill.

## Verification

- `.venv/bin/ruff check src/tac/optimization/repair_campaign_scorer.py src/tac/tests/test_repair_campaign_scorer.py src/comma_lab/scheduler/frontier_rate_attack_feedback_cycle.py tools/build_frontier_rate_attack_feedback_refresh.py`
- `.venv/bin/python -m py_compile src/tac/optimization/repair_campaign_scorer.py src/comma_lab/scheduler/frontier_rate_attack_feedback_cycle.py tools/build_frontier_rate_attack_feedback_refresh.py`
- `.venv/bin/python -m pytest src/tac/tests/test_repair_campaign_scorer.py -q`
- `.venv/bin/python -m pytest src/tac/tests/test_frontier_rate_attack_feedback.py -q`
- `.venv/bin/python -m pytest src/tac/tests/test_repair_campaign_scorer.py src/tac/tests/test_repair_campaign_score_queue.py src/tac/tests/test_frontier_rate_attack_bootstrap.py -q`
- `.venv/bin/python tools/lane_maturity.py validate`
- Review policy checks on touched source/test/tool files: 0 violations.

## Remaining Work

The next concrete actuator is to convert posterior acquisition routes into
bounded follow-up queue selection priority, while preserving the distinction
between artifact evidence surfaces and executable child queues.
