# Codex Findings: Final-Rate Portfolio Preflight

## Context

The queue-owned final-rate attack had a remaining manual-control gap: post-feedback
child queues could be executed from a feedback artifact map even if the refresh did
not also prove that the executable materializer portfolio was covered. That left
`dqs1_pairset_drop_pair` and `inverse_scorer_cell_candidate_v1` dependent on
operator memory rather than a reusable, auditable preflight.

## Landing

- Added `frontier_final_rate_attack_portfolio_coverage_preflight.v1` in
  `src/comma_lab/scheduler/frontier_final_rate_attack_autoloop.py`.
- The preflight verifies the `frontier_rate_attack_portfolio_coverage.v1` artifact,
  false-authority fields, bounded follow-up readiness, bound pathway count, and
  required deferred bindings for DQS1 and inverse-scorer target kinds.
- `execute_post_feedback_child_queues(..., require_portfolio_coverage=True)` now
  refuses to select or run child queues when portfolio coverage is missing or
  invalid, while preserving a false-authority custody report with blockers.
- `tools/build_frontier_final_rate_attack_queue.py` now requires the portfolio
  coverage preflight by default for the real post-execute feedback child-queue
  execution path. A legacy advisory escape hatch exists as
  `--allow-missing-post-feedback-portfolio-coverage`.

## Verification

- `.venv/bin/ruff check src/comma_lab/scheduler/frontier_final_rate_attack_autoloop.py tools/build_frontier_final_rate_attack_queue.py src/tac/tests/test_frontier_rate_attack_bootstrap.py`
- `.venv/bin/pytest src/tac/tests/test_frontier_rate_attack_bootstrap.py -q -k 'post_feedback_child_queue_execution'`
- `.venv/bin/pytest src/tac/tests/test_frontier_rate_attack_bootstrap.py -q`

## Authority

This is local scheduler custody and planner hardening only.

- `score_claim=false`
- `promotion_eligible=false`
- `rank_or_kill_eligible=false`
- `ready_for_exact_eval_dispatch=false`

