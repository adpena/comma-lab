# Codex Findings: Frontier Portfolio Coverage Surface

UTC: 2026-05-27T21:44:35Z

## Finding

The final-rate bootstrap now accounts for archive-rate materializer coverage,
but the feedback cycle still exposed DQS1, inverse-steganalysis, operation-chain,
and repair-budget follow-up queues as separate artifacts. That left the operator
and later automation without one typed view of which deferred executable paths
were actually bound to queue-owned follow-up work.

## Fix Landed

- Added `frontier_rate_attack_portfolio_coverage.v1`.
- Wrote `frontier_rate_attack_portfolio_coverage.json` from
  `write_frontier_refresh_artifacts`.
- Bound the major queue-owned pathways into one false-authority surface:
  archive-rate materializer sweep, operation-chain compiler, DQS1 local-first
  feedback, inverse-steganalysis acquisition, and rate-savings-to-distortion
  repair budget.
- Added explicit deferred-target binding rows for `dqs1_pairset_drop_pair` and
  `inverse_scorer_cell_candidate_v1`.
- Wired an operator inspect command into feedback-refresh reports.

## Verification

- `ruff check src/comma_lab/scheduler/frontier_rate_attack_feedback_cycle.py
  src/tac/tests/test_frontier_rate_attack_feedback.py` passed.
- Focused feedback test passed:
  `test_frontier_feedback_compiler_turns_eureka_near_misses_into_beyond_drop_two_hints`.
- Full feedback test file passed: `57 passed in 25.35s`.
- A direct artifact smoke wrote a portfolio coverage file and verified false
  score/dispatch authority with two bound pathways.

## Next Integration Edge

Use this portfolio coverage surface as the preflight gate for bounded local
autoloops: the autoloop should refuse "all executable" campaign claims unless
archive-rate target coverage and portfolio coverage are both present, false
authority, and have no unclassified/deferred-unbound gaps for the requested
campaign mode.
