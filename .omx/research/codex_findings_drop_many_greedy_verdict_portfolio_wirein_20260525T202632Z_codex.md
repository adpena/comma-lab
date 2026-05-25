# Codex Findings: Drop-Many Greedy Verdict Portfolio Wire-In

Captured at: 2026-05-25T20:26:32Z

## Finding

The DQS1 drop-many greedy independent reducer verdict was visible to operator
briefing, but it was not a typed input to the cross-family portfolio planner or
the canonical probe-outcomes ledger. That meant the optimizer could still treat
source-selector-inherited drop-many rows as ordinary pairset acquisition
candidates, even after the empirical greedy verdict deferred that independent
assumption.

## Landing

Added `dqs1_drop_many_greedy_verdict_feedback_model.v1` consumption to
`tac.optimization.cross_family_candidate_portfolio`.

The portfolio now:

- appends the Build-1c verdict to `.omx/state/probe_outcomes.jsonl` through
  `tac.probe_outcomes_ledger.register_probe_outcome` with explicit
  false-authority fields;
- ingests `dqs1_drop_many_build_1c_greedy_independent_heuristic_verdict.v1`
  payloads through the Python API and CLI;
- rejects any truthy score/promotion/dispatch authority in the verdict tree;
- marks independent `drop_many_beam_pairwise_interaction_waterfill` rows with
  a planning blocker when the verdict is `NEGATIVE`, `PARTIAL`, or Catalog 313
  `DEFER`;
- emits a hold action for those independent rows so DQS1 local-first queue
  selection skips them instead of materializing source-inherited independent
  drop-many candidates by accident;
- leaves interaction-aware successors such as learned component-marginal combos
  queueable for local controls;
- exposes the compiled verdict model in the portfolio observation-feedback
  block and CLI action summary.

## Authority Boundary

This is planning signal only. It does not claim score, promote, rank/kill, or
authorize exact-eval dispatch. Independent drop-many rows are blocked from
normal operator action priority until they are backed by interaction-aware,
component-marginal, pair-frame geometry, or inverse-scorer successor evidence.

## Verification

- `.venv/bin/ruff check src/tac/optimization/cross_family_candidate_portfolio.py tools/plan_cross_family_candidate_portfolio.py src/tac/tests/test_cross_family_candidate_portfolio.py`
- `.venv/bin/ruff check src/tac/optimization/cross_family_candidate_portfolio.py tools/plan_cross_family_candidate_portfolio.py src/tac/tests/test_cross_family_candidate_portfolio.py src/tac/tests/test_dqs1_local_first_queue_builder.py`
- `PYTHONPATH=. .venv/bin/pytest src/tac/tests/test_cross_family_candidate_portfolio.py src/tac/tests/test_plan_cross_family_candidate_portfolio_cli.py src/tac/tests/test_dqs1_local_first_queue_builder.py -q`

Result: 68 focused tests passed.

Repository-local smoke artifacts:

- `.omx/research/codex_drop_many_verdict_portfolio_wirein_smoke_20260525T202632Z/portfolio.json`
- `.omx/research/codex_drop_many_verdict_portfolio_wirein_smoke_20260525T202632Z/action_summary.json`
- `.omx/research/codex_drop_many_verdict_portfolio_wirein_smoke_20260525T202632Z/dqs1_followup_queue.json`
- `.omx/research/codex_drop_many_verdict_portfolio_wirein_smoke_20260525T202632Z/selected_pairset_acquisition.json`

Smoke results:

- The real greedy verdict compiled as active with
  `independent_greedy_status=deferred_requires_interaction_or_component_model`.
- 34 independent drop-many rows emitted hold actions.
- The generated 14-experiment DQS1 follow-up queue contains zero held drop-many
  candidates.
- `tools/experiment_queue.py ... validate` passed with 14 experiments and 84
  steps.

## Next Gap

The next higher-EV step is to feed the compiled component-marginal / pairwise
interaction model directly into DQS1 queue selection so learned multi-drop rows
become the default local-first follow-up when enough marginal evidence exists,
rather than only appearing in portfolio action ordering.
