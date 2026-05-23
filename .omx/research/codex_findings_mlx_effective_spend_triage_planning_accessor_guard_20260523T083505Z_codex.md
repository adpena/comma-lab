# Codex Findings: MLX Effective Spend-Triage Planning Accessor Guard

UTC: 2026-05-23T08:35:05Z
Lane: `lane_codex_mlx_effective_spend_triage_planning_accessor_guard_20260523`
Agent: Codex

## Finding

`tac.optimization.mlx_effective_spend_triage_selection` already recomputed and
ranked by normalized full-video MLX objective fields, but the strict candidate
filter still read raw/window aliases directly for `delta_vs_baseline_score`,
`scorer_delta_vs_baseline`, `observed_scorer_gain_vs_baseline`, and
`byte_budget_margin_vs_break_even`.

That left one remaining authority leak in the spend-triage chain: a valid
normalized full-video planning row could still be accepted or rejected based on
local singleton/window aliases instead of the canonical MLX planning accessor.

## Landing

Changed the selector to route MLX target semantics through
`scorer_response_planning_value_for_target`, preserving the raw/window values
only as provenance in the selected row. The emitted selection manifest now
records:

- `selection_planning_value_accessor=scorer_response_planning_value_for_target`
- `selection_planning_value_scope=normalized_full_video`
- recomputed normalized full-video gain, projected delta, break-even bytes, and
  normalized byte-budget margin

Added a regression where the raw/window total and scorer deltas are deliberately
positive while the normalized full-video objective is a calibrated improvement.
The row remains selectable, and the old raw-alias blocker names do not appear.

Sidecar adversarial review caught an additional option-surface trapdoor:
`require_prediction_negative=True` with `prediction_field=delta_vs_baseline_score`
was still reading the raw alias. That path now also resolves planner-target
aliases through `scorer_response_planning_value_for_target`, and selected rows
record the prediction value accessor/scope.

## Verification

Commands run:

```bash
.venv/bin/python -m pytest -q src/tac/tests/test_mlx_effective_spend_triage_selection.py
.venv/bin/python -m pytest -q src/tac/tests/test_mlx_effective_spend_triage_selection.py src/tac/tests/test_cross_family_candidate_portfolio.py src/tac/tests/test_decoder_q_selective_window_bridge.py src/tac/tests/test_scorer_response_dataset.py src/tac/cathedral_consumers/distilled_scorer_surrogate_canonical_equation_consumer/tests/test_consumer.py
.venv/bin/python -m pytest -q src/tac/tests/test_mlx_effective_spend_triage_selection.py src/tac/tests/test_cross_family_candidate_portfolio.py src/tac/tests/test_plan_cross_family_candidate_portfolio_cli.py src/tac/tests/test_decoder_q_selective_window_bridge.py src/tac/tests/test_scorer_response_dataset.py src/tac/cathedral_consumers/distilled_scorer_surrogate_canonical_equation_consumer/tests/test_consumer.py
.venv/bin/ruff check src/tac/optimization/mlx_effective_spend_triage_selection.py src/tac/tests/test_mlx_effective_spend_triage_selection.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/review_tracker.py scan
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/review_tracker.py mark-file src/tac/optimization/mlx_effective_spend_triage_selection.py --reviewer codex --status reviewed
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/review_tracker.py mark-file src/tac/tests/test_mlx_effective_spend_triage_selection.py --reviewer codex --status reviewed
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/review_tracker.py policy-check src/tac/optimization/mlx_effective_spend_triage_selection.py src/tac/tests/test_mlx_effective_spend_triage_selection.py
.venv/bin/python tools/lane_maturity.py validate
```

Results:

- focused selector suite: 15 passed
- focused selector suite after adversarial option-surface fix: 16 passed
- normalized objective / scorer-response / bridge / portfolio downstream suite:
  142 passed
- normalized objective / scorer-response / bridge / portfolio / CLI downstream
  suite after cross-family boundary guard: 148 passed
- ruff: passed
- review tracker: policy-check clean
- lane registry: 1172 lanes validated cleanly

## Integration Notes

This closes the MLX effective-spend selector side of the normalized-objective
guard chain. Remaining high-value consumers to keep auditing are raw advisory
tools that predate the scorer-response dataset and still operate in explicitly
local/window scope, especially decoder-q advisory batch summarizers and older
waterbucket candidate planners. Those should either adopt the same canonical
planning accessor when consuming scorer-response rows or carry an explicit
`local_window_only_not_spend_triage_authority` tag.

No `.gitignore` change was needed in this landing: no new generated artifact
root or large local cache namespace was introduced.
