# Codex Findings: Cross-Family Manual MLX Boundary Guard

UTC: 2026-05-23T08:35:05Z
Lane: `lane_codex_cross_family_manual_mlx_boundary_guard_20260523`
Agent: Codex

## Finding

Sidecar adversarial review found a manual-candidate boundary leak in
`tac.optimization.cross_family_candidate_portfolio`: `--candidate-json` rows
could set `source_kind=mlx_effective_spend_triage_selection` and inherit the
formal MLX operator action without passing through the typed
`mlx_effective_spend_triage_candidate_selection.v1` schema. MLX-like manual
rows could also omit the normalized full-video objective and rely only on a
caller-supplied `predicted_score_mean`.

That is not score authority, but it can distort the exact-eval action queue and
create signal loss by making a local/window MLX row look like a formally gated
normalized planner row.

## Landing

Added a fail-closed manual-candidate boundary:

- manual rows cannot use reserved formal `source_kind` values such as
  `mlx_effective_spend_triage_selection`;
- MLX-like manual rows must carry a valid normalized full-video objective;
- manual MLX `predicted_score_mean` must match
  `incumbent_score + projected_full_video_delta_vs_baseline_score`;
- accepted manual MLX rows stay `source_kind=manual_candidate`, so the operator
  action remains manual review rather than formal MLX materialization.

Added library and CLI regressions for missing normalized objective and reserved
source-kind spoofing.

## Verification

Commands run:

```bash
.venv/bin/python -m pytest -q src/tac/tests/test_cross_family_candidate_portfolio.py src/tac/tests/test_plan_cross_family_candidate_portfolio_cli.py
.venv/bin/python -m pytest -q src/tac/tests/test_mlx_effective_spend_triage_selection.py src/tac/tests/test_cross_family_candidate_portfolio.py src/tac/tests/test_plan_cross_family_candidate_portfolio_cli.py src/tac/tests/test_decoder_q_selective_window_bridge.py src/tac/tests/test_scorer_response_dataset.py src/tac/cathedral_consumers/distilled_scorer_surrogate_canonical_equation_consumer/tests/test_consumer.py
.venv/bin/ruff check src/tac/optimization/cross_family_candidate_portfolio.py src/tac/tests/test_cross_family_candidate_portfolio.py src/tac/tests/test_plan_cross_family_candidate_portfolio_cli.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/review_tracker.py policy-check src/tac/optimization/mlx_effective_spend_triage_selection.py src/tac/optimization/cross_family_candidate_portfolio.py src/tac/tests/test_mlx_effective_spend_triage_selection.py src/tac/tests/test_cross_family_candidate_portfolio.py src/tac/tests/test_plan_cross_family_candidate_portfolio_cli.py
.venv/bin/python tools/lane_maturity.py validate
```

Results:

- focused cross-family portfolio and CLI suite: 18 passed
- normalized objective / scorer-response / bridge / portfolio / CLI downstream
  suite: 148 passed
- ruff: passed
- review tracker: policy-check clean
- lane registry: 1172 lanes validated cleanly

## Integration Notes

The formal MLX path remains `--mlx-selection`, which validates the selection
schema, false-authority markers, normalized objective, and exact-auth-required
boundary before producing the MLX operator action. Manual rows remain available
for exploratory portfolio planning, but cannot impersonate formal MLX source
kinds or carry unnormalized MLX window evidence into ranking.

No `.gitignore` change was needed in this landing: no new generated artifact
root or large local cache namespace was introduced.
