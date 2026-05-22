# Codex Findings: Cross-Family CLI Action Summary

Date: 2026-05-22T17:56:04Z

## Verdict

PROCEED. The cross-family portfolio CLI is now more operator-usable without
creating score, promotion, rank/kill, or dispatch authority.

## What Landed

- `tools/plan_cross_family_candidate_portfolio.py`
  - Adds `--summary-json-out` for a compact planning-only action handoff.
  - Adds `--top-actions N` to expose top operator next actions in stdout and
    summary artifacts.
  - Adds `--require-active-pairset-observation-model` to fail closed when the
    operator expects pairset observation-response learning but the model is not
    active.
  - Appends a CLI action summary to Markdown output, including observation
    response status and false-authority fields.
- `src/tac/tests/test_plan_cross_family_candidate_portfolio_cli.py`
  - Covers summary JSON/Markdown output and fail-closed inactive-model behavior.
- `docs/runbooks/cross_family_candidate_portfolio.md`
  - Documents the operator command, expected inputs/outputs, and authority
    boundary.
- `tac.optimization.cross_family_candidate_portfolio`
  - Scopes the exact pairset response model by selector family so
    `diversity_spaced` evidence does not leak into `prefix_variant` or
    `drop_one_from_best` candidates.
  - Records selected-pair identity verification counts in the model summary
    and caps regression-only extrapolations at the best observed exact-axis
    score.
- `tools/append_mlx_dynamic_sweep_observation.py`
  - Adds `--selected-pair-indices` so future exact pairset observations can
    carry candidate-identity evidence at append time.

## Live Smoke

Command:

```bash
.venv/bin/python tools/plan_cross_family_candidate_portfolio.py --incumbent-score 0.2053300290 --incumbent-score-by-axis contest_cpu=0.19202894881608987 --pairset-acquisition experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/pairset_acquisition/dqs1_pairset_acquisition.json --observation-jsonl experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/dynamic_learned_sweep/dqs1_dynamic_sweep_observations.jsonl --json-out /tmp/cross_family_cli_check_portfolio.json --md-out /tmp/cross_family_cli_check_portfolio.md --summary-json-out /tmp/cross_family_cli_check_action_summary.json --top-actions 3 --require-active-pairset-observation-model
```

Latest selector-scoped result:

- Output directory:
  `experiments/results/cross_family_candidate_portfolio/20260522T180032Z_observed_pairsets_selector_scoped`
- `pairset_observation_response_model.active = true`
- `active_selector_kinds = ["diversity_spaced"]`
- `inactive_selector_reasons.drop_one_from_best = "need_two_distinct_selected_pair_counts"`
- `axis_observation_counts.contest_cpu = 7`
- `identity_counts.candidate_id_match_count = 7`
- `identity_counts.selected_pair_indices_verified_count = 7`
- `updated_candidate_count = 7`
- selector-family leakage fixed after adversarial review: the response model is
  active for `diversity_spaced` only, inactive for selector kinds without two
  distinct exact observations, and regression-only predictions are capped at
  the best observed score.
- top stdout action after identity-verified regeneration:
  `pairset_drop_one_rank021_pair0371`
- top action: `materialize_pairset_archive_and_run_local_controls`
- all top-action authority fields remain false

## Verification

```bash
.venv/bin/python -m pytest src/tac/tests/test_cross_family_candidate_portfolio.py src/tac/tests/test_plan_cross_family_candidate_portfolio_cli.py src/tac/tests/test_mlx_dynamic_sweep_observations.py -q
.venv/bin/ruff check src/tac/optimization/cross_family_candidate_portfolio.py tools/plan_cross_family_candidate_portfolio.py src/tac/tests/test_cross_family_candidate_portfolio.py src/tac/tests/test_plan_cross_family_candidate_portfolio_cli.py src/tac/tests/test_mlx_dynamic_sweep_observations.py
```

Results:

- `24 passed`
- Ruff passed

## Authority

This landing only improves planning visibility. The summary output is still
blocked by `auth_axis_gate_required_before_dispatch`,
`portfolio_planning_only_requires_separate_lane_claim`, and
`score_claim_requires_contest_auth_eval`. It cannot claim a score, promote a
candidate, rank/kill a lane, launch a GPU job, or replace exact contest-axis
auth eval.
