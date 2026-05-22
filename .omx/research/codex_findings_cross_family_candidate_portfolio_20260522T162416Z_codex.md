# Codex Findings: Cross-Family Candidate Portfolio

Date: 2026-05-22T16:24:16Z

## Verdict

PROCEED. The cross-family portfolio planner is validated as a planning-only
queue composer across MLX spend-triage rows, DQS1 pairset acquisition rows, and
byte-closed outside-class HFV2 manifests.

## What Landed

- `src/tac/optimization/cross_family_candidate_portfolio.py`
  - Fuses candidate rows across families into one Bayesian design portfolio.
  - Preserves false-authority fields at source, portfolio, row, and policy
    levels.
  - Keeps exact archive custody readiness advisory-only; dispatch still requires
    separate lane claim, controls, and auth-axis gating.
- `tools/plan_cross_family_candidate_portfolio.py`
  - CLI wrapper with source artifact custody hashes and markdown report output.
- `src/tac/tests/test_cross_family_candidate_portfolio.py`
  - Source fusion, false-authority refusal, custody-advisory semantics, and CLI
    determinism coverage.

## Empirical Planning Artifact

Generated ignored operator artifact:

- `experiments/results/cross_family_candidate_portfolio_20260522T162416Z/portfolio.json`
- `experiments/results/cross_family_candidate_portfolio_20260522T162416Z/portfolio.md`

Inputs:

- `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/mlx_effective_spend_triage_observed_window_selection_top32.json`
- `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/pairset_acquisition/dqs1_pairset_acquisition.json`
- `experiments/results/hfv2_sparse_sidecar_magic_bin_candidate_20260522T160241Z/hfv2_sparse_manifest.json`

Observed summary:

- `candidate_count_before_top_k = 208`
- `ranked_candidate_count = 40`
- `candidate_archive_custody_ready_count = 1`
- `recommended_next_candidate_id = pairset_diversity_k002`
- `recommended_next_action = materialize_pairset_archive_and_run_local_controls`
- `score_claim = false`
- `ready_for_exact_eval_dispatch = false`

The Bayesian rank places the custody-ready HFV2 magic-bin row first due outside
class information gain, but the operator action queue correctly demotes it
behind unblocked DQS1 pairset materialization because its exact CPU/CUDA
anchors and lane claim are missing.

## Verification

- `.venv/bin/python -m pytest src/tac/tests/test_cross_family_candidate_portfolio.py -q`
- `.venv/bin/python -m ruff check src/tac/optimization/cross_family_candidate_portfolio.py tools/plan_cross_family_candidate_portfolio.py src/tac/tests/test_cross_family_candidate_portfolio.py`
- `.venv/bin/python -m pytest src/tac/tests/test_cross_family_candidate_portfolio.py src/tac/tests/test_mlx_effective_spend_triage_selection.py -q`

## Residual Risk

This planner does not materialize archives, claim lanes, dispatch auth evals, or
make score claims. It is an operator queue composer for choosing the next
byte-closed materialization/control step.
