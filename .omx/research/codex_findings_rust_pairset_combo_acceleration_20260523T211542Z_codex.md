# Codex Findings: Rust Pairset Combo Acceleration

UTC: 2026-05-23T21:15:42Z
Agent: Codex
Lane: codex_rust_combo_acquisition_accel_20260523

## Summary

The cross-family portfolio planner could synthesize learned multi-drop DQS1
pairset rows, but its search was first-order and Python-loop-bound. That was
too weak for the inverse-scorer/water-fill framing: measured drop-two rows
contain second-order synergy or antagonism that should alter which bucket is
filled next.

## Change

- Added explicit pairwise interaction terms to the pairset component-marginal
  model. Drop-two observations now estimate
  `observed_drop_two_delta - sum(drop_one_marginals)` per axis.
- Replaced prefix-only combo generation with beam-search combo generation over
  favorable drop-one rows, scoring each combo as first-order marginals plus
  measured pairwise interaction terms.
- Added `runtime-rs/crates/pairset-combo-planner`, a Rust/Rayon native CLI for
  the same combo search. Python remains the semantic oracle and fallback; the
  Rust binary is used only when explicitly available or already built.
- Added `tac.optimization.pairset_combo_rust_bridge` so queue/planner callers
  can use native parallel search without giving it score, dispatch, or
  promotion authority.

## Profile

Synthetic stress profile: 128 independent base-pair groups, 48 marginal rows
per group, combo counts `[2,3,4,5,8,13,16]`, beam width 64, pairwise
antagonism terms every five adjacent pairs.

- Rust release native path including subprocess JSON boundary: `0.362680s`.
- Python fallback path: `61.020026s`.
- Measured speedup including subprocess overhead: `168.25x`.
- Native and Python both emitted `896` combos.

## Authority

All new rows remain `score_claim=false`, `promotion_eligible=false`,
`rank_or_kill_eligible=false`, `ready_for_exact_eval_dispatch=false`, and
`promotable=false`. The native binary ranks planning candidates only; exact
auth-axis materialization, local controls, dispatch claims, and contest CPU/CUDA
eval remain separate gates.

## Verification

- `.venv/bin/python -m pytest src/tac/tests/test_cross_family_candidate_portfolio.py -q`
- `PACT_PAIRSET_COMBO_PLANNER_BIN=/Users/adpena/Projects/pact/runtime-rs/target/debug/pairset-combo-planner .venv/bin/python -m pytest src/tac/tests/test_cross_family_candidate_portfolio.py::test_portfolio_combo_search_uses_pairwise_interaction_terms src/tac/tests/test_cross_family_candidate_portfolio.py::test_portfolio_synthesizes_learned_multi_drop_candidates_from_local_component_marginals -q`
- `cargo test -p pairset-combo-planner`
- `cargo build -p pairset-combo-planner --release`
- `.venv/bin/python -m ruff check src/tac/optimization/cross_family_candidate_portfolio.py src/tac/optimization/pairset_combo_rust_bridge.py src/tac/tests/test_cross_family_candidate_portfolio.py`
