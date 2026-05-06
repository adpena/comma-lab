# PR106 YShift Scorer-Backed Selector - 2026-05-06 Codex

## Scope

This tranche adds the deterministic local reducer needed by a scorer-backed
CUDA yshift search. It does not load scorers and does not dispatch.

## Patch

- `experiments/build_pr106_yshift_sidechannel.py` now provides:
  - `build_yshift_candidate_grid(radius=3)`
  - `choose_yshift_candidates_from_scores(score_table, candidates, require_improvement=True)`
- The selector requires finite measured scores, exactly one all-zero no-op
  candidate, and keeps the no-op candidate unless another candidate strictly
  improves the measured objective.
- `src/tac/tests/test_pr106_yshift_sidechannel.py` covers deterministic grid
  shape/order, no-op retention, strict improvement, and NaN rejection.

## Evidence Discipline

Evidence grade: `empirical` local selector/custody only.

The score table consumed by this reducer must come from the authorized CUDA
compress-time scorer path before it can affect a promoted archive. This patch
is scaffolding for that search, not score evidence.
