# Codex Findings: MLX Effective Spend Triage CLI Enforcement

UTC: 2026-05-22T09:24:44Z
Lane: `lane_mlx_effective_spend_triage_cli_enforcement_20260522`

## Summary

Added an automation-facing fail-closed CLI switch to the LL scorer-response
planner:

`--require-effective-mlx-spend-triage`

The switch writes the JSON and Markdown plan artifacts first, then exits with
code `2` unless `effective_mlx_spend_triage_gate.mlx_exact_eval_spend_triage_allowed`
is exactly `true`.

## Fix

- `tools/plan_ll_scorer_response_next.py` now exposes
  `--require-effective-mlx-spend-triage`.
- CLI stdout includes both:
  - `effective_mlx_spend_triage_allowed`
  - `effective_mlx_spend_triage_gate`
- A blocked effective gate returns exit code `2` only after output files are
  written, preserving inspectability for CI and operator automation.
- Added CLI regression tests for:
  - blocked one-row MLX dataset with passing constituent gates
  - passing synthetic validated MLX dataset with family diversity, fold
    coverage, row-level source-authority false proof, strict parity,
    calibration, and production contract gates
- Fixed the review finding that the new tests were asserting pretty-printed
  stdout substrings; they now parse stdout JSON and assert contract fields.

## Authority Boundary

This change does not create score authority. A strict effective MLX spend-triage
gate remains `[macOS-MLX research-signal]` only and cannot promote, rank/kill,
or dispatch without contest auth-eval custody.

Canonical surface: `ll_effective_mlx_spend_triage_gate.v1` in
`tac.optimization.scorer_response_dataset`. This memo introduces no new
canonical equation.

## Verification

- `.venv/bin/ruff check tools/plan_ll_scorer_response_next.py src/tac/tests/test_plan_ll_scorer_response_next_cli.py`
- `.venv/bin/python -m pytest -q src/tac/tests/test_plan_ll_scorer_response_next_cli.py`
- `.venv/bin/python -m pytest -q src/tac/tests/test_plan_ll_scorer_response_next_cli.py src/tac/tests/test_scorer_response_dataset.py`
- `.venv/bin/python tools/lane_maturity.py validate`

Results:

- Ruff: pass
- Focused CLI tests: 10 passed
- Planner/dataset regression slice: 81 passed
- Lane registry validation: 1136 lanes validated cleanly

## Partner WIP

`tools/build_hfv1_sparse_sidecar_candidate.py` is unrelated dirty work and was
left unstaged.
