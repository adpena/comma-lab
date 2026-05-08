# Phase A Pareto Solver Integration - 2026-05-08

## Scope

This note records the local planner/report integration for Phase A Pareto
selection. No remote dispatch was attempted. No score claim, promotion claim,
rank claim, kill claim, or dispatch-clearance claim is made.

## Landed planning surfaces

- `tac.score_geometry.planner_axis_marginals(...)` exposes prediction-only
  target-axis marginals for `cuda_internal` and `cpu_leaderboard` planning.
  The CPU-leaderboard path applies the calibrated chain-rule scales from
  CUDA-side candidate deltas to CPU-axis score response.
- `tools/phase_a_pareto_summary.py` now emits a `Solver planning targets`
  section and machine-readable JSON fields for:
  - sub-0.17 byte budget under explicit CPU-floor assumptions;
  - floor feasibility;
  - per-lane subtarget byte gap for full-archive-comparable rows;
  - CUDA-internal vs CPU-leaderboard planner priority.
- `reports/phase_a_pareto_20260508.md` was regenerated from the tool.

## Current planner signal

Under the explicit prediction-only floor assumptions
`d_seg=6.0e-4`, `d_pose=3.5e-5`, and target score `0.170`, the closed-form
byte budget is `137,103 B`, requiring `41,041 B` savings versus the PR101
brotli byte anchor. Full-archive-comparable rows are annotated with their
remaining byte gap to that target.

At the PR107/apogee-ish operating point used by the report's axis advisor
(`d_seg_cuda=6.88e-4`, `d_pose_cuda=1.74e-4`, `B=178,392`), CUDA-internal
planner marginals prioritize pose, while CPU-leaderboard chain-rule marginals
prioritize seg. This is a routing prior only; paired exact `[contest-CUDA]`
and `[contest-CPU]` archive custody remains mandatory before any score use.

## Verification

```text
.venv/bin/python -m pytest src/tac/tests/test_score_geometry.py -q
38 passed

.venv/bin/python -m pytest src/tac/tests/test_phase_a_pareto_summary.py -q
4 passed

.venv/bin/python -m py_compile src/tac/score_geometry.py tools/phase_a_pareto_summary.py src/tac/tests/test_phase_a_pareto_summary.py
PASS
```
