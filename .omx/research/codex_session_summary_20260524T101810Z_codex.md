# Codex Session Summary - 2026-05-24T10:18:10Z

## Scope

Advanced the local MLX effective-spend-triage to dynamic learned-sweep feedback
path. The goal was to stop treating one-off MLX scorer-response rows as manual
analysis and instead feed them back into the replanner as typed, false-authority
observations.

## Landed Artifacts

- Added `tac.optimization.mlx_effective_spend_triage_learned_sweep_adapter`.
  It adapts strict normalized MLX spend-triage selections into
  `mlx_dynamic_learned_sweep` candidate payloads.
- Added `tools/adapt_mlx_effective_spend_triage_to_learned_sweep.py`.
- Added `tac.optimization.mlx_dynamic_learned_sweep_observation_harvest`.
  It harvests local MLX learned-sweep rows into
  `mlx_dynamic_sweep_observation.v1` JSONL with false-authority fields stamped.
- Added `tools/harvest_mlx_dynamic_learned_sweep_observations.py`.
- Hardened `mlx_effective_spend_triage_selection` so legacy strict-gate rows can
  be normalized to full-video objective basis without inheriting authority.
- Exported the adapter and observation harvester from `tac.optimization`.

## Live Evidence

Generated live artifacts under:

`experiments/results/mlx_effective_spend_triage_learned_sweep_live_20260524T100114Z/`

- `learned_sweep_quality_candidates.json`: 8 adapted candidate rows, no score or
  dispatch authority.
- `local_mlx_smoke_observations.jsonl`: 8 harvested local MLX smoke observations.
- `local_mlx_smoke_observations.summary.json`: row_count=8,
  score_claim=false, ready_for_exact_eval_dispatch=false.
- `learned_sweep_plan.after_observation.json`: observation_row_count=8,
  suppressed_observed_row_count=8, ranked_row_count=120,
  local_ready_row_count=56, score_claim=false,
  ready_for_exact_eval_dispatch=false.

## Verification

- `pytest -q src/tac/tests/test_mlx_dynamic_learned_sweep_observation_harvest.py src/tac/tests/test_mlx_effective_spend_triage_selection.py src/tac/tests/test_mlx_effective_spend_triage_learned_sweep_adapter.py src/tac/tests/test_mlx_dynamic_learned_sweep.py`
  passed: 43 tests.
- `pytest -q src/tac/tests/test_mlx_dynamic_sweep_observations.py src/tac/tests/test_cross_family_candidate_portfolio.py`
  passed: 33 tests.
- `ruff check` on touched MLX adapter/harvest/selection files passed.
- `git diff --check` passed.

## Remaining Gaps

- This still consumes already-measured MLX scorer-response evidence; it does not
  yet launch or execute fresh local MLX sweeps from the dynamic plan.
- Exact auth dispatch remains correctly gated behind archive materialization,
  lane claim, and exact-eval custody.
- Next implementation step is the local runner/actuator that takes
  `ready_for_local_sweep` rows, executes the selected local MLX operation, writes
  observation JSONL, and reruns the planner automatically.
