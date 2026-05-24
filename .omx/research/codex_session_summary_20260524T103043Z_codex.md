# Codex Session Summary - 2026-05-24T10:30:43Z

## Scope

Advanced the MLX dynamic learned-sweep stack from observation harvesting into a
real local actuator. The actuator executes supported `ready_for_local_sweep`
rows, writes local MLX scorer-response artifacts, appends canonical observation
rows, and can immediately replan from the updated observation JSONL.

## Landed Artifacts

- Added `tac.optimization.mlx_dynamic_learned_sweep_local_actuator`.
  It admits only false-authority `mlx_dynamic_learned_sweep_row.v1` rows with
  `sweep_config_id="mlx_local_response"` and `ready_for_local_sweep=true`.
- Added `tools/run_mlx_dynamic_learned_sweep_local.py`.
  It runs the local actuator, appends `mlx_dynamic_sweep_observation.v1`, and
  optionally writes a replanned `mlx_dynamic_learned_sweep_plan.v1`.
- Added focused tests in
  `src/tac/tests/test_mlx_dynamic_learned_sweep_local_actuator.py`.
- Exported `execute_local_mlx_sweep_rows` from `tac.optimization`.

## Live Evidence

Used the live plan root:

`experiments/results/mlx_effective_spend_triage_learned_sweep_live_20260524T100114Z/`

Generated:

- `local_mlx_recursive_observations.jsonl`: 9 total observations after appending
  one fresh local MLX micro observation.
- `local_mlx_actuator_summary.json`: executed_row_count=1,
  observation_row_count=1, score_claim=false,
  ready_for_exact_eval_dispatch=false.
- `local_mlx_actuator_run_20260524T1020Z/.../candidate_response.json` and
  `baseline_response.json`, plus per-component `.npy` arrays.
- `learned_sweep_plan.after_local_actuator.json`: observation_row_count=9,
  suppressed_observed_row_count=9, ranked_row_count=119,
  local_ready_row_count=55, score_claim=false,
  ready_for_exact_eval_dispatch=false.

The executed row was:

`mlx_scorer_response:window:496:497::mlx_local_response::micro`

The appended observation was `[macOS-MLX research-signal]`, replanning-only, and
recorded `observed_score_or_delta=-4.091056582945091e-7`.

## Verification

- `pytest -q src/tac/tests/test_mlx_dynamic_learned_sweep_local_actuator.py src/tac/tests/test_mlx_dynamic_learned_sweep_observation_harvest.py src/tac/tests/test_mlx_effective_spend_triage_selection.py src/tac/tests/test_mlx_effective_spend_triage_learned_sweep_adapter.py src/tac/tests/test_mlx_dynamic_learned_sweep.py src/tac/tests/test_mlx_dynamic_sweep_observations.py src/tac/tests/test_cross_family_candidate_portfolio.py`
  passed: 79 tests.
- `ruff check` on touched MLX local actuator, harvest, adapter, selection, CLI,
  and test files passed.
- `git diff --check` passed.

## Remaining Gaps

- The local actuator supports `mlx_local_response` only. `macos_cpu_advisory`
  rows are still deliberately refused until a real local CPU advisory executor
  exists.
- This is still local research signal. Exact dispatch authority remains gated by
  archive materialization, lane claim, exact auth eval custody, and the existing
  false-authority contracts.
- The next high-EV step is a bounded autopilot loop over multiple
  `mlx_local_response` rows with machine-resource controls, append-only
  observation custody, and periodic replans.
