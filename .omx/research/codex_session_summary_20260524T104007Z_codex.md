# Codex Session Summary - 2026-05-24T10:40:07Z

## Scope

Advanced the MLX dynamic learned-sweep stack from a one-row local actuator into
a bounded local autopilot. The new loop executes multiple `mlx_local_response`
rows, appends canonical observations, and replans after each bounded batch while
preserving the existing false-authority contract.

## Landed Artifacts

- Added `tac.optimization.mlx_dynamic_learned_sweep_local_autopilot`.
  It wraps the local MLX actuator with `max_iterations`,
  `max_new_observations`, `rows_per_replan`, optional `max_seconds`, and strict
  `sweep_config_id="mlx_local_response"` admission.
- Added `tools/run_mlx_dynamic_learned_sweep_autopilot.py`.
  It runs the bounded local loop from a learned-sweep plan, selection payload,
  and candidate payload, then writes a false-authority summary manifest.
- Added focused tests in
  `src/tac/tests/test_mlx_dynamic_learned_sweep_local_autopilot.py`.
- Exported `run_local_mlx_sweep_autopilot` from `tac.optimization`.

## Live Evidence

Used the live plan root:

`experiments/results/mlx_effective_spend_triage_learned_sweep_live_20260524T100114Z/`

Generated:

- `local_mlx_autopilot_observations.jsonl`: copied from the existing recursive
  observation ledger and appended two new `[macOS-MLX research-signal]`
  observations.
- `local_mlx_autopilot_summary.json`: cycle_count=2, executed_row_count=2,
  initial_observation_row_count=9, final_observation_row_count=11,
  stopping_reason=`max_new_observations_reached`, score_claim=false,
  ready_for_exact_eval_dispatch=false.
- `local_mlx_autopilot_run_20260524T1045Z/cycle_0001/` and `cycle_0002/`
  with per-cycle actuation artifacts and replanned
  `learned_sweep_plan.after_cycle.{json,md}` files.

The executed rows were:

- `mlx_scorer_response:window:98:99::mlx_local_response::micro`
- `mlx_scorer_response:window:501:502::mlx_local_response::micro`

The fresh observations recorded local MLX deltas only:

- window 98:99 observed_score_or_delta=`-2.3416463149950518e-06`
- window 501:502 observed_score_or_delta=`-2.131690902562276e-06`

Final replanned summary: observation_row_count=11,
suppressed_observed_row_count=11, ranked_row_count=117,
local_ready_row_count=53, score_claim=false,
ready_for_exact_eval_dispatch=false.

## PixelShuffle Intake Check

An xhigh sidecar review found no literal public submission named
`pixelshuffle`. The related public PRs are HNeRV-family submissions that use
`nn.PixelShuffle(2)` as a decoder primitive, not as the submission identity.
Internal PixelShuffle/PSD surfaces are historical postfilter lanes and current
architecture primitives. No new public-PR intake is needed solely for the name;
if it matters now, treat it as a primitive-level lowering/layout/fusion axis in
the HNeRV/NeRV/BoostNeRV path.

## Verification

- `pytest -q src/tac/tests/test_mlx_dynamic_learned_sweep_local_autopilot.py src/tac/tests/test_mlx_dynamic_learned_sweep_local_actuator.py src/tac/tests/test_mlx_dynamic_learned_sweep_observation_harvest.py src/tac/tests/test_mlx_effective_spend_triage_selection.py src/tac/tests/test_mlx_effective_spend_triage_learned_sweep_adapter.py src/tac/tests/test_mlx_dynamic_learned_sweep.py src/tac/tests/test_mlx_dynamic_sweep_observations.py src/tac/tests/test_cross_family_candidate_portfolio.py`
  passed: 84 tests.
- `ruff check` on touched MLX autopilot, actuator, harvest, adapter, selection,
  CLI, and test files passed.
- `git diff --check` passed.

## Remaining Gaps

- Autopilot currently executes `mlx_local_response` only. CPU advisory,
  optimizer-scheduler paired ablations, exact-gated rows, and archive
  materialization still require dedicated actuators.
- The loop is bounded and local, not a queue scheduler. The next integration
  step is to let the existing queue/DAG enqueue these autopilot runs as a
  substrate worker with lane claims, resource caps, and artifact harvesting.
- MLX observations remain research signal. Exact CUDA/CPU auth eval custody,
  archive materialization, lane claim, and false-authority gates still decide
  promotion and score claims.
- PixelShuffle should not be resurrected as a separate public intake target
  without fresh exact/component evidence; it should be modeled as a primitive
  axis inside HNeRV/NeRV/BoostNeRV lowering and layout experiments.
