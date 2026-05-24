# Codex Session Summary - 2026-05-24T10:52:15Z

## Scope

Moved the MLX learned-sweep local autopilot out of manual CLI-only operation
and into the existing queue/DAG authority surface. The new adapter compiles one
bounded local MLX autopilot run into `experiment_queue.v1`, preserving local
resource caps, queue state custody, postconditions, telemetry, and false-score
authority.

## Landed Artifacts

- Added `comma_lab.scheduler.mlx_learned_sweep_autopilot_queue`.
  It validates `mlx_dynamic_learned_sweep_plan.v1`,
  `mlx_effective_spend_triage_candidate_selection.v1`, candidate payloads, and
  at least one ready `mlx_local_response` row before emitting queue work.
- Added `tools/build_mlx_learned_sweep_autopilot_queue.py`.
  It writes a guarded `experiment_queue.v1` queue whose single step calls
  `tools/run_mlx_dynamic_learned_sweep_autopilot.py`.
- Exported `MLX_LEARNED_SWEEP_AUTOPILOT_QUEUE_SCHEMA` and
  `build_mlx_learned_sweep_autopilot_queue` from `comma_lab.scheduler`.
- Updated local actuator/autopilot summaries so `gpu_launched` remains false
  and local MLX device usage is recorded as `local_mlx_device_used`, avoiding
  conflict with proxy false-authority semantics.
- Added queue and DAG tests:
  `src/tac/tests/test_mlx_learned_sweep_autopilot_queue.py` and a
  `test_staircase_dag.py` projection check.

## Live Queue Evidence

Generated queue:

`experiments/results/mlx_effective_spend_triage_learned_sweep_live_20260524T100114Z/mlx_autopilot_queue_20260524T104902Z.json`

Validation:

- `tools/experiment_queue.py --queue ... validate`: valid, one experiment,
  one `local_mlx` step, auto-parallelism `local_mlx=1`.
- `tools/experiment_queue.py --queue ... next`: one ready
  `run_mlx_learned_sweep_autopilot` step.
- `tools/experiment_queue.py --queue ... run-worker --execute --max-steps 1`:
  succeeded with `failure_count=0`, `success_count=1`,
  `resource_kind=local_mlx`, telemetry artifact records captured.

Queue-owned output:

`experiments/results/mlx_effective_spend_triage_learned_sweep_live_20260524T100114Z/local_mlx_queue_autopilot_run_20260524T104902Z/mlx_learned_sweep_autopilot_live_20260524t104902z/local_mlx_autopilot_summary.json`

Live summary: cycle_count=1, executed_row_count=1,
initial_observation_row_count=12, final_observation_row_count=13,
local_mlx_device_used=true, gpu_launched=false, score_claim=false,
ready_for_exact_eval_dispatch=false.

Executed row:

`mlx_scorer_response:window:59:60::mlx_local_response::micro`

The appended observation remains `[macOS-MLX research-signal]` and is
replanning-only. It recorded `observed_score_or_delta=5.709802667768216e-06`.

## Verification

- `pytest -q src/tac/tests/test_mlx_dynamic_learned_sweep_local_autopilot.py src/tac/tests/test_mlx_dynamic_learned_sweep_local_actuator.py src/tac/tests/test_mlx_dynamic_learned_sweep_observation_harvest.py src/tac/tests/test_mlx_effective_spend_triage_selection.py src/tac/tests/test_mlx_effective_spend_triage_learned_sweep_adapter.py src/tac/tests/test_mlx_dynamic_learned_sweep.py src/tac/tests/test_mlx_dynamic_sweep_observations.py src/tac/tests/test_cross_family_candidate_portfolio.py src/tac/tests/test_mlx_learned_sweep_autopilot_queue.py src/tac/tests/test_mlx_execution_queue.py src/tac/tests/test_staircase_dag.py`
  passed: 122 tests.
- `ruff check` on touched MLX/autopilot/queue/DAG files passed.
- `git diff --check` passed.

## Remaining Gaps

- The queue compiler emits one bounded local MLX autopilot step. Broader queue
  generation for batches of campaigns, storage-tier placement, and sustained
  worker execution are still next-step integration work.
- CPU-advisory, optimizer-scheduler paired ablations, archive materialization,
  and exact-gated rows still require separate actuators/queue compilers.
- MLX observations remain local research signal only. Exact CPU/CUDA auth eval
  custody remains the promotion/score authority path.
