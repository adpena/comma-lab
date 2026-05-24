# Codex Session Summary - 2026-05-24T13:12:24Z

Evidence grade: `[repo-local/queue-build/MLX-research-signal]`

## Scope

Moved MLX learned-sweep runtime telemetry closer to queue/DAG-owned behavior.
The auto-batch queue builder can now discover compatible prior local MLX
`experiment_queue.v1` SQLite states without hand-feeding every state path,
filter them against the current ready `mlx_local_response` queue-candidate
surface, and feed the selected state-derived telemetry into adaptive
runtime-balanced root planning.

This remains local MLX research signal only. It is not score, promotion,
rank/kill, exact-dispatch authority, or contest CPU/CUDA evidence.

## Code Changes

- `tools/build_mlx_learned_sweep_autopilot_queue.py`
  - Added opt-in `--auto-batch-discover-runtime-telemetry-states`.
  - Added `--auto-batch-runtime-telemetry-state-dir`.
  - Added repeatable `--auto-batch-runtime-telemetry-state-pattern`.
  - Added `--auto-batch-runtime-telemetry-state-limit`.
  - Discovery defaults to `experiment_queue_mlx_learned_sweep*.sqlite` under
    `.omx/state`.
  - Discovery skips symlinks and non-files.
  - Discovery loads candidate states through the same
    `experiment_queue_worker_result.v1` conversion path as explicit states.
  - A state is selected only if successful, non-timeout, postcondition-clean
    step telemetry references at least one ready queue-candidate id from the
    current learned-sweep plan.
  - Selected discovered states are deterministic newest-first with a limit.
  - Discovery requires `--auto-batch-from-plan`; otherwise it fails with a
    clean `FATAL:` error.
  - CLI summary now reports discovery enabled/count/paths and the combined
    explicit+discovered state count.

- `src/tac/tests/test_mlx_learned_sweep_autopilot_queue.py`
  - Added compatible-state discovery coverage.
  - Added incompatible-state skip coverage.
  - Added missing discovery state-dir fail-closed coverage.

## Fresh Queue-Build Proof

Command class:

- `tools/build_mlx_learned_sweep_autopilot_queue.py`
- `--auto-batch-from-plan`
- `--auto-batch-adaptive-rows-per-root`
- `--auto-batch-discover-runtime-telemetry-states`
- `--auto-batch-runtime-telemetry-state-dir .omx/state`
- `--auto-batch-runtime-telemetry-state-limit 2`

Artifacts:

- Root plan:
  `experiments/results/mlx_effective_spend_triage_learned_sweep_live_20260524T100114Z/mlx_autopilot_auto_discovered_state_adaptive_row_group_batch_roots_20260524T131125Z.json`
- Queue:
  `experiments/results/mlx_effective_spend_triage_learned_sweep_live_20260524T100114Z/mlx_autopilot_auto_discovered_state_adaptive_row_group_batch_queue_20260524T131125Z.json`

CLI summary:

- `auto_batch_runtime_telemetry_state_discovery_enabled=true`
- `auto_batch_runtime_telemetry_state_discovered_count=2`
- `auto_batch_runtime_telemetry_state_count=2`
- discovered paths:
  - `.omx/state/experiment_queue_mlx_learned_sweep_state_harvest_adaptive_row_group_batch_live_20260524T125348Z.sqlite`
  - `.omx/state/experiment_queue_mlx_learned_sweep_runtime_adaptive_row_group_batch_live_20260524T124523Z.sqlite`
- `auto_batch_runtime_telemetry_used=true`
- `auto_batch_runtime_telemetry_key_count=6`
- `auto_batch_runtime_cost_policy=telemetry_seconds_per_queue_candidate_with_planned_fallback`
- `auto_batch_selected_root_count=2`
- `experiment_count=2`
- `step_count=2`
- score, promotion, rank/kill, exact-dispatch, dispatch, and GPU-launch
  authority fields false.

Root plan:

- `runtime_telemetry_used=true`
- `runtime_telemetry_key_count=6`
- `runtime_telemetry_schema_counts={"experiment_queue_worker_result.v1": 2}`
- `runtime_cost_policy=telemetry_seconds_per_queue_candidate_with_planned_fallback`
- `row_grouping.strategy=positive_utility_runtime_balanced_waterfill`
- `root_runtime_cost_estimates=[38.79110747901723, 41.11885362499743]`
- `root_runtime_observed_queue_candidate_counts=[3, 3]`
- `runtime_telemetry_source_queue_ids`:
  - `mlx_learned_sweep_runtime_adaptive_row_group_batch_live_20260524T124523Z`
  - `mlx_learned_sweep_state_harvest_adaptive_row_group_batch_live_20260524T125348Z`

Queue validation:

- queue id
  `mlx_learned_sweep_auto_discovered_state_adaptive_row_group_batch_live_20260524T131125Z`
- schema `experiment_queue.v1`
- `experiment_count=2`
- `step_count=2`
- `local_mlx` parallelism cap `2`
- `valid=true`

## Verification

- Focused queue test:
  `pytest -q src/tac/tests/test_mlx_learned_sweep_autopilot_queue.py`:
  `39 passed in 1.84s`.
- Focused MLX/scheduler tranche:
  `pytest -q src/tac/tests/test_experiment_queue.py src/tac/tests/test_staircase_dag.py src/tac/tests/test_mlx_effective_spend_triage_selection.py src/tac/tests/test_mlx_effective_spend_triage_learned_sweep_adapter.py src/tac/tests/test_mlx_dynamic_learned_sweep_observation_harvest.py src/tac/tests/test_mlx_dynamic_learned_sweep_local_actuator.py src/tac/tests/test_mlx_dynamic_learned_sweep_local_autopilot.py src/tac/tests/test_mlx_learned_sweep_autopilot_queue.py`:
  `160 passed in 11.01s`.
- `ruff check` on touched scheduler, optimization, tool, and queue-test files:
  passed.
- `git diff --check`:
  passed.

## Remaining Gaps

- Discovery is opt-in at the queue-build CLI. The next stronger integration is
  to make compatible-state discovery a named queue/DAG policy rather than a
  command-line convention.
- Runtime telemetry still assigns multi-candidate elapsed time by even split;
  it is useful for scheduling and batch balancing, not exact per-row profiling.
- Candidate-root chained walks still need explicit duplicate/exhaustion
  semantics.
- Archive materialization, byte-closed export, exact CPU/CUDA auth-eval
  dispatch, and promotion gates remain downstream authority surfaces.
