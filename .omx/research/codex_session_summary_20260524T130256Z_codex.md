# Codex Session Summary - 2026-05-24T13:02:56Z

Evidence grade: `[repo-local/live-queue/MLX-research-signal]`

## Scope

Closed the runtime-telemetry loop one step further for MLX learned-sweep
autopilot batching. The queue builder can now mine a prior
`experiment_queue.v1` SQLite state directly via
`--auto-batch-runtime-telemetry-state`, convert successful/failed worker events
into `experiment_queue_worker_result.v1` telemetry, and feed that into
adaptive runtime-balanced batch-root planning.

This is still local MLX research signal only. It is not score, promotion,
rank/kill, exact-dispatch authority, or contest CPU/CUDA evidence.

## Code Changes

- `tools/build_mlx_learned_sweep_autopilot_queue.py`
  - Added repeatable `--auto-batch-runtime-telemetry-state`.
  - Reads SQLite `queue_events` rows with `step_succeeded` / `step_failed`.
  - Emits advisory `experiment_queue_worker_result.v1` payloads with
    `source_state_path`, `source_queue_ids`, `queue_id`, `experiment_id`, and
    `step_id`.
  - Fails closed on missing state paths with a clean `FATAL:` CLI error.

- `src/tac/optimization/mlx_learned_sweep_batch_roots.py`
  - Runtime telemetry summaries now preserve
    `runtime_telemetry_source_state_paths` and
    `runtime_telemetry_source_queue_ids`.
  - Empty telemetry summaries explicitly expose empty provenance lists.

- `src/tac/tests/test_mlx_learned_sweep_autopilot_queue.py`
  - Added SQLite-backed state telemetry fixture coverage.
  - Added CLI coverage proving state-derived telemetry drives adaptive
    auto-batch planning.
  - Added missing-state fail-closed CLI coverage.

## Fresh State-Harvest Proof

Input state:

- `.omx/state/experiment_queue_mlx_learned_sweep_runtime_adaptive_row_group_batch_live_20260524T124523Z.sqlite`

Artifacts:

- Root plan:
  `experiments/results/mlx_effective_spend_triage_learned_sweep_live_20260524T100114Z/mlx_autopilot_state_harvest_adaptive_row_group_batch_roots_20260524T125348Z.json`
- Queue:
  `experiments/results/mlx_effective_spend_triage_learned_sweep_live_20260524T100114Z/mlx_autopilot_state_harvest_adaptive_row_group_batch_queue_20260524T125348Z.json`
- Queue state:
  `.omx/state/experiment_queue_mlx_learned_sweep_state_harvest_adaptive_row_group_batch_live_20260524T125348Z.sqlite`

Root plan:

- `runtime_telemetry_used=true`
- `runtime_telemetry_key_count=6`
- `runtime_telemetry_payload_count=1`
- `runtime_telemetry_schema_counts={"experiment_queue_worker_result.v1": 1}`
- `runtime_cost_policy=telemetry_seconds_per_queue_candidate_with_planned_fallback`
- `row_grouping.strategy=positive_utility_runtime_balanced_waterfill`
- `root_runtime_cost_estimates=[42.79924875003053, 43.838665916991886]`
- `root_runtime_observed_queue_candidate_counts=[3, 3]`
- `runtime_telemetry_assignment_policy=elapsed_seconds_even_split_by_queue_candidate`
- `runtime_telemetry_source_queue_ids=["mlx_learned_sweep_runtime_adaptive_row_group_batch_live_20260524T124523Z"]`
- `runtime_telemetry_source_state_paths=[".omx/state/experiment_queue_mlx_learned_sweep_runtime_adaptive_row_group_batch_live_20260524T124523Z.sqlite"]`

Queue validation:

- schema `experiment_queue.v1`
- queue id `mlx_learned_sweep_state_harvest_adaptive_row_group_batch_live_20260524T125348Z`
- `experiment_count=2`
- `step_count=2`
- `local_mlx` parallelism cap `2`
- `valid=true`

Worker execution:

- state event counts:
  - `step_process_started=2`
  - `step_running=2`
  - `step_succeeded=2`
  - `worker_started=1`
  - `worker_stopped=1`
- `success_count=2`
- `failure_count=0`
- no running worker remained after completion.

Exact-filter proof:

- Pass 1 executed exactly 3 requested queue-candidate ids:
  - `mlx_scorer_response:window:479:480::mlx_local_response::intermediate`
  - `mlx_scorer_response:window:544:545::mlx_local_response::micro`
  - `mlx_scorer_response:window:98:99::mlx_local_response::intermediate`
- Pass 2 executed exactly 3 requested queue-candidate ids:
  - `mlx_scorer_response:window:496:497::mlx_local_response::intermediate`
  - `mlx_scorer_response:window:501:502::mlx_local_response::intermediate`
  - `mlx_scorer_response:window:59:60::mlx_local_response::intermediate`
- Both summaries reported:
  - `executed_filter_match=true`
  - `executed_filter_violation_count=0`
  - `executed_queue_candidate_id_count=3`
  - `executed_row_count=3`
  - `new_observation_row_count=3`
  - `local_mlx_device_used=true`
  - all score, promotion, rank/kill, exact-dispatch, and GPU-launch authority
    fields false.

Observation ledgers:

- `mlx_autopilot_state_harvest_adaptive_row_group_batch_observations_20260524T125348Z_pass_0001_...jsonl`
  has 3 rows.
- `mlx_autopilot_state_harvest_adaptive_row_group_batch_observations_20260524T125348Z_pass_0002_...jsonl`
  has 3 rows.

Queue performance:

- schema `experiment_queue_performance_summary.v1`
- `telemetry_only=true`
- `score_claim=false`
- `score_claim_valid=false`
- `rank_or_kill_eligible=false`
- `promotion_eligible=false`
- `ready_for_exact_eval_dispatch=false`
- `local_mlx.elapsed_seconds_mean=36.59100377050345`
- `local_mlx.elapsed_seconds_sum=73.1820075410069`
- `local_mlx.success_count=2`
- `local_mlx.failure_count=0`

## Verification

- Focused MLX/scheduler tranche:
  `pytest -q src/tac/tests/test_experiment_queue.py src/tac/tests/test_staircase_dag.py src/tac/tests/test_mlx_effective_spend_triage_selection.py src/tac/tests/test_mlx_effective_spend_triage_learned_sweep_adapter.py src/tac/tests/test_mlx_dynamic_learned_sweep_observation_harvest.py src/tac/tests/test_mlx_dynamic_learned_sweep_local_actuator.py src/tac/tests/test_mlx_dynamic_learned_sweep_local_autopilot.py src/tac/tests/test_mlx_learned_sweep_autopilot_queue.py`:
  `158 passed in 10.71s`.
- `ruff check` on touched scheduler, optimization, tool, and queue-test files:
  passed.
- `git diff --check`:
  passed.

## Remaining Gaps

- Runtime telemetry still assigns multi-candidate elapsed time by even split;
  it is useful for scheduling and batch balancing, not exact per-row profiling.
- State-harvest telemetry is wired at the queue-build CLI; the next stronger
  integration is a queue/DAG policy that automatically selects recent trusted
  compatible state files.
- Candidate-root chained walks still need explicit duplicate/exhaustion
  semantics.
- Archive materialization, byte-closed export, exact CPU/CUDA auth-eval
  dispatch, and promotion gates remain downstream authority surfaces.
