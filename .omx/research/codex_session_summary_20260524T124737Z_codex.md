# Codex Session Summary - 2026-05-24T12:47:37Z

Evidence grade: `[repo-local/live-queue/MLX-research-signal]`

## Scope

Runtime telemetry now feeds adaptive MLX learned-sweep batch-root planning.
The planner can ingest prior local MLX autopilot summaries and
`experiment_queue_worker_result.v1` payloads, estimate per-`queue_candidate_id`
runtime cost, and water-fill positive-utility rows by observed local runtime
seconds with planned `cost_units` fallback.

This remains local MLX research signal. It is not score, promotion, rank/kill,
exact-dispatch authority, or contest CPU/CUDA evidence.

## Code Changes

- `src/tac/optimization/mlx_learned_sweep_batch_roots.py`
  - Added runtime telemetry parsing for:
    - `mlx_dynamic_learned_sweep_local_autopilot.v1`
    - `experiment_queue_worker_result.v1`
  - Added strict truthy-authority rejection for telemetry payloads.
  - Local MLX summaries also require explicit false authority fields.
  - Adaptive grouping switches to
    `positive_utility_runtime_balanced_waterfill` when telemetry is present.
  - Row refs now include `runtime_cost_estimate` and `runtime_cost_source`.
  - Multi-candidate elapsed time is labeled with
    `runtime_telemetry_assignment_policy=elapsed_seconds_even_split_by_queue_candidate`
    and `runtime_telemetry_even_split_observation_count`.

- `tools/build_mlx_learned_sweep_autopilot_queue.py`
  - Added repeatable `--auto-batch-runtime-telemetry`.
  - CLI summary now reports runtime telemetry usage, key count, and cost policy.

- `src/tac/tests/test_mlx_learned_sweep_autopilot_queue.py`
  - Added runtime-informed grouping coverage.
  - Added CLI telemetry coverage.
  - Added duplicate telemetry averaging coverage.
  - Added worker-result even-split coverage.
  - Added fail-closed tests for unsupported schema, invalid elapsed seconds,
    filter mismatch, filter violations, and truthy authority flags.

## Fresh Live Proof

Inputs:

- Prior runtime telemetry summaries from the 2026-05-24T12:26:46Z adaptive
  local MLX batch, two payloads and six queue-candidate runtime observations.

Artifacts:

- Runtime-informed root plan:
  `experiments/results/mlx_effective_spend_triage_learned_sweep_live_20260524T100114Z/mlx_autopilot_runtime_adaptive_row_group_batch_roots_20260524T124523Z.json`
- Queue:
  `experiments/results/mlx_effective_spend_triage_learned_sweep_live_20260524T100114Z/mlx_autopilot_runtime_adaptive_row_group_batch_queue_20260524T124523Z.json`
- Queue state:
  `.omx/state/experiment_queue_mlx_learned_sweep_runtime_adaptive_row_group_batch_live_20260524T124523Z.sqlite`

Root plan:

- `runtime_telemetry_used=true`
- `runtime_telemetry_key_count=6`
- `runtime_cost_policy=telemetry_seconds_per_queue_candidate_with_planned_fallback`
- `row_grouping.strategy=positive_utility_runtime_balanced_waterfill`
- `root_runtime_cost_estimates=[34.20627087500179, 36.636819540988654]`
- `root_runtime_observed_queue_candidate_counts=[3, 3]`
- `runtime_telemetry_assignment_policy=elapsed_seconds_even_split_by_queue_candidate`
- all score/dispatch/promotion/rank-kill authority fields false.

Queue validation:

- schema `experiment_queue.v1`
- `experiment_count=2`
- `step_count=2`
- `local_mlx` parallelism cap `2`
- `valid=true`

Worker execution:

- `success_count=2`
- `failure_count=0`
- `claim_refused_count=0`
- `resource_limits={"local_mlx": 2}`
- elapsed `42.79924875003053s` and `43.838665916991886s`
- no failed postconditions
- no postcondition errors
- `timed_out=false` for both steps

Exact-filter proof:

- Root 1 requested and executed exactly 3 queue-candidate ids.
- Root 2 requested and executed exactly 3 queue-candidate ids.
- Both summaries reported:
  - `executed_filter_match=true`
  - `executed_filter_violation_count=0`
  - `executed_queue_candidate_id_count=3`
  - `executed_row_count=3`
  - `new_observation_row_count=3`
  - `local_mlx_device_used=true`
  - full false-authority fields false.

Observation ledgers:

- `mlx_autopilot_runtime_adaptive_row_group_batch_observations_20260524T124523Z_pass_0001_...jsonl`
  has 3 rows.
- `mlx_autopilot_runtime_adaptive_row_group_batch_observations_20260524T124523Z_pass_0002_...jsonl`
  has 3 rows.

Queue performance:

- schema `experiment_queue_performance_summary.v1`
- `telemetry_only=true`
- `score_claim=false`
- `score_claim_valid=false`
- `rank_or_kill_eligible=false`
- `promotion_eligible=false`
- `ready_for_exact_eval_dispatch=false`
- `local_mlx.elapsed_seconds_mean=43.31895733351121`
- `local_mlx.elapsed_seconds_sum=86.63791466702241`
- `local_mlx.success_count=2`
- `local_mlx.failure_count=0`

## Verification

- `pytest -q src/tac/tests/test_mlx_learned_sweep_autopilot_queue.py`:
  `35 passed in 1.33s`.
- Focused MLX/autopilot queue suite:
  `49 passed in 1.51s`.
- Broader MLX/scheduler tranche:
  `156 passed in 10.31s`.
- `ruff check` on touched files:
  passed.

## Remaining Gaps

- Runtime telemetry currently estimates multi-candidate batch cost by even split;
  useful for scheduling, not exact per-row profiling.
- The next planner step should harvest these live worker results back into the
  runtime telemetry input set automatically rather than requiring explicit CLI
  paths.
- Candidate-root chained walks still need explicit duplicate/exhaustion
  semantics.
- Archive materialization, byte-closed export, exact CPU/CUDA auth-eval
  dispatch, and promotion gates remain downstream authority surfaces.
