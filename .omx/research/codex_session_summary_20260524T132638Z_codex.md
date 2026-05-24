# Codex Session Summary - 2026-05-24T13:26:38Z

Evidence grade: `[repo-local/queue-dag-proof/MLX-research-signal]`

## Scope

Promoted MLX learned-sweep runtime telemetry from an opt-in CLI convention into
a named advisory policy that is produced by the queue builder, normalized by the
staircase DAG, and carried into executor task specs with explicit false-authority
and executor-contract safeguards.

This remains local MLX research signal only. It is not score, promotion,
rank/kill, exact-dispatch authority, or contest CPU/CUDA evidence.

## Code Changes

- `src/comma_lab/scheduler/mlx_learned_sweep_autopilot_queue.py`
  - Added `MLX_RUNTIME_TELEMETRY_STATE_DISCOVERY_POLICY_SCHEMA`.
  - Added strict `runtime_telemetry_policy` validation.
  - Queue metadata now carries the policy when supplied.
  - Batch queue construction can pass a global or per-run policy into each
    subqueue.
  - Policy payloads are forced to explicit false score/promotion/dispatch
    authority.

- `tools/build_mlx_learned_sweep_autopilot_queue.py`
  - Auto-discovered compatible runtime telemetry states now produce a named
    `runtime_telemetry_state_policy`.
  - The root plan records that policy.
  - Auto-batch run specs pass the policy into queue construction.
  - CLI summary reports policy schema and policy id.

- `src/comma_lab/scheduler/staircase_dag.py`
  - Added fail-closed normalization for `runtime_telemetry_policy`.
  - DAG node metadata wraps the policy with proxy-evidence blockers.
  - Dispatch task specs carry the policy and executor contract:
    `planner_may_use_for_runtime_balancing`,
    `executor_must_not_treat_policy_as_score_authority`, and
    `exact_auth_eval_required_before_promotion`.

- `src/comma_lab/scheduler/__init__.py`
  - Exports the runtime telemetry policy schema constant.

- Tests:
  - `src/tac/tests/test_mlx_learned_sweep_autopilot_queue.py`
  - `src/tac/tests/test_staircase_dag.py`
  - Added assertions that root plans, queue metadata, DAG nodes, and Dask task
    specs preserve the named policy and its false-authority contract.

## Fresh Queue/DAG Proof

Input plan:

- `experiments/results/mlx_effective_spend_triage_learned_sweep_live_20260524T100114Z/local_mlx_chain_autopilot_run_20260524T105523Z/mlx_learned_sweep_chain_live_20260524t105523z/step_0002/cycle_0001/learned_sweep_plan.after_cycle.json`

Generated artifacts:

- Root plan:
  `experiments/results/mlx_effective_spend_triage_learned_sweep_live_20260524T100114Z/mlx_autopilot_named_policy_adaptive_row_group_batch_roots_20260524T132520Z.json`
- Queue:
  `experiments/results/mlx_effective_spend_triage_learned_sweep_live_20260524T100114Z/mlx_autopilot_named_policy_adaptive_row_group_batch_queue_20260524T132520Z.json`
- Queue state:
  `.omx/state/experiment_queue_mlx_learned_sweep_named_policy_adaptive_row_group_batch_live_20260524T132520Z.sqlite`
- DAG:
  `experiments/results/mlx_effective_spend_triage_learned_sweep_live_20260524T100114Z/mlx_autopilot_named_policy_adaptive_row_group_batch_dag_20260524T132520Z.json`
- Dispatch plan:
  `experiments/results/mlx_effective_spend_triage_learned_sweep_live_20260524T100114Z/mlx_autopilot_named_policy_adaptive_row_group_batch_dispatch_20260524T132520Z.json`

Queue-build summary:

- `auto_batch_runtime_telemetry_policy_schema=mlx_runtime_telemetry_state_discovery_policy.v1`
- `auto_batch_runtime_telemetry_policy_id=mlx_learned_sweep_runtime_telemetry_state_discovery`
- `auto_batch_runtime_telemetry_state_discovered_count=2`
- discovered states:
  - `.omx/state/experiment_queue_mlx_learned_sweep_state_harvest_adaptive_row_group_batch_live_20260524T125348Z.sqlite`
  - `.omx/state/experiment_queue_mlx_learned_sweep_runtime_adaptive_row_group_batch_live_20260524T124523Z.sqlite`
- `auto_batch_runtime_telemetry_used=true`
- `auto_batch_runtime_cost_policy=telemetry_seconds_per_queue_candidate_with_planned_fallback`
- `experiment_count=2`
- `step_count=2`
- score, promotion, rank/kill, exact-dispatch, dispatch, and GPU-launch
  authority fields false.

Queue validation:

- queue id
  `mlx_learned_sweep_named_policy_adaptive_row_group_batch_live_20260524T132520Z`
- `experiment_count=2`
- `step_count=2`
- `local_mlx` parallelism cap `2`
- `valid=true`

Dispatch proof:

- `selected_count=2`
- `ready_count=2`
- `blocked_count=0`
- top-level `score_claim=false`
- top-level `promotion_eligible=false`
- top-level `ready_for_exact_eval_dispatch=false`
- each `dask_task_specs[].runtime_telemetry_policy.schema` is
  `mlx_runtime_telemetry_state_discovery_policy.v1`
- each `dask_task_specs[].runtime_telemetry_policy.policy_id` is
  `mlx_learned_sweep_runtime_telemetry_state_discovery`
- each policy has `score_claim=false`
- each policy carries dispatch blockers:
  - `optimizer_candidate_queue_is_planning_only`
  - `requires_exact_eval_readiness_gate`
  - `requires_lane_dispatch_claim_before_gpu_or_remote_eval`
  - `requires_non_proxy_score_evidence_before_promotion`
  - `runtime_telemetry_policy_is_advisory_only`
  - `runtime_telemetry_policy_does_not_grant_score_authority`

## Verification

- Focused queue/DAG tests:
  `pytest -q src/tac/tests/test_mlx_learned_sweep_autopilot_queue.py src/tac/tests/test_staircase_dag.py`:
  `60 passed in 2.13s`.
- Broader MLX/scheduler tranche:
  `pytest -q src/tac/tests/test_experiment_queue.py src/tac/tests/test_staircase_dag.py src/tac/tests/test_mlx_effective_spend_triage_selection.py src/tac/tests/test_mlx_effective_spend_triage_learned_sweep_adapter.py src/tac/tests/test_mlx_dynamic_learned_sweep_observation_harvest.py src/tac/tests/test_mlx_dynamic_learned_sweep_local_actuator.py src/tac/tests/test_mlx_dynamic_learned_sweep_local_autopilot.py src/tac/tests/test_mlx_learned_sweep_autopilot_queue.py`:
  `160 passed in 10.76s`.
- `ruff check` on touched scheduler, optimization, tool, and queue-test files:
  passed.
- `git diff --check`:
  passed.

## Remaining Gaps

- The named runtime telemetry policy is now queue/DAG-visible, but actual task
  execution is still deferred; this proof stops at validated dispatch specs.
- Runtime telemetry remains advisory scheduling signal, not per-row exact
  profiling or score authority.
- Candidate-root chained walks still need explicit duplicate/exhaustion
  semantics.
- Archive materialization, byte-closed export, exact CPU/CUDA auth-eval
  dispatch, and promotion gates remain downstream authority surfaces.
