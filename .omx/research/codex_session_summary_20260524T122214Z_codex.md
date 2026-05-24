# Codex Session Summary - 2026-05-24T12:22:14Z

Evidence grade: `[repo-local/live-queue/MLX-research-signal]`

## Scope

Continued the MLX learned-sweep queue tranche by adding adaptive row grouping
for automatic batch roots. Fixed grouping still exists for deterministic
regression paths. The new mode water-fills positive-utility
`mlx_local_response` rows across requested local MLX roots, balances estimated
`cost_units`, and preserves exact `queue_candidate_id` filters as the execution
authority.

This remains local MLX research signal. It is not a score claim, promotion
claim, rank/kill claim, or exact-eval dispatch recommendation.

## Code Changes

- `src/tac/optimization/mlx_learned_sweep_batch_roots.py`
  - Added `adaptive_rows_per_root`.
  - Added `positive_utility_cost_balanced_waterfill` grouping metadata.
  - Rejects adaptive grouping when no ready row has positive acquisition and
    positive learning value per cost.
  - Emits `max_rows_per_root`, `selected_total_row_count`,
    `unselected_ready_row_count`, root sizes, root costs, and explicit false
    authority fields.
- `tools/build_mlx_learned_sweep_autopilot_queue.py`
  - Added `--auto-batch-adaptive-rows-per-root`.
  - Keeps legacy fixed-row behavior unless the adaptive flag is supplied.
- `src/tac/tests/test_mlx_learned_sweep_autopilot_queue.py`
  - Covers adaptive root sizing, nonpositive-utility refusal, and CLI wiring.

## Live Adaptive-Root Proof

Source plan:

`experiments/results/mlx_effective_spend_triage_learned_sweep_live_20260524T100114Z/local_mlx_chain_autopilot_run_20260524T105523Z/mlx_learned_sweep_chain_live_20260524t105523z/step_0002/cycle_0001/learned_sweep_plan.after_cycle.json`

Artifacts:

- Root plan:
  `experiments/results/mlx_effective_spend_triage_learned_sweep_live_20260524T100114Z/mlx_autopilot_adaptive_row_group_batch_roots_20260524T122013Z.json`
- Queue:
  `experiments/results/mlx_effective_spend_triage_learned_sweep_live_20260524T100114Z/mlx_autopilot_adaptive_row_group_batch_queue_20260524T122013Z.json`
- Queue state:
  `.omx/state/experiment_queue_mlx_learned_sweep_adaptive_row_group_batch_live_20260524T122013Z.sqlite`

Root plan:

- `adaptive_rows_per_root=true`
- `max_rows_per_root=3`
- `eligible_positive_utility_row_count=17`
- `selected_total_row_count=6`
- `selected_root_count=2`
- root sizes `[3, 3]`
- root costs `[9.0, 12.0]`
- `score_claim=false`
- `promotion_eligible=false`
- `ready_for_exact_eval_dispatch=false`

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
- elapsed `35.0710s` and `40.0773s`
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
  - `gpu_launched=false`
  - `score_claim=false`
  - `promotion_eligible=false`
  - `ready_for_exact_eval_dispatch=false`

Observation ledgers:

- `mlx_autopilot_adaptive_row_group_batch_observations_20260524T122013Z_pass_0001_...jsonl`
  has 3 rows.
- `mlx_autopilot_adaptive_row_group_batch_observations_20260524T122013Z_pass_0002_...jsonl`
  has 3 rows.

## Verification

Focused:

```text
.venv/bin/python -m pytest -q \
  src/tac/tests/test_mlx_dynamic_learned_sweep_local_actuator.py \
  src/tac/tests/test_mlx_dynamic_learned_sweep_local_autopilot.py \
  src/tac/tests/test_mlx_learned_sweep_autopilot_queue.py
```

Result: `36 passed in 1.20s`.

Broader MLX/scheduler tranche:

```text
.venv/bin/python -m pytest -q \
  src/tac/tests/test_mlx_dynamic_learned_sweep_local_autopilot.py \
  src/tac/tests/test_mlx_dynamic_learned_sweep_local_actuator.py \
  src/tac/tests/test_mlx_dynamic_learned_sweep_observation_harvest.py \
  src/tac/tests/test_mlx_effective_spend_triage_selection.py \
  src/tac/tests/test_mlx_effective_spend_triage_learned_sweep_adapter.py \
  src/tac/tests/test_mlx_dynamic_learned_sweep.py \
  src/tac/tests/test_mlx_dynamic_sweep_observations.py \
  src/tac/tests/test_cross_family_candidate_portfolio.py \
  src/tac/tests/test_mlx_learned_sweep_autopilot_queue.py \
  src/tac/tests/test_mlx_execution_queue.py \
  src/tac/tests/test_staircase_dag.py
```

Result: `145 passed in 4.44s`.

Additional checks:

- `ruff check` on the adaptive batch-root, CLI, and test files passed.
- `git diff --check` passed.

## Remaining Gaps

- Adaptive grouping currently uses acquisition and cost estimates already in
  the learned-sweep plan. It does not yet learn runtime-per-row from previous
  queue telemetry.
- Candidate-root chained walks still need an explicit duplicate/exhaustion
  contract. Exact row roots remain intentionally single-cycle.
- Archive materialization, byte-closed export, exact CPU/CUDA auth-eval
  dispatch, and promotion gates remain downstream authority surfaces.
