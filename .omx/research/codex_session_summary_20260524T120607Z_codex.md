# Codex Session Summary - 2026-05-24T12:06:07Z

Evidence grade: `[repo-local/live-queue/MLX-research-signal]`

## Scope

Continued the MLX learned-sweep tranche by moving exact-row auto batch roots
from one row per root to grouped exact-row roots. The point of this change is
machine saturation: one queue root can now execute multiple acquisition-selected
`queue_candidate_id` rows while the experiment queue still proves that every
requested row, not merely a matching optimization pass, actually executed.

This remains local MLX research signal. It does not create score authority,
promotion eligibility, rank/kill authority, or exact-eval dispatch authority.

## Code Changes

- `src/tac/optimization/mlx_dynamic_learned_sweep_local_actuator.py`
  now emits exact filter count/set proof fields:
  `executed_queue_candidate_id_count`, `executed_queue_candidate_id_set`,
  `executed_candidate_id_count`, and `executed_candidate_id_set`.
- `src/tac/optimization/mlx_dynamic_learned_sweep_local_autopilot.py`
  rolls grouped exact-row filter proof up to the top-level summary.
- `src/comma_lab/scheduler/mlx_learned_sweep_autopilot_queue.py`
  now requires grouped exact-row postconditions to match both the executed
  queue-candidate set and count.
- `src/tac/optimization/mlx_learned_sweep_batch_roots.py` no longer attaches
  a single `optimization_pass_id` filter to a grouped root when that root spans
  multiple pass ids. Exact `queue_candidate_id` filters become the authority.
- Tests cover multi-row exact filters in the actuator, queue builder, and
  auto-root planner.

## Live Grouped-Root Proof

Live source root:

`experiments/results/mlx_effective_spend_triage_learned_sweep_live_20260524T100114Z/`

Auto grouped-row artifacts:

- Root plan:
  `mlx_autopilot_auto_row_group_batch_roots_20260524T120434Z.json`
- Queue:
  `mlx_autopilot_auto_row_group_batch_queue_20260524T120434Z.json`
- Queue state:
  `.omx/state/experiment_queue_mlx_learned_sweep_auto_row_group_batch_live_20260524T120434Z.sqlite`

Root plan:

- `selected_root_count=2`
- `rows_per_root=2`
- `queue_candidate_disjoint_guaranteed=true`
- Root 1 spans `micro` + `intermediate` and correctly omits the pass filter
  from its run spec.
- Root 2 stays inside `intermediate` and carries that pass filter.

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
- elapsed `25.0547s` and `25.0539s`
- no failed postconditions
- no postcondition errors
- `timed_out=false`

Both roots reported:

- schema `mlx_dynamic_learned_sweep_local_autopilot.v1`
- `executed_filter_match=true`
- `executed_queue_candidate_id_count=2`
- `executed_row_count=2`
- `new_observation_row_count=2`
- `local_mlx_device_used=true`
- `gpu_launched=false`
- `score_claim=false`
- `promotion_eligible=false`
- `ready_for_exact_eval_dispatch=false`

Observation ledgers:

- `mlx_autopilot_auto_row_group_batch_observations_20260524T120434Z_pass_0001_...jsonl`
  has `2` rows.
- `mlx_autopilot_auto_row_group_batch_observations_20260524T120434Z_pass_0002_...jsonl`
  has `2` rows.

## Verification

Focused:

```text
.venv/bin/python -m pytest -q \
  src/tac/tests/test_mlx_dynamic_learned_sweep_local_actuator.py \
  src/tac/tests/test_mlx_dynamic_learned_sweep_local_autopilot.py \
  src/tac/tests/test_mlx_learned_sweep_autopilot_queue.py
```

Result: `33 passed in 1.05s`.

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

Result: `142 passed in 4.29s`.

Additional checks:

- `ruff check` on touched MLX/autopilot/queue/tool/test files passed.
- `git diff --check` passed.

## Remaining Gaps

- The grouped-root proof executes four local MLX rows across two roots. Next
  step is adaptive root sizing from available local MLX slots and historical
  per-row runtime/utility, not a fixed `rows_per_root`.
- Candidate-root chained walks still need their own explicit mode. Exact row
  roots are intentionally single-cycle; candidate roots can be multi-cycle only
  when duplicate/exhaustion semantics are made explicit.
- Archive materialization, byte-closed export, exact CPU/CUDA auth-eval
  dispatch, and promotion gates remain downstream authority surfaces.
