# 5D Coverage Acquisition Queue Landing

UTC: 2026-05-27T05:48:51Z
Agent: codex
Lane: `lane_pair_frame_5d_coverage_acquisition_queue_20260527`

## Verdict

The 5D canvas coverage audit is now queue-owned instead of an orphaned advisory
side report.  Coverage work orders compile into executable acquisition-plan
rows, the refresh cycle emits the queue, and the autonomous-chain parent binds
it as a non-advisory child before the 5D extended-operator refire.

All new surfaces remain false-authority:

- no score claim;
- no promotion eligibility;
- no rank/kill eligibility;
- no exact-dispatch authority;
- no receiver-side optimization.

## Built And Wired

- `src/comma_lab/scheduler/pair_frame_5d_coverage_acquisition_queue.py`
  builds both per-work-order acquisition plans and the queue that consumes a
  coverage audit.
- `tools/emit_5d_canvas_coverage_acquisition_plan.py` emits one typed
  acquisition plan from a coverage work order.
- `tools/build_5d_canvas_coverage_acquisition_queue.py` builds the queue from a
  coverage audit and live canvas path.
- `frontier_rate_attack_feedback_cycle` and
  `tools/build_frontier_rate_attack_feedback_refresh.py` now emit
  `pair_frame_5d_coverage_acquisition_queue.json` plus a report summary when
  the coverage audit has work orders.
- `build_frontier_autonomous_chain_optimization_queue` now auto-binds
  `pair_frame_5d_coverage_acquisition_queue` as a local child action with a
  16-step, 6-experiment, single-worker budget, then continues to the 5D
  extended-operator child queue.

## Live Proof

Live refresh artifact root:

`experiments/results/frontier_5d_coverage_acquisition_20260527T054702Z`

Live refresh emitted:

- `pair_frame_5d_canvas_coverage_audit.json`
- `pair_frame_5d_coverage_acquisition_queue.json`
- `pair_frame_5d_extended_operator_queue.json`
- `autonomous_chain_optimization_queue.json`

Coverage queue validation:

`valid=true`, `experiment_count=6`, `step_count=9`

Coverage queue worker:

`success_count=9`, `failure_count=0`

The worker emitted five acquisition plans, including the MLX-local negative
delta probe work order, then refreshed the 5D canvas, re-audited it, rebuilt the
8-op extended-operator queue, and refired all 8 extended operators.

Autonomous parent validation:

`valid=true`, `experiment_count=3`, `step_count=27`

Autonomous parent bounded run:

`success_count=9`, `failure_count=0`, first chain
`global_many_op_rate_distortion_receiver_campaign`.

The parent work order included four child queues:

- `operation_materializer_execution_queue`
- `repair_budget_waterfill_queue`
- `pair_frame_5d_coverage_acquisition_queue`
- `pair_frame_5d_extended_operator_queue`

## Verification

- `.venv/bin/ruff check ...` on touched code, tools, and tests: passed.
- `.venv/bin/python -m pytest src/tac/tests/test_pair_frame_5d_coverage_acquisition_queue.py src/tac/tests/test_pair_frame_5d_extended_operator_queue.py -q`: 8 passed.
- `.venv/bin/python -m pytest src/tac/tests/test_frontier_rate_attack_feedback.py -q`: 50 passed.
- `tools/experiment_queue.py validate` on the live coverage acquisition queue:
  passed.
- `tools/experiment_queue.py run-worker --execute --max-steps 16 --max-experiments 6 --max-parallel 1` on the live coverage acquisition queue:
  9/9 succeeded.
- `tools/experiment_queue.py validate` on the live autonomous-chain queue:
  passed.
- `tools/experiment_queue.py run-worker --execute --max-steps 12 --max-experiments 1 --max-parallel 1` on the live autonomous-chain queue:
  9/9 succeeded.

`src/tac/tests/test_frontier_rate_attack_feedback_cycle.py` does not exist in
this checkout; the refresh CLI and queue worker runs were used as the cycle
verification surface.

## Remaining Work

This landing does not claim optimality.  It makes the missing surface explicit:
paired CPU/CUDA axis anchors and MLX negative-delta cells are now acquisition
rows with typed blockers instead of invisible assumptions.  The next tranche
should feed actual MLX-local and paired-auth evidence into those rows, then let
the refreshed 5D canvas and extended-operator queue absorb the new cells.
