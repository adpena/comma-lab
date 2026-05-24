# Codex Findings: PR95 MLX Matrix Queue Execution And Harvest

UTC: 2026-05-24T09:14:34Z
Agent: Codex
Lane: `codex_pr95_mlx_optimizer_matrix_queue_20260524`

## Scope

Adversarial continuation of the PR95/HNeRV MLX reproduction lane, focused on
making the optimizer matrix queue executable rather than merely plan-shaped.

## Concrete Findings

1. The first local run of
   `experiments/results/pr95_mlx_optimizer_matrix_queue_20260524T090145Z/experiment_queue.json`
   failed all three queue steps after command success. The bug was a plan/runtime
   identity split: plan-side `candidate_id` values encoded `seed17_c36`, while
   runtime sidecars encoded `seed17_steps1`.

2. Candidate identity now includes stage, optimizer descriptor, seed, step
   count, and channel count on both the plan and runtime sides:
   `pr95_hnerv_mlx_stage{stage}_{descriptor}_seed{seed}_steps{steps}_c{channels}`.
   A new queue execution regression catches the exact failure class by building,
   executing, and harvesting one matrix cell under isolated SQLite state.

3. PR95 public archive export and runtime-consumption proof plumbing now flows
   through plan generation, matrix queue generation, queue postconditions, timing
   smoke execution, and representation sidecars. These artifacts remain
   false-authority research signals and do not clear exact score, promotion,
   rank/kill, or dispatch gates.

4. Fresh corrected artifact:
   `experiments/results/pr95_mlx_optimizer_matrix_queue_20260524T091004Z/experiment_queue.json`.
   Worker execution completed `3/3` local MLX cells with no failed
   postconditions. Queue performance reports `local_mlx` mean worker elapsed
   time `0.21615291635195413s` for the one-step timing cells.

5. Harvested artifact:
   `experiments/results/pr95_mlx_optimizer_matrix_queue_20260524T091004Z/optimizer_candidate_queue.json`.
   It contains three timing-ranked optimizer candidate rows,
   `dispatch_ready_count=0`, `score_claim=false`, `promotion_eligible=false`,
   and `ready_for_exact_eval_dispatch=false`.

## Subagent Signal Integrated

- Optimizer wiring audit: harvested PR95 timings are correctly non-authoritative
  but still need a typed adapter into cost priors, neutral optimizer atoms,
  scheduler telemetry, and learned-sweep/cathedral consumers. Do not loosen
  authority to solve that gap.

- Exact-readiness audit: the generic materializer registry already has useful
  receiver vocabulary, but exact-readiness needs a neutral materializer receiver
  runtime-proof schema before family-agnostic materializers can become executable.

## Verification

- `.venv/bin/python -m ruff check tools/run_pr95_mlx_timing_smoke.py tools/build_pr95_mlx_optimizer_matrix_queue.py src/tac/local_acceleration/pr95_hnerv_mlx.py src/comma_lab/scheduler/local_training_queue.py src/tac/tests/test_pr95_mlx_optimizer_matrix_queue.py src/tac/tests/test_run_pr95_mlx_timing_smoke.py src/tac/tests/test_run_pr95_mlx_timing_smoke_plan.py`
- `.venv/bin/python -m pytest src/tac/tests/test_pr95_mlx_optimizer_matrix_queue.py src/tac/tests/test_run_pr95_mlx_timing_smoke.py src/tac/tests/test_run_pr95_mlx_timing_smoke_plan.py src/tac/tests/test_local_training_execution_queue.py src/tac/tests/test_local_training_optimizer_candidate_harvest.py src/tac/tests/test_pr95_hnerv_mlx.py -q`
- `.venv/bin/python tools/lane_maturity.py validate`
- `.venv/bin/python tools/review_gate_hook.py`
- `.venv/bin/python tools/experiment_queue.py --queue experiments/results/pr95_mlx_optimizer_matrix_queue_20260524T091004Z/experiment_queue.json status`
- `.venv/bin/python tools/experiment_queue.py --queue experiments/results/pr95_mlx_optimizer_matrix_queue_20260524T091004Z/experiment_queue.json performance`

## Next Integration Tasks

1. Add a harvest-to-optimizer-atoms/telemetry adapter that converts
   `optimizer_candidate_queue_v1` timing rows into false-authority scheduler
   cost observations and neutral optimizer atoms.

2. Add learned-sweep intake tests that reject raw harvested timing queues unless
   an explicit adapter supplies calibrated quality evidence.

3. Add the neutral materializer receiver runtime-proof schema to exact-readiness,
   then wire one low-risk family-agnostic archive/packet recompress materializer
   through proof-producing queue execution.
