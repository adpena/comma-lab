# Codex Session Summary - 2026-05-24T11:12:29Z

Evidence grade: `[repo-local/live-queue/MLX-research-signal]`

## Scope

Continued the next two-week optimization tranche by moving the MLX
learned-sweep autopilot from single-root queue execution toward multi-root,
queue-owned local MLX saturation. This session focused on proving that two
independent learned-sweep roots can be compiled into one `experiment_queue.v1`,
projected through the staircase DAG as separate `local_mlx` nodes, and executed
concurrently without leaking score/promotion authority.

## Code And Test State

- `src/comma_lab/scheduler/mlx_learned_sweep_autopilot_queue.py` exposes
  `build_mlx_learned_sweep_autopilot_batch_queue(...)` and
  `MLX_LEARNED_SWEEP_AUTOPILOT_BATCH_QUEUE_SCHEMA`.
- `tools/build_mlx_learned_sweep_autopilot_queue.py` accepts `--batch-spec`
  for independent roots.
- `src/tac/tests/test_mlx_learned_sweep_autopilot_queue.py` covers batch queue
  construction, duplicate run-id refusal, and the batch CLI.
- `src/tac/tests/test_staircase_dag.py` now covers MLX autopilot queue
  projection and two-root batch projection into two `local_mlx` Dask task
  specs with queue writeback and no authority fields.

## Live Batch Proof

Live artifact root:

`experiments/results/mlx_effective_spend_triage_learned_sweep_live_20260524T100114Z/`

Inputs:

- Plan:
  `local_mlx_chain_autopilot_run_20260524T105523Z/mlx_learned_sweep_chain_live_20260524t105523z/step_0002/cycle_0001/learned_sweep_plan.after_cycle.json`
- Selection:
  `normalized_effective_spend_triage_selection.json`
- Candidate payload:
  `learned_sweep_quality_candidates.json`

Batch artifacts:

- Spec:
  `mlx_autopilot_batch_spec_20260524T110916Z.json`
- Queue:
  `mlx_autopilot_batch_queue_20260524T110916Z.json`
- DAG:
  `mlx_autopilot_batch_dag_20260524T110916Z.json`
- Dispatch plan:
  `mlx_autopilot_batch_dispatch_plan_20260524T110916Z.json`
- Output roots:
  `local_mlx_batch_autopilot_run_20260524T110916Z/intermediate/...`
  and `local_mlx_batch_autopilot_run_20260524T110916Z/macro/...`

Live queue validation:

- Queue schema: `experiment_queue.v1`
- Queue id: `mlx_learned_sweep_batch_live_20260524t110916z`
- Experiments: `2`
- Steps: `2`
- Auto parallelism: `local_mlx: 2`
- Ready roots before execution: `intermediate`, `macro`
- False-authority fields: `score_claim=false`,
  `promotion_eligible=false`, `ready_for_exact_eval_dispatch=false`

Live worker execution:

```text
.venv/bin/python tools/experiment_queue.py \
  --queue experiments/results/mlx_effective_spend_triage_learned_sweep_live_20260524T100114Z/mlx_autopilot_batch_queue_20260524T110916Z.json \
  run-worker --execute --max-steps 2 --max-parallel 2 \
  --max-idle-cycles 0 --idle-sleep-seconds 0 --poll-interval-seconds 5 \
  --no-reload-definition
```

Worker result:

- `success_count=2`
- `failure_count=0`
- `claim_refused_count=0`
- `resource_limits={"local_mlx": 2}`
- Started experiments:
  - `..._intermediate`
  - `..._macro`
- Elapsed:
  - intermediate: `15.030704750039149s`
  - macro: `25.041431624966208s`
- Both terminal rows had no failed postconditions and no postcondition errors.

Result summaries:

- Both summaries have schema `mlx_dynamic_learned_sweep_local_autopilot.v1`.
- Both summaries report `gpu_launched=false`,
  `local_mlx_device_used=true`, `score_claim=false`,
  `promotion_eligible=false`, and
  `ready_for_exact_eval_dispatch=false`.
- Each root appended one observation, growing its copied observation ledger
  from `15` to `16` rows.
- Latest appended observations remained `mlx_dynamic_sweep_observation.v1`
  with `score_claim=false`, `promotion_eligible=false`,
  `ready_for_exact_eval_dispatch=false`, and `gpu_launched=false`.

Queue/DAG after execution:

- Queue status: `2 succeeded`, `0 orphaned`, no ready steps left.
- Performance summary: `experiment_queue_performance_summary.v1`,
  `telemetry_only=true`, `score_claim=false`,
  `promotion_eligible=false`, `rank_or_kill_eligible=false`,
  `ready_for_exact_eval_dispatch=false`.
- Live dispatch plan selected two independent `local_mlx` nodes with
  task resources `{"local_mlx": 1, "machine:m5": 1}` and required queue
  writeback.

## Adversarial Review Hardening

Dirac's read-only review found three P2 gaps after the first live batch proof:

- Batch CLI parsed top-level `--optimization-pass-id` but did not pass it into
  batch roots when per-run specs omitted the field.
- Queue postconditions checked the summary false-authority contract but only
  required the observation JSONL path to exist.
- The DAG batch test proved independent roots, but not that queue-level
  `local_mlx` concurrency still caps dispatch when the machine pool has more
  MLX slots than the queue allows.

Hardening landed in the working tree:

- `tools/build_mlx_learned_sweep_autopilot_queue.py` now passes top-level
  `--optimization-pass-id` into batch queue construction.
- `src/comma_lab/scheduler/mlx_learned_sweep_autopilot_queue.py` defaults that
  pass into each run unless a run spec overrides it.
- `src/comma_lab/scheduler/experiment_queue.py` supports
  `jsonl_false_authority` postconditions.
- MLX autopilot queue steps now attach `jsonl_false_authority` to the
  observation JSONL with schema
  `mlx_dynamic_sweep_observation.v1`.
- Tests now cover JSONL truthy-authority refusal, batch CLI top-level pass
  propagation, JSONL postcondition emission, and queue concurrency limiting
  under a richer `local_mlx=4` pool.

Hardened live rerun:

- Spec: `mlx_autopilot_batch_spec_20260524T111706Z.json`
- Queue: `mlx_autopilot_batch_queue_20260524T111706Z.json`
- DAG: `mlx_autopilot_batch_dag_20260524T111706Z.json`
- Dispatch plan: `mlx_autopilot_batch_dispatch_plan_20260524T111706Z.json`
- Queue validation: `local_mlx: 2`, two ready roots, and two
  `jsonl_false_authority` postconditions before execution.
- Worker result: `success_count=2`, `failure_count=0`,
  `claim_refused_count=0`, `resource_limits={"local_mlx": 2}`.
- Execution elapsed:
  - intermediate: `15.029566625016741s`
  - macro: `25.04086820804514s`
- Both hardened steps had no failed postconditions and no postcondition errors.
- Queue status after execution: `2 succeeded`, `0 orphaned`, `0 ready`.
- Performance summary remained telemetry-only with `score_claim=false`,
  `promotion_eligible=false`, `rank_or_kill_eligible=false`, and
  `ready_for_exact_eval_dispatch=false`.
- Hardened dispatch plan selected two `local_mlx` tasks, each with resources
  `{"local_mlx": 1, "machine:m5": 1}` and one `jsonl_false_authority` guard.

## Verification

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

Result after hardening: `131 passed in 4.10s`.

Additional checks:

- `ruff check` on touched MLX queue/autopilot/test/tool surfaces: passed.
- `git diff --check`: passed.

## Remaining Gaps

- The batch queue proves local MLX fan-out across independent roots, but it
  still executes one observation per root. The next tranche step is larger
  local saturation: multi-root plus larger `max_new_observations`,
  pass-diverse row allocation, and machine/resource-aware selection.
- The planner still needs the inverse-scorer/action-surface water-fill layer
  to choose roots by marginal utility instead of manually chosen
  `intermediate`/`macro` roots.
- Local MLX remains `[macOS-MLX research-signal]`; this session did not create
  score authority, exact-eval authority, promotion eligibility, or rank/kill
  authority.
- PixelShuffle should be treated as a NeRV/HNeRV/BoostNeRV primitive for
  local lowering/fusion smoke, not as a public submission intake item.
