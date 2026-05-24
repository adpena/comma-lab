# Codex Session Summary - 2026-05-24T10:56:58Z

## Scope

Extended the MLX learned-sweep queue compiler from one manual-sized queue step
to a queue-owned dependency chain. This lets `experiment_queue.v1` run multiple
bounded one-cycle MLX autopilot steps in sequence, with each step consuming the
previous step's replanned `mlx_dynamic_learned_sweep_plan.v1` artifact.

## Landed Artifacts

- Added `chain_steps` support to
  `comma_lab.scheduler.mlx_learned_sweep_autopilot_queue`.
- Added `--chain-steps` to
  `tools/build_mlx_learned_sweep_autopilot_queue.py`.
- Multi-step chains require `max_iterations=1` so every step has a stable
  `cycle_0001/learned_sweep_plan.after_cycle.json` handoff path.
- Chained steps declare queue dependencies:
  `run_mlx_learned_sweep_autopilot_0002` requires
  `run_mlx_learned_sweep_autopilot_0001`, etc.
- Each chained step postconditions the summary JSON, false authority, appended
  observation ledger, and the produced next learned-sweep plan.
- Added regression coverage for chained queue dependencies and rejection of
  ambiguous multi-cycle chained steps.

## Live Queue Evidence

Generated queue:

`experiments/results/mlx_effective_spend_triage_learned_sweep_live_20260524T100114Z/mlx_autopilot_chain_queue_20260524T105523Z.json`

Validation:

- `tools/experiment_queue.py --queue ... validate`: valid, one experiment, two
  `local_mlx` steps.
- `tools/experiment_queue.py --queue ... next`: first chained step ready.
- `tools/experiment_queue.py --queue ... run-worker --execute --max-steps 2`:
  succeeded with `failure_count=0`, `success_count=2`, no postcondition errors,
  and no orphaned steps.

The chain wrote:

- `step_0001/local_mlx_autopilot_summary.json`
- `step_0001/cycle_0001/learned_sweep_plan.after_cycle.json`
- `step_0002/local_mlx_autopilot_summary.json`
- `step_0002/cycle_0001/learned_sweep_plan.after_cycle.json`
- shared append-only observation ledger:
  `local_mlx_chain_autopilot_observations_20260524T105523Z.jsonl`

Executed rows:

- `mlx_scorer_response:window:467:468::mlx_local_response::micro`
- `mlx_scorer_response:window:109:110::mlx_local_response::micro`

Final chain state:

- observation ledger grew from 13 to 15 rows.
- final plan summary: observation_row_count=15,
  suppressed_observed_row_count=15, ranked_row_count=113,
  local_ready_row_count=49.
- local device usage was recorded as `local_mlx_device_used=true`;
  `gpu_launched=false`, `score_claim=false`,
  `ready_for_exact_eval_dispatch=false`.

## Verification

- `pytest -q src/tac/tests/test_mlx_dynamic_learned_sweep_local_autopilot.py src/tac/tests/test_mlx_dynamic_learned_sweep_local_actuator.py src/tac/tests/test_mlx_dynamic_learned_sweep_observation_harvest.py src/tac/tests/test_mlx_effective_spend_triage_selection.py src/tac/tests/test_mlx_effective_spend_triage_learned_sweep_adapter.py src/tac/tests/test_mlx_dynamic_learned_sweep.py src/tac/tests/test_mlx_dynamic_sweep_observations.py src/tac/tests/test_cross_family_candidate_portfolio.py src/tac/tests/test_mlx_learned_sweep_autopilot_queue.py src/tac/tests/test_mlx_execution_queue.py src/tac/tests/test_staircase_dag.py`
  passed: 124 tests.
- `ruff check` on touched MLX/autopilot/queue/DAG files passed.
- `git diff --check` passed.

## Remaining Gaps

- Chains are sequential because every step consumes the previous replan. Parallel
  saturation now needs sibling chain generation over independent lane roots or
  disjoint candidate families.
- Storage-tier placement for generated queue outputs is not yet part of this
  queue compiler.
- CPU-advisory, optimizer-scheduler paired ablations, archive materialization,
  and exact-gated rows still require dedicated actuators/queue compilers.
- Exact auth eval remains separate authority; this chain only emits local MLX
  research observations for replanning.
