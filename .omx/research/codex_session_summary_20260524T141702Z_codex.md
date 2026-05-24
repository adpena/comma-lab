# Codex Session Summary: MLX Concurrency Probe And Exhaustion Guard

UTC: 2026-05-24T14:17:02Z
Lane: `lane_mlx_learned_sweep_auto_batch_roots_20260524`
Evidence grade: `[macOS-MLX research-signal]`

## What Landed

- Built a widened local MLX learned-sweep queue from:
  `experiments/results/mlx_effective_spend_triage_learned_sweep_live_20260524T100114Z/mlx_autopilot_named_policy_cumulative_observation_replan_20260524T140642Z.json`.
- Queue artifact:
  `experiments/results/mlx_effective_spend_triage_learned_sweep_live_20260524T100114Z/mlx_autopilot_named_policy_cumulative_batch_queue_20260524T141114Z.json`.
- Root plan:
  `experiments/results/mlx_effective_spend_triage_learned_sweep_live_20260524T100114Z/mlx_autopilot_named_policy_cumulative_batch_roots_20260524T141114Z.json`.
- Staircase DAG:
  `experiments/results/mlx_effective_spend_triage_learned_sweep_live_20260524T100114Z/mlx_autopilot_named_policy_cumulative_batch_dag_20260524T141114Z.json`.
- Raised local MLX concurrency from 2 to 4 for this bounded probe.

## Queue Execution Evidence

- Queue id: `mlx_learned_sweep_named_policy_cumulative_batch_live_20260524T141114Z`.
- State DB:
  `.omx/state/experiment_queue_mlx_learned_sweep_named_policy_cumulative_batch_live_20260524T141114Z.sqlite`.
- Worker result: `success_count=4`, `failure_count=0`, `claim_refused_count=0`, `stop_reason=max_steps_reached`.
- No failed postconditions, no postcondition errors, and no timeouts.
- Queue status after execution: `{"succeeded": 4}` with `orphaned_step_count=0` and `ready_steps=[]`.
- Local MLX performance telemetry:
  - `success_count=4`
  - `failure_count=0`
  - `elapsed_seconds_mean=60.30730101051449`
  - `elapsed_seconds_sum=241.22920404205797`
  - `artifact_record_count=12`

## New Observation Ledgers

- `experiments/results/mlx_effective_spend_triage_learned_sweep_live_20260524T100114Z/mlx_autopilot_named_policy_cumulative_batch_observations_20260524T141114Z_pass_0001_mlx_scorer_response_window_496_497__mlx_local_response__macro.jsonl`
  - 2 rows.
- `experiments/results/mlx_effective_spend_triage_learned_sweep_live_20260524T100114Z/mlx_autopilot_named_policy_cumulative_batch_observations_20260524T141114Z_pass_0002_mlx_scorer_response_window_98_99__mlx_local_response__macro.jsonl`
  - 2 rows.
- `experiments/results/mlx_effective_spend_triage_learned_sweep_live_20260524T100114Z/mlx_autopilot_named_policy_cumulative_batch_observations_20260524T141114Z_pass_0003_mlx_scorer_response_window_501_502__mlx_local_response__macro.jsonl`
  - 2 rows.
- `experiments/results/mlx_effective_spend_triage_learned_sweep_live_20260524T100114Z/mlx_autopilot_named_policy_cumulative_batch_observations_20260524T141114Z_pass_0004_mlx_scorer_response_window_479_480__mlx_local_response__macro.jsonl`
  - 2 rows.

## Cumulative Replan Evidence

- Replan JSON:
  `experiments/results/mlx_effective_spend_triage_learned_sweep_live_20260524T100114Z/mlx_autopilot_named_policy_cumulative_observation_replan_20260524T141312Z.json`.
- Replan summary:
  `experiments/results/mlx_effective_spend_triage_learned_sweep_live_20260524T100114Z/mlx_autopilot_named_policy_cumulative_observation_replan_summary_20260524T141312Z.json`.
- Replan markdown:
  `experiments/results/mlx_effective_spend_triage_learned_sweep_live_20260524T100114Z/mlx_autopilot_named_policy_cumulative_observation_replan_20260524T141312Z.md`.
- `observation_jsonl_count=12`.
- `raw_observation_row_count=32`.
- `observation_row_count=32`.
- `duplicate_observation_row_count=0`.
- `suppressed_observed_row_count=32`.
- `ranked_row_count=96`.
- `local_ready_row_count=32`.

## Exhausted Local MLX Guard

The next attempted MLX queue build from the cumulative replan had no remaining `mlx_local_response` rows. Before this patch, runtime telemetry discovery failed first with:

`cannot discover runtime telemetry states without ready mlx_local_response queue_candidate_id rows`

That was misleading because telemetry discovery is advisory. The builder now returns no discovered telemetry in that condition and lets the batch-root planner report the real availability verdict:

`FATAL: plan has no ready mlx_local_response rows for automatic batch roots`

Changed files:

- `tools/build_mlx_learned_sweep_autopilot_queue.py`
- `src/tac/tests/test_mlx_learned_sweep_autopilot_queue.py`

## Authority Boundary

- `score_claim=false`.
- `score_claim_valid=false`.
- `promotion_eligible=false`.
- `rank_or_kill_eligible=false`.
- `ready_for_exact_eval_dispatch=false`.
- The 4-way local MLX batch is throughput/advisory evidence only. It does not create score, promotion, rank/kill, or exact-dispatch authority.

## Verification

- `ruff check ...`: pass.
- `git diff --check`: pass.
- Focused regression:
  `.venv/bin/python -m pytest -q src/tac/tests/test_experiment_queue.py src/tac/tests/test_staircase_dag.py src/tac/tests/test_mlx_effective_spend_triage_selection.py src/tac/tests/test_mlx_effective_spend_triage_learned_sweep_adapter.py src/tac/tests/test_mlx_dynamic_learned_sweep_observation_harvest.py src/tac/tests/test_mlx_dynamic_learned_sweep_local_actuator.py src/tac/tests/test_mlx_dynamic_learned_sweep_local_autopilot.py src/tac/tests/test_mlx_learned_sweep_autopilot_queue.py src/tac/tests/test_mlx_dynamic_sweep_observations.py`
  - `178 passed in 11.67s`.
- `.venv/bin/python tools/lane_maturity.py validate`
  - `OK - 1242 lane(s) validated cleanly.`

## Next Step

The current MLX local-response surface is exhausted for this learned-sweep plan. The next production move is to wire a CPU-advisory or exact-calibration actuator for the remaining `macos_cpu_advisory` rows, or regenerate a higher-level learned-sweep candidate payload with new candidate families before returning to the MLX queue builder.
