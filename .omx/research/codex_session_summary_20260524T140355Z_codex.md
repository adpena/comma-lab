# Codex Session Summary: MLX Learned Sweep Queue Batch

UTC: 2026-05-24T14:03:55Z
Lane: `lane_mlx_learned_sweep_auto_batch_roots_20260524`
Evidence grade: `[macOS-MLX research-signal]`

## What Landed

- Built the next local MLX learned-sweep queue from the cumulative observation replan:
  `experiments/results/mlx_effective_spend_triage_learned_sweep_live_20260524T100114Z/mlx_autopilot_named_policy_cumulative_batch_queue_20260524T135713Z.json`.
- Emitted the matching staircase DAG:
  `experiments/results/mlx_effective_spend_triage_learned_sweep_live_20260524T100114Z/mlx_autopilot_named_policy_cumulative_batch_dag_20260524T135713Z.json`.
- Executed the bounded queue worker with `--max-parallel 2`, `--max-steps 2`, and `--max-experiments 2`.
- Both local MLX autopilot steps succeeded; no claim refusals, no failed postconditions, no postcondition errors, and no timeouts.

## Queue Execution Evidence

- Queue id: `mlx_learned_sweep_named_policy_cumulative_batch_live_20260524T135713Z`.
- State DB:
  `.omx/state/experiment_queue_mlx_learned_sweep_named_policy_cumulative_batch_live_20260524T135713Z.sqlite`.
- Status after execution: `{"succeeded": 2}` with `orphaned_step_count=0` and `ready_steps=[]`.
- Local MLX performance telemetry:
  - `success_count=2`
  - `failure_count=0`
  - `elapsed_seconds_mean=38.177212000009604`
  - `elapsed_seconds_sum=76.35442400001921`
  - `artifact_record_count=6`

## New Observation Ledgers

- `experiments/results/mlx_effective_spend_triage_learned_sweep_live_20260524T100114Z/mlx_autopilot_named_policy_cumulative_batch_observations_20260524T135713Z_pass_0001_mlx_scorer_response_window_109_110__mlx_local_response__smoke.jsonl`
  - 3 rows.
- `experiments/results/mlx_effective_spend_triage_learned_sweep_live_20260524T100114Z/mlx_autopilot_named_policy_cumulative_batch_observations_20260524T135713Z_pass_0002_mlx_scorer_response_window_544_545__mlx_local_response__smoke.jsonl`
  - 3 rows.

## Cumulative Replan Evidence

- Replan JSON:
  `experiments/results/mlx_effective_spend_triage_learned_sweep_live_20260524T100114Z/mlx_autopilot_named_policy_cumulative_observation_replan_20260524T135939Z.json`.
- Replan summary:
  `experiments/results/mlx_effective_spend_triage_learned_sweep_live_20260524T100114Z/mlx_autopilot_named_policy_cumulative_observation_replan_summary_20260524T135939Z.json`.
- Replan markdown:
  `experiments/results/mlx_effective_spend_triage_learned_sweep_live_20260524T100114Z/mlx_autopilot_named_policy_cumulative_observation_replan_20260524T135939Z.md`.
- `observation_jsonl_count=6`.
- `raw_observation_row_count=18`.
- `observation_row_count=18`.
- `duplicate_observation_row_count=0`.
- `suppressed_observed_row_count=18`.
- `ranked_row_count=110`.
- `local_ready_row_count=46`.

## Authority Boundary

- `score_claim=false`.
- `score_claim_valid=false` in queue telemetry.
- `promotion_eligible=false`.
- `rank_or_kill_eligible=false` in queue telemetry.
- `ready_for_exact_eval_dispatch=false`.
- The generated DAG remains planning-only and retains exact-eval, lane-claim, non-proxy-score, and promotion blockers.

## Verification

- `ruff check ...` on the MLX queue, replan, actuator, scheduler, DAG, and focused tests: pass.
- `git diff --check`: pass.
- Focused regression:
  `.venv/bin/python -m pytest -q src/tac/tests/test_experiment_queue.py src/tac/tests/test_staircase_dag.py src/tac/tests/test_mlx_effective_spend_triage_selection.py src/tac/tests/test_mlx_effective_spend_triage_learned_sweep_adapter.py src/tac/tests/test_mlx_dynamic_learned_sweep_observation_harvest.py src/tac/tests/test_mlx_dynamic_learned_sweep_local_actuator.py src/tac/tests/test_mlx_dynamic_learned_sweep_local_autopilot.py src/tac/tests/test_mlx_learned_sweep_autopilot_queue.py src/tac/tests/test_mlx_dynamic_sweep_observations.py`
  - `177 passed in 16.16s`.
- `.venv/bin/python tools/lane_maturity.py validate`
  - `OK - 1242 lane(s) validated cleanly.`

## Next Step

Build the next queue from `mlx_autopilot_named_policy_cumulative_observation_replan_20260524T135939Z.json`, keep `local_mlx` concurrency at 2 or raise only after measuring memory pressure, and preserve the same authority boundary until a full-sample contest CPU/CUDA calibration payload exists.
