# Codex Session Summary: MLX Learned Sweep Second Queue Batch

UTC: 2026-05-24T14:07:50Z
Lane: `lane_mlx_learned_sweep_auto_batch_roots_20260524`
Evidence grade: `[macOS-MLX research-signal]`

## What Landed

- Built a follow-up local MLX learned-sweep queue from:
  `experiments/results/mlx_effective_spend_triage_learned_sweep_live_20260524T100114Z/mlx_autopilot_named_policy_cumulative_observation_replan_20260524T135939Z.json`.
- Queue artifact:
  `experiments/results/mlx_effective_spend_triage_learned_sweep_live_20260524T100114Z/mlx_autopilot_named_policy_cumulative_batch_queue_20260524T140508Z.json`.
- Root plan:
  `experiments/results/mlx_effective_spend_triage_learned_sweep_live_20260524T100114Z/mlx_autopilot_named_policy_cumulative_batch_roots_20260524T140508Z.json`.
- Staircase DAG:
  `experiments/results/mlx_effective_spend_triage_learned_sweep_live_20260524T100114Z/mlx_autopilot_named_policy_cumulative_batch_dag_20260524T140508Z.json`.
- The selected water-fill roots covered 6 queue-candidate rows:
  - `mlx_scorer_response:window:59:60::mlx_local_response::micro`
  - `mlx_scorer_response:window:109:110::mlx_local_response::micro`
  - `mlx_scorer_response:window:467:468::mlx_local_response::intermediate`
  - `mlx_scorer_response:window:467:468::mlx_local_response::micro`
  - `mlx_scorer_response:window:109:110::mlx_local_response::intermediate`
  - `mlx_scorer_response:window:544:545::mlx_local_response::intermediate`

## Queue Execution Evidence

- Queue id: `mlx_learned_sweep_named_policy_cumulative_batch_live_20260524T140508Z`.
- State DB:
  `.omx/state/experiment_queue_mlx_learned_sweep_named_policy_cumulative_batch_live_20260524T140508Z.sqlite`.
- Worker result: `success_count=2`, `failure_count=0`, `claim_refused_count=0`, `stop_reason=max_steps_reached`.
- No failed postconditions, no postcondition errors, and no timeouts.
- Queue status after execution: `{"succeeded": 2}` with `orphaned_step_count=0` and `ready_steps=[]`.
- Local MLX performance telemetry:
  - `success_count=2`
  - `failure_count=0`
  - `elapsed_seconds_mean=43.19945991699933`
  - `elapsed_seconds_sum=86.39891983399866`
  - `artifact_record_count=6`

## New Observation Ledgers

- `experiments/results/mlx_effective_spend_triage_learned_sweep_live_20260524T100114Z/mlx_autopilot_named_policy_cumulative_batch_observations_20260524T140508Z_pass_0001_mlx_scorer_response_window_59_60__mlx_local_response__micro.jsonl`
  - 3 rows.
- `experiments/results/mlx_effective_spend_triage_learned_sweep_live_20260524T100114Z/mlx_autopilot_named_policy_cumulative_batch_observations_20260524T140508Z_pass_0002_mlx_scorer_response_window_467_468__mlx_local_response__micro.jsonl`
  - 3 rows.

## Cumulative Replan Evidence

- Replan JSON:
  `experiments/results/mlx_effective_spend_triage_learned_sweep_live_20260524T100114Z/mlx_autopilot_named_policy_cumulative_observation_replan_20260524T140642Z.json`.
- Replan summary:
  `experiments/results/mlx_effective_spend_triage_learned_sweep_live_20260524T100114Z/mlx_autopilot_named_policy_cumulative_observation_replan_summary_20260524T140642Z.json`.
- Replan markdown:
  `experiments/results/mlx_effective_spend_triage_learned_sweep_live_20260524T100114Z/mlx_autopilot_named_policy_cumulative_observation_replan_20260524T140642Z.md`.
- `observation_jsonl_count=8`.
- `raw_observation_row_count=24`.
- `observation_row_count=24`.
- `duplicate_observation_row_count=0`.
- `suppressed_observed_row_count=24`.
- `ranked_row_count=104`.
- `local_ready_row_count=40`.

## Authority Boundary

- `score_claim=false`.
- `score_claim_valid=false`.
- `promotion_eligible=false`.
- `rank_or_kill_eligible=false`.
- `ready_for_exact_eval_dispatch=false`.
- All rows remain planning / local-research-signal only; no score, promotion, rank/kill, or exact-dispatch authority was created.

## Verification

- Queue validate: `valid=true`, `experiment_count=2`, `step_count=2`, `local_mlx` concurrency cap 2.
- DAG emission succeeded with planning-only blockers intact.
- Prior focused verification in this tranche remains green:
  - ruff check pass.
  - `git diff --check` pass.
  - targeted pytest suite: `177 passed in 16.16s`.
  - lane maturity validate: `OK - 1242 lane(s) validated cleanly.`

## Next Step

Use `mlx_autopilot_named_policy_cumulative_observation_replan_20260524T140642Z.json` as the next queue source. The next actuator improvement should use measured queue telemetry to decide whether `local_mlx` concurrency can safely rise above 2, while still preserving MLX as advisory/local research signal only.
