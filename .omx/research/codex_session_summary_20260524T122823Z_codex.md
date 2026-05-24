# Codex Session Summary - 2026-05-24T12:28:23Z

Evidence grade: `[repo-local/live-queue/MLX-research-signal]`

## Scope

Hardened the adaptive MLX learned-sweep batch-root mode after sidecar audit.
The adaptive mode still water-fills positive-utility `mlx_local_response` rows
across requested local MLX roots, but now fails closed earlier on exact-filter
ambiguity and exposes full false-authority metadata on CLI summaries.

This remains local MLX research signal. It is not score, promotion, rank/kill,
or exact-dispatch authority.

## Audit Fixes

- Mixed-pass roots now set singular `optimization_pass_id=null` and carry
  `representative_optimization_pass_id` separately. Run specs still omit the
  pass filter for mixed roots.
- Root planning rejects duplicate selected `queue_candidate_id` values before
  queue emission.
- Queue building rejects duplicate ready rows for requested `queue_candidate_id`
  filters before execution, so exact-row ambiguity cannot run and fail only
  after side effects.
- Fixed-mode CLI now rejects explicit `--auto-batch-rows-per-root 0`.
- Adaptive-mode planner errors are caught as `FATAL:` CLI errors instead of
  uncaught tracebacks.
- CLI summaries now include `score_claim_valid`, `rank_or_kill_eligible`,
  `promotable`, `dispatch_attempted`, and `gpu_launched`, all false.

## Fresh Live Proof

Artifacts:

- Root plan:
  `experiments/results/mlx_effective_spend_triage_learned_sweep_live_20260524T100114Z/mlx_autopilot_adaptive_row_group_batch_roots_20260524T122646Z.json`
- Queue:
  `experiments/results/mlx_effective_spend_triage_learned_sweep_live_20260524T100114Z/mlx_autopilot_adaptive_row_group_batch_queue_20260524T122646Z.json`
- Queue state:
  `.omx/state/experiment_queue_mlx_learned_sweep_adaptive_row_group_batch_live_20260524T122646Z.sqlite`

Root plan:

- `adaptive_rows_per_root=true`
- `rows_per_root=null`
- `max_rows_per_root=3`
- `eligible_positive_utility_row_count=17`
- `selected_total_row_count=6`
- `selected_root_count=2`
- root sizes `[3, 3]`
- `queue_candidate_disjoint_guaranteed=true`
- mixed root has `optimization_pass_id=null`,
  `representative_optimization_pass_id="micro"`, and
  `optimization_pass_ids=["intermediate", "micro"]`
- `score_claim=false`
- `score_claim_valid=false`
- `rank_or_kill_eligible=false`
- `promotable=false`
- `promotion_eligible=false`
- `ready_for_exact_eval_dispatch=false`
- `dispatch_attempted=false`
- `gpu_launched=false`

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
- elapsed `35.0688s` and `40.0754s`
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
  - full false-authority fields false.

Observation ledgers:

- `mlx_autopilot_adaptive_row_group_batch_observations_20260524T122646Z_pass_0001_...jsonl`
  has 3 rows.
- `mlx_autopilot_adaptive_row_group_batch_observations_20260524T122646Z_pass_0002_...jsonl`
  has 3 rows.

## Verification

- `pytest -q src/tac/tests/test_mlx_learned_sweep_autopilot_queue.py`:
  `26 passed in 1.17s`.
- Focused MLX/autopilot queue suite:
  `40 passed in 1.39s`.
- Broader MLX/scheduler tranche:
  `149 passed in 4.64s`.

## Remaining Gaps

- Adaptive grouping still uses planned acquisition/cost estimates; it does not
  yet learn runtime-per-row from prior queue telemetry.
- Candidate-root chained walks still need explicit duplicate/exhaustion
  semantics.
- Archive materialization, byte-closed export, exact CPU/CUDA auth-eval
  dispatch, and promotion gates remain downstream authority surfaces.
