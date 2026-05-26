# Codex Findings: MLX Repair Dynamics Component Queue

## Scope

Preserve the repair-dynamics signal from the empirical K=16 palette prior and bind it to a queue-owned MLX component-response path without granting score, promotion, rank/kill, dispatch, or budget-spend authority.

## Landed State

- Current `main`/`origin/main` commit `baf4961ef` contains the code wiring for repair dynamics priors in targeted component acquisition, refresh CLI input, repair waterfill queue propagation, MLX repair-dynamics command hints, and regression coverage.
- Durable component queue artifacts were written under `.omx/research/frontier_mlx_repair_dynamics_component_queue_20260526T151051Z/`.
- The queue contains 16 acquisition rows, 2 queued receiver-closed candidates, and selects `repair_dynamics_frame0_palette_interaction_waterfill` as the first family for both candidates.
- The embedded K=16 prior remains false-authority and records the key empirical dynamics: 15 frame-0 modes, 1 identity mode, 0 frame-1 modes, and first-class chroma/luma/RGB/roll repair families.

## MLX Execution Attempt

The queue validates as `experiment_queue.v1` and includes an MLX scorer-response step:

- `build_mlx_component_cache`
- `local_mlx_component_response`
- `harvest_targeted_component_correction_response_01`

A bounded worker run was started with `--execute --max-steps 6 --max-experiments 1 --max-parallel 1`. The worker completed both work-order emission steps, then entered `local_cpu_component_advisory`. That CPU bridge was stopped with `SIGTERM` before continuing because the operator requested MLX-only execution and immediate commit/no-signal-loss handling.

The resulting classification is persisted at `.omx/research/frontier_mlx_repair_dynamics_component_queue_20260526T151051Z/worker_stop_classification.json`.

## Blocker

The MLX response path is wired, but the current targeted-component queue still requires a CPU advisory/cache-hash bridge before MLX response. The next executor change should teach the queue to reuse or import an existing false-authority local advisory cache identity without rerunning CPU evaluation, then continue directly into the MLX cache and `tools/run_mlx_scorer_response_from_local_advisory.py --device gpu`.

## Compliance

No score claim was made. No promotion, rank/kill, paid dispatch, or budget-spend authority was granted. The MLX path remains `[macOS-MLX research-signal]` only until exact CPU/CUDA auth-axis evidence exists.
