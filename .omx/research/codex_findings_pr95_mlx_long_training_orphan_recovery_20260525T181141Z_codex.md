# Codex Findings: PR95 MLX Long Training Orphan Recovery

UTC: 2026-05-25T18:11:41Z
Agent: codex
Scope: PR95/HNeRV MLX long-training infrastructure, queue visibility, false-authority safeguards.

## Findings

1. The shared subagent ledger claimed the long-training infrastructure was landed, but the referenced commit only changed lane metadata. The implementation file was still untracked, and the claimed CLI, tests, memo, probe row, and lane registry entries were absent from the repository.
2. The recovered module was useful but not yet queue-observable: it had no operator entry point, no plan/report schema, no observer support, and no regression tests guarding false-authority fields.
3. The recovered module also used a stale `[predicted]` axis marker for MLX provenance. PR95 MLX long-training is a local research substrate and must carry `[macOS-MLX research-signal]` tagging while refusing score, promotion, rank/kill, dispatch, and exact-readiness authority.

## What Landed

- `tac.local_acceleration.pr95_hnerv_mlx_long_training` now emits explicit plan/provenance/telemetry schemas with MLX evidence tags, exact-readiness refusal blockers, and reusable false-authority fields.
- The plan/provenance payloads explicitly label the current implementation `rgb_frame_mse_local_mlx_research_mvp` so downstream queues do not mistake RGB MSE training for SegNet/PoseNet scorer-faithful PR95 optimization.
- `tools/run_pr95_mlx_long_training.py` is a plan-only default operator CLI with an explicit `--execute-smoke` escape hatch for real local MLX work.
- `experiment_queue_observer` recognizes PR95 MLX long-training plan reports so queue rows surface lane id, total epochs, candidate registry count, telemetry path, and readiness blockers.
- `local_training_queue` now accepts `pr95_mlx_long_training_plan.v1` and `--output-report`, so the same report can compile into a bounded `experiment_queue.v1` row with false-authority postconditions.
- The recovered plan report is materialized as `pr95_mlx_long_training_plan_smoke_report.json` and compiled into validated queue `pr95_mlx_long_training_plan_smoke_queue.json`.
- Regression tests cover provenance, telemetry headers, candidate registry IDs, CLI report writing, and queue-observer visibility.

## Authority Boundary

This landing does not claim a score, promotion, dispatch readiness, or rank/kill authority. The current loss is RGB-frame MSE, not contest scorer-faithful SegNet/PoseNet loss. It makes the local MLX long-training substrate durable and queue-visible so later scorer-loss wiring, full-frame inflate parity, and paired contest CPU/CUDA auth eval can consume its outputs without inheriting false authority.

## Next Engineering

Wire a queue builder that creates PR95 MLX long-training plan rows for the top candidate families, then execute bounded source-video smoke runs and harvest their telemetry into the same observer/report path.
