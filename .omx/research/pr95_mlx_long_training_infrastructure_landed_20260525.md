# PR95 MLX Long-Training Infrastructure Landing

UTC: 2026-05-25
Agent: codex
Evidence grade: `[macOS-MLX research-signal]`

## Scope

This landing recovers the PR95/HNeRV MLX long-training infrastructure from an orphaned partner draft and wires it into normal operator and queue-observer flows. The prior subagent ledger claimed the module, CLI, tests, memo, and probe row had landed, but the referenced commit only changed lane metadata. This memo records the corrected landing.

## Landed Surfaces

- `src/tac/local_acceleration/pr95_hnerv_mlx_long_training.py`
  - canonical PR95 MLX long-training plan/provenance/telemetry schemas
  - trainable decoder plus per-pair latent bundle
  - explicit `rgb_frame_mse_local_mlx_research_mvp` fidelity class so the local loss is not mistaken for scorer-faithful PR95 training
  - checkpoint export that includes trained latents in the `.pt` for downstream `--latents-from-pt` packaging
  - explicit false-authority fields and exact-readiness refusal blockers
- `tools/run_pr95_mlx_long_training.py`
  - plan-only default CLI
  - explicit `--execute-smoke` for local MLX execution
  - queue-observable JSON report output
- `src/comma_lab/scheduler/experiment_queue_observer.py`
  - PR95 MLX long-training plan reports are surfaced as queue artifacts with lane id, candidate count, telemetry path, and readiness blockers
- `src/comma_lab/scheduler/local_training_queue.py`
  - `pr95_mlx_long_training_plan.v1` reports compile into `experiment_queue.v1` rows
  - `--output-report` is accepted as the declared manifest writer so the CLI is queue-owned instead of a loose operator command
- `.omx/research/codex_pr95_stage6_stage7_full_profile_queue_20260525T1714Z/pr95_mlx_long_training_plan_smoke_queue.json`
  - validated queue-owned plan-only smoke row for the recovered long-training report
- Tests:
  - `src/tac/tests/test_pr95_mlx_long_training_infrastructure.py`
  - `src/tac/tests/test_experiment_queue_observer.py`
  - `src/tac/tests/test_local_training_execution_queue.py`
  - `src/tac/tests/test_lane_maturity_harness.py`

## Authority Boundary

The long-training output remains local research signal only. The current training loss is RGB-frame MSE, not SegNet/PoseNet scorer-faithful PR95 training. It cannot claim score, promotion, dispatch readiness, or rank/kill authority. SegNet/PoseNet or calibrated scorer loss, exact CPU/CUDA auth eval, and full-frame inflate parity remain required before any contest-axis claim.

## Next

Generate top-family long-training plan reports through the new queue compiler path, then execute bounded source-video smokes with `--execute-smoke` and harvest telemetry through the observer.
