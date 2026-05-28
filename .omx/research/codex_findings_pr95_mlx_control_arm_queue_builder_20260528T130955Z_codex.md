# Codex Findings: PR95 MLX Control-Arm Queue Builder

UTC: 2026-05-28T13:09:55Z

## Scope

The PR95/HNeRV MLX timing planner now writes an execution queue directly when invoked with `--plan-only --write-execution-queue`.

Artifacts:

- `.omx/research/pr95_mlx_control_arm_queue_builder_smoke_20260528T130916Z/plan.json`
- `.omx/research/pr95_mlx_control_arm_queue_builder_smoke_20260528T130916Z/representation_training_plan.json`
- `.omx/research/pr95_mlx_control_arm_queue_builder_smoke_20260528T130916Z/queue.json`
- `.omx/research/pr95_mlx_control_arm_queue_builder_smoke_20260528T130916Z/queue_validate.json`

## Findings

- The timing-smoke planner now compiles the local MLX execution queue without a separate manual `build_local_training_execution_queue.py` call.
- The generated queue validates as `experiment_queue.v1`.
- The smoke queue used `local_mlx` concurrency `2` and retained the default harvested no-cliff preset `blocks02_kahan_fp32`.
- The summary keeps all authority fields false and reports the queue path, queue id, and experiment count.

## Authority

This is queue-construction and local-control evidence only. It is not a score claim, not promotion authority, and not exact-eval dispatch authority.

## Integration

This reduces the PR95 MLX reproduction lane from a manual plan-builder chain into a single queue-owned control-arm entrypoint:

```bash
.venv/bin/python tools/run_pr95_mlx_timing_smoke.py \
  --stage 8 \
  --steps 1 \
  --batch-size 1 \
  --synthetic-pairs 1 \
  --seed 23 \
  --base-channels 36 \
  --output-dir <run-dir> \
  --write-mlx-gpu-drift-attestation \
  --plan-only \
  --write-execution-queue
```

The produced `<run-dir>/queue.json` can be executed by `tools/experiment_queue.py run-worker --execute`, preserving the scheduler as authority.
