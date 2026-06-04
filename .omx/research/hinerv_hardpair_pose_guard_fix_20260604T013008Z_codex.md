# HiNeRV hard-pair pose guard fix

Date: 2026-06-04T01:30:08Z  
Author: codex  
Authority: `[macOS-MLX research-signal]`, no score claim

## Finding

The HiNeRV hard-pair successor consumed the full-video hitlist correctly, but
the generic pose-instability monitor treated sampled hard-pair pose loss as if
it were an unbiased full-video instability estimate. That is false for
prioritized hard-pair refits: the sampler intentionally oversamples the
highest-error tail, so sampled `per_axis.pose >= 1000` is expected during the
repair window and must not stop the run.

A related resumed-run control was also made explicit: the pose guard's
`min_epoch` is now interpreted as local epochs since the resume checkpoint,
not absolute global epoch zero.

## Change

- `tools/run_compact_renderer_mlx_spine_runner.py` now initializes
  `_PoseInstabilityEpochMonitor` with the checkpoint resume start epoch.
- For HiNeRV runs with `prioritized_pair_indices`, the monitor still records
  bad pose-axis telemetry but suppresses hard stops from the biased hard-pair
  sampled axis.
- Ordinary non-prioritized runs keep the existing sustained-pose-instability
  fail-fast behavior.

## Evidence

Focused regression:

```bash
uv run pytest src/tac/tests/test_compact_renderer_mlx_spine_runner.py -q -k 'pose_instability_epoch_monitor or resume_start_epoch_for_pose_monitor'
uv run ruff check tools/run_compact_renderer_mlx_spine_runner.py src/tac/tests/test_compact_renderer_mlx_spine_runner.py
```

Result: 4 tests passed; ruff passed.

Hard-pair receiver-proven full600 MLX replay from the pre-fix short packet:

- archive bytes: 214,498
- full-video MLX score: 69.85158383259508
- avg SegNet distortion: 0.30224286392331123
- avg PoseNet distortion: 155.9023531214396
- exact CPU/CUDA spend gate: blocked, `mlx_response_not_within_cpu_spend_band`

This replay is not a promotion candidate. It is useful evidence that hard-pair
sampling is directionally score-improving versus the prior epoch7749 replay
while still far from frontier, and that the early sampled-pose axis should be
logged as hard-pair repair pressure rather than used as a kill switch.

