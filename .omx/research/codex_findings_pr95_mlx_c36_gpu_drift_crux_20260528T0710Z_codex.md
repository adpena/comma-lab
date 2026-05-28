# Codex Findings - PR95 MLX C36 GPU Drift Crux - 2026-05-28T07:10Z

## Verdict

The real-width PR95/HNeRV MLX GPU drift is primarily Conv2d accumulation-order
drift, not an unresolved full-decoder mystery.

Queue-owned stage-8/base-c36 trace with native MLX Conv2d:

- Artifact: `.omx/research/pr95_mlx_stage8_c36_gpu_trace_optimized_20260528Tlocal/`
- Mode: `optimized`
- First cliff at `cliff_threshold=1e-3`: `rgb_0`
- Output max/mean drift: `0.00702667236328125` / `0.000669580593239516`
- Largest rows: `rgb_0`, `output`, `rgb_1`

Same queue-owned trace with explicit fixed-order Kahan Conv2d:

- Artifact: `.omx/research/pr95_mlx_stage8_c36_gpu_trace_kahan_20260528Tlocal/`
- Mode: `kahan_fp32`
- First cliff at `cliff_threshold=1e-3`: none
- Output max/mean drift: `3.0517578125e-05` / `3.816443495452404e-06`
- Largest rows: `rgb_0`, `rgb_1`, `output`

## Engineering Consequence

For PR95-class MLX reproduction, use native `optimized` Conv2d for training
throughput and `kahan_fp32` for GPU parity/calibration/export-readiness traces
when comparing against PyTorch. This is a real engineering adaptation, not a
score claim: all artifacts remain `[macOS-MLX research-signal]` and fail closed
for exact auth authority.

The next performance question is whether Kahan is needed only at RGB heads or
whether full fixed-order Conv2d is cheap enough at export/calibration time. The
next implementation step should add per-module Conv2d accumulation-mode
selection so training can stay native while parity traces can selectively use
Kahan on `rgb_0`/`rgb_1` first, then broaden only if a trace proves it.
