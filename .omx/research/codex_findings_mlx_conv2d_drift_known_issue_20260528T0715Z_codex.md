# Codex Findings - MLX Conv2d Drift Known-Issue Check - 2026-05-28T07:15Z

## Verdict

The PR95 MLX drift is a known numerical-computing class, not an MLX-specific
scandal: float32 Conv2d is a reduction-heavy operation, and CPU/GPU/backend
implementations are not bitwise portable. PyTorch documents the same class:
floating-point operations are not associative, and CPU/GPU results can differ
even for identical inputs after randomness is controlled.

MLX 0.31.2 does not expose a `precise=True` or deterministic-reduction flag for
`mx.conv2d`; the public signature is limited to input, weight, stride, padding,
dilation, groups, and stream. Metal is the relevant MLX GPU backend on this
machine. The local runtime also proves `fixed_fp64` cannot execute on MLX GPU:
`float64 is not supported on the GPU`.

## Engineering Adaptation Landed

- Extracted one shared fixed-order MLX Conv2d primitive:
  `tac.local_acceleration.mlx_scorer_adapters.mlx_reference_conv2d_nhwc`.
- Refactored `MLXReferenceConv2dAdapter` to delegate to that primitive, removing
  the duplicated fixed/Kahan/FP64 loop.
- Wired PR95 `HNeRVDecoderMLX` to explicit `conv2d_accumulation_mode`:
  `optimized`, `fixed_fp32`, `kahan_fp32`, `fixed_fp64`.
- Wired PR95 probe/export tools to record and select the accumulation mode.
- Kept `optimized` as the default training path for throughput; fixed modes are
  explicit parity/debug/calibration paths until they prove enough score value to
  justify runtime cost.

## Measurements

CPU per-op reports after the refactor:

- `optimized`: `.omx/research/pr95_mlx_per_op_drift_optimized_after_shared_conv_20260528Tlocal/report.json`
  - full decoder max_abs `4.5776e-05`, mean_abs `4.8007e-06`
- `fixed_fp32`: `.omx/research/pr95_mlx_per_op_drift_fixed_fp32_after_shared_conv_20260528Tlocal/report.json`
  - full decoder max_abs `3.0518e-05`, mean_abs `4.8142e-06`
- `kahan_fp32`: `.omx/research/pr95_mlx_per_op_drift_kahan_fp32_after_shared_conv_20260528Tlocal/report.json`
  - full decoder max_abs `3.0518e-05`, mean_abs `4.4156e-06`
- `fixed_fp64`: `.omx/research/pr95_mlx_per_op_drift_fixed_fp64_after_shared_conv_20260528Tlocal/report.json`
  - full decoder max_abs `3.0518e-05`, mean_abs `4.3931e-06`

GPU per-op reports:

- `optimized`: `.omx/research/pr95_mlx_per_op_drift_gpu_optimized_after_shared_conv_20260528Tlocal/report.json`
  - Conv2d max_abs `1.6263e-03`; full decoder max_abs `5.3787e-03`
- `fixed_fp32`: `.omx/research/pr95_mlx_per_op_drift_gpu_fixed_fp32_after_shared_conv_20260528Tlocal/report.json`
  - Conv2d passes local band, but full decoder remains framework-different:
    max_abs `3.6774e-03`
- `kahan_fp32`: `.omx/research/pr95_mlx_per_op_drift_gpu_kahan_fp32_after_shared_conv_20260528Tlocal/report.json`
  - Conv2d passes local band, but full decoder remains framework-different:
    max_abs `3.6926e-03`

## Sources Consulted

- MLX `mx.conv2d` 0.31.2 public signature: no `precise`, deterministic, or
  accumulation-mode parameter:
  https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.conv2d.html
- MLX unified-memory/device docs: MLX arrays can execute on CPU or GPU without
  manual copies, which is the relevant local-substrate advantage:
  https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html
- PyTorch numerical-accuracy notes: floating point operation order affects
  results; bitwise identity is not guaranteed across platforms:
  https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html
- PyTorch reproducibility notes: CPU/GPU reproducibility is not guaranteed even
  with identical seeds; deterministic algorithms can cost throughput:
  https://docs.pytorch.org/docs/stable/notes/randomness.html

## Optimal Path

1. Keep MLX GPU training as the high-throughput optimizer substrate.
2. Keep archive export/readiness gates on MLX CPU plus PyTorch parity, then
   exact CPU/CUDA auth eval.
3. Use `kahan_fp32` or `fixed_fp64` on CPU as calibration/proof paths, not
   default training, unless a measured candidate shows better score movement.
4. For GPU training, treat fixed/Kahan Conv2d as a diagnostic mitigation only:
   it fixes isolated Conv2d drift but not the whole decoder on Metal.
5. Continue deeper GPU drift isolation at layer-trace granularity before writing
   custom Metal kernels; a custom kernel is justified only if the layer trace
   shows a localized high-impact kernel and the throughput/score tradeoff wins.
