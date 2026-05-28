# PR95 MLX C36 Conv2d Drift Scope

Generated: 2026-05-28T12:19Z
Author: Codex
Lane: `pr95_hnerv_mlx_reproduction`
Evidence grade: `[macOS-MLX research-signal]`

## Verdict

The production-width PR95 MLX GPU drift is not removed by pinning only the RGB
heads, or refine+RGB heads, to Kahan Conv2d accumulation. Pinning the six
upsample blocks to Kahan closes the observed C36 trace cliff to the same
3.0517578125e-05 max-delta band as full-decoder Kahan.

The visible `rgb_0` cliff is therefore an amplification point for upstream
feature drift, not proof that the RGB head arithmetic alone is the root cause.

## C36 GPU Drift Matrix

All rows used:

```bash
PYTHONPATH=. .venv/bin/python tools/run_pr95_mlx_timing_smoke.py \
  --stage 8 --steps 1 --batch-size 1 --synthetic-pairs 1 --seed 37 \
  --base-channels 36 --write-pr95-public-archive-export \
  --write-mlx-gpu-drift-attestation --mlx-gpu-drift-conv2d-accumulation-mode optimized \
  --pytorch-export-atol-max 0.01 --pytorch-export-atol-mean 0.001
```

| Scope | Artifact | Max abs | Mean abs | First cliff |
| --- | --- | ---: | ---: | --- |
| optimized baseline | `.omx/research/pr95_mlx_stage8_c36_gpu_trace_optimized_20260528Tlocal/` | 0.00702667236328125 | 0.000669580593239516 | `rgb_0` |
| `rgb_heads_kahan_fp32` | `.omx/research/pr95_mlx_stage8_c36_gpu_trace_rgb_heads_kahan_20260528Tlocal/` | 0.00702667236328125 | 0.0006695800111629069 | `rgb_0` |
| `refine_rgb_heads_kahan_fp32` | `.omx/research/pr95_mlx_stage8_c36_gpu_trace_refine_rgb_kahan_20260528Tlocal/` | 0.00702667236328125 | 0.0006695798365399241 | `rgb_0` |
| `blocks_kahan_fp32` | `.omx/research/pr95_mlx_stage8_c36_gpu_trace_blocks_kahan_20260528Tlocal/` | 0.000030517578125 | 0.0000038229563870118 | none |
| `blocks_refine_kahan_fp32` | `.omx/research/pr95_mlx_stage8_c36_gpu_trace_blocks_refine_kahan_20260528Tlocal/` | 0.000030517578125 | 0.000003824042778433068 | none |
| full `kahan_fp32` | `.omx/research/pr95_mlx_stage8_c36_gpu_trace_kahan_20260528Tlocal/` | 0.000030517578125 | 0.000003816443495452404 | none |

## Integration

The reusable control surface is now:

- global Conv2d accumulation mode;
- named per-module override presets;
- queue-owned CLI fields;
- plan and representation-manifest custody fields;
- forward-attestation and decoder-trace custody fields.

Next implementation target: make `blocks_kahan_fp32` the precision-calibration
candidate for PR95/HNeRV C36 MLX GPU parity, then profile the six upsample
blocks to decide whether a tighter subset or better fixed-order vectorization
can match the same drift band with lower wall-clock cost.
