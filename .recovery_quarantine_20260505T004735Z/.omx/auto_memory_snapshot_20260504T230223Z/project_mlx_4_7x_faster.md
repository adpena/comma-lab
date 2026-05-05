---
name: MLX is 4.7x Faster Than PyTorch MPS for Training
description: Benchmarked on M5 Max — MLX fwd+bwd 14ms vs PyTorch MPS 65ms for 2-conv 64ch 384x512
type: project
originSessionId: ecc348c1-2829-48d3-9d4c-10ed2aa188a8
---
## Benchmark Results (M5 Max, MLX 0.31.1, 2026-04-11)

| Framework | Forward only | Fwd + Backward |
|-----------|-------------|----------------|
| MLX | 3.0 ms | **14.0 ms** |
| PyTorch MPS | 2.4 ms | 65.5 ms |
| PyTorch MPS channels_last | 1.8 ms | 65.5 ms |

MLX forward is slightly slower than MPS, but MLX backward is MUCH faster.
Training (fwd+bwd) is 4.7x faster in MLX.

## Why
MLX's lazy evaluation + fused compute graph optimizes the full forward+backward
together. PyTorch MPS dispatches ops one at a time via Metal command buffers.

## Implications
- Phase 1 pre-training (L1 loss, no scorer): train in MLX, convert to PyTorch
- Phase 2 (scorer fine-tune): must stay in PyTorch (frozen scorers are PyTorch)
- CPU lane (postfilter training): could benefit too for scorer-free loss components
- The 4.7x speedup turns 5-hour Phase 1 into 1 hour

## Implementation path
1. Port MaskRenderer to MLX (NHWC layout, mlx.nn.Conv2d)
2. Train Phase 1 in MLX with L1 loss
3. Convert MLX weights → PyTorch (transpose NHWC→NCHW per layer)
4. Continue Phase 2 in PyTorch with scorer loss

## What the council got wrong
Prince Canuma said "MLX conv2d is 10-150x slower." That was forward-only benchmarks
from 2024. On M5 Max in 2026 with MLX 0.31.1, training (fwd+bwd) is 4.7x FASTER.
Always benchmark YOUR workload on YOUR hardware.

**Why:** 4.7x training speedup is transformative for our timeline.
**How to apply:** Port renderer to MLX for Phase 1 pre-training.
