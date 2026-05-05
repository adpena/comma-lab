---
name: MPS Extreme Optimization — 60-80% Training Speedup
description: Prince Canuma + council identified 6 optimizations for M5 Max. GT scorer cache is 40-50% alone.
type: project
originSessionId: ecc348c1-2829-48d3-9d4c-10ed2aa188a8
---
## P0: GT Scorer Output Cache (40-50% speedup, 2 hours)
GT frames never change. Scorers are frozen. Precompute GT scorer outputs once,
look up during training instead of re-running frozen scorers every iteration.
Memory: ~750MB for 600 pairs. Trivial on 128GB.
THIS IS THE SINGLE LARGEST OPTIMIZATION AVAILABLE.

## P1: MPS FP16 Autocast on Scorer Forward (20-40% on scorers, 30 min)
torch.autocast("mps", dtype=torch.float16) works since PyTorch 2.1+.
Currently only enabled for CUDA. Apply to frozen scorer forward passes.
Renderer stays FP32 for gradient quality.

## P2: channels_last Memory Format (15-25% on conv2d, 30 min)
model.to(memory_format=torch.channels_last) for all models.
MPS has optimized paths for NHWC. Free speedup.

## P3: Pre-stage Frames on MPS (5-10%, 15 min)
Move all frames to MPS once at startup. Unified memory = no actual copy.
Eliminates per-pair .to(device) allocation overhead.

## P4: Serialize GPU Processes (10-20%, 5 min)
MPS has single command queue. Multiple processes serialize with overhead.
Run 1 GPU process, keep CRF retrains CPU-only.

## P5: torch.compile Postfilter (5-10%, 15 min)
torch.compile(backend="aot_eager") fuses pointwise ops. Don't compile scorers.

## DO NOT pursue
- MLX migration (conv2d too slow, scorer porting impossible)
- Neural Engine (inference only, no gradients)
- Scorer distillation (approximation error kills score)
- Gradient checkpointing (128GB = memory is not bottleneck)

**Combined P0-P4: 60-80% total speedup. One afternoon of work.**
