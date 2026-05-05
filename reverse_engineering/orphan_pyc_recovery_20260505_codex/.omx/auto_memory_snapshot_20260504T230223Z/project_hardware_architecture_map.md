---
name: Hardware-Architecture Map for Task-Aware Compression
description: Different hardware targets need different optimal architectures. Paper contribution.
type: project
originSessionId: ecc348c1-2829-48d3-9d4c-10ed2aa188a8
---
## The insight (user + council)
The contest scores CPU and GPU submissions identically, but they represent
fundamentally different real-world deployment constraints. The optimal
architecture differs by hardware target:

### CPU-only (edge, IoT, comma four on-device)
- Best: Dilated postfilter h=64, INT8, 46KB
- Score: 1.27-1.33
- Inference: <30s for 1200 frames
- Deployment: OTA update, runs on any CPU

### GPU (cloud data pipeline, fleet training data)
- Best: Mask-conditioned renderer, FP4, ~200KB
- Score: 0.15-0.60 (projected)
- Inference: requires CUDA/MPS GPU
- Deployment: cloud decode before training data prep

### Apple Silicon (development, on-device Mac)
- Best: Same renderer, MPS-optimized (manual grid_sample)
- 11.3x faster than CPU fallback
- channels_last + FP16 autocast for additional speedup

### Snapdragon NPU (comma four production)
- Best: INT4 postfilter h=16, ~1.6KB
- Score: ~1.5-1.8 (estimated)
- Inference: <1ms per frame on Hexagon DSP
- Deployment: smallest possible model

## Paper contribution
A "hardware-aware compression architecture selection guide" that shows:
1. The Pareto frontier shifts based on hardware constraints
2. CPU solutions prioritize simplicity and universality
3. GPU solutions can use heavier models for better quality
4. The scoring formula doesn't account for this — same metric, different constraints

This is a unique analysis that no other submission will have.

**Why:** Positions our work as a systematic framework, not just a single solution.
**How to apply:** Present both CPU and GPU results with hardware context in the paper.
