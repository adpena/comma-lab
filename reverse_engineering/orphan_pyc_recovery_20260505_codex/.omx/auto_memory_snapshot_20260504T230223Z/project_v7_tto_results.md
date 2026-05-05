---
name: v7 TTO Results — Auth 0.37 [UNLIMITED-COMPUTE], Best Hinge Loss Run
description: v7 TTO with 500 steps hinge loss on all 600 pairs. Auth 0.37. Proxy 0.195. Teacher frames for distillation.
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## v7 TTO Results (2026-04-20, Vast.ai RTX 4090)

**Auth: 0.37 [UNLIMITED-COMPUTE]** — PoseNet=0.00250, SegNet=0.00094, Rate=0.00489
**Proxy: 0.195** — auth/proxy ratio ~1.9x

### Config
- 500 steps, hinge loss (margin=0.5), embedding loss (512D)
- seg_weight=100, pose_weight=10, compress_weight=0.5
- seg_odd_only=True, simulate_resize=True
- Constant LR 0.005, early_stop_patience=500
- Checkpoint: v5_lagrangian_renderer/renderer_best.pt (MD5: cff8dca4)

### Timing
- TTO: 2409s (40 min) on RTX 4090
- Eval prep (upscale + write .raw): 40s
- evaluate.py (DALI + CUDA): 47s
- Total eval: 87s — well within 30-min budget

### Gap to Quantizr (0.33)
| Metric | v7 TTO | Quantizr | Gap |
|--------|--------|----------|-----|
| PoseNet | 0.00250 | 0.00051 | 4.9x |
| SegNet | 0.00094 | 0.00061 | 1.5x |
| Rate | 0.00489 | 0.00799 | Better |

PoseNet is the main gap — FiLM on pose should close most of this.

### Teacher Frames
- `experiments/results/tto_v7_hinge_500/tto_frames.pt` (708MB, 1200 frames)
- Used as distillation teacher for contest-compliant renderer training
- GT poses extracted: `experiments/results/gt_poses.pt` (8.7KB, 600 × 6D fp16)
