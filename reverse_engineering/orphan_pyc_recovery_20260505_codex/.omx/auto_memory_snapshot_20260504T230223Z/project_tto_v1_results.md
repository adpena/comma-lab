---
name: TTO v1 Results — Proxy Score 0.5896 (baseline 0.6296)
description: First full 1200-frame TTO result. +0.04 improvement. Rank-6 gradient bottleneck confirmed.
type: project
originSessionId: ecc348c1-2829-48d3-9d4c-10ed2aa188a8
---
## TTO v1 Results (2026-04-14)
- Proxy baseline: 0.6296 (PoseNet 0.0172, SegNet 0.00215)
- Proxy TTO: 0.5896 (PoseNet 0.0158, SegNet 0.00193)
- Improvement: +0.04 (6.3%)
- PoseNet improvement: 8.5% (0.0172→0.0158)
- SegNet improvement: 10.2% (0.00215→0.00193)
- Auth eval: NOT RUN (NameError crashed post-processing, tto_frames.pt saved)
- Config: lr=0.005, seg=100, pose=10, compress=0.5, patience=150
- Timing: 7806s TTO (130 min), 60 batches × ~130s average

## Key Finding
Rank-6 gradient bottleneck confirmed. PoseNet output MSE only provides 6 gradient
directions in pixel space. Most batches early-stop at step 151. The optimizer
exhausts all 6 directions in ~50 steps then stalls.

## Next: TTO v3 with Embedding Loss
Expected to break through 0.012 PoseNet floor by providing rank-256 gradient.
Projected score with embedding loss: 0.46 (if PoseNet reaches 0.002).
