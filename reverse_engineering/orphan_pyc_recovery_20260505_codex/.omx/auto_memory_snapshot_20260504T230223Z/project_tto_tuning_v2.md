---
name: TTO Tuning v2 — Council-Approved Config Changes
description: Current TTO config is suboptimal — compress fights PoseNet, lr too low, seg/pose ratio wrong, early stop too aggressive
type: project
originSessionId: ecc348c1-2829-48d3-9d4c-10ed2aa188a8
---
## Current TTO Config (v1 — running now, expected auth ~0.80)
- lr=0.005, seg_weight=100, pose_weight=10, compress_weight=0.5 (annealing)
- early_stop_patience=150 (hardcoded in coupled_trajectory_optimize)
- Average PoseNet ~0.012 across 50+ batches (baseline 0.017)

## Proposed TTO Config (v2 — launch IMMEDIATELY after v1 completes)
- lr=0.01 (2x higher — most batches plateau at step 50, need more gradient)
- seg_weight=10 (SegNet already 99.8%, don't waste gradient)
- pose_weight=50 (PoseNet is the bottleneck, 5x current)
- compress_weight=0.0 (renderer frames already compressible, fights PoseNet)
- early_stop_patience=300 (current kills at 151, may miss late improvement)

## Code change needed
Thread early_stop_patience from run_batched_tto CLI through to coupled_trajectory_optimize.
Currently hardcoded at 150 in constrained_gen.py.

## CLI for v2 launch
```
--tto-lr 0.01 --seg-weight 10 --pose-weight 50 --compress-weight 0.0
```

## Why these changes
1. compress_weight=0.5 annealing was designed for from-noise, not warm-start
2. lr=0.005 causes early plateau (PoseNet peaks at ~step 50, stalls for 100)
3. 100:10 seg:pose ratio wastes 90% of gradient on solved SegNet
4. patience=150 kills runs that might improve at step 200-300
