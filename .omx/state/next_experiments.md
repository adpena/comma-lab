# Next Experiments -- 2026-04-15

## Current Best: auth=0.43 (TTO v5a, Lagrangian fixed, PoseNet gradients working)
## Renderer Baseline: auth=0.87
## Target: sub-0.20

## Active Runs (both on Modal T4)

### TTO v5b (RUNNING)
- App: ap-24iBdzVH05oWpQ73kl2OL3
- Config: embedding loss (512D) + seg_odd_only, seg=100, pose=10, compress=0.5
- Purpose: first VALID embedding loss test (v3 had zero gradients)
- Pre-registered criteria:
  - Success: proxy < 0.50
  - Kill: proxy > 0.55

### TTO v5c (RUNNING)
- App: ap-VRS7zqEOIm0jAEGGuqCqQE
- Config: extreme PoseNet focus, pose=100, seg=1, compress=0.01, steps=1000
- Purpose: find PoseNet optimization CEILING
- Pre-registered criteria:
  - Success: pose_dist < 0.005
  - Kill: pose_dist > 0.015

## Queue (after active runs complete)

### Priority 1: Weight Tuning Based on v5b/v5c Results
1. **TTO v5d**: Optimal weight balance from v5b/v5c Pareto analysis
   - If v5c shows PoseNet can go much lower, find the sweet spot
   - Config: interpolate between v5a and v5c weights
2. **TTO v5e**: Longer TTO (2000 steps) with best-found weights
   - For pairs that haven't converged at 500-1000 steps

### Priority 2: Sensitivity-Guided TTO
3. **Masked TTO**: Use PoseNet sensitivity map to constrain optimization
   - Only modify PoseNet-sensitive pixels
   - Preserve SegNet-friendly pixels unchanged
   - Requires: PoseNet sensitivity map (running locally now)

### Priority 3: Ensemble & Archive
4. **Best-of-N ensemble**: Per-pair selection across v5a/v5b/v5c
   - Guaranteed >= best individual approach
5. **Archive LZMA**: Switch from ZIP_STORED to LZMA for ~25% size reduction
   - Free rate improvement, no quality loss

### Priority 4: Architecture
6. **Per-pair Lagrangian**: Auto-tune seg/pose weights per batch
   - Some pairs may benefit from more SegNet focus, others more PoseNet
7. **Multi-pass TTO**: Two TTO passes -- SegNet-focused then PoseNet-focused
   - Avoids gradient conflict between SegNet and PoseNet objectives

## Decision Gates
- 2026-04-15: v5b/v5c results -> determine optimal weight balance
- 2026-04-17: Sensitivity map + masked TTO experiment
- 2026-04-19: Ensemble from all v5 variants -> best submission candidate
- 2026-04-21: Lock final approach, begin full-scale TTO
- 2026-05-03: DEADLINE

## Battle Plan to Sub-0.2
```
Step 1: v5b/v5c -> find Pareto frontier (IN PROGRESS)
Step 2: v5d -> optimal weight point on frontier
Step 3: Sensitivity-guided TTO -> targeted pixel optimization
Step 4: Ensemble across all variants -> per-pair best selection
Step 5: Archive LZMA compression -> free rate savings
Step 6: Longer TTO (2000+ steps) -> squeeze remaining gains

Each step: auth eval after every experiment. No blind optimization.
```

## Killed Techniques (NEVER retry)
- KL distill loss mode
- Adaptive rebalance weights
- PoseNet gradient caps
- SegNet loss weight > 100
- PSD architecture
- Any TTO without rgb_to_yuv6 patch (pre-v5a: all had zero PoseNet gradients)
