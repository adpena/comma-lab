# Current Focus -- 2026-04-15T07:00:00Z

## Scores
- **Renderer baseline**: auth=0.87 (seg=0.21, pose=0.56, rate=0.10)
- **TTO v3 (embedding loss)**: auth=0.74 (seg=0.21, pose=0.43, rate=0.10) -- DEAD
  - Embedding loss barely helped PoseNet (0.0172 -> 0.0173 WORSE proxy)
  - Config: lr=0.01, seg=10, pose=50, compress=0, embed_loss=True, seg_odd_only=True
- **TTO v4 (RUNNING on Modal)**: v1 config + seg_odd_only + antialias
  - Config: lr=0.005, seg=100, pose=10, compress=0.5, seg_odd_only=True, antialias=0.2
  - Expected: proxy ~0.59 (v1 proved 6.3% improvement), auth TBD
- **Joint Pair Generator v1 (RUNNING on Modal)**: 576K param Y-shaped U-Net
  - 5000 epochs, base_ch=48, T4
- **Target**: sub-0.50 auth

## Sub-0.5 Score Budget
```
Score = 100*seg + sqrt(10*pose) + 25*rate < 0.50
  SegNet:  < 0.15 -> seg_dist < 0.0015
  PoseNet: < 0.25 -> pose_dist < 0.00625
  Rate:    < 0.10 -> archive < 150KB
```

## Active Experiments

### TTO v4 (Modal T4, RUNNING)
- App: ap-U9fxnDuw13vda1ejuBHRTC
- Improvements over v3:
  1. Back to v1 output MSE loss (proven 6.3% improvement)
  2. seg_odd_only=True (frees even frames for PoseNet)
  3. antialias_weight=0.2 (penalizes PoseNet-invisible sub-2x2 noise)
- Archive compression: ZIP_DEFLATED level 9 (saves ~20% on renderer.bin)

### Joint Pair Generator v1 (Modal T4, RUNNING)
- App: ap-8kcsjA0IDtrWJEpNqvyUcL
- Y-shaped U-Net, mask-conditioned pair generation
- Backup path if TTO doesn't reach sub-0.50

### Ensemble Pipeline (READY)
- src/tac/ensemble.py: per-pair best-of-N selection
- Combine TTO v4 + joint pair results when both available

## Code Changes This Session
1. antialias regularization in constrained_gen.py + renderer_tto.py + modal deploy
2. Archive compression: ZIP_STORED -> ZIP_DEFLATED (-20% archive size)
3. renderer_tto.py: deduplicated 210 lines via tac.scorer imports

## CRITICAL BUG FOUND: TTO PoseNet Gradients Were ZERO (2026-04-15)

**Root cause**: Upstream `frame_utils.py:rgb_to_yuv6()` is decorated with `@torch.no_grad()`.
PoseNet's `preprocess_input()` calls this function. The training pipeline had a workaround
(patching the scorer code path), but the TTO pipeline loaded scorers through a different
code path that never received the fix. Result: every TTO run in the entire project history
optimized with ZERO PoseNet gradients. The optimizer was completely blind to PoseNet.

**Discovery**: Skunkworks council adversarial review. The Contrarian demanded: "if 50 steps
make PoseNet WORSE, something is fundamentally wrong." Hotz traced the call chain from
`preprocess_input()` through `rgb_to_yuv6()` to the `@torch.no_grad()` decorator. 13-0
unanimous vote to fix immediately.

**Impact**: All TTO scores (v1 through v4) were running with only SegNet+rate gradients.
PoseNet improvements were pure noise. The auth=0.87 renderer baseline is unaffected (training
pipeline had the fix). But TTO on top of the renderer — our primary path to sub-0.50 — was
fundamentally broken.

**Projected score with fix**: Renderer auth=0.87 has PoseNet=0.031 (35% of total score).
TTO with actual PoseNet gradients could reduce PoseNet by 5-10x (based on SegNet's proven
TTO response). Projected auth: 0.87 -> ~0.35.

**Lesson**: A single `@torch.no_grad()` decorator in a dependency silently invalidated an
entire optimization pipeline. The bug was invisible because: (1) TTO still reduced SegNet
(masking the PoseNet failure), (2) proxy scores improved (SegNet-dominated), (3) no gradient
norm monitoring was in place for individual scorer components.

## Key Insights
1. Embedding loss is DEAD (v3 proved PoseNet got WORSE with embedding MSE)
2. seg_odd_only already implemented and tested
3. PoseNet operates on 2x-downsampled YUV -> sub-2x2 noise is invisible
4. Archive already at 4-bit quantization (150KB), DEFLATE saves ~30KB more
5. Rate contribution: 0.10 -> 0.08 with DEFLATE (saves 0.02 points)
6. **TTO PoseNet gradients were ZERO** -- all TTO experiments were SegNet-only optimizations
7. Fix: ensure `rgb_to_yuv6()` runs WITHOUT `@torch.no_grad()` in TTO scorer loading path

## Deadline
- May 3, 2026 (~18 days remaining)
