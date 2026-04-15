# Current Focus -- 2026-04-15T14:30:00Z

## Scores
- **Renderer baseline**: auth=0.87 (seg=0.21, pose=0.56, rate=0.10)
- **TTO v5a (Lagrangian fixed)**: auth=0.43 (first valid TTO with PoseNet gradients)
- **TTO v3 (embedding loss)**: DEAD -- embedding loss made PoseNet WORSE
- **TTO v5b (embedding loss, RUNNING on Modal)**: app=ap-24iBdzVH05oWpQ73kl2OL3
  - Config: lr=0.005, seg=100, pose=10, compress=0.5, embed_loss=True, seg_odd_only=True, patience=150
  - Purpose: first VALID test of 512D embedding loss (v3 had zero gradients)
- **TTO v5c (aggressive PoseNet, RUNNING on Modal)**: app=ap-VRS7zqEOIm0jAEGGuqCqQE
  - Config: lr=0.005, seg=1, pose=100, compress=0.01, seg_odd_only=True, patience=300, steps=1000
  - Purpose: find PoseNet optimization CEILING -- how low can pose_dist go if we sacrifice SegNet?
- **Target**: sub-0.20 auth

## Score Decomposition (auth=0.43, v5a)
```
Score = 100*seg + sqrt(10*pose) + 25*rate
  auth=0.43 -> significant improvement from renderer baseline 0.87
  Key: PoseNet gradients now WORKING (fix validated)
```

## Sub-0.2 Score Budget
```
Score = 100*seg + sqrt(10*pose) + 25*rate < 0.20
  SegNet:  < 0.08 -> seg_dist < 0.0008
  PoseNet: < 0.08 -> pose_dist < 0.00064
  Rate:    < 0.04 -> archive < ~60KB
```

## Active Experiments

### TTO v5b (Modal T4, RUNNING)
- App: ap-24iBdzVH05oWpQ73kl2OL3
- Config: embedding loss (512D) + seg_odd_only + standard weights
- Question: does embedding loss help NOW that gradients actually flow?
- Pre-registered criteria:
  - Success: proxy < 0.50 (improvement over v5a)
  - Kill: proxy > 0.55 (no improvement)

### TTO v5c (Modal T4, RUNNING)
- App: ap-VRS7zqEOIm0jAEGGuqCqQE
- Config: pose_weight=100, seg_weight=1 (extreme PoseNet focus)
- Question: what is the PoseNet optimization FLOOR when we don't care about SegNet?
- Pre-registered criteria:
  - Success: pose_dist < 0.005 (significant reduction)
  - Kill: pose_dist > 0.015 (no improvement over v5a)

## Archive Compression Analysis
- Current: 150KB (ZIP_STORED), 4-bit quantized, 285K params
- LZMA: 113KB (-25%) -- easy win, no quality loss
- DEFLATE: 119KB (-21%) -- simpler, nearly as good
- 3-bit: ~112KB (quality risk, needs testing)
- 2-bit: ~76KB (likely too destructive)
- Rate impact: negligible (rate is ~10% of total score, compression saves <0.02 points)
- Verdict: apply DEFLATE/LZMA as free improvement, but PoseNet/SegNet dominate score

## PoseNet Sensitivity Map
- Running locally on MPS with 20 frames (quick test)
- Previous maps were ALL ZERO (gradient bug)
- This is the first valid sensitivity map with real gradients
- Will reveal which pixels PoseNet is sensitive to vs insensitive to

## CRITICAL BUG FOUND: TTO PoseNet Gradients Were ZERO (2026-04-15)

**Root cause**: Upstream `frame_utils.py:rgb_to_yuv6()` is decorated with `@torch.no_grad()`.
PoseNet's `preprocess_input()` calls this function. The training pipeline had a workaround
(patching the scorer code path), but the TTO pipeline loaded scorers through a different
code path that never received the fix. Result: every TTO run in the entire project history
optimized with ZERO PoseNet gradients. The optimizer was completely blind to PoseNet.

**Status**: FIXED in v5a. auth=0.43 confirms fix is working (50% improvement over renderer 0.87).

## Battle Plan to Sub-0.2

1. **v5b/v5c results** (in progress): determine optimal weight balance for PoseNet vs SegNet
2. **Longer TTO**: increase steps from 500 to 2000+ for pairs that haven't converged
3. **Per-pair adaptive weights**: Lagrangian-style automatic balancing per batch
4. **Archive compression**: LZMA compression for free rate savings
5. **Sensitivity-guided TTO**: only optimize PoseNet-sensitive pixels, leave SegNet-safe areas alone
6. **Ensemble**: best-of-N selection across v5a/v5b/v5c per pair

## Key Insights
1. PoseNet gradients fix was the single biggest breakthrough -- auth 0.87 -> 0.43
2. Embedding loss needs re-evaluation now that gradients work (v3 was invalid)
3. Archive compression is a free win but small impact on score
4. PoseNet sensitivity map will enable targeted optimization
5. Volume commits now every batch for real-time monitoring

## Deadline
- May 3, 2026 (~18 days remaining)
