---
name: Quantizr definitive binary analysis — NOT warp paradigm, simpler than we thought
description: Full archive reverse-engineering from PR#55. JointFrameGenerator (no warp/flow/gate). SharedMaskDecoder + two heads. Brotli on everything. GT poses (not optimized). 87K params. 5-stage freeze/unfreeze training.
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---

## Archive Structure (299,970 bytes total)

| File | Size | % | Content |
|------|------|---|---------|
| mask.obu.br | 219,472 | 73% | Brotli Q11 AV1 OBU, 600 frames, CRF 50, monochrome 512x384 |
| model.pt.br | 66,841 | 22% | Brotli-compressed FP4 PyTorch pickle (old serialization) |
| pose.npy.br | 13,185 | 4% | Brotli-compressed float32 NumPy (600,6) — GT poses, NOT optimized |

## Architecture: JointFrameGenerator (87,836 params)

**NOT AsymmetricPairGenerator.** No warp, no optical flow, no gate, no residual.

- **SharedMaskDecoder**: mask(5-class) + 2D coord grid → U-Net (c1=56, c2=64, one stride-2 down)
- **frame2_head** (static): 2x SepResBlock(56) → SepConv(56→52) → Conv(52→3) → sigmoid*255
- **frame1_head** (FiLM-conditioned): FiLMSepResBlock(56) + SepResBlock(56) → SepConv(56→52) → Conv(52→3) → sigmoid*255
- FiLM only on frame1_head.block1: pose_mlp(6→48→48) → gamma,beta for 56 channels
- All DSConv with depth_mult=1, GroupNorm(2 groups), SiLU everywhere
- Output 384x512, then F.interpolate(bilinear) to 874x1164

**Why:** frame2 is mask-determined (no pose needed). frame1 uses pose to encode the
ego-motion relationship. Both frames rendered from the SAME mask (half-frame paradigm).

## Param Breakdown
- FP4 conv weights: 75,904 (86.4%)
- FP16 FiLM projections: 5,488 (6.3%)
- FP16 pose_mlp: 2,688 (3.1%) — Linear(6,48)+SiLU+Linear(48,48)
- FP16 GroupNorm: 2,272 (2.6%)
- FP16 conv biases: 1,142 (1.3%)
- FP16 output heads: 312 (0.4%)
- FP16 embedding: 30 (0.0%) — Embedding(5,6)

## Training Pipeline (5 stages, ~1080 total epochs)

1. **ANCHOR** (400ep, lr=5e-4): Freeze frame1+pose, train shared+frame2. SegNet CE+KL(T=2.0), error_boost=9x. QAT after ep200.
2. **ANCHOR_BOOST** (80ep, lr=1e-5): Same freeze, error_boost=49x (extreme hard mining).
3. **FINETUNE** (320ep, lr=5e-5): Freeze shared+frame2, train frame1+pose. PoseNet MSE only.
4. **JOINT** (160ep, lr=1e-5): All unfrozen, combined loss, 30x pose_weight.
5. **MICRO** (120ep, lr=5e-6): Same as FINETUNE, lower LR polish.

Key: optimizer RESET at QAT transition. EMA=0.99. Cosine LR with warmup.

## Compression Tricks
- Brotli Q11 on ALL artifacts (model, masks, poses)
- AV1 OBU low-overhead bitstream (not MKV container — saves container overhead)
- PyTorch old-style serialization (_use_new_zipfile_serialization=False)
- Poses stored as float32 numpy (not fp16 — Brotli compresses better on float32)

## What He Said
- "sub 0.30 is possible just by sweeping through different conv dims"
- "further improvement could possibly be made with some neural mask encoding"
- PR#53 was his earlier "mask2mask" (0.60, 386KB, used optical flow + both masks)
- PR#55 DROPPED optical flow, added FiLM pose, half-frame masks

**How to apply:**
- Consider Brotli compression on our archive artifacts
- Consider JointFrameGenerator architecture (simpler than our warp paradigm)
- His 5-stage freeze/unfreeze is more surgical than our 3-phase approach
- GT poses work fine — he doesn't optimize them at all
- error_boost hard mining is critical for his SegNet quality
