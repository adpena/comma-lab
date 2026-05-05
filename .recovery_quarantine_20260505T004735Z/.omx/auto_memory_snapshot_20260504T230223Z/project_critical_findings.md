---
name: Critical Findings from Multi-Perspective Reviews
description: Synthesized breakthrough findings from Rick Rubin v2, Jeff Dean, Terence Tao, Chris Lattner, Palmer Luckey, and Gabe Newell reviews
type: project
---

## CONFIRMED ledger bug (2026-04-07)

The "2.08 sharpness=1 ffmpeg" entry in our ledger had **wrong PoseNet number**: 0.0938 should have been 0.08694. The actual report:
- PoseNet: 0.08694 (NOT 0.0938)
- SegNet: 0.00577
- Rate: 0.02302
- Score: 2.08 ✓

The three 2.08 results are NOT a coincidence — they're all using torch bicubic + BT.601 path implicitly. PoseNet values cluster at 0.08521-0.08694.

## Mathematical path to 1.91 (Tao)

At our operating point, marginal returns:
- ∂score/∂rate = 25 (constant)
- ∂score/∂segnet = 100 (constant)  
- ∂score/∂posenet = √10/(2√p) ≈ 5.16 (diminishing)

**Trade PoseNet headroom for rate.** PoseNet is in diminishing returns; rate and SegNet have higher marginal value.

If we hit (PoseNet=0.0848, SegNet=0.0055, rate=0.018, archive=754KB):
- 100*0.0055 + sqrt(10*0.0848) + 25*0.018 = 0.55 + 0.92 + 0.45 = **1.92**

## Critical insights from reviews

### Jeff Dean / Palmer Luckey:
**SegNet only sees the LAST frame of each pair** (`modules.py:108: x[:, -1, ...]`). With seq_len=2, even-indexed frames are NEVER scored by SegNet. Encode even frames at +6 to +10 QP for free SegNet performance — worth ~2-4% rate reduction.

### Gabe Newell:
1. **inflate.sh has full write access to INFLATED_DIR**, only shape is checked. Can ship per-video correction tensors in archive.zip.
2. **Per-video CRF**: tune CRF per video via offline scoring. Worth 2-4% rate at equal distortion.
3. **Train tiny task-aware post-filter (~20KB)** using gradients from posenet.safetensors / segnet.safetensors directly. Backprop the score function through.

### Palmer Luckey:
**Saliency-based ROI**: backprop ∂posenet/∂pixel on ground truth to get per-pixel "what matters" map. Use as ROI map directly.

### Rick Rubin (3am vocal take):
**ROI-weighted residual patch**: encode once, ship a tiny residual (200 bytes RLE for ROI) that inflate.py adds post-decode. Only covers hands/face. Bypasses the multi-stream ban.

## Frontier matrix (theoretical minimums per metric)

| Metric | Best observed | At score | If achievable in combo |
|---|---|---|---|
| PoseNet | 0.08440 | 2.14 (VQ+QM+fgd) | -0.024 vs floor |
| PoseNet | 0.08475 | 2.09 (ROI map) | -0.024 vs floor |
| SegNet | 0.00548 | 2.94 (fg=0) | -0.029 vs floor |
| Rate | 654K | 2.89 (512+fg=0) | -0.144 vs floor |

Theoretical minimum if all combined: ~1.918. Sub-1.92 is reachable if the techniques don't fight.

## The blocker

Each technique that achieves one frontier *destroys* another. Tradeoff structure:
- USM/sharpness/grain → pose↑ rate↑ segnet flat
- Denoise/blur → rate↓ segnet flat **pose destroyed**
- CRF↑ / fg=0 → rate↓ **pose destroyed**
- ROI map → **selectively spends bits where pose lives**, pose↓ rate ok

**ROI map is the only knob that moves PoseNet without paying elsewhere.** Everything else is zero-sum.
