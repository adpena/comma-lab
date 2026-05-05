---
name: Council Shower Thoughts Session (2026-04-10)
description: Deep creative review — saliency inversion, per-channel quant, TTO, hard-frame curriculum, contrarian thesis
type: project
originSessionId: ecc348c1-2829-48d3-9d4c-10ed2aa188a8
---
Panel: Hinton (KL distill), Karpathy (architecture), quantization engineer (LSQ/per-channel),
AV1 compression veteran (rate/grain), contrarian (hard-frame upsampling).

Key findings:

1. **Saliency is fighting SegNet** — PoseNet-only alpha=20 constrains boundary pixels.
   Fix: dual saliency alpha_seg=5000.

2. **Per-channel quant is free** — recovers ~3-4 bits per multi-channel conv. Implemented.

3. **KL distill schedule**: T=5 (ep 0-200) → T=2 (200-800) → T=1 (800-2500). Stepwise,
   not linear. The phase transition at T~1.5 concentrates gradients at true boundaries.

4. **DualHead architecture**: 1x1 seg_head (sharp) + 3x3 pose_head (smooth). Same params.

5. **Test-time optimization**: 5 Adam steps per frame at inflate time, optimizing pixels
   against frozen scorer. Highest-variance, highest-EV. ~5 min inflate budget available.

6. **Contrarian**: Train on hard frames, not uniformly. SegNet only changes at boundary
   pixels (~5% of image). Oversample worst 20% pairs by SegNet disagreement.

7. **Multi-pass inflate**: Run CNN twice. Second pass corrects residual errors. Free
   within inflate budget. Train with chained loss for best results.

**Why:** The panel identified that we're optimizing a nearly-saturated PoseNet while
SegNet has 590x more marginal leverage and our training actively fights it.

**How to apply:** Tier 1 (dual sal, per-channel, hard-frame) first. Tier 2 (KL distill,
DualHead, TTO) after. Score target: 1.15-1.30 with Tier 1, potentially sub-1.0 with all.
