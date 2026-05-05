---
name: Competitive Intelligence — neural_inflate PR #49
description: Competitor at 1.89 uses PixelUnshuffle/Shuffle REN architecture, achieves seg=0.00434 (much better than our 0.00576) but pose=0.071 (much worse than our 0.033)
type: project
---

PR #49 `neural_inflation` merged 2026-04-09T02:46. Score: 1.89.

**Their approach (from train_ren.py):**
- Architecture: REN with PixelUnshuffle(2) → 4 conv layers at half-res → PixelShuffle(2)
- h=32 features, 4 layers (12→32→32→32→12 at half-res)
- Zero-initialized output layer (same as us)
- Direct DistortionNet training (PoseNet + SegNet loss)
- Consecutive frame pairs for temporal coherence
- GPU required for inflation
- Archive: 917KB (53KB larger than ours)

**Key metrics:**
- PoseNet: 0.07148 (ours: 0.03317 — we're 2.15× better)
- SegNet: 0.00434 (ours: 0.00576 — they're 24% better!)
- Rate: 0.02443 (ours: 0.02302)

**Why their SegNet is better:**
The PixelShuffle trick processes at half-resolution, which aligns each conv's
RF with SegNet's internal processing resolution (512×384 → each pixel covers
roughly what SegNet sees). Our full-resolution filter wastes corrections on
spatial frequencies that SegNet downsamples away.

**What to steal:**
1. PixelUnshuffle/Shuffle architecture — likely the #1 reason for their SegNet win
2. 4 layers at half-res — matches both PoseNet and SegNet RF scales
3. If we combine their arch + our training (QAT+EMA+saliency+best-checkpoint):
   Projected: seg=0.004 + pose=0.033 + rate=0.023 → score ≈ 1.58

**Stealth note:** We lead them by 0.163 (1.727 vs 1.89). We should NOT submit
until we've extracted all the value from their PixelShuffle insight. Their
technique is public; ours isn't. Use their architecture, stack our training.
