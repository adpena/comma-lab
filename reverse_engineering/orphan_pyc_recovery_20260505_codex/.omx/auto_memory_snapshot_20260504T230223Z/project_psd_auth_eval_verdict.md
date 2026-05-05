---
name: PSD Auth Eval Verdict — Stay With Dilated
description: PSD auth 1.49 at ep809. SegNet 12.8% better but PoseNet 5x worse. Council unanimous: don't commit to PSD.
type: project
originSessionId: ecc348c1-2829-48d3-9d4c-10ed2aa188a8
---
## Auth Result (2026-04-11)
PSD h=64, ep 809: auth score 1.49 (pose=0.01108, seg=0.00532, rate=0.02522)
Proxy-auth distortion gap: <0.01 (excellent transfer, NOT like KL distill)
The 0.62 apparent gap was entirely rate penalty from 95KB model.

## Council Verdict: STAY WITH DILATED
- CRF 36 on dilated (proven, zero risk): projected 1.28
- Dilated long2500 (already running): projected 1.30
- PSD h48 smoke test (bounded side lane): 400 epochs max

## Kill List
- Do NOT deploy PSD h64 to Modal
- Do NOT pursue PSD h32 (capacity too low)
- Do NOT pursue pruning/distillation of PSD (complexity without need)

## The PSD Insight (durable, for the paper)
PixelUnshuffle alignment with scorer resolution genuinely helps SegNet (+12.8%).
This is the first architecture to improve SegNet below baseline.
Worth documenting but not worth betting the competition on.

## Projected Best Path
Dilated h64 + long training + CRF 36: score **1.23-1.25**

**Why:** 22 days left, PSD needs 1000+ more epochs with uncertain PoseNet convergence.
**How to apply:** Priority 1 is CRF sweep on proven dilated checkpoint. No architecture changes.
