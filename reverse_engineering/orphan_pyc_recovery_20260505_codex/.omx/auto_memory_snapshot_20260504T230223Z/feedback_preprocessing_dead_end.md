---
name: Preprocessing is a Dead End for PoseNet
description: ALL preprocessing approaches (blur, chroma-only, gentle) kill PoseNet by 90-105%. Only encoder params are safe.
type: feedback
---

Do NOT propose preprocessing-based approaches for this video compression challenge. Every variant we tested destroyed PoseNet:
- Full blur outside corridor: +104% PoseNet (score 2.52)
- Gentle blur (σ=0.8, blend=0.25): +90% PoseNet (score 2.47)
- Chroma-only degradation: +105% PoseNet (score 2.51)

PoseNet uses the ENTIRE frame including color information. ANY pixel modification hurts it.

**Why:** PoseNet estimates camera pose from all geometric and color cues in the frame — perspective lines, color gradients, horizon, building edges, road texture, sky color. Degrading any region destroys cues it relies on.

**How to apply:** 
- Focus on encoder-side parameter tuning (sharpness, CRF, codec params) which preserves the video signal
- The public leader's ROI approach must be doing something fundamentally different from what we tested, or their scorer weights things differently
- Do not spend more time on blur/chroma/preprocessing experiments unless the scoring function changes
