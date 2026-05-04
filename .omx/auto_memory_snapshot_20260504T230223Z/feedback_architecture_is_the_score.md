---
name: Architecture IS the Score
description: The fundamental insight from Quantizr analysis — pair-wise generation architecture determines PoseNet score, not training tricks or loss engineering
type: feedback
originSessionId: ecc348c1-2829-48d3-9d4c-10ed2aa188a8
---
The architecture IS the score. Pair-wise generation gives near-zero PoseNet by construction. Training tricks and loss engineering are secondary.

**Why:** Quantizr's 0.60 comes from `AsymmetricPairGenerator(mask1, mask2) → (frame1, frame2)` — both frames from one forward pass. PoseNet distortion 0.00066 is architectural, not trained. Our coupled trajectory loss is "a crutch for an architecture that should have pair-wise generation built in" (Quantizr adversarial voice).

**How to apply:**
- When choosing between architectural changes and training tricks, architecture wins
- The scorer is volatile — small optimizations interact unpredictably
- The 386KB submission proves the concept; sub-0.50 comes from the right architecture, not more epochs
- Quantizr's submission contains the fundamental decisions; small tricks are withheld or unstacked
- Focus engineering effort on the architecture (pair-wise, resolution chain, conditioning) not the optimizer
