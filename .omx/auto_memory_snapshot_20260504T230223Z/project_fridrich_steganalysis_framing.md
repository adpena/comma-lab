---
name: Fridrich Steganalysis Framing — Competition IS Inverse Steganalysis
description: The competition is steganalysis-in-reverse. Scorer = detector. Null-space = embedding space. Detection boundary = optimal operating point.
type: project
originSessionId: ecc348c1-2829-48d3-9d4c-10ed2aa188a8
---
## The Core Reframing (2026-04-11)

The comma video compression competition IS inverse steganalysis:
- **Steganalysis**: detect if an image was modified
- **Our competition**: generate frames the scorer can't distinguish from originals
- **Scorer = steganalysis detector** (PoseNet + SegNet measure modification)
- **Null-space projection = steganographic embedding** (hide rate savings in scorer blind spots)
- **Detection boundary = optimal operating point** (operate exactly where scorer transitions from "same" to "different")

## Reformulated Objective

Instead of: `minimize(100*seg + sqrt(10*pose) + 25*rate)`
Fridrich's formulation: `minimize(rate) subject to seg < threshold AND pose < threshold`

Below the detection boundary, there's FREE ROOM to optimize rate. Above it, you're wasting bits fighting the scorer.

## User's Connection to SynthID

User was thinking about Google DeepMind's SynthID watermarking (detectable only via steganalysis) and signal processing / video channels. This connects: SynthID embeds information in neural network null spaces — exactly what we're doing in reverse. User interested in the algorithm behind SynthID.

## Jessica Fridrich is co-leader alongside Yousfi

She founded the field. Yousfi extended her work to win competitions. She's now co-leading the grand council.

**Why:** This reframing changes how we optimize. Instead of minimizing a weighted sum, we find the detection boundary and operate on it. The constrained formulation is more natural and may converge faster.

**How to apply:** Implement Fridrich's detection-boundary formulation as an alternative to the weighted-sum loss. Use augmented Lagrangian to enforce the constraint form.
