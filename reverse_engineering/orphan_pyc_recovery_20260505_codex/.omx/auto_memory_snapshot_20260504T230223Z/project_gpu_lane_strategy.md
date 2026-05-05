---
name: GPU Lane Strategy — Mask-Conditioned Renderer
description: Two-lane competition strategy after mask2mask (0.60). CPU lane + GPU lane + best writeup.
type: project
originSessionId: ecc348c1-2829-48d3-9d4c-10ed2aa188a8
---
## The mask2mask approach (reverse engineered)
- Paradigm: Segment → Compress Masks → Neural Render
- Archive: mask.mp4 (227KB AV1 masks) + model.pt (195KB FP4) + arch.br (5KB)
- Architecture: U-Net renderer (36→60→36), flow warper, 308K params in FP4
- SegNet near-perfect because input IS the segmentation mask
- Score: 0.60 (self-reported, not yet eval-verified by organizers)

## Two-lane strategy (council approved)
**CPU Lane**: dilated h=64 + CRF 35/36 retrain → target 1.20-1.30
**GPU Lane**: mask-conditioned SPADE renderer → target sub-1.0

## Key architecture decisions for GPU lane
- SPADE generator (mask-conditioned, spatially-adaptive normalization)
- ~300K-4M params depending on quality target
- FP4 quantization with 8-value codebook (adopted from mask2mask)
- Mask video at 512×384 (scorer resolution), AV1 encoded
- Flow-based warping for temporal consistency (PoseNet pairs)

## Prize strategy (Arrow)
- Score prize: mask2mask likely wins if verified. Aim for 2nd/3rd ($250-500)
- Writeup prize ($1000): STRONGLY in play. Our paper is the best.
- Total expected: $1250-1500

## Timeline
- Days 1-7: Build + train GPU lane. Continue CRF retrains.
- Day 7: Gate — if GPU proxy > 1.5, reduce investment
- Day 10: Gate — if GPU proxy > 1.0, shift to writeup
- Days 14-21: Polish writeup, submit both lanes

## Critical insight (Collier)
mask2mask VALIDATES our thesis. Our paper maps the full journey from codec
tweaks to neural conditional generation. mask2mask is the endpoint we predicted.
We don't need to beat it to write the best paper about it.

**Why:** Existential competitive shift requires two-lane response.
**How to apply:** Build GPU lane in parallel with CPU lane. Writeup is the primary prize target.
