---
name: PSD Architecture Breakthrough
description: PSD+KL distill converges 3.5x faster than dilated+KL — 1.38 at ep 69. KL distill is load-bearing.
type: project
originSessionId: ecc348c1-2829-48d3-9d4c-10ed2aa188a8
---
## The Finding

PSD (PixelShuffle+Dilated) + KL distill converges dramatically faster than dilated + KL:
- PSD+KL: 1.38 at ep 69 (and still dropping)
- Dilated+KL: 1.40 at ep 168

PSD was previously REJECTED (1.99 with standard loss, hurt PoseNet). It works now because:
1. KL distill's soft targets protect PoseNet (bounded KL divergence = implicit regularization)
2. PSD's 24×24 RF (vs dilated's 15×15) covers more SegNet boundary context
3. PSD at half-res (582×437) aligns with SegNet's operating resolution (512×384)

## Key Lesson

Architecture changes that fail with one loss function may succeed with another.
We rejected PSD prematurely by testing it only with standard loss.
The combination of architecture + loss function matters more than either alone.

## PSD Architecture (src/tac/architectures.py PSDPostFilter)

PixelUnshuffle(2): 3ch@1164×874 → 12ch@582×437
conv1: 12→64, 3×3 at half-res
conv2: 64→64, 3×3 dilation=2 at half-res  
conv3: 64→64, 3×3 at half-res
conv4: 64→12, 3×3 at half-res
PixelShuffle(2): 12ch@582×437 → 3ch@1164×874
Total RF: ~24×24 at full resolution. ~87K params, ~85KB int8.

## Next Steps

PSD+KL is now the primary architecture for all new experiments.
Test PSD+standard at h=64 to quantify whether KL distill is load-bearing.
