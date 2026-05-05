---
name: Mask CRF Pareto frontier — CRF56 is optimal (280KB, 4.1x penalty)
description: Full AV1 CRF sweep at 384x512 with upstream DistortionNet scoring. CRF56=280KB is sweet spot. CRF63 cliff at 11.4x penalty. Train with CRF-matched masks.
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## Mask CRF Pareto Frontier (2026-04-21, 30 pairs, upstream scorer)

| CRF | Size | pose_d | seg_d | Distortion | Penalty vs lossless |
|-----|------|--------|-------|------------|---------------------|
| 50 | 411KB | 0.195 | 0.003 | 1.67 | 3.5x |
| 54 | 326KB | 0.287 | 0.003 | 2.01 | 4.2x |
| **56** | **280KB** | **0.263** | **0.004** | **1.98** | **4.1x** |
| 58 | 239KB | 0.429 | 0.004 | 2.47 | 5.1x |
| 60 | 194KB | 0.796 | 0.005 | 3.28 | 6.8x |
| 63 | 106KB | 2.288 | 0.007 | 5.45 | 11.4x |

## Key findings
1. PoseNet is EXTREMELY sensitive to mask boundary errors
2. The cliff is between CRF58 and CRF60 (accuracy 99.6% → 99.5%)
3. CRF56 is the sweet spot: 280KB, 4.1x penalty, best rate/quality tradeoff
4. Train renderer WITH CRF-matched masks to reduce the penalty

## Archive math with CRF56
- FP4 renderer (60KB) + CRF56 masks (280KB) + fp16 poses (8KB) = 348KB
- Rate = 25 * 348000 / 37545489 = 0.232

**How to apply:** Use CRF56 masks for deployment. Train with CRF56-encoded masks
(not lossless) so the renderer learns to handle AV1 boundary artifacts.
