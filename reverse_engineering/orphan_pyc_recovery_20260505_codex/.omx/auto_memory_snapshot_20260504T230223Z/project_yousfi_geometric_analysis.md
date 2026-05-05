---
name: Yousfi geometric analysis — PoseNet manifold, centroid errors, Fridrich strategy
description: PoseNet output is learned 6D embedding NOT SE(3). True manifold M=R¹×S. Centroids wrong for 3 reasons. Fridrich strategy: match dim0 (speed), ignore dims 1-5 (max 0.18 pts). Masks dominate archive, not poses.
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---

## PoseNet Output Space

NOT classical pose (tx,ty,tz,rx,ry,rz). Learned 6D embedding trained on openpilot data.
FastViT-T12 extracts features → Hydra head → 12D (first 6 scored).

Dim 0: forward speed on learned scale. Mean=31.26, std=1.26, range=[23.5, 35.1].
  Temporal autocorrelation=-0.52 (mean-reverting, cruise control).
Dim 1: yaw-related (steering), coupled with dim 5 (corr=0.53). Autocorr=0.79.
Dim 2: pitch-related, anticorrelated with dim 4 (corr=-0.46). Autocorr=0.87.
Dim 3: fast angular mode (steering corrections). Autocorr=0.69.
Dim 4: fastest-changing (bumps, vibrations). Autocorr=0.55.
Dim 5: near-constant (scene-level: camera mount, road grade). Autocorr=0.98.

## True Manifold: M = R¹ × S

R¹ = speed line [23.5, 35.1]. S = ~2-3 dim surface in R⁵.
Shannon effective dimensionality: 1.017. Participation ratio: 1.004.
NOT SE(3), SE(2), or H(3). Learned manifold from FastViT + openpilot data.

## Why Centroids Are Wrong

1. Spatial averaging destroys structured flow field (depth-dependent)
2. Conflates yaw, lateral translation, road curvature, mask changes
3. Wrong domain: PoseNet sees YUV6 at 192×256, not segmentation masks

## Fridrich Strategy for Beating PoseNet

"Match radial zoom exactly (dim 0). Let everything else be approximately correct.
Dims 1-5 contribute 0.18 points MAXIMUM even if you predict the MEAN.
Spend remaining bit budget on SegNet, where 100x multiplier matters."

## Archive Budget Reality

Masks: 222KB (89%). Renderer: 22KB (9%). Poses: 7KB (3%).
Masks dominate. Pose compression yields kilobytes. Mask compression yields tens of KB.
Engineering effort → mask compression, not pose compression.

## Validated: Zoom works

optimize_zoom_scalars on current checkpoint:
  Baseline loss: 29.6 → Best with zoom: 6.6 (4.5x improvement)
  600 scalars, 1.2KB archive cost. Rank-1 discovery empirically confirmed.
