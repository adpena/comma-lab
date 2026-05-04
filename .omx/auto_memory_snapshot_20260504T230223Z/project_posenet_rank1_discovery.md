---
name: MAJOR DISCOVERY — PoseNet is rank-1, optimal conditioning is scalar radial zoom
description: PoseNet Jacobian rank 1.008. Dim 0 captures 99.8% of variance. Optimal frame conditioning is a single scalar radial zoom from vanishing point (256,174). All existing mechanisms (FiLM, affine, MotionPredictor) are massively overparameterized.
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---

## The Finding

PoseNet's 6D output has effective rank 1.008. GT pose dim 0 captures 99.80% of total
variance (mean=31.26, std=1.26, range=11.6). Dims 1-5 are noise-level (range 0.06-0.20).

The frame-to-frame relationship is a **scalar radial zoom** centered at the Focus of
Expansion (vanishing point) at (256, 174) in scorer coordinates.

Optimal warp: grid = FoE + exp(s_i) * (coord_grid - FoE)
Archive cost: 600 FP16 scalars = 1.2KB

## Overparameterization of Existing Methods

| Method | Params for 1 DOF | Archive | Validated? |
|--------|-----------------|---------|-----------|
| Radial zoom (proposed) | 0 model + 600 scalars | 1.2KB | Theory |
| ego_flow 6-param affine | 0 model + 3600 | 1.8KB | Exists |
| FiLM 6D poses | 1440 model + 3600 | 1.8+7.2KB | Quantizr 0.33 |
| Dense MotionPredictor | 50K model + 0 | 0 extra | Our current |
| szabolcs-cs 6-DOF affine | 10.8K model + 7200 | 2.4KB | 0.36 |

## Physical Interpretation

This driving video is highway dashcam at 20fps. The dominant motion is forward
translation → radial expansion from the FoE. Lateral motion (curves) contributes
0.2% of pose variance. The FoE is the vanishing point of the road.

PoseNet's output dim 0 ≈ 31 + speed_variation. It's NOT SE(3) or Lie algebra —
it's an arbitrary learned embedding where one dimension dominates.

## How to Apply

1. Replace MotionPredictor (50K params) with RadialZoomWarp (600 scalars)
2. Archive savings: ~50KB model → ~1.2KB archive (rate improvement ~0.03)
3. Quality: captures 99.8% of PoseNet signal
4. For the remaining 0.2%: add 1 more param (lateral shift) if needed
5. This is compatible with our AsymmetricPairGenerator — just replace
   the motion source from MotionPredictor to stored zoom scalars
