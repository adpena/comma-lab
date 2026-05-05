---
name: Shannon's Optimal Capacity Allocation
description: The scoring formula has known multipliers — solve the convex optimization for optimal seg/pose/rate allocation
type: project
originSessionId: ecc348c1-2829-48d3-9d4c-10ed2aa188a8
---
## The Insight
Score = 100*seg + sqrt(10*pose) + 25*rate

This is a weighted sum with a CONCAVE term (sqrt on pose). Given fixed total model
capacity C, what's the optimal allocation between SegNet quality, PoseNet quality,
and compression rate?

## Why This Matters
- sqrt(10*pose) has diminishing returns — each unit of PoseNet improvement is worth
  LESS as you approach zero
- 100*seg is linear — each unit of SegNet improvement is worth the same
- 25*rate is linear — each byte saved is worth the same

There's a mathematically optimal point where marginal improvement in any dimension
gives equal score reduction. This determines whether to focus on PoseNet, SegNet, or rate.

## To Implement
Compute the partial derivatives and set equal:
- d(score)/d(seg) = 100
- d(score)/d(pose) = sqrt(10) / (2*sqrt(pose)) = 5/sqrt(10*pose)
- d(score)/d(rate) = 25

At optimal: 100 * d(seg)/d(C) = [5/sqrt(10*pose)] * d(pose)/d(C) = 25 * d(rate)/d(C)

Visualize as a 3D Pareto surface on the writeup site.
