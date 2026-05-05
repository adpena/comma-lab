---
name: EUREKA — Pose-Space TTO (optimize FiLM conditioning, not pixels)
description: Optimize the 6D FiLM pose vector per pair instead of 707M pixels. 196,608x compression of optimization space. 14.4KB archive. Contest-compliant.
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## The Insight (2026-04-20)

The renderer + FiLM is a conditional generator: frame = renderer(mask, pose).
Instead of pixel-space TTO (optimize 707M pixel values), optimize the 6D POSE VECTOR
per pair to minimize the contest score.

- Pixel-space TTO: 1,179,648 values per pair → proxy 0.195, auth 0.37
- Pose-space TTO: 6 values per pair → convergence near-instant, same archive cost

The "optimized pose" is not physically meaningful — it's a 6D instruction to the
FiLM layer for how to render that pair optimally for the scorers.

## Math
- 196,608x compression of optimization space
- Archive cost: 600 × 6 × 4 = 14.4KB (same as current poses.pt)
- Rate impact: 0.0096 (negligible)
- If PoseNet auth drops from 0.0145 to 0.005: saves 0.157 on score

## Can combine with latent codes
- Optimize pose (6D) + latent (16D) = 22D per pair
- Archive: 600 × 22 × 4 = 52.8KB
- Rate: 0.035 (still negligible)
- This is a 22-dimensional conditioning-space TTO

## Implementation
- At compress time: freeze renderer, optimize poses via Adam against full scorers
- At inflate time: load optimized poses from archive, single forward pass
- Fully contest-compliant. No scorers at inflate time.

## Also discovered
- Auth SegNet (0.00102) is 2x BETTER than proxy (0.002) — roundtrip helps SegNet
- Auth PoseNet (0.01454) is 2x WORSE than proxy (0.007) — roundtrip hurts PoseNet
- Archive compression (FP4 + entropy masks) saves 0.055 for free
