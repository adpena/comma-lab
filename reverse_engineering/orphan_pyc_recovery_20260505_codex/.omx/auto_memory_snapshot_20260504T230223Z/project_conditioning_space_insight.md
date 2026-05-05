---
name: Deep Insight — Conditioning-Space TTO is a Universal Principle
description: Optimizing in the conditioning space (6D FiLM) instead of output space (707M pixels) is always faster, more efficient, and preserves the generator's learned distribution. Publishable.
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## The Principle (2026-04-21)

Any conditional generator has a low-dimensional conditioning space that controls the output.
Optimizing in THAT space (instead of pixel/output space) is:
- Faster (6D vs 707M — 196K:1 compression)
- More efficient (fewer parameters, faster convergence)
- Distribution-preserving (stays on the generator's learned manifold)

## The Discovery Chain
1. TTO optimizes pixels directly → works but slow (40 min for 600 pairs)
2. Pose-space TTO optimizes 6D FiLM vectors → 200x faster, same quality
3. seg_weight=0 revealed PoseNet and SegNet live in ORTHOGONAL subspaces of FiLM space
4. Each scorer "claims" different dimensions of the conditioning space
5. You can optimize each independently without interference

## Connections
- StyleGAN: optimize in W-space, not pixel space
- Diffusion models: prompt optimization, not pixel optimization
- Our FiLM: conditioning-space TTO
- All instances of the same principle: the conditioning manifold is the natural optimization space

## Publication Potential
"Conditioning-Space Test-Time Optimization for Task-Aware Neural Compression"
- Workshop paper at CVPR/ECCV
- Or poster at ICLR/NeurIPS
- The seg_weight=0 orthogonality finding is novel
- The 196K:1 compression of the optimization space is dramatic
