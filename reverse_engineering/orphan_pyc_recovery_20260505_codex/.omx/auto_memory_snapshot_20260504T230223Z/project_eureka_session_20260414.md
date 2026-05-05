---
name: Grand Council Eureka Session 2026-04-14
description: Peak flow insights — TTO is the last mile, priority-queue frames, DALI decode risk, Quantizr strategy decoded
type: project
originSessionId: ecc348c1-2829-48d3-9d4c-10ed2aa188a8
---
## Top Eureka Moments

**Fridrich**: Renderer is a warm-start factory, not the final output. TTO bridges the manifold gap (0.031→0.005). Stop training harder, just TTO everything.

**Tao**: sqrt in scoring formula is CONCAVE — Jensen's inequality means fixing worst frames gives SUPER-LINEAR improvement. TTO should prioritize worst-PoseNet frames first (priority queue).

**Hotz**: 500 steps × 600 pairs × 2 forward passes × 3ms = 30 minutes. EXACTLY fits inflate budget. Someone designed the contest knowing this attack exists. Quantizr is already doing it.

**LeCun**: v4 supervised regression proves PoseNet supervision and warp geometry are adversarial at current model scale. Don't add more losses — train geometry only, let TTO handle PoseNet.

**Shannon**: Quantizr's FP4 codebook is a mu-law compander for neural weights. Adopt it directly.

## What Quantizr Is Actually Doing
- Joint pair generation (both frames in one forward pass)
- Hand-tuned FP4 codebook for EfficientNet weight distribution
- Mask-as-video at ~2KB (more mask precision than our 239B)
- 100% of inflate budget allocated to TTO (generator is fast enough)
- Risk for us: splitting time between rendering and TTO loses on BOTH

## Critical Risks
1. **DALI decode divergence**: TTO optimizes against PyTorch CPU decode, auth uses NVDEC GPU decode. Different pixel values. PoseNet is sensitive (29x factor proven). Mitigation: add Gaussian noise during TTO for robustness.
2. **Rate explosion**: TTO frames may be less compressible. Monitor rate.
3. **Deadline crunch**: 19 days = 10 iteration days after overhead.

## Moon Shots
- Maxwell: Learned TTO correction network (amortized TTO — 1 forward pass replaces 500 gradient steps)
- Dykstra: Alternating projections between PoseNet/SegNet constraints (could cut steps 500→100)

## Binding Consensus
Ship TTO as immediate priority. Priority-queue frame ordering. Gaussian noise augmentation. Auth eval within 48 hours.
