---
name: Session 2026-04-24 comprehensive results — 29 commits, proxy 0.441, full trick stack
description: Massive session. eval_roundtrip fixed across 15 files. HWC/CHW bug. Radial zoom (rank-1 PoseNet). Brotli. Half-frame. WILDE+SHIRAZ profiles. Freeze/unfreeze. Full competitive reverse-engineering. Council greenup achieved.
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---

## Session Outcomes

29 commits. 4090 proxy 0.441 at epoch 2125 (still improving).
Two deployable profiles (WILDE + SHIRAZ) greenup'd by 3 consecutive clean council passes.

## Critical Bugs Fixed

1. eval_roundtrip: 15 files had broken/missing roundtrip (GT cache bypass, HWC/CHW
   mismatch causing 50x scorer inflation, noise_std=0, bilinear-only instead of STE)
2. error_boost_phase3: stored in config but never consumed (dead code)
3. Curriculum difficulty: used PoseNet-only (wrong), now uses full contest formula
4. Forensics: zero-padded gradients, 4-connected boundaries, geometric mean collapse

## Innovations Implemented

1. RadialZoomWarp: 600 scalars replace 50K-param MotionPredictor. Validated 10.6x
   PoseNet improvement. Based on rank-1 Jacobian discovery (dim 0 = 99.8% variance).
2. Freeze/unfreeze training (Quantizr 5-stage adapted)
3. Error_boost 9x→49x per-pixel magnification with clamp
4. PCGrad + focal STE wired into train_distill.py (SHIRAZ)
5. Brotli compression end-to-end
6. Half-frame masks + binary poses
7. Forensic analysis tools (boundary, sensitivity, roundtrip, class boundary)
8. Canonical pipeline.py (video → archive with convergence-driven iteration)
9. Replicate padding + dilated ResBlocks in renderer

## Intelligence Gathered

- Quantizr: JointFrameGenerator (NOT warp), 87K params, GroupNorm, 5-stage freeze,
  error_boost 9x/49x, GT poses (NOT optimized), Brotli on everything
- szabolcs-cs: shared latent + 6-DOF affine, block FP at 1.017 bits/weight,
  better PoseNet than Quantizr, tar.xz double compression
- kaileh57: dilated convs = "single largest win"
- Yousfi: boundary artifacts hint, SegNet/PoseNet orthogonal

## Deployable Profiles

WILDE: base_ch=32, mid_ch=48, freeze/unfreeze, error_boost 9→49, hinge margin=1.0
SHIRAZ: same architecture, PCGrad, focal STE γ=2, continuous adaptive, curriculum

## Key Strategic Insights

- SegNet gap (5.7x vs Quantizr) IS the competition. Rate is solved.
- PoseNet is rank-1 — radial zoom from FoE captures 99.8% of variance
- SegNet/PoseNet are architecturally orthogonal (SegNet sees only frame_t1)
- CLADE is our advantage over Quantizr's GroupNorm
- Lane markings encode vehicle speed (zero archive cost motion estimation)
- Curriculum must weight by FULL score formula, not PoseNet-only
