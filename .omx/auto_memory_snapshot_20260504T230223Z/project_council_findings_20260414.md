---
name: Council Findings Session 2026-04-14
description: Lagrangian annealing, coupled trajectory, scorer architecture, DDELab intel, advisory council insights
type: project
originSessionId: ecc348c1-2829-48d3-9d4c-10ed2aa188a8
---
## Lagrangian Annealing Discovery
- Clamping λ from 10000→1000 on resume freed PoseNet from over-constraint
- v5 best (ep12600): auth=0.87, 13% over v3 (1.00)
- Late checkpoint (ep16999): auth=1.37 — weaker caps let model drift
- λ-sweep: 0.87 is ceiling for asymmetric warp (5 different λ_cap values all plateau)
- This is architectural, not optimization — warp interpolation artifacts limit PoseNet

## Coupled Trajectory Optimizer
- Independent optimization: PoseNet DIVERGES (SegNet dominates after step 400)
- Coupled (4D-Var): PoseNet CONVERGES (0.163 best with annealing + snapshot)
- Compress weight annealing: full weight 0-40%, then decay to 10% by end
- PoseNet snapshot: track best PoseNet across all steps, return that state
- From noise: PoseNet floor ~0.16 (not competitive with renderer's 0.031)
- From warm start (GT/renderer): PoseNet stays near starting value (breakthrough)

## Scorer Architecture (confirmed)
- SegNet: smp.Unet('efficientnet-b4'), ImageNet pretrained, 6 classes, CrossEntropyLoss
- SegNet.preprocess_input: just resize to (384,512). Raw RGB, no YUV.
- PoseNet.preprocess_input: resize, then rgb_to_yuv6 → (B, T*6, 192, 256)
- PoseNet operates in YUV420 space internally
- Scorer input resolution: (512, 384) = (W, H)

## DDELab/Steganalysis Intel
- DDELab = Fridrich + Yousfi's joint lab at Binghamton
- SRNet, OneHotConv (dilation=8 for DCT blocks), EfficientNet with nostride surgery
- comma scorers are DRIVING models, NOT forensic detectors — no nostride, no OneHotConv
- Focus on semantic/geometric fidelity, not pixel-level statistical anomalies
- Generated frames don't need to be forensically perfect

## Advisory Council Insights
- Shannon: 472K FP4 params = 1.89M bits of learned prior to expand 912K mask bits to 4.7M RGB bits
- Dykstra: TTO warm-start = alternating projections near intersection of SegNet+PoseNet constraint sets
- Maxwell: PoseNet divergence = temporal field discontinuities from warp interpolation; anisotropic diffusion along motion trajectories is optimal
- Karpathy: snap-to-best = early stopping with checkpointing, loss landscape locally convex near GT
- LeCun: joint pair gen = Y-shaped U-Net, conv not transformer at 500K params
- Tao: concatenated 10-channel mask input, not separate processing
