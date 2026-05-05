---
name: CARVE arXiv 2604.21713 (3D Visual Geometry) — NOT-APPLICABLE
description: 2026-04-28 evaluated CARVE per user mandate. NOT-APPLICABLE across all 9 of our 3D geometry lanes (Lane HM/CG/GE/SE3/GP/FL/M-V3/HF + Pose TTO loop). Mechanism: 1.2B-param VGGT-derived feed-forward ViT for uncalibrated multi-view geometry. Wrong scale (1000× our 100K renderer), wrong hardware (H100/H200), wrong output (absolute trajectory, 9-DOF + arbitrary scale). PoseNet wedge ceiling = 0.146 pts even with perfect pose; CARVE doesn't move the dominant SegNet (38%) or Rate (44%) wedges.
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## Paper identity (verified)
- arXiv 2604.21713v1, 23 Apr 2026, last-modified 24 Apr 2026
- Title: "Unlocking the Power of Critical Factors for 3D Visual Geometry Estimation"
- Authors: Guangkai Xu et al. (Zhejiang Univ State Key Lab CAD&CG + Tsinghua + Ant Group)
- Code: github.com/aim-uofa/CARVE (BSD-2, 1 commit, 9 stars — release-stub only)
- CVPR 2026 accepted

## Mechanism
- DINOv2 ViT encoder + transformer + 3 heads (depth, pointmap, camera)
- Camera output: 9-dim (quaternion 4 + translation 3 + FoV 2)
- 1.214B params, 211.87 TFLOPs/inference, 25.71 GiB peak HBM
- Trained 30K iterations on Hypersim/ScanNet++/VKITTI2/BlendedMVS/TartanAir/ARKitScenes
- Loss innovations: L_reg(W_inv) fixed inverse-depth weight (replaces learnable confidence — that DEGRADES), L_F per-frame scale-shift, L_consis enforcing intrinsics·depth = pointmap
- Cross-resolution fusion: ViT processes I_low + I_high separately, high-res via β·CrossAttn with zero-init β

## KITTI camera-pose results (Table 8)
- CARVE: FoV-Rel=0.078, ATE=0.664, RPE-R=0.016, RPE-T=1.740
- VGGT: 0.084 / 1.113 / 0.015 / 2.177
- Pi3: 0.094 / 0.572 / 0.016 / 2.270
- Improvements are second-decimal, ~20% RPE-T over VGGT

## Per-lane verdict (1-10)
- Lane HM (4-pt homography solver): 1/10 — analytic vs 1.2B feed-forward, no overlap
- Lane CG (Faugeras + EON intrinsics): 1/10 — CARVE PREDICTS K instead of using known K, opposite design
- Lane GE/SE3 (Lie algebra): 1/10 — CARVE outputs quaternions, no Lie machinery
- Lane GP (poly-fit pose): 1/10 — different problem (compress poses vs estimate poses)
- Lane FL (RAFT-derived): 2/10 — could replace at compress time, but output is uncalibrated absolute trajectory requiring same Lane-A least-squares calibration we already do
- Lane M-V3-clean (PoseNet-embedding distill): 2/10 — KL distill from CARVE→PoseNet would learn wrong manifold
- Lane HF (Telescope hyperbolic foveation): 1/10 — orthogonal
- Pose TTO loop: 2/10 — Lane-A poses are scorer-measured (load-bearing); CARVE is different coord frame
- NEW lane potential: 2/10 — 7.8% FoV error is useless (we have exact K=910px)

## Why NOT applicable to us (cost/score math)
- Inference cost: ~6 min on RTX 4090 just to forward 1200 frames (38 chunks × 3s on H200, ~2× on 4090)
- No public weights yet → multi-week multi-GPU training to get them = out of budget
- $5+/eval cycle on Vast.ai H100 ($3.30/hr)
- PoseNet wedge ceiling: 2× PoseNet improvement saves at most 0.093 pts; CARVE's 20% RPE-T over VGGT translates to <0.02 pts on our score
- Cost-benefit: $5+/eval for ≤0.02 expected delta is dominated by every other in-flight lane

## Strategic note (anti-arbitrariness)
PoseNet is 18% wedge; perfect pose saves at most 0.146 pts → 0.91 vs Quantizr 0.33. The remaining headroom is SegNet (38%) and Rate (44%), neither of which CARVE addresses. Lane G v3 already hit PoseNet=0.0035 via our scorer's own gradient (Pose TTO). The CARVE slot should be reallocated to Lane EC-V2 / Lane EBR / Lane PRIOR which directly attack dominant wedges.

## Cross-references
- `.omx/research/lane_g_v3_stacking_skunkworks_20260428.md` — wedge attribution
- `project_lane_taxonomy_stacking_strategy_20260427` — full lane taxonomy
- `feedback_baseline_poses_load_bearing` — why Lane A poses can't be replaced
- `project_lane_m_v2_audit_council_findings_20260428` — Lane M lineage
- `project_nitrobrew_evaluation_NOT_APPLICABLE_20260428` — sibling NOT-APPLICABLE evaluation
