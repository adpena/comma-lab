# Research Roadmap — Task-Aware Neural Video Compression

## Active Priority: Beat Quantizr (deadline May 3, 2026)

### NOW — Asymmetric Warp Training (deployed on Modal T4)
- AsymmetricPairGenerator with Fridrich 3-phase curriculum
- Warp-based pair generation (frame_t = warp(frame_t1) + gate * residual)
- Dashboard: check Modal for live training logs

### NEXT — Highest-Leverage Stacking (post-validation)

#### 1. Ego-Motion Pre-Computation (2 days, PoseNet -0.3 to -0.5)
- **Tools:** TartanVO, DPVO, or VO from openpilot itself
- **Concept:** Compute exact 6-DOF camera trajectory at compress time
- **Store:** ~100 bytes per frame (6 floats * 4 bytes * 1200 frames = 28KB)
- **Use:** Condition the warp on KNOWN ego-motion rather than predicted
- **Effect:** PoseNet becomes nearly deterministic — the warp IS the ego-motion
- **Papers:** TartanVO (Wang et al., CoRL 2021), DPVO (Teed et al., NeurIPS 2023)
- **Status:** NOT STARTED

#### 2. Learned Entropy Coding on Weights (1 day, rate -0.1)
- **Tools:** CompressAI entropy modules, arithmetic coding
- **Concept:** Replace raw FP4 byte packing with learned probability model
- **Effect:** 20-30% archive size reduction at zero quality cost
- **Papers:** COIN/COIN++ (Dupont et al., NeurIPS 2021 / TMLR 2022)
- **Framework:** `pip install compressai` — has ready-made entropy models
- **Status:** NOT STARTED

#### 3. Depth Conditioning (1 day, PoseNet -0.05 to -0.2)
- **Tools:** ZoeDepth, MiDaS (both open-source, run in seconds on GPU)
- **Concept:** Pre-compute monocular depth at compress time, store as side info
- **Store:** uint8 depth per frame, delta-coded + entropy → ~100KB for 1200 frames
- **Use:** Add as extra input channel to renderer (geometric awareness)
- **Effect:** Renderer knows which pixels are close/far → better PoseNet preservation
- **Papers:** ZoeDepth (Bhat et al., 2023), PixelNeRF (Yu et al., CVPR 2021)
- **Status:** NOT STARTED

#### 4. Occlusion Mask from Depth + Ego-Motion (0.5 days, both -0.05)
- **Concept:** Compute WHERE warping fails (new content visible due to ego-motion)
- **Store:** Binary mask, ~5KB compressed per video
- **Use:** Extra input channel to renderer → focus correction on disoccluded regions
- **Papers:** SynSin (Wiles et al., CVPR 2020), Worldsheet (Hu et al., ICCV 2021)
- **Status:** NOT STARTED — depends on #1 and #3

#### 5. AWQ-Style Mixed-Precision Quantization (1 day, rate + quality)
- **Tools:** AWQ (Lin et al., MLSys 2024)
- **Concept:** Quantize each weight proportional to its influence on scorer loss
- **Effect:** Higher quality per byte — salient weights at FP8, others at FP4
- **Status:** NOT STARTED

### MEDIUM-TERM — Post-Validation Architecture Improvements

#### 6. RAFT-Lite Correlation Volume for MotionPredictor (3 days)
- Compute all-pairs pixel similarity between frames at low resolution
- Iteratively refine flow with GRU-style update
- Much better motion estimation than our current conv stack
- **Paper:** RAFT (Teed & Deng, ECCV 2020)

#### 7. NeRV-Style Positional Encoding (0.5 days)
- Add sinusoidal encoding of frame index as input to renderer
- Per-frame awareness without per-frame parameters
- **Paper:** NeRV (Chen et al., NeurIPS 2021), E-NeRV (Li et al., ECCV 2022)

#### 8. RIFE Frame Interpolation for Backward-Delta (1 day)
- Store only keyframes, interpolate intermediates
- Reduces number of frames that need full rendering
- **Paper:** RIFE (Huang et al., ECCV 2022)

#### 9. Scale-Space Warping (2 days)
- Instead of single-scale optical flow, use multi-scale warp
- Decoder selects optimal blur/sharpness per pixel
- **Paper:** Scale Space Flow / SSF (Agustsson et al., CVPR 2020)

#### 10. Knowledge Distillation (3 days)
- Train large renderer (2M params, depth=2) for max quality
- Distill into small renderer (287K params) for deployment rate
- Teacher-student with MSE on teacher outputs + scorer loss
- Standard technique in model compression

### LONG-TERM — Post-Competition Research Directions

#### 11. 3D Gaussian Splatting as Video Codec
- **Concept:** Fit 3D Gaussians to driving scene, store Gaussian params, render dashcam view
- **Challenge:** Monocular reconstruction quality, archive size (2-5MB), training time
- **Papers:** Street Gaussians (Yan et al., ECCV 2024), 4D-GS (Wu et al., CVPR 2024)
- **Timeline:** 1-2 months research project
- **Potential:** If monocular reconstruction improves, this could be the ultimate video codec

#### 12. 4D Gaussian Splatting (3D + Time)
- Deformable Gaussians that move/rotate over time
- Separate static background from dynamic objects
- **Papers:** Deformable 3D Gaussians (Yang et al., CVPR 2024), SC-GS (Huang et al., CVPR 2024)
- **Application:** Driving scene with moving cars/pedestrians

#### 13. Neural Radiance Fields for Video (NeRF-as-Codec)
- D-NeRF, HyperNeRF for dynamic scenes
- Model weights ARE the compressed video
- **Challenge:** Training time, monocular input, quality on driving scenes

#### 14. Video Coding for Machines (VCM)
- MPEG standard in progress for machine-consumption video
- Formalized rate-accuracy tradeoff framework
- Our Fridrich approach is a concrete instantiation of VCM principles
- **Papers:** VCM ad-hoc group publications (Duan, Sun, et al.)

#### 15. Ego-Motion Decomposition
- Separate global camera motion from local object motion
- Global: single 6-DOF transform (ego-motion)
- Local: per-object bounding box + motion vector
- **Paper:** MCNet (Villegas et al., ICML 2017)
- Improves both rate (less to encode) and quality (each motion type optimized separately)

## Key Principles

1. **Unlimited compute at compress time, constrained at inflate time.** Pre-compute everything.
2. **The architecture IS the score.** Pair-wise warp generation > loss engineering.
3. **Task-aware > perception-aware.** Optimize for scorer, not human eyes.
4. **Decode on target.** No precomputed frames cross environment boundaries.
5. **Council decides design.** Skunkworks team reviews everything before GPU time.

## Reference Papers (by relevance)

### Must-Read (directly applicable)
- TartanVO (Wang et al., CoRL 2021) — visual odometry for ego-motion
- RAFT (Teed & Deng, ECCV 2020) — optical flow
- CompressAI framework — learned entropy coding
- AWQ (Lin et al., MLSys 2024) — activation-aware weight quantization
- COIN/COIN++ (Dupont et al.) — neural codec as weight compression
- NeRV/E-NeRV (Chen et al.) — neural video representation

### Should-Read (architectural insights)
- SynSin (Wiles et al., CVPR 2020) — novel view synthesis with occlusion
- ZoeDepth (Bhat et al., 2023) — monocular depth estimation
- RIFE (Huang et al., ECCV 2022) — frame interpolation
- PWC-Net (Sun et al., CVPR 2018) — coarse-to-fine optical flow
- MCNet (Villegas et al., ICML 2017) — motion decomposition

### Explore Later (long-term research)
- Street Gaussians (Yan et al., ECCV 2024) — driving-specific 3DGS
- 4D Gaussian Splatting (Wu et al., CVPR 2024) — dynamic 3DGS
- DCVC-DC (Li et al.) — state-of-art learned video codec
- VCM MPEG standard publications
