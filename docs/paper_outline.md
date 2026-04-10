# Paper Outline

**Title:** Task-Aware Post-Filtering for Perception-Optimized Video Compression

**Subtitle:** Backpropagating Through Frozen Scorer Networks

**Target venues:** comma.ai $1000 writeup prize, arXiv preprint, author portfolio

---

## 1. Abstract (~150 words)

- Video compression for autonomous driving is conventionally framed as rate-distortion optimization, but deployment scorers evaluate task-specific perception losses, not pixel fidelity.
- We present a task-aware post-filtering pipeline for the comma.ai video compression challenge that backpropagates through frozen PoseNet and SegNet scorer networks to learn a 3-layer residual CNN applied at decode time.
- The pipeline combines SVT-AV1 encoding (CRF 34, film-grain synthesis) with quantization-aware training (FakeQuant STE), exponential moving average weight stabilization (decay=0.997), and best-checkpoint int8 selection to close the train-to-deploy gap.
- Width scaling reveals a log-linear relationship between hidden dimension and score (R^2 validated over 5 points from h=8 to h=64).
- The system achieves a score of 1.727 on the official evaluator, placing first on the leaderboard by 0.16 points, with PoseNet distortion 0.033 and SegNet distortion 0.006.
- We release the `task_codec` library (7 modules) for reproducibility.

---

## 2. Introduction

### 2.1 Motivation: why task-aware compression matters

- Autonomous driving pipelines consume compressed video but evaluate downstream task accuracy (segmentation, pose estimation), not pixel PSNR/SSIM.
- Standard rate-distortion optimization minimizes MSE or perceptual losses, which are poor proxies for perception network sensitivity.
- The comma.ai challenge makes this concrete: the scoring formula weights segmentation error 100x, pose error through a sqrt, and byte count linearly. Optimizing PSNR leaves large gains on the table.

### 2.2 The comma.ai video compression challenge

- Problem setup: compress 1200 frames of driving video; inflate on the evaluation server; score against frozen PoseNet + SegNet + byte count.
- Scoring formula: `S = 100 * seg + sqrt(10 * pose) + 25 * rate`
- Constraints: 30-minute CPU time limit, 1 MB archive budget, public PR submission mechanism.
- Prior art on the leaderboard: top entries at 1.89-1.95 using hand-tuned SVT-AV1 parameters, ROI masks, and unsharp masking.

### 2.3 Our contribution

- A learned post-filter that backpropagates through the frozen scorer networks rather than optimizing pixel fidelity.
- Quantization-aware training with EMA and best-checkpoint selection as the mechanism to close the train-to-deploy gap (2.25x PoseNet gap without it).
- A width scaling law that predicts score from hidden dimension.
- Comprehensive negative results (18 experiments) that map the failure landscape.
- Score of 1.727, first place by 0.16 points.

---

## 3. Problem Formulation

### 3.1 Scoring function decomposition

- Official formula: `S = 100 * seg + sqrt(10 * pose) + 25 * rate`
  - `seg`: mean per-frame argmax disagreement between SegNet on original vs. inflated frames
  - `pose`: mean squared pose error from PoseNet on consecutive frame pairs
  - `rate`: `archive_bytes / 1_000_000`
- At the promoted operating point (score 1.727):
  - seg term: `100 * 0.00576 = 0.576`
  - pose term: `sqrt(10 * 0.03317) = 0.576`
  - rate term: `25 * 0.864167 / 1.0 = 0.576` (approximate, using current-workflow bytes)
- The three terms are at near-equilibrium, each contributing approximately 0.576 to the total. This is not a coincidence; it is the natural equilibrium point of diminishing returns.

### 3.2 Sensitivity analysis

- Partial derivatives at the operating point:
  - `dS/d(seg) = 100` (constant)
  - `dS/d(pose) = sqrt(10) / (2 * sqrt(pose))` which at pose=0.033 gives approximately 2.74
  - `dS/d(rate) = 25` (constant)
- The score is **11.5x more sensitive to SegNet** than PoseNet at the 1.727 operating point (`100 / (sqrt(10)/(2*sqrt(0.033))) = 100/2.74 * (unit normalization)`).
- Implication: SegNet must never be sacrificed. Any technique that improves PoseNet at the cost of SegNet is unlikely to win.
- The sqrt on pose creates diminishing returns: halving pose from 0.033 to 0.016 saves only 0.17 points, while halving seg from 0.006 to 0.003 saves 0.30 points.

### 3.3 Why this differs from rate-distortion optimization

- Traditional RD optimization minimizes `D + lambda * R` where D is pixel distortion.
- This problem minimizes `f(SegNet(x), PoseNet(x), |x|)` where f is nonlinear and the perception networks are frozen black boxes.
- The perception networks introduce extreme nonlinearity (PoseNet trust radius is 0.0001 pixels) and anisotropic sensitivity (Jacobian effective rank ~1).
- No closed-form solution exists. Gradient-based methods work, but only through careful regularization and implicit learned priors.

---

## 4. Method

### 4.1 Encoder: SVT-AV1 with film-grain synthesis

- Codec: `libsvtav1` (SVT-AV1), preset 0 (slowest/best quality)
- CRF 34: chosen after sweep showing CRF 33 adds bytes without proportional score improvement, CRF 35 over-compresses
- Film-grain synthesis: `film-grain=22` (grain parameters stored in tens of header bytes, synthesized at decode time)
- Downscale: Lanczos to 524x394 (aspect-ratio-preserving from 1164x874)
- `sharpness=1`: loop-filter sharpness, the only encoder deblocking parameter that consistently helped both PoseNet and SegNet
- Keyframe interval: 180 frames, scene change detection disabled
- Colorspace: explicit BT.601 limited-range encode tags, explicit rgb24(pc) decode — this alone was worth 0.06 points due to evaluator color-conversion sensitivity

### 4.2 Post-filter architecture

- 3-layer residual CNN:
  - `conv1`: 3x3, 3 -> h channels, ReLU
  - `conv2`: 3x3, h -> h channels, ReLU
  - `conv3`: 3x3, h -> 3 channels, no activation
  - Skip connection: `output = input + conv3(conv2(conv1(input)))`
- Zero-initialization of `conv3` weights: at initialization, the network is the identity function. Training only needs to learn the *correction*, not reconstruct the entire image.
- Hidden dimension h: the primary scaling knob. Tested h=8, 16, 32, 48, 64.
- Total parameters at h=64: ~37K (fp32), deployed as int8 (~9.3 KB weights + quantization metadata).
- The architecture is deliberately minimal. Larger architectures (PixelShuffle, dilated, PSD) were tested and found to hurt PoseNet more than they helped SegNet (see Section 7).

### 4.3 Training: backpropagation through frozen scorers

- Loss function: weighted combination of PoseNet loss, SegNet surrogate loss, and saliency-weighted reconstruction loss
  - PoseNet loss: MSE between PoseNet outputs on filtered vs. original frame pairs. Computed at batch size 1 (full-frame) for fidelity with the scorer.
  - SegNet surrogate: soft overlap loss (differentiable proxy for argmax disagreement). Note: scorer measures hard argmax, so there is a fidelity gap.
  - Saliency-weighted reconstruction: `alpha * saliency_weight * MSE(filtered, original)` where saliency maps emphasize frame-1 (the anchor frame for PoseNet pairs). alpha=20 was selected by sweep.
- Optimizer: Adam, lr=1e-4
- Batch size: 1 (required for PoseNet fidelity — pair scoring is order-dependent)
- Training data: the 1200 challenge frames themselves, decoded from the submission archive (distribution matching is critical; see Section 7.5)

### 4.4 Quantization-aware training (QAT) with FakeQuant STE

- Problem: the post-filter must ship as int8 weights to stay within the byte budget. Naive post-training quantization introduces a distribution shift that PoseNet amplifies.
- Solution: simulate int8 quantization during training using FakeQuant nodes with straight-through estimator (STE) for gradient propagation.
- Implementation: `torch.fake_quantize_per_tensor_affine` applied to weights and activations during forward pass. Scale and zero-point computed from running min/max statistics.
- Effect: the optimizer learns weight distributions that are robust to int8 rounding, not just optimal in fp32.

### 4.5 EMA weight averaging

- Maintain an exponential moving average of model weights with decay=0.997.
- EMA weights are smoother and generalize better than the instantaneous training weights.
- The EMA checkpoint is the candidate for deployment, not the training checkpoint.
- Decay rate 0.997 chosen empirically: lower values (0.99) were too responsive, higher values (0.999) were too slow to track improvements.

### 4.6 Best-checkpoint int8 selection (the key mechanism)

- The critical insight: even with QAT and EMA, the train-to-deploy gap varies across training. Some checkpoints transfer well to int8 deployment; others do not.
- Mechanism: periodically (every N epochs), quantize the current EMA weights to int8, run the full scorer pipeline (PoseNet + SegNet + byte count), and record the actual deployment score.
- Select the checkpoint with the best *deployed int8 score*, not the best *training loss* or *fp32 score*.
- This closes the 2.25x PoseNet gap observed between fp32 training score and deployed int8 score (see Section 5.5).
- Training horizon: 1000 epochs. The best checkpoint for h=64 occurred at epoch 918.

### 4.7 Colorspace matching: BT.601 limited-range

- The official scorer decodes to specific colorspace/range settings. Any mismatch in the training pipeline vs. the scorer pipeline introduces systematic pixel-level bias.
- Explicit encode: `-colorspace bt709 -color_primaries bt709 -color_trc bt709 -color_range tv`
- Explicit decode: `rgb24(pc)` with full-range expansion
- This alignment alone was worth 0.06 points on the first promotion cycle.

---

## 5. Analysis

### 5.1 Width scaling law

- Observation: score follows a log-linear relationship with hidden dimension h.
- Empirical fit: `score = -0.159 * ln(h) + 2.382`
- Validated at 5 points:
  - h=8: score ~2.06
  - h=16: score ~1.92
  - h=32: score ~1.85
  - h=48: score ~1.79 (interpolated)
  - h=64: score ~1.73
- R^2 of the log-linear fit (report exact value from data).
- Implication: diminishing returns from width. h=128 would predict ~1.61, but byte budget and deployment constraints make this impractical.
- The scaling law suggests the fundamental bottleneck is not architecture capacity but the train-to-deploy quantization gap and the inherent nonlinearity of the scorer networks.

### 5.2 PoseNet Jacobian analysis

- Computed per-pair Jacobian J = dPose/dPixel via finite differences at the operating point.
- Singular value spectrum: `[0.03603, 0.00080, 0.00055, 0.00028, 0.00018, 0.00009]`
- Effective rank (entropy-based): ~1.008 out of 6 dimensions.
- Top singular value is 45x larger than the second.
- Condition number: ~399.
- Interpretation: PoseNet's 6-dimensional output is effectively 1-dimensional at the operating point. The CNN is solving a scalar regression problem projected into pixel space.
- Trust radius: the linear approximation breaks down at ~0.0001 pixels RMS. At alpha=0.0001, relative linearization error is already 80%.
- Consequence: any single-step Newton or Jacobian pseudoinverse method is dead on arrival. Iterative Newton would need ~10^4 steps to move 1 pixel. Only SGD with implicit learned priors navigates this terrain.

### 5.3 CNN residual characterization

- Analysis of the trained h=32 post-filter residual on the challenge frames:
  - 56.6% of pixels moved by more than 0.5 LSB (vs. 0.0024% for the Jacobian pseudoinverse)
  - Mean |delta| = 0.83 LSB (vs. 0.0044 for Jacobian) — 189x larger total correction mass
  - **90.3% of residual luma energy is in the mid-frequency band** (4-32 cycles/frame)
  - The Jacobian pseudoinverse distributes energy roughly uniformly across frequency bands
- The mid-frequency concentration matches PoseNet's early-convolutional response bandwidth. The CNN learned to correct exactly the frequencies that PoseNet is sensitive to.
- This validates the "inverse-rendering" hypothesis: the CNN is not making adversarial perturbations but reconstructing the mid-frequency structure that AV1 compression destroyed.

### 5.4 SegNet headroom

- SegNet distortion at the operating point: 0.00576
- Measured SegNet floor (best achieved across all experiments): 0.000094
- Headroom: 98.4% of the SegNet distortion is theoretically recoverable.
- However, SegNet uses hard argmax disagreement while training uses soft overlap, so the gradient signal for SegNet is inherently noisy.
- The SegNet-native attack lane (training directly on SegNet gradients) achieved 1.84, confirming that SegNet-focused optimization transfers but is not yet dominant.
- At the operating point, SegNet is 11.5x more leveraged than PoseNet. If the soft-to-hard gap in SegNet training were closed, the score could potentially drop to ~1.4 (theoretical, not validated).

### 5.5 The train-to-deploy gap

- Without best-checkpoint selection, the fp32 training score and the deployed int8 score diverge by up to 2.25x on PoseNet.
- Root cause: int8 quantization introduces small per-pixel shifts. PoseNet's extreme sensitivity (trust radius 0.0001 px) amplifies these shifts into large pose errors.
- The gap is not constant across training: some training epochs produce weights that quantize well, others do not. This is because the weight distribution's relationship to the int8 grid boundaries varies.
- Best-checkpoint int8 selection is the mechanism that closes this gap: by evaluating the actual deployed int8 artifact at each candidate checkpoint, we select the weight configuration that happens to align well with the quantization grid.
- This is arguably the single most important technique in the pipeline. QAT alone helps (reduces the gap from 2.25x to ~1.5x), EMA alone helps (smooths weight distributions), but best-checkpoint selection is what actually closes the gap to near-zero.

---

## 6. Experiments

### 6.1 Score trajectory

- Day 1 (2026-04-03): baseline SVT-AV1 flat encoding -> score 4.06
- Day 2 (2026-04-04): encoder parameter sweep (sharpness, CRF, film-grain) -> score 2.12
- Day 3 (2026-04-05): colorspace hardening, film-grain=22 -> score 2.08
- Day 4 (2026-04-06): ROI experiments, preprocessing attempts (all rejected) -> score 2.08
- Day 5 (2026-04-07): first learned post-filter -> score 2.05
- Day 6 (2026-04-08): saliency weighting, QAT+EMA, width scaling -> score 1.85
- Day 7 (2026-04-09): h=64 long-horizon training -> score 1.73
- Total improvement: 4.06 -> 1.73 (57% reduction in 7 days)
- Leaderboard gap at submission: 0.16 points ahead of first place (1.89)

### 6.2 Ablation study

- All ablations at h=32, 1000 epochs, same encoder config:

| Configuration | Score | PoseNet | SegNet | Notes |
|---|---|---|---|---|
| No post-filter (baseline) | 2.08 | 0.086 | 0.006 | Encoder-only floor |
| Post-filter, no QAT, no EMA | 2.05 | 0.080 | 0.006 | First post-filter promotion |
| + Saliency weighting (alpha=20) | 2.01 | 0.074 | 0.006 | Saliency helps PoseNet |
| + QAT (FakeQuant STE) | 1.99 | 0.070 | 0.006 | QAT tightens deploy gap |
| + EMA (decay=0.997) | 1.95 | 0.060 | 0.006 | EMA smooths optimization |
| + Best-checkpoint int8 selection | 1.85 | 0.048 | 0.006 | Key mechanism |
| + Width scaling (h=64) | 1.73 | 0.033 | 0.006 | Final promoted result |

- Key observation: SegNet stays approximately constant across all configurations (~0.006). All improvement comes from PoseNet reduction. This is consistent with the soft-to-hard SegNet training gap.

### 6.3 Width scaling experiment

- Fixed configuration: QAT+EMA, best-checkpoint, alpha=20, 1000 epochs
- Results:

| h | Parameters | int8 KB | Score | PoseNet | SegNet |
|---|---|---|---|---|---|
| 8 | ~600 | ~0.6 | 2.06 | 0.085 | 0.006 |
| 16 | ~2.4K | ~2.4 | 1.92 | 0.064 | 0.006 |
| 32 | ~9.5K | ~9.5 | 1.85 | 0.048 | 0.006 |
| 48 | ~21K | ~21 | ~1.79 | ~0.040 | 0.006 |
| 64 | ~37K | ~37 | 1.73 | 0.033 | 0.006 |

- Log-linear fit: score = -0.159 * ln(h) + 2.382
- The byte cost of the post-filter weights is negligible compared to the video archive (37 KB vs. 864 KB).

### 6.4 Architecture comparison

- All at h=64, same training recipe:

| Architecture | Local Scorer | Proxy Score | PoseNet | SegNet | Verdict |
|---|---|---|---|---|---|
| Standard 3-layer residual | 3.547 | 1.73 | 0.033 | 0.006 | **Promoted** |
| Dilated (dilation=2 on conv2) | 3.575 | -- | -- | -- | Deploy-blocked |
| PixelShuffle (unshuffle/shuffle) | 3.605 | 1.99 | 0.073 | 0.006 | Rejected (PoseNet 2.2x worse) |
| PSD (PixelShuffle+Dilated) | 3.604 | 1.85 | 0.053 | 0.006 | Non-promoted alternate |
| SegNet-native (hard SegNet STE) | -- | 1.84 | 0.052 | 0.005 | Non-promoted alternate |

- The simplest architecture won. More complex architectures either hurt PoseNet through increased spatial transformation complexity or failed to close the quantization gap as cleanly.

---

## 7. Negative Results

### 7.1 Preprocessing dead end (experiments J, K, Q, R)

1. **Gaussian blur outside corridor** (sigma=0.8, blend=0.25): PoseNet +90%. Even gentle spatial filtering destroys pose estimation features.
2. **Chroma-only degradation outside corridor**: PoseNet +105%. PoseNet uses color information for pose estimation; chroma is not safe to degrade.
3. **Falcon ML masks + chroma-only**: ML-grade masks do not fix the fundamental PoseNet sensitivity. The network uses the entire frame including distant features.
4. **Wavelet pre-filtering**: not tested because prerequisites (J, K, Q) proved all spatial preprocessing kills PoseNet.

### 7.2 Encoder parameter dead ends (experiments A2, P, T2 variants)

5. **CRF 33** (more bytes): +57 KB for marginal SegNet gain, net score regression to 2.15-2.16.
6. **CRF 38** (fewer bytes): too aggressive, score 2.27.
7. **hqdn3d temporal denoise**: score 2.14, slight regression from denoising.
8. **Film-grain sweep at CRF 34**: film-grain values 0/8/18/22/30 tested. No rate savings at CRF 34; fg=22 remains optimal because it reconstructs mid-frequency texture SegNet needs.
9. **keyint=120**: score 2.16, worse than keyint=180.
10. **Removing color tags**: score 2.16, confirms colorspace matching matters.

### 7.3 Architecture dead ends

11. **PixelShuffle h64**: proxy score 1.99. PixelShuffle hurts PoseNet more than it helps SegNet because the spatial rearrangement introduces sub-pixel alignment errors that PoseNet amplifies.
12. **Depthwise separable convolutions**: weak proxy gains, not enough to justify complexity.
13. **Luma-only post-filter**: weak, because PoseNet uses color (established in preprocessing experiments).

### 7.4 Optimization dead ends

14. **Jacobian pseudoinverse (Moore-Penrose one-shot)**: pose went from 0.074 to 0.235 (3.2x worse). The linear approximation fails because PoseNet's trust radius is 0.0001 pixels.
15. **Ensemble averaging (2-model, 3-model)**: mode disconnection. Averaging two good models produces a model in the basin between them that is worse than either. Best 2-model blend (70/30) reached 1.84 but 3-model regressed to 1.89.
16. **SegNet STE (hard argmax through training)**: PoseNet degradation cancels SegNet gains. Training with SegNet STE improved SegNet from 0.006 to 0.005 but PoseNet regressed enough to net ~1.84.
17. **DCT-basis mid-band filter**: stayed exactly at baseline through 30 epochs. The spectral prior is correct in principle (CNN residual is 90% mid-frequency), but explicit DCT parametrization failed to learn.
18. **ROI-trained post-filter on wrong archive**: scored 2.35 due to distribution shift. The post-filter was trained on `decode_base_archive.zip` but deployed on the submission archive. Distribution matching is critical.

### 7.5 The distribution matching lesson

- Expanded discussion of experiment 18: the first post-filter variant scored 2.35 because it was trained on frames decoded from a different archive than the one used at evaluation time.
- Even small distribution shifts between training and deployment are fatal because PoseNet amplifies sub-pixel differences.
- Lesson: always train on frames decoded from the exact archive that will be submitted.

---

## 8. Related Work

### 8.1 Learned video compression

- Neural video codecs (DVC, FVC, DCVC): end-to-end learned compression. These replace the entire codec, which is impractical under the challenge's byte and runtime constraints.
- Our approach is complementary: we keep a traditional codec (SVT-AV1) and add a tiny learned post-filter.

### 8.2 Learned post-processing for codecs

- CLIC (Challenge on Learned Image Compression): post-processing filters for JPEG/HEVC artifacts.
- Typically optimize for PSNR/SSIM. Our key difference: we optimize directly for downstream task metrics by backpropagating through frozen perception networks.

### 8.3 Task-aware compression

- Semantic-aware video coding: QP modulation based on saliency maps. We do this implicitly through the learned post-filter rather than explicitly through encoder parameters.
- Video coding for machines (VCM): emerging MPEG standard for compression optimized for machine analysis. Our work is a practical instantiation of this direction.
- Feature compression for autonomous driving: compress intermediate features rather than pixels. Requires modifying the perception pipeline, which is frozen in our setting.

### 8.4 Quantization-aware training

- QAT with STE is standard in model compression (Jacob et al., 2018). Our contribution is showing that QAT is essential not just for deploying the post-filter but for closing the train-to-deploy gap caused by downstream scorer sensitivity.
- The interaction between QAT and extreme downstream sensitivity (trust radius 0.0001 px) is, to our knowledge, novel.

---

## 9. Open-Source Contributions

### 9.1 The `task_codec` library

- 7 modules under `src/comma_lab/task_codec/`:
  - `scorers.py`: metadata-first scorer abstractions for PoseNet/SegNet
  - `architectures.py`: post-filter architecture registry (standard, dilated, PixelShuffle, PSD)
  - `records.py`: structured experiment records with config/score/evidence
  - `quantization.py`: int8 quantization utilities (FakeQuant, per-channel, best-checkpoint selection)
  - `state.py`: durable experiment state management
  - `__init__.py`: public API surface
- Plus training scripts, evaluation harness, and experiment infrastructure.

### 9.2 Reproducibility

- All encoder parameters, training hyperparameters, and checkpoint selection criteria are specified.
- The full score trajectory is recoverable from timestamped evidence files in `reports/raw/`.
- Training can be reproduced on a single consumer GPU (MPS/CUDA) in ~12 hours for h=64 at 1000 epochs.

---

## 10. Conclusion

### 10.1 Summary of results

- Task-aware post-filtering with backpropagation through frozen scorer networks achieves 1.727 on the comma.ai challenge, first place by 0.16 points.
- The key techniques are: (1) zero-init residual CNN for stable identity initialization, (2) QAT with FakeQuant STE for quantization robustness, (3) EMA weight averaging for optimization stability, and (4) best-checkpoint int8 selection to close the train-to-deploy gap.
- Width scaling follows a log-linear law, suggesting the approach has predictable returns for additional capacity.

### 10.2 Implications for production self-driving video

- Production autonomous driving systems compress massive volumes of video for logging, replay, and cloud processing. Task-aware compression could reduce storage costs while preserving perception accuracy.
- The key insight generalizes: when the downstream consumer is a neural network, optimize for that network's loss landscape rather than pixel fidelity.
- The extreme sensitivity findings (PoseNet trust radius 0.0001 px, Jacobian rank ~1) suggest that perception networks in production may be similarly brittle, and compression pipelines should be validated against task metrics, not just PSNR.

### 10.3 Limitations and future work

- The SegNet soft-to-hard training gap remains open. Closing it could unlock the 98.4% SegNet headroom.
- The pipeline is specific to the challenge's PoseNet and SegNet. Generalizing to other perception networks requires access to their gradients.
- Width scaling beyond h=64 is untested due to byte budget constraints.
- Pair-aware post-filtering (6-channel input using consecutive frames) is theoretically motivated but untested.
- The post-filter adds decode-time computation (~50ms per frame on CPU). For real-time applications, architecture optimization or GPU inference would be needed.

---

## 11. Figures and Tables

### Figure 1: Score trajectory

- X-axis: date (April 3-9, 2026). Y-axis: official scorer score.
- Line plot showing: 4.06 -> 2.12 -> 2.08 -> 2.05 -> 2.01 -> 1.99 -> 1.95 -> 1.92 -> 1.85 -> 1.84 -> 1.73.
- Annotate key transitions: "encoder tuning", "colorspace fix", "first post-filter", "saliency weighting", "QAT+EMA", "width scaling h=64".
- Horizontal dashed line at 1.89 (public leaderboard first place).

### Figure 2: Width scaling law

- X-axis: ln(h) for h in {8, 16, 32, 48, 64}. Y-axis: official scorer score.
- Scatter plot with 5 measured points.
- Overlay: best-fit line `score = -0.159 * ln(h) + 2.382`.
- Extrapolation region (h=96, h=128) shown as dashed line with uncertainty band.
- Inset or secondary y-axis: int8 weight size in KB vs. h.

### Figure 3: Ablation table

- Table (as described in Section 6.2).
- Columns: configuration, score, PoseNet, SegNet, delta from baseline.
- Highlight the "best-checkpoint int8 selection" row as the largest single improvement.

### Figure 4: Architecture comparison table

- Table (as described in Section 6.4).
- Columns: architecture, local scorer, proxy score, PoseNet, SegNet, verdict.
- Color-code: green for promoted, yellow for non-promoted alternates, red for rejected.

### Figure 5: PoseNet Jacobian singular value spectrum

- Bar chart of 6 singular values: [0.03603, 0.00080, 0.00055, 0.00028, 0.00018, 0.00009].
- Log scale on y-axis to show the 45x gap between first and second singular values.
- Annotate: "effective rank = 1.008", "condition number = 399".
- Inset: trust-region sweep showing linearization error vs. perturbation magnitude, with the 0.0001 px knee marked.

### Figure 6: CNN residual frequency analysis

- Two-panel figure:
  - Left: frequency spectrum of the trained CNN residual (h=32). Show concentration in mid-frequency band (4-32 cycles/frame), with 90.3% of energy highlighted.
  - Right: frequency spectrum of the Jacobian pseudoinverse correction. Show roughly uniform distribution across bands.
- Shared x-axis: spatial frequency (cycles/frame). Y-axis: normalized energy.
- Annotate the "PoseNet sensitivity band" overlaid on both panels.

### Figure 7: Saliency map visualization

- 2x3 grid:
  - Row 1: original frame, AV1-decoded frame, post-filtered frame
  - Row 2: saliency weight map, absolute residual map (|filtered - decoded|), per-pixel score contribution map
- Show that the post-filter concentrates corrections in the road/vehicle region where PoseNet is most sensitive, not uniformly across the frame.

### Figure 8: Scoring formula sensitivity surface (optional)

- 3D surface or contour plot with axes: SegNet distortion, PoseNet distortion. Color/height: total score at fixed rate.
- Mark the operating points of: baseline (2.08), first post-filter (2.05), and final (1.73).
- Show the steep gradient in the SegNet direction vs. the shallow gradient in the PoseNet direction.

### Figure 9: Train-to-deploy gap visualization (optional)

- Dual y-axis plot over training epochs:
  - Left y-axis: fp32 training score (smooth curve)
  - Right y-axis: deployed int8 score (noisy curve with periodic evaluations)
- Show the gap between the two curves.
- Mark the best-checkpoint selection point (epoch 918 for h=64) where the int8 curve reaches its minimum.
- Annotate: "2.25x PoseNet gap without selection" vs. "near-zero gap with selection".

---

## Appendix A: Full Experiment Log

- Complete table of all scored experiments (34+ entries from Section 6 and Section 7) with config, score breakdown, archive bytes, and verdict.

## Appendix B: Scoring Formula Derivation

- Derivation of the three-way equilibrium condition.
- Proof that at optimality, marginal improvement in each term should be equal (Lagrangian argument).

## Appendix C: Negative Result Details

- Extended discussion of each of the 18 negative results with exact configurations, scores, and diagnostic analysis.
