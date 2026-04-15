# 2. Method

## 2.1 Problem formulation

The challenge video is a single 60-second clip from a Comma EON dashcam (AR0231AT sensor, 1164x874, 20 fps, 1200 frames). Two frozen scorer networks define the distortion:

- **SegNet**: an EfficientNet-B4 U-Net [Tan and Le 2019] trained on comma10k for 6-class road scene segmentation. It processes each frame independently at 384x512, comparing predicted class logits against ground truth via cross-entropy.

- **PoseNet**: a convolutional ego-motion estimator that consumes consecutive frame *pairs* in YUV 4:2:0 format (6 channels at 192x256) and predicts 6-DOF relative pose $[t_x, t_y, t_z, r_x, r_y, r_z]$. Distortion is L2 distance between poses predicted from compressed vs. original frames.

The scoring formula $S = 100\bar{d}_\text{seg} + \sqrt{10\bar{d}_\text{pose}} + 25r$ has a specific geometric structure. The partial derivative $\partial S / \partial \bar{d}_\text{pose} = \sqrt{10} / (2\sqrt{\bar{d}_\text{pose}})$ diverges as PoseNet loss approaches zero --- diminishing returns. At $\bar{d}_\text{pose} = 0.0005$, a 10% PoseNet improvement is worth the same as a 0.007% SegNet improvement. This means there is a *knee* in the PoseNet curve beyond which further optimization yields negligible score gains. Identifying this knee and reallocating effort to SegNet and rate was a turning point in our strategy.

A second structural observation: SegNet evaluates only even-indexed frames (frame $t+1$ in each consecutive pair). The 600 odd-indexed frames are invisible to SegNet and can be optimized purely for PoseNet fidelity and compressibility.

## 2.2 Architecture: asymmetric warp renderer

The renderer is a conditional generative model that produces frame *pairs* from compact semantic masks. The architecture follows the asymmetric warp paradigm identified through reverse-engineering of the leading submission [Quantizr 2026]:

$$\text{frame}_2 = G(m_2) \qquad \text{frame}_1 = W(\text{frame}_2, F(m_1, m_2)) + \sigma(g) \odot R(m_1, m_2)$$

where $G$ is the generator (CLADE U-Net with SPADE normalization), $F$ predicts optical flow, $\sigma(g)$ is a learned gating map, and $R$ is a residual correction. The second frame is rendered directly from its mask; the first is produced by warping frame 2 according to learned flow, with a gated residual to correct occlusions and disocclusions.

This design is motivated by PoseNet's evaluation protocol. PoseNet measures ego-motion by comparing consecutive frames. If both frames in a pair share a geometric relationship consistent with rigid-body camera motion, PoseNet distortion approaches zero regardless of whether the frames look photorealistic. The warp provides this geometric consistency by construction.

**Parameters.** The full model has 287K parameters. Channel widths are (36, 60) for the U-Net encoder/decoder. The flow predictor outputs 5 channels (2 for flow, 1 for gate, 2 for residual corrections). Masks encode 5 semantic classes as grayscale values at 63-pixel intervals, stored as compressed video.

**Quantization.** Weights are quantized to FP4 using a custom 8-level codebook $\{0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0\}$ with a sign bit, reducing the archive from ~1.1 MB (FP32) to ~150 KB.

## 2.3 Training: Lagrangian annealing with Fridrich constraints

Standard multi-objective training minimizes a weighted sum of losses. We found this unstable: PoseNet and SegNet gradients frequently conflict, and static weights cannot track the shifting Pareto frontier during training.

Instead, we adopt a constrained formulation inspired by Fridrich's steganalysis framework [Fridrich 2009]:

$$\min_\theta \; \mathcal{L}_\text{rate}(\theta) \quad \text{s.t.} \quad \bar{d}_\text{seg}(\theta) < \epsilon_s, \quad \bar{d}_\text{pose}(\theta) < \epsilon_p$$

This is solved via augmented Lagrangian relaxation with penalty parameter $\rho$ and multipliers $\lambda_s, \lambda_p$:

$$\mathcal{L} = \mathcal{L}_\text{rate} + \lambda_s (\bar{d}_\text{seg} - \epsilon_s) + \frac{\rho}{2}(\bar{d}_\text{seg} - \epsilon_s)_+^2 + \lambda_p (\bar{d}_\text{pose} - \epsilon_p) + \frac{\rho}{2}(\bar{d}_\text{pose} - \epsilon_p)_+^2$$

**Lagrangian annealing.** During training, the penalty caps ($\rho_\text{max}$, $\lambda_\text{cap}$) control how aggressively the optimizer enforces constraints. We discovered that temporarily *reducing* these caps allows the optimizer to explore the Pareto frontier more freely, sometimes finding better basins. Our best checkpoint (auth 0.87) came from resuming training with $\lambda_\text{cap}$ reduced from 10,000 to 1,000 --- the model found a 13% better solution in 200 epochs, though it drifted under the weaker caps over subsequent thousands of epochs. The practical protocol: anneal caps briefly, snapshot the transient improvement, re-tighten or discard late epochs.

## 2.4 Test-time optimization: coupled trajectory

At decompression time, we run gradient-based optimization directly on the generated frames. This is conceptually similar to test-time training [Sun et al. 2020] but operates on the output pixels rather than the model parameters.

For each batch of consecutive frame pairs, we optimize:

$$\min_{\Delta} \; w_s \cdot \bar{d}_\text{seg}(x + \Delta) + w_p \cdot \bar{d}_\text{pose}(x + \Delta) + w_c \cdot \|\Delta\|_2$$

where $x$ are the renderer outputs and $\Delta$ is a learned perturbation. The key is the *coupled trajectory*: PoseNet evaluates pairs $(x_{t}, x_{t+1})$, so the gradient of $\bar{d}_\text{pose}$ flows through both frames simultaneously. This is analogous to 4D-Var data assimilation [Courtier et al. 1994], where observations at multiple time steps constrain the state trajectory jointly.

**Configuration.** Learning rate 0.005, 500 steps per batch, weights $w_s = 100$, $w_p = 10$, $w_c = 0.5$, patience 150 (early stopping if no improvement). SegNet loss is computed on even-indexed frames only (`seg_odd_only=True`), exploiting the scorer's frame selection. Anti-aliasing regularization at weight 0.2 prevents high-frequency artifacts.

## 2.5 The gradient fix: differentiable BT.601 YUV conversion

PoseNet's preprocessing converts RGB frames to YUV 4:2:0 via the function `rgb_to_yuv6` in the upstream scorer code. This function was decorated with `@torch.no_grad()`, which detaches the autograd graph at the color-space conversion boundary. Any gradient flowing backward through PoseNet hits this wall and becomes zero.

During training, our pipeline applied a differentiable patch (`_patch_scorers_for_training`) that replaced the preprocessing function. But the TTO pipeline loaded scorers through a different code path that did not apply the patch. Two code paths, one patched, one not.

The fix replaces the upstream YUV conversion with a differentiable implementation of the BT.601 matrix transform:

$$\begin{bmatrix} Y \\ U \\ V \end{bmatrix} = \begin{bmatrix} 0.299 & 0.587 & 0.114 \\ -0.169 & -0.331 & 0.500 \\ 0.500 & -0.419 & -0.081 \end{bmatrix} \begin{bmatrix} R \\ G \\ B \end{bmatrix} + \begin{bmatrix} 0 \\ 0.5 \\ 0.5 \end{bmatrix}$$

This is applied through `make_scorers_differentiable()`, which patches the preprocessing function on all scorer instances. A runtime gradient validation check (~1ms) verifies that gradients flow through PoseNet before any optimization loop begins.

The impact: PoseNet distortion dropped from 0.012 (blind TTO) to 0.00209 (gradient-aware TTO), an 8.2x improvement. Overall score went from 0.70 to 0.43.

## 2.6 Score formula analysis

Several properties of the scoring formula shaped our optimization strategy:

**PoseNet diminishing returns.** The sqrt on PoseNet means the marginal value of PoseNet improvement decreases as $1/\sqrt{\bar{d}_\text{pose}}$. Below $\bar{d}_\text{pose} \approx 0.002$, further PoseNet work yields less than equivalent effort on SegNet or rate.

**SegNet odd-frame blindness.** SegNet processes only the even-indexed frame in each pair. Odd frames contribute to PoseNet and rate but not SegNet. This creates 600 frames with one fewer constraint, which can be made smoother for better compressibility.

**Rate is cheap.** At our archive size (~150 KB), the rate contribution is 0.10. Halving the archive saves only 0.05. Below ~100 KB, rate optimization has negligible marginal value compared to scorer distortion reduction.

**The detection boundary.** Fridrich's formulation reframes the problem: instead of minimizing a weighted sum, find the *detection boundary* --- the scorer distortion level below which further reduction yields negligible score improvement --- and minimize rate while staying below it. This is more natural than tuning loss weights and converges faster in practice.
