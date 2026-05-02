# 2. Method

## 2.1 Problem formulation

The challenge video is a single 60-second clip from a Comma EON dashcam (AR0231AT sensor, 1164x874, 20 fps, 1200 frames). Two frozen scorer networks define the distortion:

- **SegNet**: a `smp.Unet('tu-efficientnet_b2', classes=5)` [Tan and Le 2019] trained on comma10k for 5-class road scene segmentation. It processes only the *last frame* in each pair at 384x512, comparing argmax class predictions against ground truth to produce a disagreement rate. The vanilla stride-2 stem loses half-resolution information immediately --- artifacts below (256, 192) are invisible to it.

- **PoseNet**: a FastViT-T12 backbone ego-motion estimator that consumes consecutive frame *pairs* in YUV 4:2:0 format (12 channels: 2 frames x 6 channels at 512x384) and predicts 12-dimensional output, of which the first 6 dimensions $[t_x, t_y, t_z, r_x, r_y, r_z]$ are used. Distortion is L2 distance between poses predicted from compressed vs. original frames.

The scoring formula $S = 100\bar{d}_\text{seg} + \sqrt{10\bar{d}_\text{pose}} + 25r$ has a specific geometric structure. The partial derivative $\partial S / \partial \bar{d}_\text{pose} = \sqrt{10} / (2\sqrt{\bar{d}_\text{pose}})$ diverges as PoseNet loss approaches zero --- diminishing returns. At $\bar{d}_\text{pose} = 0.0005$, a 10% PoseNet improvement is worth the same as a 0.007% SegNet improvement. This means there is a *knee* in the PoseNet curve beyond which further optimization yields negligible score gains. Identifying this knee and reallocating effort to SegNet and rate was a turning point in our strategy.

A second structural observation: SegNet evaluates only even-indexed frames (frame $t+1$ in each consecutive pair). The 600 odd-indexed frames are invisible to SegNet and can be optimized purely for PoseNet fidelity and compressibility.

## 2.2 Architecture: asymmetric warp renderer

The renderer is a conditional generative model that produces frame *pairs* from compact semantic masks. The architecture follows the asymmetric warp paradigm identified through reverse-engineering of the leading submission [Quantizr 2026]:

$$\text{frame}_2 = G(m_2) \qquad \text{frame}_1 = W(\text{frame}_2, F(m_1, m_2)) + \sigma(g) \odot R(m_1, m_2)$$

where $G$ is the generator (CLADE U-Net with SPADE normalization), $F$ predicts optical flow, $\sigma(g)$ is a learned gating map, and $R$ is a residual correction. The second frame is rendered directly from its mask; the first is produced by warping frame 2 according to learned flow, with a gated residual to correct occlusions and disocclusions.

This design is motivated by PoseNet's evaluation protocol. PoseNet measures ego-motion by comparing consecutive frames. If both frames in a pair share a geometric relationship consistent with rigid-body camera motion, PoseNet distortion approaches zero regardless of whether the frames look photorealistic. The warp provides this geometric consistency by construction.

**Parameters.** The full model has 287K parameters. Channel widths are (36, 60) for the U-Net encoder/decoder. The flow predictor outputs 5 channels (2 for flow, 1 for gate, 2 for residual corrections). Masks encode 5 semantic classes as grayscale values at 63-pixel intervals, stored as compressed video.

**Quantization.** Weights are quantized to FP4 using a custom 8-level codebook $\{0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0\}$ with a sign bit, reducing the archive from ~1.1 MB (FP32) to ~150 KB. A separate Int4+LZMA2 scheme achieves 2.18 bits per weight (Section 4.19).

### 2.2.1 CLADE per-class normalization

Each ResBlock in the U-Net uses class-adaptive layer-wise denormalization (CLADE) [Park et al. 2019] rather than plain GroupNorm. After GroupNorm normalizes features, per-class affine parameters $\gamma_c, \beta_c$ are looked up from the segmentation mask and applied as spatially-varying modulation:

$$h_{i,j} = \gamma_{c(i,j)} \cdot \hat{h}_{i,j} + \beta_{c(i,j)}$$

where $c(i,j)$ is the semantic class at pixel $(i,j)$ and $\hat{h}$ is the GroupNorm output. With 5 classes, this adds only $2 \times C$ parameters per normalization layer (where $C$ is the channel width), but provides class-specific feature statistics --- the network learns different appearance distributions for road, sky, lane markings, vehicles, and background.

This is a competitive differentiator. Quantizr's renderer uses plain GroupNorm; szabolcs-cs uses no conditioning at all. CLADE gives the renderer direct knowledge of *which class* each pixel belongs to at every layer of the network, enabling it to specialize texture generation per class. The mask does double duty: it provides the spatial layout (through the embedding input) and the normalization statistics (through CLADE).

### 2.2.2 Dilated residual blocks

The ResBlock cascade uses a symmetric dilation pattern $[1, 2, 4, 1]$ that expands the receptive field from $7 \times 7$ (standard $3 \times 3$ convolutions) to $31 \times 31$ at zero additional parameter cost [Yu and Koltun 2016]. This was validated independently by kaileh57's PR#58 ablation on the same challenge, where dilated convolutions were reported as the "single largest win" (1.92 vs. 2.03).

The expanded receptive field is particularly important for SegNet, whose EfficientNet-B2 backbone has a receptive field of approximately $200 \times 200$ pixels. Without dilation, the renderer's effective view at each layer is smaller than a single SegNet filter --- it cannot reason about the spatial context that SegNet uses to make classification decisions.

### 2.2.3 Replicate padding

All convolutional layers use `padding_mode='replicate'` instead of the default zero padding. This change was prompted by Yousfi's comment on PR#55 (2026-04-19): "artifacts at the image boundary, due to padding in the conv layers." Zero padding introduces artificial edge statistics that SegNet's learned filters can detect --- effectively a steganographic signature at the frame boundary. Replicate padding extends the edge pixels outward, producing more natural boundary statistics.

This is a direct application of Fridrich's principle: any systematic pattern introduced by the embedding process (here, frame generation) is a potential signal for the detector (here, SegNet). Zero-padded boundaries are such a pattern.

### 2.2.4 Radial zoom warp: the rank-1 PoseNet Jacobian

The standard motion model uses a learned `MotionPredictor` network (~50K parameters) to predict per-pixel optical flow from consecutive mask pairs. We discovered that this is dramatically overparameterized for the actual task.

Computing the output Jacobian $J = \partial \text{pose} / \partial \text{pixels}$ at 5 evenly-spaced pairs across the video reveals that PoseNet's 6-dimensional output has **effective rank 1.008** (mean across pairs; range 1.003--1.013). The first singular value captures 99.8% of total variance; the condition number exceeds 350. One degree of freedom is doing all the work.

The geometric interpretation is immediate. The challenge video is a forward-facing dashcam on a highway. The dominant ego-motion is forward translation, which produces a radial expansion pattern centered at the focus of expansion (FoE). The FoE position is determined by the camera calibration of the comma EON's AR0231AT sensor: pixel $(256, 174)$ at the scorer's $384 \times 512$ resolution. The warp for pair $i$ is

$$\text{grid} = \text{foe} + e^{s_i} \cdot (\text{coord\_grid} - \text{foe})$$

where $s_i$ is a single learned scalar per pair, bounded by $|s_i| \leq 0.15$ (approximately $\pm 15\%$ zoom, sufficient for highway speeds). The 600 scalars replace the 50K-parameter `MotionPredictor` at 1.2 KB (FP16) or 300 bytes (FP4).

Validated empirically: radial zoom scalars achieve a 10.6x PoseNet improvement over the baseline renderer, comparable to the full `MotionPredictor`. The 83x parameter reduction (50K to 600) translates directly to archive savings.

This finding generalizes beyond the challenge. Any ego-motion estimator operating on forward-facing video from a calibrated camera will have a near-rank-1 Jacobian, because forward translation dominates the motion field. The radial zoom is the sufficient statistic.

## 2.3 Training: Lagrangian annealing with Fridrich constraints

Standard multi-objective training minimizes a weighted sum of losses. We found this unstable: PoseNet and SegNet gradients frequently conflict, and static weights cannot track the shifting Pareto frontier during training.

Instead, we adopt a constrained formulation inspired by Fridrich's steganalysis framework [Fridrich 2009]:

$$\min_\theta \; \mathcal{L}_\text{rate}(\theta) \quad \text{s.t.} \quad \bar{d}_\text{seg}(\theta) < \epsilon_s, \quad \bar{d}_\text{pose}(\theta) < \epsilon_p$$

This is solved via augmented Lagrangian relaxation with penalty parameter $\rho$ and multipliers $\lambda_s, \lambda_p$:

$$\mathcal{L} = \mathcal{L}_\text{rate} + \lambda_s (\bar{d}_\text{seg} - \epsilon_s) + \frac{\rho}{2}(\bar{d}_\text{seg} - \epsilon_s)_+^2 + \lambda_p (\bar{d}_\text{pose} - \epsilon_p) + \frac{\rho}{2}(\bar{d}_\text{pose} - \epsilon_p)_+^2$$

**Lagrangian annealing.** During training, the penalty caps ($\rho_\text{max}$, $\lambda_\text{cap}$) control how aggressively the optimizer enforces constraints. We discovered that temporarily *reducing* these caps allows the optimizer to explore the Pareto frontier more freely, sometimes finding better basins. Our best checkpoint (auth 0.87) came from resuming training with $\lambda_\text{cap}$ reduced from 10,000 to 1,000 --- the model found a 13% better solution in 200 epochs, though it drifted under the weaker caps over subsequent thousands of epochs. The practical protocol: anneal caps briefly, snapshot the transient improvement, re-tighten or discard late epochs.

### 2.3.1 Fridrich inverse steganalysis losses

Beyond the Lagrangian structure, three loss terms are drawn directly from the steganalysis literature:

**UNIWARD texture weighting** [Holub and Fridrich 2014]. Errors in textured regions are harder for CNN detectors to identify than errors in smooth regions. We compute a local variance map (sliding $5 \times 5$ window) as a proxy for texture complexity, then weight per-pixel L1 loss inversely: smooth regions receive $3\times$ weight, textured regions $0.3\times$. This is the spatial-domain analog of UNIWARD's wavelet directional filter bank distortion cost --- the renderer learns to hide its approximation errors where SegNet is least sensitive.

**$L^\infty$ penalty (square root law)** [Ker, Filler, and Fridrich 2008]. Fridrich's square root law shows that spreading many small errors is fundamentally less detectable than concentrating large ones. We add a penalty on the top-1% pixel errors (soft top-$k$ for gradient stability), encouraging the renderer to distribute its reconstruction error evenly rather than accepting a few badly-rendered pixels. This directly mirrors the capacity formula for steganographic security.

**Markov chain loss** [Filler, Judas, and Fridrich 2010]. HUGO showed that preserving local pixel dependency statistics makes modifications undetectable. We penalize the L1 difference between horizontal and vertical gradients of predicted vs. target frames --- a first-order approximation of the Markov chain transition probabilities. This encourages the renderer to preserve the local spatial structure (edges, gradients, texture orientation) that SegNet's convolutional filters detect.

These three losses are not ad-hoc regularizers. They are the spatial-domain translations of techniques that Fridrich and collaborators developed over two decades to make steganographic modifications invisible to CNN-based detectors. In our setting, the "modifications" are the renderer's approximation errors and the "detector" is SegNet.

### 2.3.2 Freeze/unfreeze training

The training schedule is adapted from Quantizr's 5-stage pipeline, discovered through binary reverse-engineering of his PR#55 submission archive:

| Phase | Epochs | Frozen | Loss focus | Adapted from |
|-------|--------|--------|------------|-------------|
| 1. Pixel warmup | ~400 | None | L1 pixel loss | Quantizr ANCHOR |
| 2. Anchor | ~300 | Motion predictor | SegNet CE + error_boost 9x | Quantizr ANCHOR_BOOST |
| 3. Fine-tune | ~300 | None | Full Lagrangian + Fridrich | Quantizr FINETUNE |
| 4. Joint | ~200 | Embedding | Joint SegNet + PoseNet | Quantizr JOINT |
| 5. Polish | ~100 | Motion + embedding | Low LR, error_boost 49x | Quantizr MICRO |

The key insight behind freeze/unfreeze is that SegNet and PoseNet optimization are approximately orthogonal in this architecture. SegNet evaluates only frame $t+1$ (the directly rendered frame); PoseNet evaluates the pair $(t, t+1)$ where frame $t$ is the warp. Freezing the motion predictor during SegNet-focused phases prevents PoseNet-oriented gradients from interfering with SegNet convergence. The error_boost values (quadratic: $w = 1 + (b-1) \cdot (e/\bar{e})^2$) are from Quantizr's training reconstruction.

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

## 2.7 Frontier prototypes: low-complexity overfitted neural codecs

The April 25 frontier work adds two experimental renderer families motivated by Cool-Chic and C3. These are not promoted results and should not be read as score claims. They are architectural prototypes added behind named profiles so they can be trained, round-tripped, and falsified without disturbing the proven baseline.

**Cool-Chic-style latent renderer.** Cool-Chic compresses a single image or video by overfitting a small neural decoder and compact learned latent representation to that instance. The implemented prototype follows the same low-complexity principle rather than the exact paper bitstream: it replaces the mask renderer with multi-resolution learned latent grids plus a tiny shared `1x1` synthesis decoder. The frame-specific information lives in low-resolution latent tensors; the decoder is shared across all frames. This gives us a controlled test for the hypothesis that the archive should spend bits on a small overfitted representation rather than on a larger convolutional renderer.

The prototype is deliberately constrained:

- It is selected only by the `coolchic_renderer_*` profiles.
- It uses standard static loss, not KL distillation or retired adaptive rebalancing.
- It records seed and deterministic settings in checkpoint metadata.
- Its state dict passes FP4 strict round-trip smoke tests.
- It is not production-ready until archive packaging, inflate compatibility, and authoritative evaluation are demonstrated end to end.

**C3-style residual renderer.** C3 emphasizes high-performance neural compression with low decoder complexity. Our prototype uses C3 as a residual idea rather than as a full replacement codec: a base renderer produces the image and a coordinate-conditioned MLP adds a bounded residual field. The residual head is zero-initialized, so the model starts exactly at the base renderer and can only earn complexity by improving the residual objective. This is important experimentally because it separates "does a coordinate MLP help?" from unrelated initialization noise.

The scientific gate for both prototypes is strict. A local forward pass, parameter count, or proxy loss is not sufficient. Promotion requires:

1. deterministic smoke training from a named profile;
2. eval-roundtrip scoring on decoded outputs, not raw tensors;
3. archive-size accounting with all neural artifacts inside `archive.zip`;
4. inflate-time runtime measurement under the contest budget;
5. proxy-vs-authoritative comparison before any claim of score improvement.

The council interpretation is that Cool-Chic and C3 are high-upside lanes, not replacements for the current floor. They are most likely to matter as rate-distortion experiments for the next training cycle: Cool-Chic as a smaller base representation, and C3 as a residual codec on top of the scorer-aligned renderer.

Initial local smoke testing on April 25 caught two engineering issues before any remote deployment: full-resolution ground-truth frames were not resized before eval-roundtrip scorer loss, and FP4 QAT parametrization buffers were not moved to MPS with the wrapped model. After fixing both, the Cool-Chic and C3 prototype profiles completed 8-frame scorer/QAT smokes and exported FP4 plus int4+LZMA2 artifacts. This is integration evidence only; the tiny-slice scorer value is not a rate-distortion result.

A subsequent 32-frame, 20-epoch trend smoke sharpened the diagnosis. The C3 residual head reduced float/scorer training loss substantially, especially the SegNet term, but the FP4-evaluated checkpoint did not improve. The immediate research question is therefore not whether the residual head can learn, but whether its learned correction can survive the current FP4 quantization and export path. Uniform int4+LZMA2 was much smaller than FP4 in these smoke artifacts, but scorer-sensitive bit allocation is required before treating self-compression as a deployable replacement.

## 2.6 Atomic decomposition framework

The proven baseline + Cool-Chic + C3 lanes converge on a common abstraction we formalize as an *atomic decomposition* of the compression pipeline. Every input element — at every pipeline stage, down to individual pixels in individual frames — is treated as a typed atom `a` with charged byte cost `bytes(a)`, predicted score-component contributions `Δseg(a)` and `Δpose(a)`, and cross-atom interactions (synergistic when `Δscore(a ⊕ b) < Δscore(a) + Δscore(b)`, antagonistic otherwise). Atoms span all granularities from pipeline-level (mask codec choice, renderer architecture) to component-level (per-tensor block bit allocation, per-pose-column delta-VLQ encoding) to pixel-level (Score-Jacobian-weighted reconstruction-target adjustments).

The compression problem becomes combinatorial atom selection over the proposal space subject to a charged-archive-size budget and the contest score. Three solution policies operate on this formalism: atom-waterfill (Boyd-style alternating projections), hard-pair selection (Fridrich-style sensitivity-weighted prioritization), and line-search / coordinate descent (sequential greedy refinement with R(D) joint objective). The promotion gate is a deterministic-archive contest-CUDA auth eval on identical bytes — atoms become evidence only after promotion.

The full formalism, worked example on the C-058 → C-067 micro-frontier chain, and connection to the Yousfi-Fridrich floor (Section 6.6) appear in `docs/paper/methodology_addendum_atomic_decomposition_yf_floor_20260502.md` as a paper companion. The atomic decomposition unifies the meta-Lagrangian framing (codex's submission writeup ledger), the cross-stream coordinator (Joint-ADMM), and the hard-pair temporal window selection (Fridrich-derived) under a single typed-atoms contract.
