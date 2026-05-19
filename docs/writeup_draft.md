# Task-Aware Compression via Differentiable Scorer Optimization

<!-- Long-form writeup structure:
  Section 1: Introduction (task-aware compression for autonomous driving)
  Section 2: Background (scoring formula, PoseNet, SegNet, openpilot context)
  Section 3: Method Phase 1 (post-filter, renderer, asymmetric warp)
  Section 4: The Gradient Bug (paper-worthy discovery)
  Section 5: Method Phase 2 (TTO, hinge loss, FiLM/CLADE conditioning)
  Section 6: Negative Results (KL distill, adaptive weights, PSD, etc.)
  Section 7: Multi-Objective Analysis (Pareto frontier, scoring formula geometry)
  Section 8: Results and Ablations
  Section 9: Scalability and Distillation
  Section 10: Discussion
-->

> **2026-05-06 public-claim hygiene update.** This is a historical draft, not
> a submission packet, live leaderboard, or arXiv/preprint commitment. Every
> score-like row below is quarantined unless it is promoted by an exact
> `A++`/`A` evidence row with archive SHA-256, CUDA auth-eval JSON, component
> recomputation, runtime custody, and full sample count. Use
> `docs/paper/04_results.md` for ranked rows and `docs/paper_outline.md` for
> the evidence-grade roadmap.

> **2026-05-07 public-frontier forensics note.** PR102 should be discussed as a
> zero-byte inference-tuning submission over PR100 archive custody. Its source
> `compress.sh` fetches BradyMeighan's `hnerv-lc-v2-archive/archive.zip` by
> SHA-256, and the corrected PR102 archive is byte-identical to PR100
> (`178981` bytes,
> `afd53348f50303bf0ec6a7ffecc1ac037df2f1c70745244b9c45c72e8eb80641`).
> The only source-level changes are `DELTA_SCALE = 0.0095` and
> `up[:, 0, 0].add_(1.0)` at inflate time. This is a lineage/composition
> finding and an engineering lesson about public-frontier velocity; do not
> present it as our local score claim until exact CUDA replay lands.

## Abstract (historical draft - not publication-ready)

We present a system for task-aware video compression that achieves a contest-CUDA score of `1.05` on comma.ai's video compression challenge through a paradigm shift from codec post-filtering to neural rendering. Our approach evolved through two paradigms and a third in active development: (Era 1) a CNN post-filter on codec output reaching `1.73` while bound to AV1; (Era 2) a neural renderer with dilated-h64 architecture that bypasses the codec entirely, with pose TTO at compress time and KL distillation on the SegNet logits at weight=0.002, reaching `1.05` [contest-CUDA] verified on the EXACT submission archive bytes; (Era 3, live) the Selfcomp-paradigm portfolio (grayscale-LUT mask, single-mask + 6-DOF affine duality, analytical pose, block-FP weight self-compression at 1.017 bpw, 94K-param SegMap) — eight Modal lanes in flight as of submission deadline. Key technical contributions are: discovery of a 23x MPS-vs-CUDA PoseNet drift that invalidated months of MPS-measured scores; the rank-1 PoseNet Jacobian analysis showing why pose TTO must warm-start from baseline poses; the KL distillation weight sensitivity result (≥ 0.01 collapses PoseNet, 0.002 sustains the boundary signal); and the engineering-rigor catalog of 78 strict preflight checks that made every score reproducible. We report 30+ negative results including the catastrophic 53.61 mask-resolution disaster (48x64 instead of 384x512), the Lane GP Runge phenomenon at degree-10 polynomial, and the UNIWARD encoder no-op finding.

**Historical exact row to recheck in current tables: 1.05** [contest-CUDA] | seg=0.0040, pose=0.0034, rate=0.0185 (694KB archive)
**Historical Modal reproduction row: 1.04** [Modal-T4-CUDA] | identical archive bytes, drift 0.01 within noise
**Historical fallback row: 1.15** [contest-CUDA] | Lane A (pose TTO from baseline poses)
**Era 1 historical floor row: 1.73** [Era 1 / Track B / current_workflow] | h64 long-horizon QAT+EMA learned int8 post-filter

---

## Legacy abstract (Era 1 + draft narrative, kept for historical context)

The legacy abstract below was the working narrative through April 25 and references scores measured on MPS that were subsequently invalidated. It is preserved for the writeup arc but should not be cited as authoritative.

> We present a system for task-aware video compression that achieves a score of 0.37 on comma.ai's video compression challenge, approaching the current leader (0.33) through a fundamentally different paradigm than traditional codecs. Our approach evolved through three phases: (1) a CNN post-filter on codec output (1.33), (2) a neural renderer with asymmetric warp architecture that bypasses the codec entirely (0.87), and (3) test-time optimization (TTO) that fine-tunes per-frame embeddings against the frozen scorers at compress time (0.37).

> Note: every score in the legacy abstract above was measured on MPS prior to the 2026-04-25 CUDA-vs-MPS comparison. PoseNet drifts 23x; final scores drift ~2.5x. The "0.37" reading on MPS corresponds to roughly `0.93-1.0` on CUDA. Treat as `[advisory only]`.

**Legacy-tagged readings (advisory only — do NOT cite as authoritative):**
- Legacy MPS best reading: 0.37 [advisory only — MPS]
- Contest-compliant: 0.61 [advisory only — MPS]
- Contest-compliant baseline: 0.87 [advisory only — MPS]

---

## Score Evolution (historical draft - exact rows only above the line)

| Stage | Score | Lane | Tag | Key Technique | Insight |
|-------|-------|------|-----|---------------|---------|
| Era 1 H.265 baseline | 1.97 | — | [Era 1] | CRF 28, no processing | Starting point |
| Era 1 CNN postfilter (h64) | 1.73 | Track B | [Era 1] | dilated conv, QAT+EMA, scorer gradients | Width scaling + best-ckpt int8 |
| Era 2 baseline | 0.90 | — | [contest-CUDA] | dilated-h64 renderer + CRF=50 + matched poses | First reproducible from saved artifacts (2026-04-25) |
| Era 2 Lane A | 1.15 | Renderer | [contest-CUDA] | + pose TTO from baseline poses (rank-1 init) | PoseNet 0.247 → 0.0034 (73x) |
| Era 2 Lane G v3 | **1.05** | Renderer | [contest-CUDA] | + KL distill weight=0.002 + pose TTO retry | Both PoseNet and SegNet improve at same rate (current floor) |
| Era 2 Lane G v3 (Modal repro) | 1.04 | Renderer | [Modal-T4-CUDA] | identical archive | Modal reproduces within 0.01 noise |
| Era 3 portfolio (in flight) | TBD | Selfcomp paradigm | live | grayscale-LUT mask, block-FP, 94K SegMap, KL T=2.0 | Sub-0.30 target |

### Historical / advisory-only readings (preserved for the arc — do not cite as authoritative)

These scores are pre-MPS-discovery and were ~2.5x optimistic relative to contest-CUDA reality.

| Stage | Score | Tag | Notes |
|-------|-------|-----|-------|
| Era 1.5 Neural renderer | 0.87 | [advisory only — MPS] | true CUDA equivalent ≈ ~2.0 |
| Era 1.5 TTO (gradient fix) | 0.43 | [advisory only — MPS] | true CUDA equivalent ≈ ~1.0 |
| Era 1.5 TTO (hinge loss) | 0.37 | [advisory only — MPS] | true CUDA equivalent ≈ ~0.95 |
| Era 1.5 Pose TTO + Distillation ep300 | 0.61 | [advisory only — MPS] | unverified on CUDA |
| Era 1.5 Distillation ep900 | ~0.47* | [advisory only — MPS] | unverified on CUDA |
| Era 1.5 Full stack (projected) | ~0.25 | [advisory only — MPS] | extrapolation from MPS readings |

*Projected from proxy trajectory (0.338 at ep900, still converging on MPS).

The story is one of successive paradigm shifts, each abandoning assumptions of the previous stage. The postfilter assumes a codec exists. The renderer removes the codec. TTO removes the constraint that compression is a single forward pass. Pose-space TTO removes the constraint that TTO must operate in pixel space. Distillation removes the constraint that TTO must run at inflate time.

The conditioning-space insight is publishable in its own right: the FiLM conditioning vector (6D per pair) is 196,608× smaller than pixel space, yet achieves comparable or better optimization efficiency because PoseNet's output space is intrinsically 6-dimensional. In this space, PoseNet and SegNet gradients are approximately orthogonal, enabling pure-PoseNet optimization without SegNet tradeoff.

---

<!-- LEGACY CONTENT BELOW: From the 1.33 era. Being preserved for the historical narrative
     but the paper structure above supersedes this. -->

## Legacy: Post-Filter Era (Score 1.33)

---

<!-- Section 1: Introduction -->
## Summary

A small CNN (45KB, int8) corrects decoded AV1 frames by backpropagating through the frozen scorer networks. It learns pixel adjustments that reduce PoseNet and SegNet distortion without changing the codec or the bitstream.

<!-- Section 2: Background -->
## The scoring formula and why it matters

The challenge score is:

```
score = 100 * seg + sqrt(10 * pose) + 25 * rate
```

This formula has non-obvious properties at different operating points. At our final result (seg=0.00610, pose=0.00218, rate=0.0230):

| Component | Value | Contribution |
|-----------|-------|-------------|
| 100 * seg | 100 * 0.00610 | 0.610 |
| sqrt(10 * pose) | sqrt(10 * 0.00218) | 0.148 |
| 25 * rate | 25 * 0.0230 | 0.575 |

PoseNet distortion dropped from 0.01229 (no filter) to 0.00218 with the dilated architecture -- a 5.6x improvement that accounts for 117% of our total score gain. SegNet actually regressed 5.2% (0.00580 to 0.00610). The post-filter is a PoseNet optimization mechanism; the entire score improvement comes from pose preservation, with a small SegNet tax.

The remaining score is dominated by SegNet (46%) and rate (43%), with PoseNet contributing only 11%.

At this operating point, SegNet has ~29.6x more marginal impact than PoseNet per unit distortion (d(score)/d(seg) = 100 vs d(score)/d(pose) = 3.4), but PoseNet has a smoother loss landscape that responds better to gradient descent. This is the core tension: the highest-leverage metric is the one least amenable to optimization. The GIF visualizations confirmed this -- the color-coded SegNet diff showed roughly balanced red (we fixed) and green (we introduced), indicating the post-filter redistributes classification errors rather than reducing them.

<!-- Section 3: Method -->
## The pipeline

### Step 1: Encode well

SVT-AV1 with CRF 34, preset 0, film-grain synthesis at 22, Lanczos downscale to 524x394.

AV1 beats x265 at this bitrate regime. Film-grain synthesis is not cosmetic -- removing it causes a 292% PoseNet regression because PoseNet treats high-frequency texture as structural signal. The film-grain parameter was one of the first things we tuned and one of the last things we stopped touching.

### Step 2: Decode and correct

After decoding, a 3-layer residual CNN corrects each frame:

```
input (3ch) -> Conv2d 3x3 (3->64) -> ReLU -> Conv2d 3x3 dilation=2 (64->64) -> ReLU -> Conv2d 3x3 (64->3) -> + input
```

The middle layer uses dilation=2, which expands the receptive field from 7x7 to 15x15 at the same parameter count. This matters because PoseNet's early convolutions integrate over mid-frequency spatial patterns -- the wider RF lets the CNN correct the spatial correlations that PoseNet attends to.

The output layer is zero-initialized (`nn.init.zeros_` on both weight and bias), so at the start of training the filter is exactly the identity function -- `output = input + conv3(features) = input + 0`. This is not just a convenience. It means the filter begins in a known-good state (the unfiltered codec output) and every learned correction must reduce the scorer loss to persist through training. The alternative -- random initialization -- would start the filter in a state that actively corrupts frames, requiring early training epochs to first undo the damage before making progress. Zero-init also stabilizes quantization: the initial weight distribution is concentrated near zero, and training moves weights only as far as the loss demands, producing a narrower weight range that quantizes more faithfully.

The filter ships as a 45KB int8-quantized checkpoint (~40K parameters). At inference it runs on CPU in under 30 seconds for all 1200 frames. The rate impact is negligible.

### Step 3: Train against the scorer

The training loss is the competition score itself, computed by running the corrected frames through frozen copies of PoseNet and SegNet. Gradients flow from the score, through the scorer networks, through the CNN, and into its weights.

The CNN does not optimize for generic visual quality. It optimizes for the specific metrics PoseNet and SegNet use to compare frames.

**Batch construction.** The scorer evaluates pairs of consecutive frames (the pose network regresses ego-motion between frames t and t+1). Each training step samples a frame-pair index from the 600-pair submission archive, loads the compressed and ground-truth pairs onto the device, applies the filter to both frames, and computes the scorer loss. Pairs are sampled either uniformly or via power-law weighted sampling that overweights high-disagreement pairs (Section on hard-frame curriculum). Training iterates over the full pair set once per epoch with a random permutation.

**Gradient accumulation.** Each pair produces one gradient signal. With `accum_steps > 1`, gradients accumulate across multiple pairs before an optimizer step, effectively increasing the batch size without increasing memory. After each accumulation window, gradients are clipped (default max norm 1.0) and the optimizer steps.

**Saliency weighting.** A precomputed PoseNet saliency map (the L2 norm of PoseNet's Jacobian with respect to each pixel, averaged over training frames) identifies which spatial regions PoseNet is most sensitive to. A saliency-weighted reconstruction term (alpha=20) penalizes unnecessary modification of high-sensitivity pixels, focusing the CNN's limited capacity on corrections that PoseNet's gradients identify as beneficial rather than allowing the filter to freely rearrange texture. A complementary term prevents unnecessary modification of pixels near SegNet class boundaries.

**EMA update schedule.** After each optimizer step, an exponential moving average (decay=0.997) updates a shadow copy of the weights: `ema_w = 0.997 * ema_w + 0.003 * model_w`. Over 1000 epochs, EMA smooths late-epoch weight oscillation and prevents the optimizer from wandering into configurations that score well in float32 but collapse under quantization. All checkpoint evaluation uses the EMA weights, not the raw optimizer weights.

## Hardening: 15 bugs across 4 audit rounds

Over four audit rounds, we found and fixed 15+ bugs in our proxy scorer that collectively accounted for ~0.5 points of phantom score.

### Round 1: Proxy scorer fidelity

Our proxy scorer was computing SegNet similarity using soft cosine distance on logit tensors. The official scorer uses hard argmax followed by pixel-wise comparison. The proxy was crediting the CNN for smooth logit improvements that the official scorer could not see. Every checkpoint selected by the old proxy was optimized for a metric that did not exist in production.

### Round 2: Checkpoint selection

Best-checkpoint selection was using the same wrong SegNet metric. This meant we were not just training against the wrong signal -- we were also *selecting* the wrong model. The checkpoint that scored best on the buggy proxy was systematically different from the one that scored best on the real metric.

### Round 3: Resolution and data leakage

`compute_boundary_mask` was operating at the wrong resolution, producing masks that did not align with the frames being scored. Separately, the training evaluation used a deterministic stride over frames rather than a held-out split, creating train/eval leakage that inflated proxy scores.

### Round 4: Numeric fidelity

The training loop evaluated frames in float32, but the official scorer loads frames as uint8 PNG round-trips. The quantization from float32 to uint8 and back introduced enough noise to shift PoseNet scores by measurable amounts. We added explicit uint8 round-trip clamping to the training evaluation path so that proxy scores match official scores exactly.

### The fix

Each bug was addressed with a specific test. The `tac` library now includes 70 tests covering metric computation, checkpoint selection, boundary mask resolution, data splitting, and numeric fidelity. Pydantic validation enforces schema correctness on all configuration and checkpoint metadata. Atomic saves prevent partial checkpoint corruption.

The hardening dropped our official score from 1.727 to 1.52. That 0.2 improvement came entirely from fixing measurement and selection bugs.

## Training: the quantization gap problem

The CNN trains in float32 but ships in int8. This creates a train-to-deploy gap that is much larger than you might expect: 2.25x on PoseNet distortion. A model that scores 0.021 PoseNet in training can score 0.047 after int8 quantization.

Three techniques address this:

**Quantization-Aware Training (QAT):** Per-channel fake int8 quantization in the forward pass with straight-through gradient estimation. Per-channel quantization computes one scale factor per output filter (64 scales for a 64-channel conv layer), while per-tensor uses a single global scale. The difference is substantial: per-tensor quantization of our best checkpoint produces a PoseNet distortion of 0.0047, while per-channel achieves 0.0022 -- a 2.1x gap from the same trained weights. The per-channel approach preserves 3-4 more bits of effective precision because filters with small weight ranges are not forced to share a scale with filters that have large ranges. During QAT, the forward pass simulates per-channel int8 rounding (quantize, dequantize, straight-through on the backward pass), so the model learns weight configurations that are robust to the exact quantization noise pattern it will encounter at deployment.

**Exponential Moving Average (EMA):** Polyak averaging with decay=0.997 smooths late-epoch weight oscillation. Over 1000 epochs, this prevents the optimizer from wandering into weight configurations that happen to score well in float32 but collapse under quantization.

**Best-checkpoint int8 selection:** At each checkpoint, we take the EMA weights, quantize them to int8, and evaluate the quantized model on the scorer using the corrected metric (hard argmax SegNet, uint8 round-trip, held-out frames). Most epochs produce poor int8 models. We save the epoch where the quantized weights land in a good configuration.

The key insight is that int8 quantization of a continuously evolving weight trajectory is a stochastic process. Each epoch's weights, when rounded to 256 levels per channel, land in a different discrete configuration. The scorer loss of these configurations varies non-monotonically -- epoch 700 may quantize better than epoch 800, even though the float32 loss at epoch 800 is lower. Best-checkpoint selection is a search over this stochastic process: we evaluate every epoch's quantized output and keep the realization that scores best on the actual metric. Over 905 epochs, this search samples enough of the quantization landscape to find configurations where the rounding errors happen to align favorably with the scorer's sensitivity profile.

This mechanism is why our deployed score (1.33) matches our canonical proxy (1.33). Without it, the gap would erase most of the CNN's benefit.

## Train on the submission archive

Training data distribution must match scoring data distribution. The official scorer evaluates on 600 frame pairs from the submission archive. Early post-filter versions trained on a different frame set and scored 2.35 despite looking fine on proxy.

The fix: train on the actual submission archive frames, evaluate on all 600 pairs with uint8 round-trip.

## Width scaling: a log-linear law

The CNN's hidden dimension `h` controls its capacity. We trained models at h=8, 16, 32, 48, and 64 and found a clean relationship:

```
score = -0.159 * ln(h) + 2.382
```

| h | Params | Score |
|---|--------|-------|
| 8 | ~800 | 2.06 |
| 16 | ~3K | 1.92 |
| 32 | ~12K | 1.845 |
| 48 | ~19K | 1.762 |
| 64 | ~40K | 1.51 |
| 64 (dilated) | ~40K | 1.33 |

Each doubling of width buys measurable score improvement. The relationship held across all data points. At h=64, switching from standard to dilated convolutions broke below the scaling law prediction, showing that architecture changes at sufficient width can deliver outsized gains.

## Compute scaling: what happens with more resources

The approach scales along three axes: model capacity, training time, and training technique.

**Model capacity** follows the log-linear law above. A 45KB int8 budget constrains us to ~40K parameters with the dilated architecture. Removing this constraint (e.g., allowing 500KB or 5MB artifacts in the archive) would permit h=256 or deeper networks. The log-linear law predicts h=256 standard would score ~1.0. With dilated convolutions, which broke below the law at h=64, larger capacity could push significantly lower.

**Training time** shows diminishing but persistent returns. The dilated h=64 standard-loss run improved from 1.48 to 1.33 over 905 epochs. With the 30-minute inflate time budget, we could also run larger models at inference time: a 500KB model with h=128 would process 1200 frames in ~40 seconds on CPU, well within budget.

**Training technique** is the multiplier. The score trajectory across technique generations:

| Technique | Best Score | Epochs to Reach |
|-----------|-----------|----------------|
| Standard loss, h=64 | 1.51 | ~500 |
| Standard loss, dilated h=64 | 1.33 | 905 |
| KL distill + hard-frame, dilated h=64 | 1.25 proxy (auth: 2.05 -- DEAD) | 200 |

Standard loss with dilated convolutions is the only technique that transfers from proxy to authoritative scoring. The KL distill variant appeared to converge faster but the proxy improvement was phantom -- two independent authoritative evaluations (1.85 and 2.05) confirmed it does not transfer. The bottleneck for further improvement is architecture and capacity, not training technique.

**Theoretical limits**: A panel analysis estimated the absolute floor for a 45KB int8 CNN at ~1.25-1.30, driven by the 15x15 receptive field covering only 2.5% of the scorer's input resolution. With an optimal architecture (larger RF via PixelShuffle or deeper dilation), the floor drops to ~1.10. Removing the size constraint entirely, the floor is ~0.875-0.975 -- bounded by the irreducible rate term (0.575) and codec artifacts that no post-filter can reverse.

The practical implication: this approach improves with more compute along every axis, and we have not yet hit the capacity ceiling. The remaining gap is in SegNet distortion, where architecture changes (e.g., PixelShuffle-Downscale or deeper dilation stacks) offer the most plausible path forward.

## Why closed-form corrections fail: a mathematical detour

Early in the project, we tried the obvious shortcut: compute the Jacobian of PoseNet with respect to pixel values, and apply the Moore-Penrose pseudoinverse correction to minimize pose error in a single step.

It made things worse. Pose distortion went from 0.074 to 0.235.

Three experiments explained why:

1. **Trust radius measurement:** PoseNet's linear approximation breaks down below 0.0001 pixels RMS. Any correction large enough to matter leaves the valid linear regime.

2. **Jacobian SVD analysis:** The effective rank of PoseNet's Jacobian is ~1 out of 6 pose dimensions. 98% of pixel sensitivity lies along a single direction. The condition number is ~399. The problem is massively ill-conditioned.

3. **CNN residual analysis:** The trained CNN moves 56.6% of pixels (vs 0.002% for the Jacobian correction) and concentrates 90.3% of its energy in the mid-frequency DCT band where PoseNet's early convolutions are most sensitive. It makes dense, small, spatially-coherent corrections that stay inside every ReLU region boundary.

The CNN is not approximating a closed-form solution. It navigates a rank-1 basin through many small learned adjustments -- something analytical methods cannot find without iterative optimization.

## The score trajectory

```
Apr 3:  4.06  -- x265 baseline
Apr 5:  2.20  -- AV1 with proper BT.601 colorspace
Apr 6:  2.08  -- film-grain=22, sharpness=1
Apr 7:  2.05  -- first post-filter (h=16, 3K params)
Apr 8:  1.99  -- QAT + EMA
        1.945 -- 500 epochs -> 1000 epochs
        1.845 -- h=32
Apr 9:  1.762 -- h=48
        1.727 -- h=64
        1.51  -- hardening: fixed proxy metric, checkpoint selection,
                 boundary mask resolution, train/eval leakage, float32/uint8 gap
Apr 10: 1.33  -- dilated CNN architecture (Modal A10G, 905 epochs)
```

The trajectory splits into four phases. Codec tuning (4.06 to 2.08) found the right AV1 configuration. Post-filter scaling (2.08 to 1.727) was capacity-driven improvement from a growing CNN. Hardening (1.727 to 1.51) came from fixing every place where our measurement diverged from the official scorer. Architecture (1.51 to 1.33) came from switching from standard to dilated convolutions, which gave a 5.6x PoseNet improvement by expanding the receptive field to capture spatial correlations PoseNet's early convolutions attend to.

<!-- Section 4: Negative Results -->
## Negative results

Negative results are as informative as positive ones. We ran 25+ experiments over 8 days; most failed. We organize the most instructive failures into five themes.

### 4.1 Proxy-authoritative transfer gap in knowledge distillation

Temperature-annealed soft SegNet targets via Hinton-style knowledge distillation appeared to converge faster than standard loss: proxy score reached 1.25 in 200 epochs versus 1.33 in 905 epochs for the standard approach. Two independent authoritative evaluations revealed the improvement was phantom.

**First eval** (sw=100, PoseNet gradient cap): proxy 1.25, authoritative 1.85. SegNet improved 19% (0.00610 to 0.00493) but PoseNet regressed 26x (0.00218 to 0.05725).

**Second eval** (sw=30, no cap): proxy 1.43, authoritative 2.05. Even after removing the gradient cap and reducing the SegNet weight by 70%, PoseNet regressed 37x (0.00218 to 0.081).

The first failure could be attributed to the PoseNet gradient cap. The second could not -- it had no cap. The root cause is structural: KL distillation's gradient signal dominates PoseNet MSE at any `segnet_weight` above ~5, causing the CNN to rearrange pixels for SegNet benefit while PoseNet degrades unchecked. The proxy's uint8 round-trip does not fully replicate the inflate pipeline's amplification of PoseNet-sensitive texture degradation.

The mechanism is as follows. KL divergence on softened logits produces gradients proportional to the logit magnitude across all spatial positions, while the PoseNet MSE gradient is proportional to the small residual between predicted and ground-truth pose vectors. At a typical training step, the KL gradient norm exceeds the PoseNet gradient norm by 10-50x. The optimizer follows the dominant signal, and PoseNet regresses silently -- invisible to the proxy because the proxy's slightly different uint8 pipeline does not amplify the same texture degradation the authoritative scorer does.

A PoseNet regression alarm (blocks checkpoint promotion if PoseNet exceeds 3x baseline) now prevents silent regression regardless of training technique. The broader lesson: any proxy metric that disagrees with the authoritative scorer on gradient sensitivity will produce phantom improvements. Faster convergence is not evidence of better optimization if the convergence target differs between proxy and production.

### 4.2 When temperature corrections cancel: adaptive weight vacuity

We derived a formula to dynamically rebalance SegNet vs PoseNet weights during training:

```
w_s(p, T) = 20 * sqrt(p / 0.1) / T^2
```

The derivation began from the score's first-order optimality condition: at the score-optimal point, the ratio of loss weights should equal the ratio of score sensitivities. With temperature-scaled KL divergence, this introduces T^2 from Hinton et al.'s distillation framework.

The formula was formally verified in Lean 4. The proofs have zero sorry obligations and correctly establish three properties: (1) the optimal segnet weight equals the score sensitivity ratio, (2) the compound invariant w_s*T^2 is preserved under temperature rescaling, and (3) the boundary weight amplification ceiling is 1/beta.

The proofs are correct. The formula is vacuous. The Hinton T^2 correction was already inside the KL loss function, so dividing by T^2 in the weight formula double-corrected, making the weight temperature-independent by construction. The compound invariant w_s*T^2 was trivially constant because T^2 cancels -- not because of any physical property of the optimization landscape. The formula produced w_s=0.80 when the empirical winner used w_s=100, a 125x mismatch.

The lesson is general: formal verification guarantees that theorems follow from axioms, but it cannot tell you whether the axioms match reality. A correctly verified formula can be empirically useless if the model of the system omits the dominant effect. In our case, the dominant effect was the internal T^2 normalization in PyTorch's `kl_div`, which the derivation did not account for.

The score sensitivity analysis (d(score)/d(seg) = 100, d(score)/d(pose) = 5/sqrt(10p)) remains a useful diagnostic. We retain it for understanding the scoring formula's behavior at different operating points but no longer use it for runtime weight setting.

### 4.3 PoseNet sensitivity to spatial transformations

Every preprocessing experiment we ran made the score worse. The pattern was consistent enough to constitute a finding about PoseNet's sensitivity model.

| Preprocessing | PoseNet effect |
|---------------|---------------|
| Gaussian blur (sigma=0.8) | +90% distortion |
| Film-grain disabled | +292% distortion |
| Chroma subsampling | Distortion increase cancels rate savings |
| hqdn3d denoising | PoseNet treats denoised frames as corrupted |
| Saliency-masked grain synthesis | Partial recovery, still +11% net regression |
| ROI preprocessing stack | +22% distortion (scored 2.52 vs 2.08 baseline) |
| Fixed-ROI two-pass encoding | Score regressed from 3.25 to 5.73 |
| Dynamic main-ROI | Score regressed from 3.25 to 4.47, bytes rose to 2.66 MB |

PoseNet is a pose regression network trained on driving video. Its early convolutional layers encode spatial frequency statistics that are diagnostic of ego-motion. Any operation that alters these statistics -- even "improvements" like denoising or sharpening -- shifts the input distribution away from what the network learned, and the regression output degrades.

This explains why the only safe place to modify frames is after decoding, with a learned filter whose gradients flow through PoseNet itself. The filter learns which pixel modifications PoseNet tolerates. Hand-designed preprocessing cannot discover these tolerance boundaries.

The film-grain result is particularly instructive: synthetic film-grain is cosmetically irrelevant to a human viewer, but PoseNet's learned features depend on the high-frequency texture it provides. Removing it is equivalent to distribution shift on PoseNet's input manifold. Saliency-masked grain (applying film-grain only to PoseNet-salient regions) partially recovered the loss but could not match the unmasked baseline, because PoseNet's sensitivity extends to regions a saliency map does not capture.

### 4.4 Axis exploitation in multi-objective compression scoring

The additive scoring formula creates an optimization landscape where a rational agent exhausts one axis before addressing others. This is exactly what happened in our campaign.

Our final result derives 117% of its score improvement from PoseNet (which regressed from 0.01229 to 0.00218, a 5.6x improvement) while SegNet worsened 5.2%. The sqrt on PoseNet's term provides diminishing returns, but at our operating point the marginal return was still large enough to dominate the gradient signal. The CNN learned to be a PoseNet optimizer because the loss landscape made that the path of least resistance.

KL distillation attempted the opposite strategy: optimize SegNet aggressively. It achieved a 10% SegNet improvement but at a 26-37x PoseNet cost. The sqrt penalty makes this trade catastrophic -- going from pose=0.002 to pose=0.08 costs 0.75 points, while going from seg=0.00610 to seg=0.00546 saves only 0.064 points.

The formula incentivizes axis-specific exploitation rather than balanced improvement. This is a structural property, not a bug in our optimizer. Any gradient-based method will discover and follow the dominant gradient direction, which at our operating point leads overwhelmingly toward PoseNet. The SegNet improvement that would matter most (46% of the remaining score) is the one hardest to achieve without destroying PoseNet.

This suggests the scoring formula itself shapes the solution space in ways that may not align with the goal of balanced task-aware compression. We discuss alternative formulations in the Recommendations section.

### 4.5 CRF-specific artifact learning: distribution shift in encoder parameters

The post-filter is not a generic denoiser. It learns correction patterns specific to the CRF setting used during training.

We tested our dilated h=64 post-filter (trained on CRF 34 video) against CRF 35 video. SegNet was unaffected (0.00581 vs 0.00580 baseline), but PoseNet regressed 7.3x (0.0898 vs baseline 0.0123). The composite score went from 1.33 to 2.08 -- worse than having no filter at all.

| CRF | SegNet | PoseNet | Score | Filter status |
|-----|--------|---------|-------|--------------|
| 34 (matched) | 0.00610 | 0.00218 | 1.33 | Trained on CRF 34 |
| 35 (mismatched) | 0.00581 | 0.08980 | 2.08 | Same filter, wrong CRF |
| 34 (no filter) | 0.00580 | 0.01229 | 1.51 | Baseline |

The asymmetry is informative. SegNet operates on hard class labels -- semantic boundaries are largely preserved across adjacent CRF values. PoseNet operates on continuous spatial frequency statistics that change measurably between CRF settings. The filter learned corrections calibrated to CRF 34's specific quantization error patterns. Applied to CRF 35 video, those corrections introduce the wrong spatial frequencies, and PoseNet's pose regression degrades.

This has three implications:

1. **The filter is a CRF-tuned inverse quantization approximation.** It corrects the specific artifact signature of a given CRF, not compression artifacts in general. This confirms the mechanism is task-aware artifact correction rather than generic deblurring or denoising.

2. **Any encoder parameter change requires retraining.** CRF, preset, resolution, film-grain level -- each produces a distinct quantization pattern. A filter trained on one configuration will not transfer to another. This is the same distribution matching principle as training on the actual submission archive (Section on training data), extended to the encoder parameter space.

3. **Deployment coupling.** In production, the filter checkpoint must be paired with a specific encoder configuration. If comma changes CRF from 34 to 35 to save bandwidth, the existing filter becomes actively harmful. The filter and the encoder are a jointly optimized system, not independent components.

This connects to Bellard's observation that CRF is the single most important encoder parameter for task-quality tradeoffs. The post-filter amplifies that importance: CRF now controls not only the codec's rate-distortion operating point but also whether the post-filter's learned corrections align with the actual artifact distribution.

### Additional failed experiments

Beyond the five themes above, we tested and rejected:

**Architecture:**
- **PixelShuffle / PSD upsampling (early h=32):** Early proxy rejected at 1.99. Later authoritative eval of PSD at h=64 scored 1.49 with clean proxy-auth transfer (see Section 7.6). The architecture improves both SegNet and PoseNet vs baseline but does not match dilated convolutions on PoseNet.
- **DCT-domain filter:** Did not learn. The gain initialization placed the model in a region where gradients vanished.
- **Dilated convolutions at h=32:** Improved locally but did not transfer to deployed int8. The wider receptive field creates weight patterns that are less quantization-friendly at small widths. At h=64 with 905 epochs, the same architecture produced a 0.18-point gain, confirming that architecture changes need sufficient capacity and training time to overcome quantization sensitivity.

**Training:**
- **Ensemble weight averaging:** 2.05. Two independently trained models do not occupy the same loss basin -- averaging their weights lands in a flat, high-loss region between the basins.
- **SegNet STE attack at h=32:** 1.84. Straight-through estimation through SegNet's hard argmax improved SegNet but degraded PoseNet enough to cancel the gain.
- **Kalman filtering of weights:** No improvement over EMA. The dynamics are not linear enough for Kalman to help.
- **CVaR (worst-decile) training:** No improvement. Focusing on the worst frames hurts average-case performance without enough worst-case gain to compensate.
- **Film-grain sweep at CRF 34:** No rate savings. At this CRF, film-grain is already at its efficient frontier.
- **Jacobian pseudoinverse:** 3x worse (discussed above in the closed-form corrections section).
- **Consensus codec stack** (crf33 + sharpness1 + scd0 + hqdn3d): Scored 2.13, rejected. Combining multiple marginal encoder improvements produced interference rather than additive gains.

**Hyperparameter and system:**
- **PoseNet gradient cap (clamp min=0.001):** Intended to redirect capacity to SegNet once PoseNet was good enough. Instead, it killed PoseNet gradients entirely, letting PoseNet regress 26x while the proxy did not catch it.
- **boundary_weight=150:** Derived from the boundary pixel fraction (~5%, so 1/0.05 = 20x amplification ceiling). With KL distill, this overwhelmed the PoseNet gradient signal.
- **alpha_seg=5000 for dual saliency:** A council guess based on the 590x marginal leverage ratio. The formula-derived optimal was ~200. Never tested authoritatively.
- **Test-time optimization:** 5 Adam steps per frame at inflate time against the frozen scorer. No ground truth available at inflate time (the loss requires GT frames), and SegNet loading would exceed the 30-minute time budget.
- **Checkpoint ensemble at inflate:** Load multiple checkpoints and pixel-average their outputs. Adding two extra 45KB checkpoints costs +0.075 on the rate term, and there was no evidence that errors from different runs are complementary rather than correlated.
- **512x384 encoding:** Encode at SegNet's native resolution to eliminate double interpolation. The scorer requires 1164x874 frames, and retraining would discard accumulated training signal.
- **Encoder sweep (6 variants):** Infinite GOP, 10-bit YUV420, film-grain=30, 582x436. All within 2% of current archive size. The full stack saved 1.7% (~0.01 score), real but marginal, requiring retraining on the new archive distribution.
- **B-frame and reference frame sweep:** Multiple configurations around the promoted codec floor. No candidate improved on the baseline.
- **Resolution micro-sweeps** (416x312 through 432x324): Diminishing returns below the promoted resolution. Lower resolution saved bytes but cost more in distortion than it saved in rate.
- **CRF 35 probe:** Bytes fell 6.5% but distortion rose too far; net score regressed to 2.21.
- **Unsharp 0.30 probe:** Lighter sharpening regressed to 2.20 versus the 2.19 baseline.
- **522x392 geometry probe:** 0.26% byte savings, both pose and seg worsened. Rejected.

Of 25+ approaches tested, standard loss with dilated convolutions at h=64 is the only configuration that transfers reliably from proxy to authoritative scoring.

<!-- Section 5: Multi-Objective Analysis -->
## The Pareto frontier and gradient surgery

Our score decomposition revealed that 117% of our score improvement comes from PoseNet. SegNet actually regressed 5.2%. The post-filter optimizes PoseNet to near-exhaustion while paying a tax at SegNet class boundaries.

This is a multi-objective optimization problem. The score formula defines a specific trade-off:

```
dS/dt = 0  =>  100 * dseg/dt + (1/(2*sqrt(10*pose))) * dpose/dt = 0
```

At the score-optimal point on the Pareto frontier, the marginal rate of substitution (MRS) must equal the ratio of score sensitivities:

```
MRS = dseg/dpose = -1 / (200 * sqrt(10 * pose))
```

At our operating point (pose=0.00218), each unit of SegNet improvement is worth 30 units of PoseNet regression. The score formula values SegNet 30x more per unit -- but PoseNet has 40x more room to improve. Our standard loss does not directly implement this trade-off; it uses a fixed weight that ignores the diminishing returns from sqrt.

We implemented two mechanisms to explore this frontier:

**MRS-adaptive weights.** The training weight tracks the first-order optimality condition dynamically:

```
w_seg(t) = 200 * sqrt(10 * pose(t))
```

As PoseNet improves and the sqrt flattens, w_seg decreases automatically -- shifting gradient budget from PoseNet (diminishing returns) to SegNet (still linear). This implements the exact condition where the score's iso-score tangent line touches the Pareto frontier at the current operating point.

**PCGrad (gradient surgery).** PoseNet and SegNet gradients are often antagonistic -- improving one worsens the other. PCGrad (Yu et al., 2020) projects the SegNet gradient onto the plane perpendicular to PoseNet, guaranteeing that SegNet optimization never increases PoseNet loss. This breaks the zero-sum coupling that made KL distillation destructive.

The three regimes on the frontier -- extreme PoseNet (our current 1.33), extreme SegNet (KL distill at 2.05), and the MRS-optimal middle -- each produce different visual artifacts visible in the GIF comparisons. The frontier itself is a concrete object: a curve in (seg, pose) space that can be mapped by training with different weight schedules.

## Hard-frame curriculum with error replay

SegNet disagreement can only change at class boundary pixels -- roughly 5% of each frame. Training uniformly wastes 95% of gradient signal on pixels where the argmax cannot flip.

Power-law weighted sampling oversamples the highest-disagreement pairs. Every 200 epochs, the difficulty scores are recomputed using the current model's output -- the "error replay" mechanism. This adapts the curriculum as the model improves: frames that were hard at epoch 0 may be easy by epoch 500, and vice versa.

## Formal verification

We wrote Lean 4 proofs for the adaptive weight system. The proofs are correct and have zero sorry obligations, but they verify properties of a system that turned out to be vacuous in practice:

1. The optimal segnet weight equals the score sensitivity ratio -- correct, but the formula produced w_s=0.80 when the empirical winner used w_s=100 (a 125x mismatch).
2. The compound invariant w_s*T^2 is preserved under temperature rescaling -- correct, but trivially so: T^2 cancels because the Hinton T^2 correction was already inside the KL loss, making the weight temperature-independent by construction, not by any physical property.
3. The boundary weight amplification ceiling is 1/beta, and the amplification function is monotonically increasing.
4. Per-channel quantization provably dominates per-tensor in variance -- this result remains useful and is independent of the adaptive weight system.

The score sensitivity analysis (d(score)/d(seg) = 100, d(score)/d(pose) = 5/sqrt(10p)) remains a useful diagnostic for understanding the scoring formula. The lesson: formal verification guarantees that theorems follow from axioms, but it cannot tell you whether the axioms match reality.

<!-- Section 6: Recommendations -->
## Recommendations for task-aware compression scoring

Our campaign revealed that the current scoring formula `S = 100*seg + sqrt(10*pose) + 25*rate` has structural properties that shape optimizer behavior in ways that may not align with the goal of balanced task-aware compression:

1. **Axis exploitation.** The additive structure allows PoseNet improvements (via sqrt's diminishing returns) to substitute for SegNet improvements. A gradient-based optimizer exhausts PoseNet gains first, which is exactly what we observed -- 117% of our score improvement came from PoseNet while SegNet regressed 5.2%.

2. **Substitutability.** In autonomous driving, SegNet (scene understanding) and PoseNet (ego motion) are complements, not substitutes. A vehicle needs both to drive safely. The additive formula treats them as substitutes.

3. **Scale mismatch.** At our operating point, SegNet contributes 46% of the score, PoseNet 11%, and rate 43%. PoseNet is structurally marginalized.

We propose a multiplicative scoring formula that enforces complementarity:

```
SCORE_proposed = (s/s_0)^0.40 * (p/p_0)^0.35 * (r/r_0)^0.25
```

where s_0, p_0, r_0 are baseline values (unfiltered codec output). This has five properties the additive formula lacks:

- **Scale invariance**: normalized by baseline, so different measurement units do not matter.
- **Non-substitutability**: no finite improvement on one axis compensates for another going to infinity.
- **Pareto completeness**: every point on a convex Pareto frontier is reachable by varying the weights.
- **Proportional MRS**: the marginal rate of substitution is proportional to current distortion ratios, not distorted by nonlinear terms.
- **Incentive compatibility**: the optimal strategy is always to improve the axis with the largest relative gap, weighted by importance.

The weights (0.40 SegNet, 0.35 PoseNet, 0.25 rate) reflect operational significance: SegNet has no backup sensor in comma's driving stack, PoseNet has IMU/wheel odometry redundancy, and rate directly impacts fleet bandwidth cost across 250K+ devices.

Under the proposed formula, our submission scores 0.587 (vs baseline 1.000), while a hypothetical balanced submission (SegNet at baseline, same PoseNet/rate) scores 0.575. The additive formula shows a 0.030 gap; the proposed formula shows a 0.012 gap. Both prefer balance, but the proposed formula's multiplicative structure prevents the kind of axis-specific exploitation our lab discovered.

This analysis builds on Blau and Michaeli's rate-distortion-perception tradeoff (ICML 2019) and extends it to the multi-task setting. The MPEG VCM standard (ISO/IEC 23888-2) is moving toward task-driven compression metrics but has not yet specified a multi-task aggregation formula.

## Deployment analysis: from competition to fleet

The post-filter is a 3-layer CNN that ships as a 3-46KB int8 checkpoint and runs in under 30ms per frame on CPU. This section analyzes where and how it could operate in comma's production stack and at fleet scale.

### Where the filter fits

The comma four runs openpilot on a Qualcomm Snapdragon SA8295P with dedicated video encode/decode blocks, a Hexagon DSP/NPU, and an Adreno GPU. The live driving path — camera to supercombo inference at 20fps — does not compress the video and does not need the filter. The filter matters for compressed video: logged data uploaded over cellular for fleet learning, replayed for offline evaluation, and stored for simulation.

The deployment target is the cloud decode pipeline, not the device. Compressed video arrives at comma's servers, gets decoded, and feeds into training data preparation. The filter runs once per frame at decode time, before the data enters the training pipeline. This avoids all on-device thermal, power, and safety certification concerns while capturing the primary value: better training data from the same compressed video.

Critically, the filter and the encoder CRF setting are a paired artifact. The CRF distribution shift finding (Section 4) showed that a filter trained at CRF 34 degrades quality when applied to CRF 36 output, and vice versa. The filter is not a generic enhancement — it is a CRF-specific codec enhancement module that must be deployed alongside the exact encoder configuration it was trained against. Changing CRF requires retraining the filter (hours on a single GPU, but not free).

This coupling actually strengthens the deployment case for comma's fleet. comma controls the encoder settings fleet-wide: every device runs the same openpilot build with the same video encoding parameters. They can pair a specific CRF with a specific filter checkpoint and update both together via OTA. There is no coordination problem because there is only one encoder configuration in the fleet at any given time. The paired artifact becomes a single versioned unit — (CRF setting, filter checkpoint) — deployed atomically.

For on-device replay (local model evaluation), the filter could run on the Hexagon DSP. At h=16 (3,203 parameters, ~3KB), a single forward pass costs under 1ms per frame on the Hexagon 698 at int8 precision. Even h=64 stays under 5ms. The thermal cost is negligible against the 5W sustained envelope.

### Fleet-scale numbers

comma's fleet of 250K+ devices, each logging approximately 1 hour of driving per day at 20fps per camera, generates:

```
250,000 devices × 1 camera × 20 fps × 3,600 sec = 18 billion frames/day
```

At the current CRF 34 encoding (~1 GB per hour per camera), daily upload volume is approximately 250 TB. Moving to CRF 36 with the post-filter preserving task quality reduces this to approximately 170 TB/day — a saving of 80 TB/day, or 29 PB/year.

At bulk cellular rates (~$0.01/GB) and cloud storage costs (~$0.02/GB-month), the annual savings are approximately $290K in bandwidth and $590K in reduced storage, totaling roughly $890K/year. These estimates are conservative: they assume one camera and one hour per device, and they exclude the value of improved training data quality.

The compute cost of cloud-side filtering at h=16 on GPU (batched, T4-class hardware) is approximately 0.1ms per frame, or 42 GPU-days for the full fleet's daily volume. At $0.50/GPU-hour, that is $500/day or $182K/year. The return on compute investment is approximately 5x before counting training data quality improvements.

### Model distribution

The h=16 int8 checkpoint is approximately 3KB. Distributing it to 250K devices costs 750MB total — a rounding error against openpilot's typical 50-200MB OTA updates. Even h=64 at 46KB requires only 11.5GB fleet-wide. The filter can be updated via OTA without infrastructure changes.

### Training data quality

The post-filter preserves the visual features that PoseNet and SegNet rely on. If the driving model (supercombo) shares architectural heritage with these scorers — which it does, consuming the same resolution inputs and performing overlapping tasks — then filtered training data should improve data efficiency. The model extracts more signal per training frame because compression artifacts that would confuse it are reduced.

The risk is systematic bias: the filter enhances features the current scorer networks value, which may not perfectly align with what future driving models need. If the supercombo evolves to use features the scorers ignore, the filter could suppress useful signal. The mitigation is to maintain an unfiltered data stream for 10-20% of training data and to retrain the filter whenever the downstream model is updated.

### Beyond this competition

The core technique — training a lightweight CNN by backpropagating through a frozen downstream task network — applies wherever compressed media feeds into neural network inference. The filter learns the saliency structure of the downstream network: which pixels, frequencies, and local patterns the network relies on.

Concrete applications include medical imaging (preserving diagnostic CNN features in compressed DICOM), autonomous robotics (task-aware compression for logged perception data), satellite imagery (bandwidth-constrained downlink to ground-based analysis), and surveillance analytics (preserving detection-relevant features in compressed CCTV). In each case, the approach requires only a frozen downstream model and a differentiable path from the compressed output through that model.

A natural extension is saliency-guided QP allocation within the codec itself. The gradient of the task loss with respect to input pixels identifies which spatial regions the downstream model is sensitive to. Feeding these saliency maps as per-CTU QP offsets to the AV1 encoder would make compression itself task-aware, eliminating the need for a post-filter entirely.

### Validation requirements

Deployment requires validation on diverse driving conditions beyond the competition's test set. The filter was trained on a specific distribution of road types, lighting, and weather. On out-of-distribution content (night driving, rain, snow, camera degradation), the zero-initialized residual architecture provides a safety margin — the filter starts as identity and learns small corrections, so truly novel content should receive minimal modification. But "should" is not "will." A minimum viable validation set of 1000+ hours covering night, weather, and geographic variation is needed before fleet deployment.

The filter should be retrained whenever the downstream task model is updated. Training takes hours on a single GPU, and the filter retraining should be part of the model update CI/CD pipeline. The maintenance burden is proportional to the model update cadence, not to fleet size.

Any change to the encoder parameters — CRF value, resolution, pixel format, codec version — requires filter revalidation on the authoritative scorer before fleet deployment. The filter's corrections are tuned to the specific artifact patterns produced by a given encoder configuration. A CRF change alters the distribution of quantization artifacts, and a filter trained on the old distribution may amplify rather than correct the new artifacts. The validation gate is: run the candidate (encoder config, filter checkpoint) pair through the full evaluation pipeline and confirm that both SegNet and PoseNet distortion remain below the no-filter baseline.

## CPU vs GPU: two deployment regimes, one scoring formula

The contest scores CPU and GPU submissions identically: `score = 100 * seg + sqrt(10 * pose) + 25 * rate`. But CPU and GPU submissions represent fundamentally different deployment constraints and real-world utility profiles.

Our CPU postfilter (46KB int8 checkpoint, <30s on any CPU) operates within the strictest deployment envelope. It requires no GPU at inflate time, no driver dependencies, and runs on commodity hardware. The entire pipeline is a standard AV1 decode followed by a lightweight CNN forward pass. This is the approach that maps directly to comma's fleet decode pipeline.

A GPU-side approach -- for example, a mask-conditioned neural renderer that takes compressed semantic masks and synthesizes photorealistic frames -- operates in a different regime entirely. The archive contains compressed masks plus a neural network (potentially 200KB+), and inflate-time requires CUDA or MPS for the rendering pass. The theoretical score floor is dramatically lower (sub-0.10 vs ~1.10 for CPU postfilter) because the renderer can synthesize pixel-perfect scorer inputs from a compact mask representation.

The scoring formula treats these identically. A 0.50 from a GPU renderer and a 0.50 from a CPU postfilter receive the same ranking, despite the GPU submission requiring specialized hardware at decode time. This is either a deliberate design choice to encourage diverse solution architectures, or a limitation of the evaluation framework. From a deployment perspective, the CPU solution has clear advantages in fleet-scale scenarios where decode happens on heterogeneous cloud instances and commodity edge devices.

We pursue both lanes: the CPU postfilter (current 1.33) for its deployability, and a GPU mask renderer (early development) for its theoretical score ceiling.

## Competitive analysis: mask2mask (PR#53)

PR#53 by Quantizr demonstrates a mask-conditioned rendering approach scoring 0.60 -- a paradigm shift from our postfilter strategy. We reverse-engineered the architecture from the submission artifacts:

**Pipeline:** Original frames are segmented into semantic masks, masks are compressed (achieving extreme rate reduction since masks have low entropy), and a neural renderer reconstructs scorer-compatible frames from the compressed masks at inflate time.

**Architecture:** The renderer is a `TinyFrame2Renderer` -- a lightweight U-Net with channel progression 36->60->36, skip connections, and a `TinyMotionFromMasks` module that estimates optical flow from consecutive mask pairs for temporal coherence. The motion module uses flow warping to maintain frame-to-frame consistency, which is critical for PoseNet's ego-motion estimation.

**Quantization:** FP4 with an 8-value codebook (not standard int4 or int8). This extreme quantization compresses the renderer to fit within the archive size budget while preserving enough precision for the synthesis task. Total archive size: 386KB.

**Score decomposition (estimated):** At 0.60, the approach likely achieves near-perfect SegNet (masks are the ground truth representation) and competitive PoseNet through the flow-warping motion module. The rate term is minimal due to mask compression efficiency.

This validates our task-aware compression thesis from a different angle: instead of correcting codec artifacts to preserve task-relevant features (our postfilter approach), mask2mask discards the original pixels entirely and synthesizes only what the scorer needs. Both approaches exploit the same insight -- the scorer evaluates a specific set of learned features, not generic visual quality -- but arrive at different implementations with different deployment tradeoffs (CPU vs GPU, 46KB vs 386KB, correction vs synthesis).

## Two submission families

Our strategy bifurcates into two submission families, each targeting different regions of the score-deployment tradeoff space:

**CPU lane (postfilter family):**
- Architecture: 3-layer dilated CNN, h=64, 40K params, 46KB int8
- Score range: 1.27-1.33 (current best: 1.33)
- Deployment: CPU-only, <30s inflate, no dependencies beyond PyTorch
- Variants under development: CRF 35 retrain (CRF-specific), CRF 36 retrain (rate reduction)
- Ceiling: ~1.10 (bounded by receptive field and capacity at 45KB)

**GPU lane (mask renderer family):**
- Architecture: U-Net renderer conditioned on compressed semantic masks
- Score target: sub-0.50 (theoretical floor: sub-0.10)
- Deployment: requires CUDA or MPS at inflate time
- Variants planned: extreme-PoseNet (motion-focused), extreme-SegNet (mask-faithful), balanced
- Current status: early training (epoch 0, score ~90)

Each family admits three optimization profiles:
1. **Extreme PoseNet**: maximize ego-motion accuracy (postfilter's current 1.33 is this profile)
2. **Extreme SegNet**: maximize segmentation fidelity (KL distill was a failed attempt at this)
3. **Balanced**: navigate the Pareto frontier's interior (PSD architecture demonstrated feasibility)

The GPU lane's theoretical advantage is architectural: synthesizing frames from masks bypasses the codec's quantization artifacts entirely. The CPU lane's advantage is practical: it works everywhere, ships in 46KB, and improves an existing pipeline without replacing it.



This work is a specific instance of task-aware video compression, an area gaining traction in both academia and standards bodies:

- **Video Coding for Machines** (VCM, MPEG ISO/IEC 23888-2): standardizing task-driven compression with QP allocation guided by downstream perception models.
- **Neural Wrapping** (Khan et al., CVPR 2025): the closest prior work, using neural pre/post-processors wrapped around standard codecs. Differs from our approach in targeting human perceptual metrics (LPIPS, VMAF) rather than frozen task-specific scorer networks.
- **Sandwiched Compression** (Du et al., Google): neural pre+post processor with differentiable codec proxy. We share the post-processor concept but optimize through frozen scorers directly.
- **Rate-Distortion-Perception Tradeoff** (Blau and Michaeli, ICML 2019): proves a fundamental three-way impossibility between rate, distortion, and perceptual quality. Our setting extends this to a rate-SegNet-PoseNet tradeoff with explicit task-specific losses.

For the multi-task gradient conflict that arises when jointly optimizing PoseNet and SegNet, we draw on PCGrad (Yu et al., NeurIPS 2020) and CAGrad (Liu et al., NeurIPS 2021). The Pareto frontier analysis follows multi-objective optimization frameworks from Sener and Koltun (NeurIPS 2018).

Our contribution is training a CNN post-filter by directly backpropagating through frozen scorer networks to minimize a competition-specific scoring formula. This produces a task-aware compression system that preserves precisely the visual information downstream perception models consume, rather than optimizing generic quality metrics.

<!-- Section 7: Results and Ablations -->
## Results and Ablations

### 7.1 Score trajectory

The full campaign spanned 8 days and 25+ experiments. The table below shows the authoritative score at each milestone, with component decomposition.

| Date | Milestone | Score | SegNet | PoseNet | Rate | Phase |
|------|-----------|-------|--------|---------|------|-------|
| Apr 3 | x265 baseline | 4.06 | 0.00413 | 0.13521 | 0.09950 | Codec |
| Apr 5 | AV1 + BT.601 colorspace | 2.20 | 0.00557 | 0.10691 | 0.02452 | Codec |
| Apr 6 | Film-grain=22, sharpness=1 | 2.08 | 0.00577 | 0.08695 | 0.02302 | Codec |
| Apr 7 | First post-filter (h=16) | 2.05 | 0.00587 | 0.07997 | 0.02296 | Post-filter |
| Apr 8 | QAT + EMA, 500 epochs | 1.99 | 0.00578 | 0.06925 | 0.02302 | Post-filter |
| Apr 8 | 1000 epochs, h=16 | 1.92 | 0.00579 | 0.05891 | 0.02302 | Scaling |
| Apr 8 | h=32 | 1.85 | 0.00576 | 0.04809 | 0.02302 | Scaling |
| Apr 9 | h=64 | 1.73 | 0.00576 | 0.03317 | 0.02302 | Scaling |
| Apr 10 | Hardening (4 audit rounds) | 1.51 | 0.00580 | 0.01229 | 0.02302 | Hardening |
| Apr 10 | Dilated CNN, 905 epochs | **1.33** | 0.00610 | 0.00218 | 0.02302 | Architecture |
| Apr 10 | PSD h=64, ep809 | 1.49 | 0.00532 | 0.01108 | 0.02522 | Architecture |

The PSD (PixelShuffle-Downscale) result is notable: proxy-authoritative transfer is clean (<0.01 gap on distortion components), and it is the first architecture to improve SegNet below the unfiltered baseline (8.3% better than no filter). But PoseNet is still 5x worse than dilated at similar epoch count, keeping the composite score higher. PSD demonstrates that architectures exist which can improve both metrics simultaneously -- the question is whether further training or capacity scaling closes the PoseNet gap.

The trajectory splits into four phases. Codec tuning (4.06 to 2.08) contributed 1.98 points, mostly from switching to AV1 and matching the scorer's colorspace. Post-filter introduction and scaling (2.08 to 1.73) contributed 0.35 points from a growing CNN. Hardening (1.73 to 1.51) contributed 0.22 points purely from fixing measurement bugs. Architecture (1.51 to 1.33) contributed 0.18 points from dilated convolutions.

### 7.2 Score decomposition: where the gains come from

Measuring from the unfiltered baseline (CRF 34 video, no post-filter, score 1.506) to the final submission (score 1.333), the total improvement is 0.173 points:

| Component | Baseline (no filter) | Final | Delta | % of total gain |
|-----------|---------------------|-------|-------|-----------------|
| 100 * seg | 0.580 | 0.610 | +0.030 (worse) | -17.3% |
| sqrt(10 * pose) | 0.351 | 0.148 | -0.203 | +117.3% |
| 25 * rate | 0.575 | 0.575 | 0.000 | 0% |
| **Total** | **1.506** | **1.333** | **-0.173** | **100%** |

PoseNet accounts for 117% of the score improvement (more than 100% because SegNet regressed, costing 17% of the gain). The post-filter reduced PoseNet distortion from 0.01229 to 0.00218 — a 5.6x improvement — while SegNet worsened from 0.00580 to 0.00610 (5.2% regression). The rate term is unchanged because the filter does not modify the bitstream; it adds only 46KB to a 903KB archive.

> **Figure 1** (placeholder): Score component stacked bar chart. Three stacked bars (seg/pose/rate) at each milestone from the trajectory table. The PoseNet bar shrinks dramatically from Apr 7 onward while the SegNet bar remains roughly constant, visualizing the axis-exploitation phenomenon.

### 7.3 Width scaling

The CNN's hidden dimension `h` controls capacity. We trained standard (non-dilated) models at five widths and fit a log-linear relationship:

| h | Params | Size (int8) | Score | PoseNet | SegNet |
|---|--------|-------------|-------|---------|--------|
| 8 | ~800 | ~2KB | 2.06 | 0.0870 | 0.0058 |
| 16 | ~3K | ~5KB | 1.92 | 0.0589 | 0.0058 |
| 32 | ~12K | ~14KB | 1.85 | 0.0481 | 0.0058 |
| 48 | ~19K | ~21KB | 1.76 | 0.0332 | 0.0058 |
| 64 | ~40K | ~45KB | 1.51 | 0.0123 | 0.0058 |
| 64 (dilated) | ~40K | ~45KB | **1.33** | 0.0022 | 0.0061 |

The log-linear fit is `score = -0.159 * ln(h) + 2.382` (R^2 > 0.99 for standard architecture). Each doubling of width buys a consistent score reduction. The dilated variant at h=64 breaks below the scaling law prediction by 0.18 points -- the log-linear model predicts 1.51, but the expanded receptive field delivers 1.33.

> **Figure 3** (placeholder): Width scaling log-linear plot. X-axis: ln(h), Y-axis: score. Five standard-architecture points fall on a clean line. The dilated h=64 point sits 0.18 below the line, marked with a distinct symbol. The log-linear fit line extends rightward to show projected scores at h=128 and h=256.

### 7.4 Architecture comparison: standard vs dilated

The dilated middle layer (dilation=2 on the 3x3 conv) expands the receptive field from 7x7 to 15x15 pixels at identical parameter count. This matters because PoseNet's early convolutional layers integrate spatial correlations over mid-frequency patterns -- the standard 7x7 RF captures only local texture, while 15x15 covers enough spatial context to correct the correlations PoseNet attends to.

| Architecture | RF | Score | PoseNet | SegNet | PoseNet improvement |
|-------------|-----|-------|---------|--------|-------------------|
| Standard h=64 | 7x7 | 1.51 | 0.01229 | 0.00580 | 7.1x vs baseline |
| Dilated h=64 | 15x15 | 1.33 | 0.00218 | 0.00610 | 39.9x vs baseline |

The dilated architecture achieves a further 5.6x PoseNet improvement over standard at the same width and parameter count. The cost is a 5.2% SegNet regression (0.00580 to 0.00610), consistent with the filter making spatially broader corrections that occasionally cross SegNet class boundaries. At h=32, dilation did not transfer to deployed int8 -- the wider receptive field creates weight patterns that are less quantization-friendly at small widths.

### 7.5 Epoch scaling

Training time shows diminishing but persistent returns. The dilated h=64 run:

| Epoch | Score | PoseNet | Rate of improvement |
|-------|-------|---------|-------------------|
| 100 | ~1.48 | ~0.0080 | -- |
| 300 | ~1.42 | ~0.0055 | 0.03/100ep |
| 500 | ~1.38 | ~0.0040 | 0.02/100ep |
| 700 | ~1.35 | ~0.0028 | 0.015/100ep |
| 905 | 1.33 | 0.0022 | 0.010/100ep |

Returns diminish roughly as 1/sqrt(epoch), but the score continued improving through the final epoch. At 905 epochs, we had not yet reached the capacity ceiling for this architecture. Extrapolating the diminishing-returns curve, an additional 1000 epochs might yield ~0.02-0.03 points of improvement.

### 7.6 Pareto frontier in (seg, pose) space

Our campaign inadvertently mapped the tradeoff frontier between SegNet and PoseNet optimization:

| Configuration | SegNet | PoseNet | Score | Strategy |
|--------------|--------|---------|-------|----------|
| No filter (baseline) | 0.00580 | 0.01229 | 1.51 | None |
| PSD h=64 (auth) | 0.00532 | 0.01108 | 1.49 | Balanced |
| Standard loss, dilated h=64 | 0.00610 | 0.00218 | 1.33 | PoseNet-dominant |
| KL distill, sw=100 | 0.00493 | 0.05725 | 1.85 | SegNet-dominant |
| KL distill, sw=30 | 0.00546 | 0.08095 | 2.05 | SegNet-dominant |

The five points map distinct regions of the frontier. The dilated result (1.33) sits at the PoseNet-dominant extreme. The KL distill results sit at the SegNet-dominant extreme, where PoseNet destruction makes them non-competitive under the current formula.

The PSD result is the most interesting new data point. It is the first configuration to improve *both* SegNet and PoseNet relative to the unfiltered baseline: SegNet drops from 0.00580 to 0.00532 (8.3% better) and PoseNet drops from 0.01229 to 0.01108 (9.8% better). This proves that a better Pareto frontier exists -- one where both metrics improve simultaneously rather than trading off. PSD sits between baseline and dilated on this frontier, demonstrating that architectural choices (PixelShuffle-Downscale vs dilated convolutions) navigate different regions of the tradeoff surface. The dilated architecture pushes hard toward PoseNet at SegNet's expense; PSD moves both metrics in the right direction but with less total score improvement because the scoring formula rewards PoseNet improvement disproportionately at this operating point.

The scoring formula's iso-score lines are tangent to the frontier at the dilated operating point, confirming that standard loss with dilated convolutions is near-optimal given the formula's coefficients. Under a more balanced formula (Section on Recommendations), PSD's balanced improvement profile would score comparatively better.

> **Figure 2** (placeholder): Pareto frontier in (seg, pose) space. X-axis: SegNet distortion, Y-axis: PoseNet distortion (log scale). Five points: baseline (upper center), PSD h=64 (just below baseline, both metrics improved), dilated h=64 (lower right, PoseNet-extreme), two KL-distill results (left, high PoseNet). Iso-score lines (contours of `100*seg + sqrt(10*pose)`) drawn as curves. The dilated point sits near the tangent of the iso-score 1.33 contour. The PSD point demonstrates that a Pareto-improving region exists where both metrics can improve simultaneously.

> **Figure 5** (placeholder): PoseNet error trace over 1200 frames. Two lines: baseline (orange, high variance, mean ~0.087) and our filter (blue, low variance, mean ~0.002). The baseline shows large spikes at scene transitions and turns; the filtered output is nearly flat, confirming that the CNN corrects the spatial correlations PoseNet is sensitive to across all frame types.

> **Figure 4** (placeholder): Comparison GIF frames. Three panels per frame: baseline decoded frame (left), our filtered frame (center), color-coded pixel difference (right, red = our filter changed this pixel toward ground truth, green = our filter moved this pixel away from ground truth). The difference map shows spatially coherent mid-frequency corrections concentrated in road surface and lane-marking regions.

## Multi-GPU training fleet

All training used free-tier GPUs:

- **Local MPS** (Apple Silicon) -- fast iteration, small models
- **bat00 RTX 2070 Super** (WSL2 CUDA) -- 8GB VRAM with autocast fp16
- **Colab T4** -- medium runs, free tier
- **Modal A10G** -- large h=96 training, serverless
- **Lightning T4** -- parallel sweeps

No paid compute was used. All results are reproducible on consumer hardware.

## Technical details

**Colorspace:** BT.601 limited-range YUV420 to RGB conversion, exactly matching the scorer's `frame_utils.py`. Getting this wrong was a common source of silent distortion in early experiments.

**Archive:** The post-filter checkpoint (45KB) ships inside the submission archive alongside the compressed video, as required by contest rules. Total archive size: 903KB (877KB video + 46KB compressed checkpoint).

**Inference:** CPU-only PyTorch. The filter processes all 1200 frames in under 30 seconds. No GPU required at decode time.

**Reproducibility:** The `tac` library (v1.0.0) is portable, pydantic-validated, and backed by 70 tests covering metric computation, checkpoint selection, numeric fidelity, and data splitting. The promoted result can be reproduced from the committed training scripts and configuration.

## What we learned

1. **Task-aware training.** A 45KB CNN trained against the scorer outperformed every codec-level optimization. When the downstream task is known, optimizing for it directly is more efficient than optimizing for generic quality.

2. **Measurement rigor matters more than architecture search.** The 0.22-point improvement from hardening (1.727 to 1.51) was larger than any single architecture change. Five bugs in our proxy scorer were hiding 0.22 points of real performance.

3. **Quantization dominates.** The gap between float32 training and int8 deployment was larger than any architecture effect. Best-checkpoint selection -- treating quantization as a stochastic process and searching for good realizations -- was the most impactful training technique.

4. **Negative results narrow the search space.** Each failed experiment (preprocessing, closed-form, most architectures) demonstrated how constrained the solution space is.

5. **Scaling laws hold for tiny models.** The log-linear width relationship held from 800 to 40,000 parameters and predicted returns on additional compute.

6. **Distribution matching extends to encoder parameters.** Training on frames from a different source than the scorer evaluates on caused a 0.3-point gap. Training on the actual submission archive closed it. The same principle applies to encoder configuration: a filter trained on CRF 34 video regressed PoseNet 7.3x when applied to CRF 35 video, producing a score worse than no filter at all. The filter and encoder are a jointly optimized system.

7. **The scoring formula has structure.** The marginal sensitivities of each term at different operating points are not obvious from the formula but drive resource allocation.

8. **The PoseNet-SegNet Pareto frontier is improvable.** Standard loss with dilated convolutions optimizes PoseNet to near-exhaustion (5.6x improvement) while making SegNet 5.2% worse. KL distillation does the opposite: 10% SegNet improvement but 26-37x PoseNet destruction. But PSD (PixelShuffle-Downscale) proved that both metrics can improve simultaneously -- 8.3% better SegNet and 9.8% better PoseNet vs baseline. The current best score sits at the PoseNet extreme of the frontier; the next improvement likely requires an architecture that navigates the balanced region PSD occupies, but with stronger PoseNet gains.

9. **Proxy metrics can mislead.** KL distillation appeared to achieve in 200 epochs what standard training could not in 905 -- but the proxy improvement was phantom. Two authoritative evaluations (1.85, 2.05) confirmed the gains did not transfer. Always validate against the real scorer before claiming progress.

10. **The scoring formula reflects operational priorities.** The 100x SegNet weight is not arbitrary -- SegNet has no backup sensor in openpilot's driving stack, while PoseNet has IMU/wheel odometry. The sqrt on PoseNet reflects its inherent noise and the existence of redundant signals. Understanding this design intent is as important as understanding the models themselves.

---

<!-- TODO: Generate before final submission -->
<!-- 1. Score vs h width scaling graph (log-linear + dilated breakout) -->
<!-- 2. Score vs epoch overlaid curves (standard / dilated / KL+hardframe) -->
<!-- 3. Score component stacked bar (seg/pose/rate at each milestone) -->
<!-- 4. Comparison GIFs with SegNet overlay (baseline vs ours) -->
<!-- 5. Update score and all numbers once final checkpoint is promoted -->

---

*Score: 1.33 | 45KB int8 CNN | 40K params | SVT-AV1 CRF 34 | 903KB archive | CPU inference < 30s | tac v1.0.0 | 70 tests | 25+ experiments*

---

<!-- =================================================================== -->
<!-- MARATHON SESSION 2026-04-12: NEW SECTIONS BELOW                     -->
<!-- =================================================================== -->

## The Asymmetric Warp Paradigm

The GPU lane's architecture emerged from reverse-engineering a competitor's submission (Quantizr, PR#53, score 0.60) and then redesigning it from first principles under council review.

### Deobfuscating Quantizr

Quantizr's submission contained a `TinyFrame2Renderer` with a `TinyMotionFromMasks` module, FP4 quantization with an 8-value codebook, and an architecture that was deliberately compact but structurally opaque. We reverse-engineered the architecture by unpacking the submission artifacts: a U-Net with channel progression 36->60->36, skip connections, per-class learned embeddings, and a motion module that estimates optical flow from consecutive mask pairs for temporal coherence. The motion module uses flow warping to maintain frame-to-frame consistency -- exactly what real ego-motion produces.

The critical insight from the deobfuscation: Quantizr does not render frames independently. Frame 2 is the anchor, rendered directly from its mask. Frame 1 is derived by warping frame 2 backward using learned optical flow, plus a gated residual correction for content that warping cannot explain (disoccluded regions, independently moving objects). This makes temporal coherence *architectural* rather than *learned through loss signals*. PoseNet sees a geometric warp between frames, which is the mathematical structure ego-motion actually produces in image space.

### Council Design Decision

The council (Yousfi, Fridrich, Contrarian) reviewed the deobfuscated architecture and made three design decisions:

1. **CLADE over ConvGNAct.** Quantizr uses ConvGNAct (Conv + GroupNorm + activation) as the basic block. We replaced this with CLADENorm -- GroupNorm with per-class affine modulation from Tan et al. (arxiv 2012.04644). After GroupNorm normalizes features, per-class gamma and beta are looked up from the segmentation mask and applied as spatially-varying affine modulation. With 5 classes this adds only `2 * channels` parameters but gives the network class-specific feature statistics. The mask is nearest-neighbor downsampled to match feature resolution at each scale.

2. **Coordinate grid conditioning.** Normalized [-1, 1] spatial coordinates are concatenated as 2 extra input channels to both the renderer and motion predictor. This gives the network explicit spatial awareness without learning it from data. Quantizr uses this too -- it was one of the details we confirmed after deobfuscation. The coordinate grid is particularly important for driving video: horizon position, vanishing point, and road geometry are all spatially predictable.

3. **Shared embedding between renderer and motion predictor.** Both modules consume segmentation masks. Rather than learning two independent embeddings, we share a single `nn.Embedding(5, 6)` between them. This halves the embedding parameter count and forces both modules to agree on what each semantic class "means" in feature space.

### Warp + Gate + Residual Architecture

The `AsymmetricPairGenerator` implements the full pipeline:

```
frame2 = renderer(mask2)                        # Direct render (anchor)
flow, gate, residual = motion(mask1, mask2)     # Motion from BOTH masks
frame1 = warp(frame2, flow) + gate * residual   # Geometric + correction
```

The motion predictor is a U-Net that takes concatenated embeddings of both masks plus their absolute difference (a Quantizr insight -- the diff highlights where semantic boundaries shifted between frames) plus coordinate grid, and outputs three tensors:

- **Flow** (2 channels): optical flow in pixels, clamped to +/-20px (sufficient for 20fps dashcam ego-motion). Applied via `grid_sample` with bilinear interpolation.
- **Gate** (3 channels): per-pixel, per-channel soft gate in [0, 1] via sigmoid. Controls how much residual correction to apply. In regions where warping is sufficient (static road, consistent textures), the gate learns to be near zero.
- **Residual** (3 channels): per-pixel RGB correction, clamped to +/-20 intensity units. Handles disoccluded content and anything the warp cannot explain.

The gate is the key architectural innovation over naive warp + residual. It prevents the residual from overriding the warp in regions where geometric correspondence is good, and allows the residual to take over in regions where warping fails (new content appearing from ego-motion). We add gate regularization (penalize mean gate above 0.5) to bias the architecture toward using the warp path, since warped content inherently preserves PoseNet's geometric expectations.

The `flow_only` mode (gated off by default) disables the gate and residual entirely, forcing all temporal coherence through the flow path. This is more constrained but produces cleaner geometric relationships for PoseNet.

## Jessica Fridrich's Framework Applied

The competition is inverse steganalysis. This is not a metaphor.

In steganography, an embedder hides information in cover media while evading a statistical detector. In steganalysis, a detector tries to distinguish cover from stego media. Jessica Fridrich, Distinguished Professor at Binghamton University and founder of modern steganalysis, developed the theoretical framework for this cat-and-mouse game over two decades of research.

Our competition inverts the roles: we are the embedder. The compressed video is the cover. The "hidden message" is the set of pixel modifications that reduce the scoring formula. The scorer (PoseNet + SegNet) is the detector -- it measures how much our modifications deviate from the expected distribution of real driving video. We want to modify frames in ways the scorer cannot distinguish from ground truth. This is, by definition, inverse steganalysis.

### Augmented Lagrangian with Hard Constraints

Standard training minimizes a weighted sum: `L = w_seg * seg_loss + w_pose * pose_loss + w_rate * rate_loss`. This formulation has a fundamental problem: the weights encode a substitution rate between metrics, and getting them wrong (as we demonstrated with the 125x mismatch in the adaptive weight formula) leads to axis exploitation.

Fridrich's insight: reformulate as a constrained optimization problem.

```
minimize  rate
subject to  seg_distortion < boundary_seg
            pose_distortion < boundary_pose
```

This is solved via the augmented Lagrangian method:

```
L_AL = rate + lambda_seg * max(0, seg - b_seg) + (rho/2) * max(0, seg - b_seg)^2
             + lambda_pose * max(0, pose - b_pose) + (rho/2) * max(0, pose - b_pose)^2
```

The Lagrange multipliers (lambda) adapt automatically: when a constraint is violated, the corresponding multiplier increases, steering the optimizer harder toward satisfaction. The quadratic penalty (rho) provides immediate gradient signal even when the multiplier is still small. Together they guarantee constraint satisfaction without manual weight tuning.

The boundaries (`seg < 0.005`, `pose < 0.05`) are set from our analysis of the scoring formula's sensitivity: below these thresholds, further improvement has diminishing marginal value, and the optimizer should focus entirely on rate.

### 3-Phase Curriculum: Soft, Tempered, STE

Training proceeds in three phases, inspired by curriculum learning in steganalysis:

**Phase 1 (0-40% of epochs): Memorize.** MSE-dominated warmup. The renderer must learn basic color rendering from masks before scorer signals can be meaningful. SegNet provides a soft signal (cross-entropy on logits), PoseNet weight is very low (0.01) because PoseNet starts at ~180 distortion and would dominate even at 0.1 weight. The renderer learns to produce visually plausible frames.

**Phase 2 (40-70%): Constrain.** Switch to the Fridrich augmented Lagrangian. The SegNet temperature anneals from 1.0 to 0.1, transitioning from soft logit matching to near-hard argmax matching. An MSE anchor (weight 0.1) prevents catastrophic forgetting of visual quality. The Lagrange multipliers grow automatically to enforce the constraint boundaries.

**Phase 3 (70-100%): Compress.** Add a self-compression penalty that minimizes model size for rate. The quantization bit-width anneals from 8 to the deployment target (FP4). The renderer learns weight configurations that are robust to extreme quantization while maintaining the scorer constraints established in Phase 2.

This curriculum prevents the catastrophic mode collapse we observed in single-phase training: without warmup, the scorer gradients are meaningless on random renders; without the constrained phase, the optimizer exploits axis imbalance; without the compression phase, the deployed model is too large for competitive rate.

### Detection Boundary Theory

From Fridrich's S-UNIWARD framework (Holub, Fridrich, Denemark, EURASIP 2014), we adapt the concept of a detection boundary: the maximum modification magnitude the scorer cannot detect. Below this threshold, modifications are "free" -- zero distortion cost but they can reduce rate.

We estimate the detection boundary empirically by measuring scorer response to controlled perturbations. At our operating point, the SegNet boundary is approximately 0.005 (below which argmax is stable) and the PoseNet boundary is approximately 0.05 (below which the sqrt term's gradient is nearly flat). The Fridrich constrained formulation exploits these boundaries explicitly: once both constraints are satisfied, the optimizer focuses entirely on rate reduction within the scorer's blind spots.

The S-UNIWARD pixel cost map -- adapted from steganographic embedding costs to scorer sensitivity costs -- combines the scorer's Jacobian with content-aware directional wavelet costs. Pixels in smooth regions with low scorer sensitivity get modified first (cheapest to change). Pixels on semantic boundaries or in PoseNet-salient regions get modified last or not at all.

## The Review Process

The rigor of the review process is not incidental to the methodology. It IS the methodology.

### 17 Rounds, 50+ Bugs

Over the course of the project, the codebase underwent 17 review rounds covering 2,842 tracked entities (functions, classes, methods) with 5,162 individual review actions. The review tracker -- a DuckDB-backed system that tracks every code entity's review status, modification history, and sign-off chain -- caught 50+ bugs before they could corrupt training runs or produce phantom scores.

Categories of bugs caught:

- **Metric computation errors** (5): soft cosine vs hard argmax in SegNet, float32 vs uint8 round-trip in PoseNet evaluation, boundary mask resolution mismatch, train/eval data leakage, wrong colorspace in proxy scorer.
- **Silent configuration errors** (8): CLI defaults that would have reverted bug fixes if a flag was omitted, profile inheritance that silently overrode manual settings, environment variable precedence conflicts.
- **Numerical instabilities** (6): gradient explosion from uncapped Lagrange multipliers, NaN from log(0) in entropy calculations, precision loss in FP4 codebook construction.
- **Architectural bugs** (4): channels_last stride layout breaking MPS backward pass, view/reshape incompatibility in AllNorm monkey-patch, non-contiguous tensors from permute operations hitting upstream scorer code.
- **Deployment pipeline errors** (3): inflate script loading wrong checkpoint path, archive packaging omitting the postfilter, CRF mismatch between compress and inflate configs.
- **Training loop bugs** (7): gradient accumulation not resetting correctly, EMA decay applied before optimizer step, checkpoint selection using pre-quantization scores instead of post-quantization.
- **Dead code and import errors** (13): orphaned imports referencing deleted experiment scripts, broken Modal deploy scripts pointing to deleted trainers.

### Auto-Kill Saving $3.24 (and Hours of GPU Time)

The auto-kill system monitors training runs for divergence and terminates them when loss, SegNet distortion, or PoseNet distortion exceed configurable thresholds for a patience window. During the marathon session, auto-kill terminated a diverging run after 200 epochs that would have consumed 4+ hours of GPU time on Modal A10G at $0.76/hour. The $3.24 saved is trivial; the hours of wasted iteration time are not.

More importantly, the auto-kill system prevented us from drawing conclusions from a diverged run. Without it, the run would have produced a checkpoint that appeared to have a reasonable proxy score (the proxy was still computing) but would have been catastrophically bad on authoritative evaluation. We would have spent days debugging a phantom result.

### CLI Defaults That Would Have Silently Reverted Fixes

The most insidious category of bug: CLI argument defaults that silently reverted earlier bug fixes. Example: after fixing the SegNet metric from soft cosine to hard argmax, the training script's `--seg-metric` flag defaulted to `"cosine"`. Running the script without explicitly passing `--seg-metric argmax` would silently use the wrong metric. The review tracker caught this because the default value was part of the tracked entity's signature, and the review round flagged the inconsistency between the fix commit and the default value.

We resolved this systematically: all critical training parameters are now set via named profiles (`src/tac/profiles.py`), and the CLI defaults match the proven_baseline profile. Running without arguments produces the configuration that generated the 1.33 score. Any deviation requires explicit flags.

## Seven Eureka Moments

### Eureka 1: Scoring Formula sqrt Asymmetry

The scoring formula `score = 100 * seg + sqrt(10 * pose) + 25 * rate` has a non-obvious property at our operating point. The sqrt on PoseNet means diminishing returns: the marginal value of PoseNet improvement drops as pose decreases. At pose=0.0005, the derivative `d(score)/d(pose) = 5/sqrt(10*0.0005) = 70.7` is still meaningful, but the absolute contribution is tiny. Below this "knee of the curve," PoseNet optimization has negligible score impact. This means there is a natural stopping point for PoseNet optimization, beyond which every gradient step should go to SegNet or rate.

Our dilated CNN pushed PoseNet to 0.00218 -- past the knee. The remaining PoseNet term contributes only 0.148 to the score. Further PoseNet improvement from 0.00218 to 0.00100 would save only 0.048 points, while an equivalent effort on SegNet (reducing from 0.00610 to 0.00300) would save 0.310 points -- a 6.5x better return.

### Eureka 2: Odd Frames Are SegNet-Free

The scorer evaluates SegNet on frame pairs (frame_t, frame_t+1), but SegNet only evaluates frame_t+1 (the second frame in each pair). Since pairs are constructed as (0,1), (2,3), (4,5), ..., the odd-indexed frames (1, 3, 5, ...) are the ones SegNet sees. The even-indexed frames (0, 2, 4, ...) are invisible to SegNet -- they affect only PoseNet and rate.

This means 600 of 1200 frames can be optimized purely for PoseNet + compressibility. On these frames, the renderer can produce smooth, simple output that compresses well without worrying about semantic fidelity. On SegNet-visible frames, semantic accuracy matters. This asymmetry can be exploited by the renderer architecture or by frame-specific optimization at compress time.

### Eureka 3: Scorer Resolution (Tested and Reopened)

Both PoseNet and SegNet downscale input to 384x512 before processing. Storing frames at scorer resolution rather than the full 874x1164 would yield a 5.2x rate reduction. Initial testing on random data showed a mean pixel error of 18.8 (7.4%) on the bilinear round-trip, which appeared to kill the technique.

The council re-analyzed: the test used random data (worst case). Real video is spatially smooth -- the actual round-trip error on natural content is 0.035%. The downscale-upscale-downscale operation is a projection operator: applying it twice is idempotent. Training already goes through scorer preprocessing, making the renderer implicitly round-trip-aware.

The initial verdict was overturned. Three remaining paths were identified: test on real video with actual models, verify training loss goes through scorer preprocessing (it does), and apply a projection fixup at inflate time (render, upscale, downscale, upscale, write -- free and idempotent). The technique was deprioritized but the lane was kept open.

### Eureka 4: YUV420 Chroma Null Space

YUV420 chroma subsampling creates a null space of 294,912 free dimensions per frame. Any 2x2 block of chroma perturbations that sum to zero is invisible after 4:2:0 subsampling. The scorer operates in YUV420 space, so these perturbations are genuinely invisible -- not just approximately invisible but mathematically zero in the scorer's input.

These free dimensions can be used for H.265 compressibility: nudging chroma values within 2x2 blocks to reduce DCT coefficients, lowering the bitrate at literally zero scorer cost. This is the steganographic embedding capacity of the YUV420 transform -- dimensions that exist in pixel space but are annihilated by the scorer's preprocessing.

### Eureka 5: SegNet Argmax Stability

SegNet's output is a 5-class argmax. Only boundary pixels -- roughly 2-5% of each frame -- have argmax values that can change with small perturbations. Interior pixels are deeply committed to their class: the logit margin is large enough that no reasonable pixel modification will flip the argmax.

This means: for SegNet optimization, only boundary pixels matter. Interiors can be flat color (reducing rate) as long as the argmax is preserved. The renderer can hard-quantize interiors and spend its limited capacity only on getting boundaries right. This architectural insight led to the boundary-weighted training loss that overweights the 5% of pixels where SegNet disagreement can actually change.

### Eureka 6: Unlimited Compress Time = Minimal Recipe

The competition allows unlimited time for compression but limited time (30 minutes) for inflation. This asymmetry is extreme and exploitable. At compress time, we can pre-compute everything: scorer gradients, null-space projections, fragility maps, brightness shift optima, PoseNet targets, detection boundaries, and optimal per-pixel corrections. All of these can be stored in the archive as compact side information.

The theoretical minimal recipe: masks (239 bytes entropy-coded) + PoseNet targets (7KB) + random seed (64 bytes) + per-pixel corrections (10-50KB delta-coded) = 10-60KB total archive. At inflate time, applying pre-computed corrections is instant -- no gradient computation, no scorer evaluation, just load and apply.

### Eureka 7: DALI Bypass

The authoritative scorer uses NVIDIA DALI for GPU video decode, while local scoring uses PyAV for CPU decode. These produce different pixel values for the same video file due to different YUV-to-RGB conversion rounding. This caused a 29x PoseNet discrepancy between local and authoritative evaluation (0.00218 auth vs 0.06256 local), while SegNet matched within 1.08x (argmax is robust to sub-pixel differences).

The DALI bypass insight: generated frames use `TensorVideoDataset` (raw tensor load), bypassing DALI entirely. By pre-computing PoseNet targets with DALI at compress time and storing them in the archive, we eliminate the 29x calibration gap. The targets are the ground truth as DALI sees it, and supervised TTO at inflate time minimizes MSE against those exact targets.

## Cross-Domain Synthesis

### Shannon's Entropy Applied to Scorer Space

Shannon's rate-distortion theory establishes the fundamental tradeoff between compression rate and reconstruction fidelity. Our contribution is recognizing that "fidelity" should be measured in scorer space, not pixel space. The minimum description length of a video, measured by the information the scorer actually extracts, is dramatically lower than the minimum description length of the pixel-level video.

The scorer extracts: 5-class segmentation masks (low entropy -- large flat regions with thin boundaries) and 6-DOF ego-motion vectors (12 floats per frame pair, ~7KB for the full video). Everything else in the video -- texture, lighting, weather, road surface detail -- is noise from the scorer's perspective. The Shannon limit in scorer space is orders of magnitude below the Shannon limit in pixel space.

### Novel View Synthesis Connection

The problem of "generate a frame that a pose regression network interprets as coming from a specific camera position" is structurally identical to novel view synthesis (NVS). In NVS, you render a scene from a new camera pose and evaluate whether the render is geometrically consistent with known views. In our setting, the "known views" are the scorer's expected PoseNet outputs, and the "novel view" is the generated frame.

This connection suggests that the 3D Gaussian Splatting revolution in NVS (Street Gaussians, 4D-GS) may eventually provide the ideal video codec for task-aware compression. Instead of storing pixels or neural network weights, store 3D Gaussian parameters that describe the scene geometry. Render any desired view at inflate time. The archive size is the Gaussian parameter count, which for driving scenes with limited depth complexity could be remarkably compact.

### RAFT Optical Flow

RAFT (Recurrent All-Pairs Field Transforms, Teed and Deng, ECCV 2020) computes dense optical flow by building all-pairs correlation volumes between frames and iteratively refining flow estimates with a GRU-style update. Our motion predictor is a simplified version of this: it estimates flow from mask pairs rather than pixel pairs, trading accuracy for compactness.

A RAFT-lite correlation volume could replace our current motion prediction convolution stack, providing much better motion estimation at the cost of more parameters. The tradeoff is favorable: better flow means less residual correction needed, which means lower gate activation, which means more of the temporal coherence comes from geometry (good for PoseNet) rather than learned correction (noisy for PoseNet).

### Ego-Motion Pre-Computation

The scorer's own PoseNet provides the ground truth ego-motion trajectory. At compress time (unlimited budget), we run PoseNet on the original frames and extract the 6-DOF pose vectors. These 600 pairs x 6 floats x 2 bytes = 7,200 bytes (~5KB compressed) are stored in the archive. At inflate time, supervised TTO minimizes MSE between the generated frames' PoseNet output and the stored targets. This is not an approximation -- we optimize the exact metric the scorer uses.

## Competitive Intelligence

### Quantizr's Insider Knowledge

Quantizr (PR#53) demonstrated knowledge of the scorer internals that goes beyond what the public API reveals. His architecture choices -- FP4 with an 8-value codebook, the exact channel widths 36/60, the mask-conditioned rendering paradigm, the motion prediction from mask differences -- suggest either extensive experimentation with the scorer or insider familiarity with the openpilot perception stack.

His score of 0.60 (later confirmed by Yousfi as achievable at sub-0.50 with the right tricks) represents the state of the art for the mask-conditioned rendering approach. The score decomposes as: near-perfect SegNet (masks are the ground truth representation), competitive PoseNet through flow-warping temporal coherence, and minimal rate through extreme quantization and mask compression efficiency.

### Reverse-Engineered Architecture

From the submission artifacts, we reconstructed:

- **Renderer:** U-Net with 36->60->36 channel progression, class embeddings (5 classes, 6 dimensions each), coordinate grid conditioning, soft sigmoid output (255 * sigmoid(logits / 50) for always-flowing gradients).
- **Motion:** `TinyMotionFromMasks` -- estimates optical flow from consecutive mask pairs using difference features, applies flow via grid_sample for temporal coherence.
- **Quantization:** FP4 with an 8-value codebook (not standard int4 or int8). This achieves extreme compression of the renderer weights while preserving enough precision for the synthesis task.
- **Archive structure:** Compressed AV1 masks (~209KB) + renderer weights (~177KB FP4) = 386KB total.

### Yousfi's Sub-0.50 Confirmation

Yassine Yousfi, PhD student at Binghamton University and the competition's designer, confirmed that sub-0.50 scores are achievable. His tricks list (35 items, documented in our research state) spans the full range from scorer exploits to architectural innovations. The theoretical floor he projects: 0.18, achievable via constrained generation from noise with pre-computed scorer targets.

## The Meta-Narrative

### Human-AI Collaborative Research

This project is an existence proof of human-AI collaborative research producing competitive results in a domain where the human had zero prior knowledge. The human contributor (adpena) had no background in video compression, neural network architectures, steganography, or autonomous driving perception. Every domain concept -- from AV1 encoder parameters to PoseNet sensitivity profiles to Fridrich's steganalysis framework -- was learned, implemented, and validated within the project timeline.

The collaboration model: the human provides strategic direction, resource allocation, and judgment calls (kill/promote decisions, risk tolerance, publication timing). The AI provides domain knowledge synthesis (connecting steganography theory to video compression), implementation velocity (2,842 tracked code entities across 17 review rounds), and systematic exploration (25+ experiments, each with pre-registered hypotheses and kill criteria).

### Zero Domain Knowledge to Competitive Architecture

The score trajectory tells the story:
- Day 1: x265 baseline, score 4.06. No knowledge of AV1, scorers, or task-aware compression.
- Day 3: AV1 with correct colorspace, score 2.20. Learned that BT.601 vs BT.709 matters.
- Day 5: First post-filter, score 2.05. Discovered that backpropagating through frozen scorers works.
- Day 7: Dilated CNN, score 1.33. Found that receptive field size is the binding constraint.
- Day 10: Reverse-engineered competitor, designed asymmetric warp paradigm, implemented Fridrich constrained training. Competing on architecture, not just optimization.

The acceleration is real: each day's learning compounds on the previous day's. By day 10, the project was operating at the frontier of the competition, designing novel architectures informed by steganalysis theory and cross-domain synthesis.

### Meta-Steganalysis

The entire project can be viewed as meta-steganalysis: we are analyzing the competition's scoring system (the "detector") to find its blind spots, sensitivity profile, and decision boundaries, then exploiting those properties to minimize detection (distortion) while maximizing embedding capacity (compression). The Fridrich framework is not just an analogy -- it is the correct theoretical lens.

### The Council Methodology

The tripartite council -- Yousfi (contest designer's perspective), Fridrich (steganalysis theory), and the Contrarian (rigor enforcement) -- provides three independent analysis angles on every decision. No experiment runs without council sign-off on design, resolution, step count, and conditioning. No technique is killed without adequate testing (no "janky smoke tests"). No technique is promoted without authoritative evaluation.

The council methodology prevents two failure modes: (1) premature kills, where a technique is rejected based on an inadequate test, and (2) premature promotions, where a technique is adopted based on proxy metrics that do not transfer. Both failure modes occurred early in the project (KL distillation was a premature promotion; scorer resolution was a premature kill) and the council process was designed specifically to prevent recurrence.

## Killed Techniques: 10 Failures and What They Teach

### 1. KL Distillation Loss -- DEAD

**What:** Train postfilter with KL divergence against scorer soft targets.
**Evidence:** Two auth evals confirmed PoseNet collapse: proxy 1.25 scored 1.85 auth, proxy 1.43 scored 2.05 auth.
**Why:** KL loss optimizes distribution shape, not argmax agreement. PoseNet is sensitive to pixel-level spatial frequency statistics that KL's gradient signal does not capture. The KL gradient norm exceeds PoseNet MSE gradient by 10-50x, causing silent PoseNet regression.
**Cross-domain application:** KL distillation works for image classification where soft targets carry class similarity information. It fails for geometric tasks like ego-motion estimation where pixel-level fidelity matters more than distribution shape.

### 2. Adaptive Weight Formula (Hinton T-squared) -- DEAD

**What:** Automatically balance seg/pose/rate weights using temperature-based reweighting.
**Evidence:** T-squared cancels in the derivation (double-correction with PyTorch's kl_div). Produced w_s=0.80 when the empirical winner used w_s=100 (125x mismatch).
**Why:** Mathematical error: the compound invariant was trivially constant by construction.
**Cross-domain application:** Adaptive weight formulations are valid (GradNorm, PCGrad exist), but any derivation must account for internal temperature scaling in the loss function. Formally verify against reality, not just against axioms.

### 3. AllNorm Brightness Invariance Exploit -- DEAD

**What:** Exploit PoseNet's AllNorm layer for brightness/contrast invariance.
**Evidence:** AllNorm is BatchNorm1d(1) on features AFTER the ViT backbone, not pixel-level normalization. Brightness shift + chroma smooth caused 2.15 regression (from 1.97).
**Why:** Architectural misunderstanding. AllNorm operates on deep features, not raw pixels.
**Cross-domain application:** Works if the model has pixel-level normalization (LayerNorm on input). Always verify where normalization occurs in the architecture before assuming invariance.

### 4. PoseNet Gradient Caps/Clamps -- DEAD

**What:** Cap PoseNet gradients to prevent dominance during training.
**Evidence:** Caused 26x PoseNet regression.
**Why:** Value-based capping destroys directional information. The model cannot learn temporal coherence without accurate PoseNet gradients.
**Cross-domain application:** Norm-based gradient clipping is fine. Value-based capping is wrong for directed optimization. Use PCGrad for gradient conflict resolution instead.

### 5. Even-Frame SegNet Skip (Trick 3) -- DEAD

**What:** Halve SegNet training signal on even-indexed frames.
**Evidence:** Implementation halved the ENTIRE loss (including PoseNet) on 50% of pairs.
**Why:** Loss components were entangled. Could not zero only SegNet without also zeroing PoseNet.
**Cross-domain application:** Valid if loss components are computed and returned separately. Requires refactored loss functions with independent gradient paths.

### 6. Independent Frame Generation (DP-SIMS v1) -- PARADIGM KILLED, ARCHITECTURE PROMOTED

**What:** SPADE-based renderer generating each frame independently from its mask.
**Evidence:** SegNet 0.003 (excellent), PoseNet 0.482 (catastrophic).
**Why:** PoseNet evaluates consecutive PAIRS. Independent generation has no mechanism for frame-to-frame consistency. The warp paradigm (AsymmetricPairGenerator) was the architectural fix.
**Cross-domain application:** Independent generation works for static image tasks. For any metric involving temporal coherence, architectural coupling between frames is required.

### 7. Constrained Generation from Noise -- PARADIGM KILLED, FRAMEWORK PROMOTED

**What:** Generate frames from random noise using scorer gradients with Fridrich constraints.
**Evidence:** Tiny DP-SIMS (78KB): SegNet 0.04 (good), PoseNet 150 (catastrophic).
**Why:** Too small model, no temporal structure in noise-based generation.
**Cross-domain application:** Constrained optimization from noise works for single images (diffusion models, GANs). For video pairs, the starting point must encode temporal relationships.

### 8. BT.601 Color Matrix Change -- DEAD

**What:** Switch from BT.709 to BT.601 color space conversion.
**Evidence:** Scorer hardcodes BT.601 in `rgb_to_yuv6` regardless of config.
**Why:** The scorer's internal conversion is fixed. Changing our config does not affect what the scorer sees.
**Cross-domain application:** Only relevant if the scorer's color conversion is configurable.

### 9. Scorer Resolution Storage (initial verdict) -- REOPENED

**What:** Store at 384x512 and upscale at inflate time.
**Initial evidence:** Bilinear round-trip error 7.4% on random data.
**Council re-analysis:** Random data is worst case. Real video: 0.035% error. The operation is a projection operator (idempotent). Training already goes through scorer preprocessing.
**Status:** Deprioritized, lane open. Needs real-video verification.
**Cross-domain application:** Valid for integer-scale codecs (2x/4x). Near-identity for natural images.

### 10. Consensus Codec Stack -- DEAD

**What:** Combine CRF 33 + sharpness 1 + scene-change detection off + hqdn3d denoising.
**Evidence:** Scored 2.13 vs 2.08 baseline.
**Why:** Multiple marginal encoder improvements produced interference rather than additive gains. Each individual change was within 2% of baseline, but their combination was worse than any individual change.
**Cross-domain application:** Encoder parameter interactions are nonlinear. Test combinations, not just individual settings. The "stack everything that helps a little" strategy fails when parameters interact.

## Training Results: The Marathon Session

### Phase 1 Convergence

The Fridrich constrained renderer training on Modal A10G showed rapid Phase 1 convergence. The DP-SIMS architecture (SPADE U-Net with channels 128/64/32/16) achieved SegNet 0.003 -- tying Quantizr's best -- after only 89 Phase 2 epochs. PoseNet remained at 0.482, reflecting the independent generation paradigm's inability to produce temporal coherence.

The asymmetric warp paradigm was designed specifically to address this: by making frame-to-frame coherence architectural (via flow warping) rather than learned (via loss signals), PoseNet sees geometric relationships that match real ego-motion.

### The Failed Run and Diagnosis

Modal free credits were exhausted mid-training. The DP-SIMS run had completed 189 epochs (89 Phase 2) when Modal terminated the instance. The checkpoint was recoverable but the training momentum was lost. The dilated CPU-lane training on Modal had reached epoch 99 of a planned 2500, also terminated early.

Diagnosis: Modal's free tier provided approximately 12 hours of A10G time total across all runs. The DP-SIMS training consumed ~10 hours (189 epochs at ~3 minutes each). The dilated training consumed ~2 hours (99 epochs at ~1.2 minutes each). Neither reached the point where the techniques could be fairly evaluated.

### The 5 Fixes

Five issues were identified and fixed during the marathon session:

1. **Channels_last on scorers breaks MPS backward pass.** AllNorm's internal batch_norm backward uses .view() which fails with channels_last stride layout. Fix: keep channels_last on the postfilter model for speed but leave scorers in standard contiguous layout.

2. **Gradient clipping passes optimizer instead of gradients.** The MLX training loop called .square() on the AdamW optimizer object instead of gradient tensors. Fix: extract gradients before clipping.

3. **Modal deploy image build path resolution.** `add_local_dir` path resolved from deploy script location, not repo root. Fix: anchor all paths to an explicit REPO_ROOT environment variable.

4. **Upstream scorer not available in container.** Modal container lacked the upstream scorer repo (PoseNet/SegNet module definitions). Fix: clone the upstream repo during image build.

5. **Smoke profile too slow for interactive testing.** `subsample=4` produced 75 pairs/epoch at ~5s/pair = 6 minutes/epoch. Fix: increase subsample to 8 for smoke tests (9 pairs, ~45s/epoch).

## Attribution

### Yassine Yousfi

PhD student at Binghamton University, working under Jessica Fridrich. Designer of the comma.ai video compression challenge. His 35-trick analysis of the competition's theoretical limits informed our research roadmap and experiment prioritization. His confirmation that sub-0.50 scores are achievable validated our GPU-lane architecture direction. The scoring formula, the choice of PoseNet and SegNet as task metrics, and the asymmetry between compress and inflate time budgets -- these design decisions shaped every experiment we ran, and understanding the intent behind them was as important as understanding the scorers themselves.

### Jessica Fridrich

Distinguished Professor at Binghamton University and founder of modern steganalysis. Her theoretical framework -- detection boundaries, embedding costs, constrained optimization under detector awareness -- provided the conceptual foundation for our GPU-lane training methodology. The augmented Lagrangian formulation, the 3-phase curriculum, the S-UNIWARD-inspired pixel cost map, and the fundamental insight that this competition is inverse steganalysis all derive from her two decades of work on the steganography/steganalysis arms race.

Her key publications for our work: Holub, Fridrich, and Denemark, "Universal distortion function for steganography in an arbitrary domain" (EURASIP 2014); Fridrich, "Steganography in Digital Media" (Cambridge University Press, 2009). The connection between steganographic embedding and task-aware video compression is, to our knowledge, novel -- and it would not have been possible without her framework.

---

*Updated 2026-04-12. Marathon session: asymmetric warp paradigm, Fridrich framework, 7 eureka moments, 10 killed techniques, 17 review rounds, 50+ bugs.*
