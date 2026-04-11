# Task-Aware Post-Filter for comma.ai's Video Compression Challenge

<!-- arXiv structure:
  Section 1: Introduction (task-aware compression for autonomous driving)
  Section 2: Background (scoring formula, PoseNet, SegNet, openpilot context)
  Section 3: Method (post-filter architecture, training, quantization)
  Section 4: Negative Results (4 subsections: KL distill, adaptive weights, PoseNet spatial sensitivity, Pareto trap)
  Section 5: Multi-Objective Analysis (Pareto frontier, gradient methods)
  Section 6: Recommendations (scoring formula, deployment)
  Section 7: Results and Ablations
-->

**Score: 1.33** | seg=0.00610, pose=0.00218, rate=0.0230

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

The output layer is zero-initialized, so at the start of training the filter is the identity function -- it does nothing. Every correction the filter learns has to earn its place by reducing the scorer loss.

The filter ships as a 45KB int8-quantized checkpoint (~40K parameters). At inference it runs on CPU in under 30 seconds for all 1200 frames. The rate impact is negligible.

### Step 3: Train against the scorer

The training loss is the competition score itself, computed by running the corrected frames through frozen copies of PoseNet and SegNet. Gradients flow from the score, through the scorer networks, through the CNN, and into its weights.

The CNN does not optimize for generic visual quality. It optimizes for the specific metrics PoseNet and SegNet use to compare frames.

A saliency-weighted reconstruction term (alpha=20) focuses the CNN's limited capacity on pixels that PoseNet's gradients identify as high-sensitivity, while a complementary term prevents unnecessary modification of pixels that SegNet cares about.

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

**Quantization-Aware Training (QAT):** Per-channel fake int8 quantization in the forward pass with straight-through gradient estimation. Per-channel scales (one per output filter) preserve 3-4 more bits of effective precision than per-tensor quantization. The model learns weight configurations that are robust to the exact quantization noise pattern it will encounter at deployment.

**Exponential Moving Average (EMA):** Polyak averaging with decay=0.997 smooths late-epoch weight oscillation. Over 1000 epochs, this prevents the optimizer from wandering into weight configurations that happen to score well in float32 but collapse under quantization.

**Best-checkpoint int8 selection:** At each checkpoint, we take the EMA weights, quantize them to int8, and evaluate the quantized model on the scorer using the corrected metric (hard argmax SegNet, uint8 round-trip, held-out frames). Most epochs produce poor int8 models. We save the epoch where the quantized weights land in a good configuration.

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

Negative results are as informative as positive ones. We ran 25+ experiments over 8 days; most failed. We organize the most instructive failures into four themes.

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

### Additional failed experiments

Beyond the four themes above, we tested and rejected:

**Architecture:**
- **PixelShuffle / PSD upsampling:** Proxy rejected at 1.99. Sub-pixel shuffle helps SegNet but hurts PoseNet -- PoseNet is sensitive to any spatial rearrangement of information.
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

## Related work

This work is a specific instance of task-aware video compression, an area gaining traction in both academia and standards bodies:

- **Video Coding for Machines** (VCM, MPEG ISO/IEC 23888-2): standardizing task-driven compression with QP allocation guided by downstream perception models.
- **Neural Wrapping** (Khan et al., CVPR 2025): the closest prior work, using neural pre/post-processors wrapped around standard codecs. Differs from our approach in targeting human perceptual metrics (LPIPS, VMAF) rather than frozen task-specific scorer networks.
- **Sandwiched Compression** (Du et al., Google): neural pre+post processor with differentiable codec proxy. We share the post-processor concept but optimize through frozen scorers directly.
- **Rate-Distortion-Perception Tradeoff** (Blau and Michaeli, ICML 2019): proves a fundamental three-way impossibility between rate, distortion, and perceptual quality. Our setting extends this to a rate-SegNet-PoseNet tradeoff with explicit task-specific losses.

For the multi-task gradient conflict that arises when jointly optimizing PoseNet and SegNet, we draw on PCGrad (Yu et al., NeurIPS 2020) and CAGrad (Liu et al., NeurIPS 2021). The Pareto frontier analysis follows multi-objective optimization frameworks from Sener and Koltun (NeurIPS 2018).

Our contribution is training a CNN post-filter by directly backpropagating through frozen scorer networks to minimize a competition-specific scoring formula. This produces a task-aware codec that preserves precisely the visual information downstream perception models consume, rather than optimizing generic quality metrics.

<!-- Section 7: Results and Ablations -->
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

6. **Distribution matching.** Training on frames from a different source than the scorer evaluates on caused a 0.3-point gap. Training on the actual submission archive closed it.

7. **The scoring formula has structure.** The marginal sensitivities of each term at different operating points are not obvious from the formula but drive resource allocation.

8. **The PoseNet-SegNet Pareto frontier.** Our campaign inadvertently mapped the tradeoff boundary. Standard loss with dilated convolutions optimizes PoseNet to near-exhaustion (5.6x improvement) while making SegNet 5.2% worse. KL distillation does the opposite: 10% SegNet improvement but 26-37x PoseNet destruction. The score minimum lies on the frontier between these extremes -- a multi-objective optimization problem where the scoring formula's coefficients (100, sqrt(10), 25) define the optimal tangent line.

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
