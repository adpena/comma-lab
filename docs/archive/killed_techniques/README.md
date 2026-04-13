# Killed Techniques Registry

Techniques that were investigated and conclusively killed for THIS competition.
Preserved for signal, cross-domain applications, and the writeup/paper.

**No signal loss.** Every kill has: what we tried, what happened, WHY it failed,
and where the technique might work in other contexts.

---

## Conclusively Killed (with evidence)

### 1. KL Distillation Loss
- **What:** Train postfilter with KL divergence against scorer soft targets
- **Evidence:** Two auth evals confirmed PoseNet collapse (proxy 1.25→auth 1.85, proxy 1.43→auth 2.05)
- **Why it failed:** KL loss optimizes distribution shape, not argmax agreement. PoseNet is sensitive to pixel-level changes that KL loss doesn't capture.
- **Where it might work:** Image classification where soft targets are meaningful. NOT for geometric tasks like ego-motion estimation.
- **Commits:** Multiple, documented in project memory
- **Status:** DEAD — do not resurrect for this competition

### 2. Adaptive Weight Formula (Hinton T²)
- **What:** Automatically balance seg/pose/rate weights using temperature-based reweighting
- **Evidence:** T² cancels in the derivation, making the formula vacuous. Produced w_s=0.80 when empirical winner used w_s=100 (125x mismatch).
- **Why it failed:** Mathematical error in the derivation. The compound invariant w_s*T² was trivially constant by construction.
- **Where it might work:** Nowhere — the math is wrong. The concept of adaptive weights is valid but needs a different formulation (e.g., GradNorm, PCGrad).
- **Status:** DEAD

### 3. AllNorm Brightness Invariance Exploit
- **What:** Exploit PoseNet's AllNorm layer for brightness/contrast invariance
- **Evidence:** AllNorm is BatchNorm1d(1) on features AFTER the ViT backbone, NOT pixel-level normalization. Brightness shift + chroma smooth caused 2.15 regression (from 1.97).
- **Why it failed:** Misunderstanding of the architecture. AllNorm operates on deep features, not raw pixels.
- **Where it might work:** If the scorer used pixel-level normalization (LayerNorm on input), brightness invariance would be exploitable. Check architecture before assuming.
- **Status:** DEAD

### 4. PoseNet Gradient Caps/Clamps
- **What:** Cap PoseNet gradients to prevent dominance during training
- **Evidence:** Caused 26x PoseNet regression
- **Why it failed:** Capping gradients destroys the directional information PoseNet provides. The model can't learn temporal coherence without accurate PoseNet gradients.
- **Where it might work:** Gradient clipping (norm-based, not value-based) is fine. Value-based capping is always wrong for directed optimization.
- **Status:** DEAD

### 5. even_frame_skip_seg (Trick 3)
- **What:** Halve SegNet training signal on even-indexed frames
- **Evidence:** Implementation halved the ENTIRE loss (including PoseNet) on 50% of pairs
- **Why it failed:** Bug — loss components were entangled. Could not zero only SegNet without also zeroing PoseNet.
- **Where it might work:** If loss components are computed and returned separately, per-component scaling is valid. Requires refactored loss functions.
- **Status:** DEAD (replaced by frame_t1-only SegNet, which is correct)

### 6. Scorer Resolution Exploit (Eureka 3)
- **What:** Store frames at 384x512 (scorer resolution) and upscale at inflate time for 5.2x rate reduction
- **Initial evidence:** Bilinear round-trip 384→874→384 is NOT identity on random data. Mean pixel error 18.8 (7.4%).
- **Council re-analysis:** The initial test was on RANDOM data (worst case). Real video is spatially smooth → 0.035% error. The round-trip D(U(x)) is a PROJECTION OPERATOR — applying it twice is idempotent. Training already goes through scorer preprocessing, making the renderer implicitly round-trip-aware.
- **Three remaining paths:**
  1. Test on real video with actual models (not just math-only mode)
  2. Verify training loss goes through scorer preprocessing (it does)
  3. Apply projection fixup at inflate: render→upscale→downscale→upscale→write (free, idempotent)
- **Quantizr's approach:** Uses Lanczos + learned REN post-filter, NOT raw bilinear. His REN compensates for upscale artifacts scorer-specifically.
- **Verification script:** `experiments/verify_scorer_resolution.py`
- **Status:** DEPRIORITIZED (lane kept open) — initial "KILLED" verdict overturned by council. Needs real-video testing before closing.

### 7. SIREN (Sinusoidal Representation Networks) for Video Memorization
- **What:** Overfit a per-pixel MLP to memorize the video. Weights ARE the compressed representation.
- **Evidence:** 21.7 dB PSNR at 1/4 res, 500 steps. BUT: test was at reduced resolution with insufficient steps — "janky smoke test" per council rules.
- **Why it was killed:** Insufficient signal from the test, not conclusive failure. The technique was deprioritized in favor of the mask-conditioned renderer approach (which proved more promising).
- **Where it might work:** NeRV (Neural Representations for Videos) literature shows this CAN work at scale with proper architecture. Our test was underpowered. Worth revisiting post-competition with proper resolution and step count.
- **Status:** DEPRIORITIZED (not conclusively killed — lane kept open per CLAUDE.md)

### 8. BT.601 Color Matrix Change
- **What:** Switch from BT.709 to BT.601 color space conversion
- **Evidence:** Scorer hardcodes BT.601 in rgb_to_yuv6 regardless of config
- **Why it failed:** The scorer's internal conversion is fixed. Changing our config doesn't affect what the scorer sees.
- **Where it might work:** Only if the scorer's color conversion were configurable (it's not).
- **Status:** DEAD

---

## Partially Killed (useful components remain)

### 9. DP-SIMS with Independent Frame Generation
- **What:** SPADE-based renderer generating each frame independently from its mask
- **Evidence:** SegNet 0.003 (excellent), PoseNet 0.482 (catastrophic). Independent generation cannot produce temporal coherence.
- **Why PoseNet failed:** PoseNet evaluates consecutive PAIRS. Independent generation has no mechanism for frame-to-frame consistency.
- **What survived:** The MaskRenderer (SPADE U-Net) architecture IS used inside AsymmetricPairGenerator. The independent generation paradigm was killed; the renderer architecture was promoted.
- **Status:** ARCHITECTURE PROMOTED, PARADIGM KILLED

### 10. Constrained Generation from Noise
- **What:** Generate frames from random noise using scorer gradients with Fridrich constraints
- **Evidence:** Tiny DP-SIMS (78KB): SegNet 0.04 (good), PoseNet 150 (catastrophic)
- **Why it failed:** Too small model, no temporal structure in noise-based generation
- **What survived:** The Fridrich constrained optimization framework (augmented Lagrangian) IS used in the training loop. The "from noise" paradigm was killed; the optimization framework was promoted.
- **Status:** FRAMEWORK PROMOTED, PARADIGM KILLED

---

## Cross-Domain Application Notes

Many killed techniques have valid applications outside this specific competition:

| Technique | Valid Domain | Why It Works There |
|-----------|-------------|-------------------|
| KL distillation | Image classification | Soft targets carry class similarity info |
| Brightness invariance | Lighting-robust recognition | If model has input normalization |
| SIREN memorization | NeRF, image fitting | With proper resolution and steps |
| Resolution exploit | Integer-scale codecs | 2x/4x scales are near-identity |
| Gradient caps | Stabilizing GAN training | Value caps prevent mode collapse |
| Independent rendering | Static image generation | No temporal coherence needed |
| Noise-based generation | Unconditional image synthesis | GANs, diffusion models |

---

## How to Use This Document

1. **Before proposing a technique:** Check if it's killed here
2. **Before killing a technique:** Verify the test was adequate (no janky smoke tests)
3. **When writing the paper:** This is the ablation study — what we tried and why it failed
4. **When exploring new domains:** Check the cross-domain notes for techniques that might transfer
