# Jack-from-Skunkworks Council — SegNet + Rate Research Survey
**Date:** 2026-04-28
**Mission:** Survey 2024-2026 literature on SegNet attack + Rate compression for the comma.ai video compression challenge frontier (Lane G v3 = 1.05; target sub-Quantizr 0.33).
**Budget context:** $200-$500 on Vast.ai (per user 2026-04-28). H100/A100/4090 multi-day runs OK; lanes up to $50 affordable.
**Method:** ~30 WebSearches + 5 WebFetches across arXiv / CVPR / ICLR / NeurIPS / GitHub. 18+ existing lane portfolio cross-referenced for distinctness.

---

## Wedge attribution (from Lane G v3 1.05 baseline)
- SegNet: 0.401 score points (38%)
- Rate: 0.462 score points (44%)
- PoseNet: 0.186 score points (18%)
- Quantizr at 0.33: ~0.090 SegNet + ~0.040 PoseNet + ~0.200 rate

---

## SegNet Survey (top 5)

### S1. Lane J-JBL — Jaccard Metric Loss + Boundary Label Smoothing
- arXiv 2302.05666 (Wang et al., NeurIPS 2023, refined 2024)
- JML is a soft-label-compatible Jaccard surrogate that subsumes Lovász-Softmax. BLS injects boundary "dark knowledge" into KD. +2-5 mIoU across 13 architectures including EfficientNet; calibration improves *near class boundaries* (which IS our argmax disagreement signal).
- **Mechanism**: replace standard KL distill on SegNet logits with JML+BLS during compress-time training. Weight boundary pixels 3-5×.
- **Cost**: $1.50 / 8h on 4090
- **Predicted band**: [0.92, 1.02]
- **Distinctness**: ✓ Lane SG protects layers; Lane MOS biases priors; Lane SAUG redraws sigma. None do *boundary pixel weighting*.

### S2. Lane J-EFD — Edge-Aware Mix Diffusion + Class-Flipping Noise
- AAAI 2026 (Wang et al.)
- Class-flipping noise with HIGHER probability AT BOUNDARIES forces robust boundary representation. Directly attacks EfficientNet-B2 stride-2 stem blind spot.
- **Mechanism**: at training time perturb GT mask labels along class boundaries (computed via morphological gradient). Force renderer to predict un-flipped argmax.
- **Cost**: $2.00 / 10h on 4090
- **Predicted band**: [0.88, 1.00]
- **Distinctness**: ✓ No existing lane perturbs labels at class boundaries.

### S3. Lane J-SEM — SCENE-style Mask Preprocess
- arXiv 2601.22189 (2026)
- Lightweight semantic-aware preprocessing achieves +10.6 VMAF over vanilla AV1.
- **Mechanism**: tiny 5K-param preprocessor that learns which mask regions to over-bit based on SegNet's class-boundary activation map. Apply BEFORE AV1 encoding of masks.mkv.
- **Cost**: $0.50 / 4h on 4090
- **Predicted band**: [0.95, 1.05]
- **Distinctness**: partial overlap with Lane MOS (decoder bias); this biases the encoder. Stack-able.

### S4. Lane J-IRD — Inter-Resolution Knowledge Distillation
- arXiv 2401.06010 (2024)
- Train teacher at HIGH-RES (768×512) and distill into student at native (384×512). Student inherits boundary structure low-res teacher would miss.
- **Mechanism**: teacher = SegNet evaluated at 768×512 GT; student = our renderer. Renderer learns 384×512 output that, when bilinear-upsampled to 768×512, matches high-res argmax.
- **Cost**: $1.00 / 6h on 4090
- **Predicted band**: [0.95, 1.04]
- **Distinctness**: ✓ No existing lane uses higher-resolution teacher.

### S5. Lane J-ADV — Detector-Informed Embedding (Yousfi/Fridrich canonical)
- arXiv 1803.09043 + Yousfi/Dworetzky/Fridrich detector-informed batch steganography (NSF 2022-2024)
- Selectively replace/rearrange image elements via gradients from target CNN detector. THE Fridrich-canonical inverse-steganalysis paradigm.
- **Mechanism**: compute SegNet logit gradients w.r.t. mask pixels; perturb only highest |∂logit/∂pixel| × low(local_variance) pixels (UNIWARD weighting). Renderer outputs argmax-stable masks even after AV1 quantization.
- **Cost**: $2.50 / 12h on 4090
- **Predicted band**: [0.85, 1.00]
- **Distinctness**: ✓ Existing lanes have UNIWARD textured-region weighting in spirit but no lane does gradient-of-detector × local-variance masking.

---

## Rate Survey (top 5)

### R1. Lane J-RFSQ — Robust Residual FSQ
- arXiv 2508.15860 (Aug 2025)
- Multi-stage FSQ with learnable scaling + invertible LayerNorm. 28.7% L1 reduction + 45% perceptual loss reduction vs FSQ/VQ-EMA/LFQ. Cosmos uses FSQ; RFSQ improves it.
- **Mechanism**: replace current FP4 codebook (Lane F lineage) with RFSQ codes for renderer.bin. Codebook itself is invertible-norm + scaling, ~2KB extra.
- **Cost**: $3.00 / 16h on 4090
- **Predicted rate gain**: ~30-50KB on renderer.bin (170KB → 120-140KB) → −0.020 to −0.034 rate
- **Predicted band**: [0.99, 1.03] assuming distortion holds
- **Distinctness**: ✓ Lane F is FP4 fake-quant (CC 8.9 SIMULATED). RFSQ is genuinely smaller storage AND learnable structure.

### R2. Lane J-NWC — Neural Weight Compression
- arXiv 2510.11234 (late 2025)
- Train neural codec on weight tensors. SOTA 4-6 bits/weight with smooth tradeoff. "Gains extend across diverse architectures, e.g., vision encoders."
- **Mechanism**: pretrain a tiny weight-codec on a corpus of small renderers we've trained (we have hundreds in reports/). Use codec to compress final renderer.bin.
- **Cost**: $5.00 / 24h on 4090
- **Predicted rate gain**: 4 bits/weight on 88K params = 44KB renderer (vs current 170KB) → −0.084 rate
- **Predicted band**: [0.92, 1.02] if distortion preserved
- **Distinctness**: ✓ Lane Ω-V2 is per-element learnable Lagrangian bits; NWC learns the *codec* itself from a corpus. Compositional.

### R3. Lane J-DCAE — Dictionary-based Cross Attention Entropy
- arXiv 2504.00496 (CVPR 2025)
- Learnable dictionary summarizes typical structures. -17% BD-rate on Kodak vs MLIC++ at 4× lower latency.
- **Mechanism**: replace AV1 static Brotli post-pass with dictionary-based entropy coder trained on mask_codec output stream. Dictionary lives in archive (<30KB), amortized across 1200 frames.
- **Cost**: $2.00 / 10h on 4090
- **Predicted rate gain**: 50-100KB shaved from masks.mkv → −0.033 to −0.066 rate
- **Predicted band**: [0.98, 1.04]
- **Distinctness**: ✓ Lane EBR is Ballé hyperprior on poses/renderer; this is a dictionary specifically for the mask bitstream.

### R4. Lane J-ASYM — Asymmetric Tiny Decoder for Masks
- arXiv 2412.17270 + CVPR 2025 (DCAE paradigm)
- Heavy compress-time encoder + tiny inflate-time decoder. Encoder doesn't ship.
- **Mechanism**: train 30-50K param mask decoder + 1M+ param mask encoder; only the tiny decoder ships. Encoder runs at compress time.
- **Cost**: $4.00 / 18h on 4090
- **Predicted rate gain**: mask data 450KB → ~150KB → −0.200 rate
- **Predicted band**: [0.78, 0.95]
- **Distinctness**: partial overlap with Lane I (Cool-Chic INR decoder). AsymLLIC is encoder-decoder split rather than coordinate-MLP. Run alongside.

### R5. Lane J-FAC — Format-Aware Coder
- arXiv 2508.19263 (2025)
- Studies entropy-coded compression of NN weights in BF16/FP8/FP4. General-purpose Zstd/Brotli FAILS to exploit structured exponent-mantissa distribution.
- **Mechanism**: replace generic Brotli on FP4 codebook indices + scales with custom rANS + dictionary tuned to OUR renderer's empirical weight distribution (computed offline).
- **Cost**: $0.30 / 2h on CPU
- **Predicted rate gain**: 5-15KB savings → −0.003 to −0.010 rate (small but FREE)
- **Predicted band**: [1.04, 1.05]
- **Distinctness**: ✓ Nothing in portfolio touches post-Brotli format-awareness.

---

## Cross-cutting (top 3)

### X1. Lane J-MAETOK — MAETok-style Renderer Tokenizer Training
- ICML 2025 (arXiv 2502.03444)
- Mask modeling on autoencoders → semantically rich latents + 76× faster training.
- **Mechanism**: hybrid with Lane MAE-V — but mask the renderer's *latent feature maps* (not input mask) during training. Forces sparser, more compressible representations.
- **Cost**: $2.00 / 10h on 4090
- **Predicted band**: [0.95, 1.03]

### X2. Lane J-LUT — LUT-based Renderer Upsampler
- arXiv 2503.06617 + VoLUT (2025)
- Replace neural upsamplers with LUT operations. LUT is small table of bytes that compresses very well.
- **Mechanism**: replace renderer's final upsampling layer (parameter-heavy) with learned 3D LUT. Smaller archive bytes, respects strict-scorer-rule.
- **Cost**: $1.00 / 6h on 4090
- **Predicted band**: [0.98, 1.04], ~20-30KB savings

### X3. Lane J-IMP — Iterative Magnitude Pruning at Scale
- arXiv 2406.01820 + ICLR 2026 papers
- PX (Spectral Foresight Pruning) confirms LTH superiority for semantic segmentation. With $200-500 budget, full IMP is now affordable.
- **Mechanism**: 10-cycle IMP on dilated-h64 baseline. Each cycle: train → prune 20% smallest → rewind → retrain. Result: 90% sparse renderer.
- **Cost**: $20-30 / 60h on 4090 (now in budget)
- **Predicted band**: [0.85, 1.00] — historically LTH gives 2-5× compression at minimal accuracy loss
- **Distinctness**: Lane Ω-V2 is per-element bit allocation; IMP is *removal*. They compose cleanly (prune first, quantize remaining).

---

## Top 3 Highest-EV Lanes to Dispatch (Cycle 1 Jack)

### TOP-1: Lane J-JBL ($1.50 / 8h, [0.92, 1.02])
SegNet wedge = 38% × 1.05 = 0.40 pts. BLS+JML +2-5 mIoU near boundaries. Conservative 20% reduction in SegNet distortion: 0.004 → 0.0032 → −0.08 score. Lowest cost, highest confidence.

### TOP-2: Lane J-NWC ($5.00 / 24h, [0.92, 1.02])
Renderer.bin 170KB → 44KB at 4 bits/weight = −126KB = −0.084 rate. If distortion holds (paper says yes for vision encoders), −0.08 score.

### TOP-3: Lane J-IMP ($25 / 60h, [0.85, 1.00])
LTH at 90% sparsity + Lane Ω-V2 quantization = 88K → 9K active × 4 bits = 4.5KB renderer. Massive rate attack, MEDIUM-HIGH risk (10-cycle dependency chain).

**Cycle 1 Jack total: $31.50 / ~80 GPU-hours combined.**

---

## NOT-applicable (transparency)

- **ARCHE (arXiv 2603.10188)**: 95M params, 222ms inference. Disqualified by parameter budget.
- **C3 (ICLR 2025)**: Already covered by Lane I (Cool-Chic).
- **HiNeRV / NeRV variants**: Lane I covers this design space.
- **HVQ-CGIC**: Targets generative distortion (LPIPS), not argmax-rate-distortion.
- **Diffusion-based image compression**: inflate-time computation budget too large.
- **HAS-VQ Hessian-Adaptive Sparse VQ**: 1.7B-scale LLM machinery; Lane Ω-V2 already handles per-element Hessian-style allocation.
- **MoE-INR, Bias for Action INR (CVPR 2025)**: extra archive bytes wipe out gains OR strict-scorer-rule conflict.
- **Stable Diffusion teacher distillation**: compute prohibitive at compress-time.
- **Brotli-G shared dictionaries**: HTTP-only tooling; Lane J-FAC is the proper extraction.
- **DCAE 95M-param decoder**: paradigm extracted as Lane J-DCAE.

---

## Strategic Priority Update

**Cycle 1 (THIS WEEK):**
1. Lane J-JBL ($1.50) — highest-confidence SegNet
2. Lane J-NWC ($5.00) — highest-EV rate
3. Lane J-IMP ($25.00) — moonshot rate
4. Continue in-flight (Ω-V2, EC, SAUG-V2, Lane I, EBR, MOS) — EC-first stack

**Cycle 2 (post-landing):**
5. If J-NWC succeeds → stack with Lane J-DCAE
6. If J-JBL succeeds → stack with Lane J-EFD
7. If both succeed → stacked combo: J-JBL + J-NWC + Lane MOS + Lane EC

**Cycle 3 (moonshot):**
8. Lane J-ASYM (asymmetric mask codec) — if Lane I underperforms
9. Lane J-ADV (gradient-detector embedding) — Fridrich-canonical

**Predicted stack outcomes:**
- Conservative (J-JBL + J-NWC): [0.85, 0.95]
- Aggressive (J-JBL + J-NWC + MOS + EC): [0.65, 0.85]
- Moonshot (J-IMP + J-NWC + J-DCAE + J-EFD + Ω-V2): [0.40, 0.65] — **could PASS Quantizr**
- Wildcard: Lane J-ADV (Yousfi-canonical) sub-0.50 if executed correctly

**Competitive read**: Quantizr at 0.33 has not adopted RFSQ (Aug 2025) nor JML+BLS for argmax distillation. Open window to leapfrog via 2025 SOTA quantization + 2024 SOTA segmentation distillation.

**Council vote (Jack-summarized):**
- Yousfi (steganalysis priority): champions J-ADV first
- Hotz (engineering speed): champions J-FAC first as 2-hour CPU experiment
- Quantizr (competitor analysis): champions J-NWC + J-IMP because Quantizr stopped at vanilla 5-stage QAT
- Fridrich (rigor): demands J-JBL first because BLS+JML is canonically-correct surrogate for argmax-disagreement scoring
- Jack (research-rigor): J-JBL + J-NWC + J-IMP parallel — three different wedges, $31.50, 24h

**Sources:** see memory entry's source list at `feedback_compute_budget_hundreds_of_dollars_20260428` cross-references and external paper URLs above.
