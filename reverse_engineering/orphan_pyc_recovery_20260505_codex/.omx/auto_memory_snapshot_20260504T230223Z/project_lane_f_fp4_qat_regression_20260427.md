---
name: Lane F FP4 QAT regression — PoseNet sensitive to FP4 on dilated-h64 ASYM
description: 2026-04-27: Lane F (FP4 QAT on dilated-h64 baseline) ran end-to-end after qat_finetune Bug 2 fix. Score 2.73 [contest-CUDA] vs baseline 2.29 = +0.44 REGRESSION. PoseNet exploded (0.247 → 0.391, +58%). Rate -0.108 saved as predicted but PoseNet wipeout > rate gain. FP4 NOT imperceptible on dilated-h64 ASYM PoseNet path despite SegNet at floor.
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
**Lane F result 2026-04-27 (commit f8ccdd08 unblock; subagent acf426ca):**

| Metric | Lane F (FP4 QAT) | Baseline 2.29 | Delta |
|---|---|---|---|
| **Final score** | **2.73** [contest-CUDA] | **2.29** [contest-CUDA] | **+0.44 (WORSE)** |
| PoseNet dist | 0.391 | 0.247 | +58% (worse) |
| SegNet dist | 0.00365 | ~0.003 | floor (unchanged) |
| Rate unscaled | 0.01561 | ~0.0184 | -15% (better, -0.108 score) |
| Archive bytes | 586,232 | ~700KB | -16% (smaller) |

**Bug 2 fix VERIFIED:** `load_asymmetric_checkpoint_fp4` succeeded after the pair_mode-aware loader landed. No shape-mismatch crash. The chained arch-drift bugs are GENUINELY closed.

**Cost:** $0.20 total (one Denmark instance destroyed for NVDEC missing $0.14 + France instance for the actual run $0.06). 50 epochs at fp4_lr=2.5e-6 finished in 47 seconds.

**The surprising finding — FP4 sensitivity:**

Original prediction (per memory `project_5stage_quantization_advantage`): "Distortion expected unchanged (FP4 noise is imperceptible in inference)." This was WRONG for dilated-h64 ASYM. PoseNet output IS sensitive to FP4-quantized weights even when SegNet is unaffected.

Same pattern as `feedback_proxy_auth_math_useless`: proxy/internal eval reported `distortion=7.52` post-FP4, suggesting acceptable. Auth eval reported PoseNet=0.391, ~150× the proxy's pose component. The proxy doesn't capture what the contest scorer measures.

**Why might FP4 hurt PoseNet specifically?**
- PoseNet reads YUV6 from FastViT-T12 attention. The ASYM renderer's decoder produces RGB → YUV6 conversion is downstream. FP4 noise on the renderer's last conv (before the rgb_to_yuv6 step) propagates as YUV-correlated noise that FastViT's attention picks up.
- SegNet (EfficientNet-B2 with stride-2 stem) is more robust because stride-2 averages noise out at the first layer.
- Council guess (post-hoc): the renderer's last 1x1 conv has ~3 weights per output pixel — FP4's 4-bit precision means each output pixel has ~3×log2(16) = 12 bits of precision total, vs FP32's 96 bits. That 8x reduction shows up in PoseNet's high-frequency sensitivity.

**Implications for FP4-anywhere strategy:**
- FP4 QAT on AsymmetricPairGenerator with dilated-h64 arch is a NET LOSS at this rate point.
- The brotli wrapper (Lane B-alt, -0.0037 score) was barely cost-effective; FP4 is worse.
- For rate gains beyond brotli: try a SMALLER arch (more parameters in fewer bits) instead of FP4-on-current-arch.

**What NOT to do:**
- Do not promote Lane F's archive (586KB at 2.73) to submission. Worse than baseline.
- Do not retry Lane F with more epochs — the post-QAT distortion was 7.41 (actually better than the 8.18 pre-QAT), so it converged. The bug isn't training; it's the final FP4 representation.
- Do not assume FP4 is benign on other archs. Re-test SegNet sensitivity if applying FP4 to a smaller renderer.

**Future lanes affected:**
- Lane K (DSConv variants) — would need FP4 quant; likely similar sensitivity, run small SegNet+PoseNet sensitivity test BEFORE committing to a full retrain.
- Lane L (PSD architecture) — same caveat.
- Lane I (Cool-Chic / C3) — already known per `feedback_qat_finetune_chained_arch_bugs` that FP4 destroys C3's float gain. Confirmed.

**Cost of this finding:** $0.20 + 47s GPU time. CHEAP for the empirical evidence. The bug fix at f8ccdd08 was the load-bearing investment; the actual run was just to measure the regression.
