---
name: Self-Compression Breakthrough
description: Yousfi's scorer-Jacobian finds layers where 2-bit IMPROVES score. FiLM is 3rd most scorer-sensitive (pixel analysis was WRONG). Deep conv quantization noise = beneficial steganalytic noise.
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## Self-Compression Results (2026-04-25)

Three approaches tested on renderer.bin (288K params):

### Yousfi's Scorer-Jacobian (BREAKTHROUGH)
Layers where 2-bit quantization IMPROVES contest score:
- `bottleneck.conv2` (32K params): scorer_sens = -0.063
- `down_res.conv2` (32K params): scorer_sens = -0.057
- `bottleneck.norm2.class_beta`: scorer_sens = -0.035

Layers that MUST have high precision:
- `up_res.conv1` (11.6K params): scorer_sens = +0.249 (most sensitive)
- `head` (108 params): scorer_sens = +0.231
- `film_bottleneck.shift` (360 params): scorer_sens = +0.220

**Why:** Quantization noise in deep bottleneck layers acts as Fridrich-style steganalytic noise that the scorers can't distinguish from natural texture. The scorers actually score it BETTER because it looks more "natural." This is the inverse steganalysis principle in action.

**Why:** Pixel sensitivity (Hotz) showed FiLM at zero sensitivity. Scorer sensitivity (Yousfi) shows FiLM shift at 0.220 — the 3rd most sensitive layer. The FiLM pose signal barely changes pixels but massively changes PoseNet output. Only scorer-aware analysis catches this.

### Action Items
1. Implement rate-constrained knapsack allocation (target 80KB archive)
2. Layers with negative sensitivity → force to 2 bits (free quality improvement)
3. FiLM layers → 8 bits (critical for PoseNet, invisible to pixel metrics)
4. Integrate as `--mixed-precision-qat` flag in qat_finetune.py
