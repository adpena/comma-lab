---
name: szabolcs selfcomp FULL reverse-engineering 2026-04-26
description: Complete deobfuscation of szabolcs-cs PR #56 (score 0.36). Rivals Quantizr without storing masks at all.
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
PR #56 by szabolcs-cs, 2026-04-23. Score **0.36** (just 0.03 above Quantizr's 0.33).

**Component breakdown:**
- Average PoseNet Distortion: 0.00040 ⇒ √(10×0.00040) = 0.063
- Average SegNet Distortion: 0.00115 ⇒ 100×0.00115 = 0.115
- Submission file size: 279,036 bytes ⇒ 25 × 279036 / 37545489 = 0.186
- **Total: 0.36** (matches reported)

**The killer trick: NO masks in archive, NO SegNet at inflate.**

Pipeline:
1. Decode AV1 video (luma only at 384×512, bicubic-resized).
2. For each pixel: embed grayscale value [0..255] through a Gaussian softmax LUT centered at 5 quantization levels `[0, 255, 64, 192, 128]` with σ=15.0 → soft 5-class probability map (5 channels).
3. Build per-frame affine-warped 3-channel latent canvas: `shared_latent_base` (3×30×40) bicubic-upsampled to 1.25× output, then per-frame `affine_grid + grid_sample` with 6 DOF (zoom, aspect, shear_x, shear_y, trans_x, trans_y). Each tanh-bounded by max_zoom_delta=0.12, max_aspect/shear=0.03, max_translation=0.08.
4. Concat 5+3 → CNN: `Conv1x1(8→hidden) → N×ResidualBlock(hidden, block_hidden) → Conv1x1(hidden→3)` with SiLU.
5. Output sigmoid×255 → RGB. Bicubic to 1164×874.

**Self-compression at 1.017 bits/weight:**
- Conv weights: `weight_qint * 2^weight_exponents` — per-block float exponent + low-bit integer mantissa. The "weight_tensor_layout: HWOI" hint reveals block-FP layout.
- Biases / latent / affine: `linear_q_per_tensor_v1` (uniform N-bit quantization with min/max).
- `linear_q_per_affine_column_v1` for the affine embedding (per-column min/max).
- Final compression: `tar.xz` over the whole payload (the "double compress" commit).

**Why they're not at 0.33:** Single residual stack from luma+latent has limited expressive power. SegNet distortion 0.00115 vs Quantizr's ~0.0007. Mask-aware FiLM (Quantizr) wins on per-class precision.

**How to apply for our DEN:**
- Their no-mask trick is brilliant but architecturally incompatible with our mask-aware DEN. Stay with masks for higher distortion ceiling.
- Their per-frame affine (6 DOF × 1200 frames) is functionally similar to our per-pair FiLM (6 dims × 600 pairs). We have it.
- **Block-FP weight encoding** is genuinely novel — replaces our FP4 (4 bits/weight) with 1-2 bits/weight. Future DEN-NEXT iteration: implement block-FP for the renderer weights to drop archive by 30-50KB.
- Their archive composition: ~200KB AV1 luma + ~70KB block-FP weights + ~10KB affine. We can match the model+pose budget; need half-frame masks to match the rate.

**Code on disk:** `/tmp/szabolcs_re/inflate.py` (263 lines, full source). Three weight codecs and the SegMap class are all there for direct adaptation.

**Their GitHub:** `szabolcs-cs/comma_video_compression_challenge` (fork). Master branch has the exact submission.
