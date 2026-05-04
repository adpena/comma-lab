---
name: szabolcs-cs PR#56 reverse-engineering — shared latent + affine + block FP compression
description: Score 0.36, 279KB archive. Shared learned image + per-frame 6-DOF affine transform. Block floating point at 1.017 bits/weight. Better PoseNet than Quantizr. tar.xz double compression.
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---

## Architecture: SegMap

Input: 5ch (Gaussian softmax from AV1 grayscale) + 3ch (affine-transformed shared latent) = 8ch
Body: Conv2d(8, hidden, 1x1) → N × ResBlock(hidden, block_hidden) → Conv2d(hidden, 3, 1x1)
Output: sigmoid * 255 → bicubic upscale to 874x1164

## Three Key Innovations

1. **Shared Latent + Per-Frame Affine**: ONE learned image (3x30x40=3600 params) + per-frame
   6-DOF affine (1200x6=7200 params). Affine = zoom, aspect, shear_x, shear_y, tx, ty.
   PoseNet-optimal because affine deltas encode ego-motion directly.
   PoseNet: 0.000397 (better than Quantizr's 0.000513).

2. **Block Floating Point Compression**: weight = qint * 2^exponents.
   Ternary/binary qint + shared exponents per block. Then tar.xz (LZMA2).
   Effective: 1.017 bits/weight — 4x more efficient than FP4.

3. **Gaussian Softmax LUT**: Analytical conversion from AV1 grayscale to 5ch
   soft class probabilities. softmax(exp(-(x-target)²/(2*15²))) over 5 classes.
   No stored lookup table.

## Archive: 279,036 bytes
- archive.zip contains payload.tar.xz (LZMA2 double compression)
- segmap_inference.pt — model weights in block FP format
- 0.mkv — 600 AV1 monochrome frames (half-frame paradigm)

## Unreleased: `learned_fullres_residual` and `lowfreq_frame_channel` flags exist but are not deployed.

**How to apply:**
- Per-frame affine transform is a powerful PoseNet technique — consider for our FiLM conditioning
- Block FP + LZMA2 could replace our FP4 + Brotli for smaller archives
- Gaussian softmax for mask input is more elegant than hard class labels
- His SegNet is 2.9x worse than Quantizr — we should NOT copy his SegNet approach
