---
name: Selfcomp 0.38 — full reverse engineering
description: 2026-04-29 complete RE of PR #56 selfcomp inflate.py (263 lines) + author description. Five paradigm shifts: grayscale-LUT mask encoding, single-mask-per-pair affine duality, analytical-pose via affine_delta, block-FP weight self-compression at 1.017bpw, 94K-param SegMap arch.
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
**Source**: github.com/commaai/comma_video_compression_challenge/pull/56 (merged 2026-04-27, Yousfi: "nice"). Author szabolcs-cs.

**Score** (GPU-eval): final 0.38, archive 279,036 bytes
- segnet 0.00122 × 100 = 0.122
- posenet √(10 × 0.000552) = 0.074
- rate 0.00743 × 25 = 0.186

Beats our Lane G v3 (1.04) on EVERY component: segnet 3.3×, posenet 2.5×, rate 2.5×.

## Architecture (SegMap, ~94K params)

Input (B, 8, 384, 512):
- 5 chan = F.embedding(grayscale_pixel, gaussian_softmax_LUT) where LUT[i,c] = softmax(exp(-(i - target[c])²/(2·σ²))), target = [0, 255, 64, 192, 128] (SegNet class colors), σ = 15.0
- 3 chan = F.grid_sample(F.interpolate(shared_latent_base[1,3,30,40] × 1.25), theta = affine(frame_affine_embedding[frame_idx]))

Body:
- layer_in: Conv2d(8, hidden, 1×1)
- num_blocks ResBlocks: Conv(h→bh, 3×3) + SiLU + Conv(bh→h, 3×3) + SiLU(skip)
- layer_out: Conv2d(hidden, 3, 1×1) + sigmoid×255 + bicubic→(1164,874)

Inflate pipeline: 1 grayscale AV1 frame → frame_indices=[2i, 2i+1] → SegMap forward(B=2) → frame1+frame2 RGB.

## Five paradigm shifts that beat us

1. **Grayscale-LUT mask** (1ch smooth values, AV1-friendly) vs our 3ch discrete-class
2. **Single mask + affine duality** (one mask warps to frame1+frame2) vs our 2 masks per pair
3. **Analytical pose via affine_delta** (6-DOF tanh-bounded, stored as embedding) vs PoseNet
4. **Block-FP weight self-compression** at 1.017 bpw (qint × 2^exp, HWOI permute) vs our FP4 4-8bpw
5. **94K-param SegMap** vs our 287K-param ASYM

## Self-compression weight format

Codecs:
- `linear_q_per_tensor_v1`: weight = min + qint × (max - min)/(2^bits - 1)
- `linear_q_per_affine_column_v1`: per-column variant for embeddings

Conv weight: `weight = qint × 2^exponents`. HWOI layout permute before encode for entropy.

## What Selfcomp explicitly admits is suboptimal

- "Underfit to segnet due to no architecture search"
- "More can be gained" on weight self-compression
- NO KL distillation (Quantizr's edge → 0.33 vs Selfcomp 0.38)

## How to apply

- Lane MM ($1/2h, predicted 0.78): grayscale-LUT mask encoding, drop-in replacement.
- Lane SS ($4/8h, predicted 0.55): single-mask-per-pair affine duality.
- Lane SA ($5/12h, predicted 0.45): full SegMap clone reimplementation.
- Lane SC++ ($5/12h, predicted 0.33): SA + KL distill T=2.0 — sub-Quantizr territory.
- Lane SO ($7/14h, predicted 0.30): SC + Hessian per-weight sub-1bpw quant.
- Lane SCK ($15/30h, predicted 0.28): full stack + arch search.

Council 5/5 unanimous: pursue MM first as cheapest validator; SA+SC++ in parallel as primary path to ≤0.5; SO+SCK as moonshot to beat Quantizr 0.33.
