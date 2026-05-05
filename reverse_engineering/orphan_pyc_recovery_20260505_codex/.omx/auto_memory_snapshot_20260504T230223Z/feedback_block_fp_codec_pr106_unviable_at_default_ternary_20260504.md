---
name: block_fp_codec ternary too lossy for PR106 HNeRV decoder
description: src/tac/block_fp_codec.py is ternary-only ({-1, 0, +1}); purpose-built for Selfcomp bimodal weight distributions. On PR106's continuous-distribution HNeRV decoder weights, max_err = 0.5-1.0 at all clip_threshold/block_size combinations. Lane #04 (block_fp_codec → PR106 decoder) is NOT viable as currently designed; needs a higher-precision codec variant (4-bit block-FP with int4 qint).
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
Empirical finding — 2026-05-04. Smoke-tested `tac.block_fp_codec.pack_block_fp` on PR106 HNeRV decoder state_dict (extracted via `experiments/extract_pr106_decoder.py`, anchor commit 45149f21).

## Per-tensor result on PR106 `blocks.0.weight` (144×36×3×3, abs_max=0.6008, abs_mean=0.0865)

| clip_threshold | block_size | pack bytes | brotli bytes | max_err |
|---:|---:|---:|---:|---:|
| 0.5  (default) | 16  | 46,732 | 917    | **0.4939** (~weight max!) |
| 0.5  (default) | 32  | 46,716 | 495    | 0.4939 |
| 0.5  (default) | 64  | 46,708 | 96     | 0.4939 |
| 0.1            | 16  | 46,732 | 9,989  | **0.8959** |
| 0.1            | 32  | 46,716 | 9,371  | 0.8959 |
| 0.05           | 16  | 46,732 | 11,062 | **0.9480** |
| 0.01           | 16  | 46,732 | 8,999  | **0.9858** |

Across all 13 Conv2d weights in PR106 decoder, max_err per tensor was 0.24 - 0.50 at default settings — comparable to weight magnitudes themselves. Brotli compresses to ~2-3KB total because the qint stream is mostly zeros (most weights round to 0 under ternary clip).

**Why:** `block_fp_codec` rounds qint to `{-1, 0, +1}` only (line 186: `ternary = torch.sign(scaled) * (scaled.abs() >= clip_threshold).to(scaled.dtype)`). There is NO n_bits parameter. The codec is hardcoded ternary by design — see header comment (line 26-28): "Each weight is then mapped to round(w / 2**e_b / clip_threshold), clamped to {-1, 0, +1}".

**Why this works for Selfcomp's renderer:** Selfcomp's 88K-param renderer (cited 0.36 score, 1.017 bpw) was trained with a regularizer that pushes weights to a bimodal distribution around 0 / ±some_magnitude. Ternary quantization preserves both modes near-losslessly. PR106's HNeRV decoder is trained without such regularization → continuous Gaussian-like distribution → ternary clobbers everything.

**Why:** Selfcomp's training regimen is part of the "Selfcomp 0.36" architecture; the codec is downstream of training, not interchangeable with arbitrary pre-trained models.

**How to apply:**
1. Lane #04 (`revival_plan_04_block_fp_codec_pr106_decoder`) is NOT viable as currently designed without higher-precision codec.
2. To revive: implement a `pack_block_fp_int4` variant (4-bit signed qint per weight, shared exponent per block) in src/tac/block_fp_codec.py. Estimated 5-6 bpw effective after brotli.
3. Alternative: train a PR106-architecture-compatible bimodal-regularized HNeRV first (significant training lift).
4. Until either fix lands, prefer Lane Ω-W-V3 (water_filling_codec_v2) which handles continuous distributions natively via per-channel block-FP-with-Hessian-aware bit-allocation.

Cross-references:
- Anchor commit: 45149f21 (extract_pr106_decoder.py)
- Companion finding: experiments/repack_pr106_with_water_filling.py demonstrates Ω-W-V3 saves 22,152 bytes on the same PR106 weights with no fidelity issue (max_err well-controlled by the Lagrangian water-fill).
- Audit: experiments/results/internal_hidden_gem_audit_20260504_claude/revival_plans/revival_plan_04_block_fp_codec_pr106_decoder.md (council 7/10 GO; this finding effectively defers it pending int4 variant).
