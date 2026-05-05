---
name: PR106 has no separate mask channel — Lanes #05 and #06 fundamentally non-applicable
description: Plans #05 (UNIWARD-delta on PR106 mask channel) and #06 (mask-grayscale-LUT on PR106 mask channel) presume a mask.mkv-style separate stream that does not exist in PR106's HNeRV architecture. PR106 archive = single 0.bin = brotli-decoder + brotli-latents only; the latent IS the implicit-neural-representation of the whole video, not a SegNet mask. Both lanes need fundamental reformulation (e.g., "UNIWARD-delta on PR106 latent stream") before they can be considered.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
Empirical finding 2026-05-04. Verified PR106 archive.zip layout vs the audit's mask-channel assumption.

## Result

```
$ unzip -l experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/archive.zip
   186131  05-04-2026 08:00   0.bin

PR106 0.bin internal structure (per extract_pr106_decoder.py output):
  decoder_brotli: 170,278 bytes  (HNeRV decoder state_dict, INT8 quant + zigzag + brotli)
  latent_brotli:   15,849 bytes  (600 frame-pair latents, 28-dim, uint8 delta-coded + brotli)
  total:          186,131 bytes
  n_tensors with 'mask' in name: 0
```

**No separate mask channel exists.** PR106 is HNeRV-style (Hybrid Neural Representation for Video): the per-pair 28-dim latent + the HNeRV decoder JOINTLY define the full RGB output. There is no SegNet-mask, no mask.mkv, no separate-stream-encoding-mask-only design like Quantizr's submission.

## Why Lanes #05 and #06 are non-applicable

- **Lane #05** (`revival_plan_05_uniward_delta_pr106_mask_channel`): UNIWARD (Universal Wavelet Relative Distortion) is a steganographic embedding cost function — Fridrich-style adaptive payload distribution that hides information in textured regions. It operates on a SIGNAL-vs-COVER decomposition (e.g., mask vs frame). PR106 has no mask cover signal to attack.
- **Lane #06** (`revival_plan_06_mask_grayscale_lut_pr106_mask_replace`): The grayscale-LUT trick (Selfcomp's σ=15 Gaussian-softmax-LUT over CLASS_TARGETS) replaces a SegNet's encoder output with a fixed lookup table. Requires (a) a SegNet-mask discrete-class output to replace, and (b) the Gaussian-softmax structure baked into the architecture. Neither exists in PR106's HNeRV.

## How to apply

1. Lanes #05 and #06 are NOT viable as currently designed for PR106.
2. To revive (significant reformulation):
   - **Lane #05 alternative**: Apply UNIWARD-style adaptive bit-allocation cost function to the LATENT residual stream after a coarser decoder approximation. Untested; requires scoring framework for "where to spend bits in the latent".
   - **Lane #06 alternative**: Apply LUT-substitution to the HNeRV decoder's stem.weight (which acts as a fixed-channel-projection). Could shrink that 1728×28 Linear layer at cost of fidelity. Untested; needs careful calibration.
3. Both reformulations are speculative and not council-approved — defer indefinitely until a stronger anchor exists.

## Pattern across audit's revival plans (running tally as of 2026-05-04)

FIVE audit lanes now empirically falsified at zero GPU cost (~$2.50+ saved):
- **Lane #02 arith_qint → PR106 latents**: 0-th-order arithmetic > brotli already (4.66 vs 3.76 bits/byte)
- **Lane #03 QZS3 → PR106 decoder**: schema-locked to JointFrameGenerator
- **Lane #04 block_fp → PR106 decoder**: ternary-only ({-1, 0, +1}) destroys continuous distributions
- **Lane #05 UNIWARD-delta → PR106 mask channel**: PR106 has no mask channel (THIS finding)
- **Lane #06 mask-grayscale-LUT → PR106 mask channel**: PR106 has no mask channel (THIS finding)

**Lane Ω-W-V3 (water_filling_codec_v2)** remains the only audit-recommended lane proven viable on PR106 weights (-22,152 bytes / -0.01475 rate Δ in stub-mode preview, anchor commit b2f958a4).

The audit's "8 PR106-stacking revival plans" had a 37.5% (3 of 8 still alive: #01 alive, #07/#08 partially) implementation-viable rate when stress-tested against PR106's actual architecture. The ae430d78 subagent applied generic "{X codec} → {PR106 component}" templates without verifying that PR106 has the relevant component.

Cross-references:
- Anchor: experiments/extract_pr106_decoder.py (commit 45149f21)
- Other falsified lanes: feedback_block_fp_codec_pr106_unviable_at_default_ternary_20260504.md, feedback_qzs3_codec_pr106_unviable_jointframegenerator_locked_20260504.md, feedback_arithmetic_qint_codec_pr106_latents_unviable_brotli_already_below_entropy_20260504.md
- Surviving lane: experiments/repack_pr106_with_water_filling.py (commit b2f958a4)
- Audit refs: experiments/results/internal_hidden_gem_audit_20260504_claude/revival_plans/revival_plan_05_uniward_delta_pr106_mask_channel.md, experiments/results/internal_hidden_gem_audit_20260504_claude/revival_plans/revival_plan_06_mask_grayscale_lut_pr106_mask_replace.md
