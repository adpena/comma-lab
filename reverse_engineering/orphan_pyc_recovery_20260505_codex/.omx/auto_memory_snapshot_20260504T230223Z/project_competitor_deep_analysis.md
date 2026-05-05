---
name: Competitor Deep Analysis — mask2mask and tensor_inversion fully deobfuscated
description: Full architecture reverse-engineering of all sub-1.0 submissions. Rules clarified. Gradient bug explains all underperformance.
type: project
originSessionId: 47bf3dd8-df75-4271-9ce1-428c19c2eb32
---
## mask2mask (Quantizr, 0.60) — FULLY DEOBFUSCATED
- Archive (386KB): FP4-quantized AsymmetricPairGenerator + mask.mp4 (5 classes as grayscale video) + marshal-obfuscated bytecode
- Inflate: decode masks → feed through generator → output alien-looking frames (flat colors, NOT photorealistic)
- Generator trained WITH scorers as loss, but scorers NOT needed at inflate time → RULE COMPLIANT
- PoseNet 0.00066 (near zero): generates PAIRS simultaneously, controls inter-frame motion directly
- SegNet 0.00264: trained to produce correct class predictions, not correct pixels
- FP4 codebook: 8 positive levels (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0) + sign bit
- Yousfi: "you can get even better than 0.50 with this strategy and some tricks ;)"

## tensor_inversion (amoghmunikote, 0.75) — RULE NON-COMPLIANT
- Loads scorer weights (73MB) at inflate time WITHOUT including in archive
- Yousfi explicitly flagged this — PR closed
- If weights included: rate alone = 51.19 → score > 51
- Technique: gradient descent through frozen scorers with pre-computed targets
- Also has the rgb_to_yuv6 @torch.no_grad bug (zero PoseNet gradients)

## CRITICAL RULE
"External libraries and tools can be used and won't count towards compressed size, unless they use large artifacts (neural networks, meshes, point clouds, etc.), in which case those artifacts should be included in the archive"
- You CAN use scorers at COMPRESS time
- You CANNOT use scorers at INFLATE time without including weights in archive
- You CAN train a CUSTOM model against scorers and include only custom weights

## Key Insight
"You don't need to compress the video. You need to compress the SCORER OUTPUTS."
The reconstruction doesn't need to look like original video — just produce same scorer outputs.

## Our Gradient Fix Changes Everything
- The @torch.no_grad on rgb_to_yuv6 affected EVERYONE doing TTO (us and tensor_inversion)
- With the fix (commit 10093136), TTO should actually work for PoseNet optimization
- Council projects: PoseNet 0.017 → 0.003, score 0.87 → 0.35
