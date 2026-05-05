---
name: Quantizr Architecture Deep Intel
description: mask2mask 0.60 uses AsymmetricPairGenerator, FP4 custom codebook, marshal-obfuscated arch, mask video encoding
type: project
originSessionId: ecc348c1-2829-48d3-9d4c-10ed2aa188a8
---
Quantizr's mask2mask (PR #53, score 0.60) architecture confirmed from archive analysis:

**Architecture:**
- Class: `AsymmetricPairGenerator` (same name as ours!)
- Input: mask pairs (5 classes) → Output: RGB frame pairs
- Marshal-serialized + Brotli-compressed (arch.br) — intentionally obfuscated
- Claims "slightly different architecture gets 10% better" (→ 0.54) and "<0.50 is easily possible"

**FP4 Quantization (study this):**
- Custom 8-level codebook: {0, 0.5, 1, 1.5, 2, 3, 4, 6} + sign bit
- Nibble-packed weights with per-block scales
- This is NOT standard FP4 — it's a learned/hand-tuned magnitude quantization

**Archive structure (386KB total):**
- model.pt.br (Brotli-compressed FP4 weights)
- mask.mp4.br (masks encoded as grayscale video, value/63 mapping, Brotli-compressed)
- arch.br (architecture definition, marshal-serialized)

**Key metrics:** PoseNet 0.00066 (47x better than us), SegNet 0.0026 (comparable), rate 0.010

**How to apply:** Our FP4 export should study his codebook. The mask-as-video encoding is clever for rate. The joint pair generation confirms coupled optimization is the right direction.

**Yousfi's comma10k-baseline reveals scorer SegNet:**
- EfficientNet-B0 + U-Net (segmentation_models_pytorch)
- 6 classes, values: {0, 41, 76, 90, 124, 161}
- Resolution 448x576, CrossEntropy loss
