---
name: QZS3 codec hardcoded for JointFrameGenerator architecture (refuses PR106 HNeRV decoder)
description: src/tac/quantizr_qzs3_codec.encode_qzs3_state_dict() is schema-locked to JointFrameGenerator-compatible state dicts (Quantizr's 88K-param renderer). On PR106 HNeRV decoder it raises ValueError("state dict is not JointFrameGenerator-compatible: missing=['shared_trunk.embedding.weight', ...]"). Lane #03 (QZS3 → PR106 decoder replacement) is NOT viable without either (a) refactoring QZS3 to be schema-agnostic, or (b) building a PR106-specific QZS3 sibling (qzs3_for_hnerv).
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
Empirical finding 2026-05-04. Smoke-tested `tac.quantizr_qzs3_codec.encode_qzs3_state_dict` on PR106 HNeRV decoder state_dict (extracted via `experiments/extract_pr106_decoder.py`, anchor commit 45149f21).

## Result

```
PR106 state_dict: 28 tensors, 228958 params
block_size=16: FAIL: ValueError: state dict is not JointFrameGenerator-compatible: missing=['shared_trunk.embedding.weight', 'shared_...
block_size=32: FAIL (same)
block_size=64: FAIL (same)
```

The codec performs an architectural schema check (JointFrameGenerator-compatible) before any byte work. PR106 HNeRV decoder uses a different module hierarchy (stem.weight, blocks.0..5.weight, skips.2..4.weight, refine.0..1.weight, rgb_0/1.weight) — completely different namespace.

## Why

QZS3 was designed alongside Quantizr's specific 88K-param renderer (JointFrameGenerator). The codec embeds knowledge about which tensors are FP4-quantized (3x3 conv weights), which are bias / embedding / Linear (qint at lower bit width), etc., via the FIXED schema. PR106's HNeRV decoder has no `shared_trunk.embedding.weight`, no FILM head, etc.

## How to apply

1. Lane #03 (`revival_plan_03_quantizr_qzs3_pr106_decoder_replacement`) is NOT viable as currently designed.
2. To revive: implement `qzs3_for_hnerv_state_dict()` in `src/tac/quantizr_qzs3_codec.py` that detects PR106 HNeRV schema and applies the right per-tensor codec choice (e.g., 4-bit FP4 for 3x3 Conv2d weights, 8-bit qint for stem Linear, 8-bit qint for biases).
3. Until that variant lands, prefer Lane Ω-W-V3 (water_filling_codec_v2) which is schema-agnostic — it operates per-Conv2d-tensor with per-channel block-FP-with-Hessian-aware-bit-allocation.

## Pattern across audit's revival plans

This is the SECOND audit lane today empirically falsified at default codec API:
- **Lane #03 QZS3 → PR106**: schema-locked to JointFrameGenerator (THIS finding)
- **Lane #04 block_fp → PR106**: ternary-only ({-1, 0, +1}), destroys PR106 continuous-distribution weights (memory: feedback_block_fp_codec_pr106_unviable_at_default_ternary_20260504.md)

The audit's "high-EV revival plans" assumed codec interchangeability that doesn't hold. The ae430d78 subagent's recommendations were architecture-agnostic at the planning level but architecture-specific at the implementation level. Cross-application requires either refactor or sibling impl.

**Lane Ω-W-V3 stands out** as the only audit-recommended lane that ACTUALLY works on PR106 weights (-22,152 bytes / -0.01475 rate Δ in stub-mode preview, anchor commit b2f958a4) because water_filling_codec_v2 was designed schema-agnostic from the start.

Cross-references:
- Anchor: experiments/extract_pr106_decoder.py (commit 45149f21)
- Companion finding (block_fp): feedback_block_fp_codec_pr106_unviable_at_default_ternary_20260504.md
- Surviving lane: experiments/repack_pr106_with_water_filling.py (commit b2f958a4)
- Audit ref: experiments/results/internal_hidden_gem_audit_20260504_claude/revival_plans/revival_plan_03_quantizr_qzs3_pr106_decoder_replacement.md
