---
name: arithmetic_qint codec on PR106 latents already-saturated by brotli
description: Plain 0-th-order arithmetic coding of PR106's latent stream cannot beat brotli — Shannon 0-th-order entropy = 4.66 bits/byte (theoretical floor 19,642 bytes); PR106 brotli achieves 3.76 bits/byte (15,849 bytes), BELOW the 0-th-order Shannon floor via LZ77+context. Lane #02 (arith_qint → PR106 latents) needs a custom context model (not plain encode_qints_arithmetic) to have any chance.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
Empirical finding 2026-05-04. Smoke-tested `tac.arithmetic_qint_codec.encode_qints_arithmetic` on PR106 latent raw stream (decoded from brotli inside PR106 0.bin).

## Result

```
PR106 latent_brotli (in archive): 15,849 bytes
raw decompressed (concat of zigzag delta uint8 streams + scales/mins float16): 33,712 bytes
Shannon entropy of raw uint8 stream: 4.6611 bits/byte
0-th-order theoretical floor: 19,642 bytes
PR106 brotli effective ratio: 3.7610 bits/byte (15,849 bytes actual)
```

PR106 brotli BEATS the 0-th-order Shannon floor by 3,793 bytes (24% below) because it exploits:
- LZ77 backreferences (long runs / templates in delta streams)
- Higher-order context models (per-byte distribution conditioned on previous N bytes)
- Static + dynamic context-tree adaptive coding

## Why this falsifies Lane #02 at default API

`tac.arithmetic_qint_codec.encode_qints_arithmetic` is a plain 0-th-order arithmetic coder (single global frequency table). Without a context model, the best it can achieve = Shannon 0-th-order entropy = 19,642 bytes — 3,793 bytes WORSE than PR106's brotli. Lane gain would be NEGATIVE.

The audit's plan #02 acknowledged this risk ("encoded ≤ 15849 - 200 bytes (arbitrary win threshold)"). Entropy analysis confirms: with default API, lane is mathematically impossible to ship.

## How to apply

1. Lane #02 (`revival_plan_02_arithmetic_qint_codec_pr106_latents`) is NOT viable as currently designed.
2. To revive: implement a CONTEXT-AWARE arithmetic coder for the PR106 latent stream:
   - Per-position conditional frequency (latent_dim=28; conditional on previous frame-pair index)
   - OR LZ77-augmented arithmetic (pre-replace runs with backreferences)
   - OR train a small LM (e.g., 3-gram or RNN bytes-as-tokens) to drive a per-byte distribution
3. Even with context model, expected gain is marginal (-50 to -300 bytes per audit's own forecast = -0.0001 score Δ).
4. Until such a variant lands, prefer Lane Ω-W-V3 (water_filling_codec_v2) for actual byte savings (-22,152 stub-mode preview) or skip lane #02 entirely.

## Pattern across audit's revival plans (running tally as of 2026-05-04)

THREE audit lanes now empirically falsified at zero GPU cost (~$1.50+ saved):
- **Lane #02 arith_qint → PR106 latents**: 0-th-order arithmetic > brotli already (THIS finding).
- **Lane #03 QZS3 → PR106 decoder**: schema-locked to JointFrameGenerator (memory: feedback_qzs3_codec_pr106_unviable_jointframegenerator_locked_20260504.md).
- **Lane #04 block_fp → PR106 decoder**: ternary-only ({-1, 0, +1}) destroys continuous distributions (memory: feedback_block_fp_codec_pr106_unviable_at_default_ternary_20260504.md).

**Lane Ω-W-V3 (water_filling_codec_v2)** remains the only audit-recommended lane proven viable on PR106 weights (-22,152 bytes / -0.01475 rate Δ in stub-mode preview, anchor commit b2f958a4).

The audit's "8 PR106-stacking revival plans" had a 50% (4 of 8) implementation-viable rate when stress-tested. The ae430d78 subagent's planning was directionally good but did not validate codec applicability against PR106's specific architecture/data distribution before committing the plans to disk.

Cross-references:
- Anchor: experiments/extract_pr106_decoder.py (commit 45149f21)
- Companion findings: feedback_block_fp_codec_pr106_unviable_at_default_ternary_20260504.md, feedback_qzs3_codec_pr106_unviable_jointframegenerator_locked_20260504.md
- Surviving lane: experiments/repack_pr106_with_water_filling.py (commit b2f958a4)
- Audit ref: experiments/results/internal_hidden_gem_audit_20260504_claude/revival_plans/revival_plan_02_arithmetic_qint_codec_pr106_latents.md
