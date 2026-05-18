# DEFER — fec6 + block-FP decoder (Ext 2)

**Date:** 2026-05-17
**Lane:** `lane_fec6_stacking_wave_5_grammar_extensions_20260517`
**Verdict:** DEFERRED-pending-float32-source-substrate-rebuild (NOT KILLED per CLAUDE.md "Forbidden premature KILL")
**Substrate compatibility:** verified incompatible (PR101's decoder is already pre-encoded with custom byte maps + Brotli; not float32 source weights)

## Why this is deferred

Per the canonical premise verifier at `.omx/tmp/fec6_stacking_wave_premise_verifier.txt` PV-2:

- The block-FP codec at `src/tac/block_fp_codec.py` expects float32 source weights at encode time (per its `encode` API surface and tests at `src/tac/tests/test_block_fp_codec.py`).
- PR101's `decoder_blob` is NOT float32 source weights. Per `experiments/results/public_pr101_hnerv_ft_microcodec_intake_20260504_codex/source/submissions/hnerv_ft_microcodec/src/codec.py:260-296`, PR101's decoder is a custom-byte-mapped int8 stream encoded with Brotli per `decompress_brotli_streams` + per-tensor `DECODER_BYTE_MAPS` (zig-zag, etc.).
- **The PR101 decoder has already been quantized + entropy-coded by its training pipeline.** Block-FP would have to (a) decompress the Brotli streams, (b) decode the custom byte maps back to int8 tensors, (c) DEQUANTIZE the int8 tensors back to float32 (which destroys information), (d) re-encode with block-FP ternary {-1,0,+1}, (e) re-Brotli the result, (f) write a NEW inflate.py that reverses block-FP.

This is NOT a fec6 grammar extension. This is a SUBSTRATE REPLACEMENT — a complete rebuild of the PR101 decoder using block-FP from-scratch.

## Substrate-compatibility evidence

Per CLAUDE.md Catalog #301:

- **Compatible substrates for block-FP**: Selfcomp, szabolcs (which already use block-FP per `src/tac/block_fp_codec.py` docstring), any custom-trained substrate that exports float32 weights and chooses block-FP as its quantization-encoding pipeline.
- **Incompatible substrates for block-FP**: substrates whose decoder is already encoded (PR101, fec6, PR106 LANES wrapping PR101). The decoder cannot be re-quantized without first dequantizing, which would either lose information or require retraining.

## Reactivation criteria

DEFER reverses to LAND when ANY of:

1. **A successor substrate (PR95 Phase 2-4 family per task #608) trains a HNeRV-like renderer from scratch and uses block-FP as its decoder quantization at export time.** This is canonical substrate-engineering scope: a new training pipeline (not a fec6 bolt-on).

2. **An operator-approved substrate-engineering scope rebuild of PR101 with block-FP-from-float32 decoder quantization.** Large scope: re-train the HNeRV decoder, re-export with block-FP, re-design the archive grammar, re-implement inflate.py. Estimated 100-300 LOC of substrate-engineering work.

3. **Empirical evidence that block-FP re-encoding of PR101's already-quantized int8 decoder yields any rate improvement.** This is unlikely because PR101's int8 + zig-zag + Brotli is already near the Shannon bound for the quantized signal; re-quantizing without information would only add lossy noise.

## Per CLAUDE.md "Forbidden premature KILL without research exhaustion"

Default verdict is **DEFERRED-pending-research**, not KILLED. The block-FP codec is well-validated for substrates that export float32 source weights (per `src/tac/tests/test_block_fp_codec.py` + the Selfcomp/szabolcs lane evidence). The deferral is about ARCHIVE-COMPATIBILITY with the PR101 family, not about block-FP itself.

## Alternative probe methodologies per Catalog #308

Per the alternative-probe enumeration discipline (N≥3 required):

1. **Alternative 1**: SPLIT-VERDICT — apply block-FP to a different substrate that exports float32 source weights at training-time (the canonical Selfcomp / szabolcs lane application).

2. **Alternative 2**: REQUEST-REINVESTIGATION — investigate whether PR101's int8 + zig-zag + Brotli already saturates the rate-bound for the HNeRV decoder. If yes, block-FP cannot improve; the deferral is permanent for the PR101 family.

3. **Alternative 3**: SUBSTRATE-PIVOT — wait for PR95 Phase 2-4 substrate to land with native block-FP export at training time. This is the proper substrate-engineering scope home for the technique.

## Per CLAUDE.md "META-ASSUMPTION ADVERSARIAL REVIEW"

**Operating-within assumption**: this DEFER memo assumes that the operator's framing of "block-FP applied to fec6's decoder.bin int8 mantissa stream" was based on a stale inventory entry (`tac.codec.block_fp_codec` per `meat_on_the_bone_inventory_and_canonical_helpers_design_20260517.md` §3.A) that hadn't been verified against the actual fec6 + PR101 grammar (which has no separate `decoder.bin` int8 slot).

**Assumption-Adversary verdict**: HARD-EARNED — the inventory entry IS the source of the stale assumption; the premise-verification step caught it; the deferral is the correct response per the operator's UNIQUE-AND-COMPLETE-PER-METHOD operating mode.

## Predicted paradigm-vs-implementation classification per Catalog #307

**IMPLEMENTATION-CARGO-CULT** — the block-FP paradigm (per-block shared fp16 exponent + ternary mantissa) is intact and well-validated on substrates that export float32 source weights. The specific implementation-mapping "apply block-FP to PR101's decoder.bin" cargo-culted the assumption that PR101 has a `decoder.bin` int8-mantissa-with-fp16-per-tensor-scale slot from the inventory framing. The paradigm reactivates the moment a compatible substrate ships native block-FP export at training time.

## Cross-references

- `src/tac/block_fp_codec.py` — block-FP canonical implementation
- `src/tac/hessian_block_fp.py` — Hessian-weighted block-FP variant
- `src/tac/block_fp_jfg.py` — JFG-specific block-FP variant
- `src/tac/codec/a6_selfcomp_blockfp_hyperprior_compose.py` — Selfcomp composition (the canonical home)
- `experiments/results/public_pr101_hnerv_ft_microcodec_intake_20260504_codex/source/submissions/hnerv_ft_microcodec/src/codec.py:260-296` — PR101's actual decoder encoding pipeline
- `tools/build_pr101_frame_exploit_selector_packet.py` — fec6 builder (wraps PR101 opaquely)
- `experiments/train_substrate_pr95_phase_2_4.py` — task #608 PR95 Phase 2-4 substrate (future home for block-FP application)
- CLAUDE.md "Forbidden premature KILL without research exhaustion"
- CLAUDE.md "KILL/FALSIFIED memory verdicts — NON-NEGOTIABLE"
- Catalog #229 premise verification
- Catalog #301 kill-memo substrate-compatibility evidence
- Catalog #307 paradigm-vs-implementation classification
- Catalog #308 alternative-probe-methodologies enumeration
