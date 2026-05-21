# Magic Codec Pair #2 Sparse PacketIR FEC6 Null-Byte Smoke Landed

**Author**: Codex
**UTC**: 2026-05-21T00:21:20Z
**Lane**: `lane_wave_3_magic_codec_pair_2_sparse_packet_ir_fec6_null_byte_cpu_smoke_20260520`
**Axis**: `[macOS-CPU advisory]`
**Promotion status**: `score_claim=false`, `promotion_eligible=false`

## What Landed

Built and ran `tools/run_magic_codec_pair_2_sparse_packet_ir_fec6_null_byte_smoke.py`, with focused regression coverage in `src/tac/tests/test_magic_codec_pair_2_sparse_packet_ir_fec6_null_byte_smoke.py`.

This is the free local validation for magic-codec cascade pair #2:

`sparse_packet_ir SRL1 x procedural-codebook residuals on FEC6 master-gradient-null byte positions`.

The smoke uses the canonical null-byte matrix and the master-gradient `.npy` tensor, then extracts the inner ZIP member bytes from the FEC6 frontier archive. The important correction is that "null byte" means zero score-gradient leverage across the seg/pose/rate master-gradient axes. It does not mean literal byte value zero.

## Critical Correction

The interrupted surface had a dangerous interpretation bug: it treated "null bytes" as literal `0x00` archive byte values. That was wrong.

Empirical check:

- FEC6 outer `archive.zip`: `178,517` bytes, SHA-256 `6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf`
- FEC6 inner member `x`: `178,417` bytes, SHA-256 `f174192aeadfccf4b50fe7d45d1c9b98cec74eedfa33d06c35d480e6b46cd4dd`
- Literal zero bytes in inner member: `605`
- Master-gradient-null bytes in matrix: `16,292`

So pair #2 must be keyed by master-gradient null indices, not by byte value.

## Full-Scale Result

Source artifact:

`experiments/results/magic_codec_pair_2_sparse_packet_ir_fec6_null_byte_smoke_20260521T002120Z/smoke_result.json`

Key outputs:

| Metric | Value |
|---|---:|
| FEC6 null positions used | `16,292` |
| Config A, in-place charged bytes | `16,292` |
| Config A, brotli audit bytes | `16,296` |
| Config B, procedural-only seed bytes | `32` |
| Config C, procedural + SRL1 residual bytes | `97,473` |
| Bytes saved, C vs A | `-81,181` |
| Empirical Delta S | `+0.05405509567341099` |
| Predicted Delta S | `-0.00109` |
| Residual z-score vs prediction | `101.1837` |
| Byte-mutation seed sensitivity | `true` |

Verdict:

`CARGO-CULTED`

Cascade verdict:

`PAIR_2_FALSIFIED_CASCADE_FURTHER_NARROWS_PIVOT_TO_PAIR_4_OR_DP1_ONLY`

## Interpretation

The pair #2 hypothesis was that a procedural predictor would make the residuals on the master-gradient-null FEC6 byte positions sparse enough for SRL1 to win. It did not. The residual had only `0.331%` zeros (`54 / 16,292`) and SRL1 paid index/value overhead on nearly every selected byte. Brotli on the same residual was far cheaper than SRL1, but still not enough to beat the in-place byte budget.

This does not invalidate the null-exploit surface, the magic codec surface, or the earlier master-gradient/per-frame/per-pair/byte-pixel work. It invalidates this specific binding:

`pcg64 seed predictor + exact int16 residual + SRL1 on FEC6 master-gradient-null byte positions`.

The earlier sensitivity surfaces remain the routing authority. Null exploit and magic codec are consumers of those surfaces, not replacements for them.

## Canonical Equation Update

Appended one canonical anchor to `procedural_codebook_from_seed_compression_savings_v1` in `.omx/state/canonical_equations_registry.jsonl`.

The accidental duplicate append from rerunning the smoke was removed before commit; the registry now has one pair #2 anchor:

`third_empirical_anchor_wave_3_magic_codec_pair_2_sparse_packet_ir_fec6_null_byte_smoke_20260521T002120Z`

## Next Action

Do not spend paid eval or exact-eval queue on pair #2 as currently formulated.

Next frontier-moving path should be one of:

1. Pair #4 orthogonality validation from the magic-codec stacking matrix.
2. DP1 procedural paired-smoke recipe authoring, where the procedural-codebook hypothesis is tied to an intermediate codebook/basis rather than raw FEC6 member-byte substitution.
3. LL scorer-surrogate / frame-pair planner wiring that consumes the null-byte matrix and per-frame decomposition as training-data prioritization, not as a direct byte-replacement claim.

Further FEC6 selector entropy polish remains below the materiality threshold unless it changes components or supplies a consumed residual/runtime adapter.
