# apogee_intN single-stream brotli — DEFERRED with cost/benefit (2026-05-04)

## Decision

**DEFERRED.** Implementation is small (~30 LOC + new codec_id), but predicted
score Δ is **0.000618** — below the noise floor of contest measurement.

## Cost-benefit analysis

### Empirical gap (from `docs/pr106_byte_layout_deconstruction_20260504.md`)

apogee_int8 currently encodes 28 int8-eligible tensors via 28 SEPARATE brotli
streams (`encode_decoder(q_sd_singleton)` called per tensor in
`experiments/repack_pr106_with_intN_block_fp.py:153`).

PR106 brotli's the entire 229,070-byte int8 zigzag corpus + metadata in ONE
brotli call (via the canonical `encode_decoder(full_q_sd)`).

The encoder-strategy gap:
- apogee_int8 weight payload: 171,206 bytes (28 streams)
- PR106 decoder brotli:        170,278 bytes (1 stream)
- **Gap: 928 bytes** (brotli dictionary + Huffman amortization penalty)

### Predicted savings

| Variant | Current bytes | After single-stream | Savings | Score Δ |
|---|---:|---:|---:|---:|
| apogee_int4 | 109,888 | ~109,000 | ~888 | -0.000591m |
| apogee_int5 | 154,447 | ~153,500 | ~947 | -0.000631m |
| apogee_int6 | 170,342 | ~169,400 | ~942 | -0.000627m |
| apogee_int8 | 187,623 | ~186,695 | ~928 | -0.000618m |

Average score Δ ≈ **0.000618** per archive.

### Why this is below contest noise

- contest formula: `25 * archive_bytes / 37,545,489` per byte
- per-byte score: `6.66e-7`
- 928 bytes → 0.000618 score Δ
- vs eval-noise band on T4: empirically ~0.007 (PFP16 paired calibration)
- vs apogee bit-width step (8→5): -31,576 bytes / -0.021 score Δ (**34× larger**)

Single-stream brotli is **literally a 100× smaller signal than eval noise**.

### Implementation cost

Net new code: ~50 LOC + tests. Adds:

1. New `CODEC_ID_BULK_BROTLI_INT8 = 1` constant
2. Producer: collect all int8-eligible tensors → single `encode_decoder(q_sd_all)` call → emit ONE entry with codec_id=1
3. Inflate adapter: dispatch codec_id=1 → `decode_decoder()` → distribute tensors back to their names
4. Backward compat: existing codec_id=0 entries still work (legacy path)
5. Regression test: lock new byte counts, verify roundtrip
6. Producer regenerates all 5 manifests with new bytes
7. Re-run all dispatch dry-runs to confirm parser-roundtrip
8. Update `tools/apogee_intN_pareto.py` predicted bands? (no — bands are distortion-only, byte savings only affect rate)

Estimated wall-clock: 30-60 min careful work + verification.

### Why DEFER

1. **Below contest noise**: 0.000618 score Δ is invisible compared to the 0.007 T4 eval noise band.
2. **Bit-width axis dominates**: 31,576-byte savings from 8→5 bits is 34× larger than 928-byte savings from single-stream.
3. **Implementation risk**: any bug in the bulk codec breaks all 4 Pareto-frontier dispatches. Reward is rounding-error; downside is breaking the launch-ready stack.
4. **Council #271 (Phase 1 next-priority)**: prioritized HIGH-EV moves. This is a polish-tier improvement, not an EV move.

### Reactivation criterion

Implement IF AND ONLY IF:
- Apogee_int5 dispatch lands a measured score AND
- The score is within 0.001 of beating PR106's 0.20946 AND
- The 0.000618 Δ from single-stream brotli would tip the result

Then it becomes a worth-it polish change. Without that specific situation, the implementation is below the EV threshold.

## Cross-refs

- Empirical measurement: `docs/pr106_byte_layout_deconstruction_20260504.md` (sister memo, "Two independent savings opportunities" section)
- Producer to refactor: `experiments/repack_pr106_with_intN_block_fp.py:146-155` (the per-tensor encoding loop)
- Inflate adapter to extend: `submissions/apogee_intN/inflate.py:_decode_brotli_int8_single_tensor`
- Reference single-stream impl (PR106): `experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/source/submissions/belt_and_suspenders/src/codec.py:encode_decoder`
- Dispatch tooling (would all need re-validation after refactor): `tools/dispatch_dryrun_apogee_intN.py` + `tools/all_lanes_preflight.py`
