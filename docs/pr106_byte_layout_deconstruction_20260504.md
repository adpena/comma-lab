# PR106 belt_and_suspenders byte-layout deconstruction (2026-05-04)

Reverse-engineered from `experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/source/submissions/belt_and_suspenders/src/codec.py` (`parse_packed_archive` + `decode_packed_decoder` + `decode_fixed_latents`).

## Archive byte layout (0.bin)

```
Total: 186,131 bytes
═══════════════════════════════════════════════════════════════════════════
Offset    Bytes   Field                                              Notes
─────────────────────────────────────────────────────────────────────────────
0         1       magic 0xFF                                         dispatch byte
1         3       dec_len (uint24 little-endian)                     value=170,278
4         170278  brotli-compressed PACKED decoder weights           dec_len bytes
170282    15849   brotli-compressed delta-encoded latents            to end
═══════════════════════════════════════════════════════════════════════════
```

## Decoder (170,278 brotli → 229,070 raw, 1.35× compression)

`decode_packed_decoder(data)`:

```
brotli.decompress(170278 bytes) → 229,070 raw bytes
Layout:
  ┌────────────────────────────────────────────────────────┐
  │ 28 × packed int8 zigzag tensor payloads                │  ← 228,958 bytes
  │   (schema hardcoded in PACKED_STATE_SCHEMA — no        │     (1 byte/param,
  │    per-tensor names/shapes in payload!)                │      schema-driven)
  ├────────────────────────────────────────────────────────┤
  │ 28 × float32 scales (one per tensor)                   │  ← 112 bytes
  └────────────────────────────────────────────────────────┘
Total: 229,070 bytes
```

Per-param cost: **0.7437 bytes/param** (170,278 / 228,958). Brotli on int8 zigzag is BELOW int8's nominal 1 byte/param.

PACKED_STATE_SCHEMA (28 entries, 228,958 params total — first 5 shown):

| name | shape | params |
|---|---|---:|
| stem.weight | (1728, 28) | 48,384 |
| blocks.0.weight | (144, 36, 3, 3) | 46,656 |
| blocks.1.weight | (144, 36, 3, 3) | 46,656 |
| blocks.2.weight | (108, 36, 3, 3) | 34,992 |
| blocks.3.weight | (80, 27, 3, 3) | 19,440 |
| ... | ... | ... |

## Latents (15,849 brotli → 33,712 raw, 2.13× compression)

`decode_fixed_latents(data)` for 600 frame-pairs × 28 latent dim:

```
brotli.decompress(15849 bytes) → 33,712 raw bytes
Layout (n=600 pairs, d=28 latent dim):
  ┌────────────────────────────────────────────────────────┐
  │ lo bytes (n*d = 16,800 uint8)                          │  ← low byte of (hi<<8)|lo
  ├────────────────────────────────────────────────────────┤
  │ mins (d × float16 = 56 bytes)                          │  ← per-dim min
  ├────────────────────────────────────────────────────────┤
  │ scales (d × float16 = 56 bytes)                        │  ← per-dim scale
  ├────────────────────────────────────────────────────────┤
  │ hi bytes (n*d = 16,800 uint8)                          │  ← high byte
  └────────────────────────────────────────────────────────┘
Total: 33,712 bytes
```

Decoding: `delta_zz = (hi << 8) | lo`; reconstruct via cumulative-sum of zigzag-decoded deltas; dequantize as `q[i] * scales[d] + mins[d]`.

Per-frame-pair cost: **15,849 / 600 = 26.4 bytes/pair** (28 dims × 16-bit zigzag delta + tiny per-dim metadata, brotli-compressed).

## Why apogee_intN is +1.5KB heavier than PR106 (int8 case) — CORRECTED 2026-05-04 evening

Empirically measured layout breakdown (apogee_int4..8 + PR106-int8):

| bits | total | name_overhead | weight_payload | latent | name_overhead % |
|---|---:|---:|---:|---:|---:|
| apogee_int4 | 109,888 | 564 | 93,471 | 15,853 | 0.51% |
| apogee_int5 | 154,447 | 564 | 138,030 | 15,853 | 0.37% |
| apogee_int6 | 170,342 | 564 | 153,925 | 15,853 | 0.33% |
| apogee_int7 | 205,050 | 564 | 188,633 | 15,853 | 0.28% |
| apogee_int8 | 187,623 | 564 | 171,206 | 15,853 | 0.30% |
| **PR106** (int8 PACKED) | **186,131** | **4** | **170,278** (brotli) | **15,849** | **0.00%** |

The **+1,492-byte gap** between apogee_int8 (187,623) and PR106 (186,131) splits as:
- **560 bytes** = per-tensor name+shape overhead (apogee_int8's 564 vs PR106's 4)
- **928 bytes** = weight-payload overhead (apogee_int8's 171,206 vs PR106's 170,278)

The **560-byte schema overhead** is the *minor* component. The **928-byte weight-payload overhead** is the *major* component — it's the cost of per-tensor brotli (28 separate brotli streams) vs PR106's single-stream brotli over the entire int8 zigzag corpus. Brotli's dictionary + Huffman tables amortize across one larger stream more efficiently than across 28 small streams.

### Two independent savings opportunities

| Optimization | Savings (bytes, all bits) | Score Δ | Trade-off |
|---|---:|---:|---|
| PACKED schema (drop names+shapes) | ~422 | 0.000281m | Lose generality across architectures |
| Single-stream brotli (vs per-tensor) | ~928 (at int8; less at lower bits) | 0.000618m | None — pure encoder improvement |
| **Combined** | **~1,350** | **~0.000899m** | Schema-locked to one architecture |

Both savings are *minor* relative to the bit-width reduction (apogee_int5 already wins -31KB / -0.021 score Δ vs PR106 by going from 8→5 bits). The single-stream brotli optimization is the cleaner win: same code path, no architecture lock-in.

### Why this matters less than initially thought

The original deconstruction memo speculated the gap was "~1,500 bytes from per-tensor name+shape overhead." Empirical measurement shows the schema overhead is actually 560 bytes — most of the gap (928 bytes) is in the brotli encoding strategy, not the schema layout. The bit-width axis dominates everything:

- 8→5 bits: -31,576 bytes (-0.021 score Δ — order of magnitude bigger)
- single-brotli + PACKED schema: -1,350 bytes (-0.0009 score Δ — rounding error in comparison)

## Why apogee_int5 BEATS PR106 by 31KB

apogee_int5 (154,555 bytes) vs PR106 (186,131 bytes):

| Component | PR106 (int8 brotli) | apogee_int5 | Δ |
|---|---:|---:|---:|
| Decoder | 170,278 | ~138,706 | -31,572 |
| Latents | 15,849 | 15,849 | 0 |
| **Total** | **186,131** | **154,555** | **-31,576** |

The savings come from **5 bits/param vs 8 bits/param** (-37.5% per weight) outweighing the per-tensor name+shape overhead. Intuition: at 8 bits/param the schema overhead is significant; at 5 bits the weight payload dominates.

## Implications for next dispatch

The PR106 packed layout is the **schema-driven extreme** of int8 — everyone who keeps int8 weights pays at least 0.74 bytes/param. To beat it:

1. **Lower bits-per-weight** (Lane #04 apogee_intN, predicted -0.05 score Δ at int4)
2. **Sensitivity-weighted bit allocation** (Lane Ω-W-V3, predicted -0.014 rate Δ via stub)
3. **Score-Jacobian residual** (Lane SJ-KL, no PR106 stacking — uses C067 base)
4. **Combination**: int5 base × per-channel water-fill on top of the int5 weights → predicted ADDITIVE rate savings

For the PR106-stacking Pareto, the entry point is **APOGEE_INTN_BITS=5** (sweet spot per `tools/apogee_intN_pareto.py`). Lane Ω-W-V3 stacks orthogonally — different codec family.

## Cross-refs

- Public-frontier intake: `experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/`
- Source codec (read-only, public): `.../source/submissions/belt_and_suspenders/src/codec.py`
- Apogee int4..8 producer: `experiments/repack_pr106_with_intN_block_fp.py`
- Apogee inflate adapter: `submissions/apogee_intN/inflate.py` (codec_id 0 = brotli-int8 single-tensor; codec_id 5 = intn block-FP)
- Pareto + dispatch tooling: `tools/apogee_intN_pareto.py` + `tools/all_lanes_preflight.py`
- Sister lane (water-fill v2 on extracted decoder): `scripts/remote_lane_omega_w_v3_pr106.sh`
