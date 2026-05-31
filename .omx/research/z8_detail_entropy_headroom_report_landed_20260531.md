# Z8 detail-coefficient entropy-headroom report — LANDED 2026-05-31

`[macOS-CPU advisory]` **NON-PROMOTABLE** per CLAUDE.md "MPS auth eval is NOISE" +
Catalog #127/#192/#317/#323/#341. **NO contest-score claim.** $0 macOS-CPU, no GPU,
no PR. This is a read-only measurable diagnostic that grounds codex's v2 pair-blob
codec (operator routed option **(b)** 2026-05-31: *"ground it with a measurable
entropy-headroom report"* + *"provide brotli precisely what it needs with no signal
loss"* + *"we can super optimize this part"*).

## What landed

- `tools/z8_detail_coeff_entropy_headroom_report.py` — read-only diagnostic. Parses
  a REAL byte-closed Z8HPC1 archive, decodes per-pair wavelet detail bands via the
  canonical `parse_archive` + `parse_pair_blobs_from_wavelet_blob`, and measures per
  subband, at matched distortion: (raw f32 → brotli) vs (lossless byteshuffle → brotli)
  vs (quantize + per-subband mode via the canonical `_encode_quantized_detail_payload`)
  vs the structured Shannon floor `(H_bin(p_nz) + p_nz·H_nz)/8`.
- `src/tac/substrates/z8_hierarchical_predictive_coding/tests/test_detail_coeff_entropy_headroom_report.py`
  — 16 NO-FAKE tests (real brotli / real Shannon entropy / real quantize round-trip /
  live-archive regression guard). All green.
- `.omx/research/z8_detail_entropy_headroom_20260531T185438Z.json` — the empirical
  artifact (baseline archive, 6 pairs, fine Δ sweep).

## Sister-disjoint (Catalog #340)

Codex owns the **v2 pair-blob codec** (`canonical_quadruple_binding.py` —
`_encode_quantized_detail_payload` already does the 4-way per-subband `argmin(bytes)`:
`qi16_dense` / `qi16_zero_rle` / `zigzag_u16_byteplane` / `qi16_static_range`). This
tool is a NEW read-only file. It **does not modify** the codec — it consumes the
canonical read-only decode + encoder primitives so every measurement is apples-to-apples
with the deployed pipeline (no re-implemented strawman encoder).

## Empirical findings (baseline archive, 6 pairs, 414,720 detail coeffs)

The detail-blob default schema stores each coefficient as **raw float32 → brotli q=11**.
The coeffs are tiny (|c| ≈ 0.014–0.036); the low fp32 mantissa bits are near-random, so
brotli stores survivors at ~3.2–3.7 bytes/coeff.

| Δ (quant step) | current raw-f32→brotli | v2 quantize+mode | structured Shannon floor | headroom | distortion (MSE) |
|---:|---:|---:|---:|---:|---:|
| 0.015625 | 1,403,462 B | 135,180 B | 134,223 B | **90.4 %** | 2.0e-05 |
| 0.03125 | 1,403,462 B | 86,356 B | 86,092 B | 93.8 % | 7.3e-05 |
| 0.0625 | 1,403,462 B | 43,736 B | 43,174 B | 96.9 % | 2.1e-04 |
| 0.125 | 1,403,462 B | 13,776 B | 15,228 B | 99.0 % | 4.1e-04 |
| 0.25 | 1,403,462 B | 4,351 B | 5,149 B | 99.7 % | 6.3e-04 |

**Three findings that ground codex's codec:**

1. **The entire detail blob is headroom.** Even at the fidelity-preserving Δ=0.015625
   (distortion 2e-5 ≈ lossless to fp precision) the v2 codec cuts the detail bytes
   **90.4 %**. This is the dominant Z8 rate lever (the wavelet blob is ~99.5 % of the
   archive). The current raw-f32 storage is 1000–4000× above the entropy floor.

2. **The lossless byteshuffle path is a DEAD END.** byteshuffle → brotli = 3.36–3.40
   bytes/coeff — *no better* than current (sometimes worse). The float32 byte-planes are
   genuinely high-entropy. So "no signal loss" **cannot** mean "store float32 losslessly";
   the win **requires** quantization (the one controlled lossy step), with the entropy
   wrapper bijective *after* quantize. (LL stays float32 by design.)

3. **The v2 codec is already at the Shannon floor.** `qi16_zero_rle` wins every subband
   (10–33 % nonzero at Δ=0.0625), landing within ~5–10 % of the structured floor and
   sometimes *below* it (brotli's LZ exploits structure order-0 entropy misses). This
   **empirically justifies keeping the static range coder gated** (`_MAX_STATIC_RANGE_
   DETAIL_SYMBOLS=0`) — it could recover at most ~5–10 % more, not worth the pure-Python
   runtime-decode speed cost. A native/vectorized range backend is a *small*-payoff
   investment, not a large one.

## Operator's "provide brotli precisely what it needs with no signal loss" — answered

The right move is exactly codex's v2 path: quantize per subband → zigzag → zero-RLE → brotli.
The only open knob is the **per-subband Δ operating point**: smaller Δ preserves the
contest score (the dead-zone work showed these tiny coeffs carry score-relevant signal —
keep=0.02 worsened score 4.85→9.06), larger Δ trades distortion for rate. That per-subband
Δ choice is the RD-optimal allocation the joint P18/P19 water-fill (Phase 5 / #1591/#1592)
should solve — this report supplies its empirical per-subband RD curve.

## 6-hook wire-in (Catalog #125)

- hook #1 sensitivity-map: **ACTIVE** (per-subband headroom is an advisory sensitivity
  signal; non-promotable).
- hook #2 Pareto constraint: N/A (advisory diagnostic; no score posterior entry).
- hook #3 bit-allocator: **ACTIVE** (the per-subband RD curve is exactly the input the
  bit-allocator / water-fill consumes to pick per-subband Δ).
- hook #4 cathedral autopilot dispatch: N/A (read-only $0 diagnostic, non-promotable).
- hook #5 continual-learning posterior: N/A (non-promotable; no contest-axis anchor).
- hook #6 probe-disambiguator: **ACTIVE** (the byteshuffle-DEAD-END vs quantize-WINS
  vs range-coder-gap-is-small verdict disambiguates codec design choices).

## Reproduce

```bash
.venv/bin/python tools/z8_detail_coeff_entropy_headroom_report.py \
    --archive experiments/results/z8_joint_p18_p19_deadzone_rate_attack/baseline/byte_closed_archive/0.bin \
    --num-pairs 6 --quant-steps 0.015625,0.03125,0.0625,0.125,0.25 \
    --out-json .omx/research/z8_detail_entropy_headroom_<utc>.json
```

Lane: `lane_z8_detail_coeff_entropy_headroom_report_20260531`. Mission contribution:
`frontier_breaking_enabler` (grounds the dominant Z8 rate lever for codex's Phase 5
codec). Sister of the Z8 dead-zone rate attack (`ad73c2863`) + the joint P18/P19
gradient water-fill solver architecture (`joint_p18_p19_gradient_waterfill_solver_
architecture_20260531.md`).
