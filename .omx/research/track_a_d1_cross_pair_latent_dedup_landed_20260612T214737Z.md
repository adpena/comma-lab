# Track-A ITEM D1 — Cross-Pair Latent Dedup: BUILT + SEALED (measured negative on current latents)

**Owner:** PARTNER D1 (cross-pair latent dedup rate lever).
**Date:** 2026-06-12T21:47Z.
**Authority discipline:** every number here is `[macOS-CPU advisory] NON-PROMOTABLE` until
`upstream/evaluate.py` on the byte-closed archive (ITEM E). `SCORE_CLAIM=False`.

## TL;DR (data, not prose)

- **Module (CORE, carrier-agnostic):** `src/tac/losses/cross_pair_latent_codec.py`
- **Adapter (thin, base_ch20):** `build_latent_blob_dedup_or_vendored` + `decode_latent_blob` (same file)
- **Tests (NO-FAKE):** `src/tac/tests/test_cross_pair_latent_codec.py` — **27 tests, all pass**
- **MEASURED deployed win on the 3 real base_ch20 latent tensors:** **0 B on all three** (byte-identical).
- **Verdict:** an **HONEST MEASURED NEGATIVE** for the cross-pair-redundancy thesis on the
  current base_ch20 latents — the vendored 1st-order-delta + lo/hi + brotli is already within
  ~1.3% of the symbol-entropy floor and beats every alternative measured. The apparatus is
  REAL, data-driven, and produces a real **−616 B** win on latents that DO have cross-pair
  structure (control), so it will fire automatically on Track B / future latents with genuine
  temporal redundancy. The default-preserving guard makes wiring it cost nothing.
- **Recursive adversarial review:** 1 finding (Lens-2, fixed + regression-tested) → counter
  RESET → **3 consecutive fresh clean lenses** → **SEALED**.

## The measured reality (why dedup is a negative on THESE latents)

Real base_ch20 latents are `(600, 28)` — one quantized pair-code per frame-PAIR. Measured on
`basin` / `arm` / `mps` `best_ema_latents.pt`:

| mechanism | Δ vs vendored (basin) | why |
|---|---|---|
| cross-pair **exact dedup** | N/A — **0 dups** (600→600 unique) | no two pair-codes are identical |
| shared-**codebook / VQ** (K=64/128/256) | **+1584 B** | codebook+index overhead > savings; latents too diverse |
| **2nd-order / motion** delta | **+1747 B** | latents are NOT temporally smooth (2nd-delta entropy 8.15 > 1st 7.37 b/sym) |
| static **global range coder** | **+113 B** | explicit freq-table side cost exceeds brotli's online-adaptive model |
| **per-dim** range coder | **+4075 B** | 28× freq-table overhead dominates |
| side-info reorder | ±8 B | brotli alignment noise |

The vendored 1st-order-delta lo/hi+brotli ≈ 15800 B; the AC symbol-entropy **floor** is
≈ 15595 B (H=7.369 b/sym). The vendored codec is **already ~1.3% above the floor** and **beats a
real working range coder** because brotli adapts its model online with no explicit freq-table
cost. There is no cross-pair redundancy left to code out on these specific latents.

## Final MEASURED deployed win (the 3 real tensors) — `[macOS-CPU advisory] NON-PROMOTABLE`

| tensor | vendored_B | our_B | ΔB | implied ΔS | is_framed |
|---|---:|---:|---:|---:|---|
| basin | 15800 | 15800 | **+0** | **+0.0000000** | False |
| arm   | 15838 | 15838 | **+0** | **+0.0000000** | False |
| mps   | 15574 | 15574 | **+0** | **+0.0000000** | False |

Full end-to-end archive proof (Lens 3): basin full archive 89570 → 89570 B (**byte-identical**),
latents parse back **bit-exact** through the unified decoder. The honest negative survives the
full archive build→parse-back path.

## The apparatus IS real (NO-FAKE control: −616 B on structured latents)

On run-structured latents (40 modes, repeat-runs — genuine cross-pair redundancy), the selector
picks **DEDUP** and produces a **real −616 B win** (latent blob 1851→1235 B; full archive
75621→75005 B) that **survives parse-back bit-exact**. Candidate post-brotli sizes prove the
formats are DISTINCT implementations, not enum-padding: DEDUP 1089 < CODEBOOK 1329 <
FRAMED_DELTA 1864. A no-op codec that returned its input would fail the strict `<` savings
assertion (`test_a_no_op_codec_would_fail_the_savings_assertion`).

## Design (CORE / ADAPTER split — carrier-agnostic)

- **CORE** (`quantize_pairs`, `dequantize_pairs`, `encode_latents_best`, `decode_latents_best`,
  the per-format encoders/decoders): operate on a generic `(n_pairs, latent_dim)` float tensor;
  **import NO base_ch20 / vendored constant** (AST-verified — the word "vendored" appears only in
  docstrings). Reusable on Track B unchanged.
- **Measure-and-select**: `encode_latents_best` builds 3 framed candidates (FRAMED_DELTA, DEDUP,
  CODEBOOK), brotli-compresses each, and returns the smallest. `structural_only=True` restricts
  selectable winners to DEDUP/CODEBOOK (used by the adapter — see the Lens-2 finding).
- **Quantization parity**: byte-for-byte the vendored per-dim asymmetric uint8 + min/max/254 grid,
  so the reconstructed latents are IDENTICAL across every format (apples-to-apples vs vendored).
- **Grammar (framed)**: `1-byte format flag + (n,d) + fp16 mins + fp16 scales + format payload`.
  DEDUP = unique-row table (delta lo/hi) + temporal-delta-zigzag index. CODEBOOK = uint8
  codebook + index + EXACT temporal-delta residual (round-trips regardless of codebook quality —
  the codebook is a predictor, the residual carries the exact remainder).
- **ADAPTER** (`build_latent_blob_dedup_or_vendored`): brotli-wraps BOTH the unframed vendored
  payload and the best STRUCTURAL framed candidate; returns the smaller. On a tie / no-win it
  returns the **EXACT vendored brotli bytes** (`is_framed=False`) → archive byte-identical. The
  inflate seam is one `if` (`decode_latent_blob(blob, is_framed)`).
- **NOT wired into driver.py** (per the contract — that is ITEM E's gate). Clean wire-in seam left:
  the driver's archive build calls `build_latent_blob_dedup_or_vendored` instead of
  `brotli.compress(codec.encode_latents(...))` and persists the 1-bit `is_framed` flag in meta;
  inflate calls `decode_latent_blob`. Default-OFF = no flag set = byte-identical.

## Recursive adversarial review (owner-run, 3-clean gate)

**FINDING (Lens 2, round 1):** the adapter falsely emitted `is_framed=True` on ~18/40 iid-noise
seeds, claiming 1–29 B "wins". Root cause: FRAMED_DELTA is information-IDENTICAL to vendored
(`framed[1:]==vendored_payload`), so its byte difference under brotli is pure **block-ALIGNMENT
noise** from the prepended 1-byte flag — NOT a real saving. **Fix:** the adapter selects with
`structural_only=True` — only DEDUP/CODEBOOK (genuine cross-pair-redundancy formats) may win;
FRAMED_DELTA is the decoder-consistent fallback, never a reported win. Regression-tested
(`test_adapter_never_emits_framed_on_iid_noise_brotli_alignment_artifact`,
`test_encode_latents_best_structural_only_excludes_framed_delta_win`). **Counter reset to 0.**

**FRESH 3-clean-pass record (post-fix):**
- **Lens 1 (bit-exact round-trip + NO-FAKE):** 336 configs (n∈{1,2,3,7,64,600,1000} ×
  d∈{1,2,28,64} × scale∈{1e-3,1,1e3} × {iid,repeat,run,const}) × all candidates × both selector
  modes → **0 round-trip failures**. n=0 raises cleanly (no silent corruption). **CLEAN → 1/3.**
- **Lens 2 (default-preserving byte-identity):** 100 iid seeds (varied n/d/scale) → 0 byte
  mismatches, **0 spurious framed-delta wins**, adapter deterministic. **CLEAN → 2/3.**
- **Lens 3 (real deployed win + survives-inflate):** full archive build→parse-back on real basin
  decoder+latents → 89570→89570 B byte-identical, latents bit-exact; redundant control −616 B
  survives parse-back end-to-end. **CLEAN → 3/3.**

**3/3 clean → SEALED.**

## NO-FAKE test list (27 tests)

quant/dequant parity (3): `test_quant_dequant_round_trips_quantized_codes`,
`test_quant_matches_vendored_encode_decode_when_available`, `test_quant_codes_in_valid_range`.
per-format bit-exact round-trip (3): `test_framed_delta_round_trip`,
`test_dedup_round_trip_on_repeated_rows`, `test_codebook_round_trip_is_exact_despite_lossy_centroids`.
adversarial inputs (6): all-equal pairs, all-distinct pairs, single pair, max-range symbols,
high-dim (256), uint32-index-width path.
NO-FAKE distinctness/savings (4): `test_dedup_strictly_beats_framed_delta_on_redundant_latents`,
`test_candidate_formats_are_distinct_implementations`, `test_selector_picks_framed_format_when_it_wins`,
`test_a_no_op_codec_would_fail_the_savings_assertion`.
default-preserving (4): byte-identical-when-no-win, tie-keeps-vendored,
**never-emits-framed-on-iid (the Lens-2 regression)**, structural_only excludes framed-delta.
real latents (6, skip-if-absent): 3× bit-exact round-trip + 3× byte-identical measured-negative guard.
discipline (1): `test_score_claim_discipline_flags`.

## 6-hook wire-in (Catalog #125)

1. sensitivity-map — N/A (lossless codec; no per-axis distortion contribution).
2. Pareto constraint — ACTIVE: the latents-section byte cost is a rate-term constraint; this lever
   is a no-op on it for the current latents (measured), a −616 B reducer when redundancy exists.
3. bit-allocator hook — N/A (the codec is exact/lossless; no bit budget to allocate).
4. cathedral autopilot dispatch — N/A (not driver-wired by contract; ITEM E gate).
5. continual-learning posterior — the measured negative + the −616 B control are the empirical
   anchors recorded in this memo.
6. probe-disambiguator — ACTIVE: `encode_latents_best` IS the measure-and-select disambiguator
   (it empirically picks the smallest format per-latent-tensor; the negative is its verdict here).

## Honest recommendation for the next mechanism

Cross-pair-redundancy coding is exhausted as a rate lever **on these latents** (near
incompressible past vendored). If a rate win on the latent section is still wanted, the only
direction with theoretical headroom is **closing the ~1.3% gap to the symbol-entropy floor**
(~205 B on basin) with a coder that has NO explicit freq-table cost — i.e. an **adaptive
context/range coder** (not the static one measured here, which loses on the freq table). That is
a small, uncertain ~205 B (≈ −0.00014 ΔS) ceiling and likely not worth the inflate-path
complexity vs the decoder-weight rate levers (ITEM B/D3). **The biggest remaining base_ch20 lever
is the d_seg distortion term (the distortion arm), not the latent rate.**
