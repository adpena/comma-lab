# Council Design Review — Lane 20: Ballé Hyperprior on Renderer qint Stream

**Status:** Phase A council review for Level 1 → Level 3 graduation.
**Anchor:** Lane G v3 = 1.05 [contest-CUDA] (DILATED_H64_HALF_FRAME_V3_ANNEALED_KLDISTILL).
**Predicted band [prediction]:** -3% to -8% archive bytes on the renderer qint segment (~5–15 KB out of ~180 KB renderer.bin); score delta -0.001 to -0.005.
**Cost estimate:** $0.50 (training small hyperprior + auth eval).
**Dependencies:** existing `src/tac/arithmetic_qint_codec.py` (static factorized prior baseline) and `src/tac/balle_hyperprior_renderer.py` (Level 1 scaffold — landed 2026-04-29).

## 1. Existing scaffold audit

`src/tac/balle_hyperprior_renderer.py` (421 LOC, 8 tests passing):

- `class ScalePriorMLP(nn.Module)` — context-vector → per-element σ via softplus
- `gaussian_rate_bits(qint, sigma)` — closed-form `-log2 N(0,σ²)` continuous approximation
- `static_factorised_rate_bits(qint)` — single global σ baseline
- `encode_balle_hyperprior` / `decode_balle_hyperprior` — emits FP16 MLP weights + magic header (BHP1)
- `amortisation_break_even_bytes(scale_prior, savings_fraction)` — threshold computation

**What is NOT present (the gap to Level 3):**

1. **No actual arithmetic coder integration.** The scaffold computes rate as a closed-form continuous Gaussian. A real codec must:
   - Discretize the Gaussian into per-symbol probabilities `p(y=k|σ) = ∫_{k-0.5}^{k+0.5} N(t|0,σ²) dt`
   - Feed those probabilities into the existing `_ArithmeticEncoder` from `arithmetic_qint_codec.py`
   - Decode by re-computing per-symbol probabilities from the SAME hyperprior σ values (the σ stream lives in side-info)
2. **No "context" wiring.** The MLP takes `(B, context_dim)` but nothing produces the context. Production design: chunk qint stream into K blocks; per-block σ is predicted from a learned per-block context vector (or a tiny per-block hyper-latent z).
3. **No training loop.** Lane 20 is a learned codec; the MLP weights need fitting on real qint data.
4. **No archive integration.** The codec is currently a freestanding utility; it must wire into either:
   - `repack_payload_tar_xz_to_arithmetic` (Selfcomp-style payload — but Lane G v3 is FP4A-encoded, not block-FP)
   - OR a renderer-archive variant `BHv1` magic-byte dispatch (analogous to OWV2)
5. **No real-archive empirical measurement.**
6. **No STRICT preflight.**
7. **No 3-clean-pass adversarial review.**
8. **No remote_lane script.**

## 2. Council deliberation

### Ballé (CHANNELED — inner council seat, 2018 entropy bottleneck author)

**Architecture:** The original 2018 paper uses 1×1 convs over 2-D latents — but our qint stream is 1-D (flattened conv weights). The minimal faithful adaptation is:

> y is the 1-D qint stream (block-flattened). Hyper-latent z is a small block-context vector (one per block of size B = 256–1024 qints). σ is predicted by the hyper-decoder `h_s : z → σ_per_block`. The hyper-encoder `h_a : y_block → z` is a small MLP. Both `z` and `y` are arithmetic-coded; z under a fixed factorized prior, y under N(0, σ²) discretized.

**Critical decision points:**
- **Block size** (B): too small → side-info dominates; too large → loses heteroscedasticity. **Recommend B=256** for ~180KB Lane G v3 (≈700 blocks). Verify empirically.
- **Side info encoding:** z is also discretized; z values quantized to int4 (range ±7) and arithmetic-coded under a learned 1-D piecewise-linear CDF. Tinier alternative: int8 z directly stored without arithmetic coding (the extra bytes vs entropy gain matter only when |z| > 100).
- **Conditional density discretization:** `p(y=k|σ) = Φ((k+0.5)/σ) - Φ((k-0.5)/σ)` where Φ is standard-normal CDF. Numerically stable for σ ∈ [0.05, 10]. Below that, hard-clamp to {-1, 0, +1} ternary ratios. Above that, the static fallback is no worse.
- **Training:** end-to-end with `bits_y + bits_z` loss; STE for the round operation on z and y. EMA decay 0.997 (CLAUDE.md).
- **GDN nonlinearity:** the 2018 paper's GDN is for 2-D images; on a 1-D weight stream, GELU is the practical equivalent (already in scaffold). Defer GDN to V2.

**Verdict — Ballé seat:** GREEN. Architecture is the canonical 2018 hyperprior shape adapted to 1-D weight streams. The block-context decoder side-info cost is the critical knob; B=256 with 8-D z gives ~16 bytes side-info per block × 700 blocks ≈ 11 KB side-info — needs to amortize against ~5–15 KB savings on the y-stream. Borderline but plausible.

### Shannon (LEAD — information theory)

**Math foundation:**
- Static factorized rate (current arithmetic codec): `R_static = N · H(p_global)` where p_global is the empirical 1-D histogram. Per-block factorization assumes y_i are i.i.d. across blocks — wrong if blocks have different scales.
- Hyperprior rate: `R_hyper = sum_b sum_i [-log2 p(y_{b,i} | σ_b)] + R_z` where R_z is bits to encode the z stream.
- **Win condition** (cross-entropy ≤ entropy bound): `R_hyper < R_static` iff the heteroscedasticity savings on y-stream exceed R_z. By Jensen's inequality, `H(Y) ≥ H(Y|Z)` always, so the y-conditional rate is bounded above by the unconditional rate; the only question is whether R_z + ε_train < the gap.

**Verifiable lower bound:**
- For Lane G v3 FP4 weights (5 bits/symbol uniform over {-7..-1, 0, 1..7}): empirical distribution is heavy near 0 (~30% zeros), tails sparse. Static cross-entropy ~3.6 bits/symbol on the qint stream. Optimal Shannon entropy on i.i.d. blocks: ~3.4 bits/symbol. Gap ~0.2 bits/symbol × 60K weight elements = ~1500 bytes max from breaking i.i.d. — IF heteroscedasticity exists across blocks.
- **The empirical question:** is there block-level heteroscedasticity in Lane G v3 FP4 weights? The Phase E measurement answers this. If σ_block stddev across blocks is < 1.2 × σ_global, hyperprior wins are negligible.

**Verdict — Shannon seat:** YELLOW. Math is sound but the Lane G v3 anchor is FP4-quantized (5-symbol alphabet) — heteroscedasticity may be small across the already-uniform-ish FP4 distribution. Recommend: also test Lane Ω-W-V2 anchor (block-FP qint stream with much wider per-block scale variation — that's where Ballé wins).

### MacKay (channeled — MDL grandmaster)

**MDL accounting:**
- Bits for hyperprior weights (need to ship): ~`MLP_params × 16` (FP16) ≈ 2–10 KB depending on arch.
- Bits for z-stream: `n_blocks × z_dim × bits_per_z_symbol`.
- Bits for y-stream: `n_y · H(y|σ̂)` where σ̂ is decoded from z.
- **Compare to:** baseline arithmetic codec bits = `n_y · H(y_static)` + ~50 bytes header.
- **Win iff:** `MLP_bits + R_z + n_y·H(y|σ̂) < n_y·H(y_static) + 50B`.

**Verifiable claim:** for Selfcomp-class block-FP streams (~11 KB qint), the MLP must be < ~2 KB or no win possible. For Lane G v3 FP4A (~25 KB qint segment within the larger renderer.bin), MLP can be up to ~5 KB.

**Verdict — MacKay seat:** GREEN with kill criterion: if MDL accounting at the END of training shows R_hyper ≥ R_static, abandon and ship the static arithmetic codec instead.

### Selfcomp (block-FP author, 0.38-scoring renderer)

> "Block-FP per-channel ALREADY adapts σ per-block (the per-channel exponent IS the σ selector — block-FP is a hand-coded hyperprior). What does Ballé add on top? The answer is: Ballé learns the σ per-block rather than choosing it greedily; the LEARNED σ minimizes total rate including the side-info cost, which the greedy max-abs-per-channel does not. Net win is small (1-3%) on already-block-FP streams; bigger (5-10%) on uniform-FP4 streams."

**Verdict — Selfcomp seat:** GREEN. But: ship Lane 20 atop the EXISTING arithmetic-codec output, not a new wire format. The Ballé hyperprior should be a drop-in REPLACEMENT for the static frequency table — its OUTPUT is the same `_ArithmeticEncoder` byte stream, just with conditional probabilities instead of a single global frequency table.

### van den Oord (VQ-VAE alternative)

> "If you're going to ship a learned codec for the qint stream, why a continuous σ predictor and not a VQ-VAE-style codebook? Lane J-NWC already does that. The differentiator: Ballé's continuous σ generalizes to unseen distributions; VQ-VAE is bounded to the codebook. For shipping a NEW renderer to the same lane, Ballé should re-train in seconds; a VQ-VAE would need new codebooks. So Ballé wins on amortization across renderers, VQ-VAE wins on absolute compression for a fixed renderer."

**Verdict — van den Oord seat:** GREEN. Ballé is the right choice for a shippable codec ON TOP of FP4A renderers, where the codec ships once and adapts via inference.

### Quantizr (adversarial — leader at 0.33)

> "I shipped FP4 + Brotli. My renderer.bin is ~64 KB. Lane 20 must beat 64 KB on the qint segment by >3 KB to be worth shipping. If your prediction is -3% to -8% on the qint segment of a ~25 KB Lane G v3 sub-payload, that's -750 B to -2 KB — BORDERLINE FOR THE LANE 20 BIT BUDGET. Add cost of the MLP weights (~2 KB) and you might NEGATIVE NET. Show me the empirical at end of Phase E or skip."

**Verdict — Quantizr seat:** YELLOW. Honest assessment: -3% net is plausible at best; net negative at worst. The empirical measurement is the gate.

### Hotz (raw engineering)

> "Just chunk the qint stream into K=4 parts, fit ONE static frequency table per chunk, ship K=4 little headers. 80% of the win at 1% of the complexity. Don't ship a NEURAL NETWORK to do what 4 histograms can do."

**Verdict — Hotz seat:** RED on full neural hyperprior; GREEN on chunked-static-prior as Lane 20-LITE. Strong contrarian — propose: implement BOTH the chunked-static (Hotz-LITE) and the full neural hyperprior; A/B in Phase E.

### Karpathy (advisory — let compute speak)

> "Run both. Phase E measures both. Ship the winner. Cost is $0.50 — don't let council debate burn 5x that in time."

**Verdict — Karpathy seat:** GREEN on dual-path implementation.

## 3. Decision

**Adopt:** Implement BOTH the full Ballé hyperprior AND the Hotz-LITE chunked-static prior. Empirically test both in Phase E. Ship the winner.

**Architecture (full Ballé):**
- `BalleHyperpriorCodec(block_size=256, z_dim=8, hidden=16, depth=3)`
- Hyper-encoder `h_a`: per-block (256 qint values) → 8-D z (small MLP, ~5 KB params at FP16)
- Round + arithmetic-code z under fixed Laplacian factorized prior
- Hyper-decoder `h_s`: 8-D z → 1 σ per block (existing `ScalePriorMLP`, ~1 KB at FP16)
- Per-symbol probability `p(y=k|σ_b) = Φ((k+0.5)/σ_b) - Φ((k-0.5)/σ_b)` for k ∈ alphabet
- Reuse `_ArithmeticEncoder` from `arithmetic_qint_codec.py` with per-symbol cumulative table updated per block

**Architecture (Hotz-LITE):**
- Split qint stream into K=4 contiguous chunks
- Fit a separate `build_freq_table` per chunk
- Header: K=4 + 4 freq tables + 4 chunk-payload sizes
- Decoder reads chunk-by-chunk

**Wire format (BHv1 magic-byte for both):**
```
magic           : 4 bytes  = b"BHv1"
version         : 2 bytes  uint16
mode            : 1 byte   (0=hotz_lite, 1=full_balle)
num_symbols     : 2 bytes  uint16
offset          : 4 bytes  int32
n_total         : 8 bytes  uint64
[mode-specific tables]
payload_size    : 8 bytes  uint64
payload         : payload_size bytes (arithmetic-coded body)
```

**Kill criteria:**
- If empirical R_hyper ≥ R_static on Lane G v3 qint stream: abandon Ballé, ship Hotz-LITE.
- If empirical R_hotz ≥ R_static + 50 B: abandon both, do not integrate Lane 20 — leave the existing arithmetic codec as-is.
- If integrated archive renders incorrectly under inflate.sh roundtrip: HARD ABANDON, do not ship.

## 4. Phase ordering (operational)

1. **Phase A** (this doc) — DONE
2. **Phase B** — extend `balle_hyperprior_renderer.py` to V2 with discretized-Gaussian arithmetic encode/decode + Hotz-LITE; preserve the Level 1 scaffold API
3. **Phase C** — 10+ tests covering both modes, including roundtrip on Lane G v3 fixture
4. **Phase D** — `compress_archive.py` integration + `inflate_renderer.py` BHv1 dispatch + remote_lane_balle.sh + profile
5. **Phase E** — empirical run on Lane G v3 qint stream → tag `[empirical:reports/lane_20_balle_real_archive.json]`
6. **Phase F** — contest-CUDA ($0.50 estimate; goes through if budget OK)
7. **Phase G** — STRICT preflight Check 91 (BHv1 must include side-info)
8. **Phase H** — 3-clean-pass adversarial review with rotating perspectives
9. **Phase I** — memory + registry

## 5. Cross-references

- CLAUDE.md "EMA — NON-NEGOTIABLE" (decay 0.997 on hyperprior weights at training time)
- CLAUDE.md "Auth eval EVERYWHERE" (Phase F is the auth-eval gate)
- CLAUDE.md "Recursive adversarial review protocol" (Phase H = 3 clean passes)
- CLAUDE.md "NEVER invent CLI flags" (Phase D Verify train_renderer.py argparse before subprocess wiring)
- `feedback_production_hardened_standard_definition_20260430.md` (the Level 3 bar)
- `project_phases_2_3_4_design_implementation_math_provenance_20260429.md` §"Lane 20 Ballé"
- `feedback_codec_stacking_composition_canonical_orders_20260429.md` (canonical order: arithmetic ALWAYS terminal — Lane 20 IS that arithmetic step)
- `src/tac/arithmetic_qint_codec.py` (the static-prior baseline)
- `src/tac/balle_hyperprior_renderer.py` (the Level 1 scaffold being extended)
