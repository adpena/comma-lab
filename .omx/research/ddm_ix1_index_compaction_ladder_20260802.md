# ddm_ix1 — the index is 0.003% of our bytes; the LAYOUT is 0.5%, and it was never raced

**UTC** 2026-08-03 · **arm** `ddm_ix1_index_compaction_ladder` · **axis** `[byte-closed rate, scorer-free]`
`score_claim=false`, `promotion_eligible=false`. **Pointer UNMOVED.**

**Vehicle:** `/Volumes/VertigoDataTier/pact/ddm_v4d_20260731/v4d_composed_dc1_fold_archive.zip`,
**360,309 B** (`dc1_fold`, our own-vehicle best, S = 0.8983775).
Rebuild parity verified: re-zipping the untouched members reproduces **360,309 B exactly**, so every
byte delta below is a true delta and not a framing artifact.

**Denominator:** gap to the PR130 demonstrated floor **0.7262358**; 1% of gap = **10,907 B**
(`tac.canonical_equations.gap_decomposition_against_floor_20260802`; PR130 = 191,052 B per `ddm_na1`).

---

## The answer first

1. **The index axis is real, solved, and ~200× too small to matter here.** Every index-shaped
   payload on the live vehicle totals **~35 B** of headroom = **0.003% of the gap**. MEASURED, not
   estimated. The operator is right that these are solved problems; the finding is that on OUR
   vehicle they are **not the binding constraint anywhere**. This kills a class of future arms.
2. **The widened charter was right by a factor of ~180.** Racing LAYOUT (struct-of-arrays) and
   sub-byte packing on the token stream — never done before on our own payload — is worth
   **−5,184 B** on the tokens member alone, **byte-closed and scorer-free**.
3. **Full measured stack: −6,144 B → ΔS −0.0040910 → −0.563% of the gap.**
   `0.8983775 → 0.8942865`. d_seg and d_pose are **invariant by construction** (§4).
4. **Round-1 self-review caught a 1,302 B fake in my own headline.** First result was −6,486 B; it
   shipped only the residual and reconstructed with a `base` array the decoder does not have. §4.
5. **The tokens are already at generic-coder incompressibility** (346,478 B raw → 346,483 B brotli,
   **+5**). The remaining win is not a better coder, it is a better **layout**.

---

## 0. MIGRATION AUDIT — is this byte VIDEO-SPECIFIC or GENERIC? (per field, MEASURED)

| member / field | bytes (in-zip) | verdict | migration |
|---|---:|---|---|
| `state/tokens.dr7t` — int4 lattice (600,24,32,4) residual | 344,942 | **VIDEO-SPECIFIC** | none — this is the payload |
| `state/tokens.dr7t` — per-cell mode `base` (24,32,4) | 1,297 | **VIDEO-SPECIFIC** | none — mode over *this* clip |
| `state/tokens.dr7t` — framing/header | ~22 | GENERIC | already negligible |
| `state/renderer.sec` — TR1 lotto mask-mods | 3,341 | **AMBIGUOUS-ROUTED** | order-0 entropy 3,305 B on 256 symbols ⇒ already incompressible. `regenerate_bank_and_apply_mask_mods` implies bank-from-seed (GENERIC) + mask-mods (video-derived). **Field-level split is OWED**; I did not open it and will not claim it. |
| `state/pose_warp.stp` — `tp_member` f16 pose (600,6) | 6,365 | **VIDEO-SPECIFIC** | none |
| `state/pose_warp.stp` — `ab_member` f16 exposure (600,2) | 1,838 | **VIDEO-SPECIFIC** | none |
| `state/pose_warp.stp` — `st_coded` s_t grid indices | 189 | **VIDEO-SPECIFIC** (indices) | none |
| `state/pose_warp.stp` — `sel_coded` selector bits | 79 | **VIDEO-SPECIFIC** (assignment) | colex → 71 B (**−8 B**) |
| `state/pose_warp.stp` — `beta_coded` magnitude idx | 151 | **VIDEO-SPECIFIC** (assignment) | order-0 arith → ~124 B (**−27 B**) |
| `state/selector.sec` | 314 | **VIDEO-SPECIFIC** | brotli instead of deflate (**−37 B**) |
| `state/pose_stub.sec` — `{"inert":true,…,"values":[]}` | 83 | **100% GENERIC** — a constant string, zero clip content | delete; declare in `inflate.py` (**−83 B**) |
| `manifest.json` — 6 × `sha256` integrity fields | ~250 | **GENERIC** — hashes *of the shipped bytes*, recomputable at decode | migrate (**custody tradeoff flagged**) |
| `manifest.json` — `tr1_metadata`, `schema`, `base`, `frame0_policy`, `pose_carrier`, 3 bool flags | ~350 | **GENERIC** — pure metadata | migrate |
| `manifest.json` — `st_grid` (11 floats), `rs_beta_mags` (13 floats) | ~70 | **GENERIC — quantization GEOMETRY, not assignment** | migrate |
| `manifest.json` — `beta_idx_counts`, `selector_num_two` | ~35 | **REDUNDANT** — derivable from the coded sections | delete |
| `manifest.json` — `pose_dim0_offset` = 32.125 | ~12 | **VIDEO-SPECIFIC** (fitted to this clip) | keep — 27 B minimal manifest |

**Measured migratable total: 743 B** (manifest 1,450 → 27 raw; pose_stub 83 → 0), realized in the §3
stack. **0.068% of the gap.**

### The Laguerre / tropical / Morse–Smale test, answered directly

> *Do any shipped bytes encode cell GEOMETRY rather than cell ASSIGNMENT?*

**Measured answer: two fields did, ~70 B, and they migrate. Everything else already ships
assignment-only.** No SegNet head weights, prototype colors, per-class constants, or decision
thresholds appear in any member — consistent with the runner's own contract (`NO scorers, NO GT
masks, NO shipped mask`). The frozen-head Laguerre/power-cell geometry is **already not paying rate
on this vehicle**; the only geometry-shaped bytes were the two quantization grids (`st_grid`,
`rs_beta_mags`), and the §3 stack removes them. This is an honest **negative on a real hypothesis**:
the structural rate cut the identification predicts has, on v4d, already been taken.

**What that leaves as genuinely video-specific: the ASSIGNMENT.** 96.2% of the archive is exactly
that — which cell each location lands in. So the compaction ladder *is* the whole game here, and
§1–§3 are its measurement.

---

## 1. PHASE A — the subset-index ladder, raced on the REAL selector payload

Live selector: **k = 224 of n = 600**, shipped as brotli(packbits) = **79 B**.

| rung | bytes | note |
|---|---:|---|
| **colex rank** (combinatorial number system) | **71** | **the winner**; `log2 C(600,224)/8` = 70.878 B floor |
| golomb-rice gaps | 75 | |
| bitmap packed (raw) | 75 | |
| bitmap + brotli q11 | 79 | **what we ship today** |
| bitmap + deflate | 86 | |
| bitmap + raw-LZMA1 | 88 | |
| elias-fano | 93 | loses badly at k/n = 0.37 — EF is for *sparse* sets |
| iid order-0 entropy reference | 71.49 | |

**`structure_gain_vs_colex` = 0.998** ⇒ the selector positions are **exchangeable**; there is no
clustering to exploit, so `log2 C(n,k)` genuinely IS the floor here and colex attains it.
**Saving: 8 B.**

**Contrast — and the reason the racer reports this ratio.** On `ddm_bp2`'s blind-set indices the same
question has the opposite answer: its landed `reports/ddm_bp2/index_cost.json` measured
`structure_gain_vs_comb` up to **1.62** — a brotli'd bitmap *beating* `log2 C(n,k)` by 62%, because
those positions are spatially clustered. **`log2 C(n,k)` is a floor only under exchangeability.**
An arm that assumes it is always the floor will over-pay on clustered sets; an arm that assumes
prior-coding always wins will over-pay on exchangeable ones. **Measure, don't assume** — that is what
`race_subset_index` is for.

Second index-shaped payload: `beta_coded`, 600 uint8 over 13 symbols, counts
`[5,5,1,10,15,420,66,52,13,1,7,1,4]`. Shipped brotli = **151 B**; order-0 entropy = **123.9 B**
(**−27 B**), and the histogram is *already in the manifest*, so a static arithmetic coder costs
nothing extra to specify.

**PHASE A INDEX TOTAL: ~35 B = 0.003% of the gap.** MEASURED. The operator's *"these are all solved
problems"* is confirmed — and the more useful half of the finding is that **on this vehicle they are
worth almost nothing**, so no further arm should spend time there.

### Where the index IS free entirely (Phase B, brief because the axis is small)

An index the receiver **derives** costs zero. Two live instances, both already true:
- **The blind set is a function of the resize kernel.** `ddm_ll1` measured `antialias=False`, strides
  2.2760/2.2734 > 2 ⇒ disjoint 2×2 windows, 22.70% blind to both scorers; `blind_mask()` /
  `window_geometry()` are built. Generic ⇒ **zero counted bytes**, and `ddm_bp2` correctly paid **0**
  for its index (its 401,285 B were *signs*, not positions).
- **`ddm_dc1`'s gauge degeneracy**: a move along an already-shipped coordinate needs no index at all.

**The derived-index rung is the migration audit one level down** — same test, same verdict rule.

---

## 2. PHASE A — the REPRESENTATION ladder on the token stream (96.2% of bytes)

### 2a. Generic recompression is exhausted

| member | raw | deflate | brotli q11 | raw-LZMA1 |
|---|---:|---:|---:|---:|
| `tokens.dr7t` | 346,478 | 346,594 | **346,483** | 351,181 |
| `renderer.sec` | 3,341 | 3,352 | 3,345 | 3,362 |
| `pose_warp.stp` | 8,654 | 8,665 | 8,658 | 8,735 |
| `selector.sec` | 535 | 320 | **277** | 355 |
| `manifest.json` | 1,450 | 759 | **671** | 800 |
| `pose_stub.sec` | 83 | 83 | **76** | 84 |

**Tokens: brotli is +5 B WORSE than stored.** That is the signature of a stream already at its
coder's entropy — the r7 adaptive arithmetic coder did its job. **Coder-swapping the tokens is dead.**
The only live generic wins are the three small members (~132 B), folded into §3.

### 2b. The model, not the coder — oracle conditional entropies on the residual

`delta` = `(code − per-cell-mode) mod 16`, 1,843,200 symbols. Shipped r7 = **1.5038 bits/sym**.

| context | oracle B | +MDL B | ctxs | bits/sym |
|---|---:|---:|---:|---:|
| H0 (iid) | 504,291 | 504,310 | 1 | 2.1888 |
| \| channel | 503,539 | 503,610 | 4 | 2.1855 |
| \| left | 452,101 | 452,321 | 16 | 1.9622 |
| \| prev-pair | 404,313 | 404,534 | 16 | 1.7548 |
| \| row | 398,008 | 398,373 | 24 | 1.7275 |
| \| row+chan | 394,880 | 396,161 | 96 | 1.7139 |
| **\| CELL-ID (r,c,k)** | **338,440** | 365,019 | 3,072 | **1.4689** |
| \| cell + prev | **309,468** | 405,318 | 20,412 | 1.3432 |

`+MDL` is the Rissanen parameter cost `(L−1)/2 · log2 n_c` per used context — the honest price of
*specifying* the model. **Cell-identity is the context that matters**, and its oracle (338,440 B) sits
**below** the shipped 346,478 B. `cell+prev` reaches 309,468 B but its model cost (+95,850 B) makes
it unaffordable statically; an adaptive coder pays less, so **~310–338 KB is the plausible realizable
band** and the remaining ceiling is **model-cost-bound, not coder-bound**.

### 2c. LAYOUT — struct-of-arrays vs array-of-structs (the rung never raced on our stream)

Best of brotli-q11 / raw-LZMA1, on the residual:

| layout | byte lane | **nibble lane** | 4 bit-planes |
|---|---:|---:|---:|
| AoS `(P,R,C,K)` native | 398,826 | 395,026 | 479,761 |
| SoA chan-major `(K,P,R,C)` | 408,793 | 404,547 | 486,922 |
| SoA row-major `(R,P,C,K)` | 394,114 | 387,335 | 457,647 |
| **SoA cell-major `(R,C,K,P)`** | 346,919 | **339,970** | 410,349 |
| SoA `(K,R,C,P)` | 347,238 | 340,326 | 410,723 |

Three MEASURED results:
- **Cell-major + nibble packing = 339,970 B**, i.e. **within 1,530 B of the cell-id oracle** (338,440).
  Grouping all 600 pairs' values for the same cell contiguously lets a *stock* coder learn the
  per-cell distribution adaptively, paying far less than the full Rissanen term. **The layout change
  realizes essentially the entire cell-conditional model gain with no new coder.**
- **Nibble packing beats byte lanes at every layout** (−1.0% to −2.0%), consistently.
- **Bit-plane transposition LOSES here, badly (+20%).** It is the rung "almost never tried" — we
  tried it, and on a 4-bit alphabet with strong per-cell value correlation it destroys the symbol
  correlation the coder was exploiting. Honest negative; the racer keeps it so the next arm need not
  re-discover it. (Note: `pose_warp`'s f16 members already ship **byte-plane** transposed — `kl1` —
  so that rung *is* live on our vehicle where it pays.)

---

## 3. BYTE-CLOSED RESULT (scorer-free, exact)

`IX1SOA02` frame = header + SoA-cell-major nibble-packed residual + coded `base`, self-describing,
codec chosen per-block by measured size. **Decode proved bit-identical to the original lattice from
the frame alone.**

| stack | archive B | Δ B | ΔS | % of gap |
|---|---:|---:|---:|---:|
| baseline (rebuild parity = original) | 360,309 | 0 | 0 | 0 |
| A. IX1SOA02 tokens | 355,125 | **−5,184** | **−0.0034518** | −0.475% |
| A + selector brotli | 355,088 | −5,221 | −0.0034764 | −0.479% |
| A + selector + manifest migrated | 354,248 | −6,061 | −0.0040357 | −0.556% |
| **A + selector + manifest + pose_stub migrated** | **354,165** | **−6,144** | **−0.0040910** | **−0.563%** |

**S: 0.8983775 → 0.8942865.**

**Why this is an exact ΔS with zero scorer time.** The re-encode is LOSSLESS and the decode was
proved bit-identical, so d_seg and d_pose are **invariant by construction**; the only term that can
move is `25·bytes/37,545,489`, computed by `stat()` on a rebuilt archive whose framing parity was
verified. Not an estimate.

**What is OWED before this is promotable:** the `IX1SOA02` decoder must land in the real
`inflate_runner_v4d.py` (a transpose + unpack — generic, rule-118 free, no LOC cost), and the
manifest/pose_stub migrations must land as `inflate.py` constants with the receiver-consumption
bijection (#417) updated. One confirmation exact eval closes it. **Pointer UNMOVED until then.**

---

## 4. ROUND-1 ADVERSARIAL REVIEW OF MY OWN RESULT

**CRITICAL, self-caught, headline-changing.** My first byte-close reported **−6,486 B**. It was a
**fake**. `factor_mode_delta` returns `(base, delta)` where `base` is the **per-cell mode over all
600 pairs** — 3,072 video-derived symbols computed from the full lattice, **not** derivable from the
residual and **not** derivable at decode. My v1 frame shipped only the residual and then
"verified" the roundtrip using a `base` computed locally from the original codes — **state the
decoder does not have.** That is the borrowed-state fake in its purest form, and it was inside my own
headline number.

Fix: `IX1SOA02` ships `base` (1,536 B raw nibbles → **1,297 B** brotli), and the roundtrip is now
proved **from the frame alone**. **True saving −5,184 B, not −6,486 B.** The check that caught it was
mechanical: *what does the decoder actually have?*

Three more attacks, run:
- **Unit/key assumptions.** `delta_s_rate_from_bytes` re-derived from `upstream/evaluate.py:63`
  (`25·bytes/37,545,489`) and unit-tested against the denominator itself, not recalled.
- **Class vs instance.** The bug class is "a codec race that compares a partial payload against a
  complete one." Guarded structurally: the racer's frames are self-describing and the tests assert
  round-trip *from the payload alone*.
- **Would the test pass if the code were broken?** `test_race_layouts_finds_the_stationary_layout`
  constructs an array where cell-major MUST win and asserts AoS loses — it fails if the transpose is
  a no-op. `test_pack_bitplanes_is_a_transposition_not_a_copy` reconstructs the input from the planes.

**Remaining incomplete coverage, named:** (a) `renderer.sec`'s 3,341 B were **not** field-split — the
bank-from-seed vs mask-mods boundary is OWED and could be a structural cut larger than everything
here; (b) the `cell+prev` adaptive realization was not built, so the 310–338 KB band is a bound not a
measurement; (c) the manifest sha256 migration trades our own custody for 250 B — flagged, not
decided.

---

## 5. SCOPE — what this arm deliberately did NOT do

**No ranker was built.** This ladder measures **BYTES for LOSSLESS re-encodings**. For lossless
compaction, bytes *are* the currency and no sensitivity model is needed or used.

**No metric was chosen, because a lossless re-encode has no distortion.** There is no quantizer, no
codebook, and no centroid anywhere in this arm — so the Bregman-centroid correctness question
(right-centroid = arithmetic mean in the **expectation** coordinate η, not in the natural parameter θ)
**does not bite here**. It bites the moment a rung becomes lossy. Named, in-scope observation for
whoever takes that up: **the token lattice IS a quantizer** (int4, 16 levels) and `st_grid` /
`rs_beta_mags` ARE its grids — whether their levels were placed by a Euclidean criterion or a
Fisher/Bregman one is **unaudited and is a live question**, since a wrong centroid shows up as bytes
via a larger residual.

### 5b. Quantize/rasterize — pricing width against what SURVIVES the crossing

The operator's framing (*"we keep brushing up against dithering and anti-aliasing"*) names the rung
directly below this arm: **a field carrying sub-quantum detail the lattice discards is paying rate
for nothing** — the same "zero bits for what the score cannot see" rule as the derived-index rung.
Two honest statements about where that bites on the live vehicle, and one measurement:

- **It does not bite the tokens.** They are already an int4 lattice — 4 bits is the width, there is
  no sub-quantum mantissa to discard. The token win is layout (§2c), not width.
- **It bites the f16 members**, `tp_member` (6,365 B) + `ab_member` (1,838 B) = **8,203 B = 2.3% of
  the archive** — the second-largest addressable block after the tokens. MEASURED: both are *already*
  byte-plane transposed (`KL1PWF01`) and coded **below** raw f16 width — 6,357 B vs 7,200 B (88%) and
  1,830 B vs 2,400 B (76%) — so the classical AA-adjacent rung (plane transposition) **is already
  taken here**. What is NOT taken is the **mantissa-width solve**: how many f16 mantissa bits survive
  `warp → composite-R → uint8 → argmax / PoseNet`. That question is **LOSSY and therefore out of this
  module by construction**, and it must be priced through the exact adjoint, not guessed.
- **The right coordinates for that solve are ≤10-dimensional per site**, not pixels: pose is
  `rank(J) ≤ 6` (`ddm_pb3`, measured) and the SegNet head is affine rank-4. A width allocation ranked
  in pixel space is ranked in the wrong basis — the same wrong-coordinates failure that killed
  `ddm_bp2`, one level down.
- **Route, do not rebuild:** #391 exact composite-R adjoint · #220 AA coverage-integrated render ·
  #283 AA-SDF rasterizer · #149 sub-pixel placement pre-D · #580 ker(A) 80.67% · #401 blind fill ·
  `ddm_ll1` (whose window solve already *is* the exact ordered-dither solve on the measured disjoint
  2×2 cell, 29× faster and 2.5× better than the exhaustive search).

**Routing, not rebuilding** — for anything score-ranked: #700 oracle facade · `ddm_at1` ·
`ddm_g3` (hard-pair registry + subset→full validity `r`) · `ddm_g4` · #141 · #391 · #583 ·
`ms3`/`ms4` (margin-Fisher; bind to `policy_bindings.optimal_metric`) · #504/#550 Bregman.
**#611 remains PENDING** and this arm did not touch it.

**A prefix is not a sample** (`ddm_bp2`, measured same day): every number here is n600 or full-payload.

---

## 6. THE APPARATUS (Phase C — so this is not re-derived)

`src/tac/optimization/ddm_ix1_representation_ladder.py` + 29 passing tests.

- `race_subset_index(positions, n)` → measured bytes for **colex / Elias-Fano / Golomb-Rice gaps /
  bitmap × {deflate, brotli, raw-LZMA1}**, plus `colex_floor_bytes`, the order-0 reference, and
  **`structure_gain_vs_colex`** — the ratio that tells you whether your positions are exchangeable.
- `colex_rank` / `colex_unrank` / `colex_encode` / `colex_decode` — exact combinatorial number
  system, exhaustively verified bijective (our L31 / PR101 precedent, now a callable primitive).
- `elias_fano_encode/decode`, `golomb_rice_gaps_encode`, `pack_nibble_lane`, `pack_bitplanes`.
- `race_layouts(array)` → axis-permutation × {byte lane, nibble lane, bit-planes} × generic coders.
- `race_generic(payload)` → includes a `stored` rung so "already at entropy" is visible.
- `delta_s_rate_from_bytes`, `gap_fraction_of_bytes(..., total_gap=...)` — **never hardcodes a
  floor**; the caller supplies it from the canonical equation.

The module's docstring carries the NON-GOALS and the routing table so the next caller cannot mistake
it for a ranker.

---

## 7. NEXT-IF-RESUMED (ranked by measured addressable bytes)

1. **Land the `IX1SOA02` decoder in `inflate_runner_v4d.py` and take the −6,144 B.** Highest
   value/effort ratio on this list: measured, byte-closed, bit-identity proved, decoder is a
   transpose + unpack. Frame builder is at `scratchpad/tokens_ix1_v2.bin` (regenerate from §3).
2. **Field-split `renderer.sec` (3,341 B).** The only member I did NOT open. If
   `regenerate_bank_and_apply_mask_mods` means bank-from-seed, part of it is GENERIC and free —
   a structural cut, which dominates any coding gain. **Potentially larger than everything above.**
3. **Build the adaptive `cell+prev` residual coder.** Oracle band 310–338 KB vs 341,294 B shipped
   ⇒ up to a further **−31 KB = −2.8% of gap**, model-cost-bound. Use `race_layouts` cell-major as
   the starting layout.
4. **Audit `st_grid` / `rs_beta_mags` level placement in the Fisher/Bregman coordinate**, not
   Euclidean. Wrong centroid ⇒ larger residual ⇒ bytes. Route through `ms3`/#504; do not hand-roll.
5. **Mantissa-width solve on the 8,203 B of f16 fields** (§5b) — plane transposition is already
   taken; the open question is how many mantissa bits survive the crossing. LOSSY: price through
   #391's exact adjoint in the rank≤6 / rank-4 coordinates, never in pixel space.
6. **Do NOT spend another arm on the index.** ~35 B, measured, twice. It is solved and it is small.

---

*STORES CONSULTED:* `ddm_pb3_parametric_blind_set_20260802.md` (the directive's origin; its §5 is
still OWED and this arm does not close it), `reports/ddm_bp2/index_cost.json` (the clustered-set
`structure_gain_vs_comb` ≤ 1.62 contrast), `ddm_ll1_window_solve` (derived blind mask),
`inflate_runner_v4d.py` + `experiments/ddm_r7_token_coder.py` (the vehicle and its coder, read at
source — the `base` catch came from reading `factor_mode_delta`, not from its docstring),
CLAUDE.md L20–L32 (intake intelligence, raced not adopted),
`tac.canonical_equations.gap_decomposition_against_floor_20260802` (the denominator).
