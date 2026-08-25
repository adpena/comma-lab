# Lever-D reactivation on the TEMPORAL FLICKER axis — build spec + honest accounting (DESIGN, #279)

**Date:** 2026-07-03 · **Status:** DESIGN-ONLY (no code edits) · **Task:** #279 (Lever-D reactivation),
sisters #72 (MCR codec, BUILT), orphan #226 (render-consumed flip-repair), #202 (byte-close), #257
(store-nothing pose carrier), #274 (down-weight lever, BUILT `6e355170d`), #205 (live witness, SACRED).
**Pointer 0.19110 UNMOVED. MEANS.**

> **NO-FAKE banner.** Every net-S number in this memo is a **SPEC / projection**, NOT a byte-closed
> `upstream/evaluate.py` exact row. The headline `−0.35 S` is the OPTIMISTIC corner (≈100% recovery @
> entropy-optimal 250 KB); the realistic expected corner is **much more modest (≈−0.05 S)** and the
> pessimistic corner is **net-NEGATIVE (worse)**. The lever moves ONLY d_seg; it does NOT touch pose or
> the INR base rate. Even the optimistic corner leaves the witness S ≈ 0.40, still **~2× above the
> 0.19110 pointer** — this makes the witness more competitive, it does NOT move the pointer. A byte-closed
> n600 A/B is the only thing that promotes any of this from spec to fact.

---

## 0. What is already built (reactivation, not new build)

- **`src/tac/boundary_math/margin_conditional_residual.py` (#72)** — the honest per-pair spatial coder.
  Reuse verbatim: `WATERLINE_BYTES_PER_FLIP = 1.27` B/flip, `SEG_VALUE_PER_FLIP = 8.48e-7` S/flip,
  `_N_SCORED_TOTAL = 117,964,800` (600×512×384), `measure_code_cost` (conditional position
  `log2 C(|B|,K)` + `H(target|margin-bin)` class bits), `waterfill_select` (admit net>0 AND
  marginal < waterline), `encode_residual`/`decode_residual` (reversible `MCR1` sidecar).
- **`tools/levelset_byte_close_and_eval.py` (#202)** — the `LVLS1` archive grammar. 6 blocks today:
  `magic | manifest | base_brotli | code_brotli | pose_sidecar [| lane_band (COUNTED) | pose_carrier
  (COUNTED)]`. Optional trailing blocks are manifest-flag-gated → **default-off = byte-identical**.
  Lane-band (#224) and pose-carrier (#205) are the two live precedents for adding a **COUNTED**
  video-derived block. The seg verdict is `SegNet(frame1).argmax != L*`; `frame1` is the witness render
  in `_render_pair` (`fk==1`), so a decode-time RGB overlay on `frame1` is exactly where an induce lands.
- **`#257` store-nothing pose carrier** — composes as the 6th block; Lever-D adds a **7th** block,
  orthogonal (pose carrier touches `frame0` which is SEG-free; Lever-D touches `frame1` which is d_seg).

**Reactivation delta over the built MCR:** (1) TEMPORAL coding across the 600 pairs (the existing coder is
per-pair independent); (2) SEGMENT-level rather than per-pixel (SegNet sees regions); (3) the **induce**
mechanism (a decode-time RGB operator + compress-time detector-informed admission — the built MCR stores
`(pos, class)` but never specified how `frame1` RGB is modified to realize the flip at decode). The
waterfill/cost KKT math is REUSED unchanged.

**The finding (a949ff63, n600):** #205 d_seg = **0.004964 @ep225**, already slightly BELOW the single-frame
popout floor 0.00520 → the residual essentially IS the temporal flicker (44% of spikes = lane; luma
temporal-delta 8.4× at spikes). Flicker is spatially COHERENT (3×3 mode filter drops popouts only 3.9%;
only 6.1% lone pixels) → per-pixel entropy OVERSTATES store-cost; regional-context entropy ≈ 250 KB.
Induce is cheap: 67% of popouts flip with a **<1-logit nudge**.

---

## 1. WHAT to store — the segment-level temporal flip-residual

**Object.** For each spatially-coherent boundary SEGMENT `s` (a node of the per-frame region-adjacency /
Morse-Smale boundary graph, Lens-1), store the set of frames where the witness `frame1` argmax disagrees
with `L*` on `s`, plus the target class. NOT per-pixel; the segment is the unit because SegNet reads
regions (the linear "store-the-flip-pixels" sidecars NO-GO'd ×3).

**Three correlation axes exploited (all real, all measured — NO closed-form-CDF assumption):**
1. **Spatial (intra-frame):** flips concentrate in the decoder-KNOWN low-margin band
   `B = {p : m(p) < τ}` (the witness regenerates its own margin field for FREE from its render →
   `boundary_set_from_margin`, ZERO stored bytes). Position cost is the conditional set-index
   `log2 C(|B|, K)` per the built coder, but lifted from pixel-index to **segment-boundary contour**
   (store the segment's low-order boundary polyline once, per the `contour_codec.py` grammar; the
   segment's pixels regenerate FREE from the contour + the decoder's margin band).
2. **Temporal (inter-frame):** a segment that flickers does so in a temporally-correlated on/off pattern.
   Store the per-segment flip-event train across the 600 frames as **temporal RLE + order-1 arithmetic
   code** (`charm_range_coder.py` is in-tree). The 250 KB estimate already used temporal context
   (positions 673→206 KB, class 1.85→0.59 bits/flip); the realized coder must actually implement it.
3. **Class:** `H(target_class | margin-bin)` conditional entropy (built `class_bits_conditional`), further
   conditioned on the segment's dominant class-pair (a boundary segment lives between exactly two classes;
   the flip target is almost always "the other side" → class cost collapses toward ~0.6 bit/flip).

**Refined 250 KB estimate (segment level).** Cross-check against the waterline:
`d_seg_now 0.004964 × 117,964,800 ≈ 585,600` total flip-px across 600 frames. The memory's 250 KB entropy
estimate over the net-recoverable subset (~350 K flip-px) implies **≈0.57 B/flip** — ~2.2× below the
1.27 B/flip waterline. That is achievable ONLY IF the segment-contour + temporal-RLE + class-context coder
realizes near-entropy. **Honest realized band: [250, 400] KB** (a real range coder lands 5–60% above a
context-model entropy estimate depending on model fidelity). This band is the #1 unknown the subset proof
must pin.

---

## 2. HOW to induce — detector-informed, decode-time, scorer-free

The seg verdict is `SegNet(frame1).argmax`. To realize a stored flip we must make SegNet argmax the target
class at those pixels — **without running SegNet at inflate** (strict-scorer rule: no scorer weights at
inflate; ~73 MB would destroy rate anyway).

**The split (inverse-steganalysis / Fridrich detector-informed embedding):**
- **COMPRESS time (unlimited compute, scorers ALLOWED):** for each candidate segment, apply the EXACT
  deterministic decode-time nudge operator (below) to `frame1`, run the frozen SegNet, and **verify** the
  argmax flips to the target on that segment. Admit a segment ONLY if its decode-time nudge actually flips
  it AND net value > 0 (collateral priced in). By construction, **every admitted segment recovers ≈100%**
  at decode; the uncertainty is the ADMISSION fraction, not the induce.
- **DECODE time (inflate, NO scorer):** apply the stored/regenerated nudge as a pure RGB overlay on
  `frame1`. Generic algorithm → **FREE (rule 118)**.

**The nudge operator (FREE, generic).** At a segment with target class `c`: blend the segment's `frame1`
RGB toward the class-`c` palette prototype (the witness already carries `--palette-anchor`; the 5 class
prototypes are decoder-known) by a bounded step along the SegNet-sensitive direction, feathered over the
segment interior, UNIWARD-weighted to the segment's high-texture pixels (undetectable-error discipline;
`<1-logit` nudge → small, texture-hidden). Two admissible variants, pick by the subset proof:
- **Variant A — class-only (cheapest):** store only `(segment, target class)`; the nudge is the fixed
  deterministic "blend toward prototype `c` by step σ." Store cost = §1. **The admission waterfill uses the
  EXACT variant-A nudge at compress time**, so admitted = "flips under the fixed decode nudge." This is the
  default; the induce is 0 extra stored bytes beyond `(segment, class)`.
- **Variant B — coded RGB delta:** if variant A's admission fraction is too low, store a small quantized
  per-segment delta (mean chroma/luma shift, ~2–4 B/segment). Higher recovery, higher rate. Priced by the
  same waterfill; only used if A is dominated on the subset proof.

**Compliance:** the flip positions/classes are VIDEO-DERIVED (from the actual GT SegNet argmax on the
actual video) → **COUNTED in archive.zip**, measured by the archive stat. The nudge operator + palette +
contour rasterizer are GENERIC → FREE in inflate.py. No hide-data-in-code; no scorer at inflate.

---

## 3. BYTE-CLOSE path — the 7th COUNTED block + inflate consumer

Mirror the lane-band / pose-carrier precedent exactly (`build_levelset_blob` / `_io_pack` /
`parse_pose_carrier` / `_render_pair`).

**Archive grammar addition (`tools/levelset_byte_close_and_eval.py`):**
- New magic `MCRT1\x00` (margin-conditional residual, temporal). New **7th** optional block appended by
  `_io_pack` AFTER `pose_carrier`, gated by a new manifest flag `seg_flip_residual` → **absent =
  byte-identical to today's grammar** (default-off guarantee, same as lane/pose-carrier).
- Block payload = the §1 coder output: `header(n_pairs, τ, grid, n_segments) | segment-contour blob
  (brotli) | temporal-RLE+AR flip-event blob | class blob | [variant-B delta blob]`. Manifest carries the
  scalar cfg (τ, nudge step σ, variant, palette ref) so inflate reproduces the nudge decode-consistently.
- `build_levelset_blob` gets `seg_flip_bytes`/`seg_flip_manifest` params; `breakdown` gets
  `seg_flip_counted_bytes` (the measured COUNTED add).

**Inflate-side consumer (`_render_pair`, `fk==1` path):** after the witness `frame1` render `rgb` (and any
lane composite), if `seg_flip_residual` is active and pair `pi` has admitted segments: parse the pair's
segment set, regenerate each segment's pixels FREE (contour + margin band), and apply the FREE nudge
operator toward the stored target class. This is a pure RGB mutation of `frame1` BEFORE `_R(...)`;
`frame0` (pose carrier) is untouched. A `parse_seg_flip_residual(blob)` reader mirrors
`parse_pose_carrier`. Everything is op-for-op reproducible (CPU-locked bit-exact per the
MLX-GPU-not-bit-identical-crossprocess finding — the codec is numpy/CPU).

**Composition:** pose carrier (block 6, `frame0`) ⊥ Lever-D (block 7, `frame1`) ⊥ lane-band (block 5,
`frame1` render, runs BEFORE the flip overlay). Order in `_render_pair`: witness render → lane composite →
**flip-residual overlay** → `_R`. Down-weight lever (#274) is a TRAINING-time loss reweight on the frozen
checkpoint's producer, orthogonal and additive (it shrinks the incompressible remainder; Lever-D stores
the compressible-regional part).

---

## 4. HONEST rate / recovery accounting — the band

**Addressable ceiling (sharper than the memory's 0.52).** The recoverable d_seg is bounded by the CURRENT
witness d_seg, not the popout floor: `d_seg_now 0.004964 × 100 = 0.4964 S` is the max seg you could recover
by driving d_seg→0 (unreachable — irreducible remainder > 0). The memory's `−0.52 S` uses the popout-floor
magnitude 0.00520 and slightly **overcounts**; the honest ceiling is **−0.4964 S**.

Let `r` = net recovery fraction of d_seg_now (net of receptive-field collateral, #51/#55), realized rate
`R_KB` → rate S = `25 × 1000·R_KB / 37,545,489`. Net ΔS(witness) = `−r × 0.4964 + rate_S`:

| corner | r (net recovery) | realized rate | rate S | seg ΔS | **net ΔS (witness)** |
|---|---|---|---|---|---|
| memory-spec (as stated) | ~1.0 of popout 0.52 | 250 KB | +0.170 | −0.520 | **−0.35** (optimistic anchor) |
| **optimistic** | 0.70 | 250 KB | +0.170 | −0.347 | **−0.177** |
| **expected** | 0.50 | 300 KB | +0.200 | −0.248 | **−0.048** |
| **pessimistic** | 0.30 | 400 KB | +0.266 | −0.149 | **+0.117 (WORSE)** |

**Break-even recovery** (r s.t. net=0): 34% @250 KB · 40% @300 KB · 54% @400 KB. The 67%-flip-cheap finding
is a GROSS number; **net-of-collateral** recovery is the unknown that decides whether we clear break-even.
The lever is real but MARGINAL: plausibly a small win (−0.05 to −0.18 S), plausibly net-negative if the
realized coder misses entropy and/or collateral eats recovery.

**Still above the pointer either way.** Witness implied S ≈ 0.75 (d_seg 0.4964 + pose ~0.176 + base rate;
note pose is OPEN/UNMEASURED — warp still catastrophic). The optimistic −0.35 corner → witness S ≈ 0.40,
**~2× above the 0.19110 pointer.** Lever-D touches ONLY d_seg; it cannot get the witness to the pointer.
This is the witness getting more competitive on ITS d_seg term, a long-bet increment — NOT a pointer move.

---

## 5. MINIMAL first build + A/B plan (await GO)

**Stage 0 — subset proof (light; gates everything; the smallest thing that produces a REAL byte-closed
recovery number).** On a FROZEN #205 checkpoint (READ-ONLY snapshot; #205 untouched), on a 32-pair subset:
1. Compute the GT-vs-witness `frame1` flip set + the decoder's free margin band (numpy/CPU).
2. Build the segment graph + run the §1 temporal+segment+class coder → **realized coded bytes** (the real
   B/flip, pinning the [250,400] KB band's slope on the subset).
3. Apply the EXACT variant-A decode-time nudge → run frozen SegNet (compress-time, allowed) → **measure
   realized net d_seg recovery** on the subset (this is the `r` unknown, measured for real).
4. Compute subset net ΔS = −(realized seg drop) + (realized rate). **Default-OFF scaffold**: the whole path
   behind `--seg-flip-residual` (byte-identical when off, proven by an A-off==B-off-1.0 bit-identity check
   like the down-weight lever's).

**GATE:** proceed to the full n600 build ONLY IF the subset shows realized B/flip < ~1.0 AND net recovery
in the net-positive corner (subset net ΔS < 0). If the subset lands in the pessimistic corner → STOP, log
the honest negative (implementation-level, per Catalog #307), keep the down-weight lever as the seg play.

**Stage 1 — full A/B (byte-closed n600, exact).** store+induce ON vs OFF, both through `#202`
`levelset_byte_close_and_eval.py`: measure **Δd_seg** (SegNet on the decoded `frame1`, the exact verdict)
vs **Δrate** (the `archive.zip` stat). Report the exact net ΔS row. This promotes the spec to a fact (or
kills it). Runs on the frozen checkpoint; any n600 SegNet/decode pass routes through the **governed
launcher** (P0 memory governor) — never raw. **No heavy/paid compute fires without operator GO.**

---

## 6. Risks / NO-FAKE / means

- **Realized-rate risk (primary):** 250 KB is an ENTROPY estimate; the real coder may land 300–400 KB →
  break-even climbs to 40–54%. Stage-0 measures it.
- **Collateral risk:** frame1 nudges have receptive-field spill (#51/#55); the waterfill prices net value,
  but net recovery `r` is unproven. Stage-0 measures it with the real scorer.
- **Induce-fidelity is NOT a risk by construction** (admission = "flips under the exact decode nudge"), but
  the ADMISSION fraction is the `r` unknown.
- **Compliance is explicit:** residual = COUNTED video-derived bytes in archive.zip (measured); nudge +
  contour rasterizer + palette = FREE generic algorithm in inflate.py; NO scorer at inflate.
- **Means, not ends:** this is a d_seg lever on the witness capstone, itself far from the pointer. Pointer
  0.19110 UNMOVED until a byte-closed exact row beats it. #205 SACRED, untouched.

**Wire-in (design memo, research-only):** sensitivity-map = the per-segment net_value/byte IS a
sensitivity contribution (fold into `tac.sensitivity_map` at build time); Pareto = the waterfill IS the
rate/d_seg KKT admission; bit-allocator = the temporal coder is a new allocator primitive; cathedral =
the 7th COUNTED block is archive-deployable; continual-learning = the Stage-0 realized B/flip + `r` are the
empirical anchors to register; probe-disambiguator = Stage-0 IS the disambiguator between the memory-spec
optimistic corner and the pessimistic corner. All N/A until Stage-0 lands real numbers.

---

## Observability surface

*(OBSERVABILITY-ADDENDUM 2026-08-25 — APPEND-ONLY per Catalog #110/#113. This
section is an INDEX into this memo's own content per Catalog #305's 6 facets;
it adds no new claim. Facets with no counterpart in this memo say so plainly.)*

1. **Per-layer inspection** — §1 "WHAT to store — the segment-level temporal flip-residual" and §2 "HOW to induce — detector-informed, decode-time, scorer-free" separate the stored layer from the induction layer, each inspectable alone.
2. **Per-signal decomposition** — §4 "HONEST rate / recovery accounting — the band" decomposes the lever into its rate cost and its recovery, as a band rather than a point.
3. **Run-to-run diff** — §3 "BYTE-CLOSE path — the 7th COUNTED block + inflate consumer" makes the lever exactly one added archive block, so an ON build diffs against OFF by that block; the flags are `--seg-flip-residual` and `--palette-anchor`.
4. **Post-hoc query** — the byte-close path is `tools/levelset_byte_close_and_eval.py` producing `archive.zip`; the coder surfaces are `charm_range_coder.py` and `contour_codec.py`; the consumer is `inflate.py`.
5. **Cite-chain** — §0 "What is already built (reactivation, not new build)" attributes every reused surface, which is what makes this a reactivation rather than a new build.
6. **Counterfactual hooks** — §5 "MINIMAL first build + A/B plan (await GO)" is the pre-registered A/B; §6 "Risks / NO-FAKE / means" enumerates what would falsify the lever.
