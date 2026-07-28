# ddm_pp1 — PRICE THE DIRECT-PARTITION LEG (ee1 R1, $0) + register the band lemma (ee1 R3)

**Pointer honesty first: 0.1910828242 [contest-CPU] UNMOVED. This arm moved no exact score.** It is a
`[macOS-CPU advisory] NON-PROMOTABLE` measurement arm: real lossless coder bytes over the cached n600 GT
partition + a $0 band-lemma falsifier. NO scorer jobs, no render, no training (cached artifacts + real
coders only; fd1r owns the scorer slot). Every byte below is a bit-exact round-trip-proven lossless coder
length on the REAL n600 cached maps (or the closed-form adaptive length PROVEN == real coded bytes to
<0.01%), NEVER a byte-closed `evaluate.py` row.

**Bottom line (2 findings):**
1. **R1 — the direct partition is CHEAP: 173.6 KB lossless (KT context-arith) / 172.2 KB lossy-optimal.**
   The pre-registered ee1 falsifier (lossless ≥350 KB AND lossy ≥250 KB ⇒ dead) is **NOT REACHED** — the
   direct-explicit partition family is **NOT DEAD**; it lands in the 120–180 KB THIRD-ROUTE band. But the
   composed explicit route lands **~0.189** (above the 0.172 bar) because the explicit context-arith
   partition is +57 KB vs PR130's learned tokens: **direct-explicit CONVERGES with (does not beat) the
   implicit token+renderer carrier** (ee1 C10). The binding constraint stays REALIZATION (fd1/R2 slot),
   not partition coding.
2. **R3 — the band lemma is CONFIRMED + REGISTERED.** Measured coherent position cost crosses the 1.2731
   water at **ρ_c = 5.0e-4** (derived uniform crossing ρ_u = 8.6e-4; context shifts the edge down ~1.7×).
   Registered as `ddm_pp1_correction_stream_position_band_v1`.

## STORES CONSULTED (recall-first; every cited receipt verified this session)
- **CLAUDE.md + AGENTS.md** (full read): NO-FAKE supreme rule (#6 search-as-solver / #8 surrogate-not-
  exact); the 1.2731 B/flip region_merge water law; serializer + post-edit sha; `.py` review gate (never
  `REVIEW_GATE_OVERRIDE`); bulk → `/Volumes/VertigoDataTier/pact`; canonical-equations registry discipline.
- **Charter** `scratchpad/pp1_charter.md` (R1 coder race + falsifier; R3 band-lemma validate-then-register).
- **ee1 memo** `.omx/research/ddm_ee1_einstein_fresh_eyes_capstone_20260728.md` — §A.2 (sufficient statistic
  = partition + pose), §C4 (direct-partition leg UNPRICED internally; ECC anchor ~150 KB), §C6 (the band
  lemma, DERIVED), §D.1 (ECC Cityscapes 2,662 B lossless → ~250 B/frame; SegNet-argmax noisier caveat), §E
  R1/R3/R6/R8. **My R1 PRICES ee1 C4's unmeasured cell; my R3 VALIDATES+REGISTERS ee1 C6.**
- **fc1 receipts** `/Volumes/VertigoDataTier/pact/ddm_fc1_20260728/`: `stage2_coders_n600.json` (support
  LZMA **421,366 B** = 0.413 B/flip @ 0.864% density; labels **41,392 B**; both round-trip-verified);
  `entropy_n600.json` (support fraction 0.00864 = 1,019,467 flips; 1.273 water; bars 0.172→**187,727 B**,
  0.15→**154,522 B**).
- **sp1 receipt** `ddm_sp1_20260728/r1_contour_support_n600.json` (contour support 444,394 B; concession
  U-min at the lossless edge — my full-partition concession reproduces the same shape).
- **r6cal .py** `src/tac/canonical_equations/ddm_r6cal_solved_object_rate_dominance_20260728.py` — the
  canonical-equation module pattern my R3 registration follows verbatim (build/populate/evaluator/provenance).
- **PR130 anchor** (ee1 §B, pi1 code-verified): 191,052 B = tokens 116,980 B (~195 B/frame @ 0.00793 bpp)
  + int4 renderer 40,252 B + HPAC prior 20,179 B + pose 23,054 B; measured **0.172141 [contest-CUDA]**;
  native realization d_seg 2.97e-4. LESSONS-ONLY anchor (no adopted bytes/constants).
- **GT cache** `experiments/results/mlx_fleet_gt_cache/gt_n600.npz`: `lstars` (600, 384, 512) int64, 5
  classes {Road,Lane,Undrivable,Movable,MyCar}, verified. **The object priced here — never coded before.**
- **#307 machinery REUSED, not rebuilt**: `tools/measure_contour_string_flip_coding.py`
  (`AdaptiveStream`/`AdaptiveStreamDecoder` + `contour_encode_frames`/`contour_decode_frames`, bit-exact
  round-trip-tested at `src/tac/tests/test_contour_string_flip_coding.py`). My tools IMPORT its functions.

## The object (honest scope note)
`lstars` is the **600 seg-scored last-frames** (one per pair; SegNet scores only the last frame,
modules.py:108) at 384×512, 5 classes — the ee1 A.2 sufficient statistic. The charter's "1200 frames"
is the full video; the seg-relevant partition sequence is 600 maps (117,964,800 pixel-frames = the flip
denominator, matching fc1). Recon: class fractions [Road 0.232, Lane 0.006, Undrivable 0.495, Movable
0.012, MyCar 0.254]; boundary **2,436 px/frame** (matches ee1 A.3's 2–4k); temporal disagreement between
consecutive scored frames **1.246%** (highly coherent).

## R1 — THE CODER RACE (lossless, n600, bit-exact round-trip). MEASURED.

| coder family | KB | round-trip | note |
|---|---:|---|---|
| **context-arith temporal (o8 + prev-5 nbhd), KT** | **173.6** | proof ✅ | **WINNER**; Laplace α=1 = 177.8 KB |
| context-arith intra o8 | 206.9 | proof ✅ | spatial-only; +temporal saves 33 KB |
| context-arith intra o6 | 219.8 | proof ✅ | |
| context-arith intra o4 (L,U,UL,UR) | 233.7 | proof ✅ | held-out ≈ plug-in (robust) |
| bz2 raster | 338.6 | ✅ | generic |
| row-RLE → LZMA | 332.3 | ✅ | 357,757 runs |
| LZMA1-x9e raster | 410.0 | ✅ | generic incumbent |
| brotli-11 raster | 424.7 | — | generic |
| PNG-Paeth residual → LZMA | 499.3 | ✅ | prediction hurts a label map |
| per-class binary planes → LZMA | 660.5 | ✅ | Road 295 / Lane 189 / Undriv 85 / Mov 62 / MyCar 29 |

**The context-adaptive arithmetic coder wins decisively at 173.6 KB** (the ECC-class strong coder); every
generic coder is 330–660 KB. Temporal-as-CONTEXT (not predict-then-residual, ee1 C3) saves 33 KB over
intra (206.9 → 173.6).

**NO-FAKE proof the 173.6 KB is a REAL decodable coder length:** a context-adaptive arithmetic coder's
byte length equals the closed-form Dirichlet-multinomial (KT/Laplace) code length of its model to <0.01%
(one range-coder flush over ~1e8 symbols). The tool computes that closed form EXACTLY over all n600
(order-independent; pays its own model-learning cost by construction — no held-out artifact, no plug-in
optimism) AND proves the correspondence on a 6-frame subset with the in-tree #307 `AdaptiveStream`:
**coded_bytes / closed_form = 1.0000, bit-exact round-trip = True.** (The subset per-frame rate is higher
only because the adaptive model is un-converged on 6 frames; the full-n600 closed form is the amortized
authority.)

**Per-class attribution (best context, approx):** Road 62.4 / **Lane 62.3** / Undriv 20.9 / Movable 13.2
/ MyCar 5.6 KB. **The Lane class is 36% of the cost at 0.72 b/px** despite being 0.59% of pixels — the
thin-dash residual is the expensive part (as predicted).

**R8 lane-dash sub-race (ee1 R8):** the #307 contour chain coder on the Lane binary field = **219.5 KB**
(round-trip ✅), which LOSES to the context-arith Lane attribution (62.3 KB). **Falsifier met: the
lane-dash dictionary/contour shows NO byte win over context-arith** — the context coder already captures
the lane structure ~3.5× better.

### Lossy concession curve (ee1 A.1 waterfill at 1.2731 B/flip; component removal → surrounding majority)

| drop comps < k | conceded flips | conceded frac | best-coder bytes | **S_partition** |
|---:|---:|---:|---:|---:|
| 1 (lossless) | 0 | 0.000% | 173,616 | 0.11560 |
| **2 (OPTIMAL)** | **866** | **0.001%** | **172,220** | **0.11541** |
| 4 | 7,313 | 0.006% | 167,691 | 0.11783 |
| 8 | 25,322 | 0.021% | 158,861 | 0.12727 |
| 16 | 51,135 | 0.043% | 149,900 | 0.14318 |
| 32 | 87,785 | 0.074% | 141,006 | 0.16828 |
| 64 | 159,786 | 0.135% | 129,412 | 0.22161 |

`S_partition = 25·retained_bytes/37,545,489 + 100·conceded_flips/117,964,800`. The curve is **U-shaped with
the min at the near-lossless edge** (k=2): the context coder already codes small components BELOW the 1.273
water level, so every deeper concession is a net S LOSS — the SAME structure sp1 found for the flip support.
Lossy buys almost nothing (173.6 → 172.2 KB).

### FALSIFIER VERDICT (ee1, pre-registered) — NOT REACHED; THIRD ROUTE OPENS

- Death requires lossless **≥350 KB** AND lossy-at-water **≥250 KB**. Measured: lossless **173.6 KB**,
  lossy-optimal **172.2 KB** — **BOTH far below the dead thresholds.** Direct-explicit partition coding is
  **NOT DEAD** (verdict_scope: FORMULATION — this object `lstars`, these coder families).
- 173.6 KB is IN the **120–180 KB THIRD-ROUTE band** → the third route OPENS in the ee1 sense. My measured
  174 KB is close to the ECC external anchor (~150 KB); the ~24 KB excess is exactly ee1 §D.1's caveat
  (SegNet-argmax maps are noisier than curated Cityscapes GT — boundary speckle the coder must pay for).

### Composed third-route arithmetic (HONEST; measured legs)

`S = 100·d_seg_realization + √(10·d_pose) + 25·(partition + renderer + pose)/37,545,489`

| route | partition | +renderer | +pose | realization d_seg | d_pose | **composed S** | vs 0.172 bar |
|---|---:|---:|---:|---:|---:|---:|---|
| **A. explicit partition + trained renderer** | 173,616 | 40,000 | 2,000 | 3e-4 | 2.33e-5 | **0.1888** | ABOVE |
| B. explicit partition + GENERIC painter + corrections | 173,616 | +421,000 support | 2,000 | ~0 (corrected) | 2.33e-5 | **0.412** | DEAD |
| PR130 (implicit token+renderer, MEASURED) | tokens 116,980 | 40,252 | 23,054 (+prior 20,179) | 2.97e-4 | 2.33e-5 | **0.172141** | AT bar |

- **Route A** ships the explicit 173.6 KB partition + a trained renderer (realization only) + pose → **0.189**,
  ABOVE the bar. Why: the explicit context-arith partition is **+57 KB** vs PR130's learned tokens (117 KB).
  The learned prior TIGHTENS the partition-trajectory description below the explicit context-arith rate.
- **Route B** (generic 0-byte painter) is DEAD: the painter's native realization (paint-face 0.0086 = 159×)
  needs corrections, and at 0.86% base error the correction SUPPORT is ~421 KB (sp1) — blowing the budget.
  This is exactly the band lemma's teeth (R3): a base inside the band pays a huge total support.
- **Convergence (ee1 C10):** explicit partition 173.6 KB ≈ PR130's FULL partition leg (tokens 117 + renderer
  40 + prior 20 = 177 KB). **The direct-explicit and implicit token+renderer representations price the
  partition at the same ~174–177 KB — neither dominates.** Direct-explicit is NOT dead, but it does NOT beat
  the implicit carrier; the sub-bar path still needs the implicit carrier (fd1's slot) OR a learned prior on
  the partition tokens to close the +57 KB gap. **The binding constraint remains REALIZATION, not partition
  coding** — which this arm confirms is CHEAP.

## R3 — THE BAND LEMMA: VALIDATED (the $0 falsifier) + REGISTERED

**The lemma (ee1 §C6, DERIVED):** a correction stream's per-error POSITION cost is bounded above by the
uniform combinatorial rate `b_pos(ρ) = log2(N/k)/8 = log2(1/ρ)/8` B/err, which crosses the 1.2731 water at
`ρ_u = 2^(-8·1.2731) = 8.6e-4`. Context/coherence LOWERS the achievable cost below the bound. Claim:
corrections rational only in a band ~1e-3..1e-2.

**The $0 falsifier (measured):** recompute the SUPPORT (position) coding price at 9 synthetic densities
(margin-thresholded boundary-COHERENT correction fields + random-subsampled INCOHERENT reference), code
positions (packbits→LZMA + #307 contour), locate the measured crossing.

| ρ (coherent, margin-thresh) | measured B/err (best) | uniform log2(1/ρ)/8 |
|---:|---:|---:|
| 0.00022 | 1.469 | 1.515 |
| 0.00056 | 1.245 | 1.349 |
| 0.00141 | 1.005 | 1.183 |
| 0.00282 | 0.803 | 1.059 |
| 0.00562 | 0.587 | 0.934 |
| 0.01112 | 0.359 | 0.811 |
| 0.02170 | 0.200 | 0.691 |
| 0.03800 | 0.118 | 0.590 |
| 0.07006 | 0.068 | 0.479 |

**Results (lemma CONFIRMED):**
- The uniform bound is an UPPER bound on the measured coherent cost at **all 9 points** (each measured <
  uniform).
- **Measured coherent water crossing: ρ_c = 5.0e-4** (context shifts the derived ρ_u = 8.6e-4 down ~1.7×,
  exactly "context lowers it").
- The random (incoherent) reference OVERSHOOTS the uniform bound (LZMA overhead on random points: 2.69 B/err
  at ρ=1e-4 vs uniform 1.66) and crosses higher (~1.5e-3) — incoherent corrections are even less rational.
- **Cross-check:** at fc1's ρ=0.864% the coherent curve interpolates ~0.44 B/err vs fc1's MEASURED 0.413
  B/err (LZMA support) — the synthesis reproduces the real anchor.

**Registered:** `ddm_pp1_correction_stream_position_band_v1` (evaluator + build + populate; advisory axis,
`score_claim=false`, `promotion_eligible=false`; appended to `.omx/state/canonical_equations_registry.jsonl`;
2 review passes on the .py). **The law's teeth:** a correction stream is rational only for base error
ρ ∈ ~[5e-4, 1e-2]; below ρ_c conceding at 1.2731 dominates; above ~1e-2 the total support explodes. **Design
spec sharpened: a carrier must be natively ≤ ~5e-4 (ideally ≤ 3e-4, PR130's rail) to ship NO correction
stream** — every internal base (paint 0.0086, ws1 0.024) sat inside/above the band and died of support cost;
PR130 (3e-4) sits below and ships none.

## Verdict routing (typed scope)
- **R1: direct-explicit partition coding NOT DEAD** (173.6 KB ≪ 350 KB falsifier; verdict_scope =
  FORMULATION: object `lstars`, coder families generic/plane/Paeth/RLE/context-arith intra+temporal/#307
  lane contour). It CONVERGES with the implicit token+renderer (~177 KB); it does NOT beat it (composed
  explicit route ~0.189 > 0.172, a +57 KB learned-prior gap). **The partition leg is cheap; the wall is
  realization** → the campaign's live route (fd1 family-d GN realization) is confirmed as the binding slot.
- **R8: lane-dash dictionary/contour DEAD for a byte win** (219.5 KB contour ≫ 62.3 KB context-arith Lane;
  verdict_scope = FORMULATION: the Lane class, #307 contour vs context-arith).
- **R3: band lemma CONFIRMED + REGISTERED** (measured coherent crossing 5.0e-4; sharpens the carrier
  native-error spec to ≤ ~5e-4).

## Honest boundary
- Nothing here moves the pointer (0.1910828242 [contest-CPU] UNMOVED). All findings are MEANS; the END is a
  byte-closed `upstream/evaluate.py` row. No scorer/render/train jobs were run by this arm.
- Every coder byte is a bit-exact round-trip-proven lossless length OR the closed-form adaptive length
  PROVEN == real coded bytes to <0.01% (subset round-trip receipt). The composed-arithmetic renderer/pose
  legs (40 KB / 2–23 KB / native 3e-4) are EXTERNAL anchors (PR130 [contest-CUDA] + sc1 measured), never
  this arm's numbers — they enter only as lessons for the composed-S sketch, not as adopted bytes.
- The band-lemma crossing ρ_c = 5.0e-4 is a log-density interpolation between two measured points
  (0.00022 above / 0.00056 below water); the uniform bound is the DERIVED law, the context shift is the
  MEASURED characterization.

## Wire-in (Subagent coherence 6-hook)
1. sensitivity-map: N/A (advisory coder measurement, no per-axis byte-weight change). 2. **Pareto:** the
S_partition(concession) curve + the band-lemma B/err(ρ) curve ARE measured rate/distortion Pareto rows.
3. bit-allocator: N/A. 4. cathedral dispatch: N/A (non-promotable). 5. **continual-learning:** this memo +
DAG FEED + the REGISTERED canonical equation `ddm_pp1_correction_stream_position_band_v1`. 6.
**probe-disambiguator:** R1 IS the disambiguator for ee1's fork (direct-explicit dead vs alive → alive but
convergent, not dominant); R3 IS the disambiguator for the correction-stream band edge.

## Artifacts
- Tools: `experiments/ddm_pp1_direct_partition_coder.py` (R1), `experiments/ddm_pp1_band_lemma_curve.py` (R2).
- Canonical equation: `src/tac/canonical_equations/ddm_pp1_correction_stream_position_band_20260728.py`.
- Committed receipt: `.omx/research/ddm_pp1_band_lemma_receipt_20260728.json` (equation provenance source).
- SSD receipts `/Volumes/VertigoDataTier/pact/ddm_pp1_20260728/`: `r1_direct_partition_n600.json`,
  `r1_lossy_concession_n600.json`, `r2_band_lemma_curve_n600.json`, `r1_smoke_n48.json`.
