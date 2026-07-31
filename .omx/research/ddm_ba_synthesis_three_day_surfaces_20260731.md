# Three-day negative-results audit — CROSS-DAY SYNTHESIS

**Date:** 2026-07-31 · **Axis:** `[macOS-CPU advisory]`, `score_claim=false` · **Pointer:**
0.1910828242 `[contest-CPU]` **UNMOVED**. This audit moved no number and does not claim to.

**Inputs:** `ddm_ba29_negative_response_surfaces_20260729.md` (a5695d1d9b) ·
`ddm_ba30_negative_response_surfaces_20260730.md` (05812c7f06) ·
`ddm_ba31_negative_surfaces_20260731.md` (93abf07892→6fb0bced5c) ·
`ddm_surface_correction_economics_20260731.md` (e68e564b6c, MAIN).
**Combined denominator:** 121 verdict-shaped statements found · 118 placed · 3 unplaceable and named.
Scope gaps stated by the arms: `.omx/state/*.jsonl`, `reports/`, non-ledger task rows.

---

## 1. ONE NUMBER APPEARS ON EVERY SURFACE, AND IT IS NOT EMPIRICAL

Six independent surfaces across four days each carry a "water level," "break-even," "exchange rate," or
"allocator threshold." **They are the same number, and it is pure arithmetic from the score definition:**

```
100 · (37,545,489/25) / (600·384·512)  =  1.2731082153320312   [closed form, ba31]
(100/N) / (25/37,545,489)              =  1.273108215332031    [MAIN, independent]
```

| surface | how it appeared | day |
|---|---|---|
| pp1 correction band | "registered region-merge water level 1.2731 B/flip" | 07-28 |
| ba29 Surface A | the window's only immovable level set | 07-29 |
| ba30 exchange | "15,018 bytes per 1e-4 of d_seg" | 07-30 |
| ja1 | allocator break-even | 07-31 |
| b2b | `w_rate` derived FROM it | 07-31 |
| ba31 | identified as the scorer's own seg↔rate rate | 07-31 |

**Consequence:** it was being treated as a measured constant to be respected. It is a *definition* to be
computed. Nothing about it can be "re-measured," and any surface that reports a different break-even for
the same units has a units error, not a new finding.

## 2. THE ONE PLACE IT ACTUALLY STEERS A RUN, IT IS UNDER-WEIGHTED BY 34.8%

The live burn (`ddm_b4s_20260731/window_01/tr1_config.json`, hash `d2e31f1f…`) ships
**`w_rate: 0.05`, `w_seg: 100.0`**. b2b's S-commensurate derivation, recomputed by MAIN from first
principles: `(25/37,545,489)·(921,600/8) = 0.076707` (921,600 = 24×32 token grid × 1200 frames).

```
0.05 / 0.076707 = 0.6518   ->  the rate gradient runs at 65.2% of what the SCORE says it is worth
```

**CONDITIONAL:** sound only if the trainer's rate term is *mean bits per token*. **Verify the units in
the trainer before acting** — this is the single unverified link in an otherwise closed chain, and it is
the difference between a real 34.8% under-weighting and a units mismatch. The resulting byte over-spend
is unmeasured either way.

## 3. THE SHARED DEGREE OF FREEDOM: CODER QUALITY — four independent arrivals

| arrival | statement |
|---|---|
| MAIN (band lemma) | coherence moves the correction band's **rate edge across 2.99×** (incoherent 1.5e-3 → uniform 8.59e-4 → coherent 5.02e-4); the law's word *"permanently"* describes our current coder |
| ba29 | coder quality sets position on **both** Surface A (byte↔flip) and C (token coding); **exhausted on C** (19 coordinates, span 1.381×, ideal ceiling −0.54%) and **completely unmeasured on A** — where the only above-water verdict sits |
| ba30 | **the admission unit must equal the coder's conditioning unit** — one knob folding four separate "aiming beats random?" negatives into one statement |
| ba31 | band edge spans **2.991×** across coder families, and **3 of 5 per-class flicker floors sit within 1.76× of it** — i.e. *inside* the measured coherence effect |

ba31's last row is the sharpest: **per-class "floors" that sit inside the span of a knob we control are
not floors.**

## 4. THE SAME OBJECT PRICED THREE WAYS, SIGN FLIPPING EACH TIME — and it decides a `do_not_spend`

All three arms independently priced **QA03's accepted correction quanta**:

| pricing | result | source |
|---|---|---|
| tr1 re-encode | **1.140× OVER** water (net +0.000222) | ba29 |
| SMEVR average-scaled | **0.842× UNDER** water (net −0.000251) | ba29 |
| coherent position | **−0.204 S** | ba31 |
| uniform bound, no interpolation | **−0.099 S** | ba31 |
| composite | **+0.060 S** | ba31 |
| position only (band lemma) | 0.545× water, **45.5% of budget unpriced (values)** | MAIN |

**Swing 0.264 S — larger than the entire rate axis (0.239 S).** ba31 adds the mechanism: QA03's
1.45 B/flip exceeds even the *uniform position bound* (1.0007) by 1.449×, so **≥51.6% of it is
label+solver cost — a term the governing law explicitly excludes.** And the **composite** — the single
pricing that comes out positive — is the sole basis on which ja1 marks the "seg corrections" pool
`do_not_spend`.

**This is the audit's largest single finding.** A pool is closed by the one pricing, out of six, that
gives the closing sign, on a quantity over half of which the governing law does not cover.
→ **#832** (SMEVR marginal re-price of the 326 accepted quanta, $0, harness exists) settles it.

## 5. TWO WAYS AN INSTRUMENT CAN MEASURE SOMETHING OTHER THAN THE SYSTEM

**(a) The number is a dataset constant** (ba30, MAIN-verified float64). pj1's renderer "capacity floor"
f = 0.504824 **is** the constant-Undrivable predictor, `0.50482448154026`, with the per-class vector
matching at **abs diff 0.00e+00 on all five classes**. fp1's f′ = 0.499366 = **98.92%** of the same
corner after 50 converged epochs. Both measured majority-class collapse.
*Scope, held:* **BR-D's firing is unaffected** — fp1's load-bearing receiver floor 0.008305 is computed
on the ground-truth argmax and is collapse-immune. Only fp1's *secondary* "trunk not small-conv-decodable"
rests on the collapsed run. → **#833** (two-landing: re-grade + degenerate-baseline control).

**(b) The quantities are not coupled at all** (ba31). gc12 reports "spend bytes to buy d_seg,
favorability ~2.7×"; gc14 measures Δbytes vs Δd_seg at **r = +0.212, t = +1.30** — same-direction drift.
An exchange requires r<0; t=1.30 means positive coupling is not established either. Honest read: **no
demonstrated coupling in either direction.** ba31's law is binding: *an exchange rate is only meaningful
between quantities shown to be causally coupled.* → **#834 corrected**; MAIN had composed the 2.7× into
a "~30 KB net-positive alternation" twenty minutes earlier. **An arithmetic composition inherits the
weakest premise of its inputs, and "is this even an exchange?" is a premise.**

## 6. LEVEL SETS THAT MOVED THE MOMENT SOMEONE LOOKED

- **Realization wall is ~1/k LSB, not 1 LSB** (k≈30 area-averaging in R). The steering-atom family was
  killed **on amplitude**, with support×coherence never measured. *(ba31 — a reactivation.)*
- **The exact-solve "floor" is a curve:** box 136,839 err @291 MB vs exact 17,927 @409 MB = **7.64× error
  for 1.41× bytes**. The amortization gap is 3.35× or 25.6× depending on which point you call the floor.
- **fl1's falsifier had no power at this operating point:** corner-C is a *proportional* split
  (k=0.14071 across all five classes to four digits), so its five numbers are the piercing ratios
  rescaled by one constant; the trigger sits **5.02× below** the nearest class. fl1 published both
  columns and reached the correctly-scoped verdict — **sound**.
- **Leverage split (ba29), the number no verdict on either side carried:** coding the object better is
  **saturated within 1.381× across 19 coordinates**; changing *what is described* moved **39.85×**
  (bytes), **145.8×** (support: 1,415,927 B transmitted → 9,711 B derived, same channel/coherence/flips),
  **417,000×** (pose) — inside 24 hours.

## 7. CALLED SOUND, WITH REASONS (not padding — this is the control group)

fl1's scoped NO-re-waterfill · fp1's receiver floor · of1 Probe 1's structural criterion · v4c rung-A +
QA58/QA61 · pi2's re-aim · gr1's n48 knee · nv1's product-vs-additive reframe · ba29 Surface C's
saturation · gc13's B10 false-positive adjudication · gc16's cadence bound · sb2's build-asymmetry ·
ph3's operator-caught two-solve conflation · and MAIN's own ALARM #3 refusal, KKT reading of λ=0, and
byte-ledger law. **ba31's note: most of these are arms overturning their own headline — the most
encouraging pattern in the window.**

## 8. WHAT THIS CHANGES

1. **#832** — SMEVR marginal re-price of QA03's 326 quanta. $0, harness exists. Settles §4's sign.
2. **#833** — degenerate-baseline control (two-landing) + pj1/fp1 re-grade, scope held to §5(a).
3. **#834** — reformulated: establish coupling FIRST; the alternation arithmetic is retracted as arithmetic.
4. **w_rate units check** (§2) — verify the trainer's rate-term units, then decide 0.05 vs 0.076707.
5. **`ps1_ladder.partial.jsonl`** — 600 rows on disk, unread; pose is the largest-weight axis (~1.24 S,
   exceeding seg 0.431 + rate 0.239 combined) and no base in the 07-30 window was chosen for it.

**None fired.** Three arms landing in sequence is not three licences to fan out; these compose, and #834
is the standing proof that composing before checking premises is how the errors get made.
