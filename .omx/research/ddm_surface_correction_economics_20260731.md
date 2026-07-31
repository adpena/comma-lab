# The correction-economics SURFACE (not a regime tag)

**Date:** 2026-07-31 · **Author:** MAIN · **Axis:** `[macOS-CPU advisory]`, `score_claim=false`,
`promotable=false` · **Pointer:** 0.1910828242 `[contest-CPU]` **UNMOVED** — this is a re-reading of
already-measured data, not a new row.

**Why this exists.** Operator, verbatim (2026-07-31): *"I've told you numerous times no binary results
ever. We are proceeding Einsteinian and according to our design and guiding philosophies and principles,
which include unification, completeness, and more."* The costate organ reports
`ddm_pp1_correction_stream_position_band_v1` as a **three-valued regime tag** {CONCEDE / CORRECT /
EXPLODE} and prints `-> corrections DEAD`. The underlying law is a **9-point measured curve on a
multi-axis surface**. This document is that surface, with every one of our own bases placed on it as a
coordinate. Nothing here is new measurement; every number is recomputed from
`.omx/research/ddm_pp1_band_lemma_receipt_20260728.json` (`lemma_confirmed=true`, N_sites=117,964,800 =
512×384×600).

---

## 0. The level set is real and independently confirmed

Fixing one argmax error gains `100/N = 8.477e-7` S. One archive byte costs `25/37,545,489 = 6.659e-7` S.
Break-even = **1.2731 B/err** — *exactly* the registered water level `W`, re-derived here from the
contest score definition alone. **W is not a tuned constant; it is the S-neutral exchange rate.** That
part of the law is solid and is the surface's natural unit.

## 1. The tag welds TWO different economic criteria with different units

| edge | criterion | units compared | crossing ρ |
|---|---|---|---:|
| lower ("CONCEDE below") | **average** position cost vs W | B/err vs B/err — a **RATE** | 5.015e-4 |
| upper ("EXPLODE above") | **total** support bytes vs the archive box | B vs B — a **BUDGET** | ~1e-2 (as stated) |

Recomputed, the budget edge is **not** at 1e-2 — it is wherever the box is:

- total support ≤ **200 KB** → ρ = **1.836e-3**
- total support ≤ **130 KB** → ρ = **1.015e-3**

The two lower edges are **3.66× apart**. A single scalar tag on a single axis cannot carry both.

## 2. The two axes move OPPOSITELY in ρ — which the tag's ordering hides

| ρ | k errors | b_pos (B/err) | rate vs W | TOTAL support B |
|---:|---:|---:|---:|---:|
| 0.0705192 *(the "corrections DEAD" base)* | 8,318,783 | 0.0679 | **0.053** | 564,922 |
| 0.0138329 | 1,631,795 | 0.2968 | 0.233 | 484,296 |
| 0.0086421 *(fc1 real anchor)* | 1,019,467 | 0.4309 | 0.338 | 439,300 |
| **0.0038892** *(window_03 endpoint, today)* | 458,789 | 0.6941 | **0.545** | 318,441 |
| 0.0010000 | 117,965 | 1.0895 | 0.856 | 128,522 |
| 0.0005015 *(the tag's lower edge)* | 59,159 | 1.2711 | 0.998 | 75,198 |
| 0.0003000 *(PR130 rail)* | 35,389 | 1.3943 | 1.095 | 49,342 |

**At the base where the organ prints "corrections DEAD," position costs 0.053× the water level —
corrections are ~19× CHEAPER per error there than conceding.** The DEAD tag at that coordinate is a
*budget* statement (565 KB total), never a rate statement. Read as "corrections don't work," it inverts
the actual economics.

**Today we are IN the tag's band and simultaneously 2.12× over the 200 KB budget edge** (3.83× over
130 KB). "IN band" is TRUE on the rate axis and FALSE on the budget axis at the same coordinate.

## 3. The average-vs-marginal collapse (the largest single loss)

The band's lower edge is where the **average** position cost crosses W. A waterfill spends at the
**margin**. The 9 τ rungs are nested supports, so differencing totals gives the marginal directly:

| τ | ρ | k | AVG B/err | **MARGINAL B/err** | marg vs W | S-positive at the margin? |
|---:|---:|---:|---:|---:|---:|:--|
| 0.008 | 2.247e-4 | 26,512 | 1.4687 | 1.4687 | 1.154 | no |
| 0.02 | 5.632e-4 | 66,438 | 1.2448 | **1.0962** | 0.861 | **yes** |
| 0.05 | 1.413e-3 | 166,700 | 1.0054 | 0.8468 | 0.665 | yes |
| 0.1 | 2.824e-3 | 333,078 | 0.8026 | 0.5994 | 0.471 | yes |
| 0.2 | 5.622e-3 | 663,192 | 0.5872 | 0.3699 | 0.291 | yes |
| 0.4 | 1.112e-2 | 1,312,199 | 0.3593 | 0.1264 | 0.099 | yes |
| 0.8 | 2.170e-2 | 2,560,396 | 0.1999 | 0.0323 | 0.025 | yes |
| 1.5 | 3.800e-2 | 4,482,583 | 0.1181 | 0.0090 | 0.007 | yes |
| 3.0 | 7.006e-2 | 8,264,825 | 0.0679 | 0.0085 | 0.007 | yes |

**Every rung from τ=0.02 upward has marginal position cost BELOW W.** Only the tightest rung (τ=0.008)
is above. On the position axis, widening the support is S-positive essentially everywhere measured.
The law's sentence *"sub-ρ_c correction machinery is **permanently** pointless"* is a statement about
the **average**, applied as if it governed the **margin**. "Permanently" is the binary.
*(DERIVED by differencing the receipt's nested-τ averages — assumes true nesting, which the τ-ordering
supports; not independently re-measured.)*

## 4. What moves each level set — all of them are OUR OWN knobs

| level set | moved by | measured range |
|---|---|---|
| rate edge (avg b_pos = W) | **coherence exploited by the support coder** | incoherent 1.5e-3 → uniform bound 8.59e-4 → coherent 5.02e-4 = **2.99×** |
| budget edge (total = box) | **the box itself** | 200 KB → 1.836e-3 · 130 KB → 1.015e-3 |
| effective operating point | **all-errors vs waterfilled subset** | the tag assumes ALL; a subset is the real object (→ #766 wr1) |

**Coder quality and the band edge are the same degree of freedom.** r7 raced 14 coders; that is a
measured axis that relocates a "law constant."

## 5. Completeness — the terms this surface still does NOT carry

1. **VALUES are unpriced.** `b_pos` is the *support/position* term only. Total = position + values. At
   today's base, position takes 0.545× of the S-neutral budget, leaving **0.579 B/err (45.5%)** for
   values before total reaches W. The organ's own sibling line reports *"QA03/QA04 white-jitter =
   MEASURED BREAK-EVEN at this base"* — ratio **1.0**, i.e. a **total** landing exactly on the level set.
   That is *consistent* with position 0.545× + values ≈ 0.455×, and would make break-even a
   **decomposition result**, not a stopping reason. **NOT VERIFIED HERE** — the resolving check is to
   read QA03/QA04's cost decomposition and confirm whether its measurement is total-inclusive.
2. **REALIZATION is a wholly independent axis.** Price says nothing about whether an edit survives
   R→uint8→SegNet. The ERF-collateral law (fp1 + QA92) measures post-hoc injection on textured renders
   as **net-worse even with perfect GT** (+0.30 S, ~85px ERF re-reads the stroke). Cheap position cost
   does **not** imply a realizable correction. Both the band lemma (price) and QA92 (realization) were
   being read as verdicts on the single word "corrections."
3. **Evidence class.** The receipt is explicit: synthetic correction-support densities, coherent +
   incoherent synthesis, cross-checked against the fc1 real anchor (interp 0.44 vs measured 0.413
   B/err). It is **NOT a byte-closed evaluate.py row.**

## 6. Where this leaves the object (a placed point, not a verdict)

At ρ = 0.0038892 we sit **inside** the rate band at 0.545× water on position, **2.12× outside** the
200 KB budget edge on totals, with **45.5% of the S-neutral budget unaccounted (values)** and the
realization axis **measured hostile** for post-hoc injection specifically. The surface's own gradient
says the actionable directions are: lower the base (moves us down-left along both axes at once),
improve support coherence (relocates the rate edge, 2.99× of measured room), or waterfill a subset
instead of correcting all (which the marginal table says is S-positive at every measured rung, and
which is exactly #766's object). None of these is visible from the word DEAD.

**Sisters:** `ddm_pp1_correction_stream_position_band_20260728.py` (the law) ·
`erf_collateral_law_no_posthoc_injection_on_textured_renders_20260731` (the realization axis) ·
`boolean_flags_are_a_ui_over_a_continuum_never_binary_judgment_20260731` (the output-form rule) ·
task #766 (wr1 reverse-waterfill — the consumer) · task #822 (lane guard: same "budget never tightens"
shape at a different surface).
