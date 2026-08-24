# ddm_af1 — the address-free law cannot be tested, because the ladder it would be tested on is not one quantity, and the cut has no members on the deciding side

`date_utc: 2026-08-24` · `axis: [desk / arithmetic on banked receipts — no scorer, no dispatch]` ·
`score_claim: false` · `promotion_eligible: false` · `pointer_moved: false` · cost $0

`verdict_scope`: **the six-rung exchange ladder as circulated on 2026-08-24, plus tba1's D1–D6 map.**
This is a re-derivation and re-classification of banked numbers. It runs no new measurement and
closes no mechanism family. What it closes is a *comparison*.

STORES CONSULTED: `ddm_bo2_born_small_distortion_row_20260824.md` · `ddm_tba1_token_bit_attribution_20260823.md`
(D1–D6 map) · `ddm_tv1`/`ddm_tv2_evaluator_tolerance_curve_20260824.md` (ladder table §5.1.4, credit
flag §5.1.5) · `ddm_w72_distortion_advisory_20260823.md` · `ddm_ni1_247x_erratum_20260822.md` ·
`ddm_dg2_diagonal_distortion_verdict_20260824.md` · `ddm_df1_dddb_field_20260824.md` (address floor §4) ·
`ddm_hc1_hpac_calibration_reliability_20260824.md` (indicator/conditional split) ·
`ddm_ad2_addressing_cost_decomposition_20260822.md` · `ddm_ae1_anti_predicted_excess_20260822.md` ·
`ddm_tri1_triple_composition_and_pair_closure_20260824.md` §3.4 (the 21.62× derivation) ·
`ddm_ds1_cheap_to_shrink_objective_20260824.md` §6 (prior flag on the same `r` values) ·
`ddm_tx1_toolbox_crosswalk_20260819.md` §0 (exchange rate, CITED) · `ddm_ar1b_archive_residue_purchase_20260822.md` ·
`ddm_dc1_decode_time_compute_20260821.md`.

---

## 0. Answer first

**The law cannot be tested, and the reason is not the one MAIN gave.** MAIN's charter said the law
rests on n≤1 because born-small was the only address-free member and was unmeasured. `ddm_bo2` then
landed and MAIN revised to "refuted at n=1." **Both readings are wrong in the same direction: they
assume the ladder has members on both sides of the cut. It does not.**

Applying a mechanical criterion (§3), **all six ladder rungs are address-FREE. Zero are
address-paying.** The discriminator is *constant* over the measured population, and a constant
carries no information. The law is not refuted by bo2 and it is not supported by anything — it is
**undefined on this evidence**, at effective **n = 6 vs 0**.

And there is a structural reason the sides can never be compared by this ladder at all:

> **Address-paying mechanisms name a subset in order to code it EXACTLY. They are lossless, so they
> have no distortion numerator and therefore no exchange ratio. Address-free mechanisms here discard
> the subset information, so they are lossy and always have one. The two classes are measured in
> different units and live on different axes.** The exchange ladder is only defined on one side of
> the cut. Asking it to separate the sides is a category error, not a small-n problem.

This is not an argument from definitions alone: **six address-paying mechanisms are BUILT and priced in
this tree** (ae1's 130,228 B of explicit flags, ad2's 12,110 B of QEVENT addresses, and four
table-storing merge codecs — §3, §6). **All six are lossless. All six have zero distortion. None has an
exchange ratio.** The sides are populated; they are just not commensurable.

**Three further findings, in descending order of how much they change what fires next:**

1. **The ladder is not one quantity — it holds THREE different conventions, and correcting them
   changes both the ordering and the covariate.** W72's `35.5364×` is an **S-ratio**
   (`S_treated ÷ S_matched-base`), not damage÷credit; its true exchange ratio is **922×**. ni1's
   `247.69×` is **seg ÷ seg-break-even-ceiling**; its true damage÷credit is **714×**. `tba1`-D3's
   `21.62×` is **derived, seg-leg-only, on a mechanism that was never built**. On one convention the
   ladder reads **33.7× · 97.2× · 686× · 714× · 922×** — and W72 moves from *third best* to *worst*.
   §2.

2. **MAIN's pose-share covariate does not survive the correction. It reverses.** On the corrected
   ladder `corr(pose_share, log ratio) = −0.155` at n=5 — indistinguishable from zero, with the sign
   *opposite* to the hypothesis. The apparent monotone trend was an artifact of W72's S-ratio sitting
   in a damage÷credit column. The one covariate that survives is **size**
   (`corr(log credit_B, log ratio) = −0.667`): *bigger* bites have *lower* ratios — which is dg2's own
   within-family law reappearing, and it is the unhelpful direction. §4.

3. **MAIN's `hc1` ↔ `df1` identification is numerically CORRECT (they agree to 1.98%) and
   strategically INVERTED.** `df1`'s address floor 109,113.50 B and `hc1`'s realized indicator block
   111,275.62 B are the same information-theoretic object measured two ways. But on this object that
   object is **the payload, not overhead**. Going address-free does not release it — the decoder
   genuinely does not know where the model is wrong. "The address tax is 181% of demand" is true
   arithmetic that licenses a false hope. §5.

**The single cheapest deciding measurement is in §9, and it is not a test of this law** — the law is
undefined, so paying to test it would be paying for a constant. §9 names what to buy instead.

---

## 1. What the operator's four families actually are, and where they already sit

The operator named **born-small · merging · regime · implicit**. MAIN read a shared discriminator off
those names and asked me to check whether that is pattern-matching on vocabulary. It is, and
`ddm_tba1` shows why: **the same space was already partitioned three days ago, by mechanism rather
than by name, and both sides of the operator's cut appear in it as D1 and D6.**

tba1's D1–D6 map (`ddm_tba1_token_bit_attribution_20260823.md`, the untested-directions table):

| tba1 | direction | operator's family | status in tba1 | ceiling |
|---|---|---|---|---:|
| **D1** | explicit position-selected treatment at any cost threshold | **address-PAYING** | SHARP — closed by arithmetic | **+9.45 B** |
| **D6** | **free (receiver-regenerable) selectors** | **regime / address-FREE** | **SHARP — budget already spent** | — |
| D2a | drop CHEAP positions | — | insufficient by 10.1× | 4,182.59 B |
| D2b | drop EXPENSIVE positions | (= ni1/nr1) | task collapse | >79,276 B |
| **D3** | **alphabet reduction in the model** (class merge → retrain) | **merging** | **the one ceiling that approaches demand** | **38,649.8 B** |
| D4 | probability-model precision | — | sharp and wrong-signed | 974.1 B |
| D5 | pair-selective coding | — | **not sharp — EMPTY** (no structure) | — |

Two things follow, and they matter more than the classification exercise MAIN chartered:

**(a) BOTH sides of the operator's cut are already closed, for DIFFERENT reasons, and neither closure
is the address tax on its own.** D1 (address-paying) is closed by the tax — ceiling **+9.45 B**, i.e.
0.022% of demand. D6 (address-free / regime) is closed by **EXHAUSTION**, not by tax: tba1's verbatim
reason is *"the shipped HPAC context already consumes the best free selector (predicted class, t−1/t−2
agreement, boundary bucket). oe1 and cx3 tested it and lost."* The free-selector budget is spent by the
model we already ship.

**(b) "regime" is D6 and is therefore not new.** A regime split is exactly a receiver-regenerable
selector — the decoder determines which regime it is in and switches representation. That is the
definition of a free selector, and the shipped HPAC context already conditions on the three best ones.
MAIN's candidate regime evidence confirms this rather than opening it:

- **`tv1`/`tv2`'s τ-inversion is real and is a description, not a lever.** bo2's own transferable
  correction is the binding one: *"the τ-inversion routes damage between axes; it does not bound it on
  either."* I add the coding reading: a regime split only pays if the decoder can *condition* on it,
  and boundary-vs-interior is already in the shipped context as `boundary bucket`.
- **`hc1`'s causal `d` at 49.85%** is genuinely decoder-derivable and genuinely free — and it is
  already the same `boundary bucket` family. hc1 itself prices the whole model-calibration axis at
  **8.44 heldout net B**.
- **`g4` / `ad2`** — see §6; ad2's one real win is on a different vehicle.

**(c) "merging" is D3, and D3 is the single most valuable open ceiling on the map** — 38,649.8 B =
**91.19% of the 42,382 B demand**. §6 inventories what is built for it.

**"implicit" is the one family with no tba1 row**, because tba1 mapped the *token-coding* axis and
implicit representations are a *decode-time compute* axis. That is `ddm_dc1`. §7.

---

## 2. The ladder is three conventions wearing one column header

Every rung below is a banked, cited number. What differs is the **denominator**, and nobody wrote the
denominator into the column header. `ddm_ds1` §6 already flagged this — *"I did not verify the
charter's `r` values (21.62×/46.3×/247.69×/349×/478.7×/687×/792×). I used them as supplied and flagged
the W72 discrepancy rather than resolving it."* This section resolves it.

### 2.1 The three conventions found

| # | convention | meaning | rungs using it |
|---|---|---|---|
| **A** | `ΔS_distortion ÷ (bytes_saved × λ_B)` | **how many times over break-even the trade is** | bo2, dg2, tolerance |
| **B** | `S_treated ÷ S_base` | how many times worse the total score is | **W72** |
| **C** | `Δd_seg ÷ d_seg_break-even_ceiling` | seg-leg-only, against a ceiling | **ni1**, **tba1-D3** |

Only **A** answers "is this trade worth making." **B** and **C** are different questions whose answers
happen to be dimensionless, which is what let them into the column.

### 2.2 W72 — the 26× understatement, derived

MEASURED inputs (`ddm_w72_distortion_advisory_20260823.md` §1): W72 `d_seg 0.02351655`,
`d_pose 1.93641210`, archive 169,489 B. Matched-lineage base = `tv1 k0_control`
(`d_seg 0.00034740`, `d_pose 0.00014701`, 180,368 B) — the same base bo2 used.

DERIVED here:

```
S(base)    = 100(0.00034740) + √(10·0.00014701) + 25·180368/37545489 = 0.193181530761251
S(W72)     = 6.864979038642395                    (w72 §1, recomputed from components)
S(W72)/S(base)                                    = 35.536414954319206   ← the ladder's "35.5364×"
S(W72)/S(dx2 contest-CUDA 0.14821987563243377)    = 46.31618404313508    ← the withdrawn "46.32×"
```

The `35.5364×` entry reproduces to **10 significant figures** as an S-ratio. It is convention **B**.
tv2 §5.1.4 introduced it with the note *"matched-lineage; the circulating 46.32× divided a macOS-CPU
numerator by the contest-CUDA pointer — #1034, corrected by `ddm_df1`."* **That correction is right and
insufficient**: it fixed the lineage of the denominator without noticing the denominator was the wrong
*kind*. Both 35.54 and 46.32 are S-ratios; neither is an exchange ratio.

On convention **A**, against the same matched base:

```
Δd_seg → ΔS  = 100·(0.02351655 − 0.00034740)                    = 2.316915
Δd_pose → ΔS = √(10·1.93641210) − √(10·0.00014701)              = 4.362126
damage                                                           = 6.679041
credit       = 10,879 B × 6.658590e-07                           = 0.00724388
ratio        = 6.679041 / 0.00724388                             = 922.0×
```

**W72's exchange ratio is 922×, not 35.54×.** The w72 memo's own text already said this in a different
form — *"the distortion it pays is ~928× that credit"* (§0, computed against the CUDA base). My 922×
is the matched-base version of the same statement. The 928× was in the memo the whole time; the ladder
took the S-ratio instead.

### 2.3 ni1 — convention C

`ddm_ni1_247x_erratum_20260822.md` is explicit that its number is *"measured ratio to NI1's own
break-even ceiling: 0.07583781 / 0.000306175 = 247.69×"* — a **seg-only ratio against a d_seg
ceiling**. On convention A, with ni1's MEASURED `d_seg 0.07583781`, `d_pose 40.53479004`, 122,250 B:

```
damage = 100(0.07583781−0.00034740) + √(10·40.53479004) − √(10·0.00014701) = 7.549041 + 20.094900 = 27.643941
credit = (180,368 − 122,250) B × λ_B = 58,118 × 6.658590e-07                                       = 0.038698
ratio                                                                                              = 714.3×
```

ni1's memo is not wrong — it answers its own question correctly and says so. The **ladder** is wrong
for putting that answer in a column asking a different question.

### 2.4 tba1-D3 — the "best measured rung" is neither best nor measured

`21.62×` does not appear in `ddm_tba1`. Its source is
`ddm_tri1_triple_composition_and_pair_closure_20260824.md` §3.4, which priced tba1's D3 **ceiling** by
transferring **dg2's MEASURED 0.9528 final-flips-per-edit** onto a hypothetical 656,333-flip edit set:
`656,333 flips ⇒ 0.556381 S ⇒ 21.62×`. tri1 labels it honestly — *"the seg leg alone refuses at
≥21.62×, before a single unit of pose is counted."*

So the rung the whole ladder is anchored on is:

- **DERIVED, not measured** — no D3 model was ever retrained; tba1 lists it as *untested*;
- **seg-leg-only** — pose is not merely small in it, it is **absent**;
- a **LOWER bound** on its own true value, by tri1's own statement.

**This is the ladder's most consequential defect, and it is a selection artifact:** D3 ranks best
*because* the numerator excludes the term that dominates every rung that was actually measured. Across
the five measured rungs pose carries **65.3%–93.3%** of damage. Restoring even the *smallest* observed
pose share to D3's numerator moves it from 21.62× to ≈62×, behind tolerance and born-small. I do not
publish 62× as a prediction — D3's pose is unmeasured and unmeasurable without building it — but the
ordering claim "21.62× is the best exchange ratio ever measured" cannot stand. **Nothing on this ladder
is below 33.7× on a like-for-like basis.**

### 2.5 The ladder, on one convention

MEASURED distortion in every row; credit as noted. Break-even = 1.0.

| mechanism | credit (B) | damage (ΔS) | **ratio (conv. A)** | pose share | as circulated |
|---|---:|---:|---:|---:|---|
| tolerance-cond k=10⁶ | 104,166 | 2.3378 | **33.7×** | 80.7% | 33.7× ✓ |
| `bo2` born-small (HG1) | 79,240 | 5.1311 | **97.2×** | 75.4% | 97.25× ✓ |
| `dg2` diagonal k060000 | 1,576 | 0.7202 | **686.3×** | 93.3% | 686× ✓ |
| `ni1` whole-body lossy | 58,118 | 27.6439 | **714.3×** | 72.7% | 247.69× ✗ (conv. C) |
| `W72` renderer rung | 10,879 | 6.6790 | **922.0×** | 65.3% | 35.5364× ✗ (conv. B) |
| `tba1`-D3 | 38,649.8 (ceiling) | seg-only, derived | **not placeable** | **unmeasured** | 21.62× ✗ (conv. C, unbuilt) |

Two flags travel with this table and must not be stripped:

- **The credit column for tolerance is a STATIC accounting.** tv2 §5.1.5 flags it and `ds1` measured
  static `−log₂p` on this exact field mispricing by **14.59× in the false-win direction**. Static
  accounting **overstates** releasable bytes, so 33.7× is a **LOWER bound**. Per
  `[[price-token-field-levers-by-real-reencode]]`, no modified stream was re-encoded.
- **bo2's credit is dx2-relative** (the alternative to shipping born-small is shipping dx2). bo2's
  sub-0.12-budget reading of the same row is 209.07×. MAIN's amendment states the governing caveat:
  *"my two bars are two CORNERS of one budget… quoting one alone quotes half a budget."*

---

## 3. The classification, with a mechanical criterion — and why it has n = 0 on the deciding side

The charter demanded a criterion someone else could apply and get the same answer, and warned that
otherwise I would be sorting by vibe. Here it is.

> **CRITERION (apply to the archive, not the name).** Does the archive contain bytes whose *function*
> is to identify WHICH positions/elements receive distinct treatment, and whose *length grows with the
> cardinality of the selected subset*? If yes → **address-PAYING**. If no → **address-FREE**.

The growth clause is what makes it mechanical: a global re-representation has a length set by the
*model*, not by |selected set|. `ddm_ad2` independently uses the same operational test — it physically
attributes bytes to `addressing` vs `value` columns and can therefore report e.g. *"NR1 QEVENT
addresses only: 12,110 B"* and *"spatial assignment IDs at implicit raster sites — addressing —
10,900 B."* That is this criterion already implemented on real packets, by another arm, before I
wrote it. I take that as evidence the criterion is operational rather than mine.

Applying it:

| mechanism | does any section name a subset, growing with its size? | class |
|---|---|---|
| tolerance (tv1/tv2) | No — the *field* is edited, then the unchanged receiver codes whatever it is given | **FREE** |
| `bo2` born-small | No — an analytic generator; decoder re-derives the field | **FREE** |
| `dg2` diagonal | No — field and model are edited; the coder codes the result | **FREE** |
| `ni1` whole-body | No — global lossy re-representation | **FREE** |
| `W72` renderer | No — a weight layout (`nested_group_dense`) | **FREE** |
| `tba1`-D3 | No — alphabet collapse **in the model**, global | **FREE** |

**Six free, zero paying.** MAIN's amendment classified W72 and tba1-D3 as address-paying; by this
criterion both are free — W72 stores a weight layout and tba1-D3 stores an alphabet, and neither
length depends on any selected subset. Correcting those two removes the last members from the paying
side.

**Address-paying mechanisms do exist on this campaign — six of them, built and priced. They are simply
not on this ladder, and cannot be:**

| mechanism | the stored address | measured |
|---|---|---|
| `ddm_ae1` explicit flags | 93,580 anti-predicted positions | **130,228 B** (0.9481× the i.i.d. reference 137,351.94 B, per tba1's calibration) |
| `ddm_ad2` NR1 QEVENT | explicit coordinate addresses | **12,110 B** |
| `packet_member_merge_receiver.py:98` | ZIP member table (names+offsets+lengths) | byte-closed, sha-verified per member |
| `pr103_arithmetic_coding.py:199` | `tensor_symbol_counts` stream boundaries | source: decoder *requires* it (`:126-129`) |
| `hnerv_pr103_lc_ac_schema.py:234` | `AC_STREAM_SPECS` layout constants | byte-parity proven by re-encode (`:289-291`) |
| `pr101_split_brotli_codec_derivers.py:135` | 28-tuple storage permutation | baked as `DECODER_STORAGE_ORDER` |

**Every one of them is LOSSLESS** — that is not a coincidence but the definition: you name a subset
*in order to code it exactly*. Their distortion is identically zero, so their exchange ratio is
`0 ÷ credit = 0`, or undefined where credit is also zero. **None can appear on a ladder whose ordinate
is damage÷credit.** That is the structural statement in §0, now carried by six built artifacts rather
than by an argument: the cut separates lossless from lossy, and the ladder only measures lossy.

**So the honest verdict is not "refuted" and not "n=1."** It is: *the classification is well-defined
(§3 criterion), the measured population is entirely on one side of it, and the other side is measured
in units the ladder does not carry. The law is **undefined on this evidence**.* MAIN's revised reading
("address-freedom removes a byte FLOOR, not a distortion CEILING") is **correct as physics** and I
endorse it — but it is a statement derivable from the definitions, not something bo2 measured, and it
should be labeled DERIVED rather than carried as bo2's result.

---

## 4. What actually predicts the ratio — MAIN's Q2, answered

MAIN asked three questions. Taking them in order.

### 4.1 Fill the missing pose shares (Q1)

- **`ni1`: pose share 72.7%** — DERIVED from ni1's MEASURED components
  (`pose ΔS 20.0949 / total 27.6439`).
- **`tba1`-D3: pose share is UNKNOWN and cannot be filled.** D3 was never built; tri1 priced the seg
  leg only. **This is the answer to MAIN's "those two rows decide it": one of the two deciding rows
  does not exist.** And it is the row carrying the best ratio — which is precisely §2.4's selection
  artifact.

### 4.2 The sign (Q2)

On the corrected ladder, n=5 rows with both legs measured:

```
corr(pose_share , log ratio)  = −0.155     ← MAIN's hypothesis, sign REVERSED, magnitude ≈ 0
corr(log credit_B, log ratio) = −0.667
corr(pose_share , log credit) = −0.481
```

**MAIN's covariate does not survive.** The worst rung (W72, 922×) has the **lowest** pose share
(65.3%); the best (tolerance, 33.7×) has 80.7%. The apparent monotone trend in MAIN's table was
produced by W72 sitting at 35.5364× — its **S-ratio** — instead of 922×. Correct that one cell and the
trend inverts.

MAIN pre-authorized this answer: *"with 7 points and an eyeballed monotone trend I am exactly the
reader who should not be trusted… '3 usable rows, not enough' is an acceptable and useful answer."*
The measured answer is stronger than that: **5 usable rows, and the correlation is −0.155.** At n=5
that is not distinguishable from zero in either direction. I do not claim pose-share is
*anti*-correlated; I claim it is **not the covariate**, and that the evidence for it was an artifact.

### 4.3 The confound (Q3) — and it is the live one

MAIN asked whether pose share is a proxy for total perturbation magnitude. **It is worse than that:
size is the covariate that survives, and pose share is the one that does not.** `corr(log credit_B,
log ratio) = −0.667` — the strongest relationship in the table, and *negative*: bigger bites have
lower ratios.

That is not a new law. It is **`dg2`'s own measured within-family law appearing across families**:
`ratio ∝ B^−0.2748`, with both ends measured (1,576 B → 686×, 942 B → 792×). dg2 already drew the
consequence: *shrinking a move raises its ratio*, so **no amount of taking smaller bites reaches
break-even**. My cross-family correlation says the same thing with five points from five mechanisms.

And it is the unhelpful direction, which should be said plainly: **the two largest bites ever measured
— 104,166 B and 79,240 B, together far exceeding the 42,382 B demand — still sit 33.7× and 97.2× over
break-even.** Size does not rescue; it merely correlates.

### 4.4 The wall is arithmetic, not classification

The relationship that *does* hold, at n=5 with no exceptions, is not a covariate over mechanisms — it
is a property of the **score at dx2's operating point**. DERIVED from the score definition and dx2's
MEASURED `d_pose = 6.37e-6`:

```
∂S/∂d_pose = 5/√(10·d_pose) = 5/√(6.37e-5) = 626.5    per unit d_pose
∂S/∂d_seg  = 100                                       (constant)
pose is 6.26× more marginally sensitive than seg
one unit of d_pose costs 626.5/λ_B = 940,844,854 bytes of rate
```

Budget: to consume the **entire** remaining gap (0.028220 S) on pose alone,

```
d_pose may rise by at most 1.246829e-04   (6.37e-6 → 1.310529e-04)
```

MEASURED rises: **W72 +1.9364 = 15,531× that entire budget. bo2 +1.5282 = 12,257×.**

This is CLAUDE.md's own operating-point table extended to the live point — the crossover it records is
`pose_avg ≈ 2.5e-4`, and dx2 sits **39× below** it. tv2's √-saturation law is real and MAIN's warning
about it is exactly right: *"the √ saturates the RATE of growth; it does nothing against a
four-order-of-magnitude blowup."*

**So the answer to "what predicts the ratio" is: nothing about the mechanism does. The score does.**
Any mechanism that perturbs the rendered frames is priced against a pose budget of `1.25e-4`, and every
measured mechanism overran it by 4 orders of magnitude. Mechanism identity then only decides *which*
4-order-of-magnitude overrun you get — which is why the ratios cluster in one decade and correlate with
nothing but size.

That is a stronger and more transferable statement than any classification, and it is the one I would
carry forward. It also **confirms MAIN's parallel arm's premise**, from a different direction — §7.

---

## 5. `df1` ↔ `hc1`: the identification is right and the inference is inverted

MAIN's amendment called this *"the strongest evidence the law will get"* and asked me to verify or
refute. **Verified numerically; refuted as strategy.**

### 5.1 They are the same object (VERIFIED)

```
df1 §4 address floor  = Σᵢ H_b(pmax_i) = 872,907.97 bits = 109,113.50 B
hc1 §2 INDICATOR      = 111,275.62 B   (= 34,674.08 "yes" + 76,601.54 "no")
agreement                                = 1.98%
```

Both are the **binary indicator's information content** — df1 computes its *entropy*; hc1 measures its
*realized code length* in the shipped stream. Realized exceeds entropy because the model is
miscalibrated, which is exactly what hc1 went to measure. **MAIN's identification is correct**, and
the 1.98% gap is itself meaningful: it is the calibration slack, and hc1 measured that recovering it
perfectly yields **8.44 heldout net B**.

One bookkeeping correction: MAIN wrote *"subtracting df1's zero-mode (34,674 B) leaves ~76,600 B."*
The subtraction is already done — hc1's table shows `34,674.08 + 76,601.54 = 111,275.62`, so 76,601.54
is the **"no" branch inside the indicator**, not a residue after removing something foreign.

### 5.2 Why "the address tax is 181% of demand" licenses a false hope (REFUTED)

MAIN's figure is right on its own terms — `111,275.62 / 42,382 = 262.6%` for the whole indicator,
181% for the "no" branch. What the framing hides is a **status difference between two things that
share a value**:

| | as an ADDRESS (df1's framing) | as a RESIDUAL (hc1's framing) |
|---|---|---|
| what it is | overhead paid **on top of** the payload, to enable selective treatment | **the payload itself** — information the decoder does not have |
| can you avoid it? | **Yes** — don't be selective. This is why df1 closed D1. | **No.** Only a better model (lower entropy) or accepted distortion. |
| what address-freedom does to it | removes it | **nothing** |

**The two framings coincide in value because they are the same entropy, and diverge completely in what
you may conclude.** df1's 3.1468× is a proof that *a hypothetical selective mechanism cannot pay for
itself* — the overhead exceeds the prize. It is **not** a term in the price of any mechanism we
actually built, because (§3) we never built a selective one on this axis. tba1 priced that road at
**D1 = +9.45 B** and closed it.

So: **address-freedom cannot release the address tax, because on this object the "address tax" and the
"payload" are the same bits.** That is the sharpest form of MAIN's own self-correction (*"I aimed you
at a term worth 3.15× in a problem whose losses run 21–686×"*) — and it is stronger, because it
explains not just that the term was small but why removing it recovers nothing.

**The 3.1468× and the 21.62×–922× are not commensurable and should never again be placed in one
sentence as if they were rungs of one scale.** One is a lossless-overhead ratio; the others are
lossy-exchange ratios. Same genus as §2's three conventions.

**Where this does leave a live axis:** if the indicator block is a residual, then the only way to shrink
it is to lower its entropy — i.e. **a better probability model**. That is `[[probability-model-axis-live-fx1-sweep-prior]]`,
and hc1 explicitly left it open while closing the *calibration* sub-axis at 8.44 B. Calibration is not
the model; hc1 fixed the model's confidence, not its conditioning.

---

## 6. The merge family (D3) — inventory

`tba1`-D3 — *alphabet reduction in the model: retrain on 4 symbols after a class merge* — is the
operator's "merging" family and **the single largest open ceiling on tba1's map: 38,649.8 B = 91.19%
of the 42,382 B demand.** Its warrant is measured: **Lane holds 33.97% of token bits at 0.59% of area**
(`[[lane_is_0.59pct_area_and_33.56pct_of_model_bits_20260822]]` records 33.56%/38,183 B on the model
axis; tba1's 33.97% is the token-bit share — two adjacent measurements of the same concentration).

**Vehicle status — the finding MAIN asked me to confirm or refute. `#917`'s complaint is CONFIRMED,
and source-verification corrected my own first answer.** I initially inferred that `region_merge`
stores a label map and is therefore address-paying. **That inference was wrong**, and the truth is
more useful. Source-verified inventory (paths and line numbers as traced):

| surface | what it merges | side-info the decoder needs | class (§3) | vehicle |
|---|---|---|---|---|
| `src/tac/boundary_math/region_merge.py:104` `solve_mdl_region_merge` | **RAG regions** — 4-connected components of the argmax map (`partition.py:118`), contracted into the largest neighbour (`:90-101`) | **none for the merge** — the *result* ships as a full dense LZMA2 label raster (`dense_raster_lzma_baseline.py:55-74`), so no merge tree, label map, or permutation is transmitted | **FREE** | **NOT in the shipping body.** Sole caller `seg_core.py:179`; of ~40 `seg_core` importers, **zero** import it |
| `src/tac/optimization/packet_member_merge_receiver.py:98/159/379` | **ZIP archive members** | **STORED, self-describing member table** — names + offsets + lengths, 3 wire magics (`:700-712`) | **PAYING** | alive infra; **INAPPLICABLE** — see below |
| **`tba1`-D3 (unbuilt)** | **classes in the model's alphabet** (5→4, then retrain) | **nothing** — a global alphabet | **FREE** | dx2 token body; **the live one** |

Four consequences, the third being the one that matters most:

1. **`region_merge`'s real disqualifier is not addressing — it is that the merge DECISION reads the
   GT oracle.** `region_merge.py:164-167` computes `flips_fixed` from `gt_argmax_hw` = `L*`, the scored
   target, and gates on `pays_rent = flips_fixed > 0 and marginal_bytes < flips_fixed * water_level`
   (`:183`). That is **encoder-only by construction**: the decoder never reconstructs the merge, it
   just decompresses the post-merge raster. So it is address-free *and* unusable as a decoder-side
   mechanism, for an unrelated reason. It also ships a **dense raster**, a different rate model from
   dx2's token stream entirely.

2. **`packet_member_merge_receiver` is byte-closed and genuinely address-paying, and it is blocked on
   this body by a one-line structural fact:** the shipping archive has **exactly one member**
   (`submissions/robust_current/jg5_sub015_runtime/runtime/archive.zip`, single entry `p`,
   `ZIP_STORED`), so `frontier_rate_attack_bootstrap.py:449-450` raises
   `packet_member_merge_requires_at_least_two_members` and no candidate is produced. The mechanism is
   fine; the object has nothing to merge.

3. **The inventory hands §0 its confirmation, with real built artifacts rather than an argument.**
   Every address-PAYING merge in the tree stores a table, and **every one of them is LOSSLESS**:
   `packet_member_merge_receiver` (member table, sha-verified per member at `:453-496`);
   `pr103_arithmetic_coding.py:199/260` merged range stream (source states the decoder *requires*
   `tensor_symbol_counts` because "the range-coded stream itself does not carry boundaries",
   `:126-129`); `hnerv_pr103_lc_ac_schema.py:234/313` (hardcoded `AC_STREAM_SPECS` layout);
   `pr101_split_brotli_codec_derivers.py:135` (storage permutation baked into `DECODER_STORAGE_ORDER`,
   `pr101_split_brotli_codec.py:76-79`, whose source warns the decoder "MUST receive the SAME
   values"). **Four built address-paying mechanisms, four stored tables, zero distortion, zero
   exchange ratios — and all four on retired vehicles.** This is §0's claim as artifacts:
   address-paying ⟺ names a subset in order to code it exactly ⟺ lossless ⟺ off the ladder.

4. **No implementation of D3 exists anywhere in the tree.** `experiments/ddm_tri1_triple_composition.py:461`
   records R2 = "alphabet collapse (class merge to 4 symbols) + model refit = tba1 D3" but its own
   header (`:3-5`) states *"This script MEASURES NOTHING"*; and the cited derivation
   `experiments/ddm_tba1_bulk_ceiling_derivation.py:231` only **refuses** alphabets >5 — it collapses
   nothing. So D3's blocker is a **retrain**, not code, and tba1's pre-registered falsifier stands: a
   model retrained on a reduced-alphabet field must (a) beat 113,777 B on a real re-encode by >0 B with
   its own model cost counted, and (b) have `d_seg` measured on the authority lane. Clause (b) is what
   §4.4 prices — a class merge **changes the decoded field**, so it is lossy and faces the same
   `1.25e-4` pose budget as every rung on the ladder.

**Reaching for either built surface as "the merge code we already have" would be the
`[[measured_object_vs_named_object_20260816]]` error** — `region_merge` merges regions under a GT
oracle and ships a raster; `packet_member_merge` merges container members. Neither can touch an
alphabet.

**Also verified and worth recording:** the quotient surfaces are *not* merges.
`whole_teacher_distilled_student.py:339` `quotient4_from_logits5` is a fixed Helmert-basis 5→4
projection — fully decoder-derivable with **zero** stored side-info, hence the purest address-free
object in the tree — but it quotients the **softmax shift symmetry**, not classes. It does not reduce
the alphabet and is not D3.

⚠ One item the inventory could not settle: whether `packet_member_merge` was ever applied to a
*historical* multi-member archive (only the current `jg5_sub015_runtime` archive was enumerated). Not
load-bearing here.

**`ad2` and `g4`, which MAIN offered as regime evidence, belong here as the lossless counterpart:**
`ad2`'s one real win is **17,957 B, tile-major time QPAIR reordering — but on the `NR1 K32` packet,
not dx2.** On **DX2** ad2 measured *"fixed-representation coder work: 0 B — CLOSED by RB1; not
reopened."* That distinction is load-bearing (`[[the-borrowed-number]]`): the 17,957 B is a
**different-vehicle** number and is a hypothesis on ours.

---

## 7. Prices, and the resource the class converts

Demand, CITED from `ddm_tx1_toolbox_crosswalk_20260819.md` §0 and not re-derived:
**gap to 0.12 = 0.028220 ⇒ 42,382 B at fixed distortion, or 150 B at zero distortion;
λ_B = 6.658590e-07 S/B.** Archive census from `ddm_ar1b`: token 113,777 · renderer 30,856 · carrier
22,010 · HPAC model 13,515 · **framing 210** = 180,368 B, **zero remainder** (ar1b §"Exact physical
census": `31+14+13,515+30,856+22,010+96+47+22 = 66,591`, `+113,777 = 180,368`). The four-term
shorthand in circulation omits the 210 B of framing; ar1b's full census is exact, as claimed.

| candidate | class (§3) | ceiling / measured | % of 42,382 B demand | distortion status |
|---|---|---:|---:|---|
| **`tba1`-D3 alphabet merge** | FREE | **38,649.8 B** (ceiling) | **91.19%** | **lossy — unpriced, faces the 1.25e-4 pose budget** |
| `ad2` QPAIR reorder (NR1) | PAYING→lossless | 17,957 B | 42.37% | **zero** — but **different vehicle** |
| `hc1` recalibration | FREE | **8.44 B** (measured, heldout net) | 0.0199% | **zero** — FAMILY CLOSED |
| `ad2` fixed-repr. coder work (DX2) | — | **0 B** | 0% | zero — CLOSED by RB1 |
| `tba1`-D1 explicit selection | **PAYING** | +9.45 B | 0.022% | zero — closed by arithmetic |
| `tba1`-D6 free selectors / regime | FREE | budget spent (oe1, cx3 lost) | — | zero |
| `film_amortized_flat_w96` | FREE | 1,078 B | 2.544% | unmeasured (w72 §4) |

**Decode-time compute — the resource MAIN flagged as unspent.** MEASURED by `ddm_dc1`
(`[contest-CUDA T4 n600]`): token decode **478.65 s**, neural render+resize **55.22 s**, full inflate
wrapper **558.63 s**, evaluator **51.90 s**, contest-auth-eval wrapper **622.40 s**, Modal call
**629.12 s**. dc1's own headroom statement: *"the gross arithmetic against 1,800 seconds is 1,170.88 s
spare after the Modal call"*, with a **governed** endpoint of **691.47 s**.

⚠ **This does not match the figures in MAIN's amendment** (*"contest-CPU decode 831.5 s of 1,800 s
(2.17×); T4 1,471 s (328.7 s spare)"*). dc1's T4 total is **629.12 s with 1,170.88 s spare**, not
1,471 s with 328.7 s spare. I did not resolve which run MAIN's numbers come from — they may be a
different rung or a CPU/T4 mix-up. **The direction is the same and is not in dispute: real headroom
exists, and dc1 measures it at 691–1,171 s on T4.** I flag the discrepancy rather than average them
(`[[available-field-vs-authoritative-field]]`).

**Per-candidate decode cost against that headroom:** D3's alphabet merge changes the *symbol count*,
not the decode algorithm — a 4-symbol arithmetic decode is **cheaper** than a 5-symbol one, so D3
spends **no** decode budget and is not at risk from the 30-min wall. `ad2`'s reordering is a permutation
applied at decode: **O(n) over 1.84M pairs, negligible.** The family that *would* spend the budget is
implicit/`dc1` (decode-time solving), and dc1's own successor row is explicitly gated on *"native wall
exceeds cold bar"* — i.e. dc1 already carries the gate. **No candidate in this memo is budget-limited.**
The headroom is real and unspent, and **nothing here converts it**, because every surviving candidate
is cheap at decode. That is worth saying plainly: the unspent resource is not the binding constraint
either.

---

## 8. Corrections to the charter

The charter said corrections are the most valuable return. Six, in descending order.

1. **"Address-paying vs address-free" does not cut the measured population — it is 6–0, not 5–1.** The
   charter (and MAIN's first amendment) classified W72 and tba1-D3 as address-paying; by the §3
   criterion both are free. There was never a deciding side to be short of.
2. **The deeper reason is that the two classes are not commensurable.** Address-paying ⇒ lossless ⇒ no
   distortion numerator ⇒ no exchange ratio. The ladder is defined only on the lossy side. This is a
   category error, not a sample-size problem, and no amount of new measurement on the current axis
   fixes it.
3. **The ladder handed to me holds three conventions.** W72's 35.5364× is an S-ratio (true: 922×);
   ni1's 247.69× is seg÷seg-ceiling (true: 714×). Both were published correctly by their own memos and
   mis-imported by the ladder. `ds1` §6 flagged this and was not followed up.
4. **"21.62×, the best exchange ratio ever measured" is none of those three words.** It is derived, not
   measured; seg-only, so not an exchange ratio in the ladder's sense; and best only because pose is
   absent from its numerator. **The true floor over like-for-like rungs is 33.7×**, and that is itself
   a lower bound on a static credit.
5. **MAIN's pose-share covariate is an artifact of correction (3).** Fixing W72's single cell takes
   `corr(pose_share, log ratio)` from an eyeballed monotone trend to **−0.155**.
6. **The `hc1`/`df1` identification is right; the strategic reading is inverted.** 111 KB is the
   payload, not overhead. Address-freedom recovers none of it.

**The better cut, if one is wanted.** Not addressing but **whether the decoded field changes**:

| | field unchanged (lossless) | field changed (lossy) |
|---|---|---|
| distortion | **exactly 0** | 65–93% of it in pose |
| ratio | **undefined** (0 ÷ credit) | 33.7× – 922×, n=5, no exceptions |
| members | ae1, ad2, hc1, tba1-D1/D6 | the whole ladder |
| binding limit | model entropy: **111,275.62 B** indicator block | pose budget: **Δd_pose ≤ 1.25e-4** |

This cut separates the measured population **perfectly**, at n=5 vs n≥4, with no counterexample. I
flag its weakness myself: **it is close to tautological** (lossless ⇒ zero distortion), and its whole
value is that it names the *right two questions* — "how low can a lossless code go?" (answer: bounded
by the 111 KB indicator entropy, attacked by the model axis) and "can anything survive a `1.25e-4`
pose budget?" (answer: nothing measured has come within 4 orders of magnitude). A tautology that
replaces a fitted covariate with two measurable quantities is worth having; a tautology presented as a
discovery is not, and I am not presenting it as one.

**MAIN's parallel lossless arm: premise CONFIRMED, do not pull it.** MAIN asked to be told loudly if my
analysis contradicts it. It does not — §4.4 reaches the same place from the score side: zero frame
change ⇒ zero pose damage ⇒ the only position not facing the `1.25e-4` budget. Two cautions to carry
into it: (a) the dx2-specific lossless evidence is **8.44 B (hc1) and 0 B (ad2/RB1)** — the encouraging
**17,957 B is NR1's, a different vehicle**; (b) the ceiling on that axis is the **111,275.62 B indicator
entropy**, and calibration is already closed at 8.44 B, so the arm must attack **conditioning**, not
confidence.

---

## 9. The single cheapest measurement that would decide it

**Not a test of the address-free law.** The law is undefined on this evidence (§0, §3); paying to test
a constant buys nothing. The cheapest decisive purchase is the one that resolves the *live* fork §4.4
opens, and it is:

> **Re-encode the dx2 token field under a merged alphabet (Lane folded into its dominant neighbour,
> 5→4 symbols) and measure (a) the real re-encoded byte count with the retrained model's own cost
> counted, and (b) `d_seg` and `d_pose` on the authority lane.**

Why this one:

- It is the **only untested direction whose ceiling approaches demand** (38,649.8 B = 91.19%) —
  tba1's own conclusion, independently reached.
- It is the operator's **"merging"** family, and §6 shows **no built surface implements it**, so its
  status is genuinely unknown rather than inherited.
- It resolves **both** open questions at once: (a) gives a *real re-encode* number, which is the only
  thing that can discharge the static-accounting flag (`ds1`'s 14.59× misprice); (b) gives D3 the
  **pose measurement it has never had**, which is the missing cell that made the whole ladder's
  ordering an artifact (§2.4, §4.1).
- It is **cheap on the axis that matters**: no new representation, no new receiver, no decode-budget
  risk (§7 — a 4-symbol decode is cheaper than 5). The cost is one retrain plus one authority row.

**Pre-registered falsifier, taken from tba1 verbatim so it is not mine to soften:** the retrained model
must **(a)** beat 113,777 B on a real re-encode by >0 B with its own model cost counted, **and (b)**
have `d_seg` measured on the authority lane. I add the term §4.4 makes binding and tba1 could not have
priced: **(c) `Δd_pose` must be measured, and the whole-gap budget is `1.25e-4`.** If D3 changes the
field at all, (c) is the clause that will decide it, and no prior rung has come within 10⁴ of passing
it.

**Cheaper still, and worth firing first if only one thing fires:** MAIN's lossless-model arm, because
it is the **only position that does not face (c) at all**, and its ceiling — the 111,275.62 B indicator
block — is 2.63× the entire demand.

---

## 10. What I did not do

- **I ran no measurement.** Every number here is arithmetic on banked receipts. The only computations I
  performed are the recomputations in §2, §4, §5 and the correlations in §4.2 — all reproducible from
  the cited memos with the formulas shown inline.
- **§6 is now source-verified, and it corrected me.** My first draft INFERRED that `region_merge.py`
  stores a label map and is address-paying. Source inspection showed the opposite — no merge side-info
  is transmitted at all (the post-merge raster is) — and that its real disqualifier is the GT-oracle
  read at `:164-167`. **I record this because the inference was wrong in exactly the way the charter
  warned about: I classified from what the mechanism's NAME implies it must do, rather than from the
  code.** The §3 criterion is sound; my application of it without reading the file was not.
- **I did not re-derive λ_B or the demand** — CITED from `tx1` §0 per charter instruction.
- **I did not resolve the decode-headroom discrepancy** between dc1's measured T4 figures and MAIN's
  amendment (§7). I reported both and used dc1's.
- **The correlations are n=5.** I report them to three decimals because that is what the arithmetic
  gives, **not** because n=5 supports three decimals. No p-value is quoted and none should be inferred;
  the claim `−0.155 ≈ 0` is a statement that the hypothesis has no support here, not that the true
  correlation is zero.
- **I did not verify tolerance's or dg2's raw scorer rows** — taken as published by tv2 and dg2. The
  static-credit flag on tolerance is tv2's own and travels with the number.
- **I did not measure the noise floor on `r`.** `ds1` established that nobody on this campaign has, and
  that every "within noise" claim about the exchange ratio is currently unlicensed. **Mine included:**
  my `−0.155` and my ordering changes are unprotected by any measured floor. The ordering changes I
  claim (26× on W72, 2.9× on ni1) are far larger than any plausible floor; the correlation is not.
- **`tba1`-D3's pose share cannot be filled** without building D3. I state that rather than estimating
  it; the ≈62× in §2.4 is explicitly labeled not-a-prediction.
- **I did not test whether the §8 "field changed?" cut predicts anything *within* the lossy side.** It
  does not — that is §4.2's whole result — and I make no claim that it orders the ladder.

---

## Own-vehicle frontier

**dx2 — S 0.14821987563243377 @ 180,368 B `[contest-CUDA T4, n600]`, archive sha `976f706d…` —
UNMOVED by this arm.** Gap to 0.12 = 0.028220 ⇒ 42,382 B at fixed distortion, or 150 B at zero
distortion. This memo bought no bytes and no distortion; it corrected a comparison and named the next
purchase.
