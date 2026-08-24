# ddm_ma2 — the two readings are one object, and both halves of the mechanism were closed before this charter was written

`date_utc: 2026-08-24` · `arm: ddm_ma2_merged_alphabet_lane_fold` ·
`axis: [macOS-CPU advisory / scorer-free arithmetic over retained coder-cost fields]` ·
`score_claim: false` · `promotion_eligible: false` · `pointer_moved: false` ·
`verdict_scope: INSTANCE:DX2_archive_976f706d_n600` for the measured reconciliation;
**the closure is inherited, not produced here** — its scope is the scope of `ddm_tri1` §3.4
(FORMULATION, the `tokens × HPAC` R2 cell) and `ddm_ws1` (FAMILY, the registered
explicit-worldsheet reference form). Cost: $0. No scorer, Modal, Metal, encoder, or
`upstream/` write.

`STORES CONSULTED:` `ddm_tri1_triple_composition_and_pair_closure_20260824.md` §3.3–3.5 ·
`ddm_ws1_optimal_worldsheet_grammar_20260821.md` · `ddm_ws0_worldsheet_grammar_price_20260821.md` ·
`ddm_ig1_implicit_carriage_gestalt_20260821.md` §"Routing table for NR1" ·
`ddm_af1_address_free_class_law_20260824.md` §2.4/§6/§9 · `ddm_tba1_token_bit_attribution_20260823.md`
§2/§6 · `ddm_ld1_lane_lossy_drop_exchange_20260822.md` · `ddm_jf1_joint_field_model_refit_20260823.md` ·
`ddm_dg2_diagonal_distortion_verdict_20260824.md` · `ddm_tx1_toolbox_crosswalk_20260819.md` §0 (CITED) ·
`.omx/state/main_hot_state.md` · dx2 runtime source at
`/Volumes/APDataStore/pact/ddm_dx2/r7/candidate_runtime_dx2/runtime/` · `.omx/state/` ledgers.

---

## 0. Answer first

**STEP 0 — the object.** The charter's 20× disambiguation is a **false dichotomy on this
vehicle**. The dx2 token stream codes **one symbol per pixel**, and its five symbols **are** the
five SegNet classes. "Coding alphabet" and "output class partition" are the same object. There is
no second reading to choose. Source, not inference: `residual_archive.py:42 NUM_CLASSES = 5`;
`f26_hpac_native.c:25 #define F26_ALPHABET 5`; the field is 600 × 384 × 512 uint8 with values 0–4;
`ld1`:65 names the symbols "class 1 (Lane)" and "class 0 (Road)" and joins them positionally to the
DALI GT argmax.

**And the charter's own reading-(a) arithmetic is the correct price for both readings.** The charter
DERIVED "≈19× the entire 0.028220 gap." `ddm_tri1` §3.4 had already MEASURED-transferred it:
**0.556381 S = 19.72× the gap = 21.62× the byte credit.** The charter's estimate is right to 1.4%.
The charter proposed to spend a slot resolving an ambiguity that does not exist, on a number it had
already computed correctly.

**RECALL GATE — the arm is CLOSED, on both halves, by work that landed before this charter.**

| half of the mechanism | status | closing evidence | date |
|---|---|---|---|
| **M1** — fold Lane out, do **not** regenerate | **CLOSED** | `ddm_tri1` §3.4: **≥21.62×** over break-even **on the seg leg alone**, before one unit of pose | 2026-08-24 |
| **M2** — fold Lane out **and regenerate it from a geometric curve** (the charter's hypothesis) | **CLOSED at OPTIMAL FORM** | `ddm_ws1`: the openpilot lane-polynomial receiver was **BUILT, RACED, and LOST**. `DEAD`. | 2026-08-21 |

`ld1` did **not** close it — it edited symbol values under a **fixed** model. `jf1` did **not**
close it — 5 symbols, epoch 2 of 60. The charter was right about both. It missed `tri1` and `ws1`.

**The three-part falsifier (a)+(b)+(c) cannot be run, and does not need to be: M2 dies on bar (a),
the RATE bar, before distortion is consulted.** Re-supplying Lane explicitly is measured three
independent ways, and every one costs more than the fold saves:

| explicit re-supply of Lane | MEASURED B | source | × the 38,649.8 B credit | net |
|---|---:|---|---:|---:|
| ws0 boundary coords, Road↔Lane | **144,092** | `ws0`:41 | **3.728×** | **+105,442 B = 2.49× demand** |
| ws0, all four Lane strata | 146,935 | `ws0`:41,63–65 | 3.802× | +108,285 B = 2.55× demand |
| g1 row-run, Lane class | 180,701 | `ws0`:89 | 4.675× | +142,051 B = 3.35× demand |
| **ws1 LBND2 openpilot polynomial** | **534,763** | `ws1`:59 | **13.836×** | +496,113 B = 11.71× demand |

**`Δd_pose` against the 1.25e-4 budget: NOT MEASURED, and not owed.** Bar (c) is unreachable — the
exclusive n600 scorer lane is not this arm's (`jf1`:101 records it with AP1/MAIN, and both `ld1` and
`jf1` queued their scorer joins for the same reason). It is also moot: M1 refuses on seg alone at
21.62×, and M2 refuses on rate at +105,442 B. Neither reaches a pose measurement. I record the
budget unmet rather than estimating it.

**The pointer did not move.** This arm bought no bytes and no distortion. It produced one exact
reconciliation, corrected two committed memos, and spent $0 confirming a closure it did not need to
build.

---

## 1. STEP 0, resolved at source: the two readings collapse

The charter asked which object "Lane folded, 5→4" names, and warned the two readings differ by 20×.
On this vehicle they are the same object, and the source says so in three places.

| surface | evidence |
|---|---|
| `runtime/residual_archive.py:42` | `NUM_CLASSES = 5`, with `:45 FIXED_STATES = 25` (= 5², the previous-class × predicted-class context radix) |
| `runtime/f26_hpac_native.c:25` | `#define F26_ALPHABET 5`, used at 29 sites: RC64 frequency/cumfreq loops, decode symbol search, logits buffer |
| `runtime/f26_hpac_native.py:159` | `if channels != 64 or int(model.num_classes) != 5:` — a hard refusal of any other alphabet |

The coded symbol *is* the class label. Reducing the alphabet to 4 **is** merging the partition.
There is no separate "coder alphabet" that could be shrunk while the output partition stays at five.

**The one real subtlety the charter identified correctly, and it does not rescue the mechanism.**
The token field is an **intermediate** consumed by the renderer, not the scored argmax. The frozen
SegNet re-derives its argmax from the rendered RGB. So a token error only becomes a score error if it
*survives* the render. `ld1`:47–52 measures the survival of **naturally occurring** errors at
**24.66%** body-wide and **29.94%** for GT-Lane.

**I nearly transferred those ratios, and they are the wrong ones.** Applying 29.94% to the charter's
arithmetic gives 0.142 S and understates the damage **3.9×**. `ld1`:56–58 explicitly forbids the
transfer: those ratios "describe naturally occurring baseline errors, not these deliberate high-cost
Lane→Road edits." The correct rate for **deliberate** edits is `dg2`'s MEASURED **0.9528
final-flips-per-edit**, which is what `tri1` used. Near-100% survival, not 30%. I record the near-miss
because it is precisely the `[[the-borrowed-number]]` genus, and the guard that caught it was reading
`ld1`'s own refusal clause rather than its table.

---

## 2. The reconciliation the charter asked for — MEASURED, and they are different index sets

The charter asked me to reconcile 38,649.8 B against 38,183 B "or say they are different objects."
**They are different objects.** Both are sums over the *same* retained per-position cost field; they
differ in *which positions* they sum.

- `tba1`:74 sums over positions whose **DECODED TOKEN** is Lane.
- `ld1`:44 sums over positions whose **DALI GT ARGMAX** is Lane.

`tba1`:84–86 notes the axis difference in prose but never quantifies it. This arm did.

MEASURED, `experiments/ddm_ma2_lane_index_set_reconciliation.py`, custody-verified inputs
(cost `99d7833d…`, tokens `cc10a7b0…`, DALI GT `91d3ff11…`, all three digests matched):

| published | value | reproduced here | abs error |
|---|---:|---:|---:|
| `tba1` Lane bytes (decoded axis) | 38,649.8 | **38,649.81741402853** | 0.017 B |
| `ld1` Lane bytes (GT axis) | 38,182.996184 | **38,182.99618413268** | **1.33e-07 B** |
| `ld1`:62 both-Lane positions | 688,847 | **688,847** | **exact** |

The run also reproduced **all five rows** of `tba1` §2 on an independent code path — Road 44,297.0 ·
Lane 38,649.8 · Undrivable 12,893.7 · Movable 11,876.8 · MyCar 6,058.8 B, with areas 23.2331% /
0.5858% / 49.5175% / 1.2380% / 25.4255% — and its total attributed bits match `tba1` §1's
910,209.280609 to **6.2e-08 bits**. That is a third independent computation of that table, and it is
what licenses using the per-class figures below.

The 2×2 Lane contingency, which is the object the reconciliation needs:

| cell | positions | bytes | B / position | × the both-Lane rate |
|---|---:|---:|---:|---:|
| decoded Lane **and** GT Lane | 688,847 | 37,408.233048 | 0.0543056 | 1.000× |
| decoded Lane, GT **not** Lane | 2,248 | 1,241.584366 | 0.5523062 | **10.170×** |
| GT Lane, decoded **not** Lane | 1,907 | 774.763136 | 0.4062733 | 7.481× |

The identity closes exactly:

```
38,649.817 − 38,182.996 = 466.804 B          (the published difference)
 1,241.584 −    774.763 = 466.821 B          (the disagreement asymmetry)
identity residual                 = −1.17e−11 B
```

The 1,907 GT-Lane-only positions reproduce `ld1`'s Lane representation-error denominator exactly
(its 571/1,907 survival row). The match on 688,847 is the **order-sensitive** validator: a wrong
raster join would not land on the published intersection.

**Correction to `ddm_af1` §6.** It calls these "two adjacent measurements of the same concentration."
They are not measurements of one concentration; they are the costs of two index sets that differ on
4,155 positions. The distinction matters because the **credit** a merge could claim is the
**decoded-axis** figure (those are the symbols that stop being coded), while the **damage** is
counted on the **GT axis** (those are the labels that become wrong). Using one number for both legs
silently moves 466.8 B across the exchange.

**New micro-finding, offered as such:** the positions where the model and the GT *disagree* about
Lane cost **10.17×** more per position than the positions where they agree. The coder pays most
where it is wrong. This is the `wj1`/`tba1` §4 cost-and-error co-location pattern appearing on the
Lane class specifically. `verdict_scope: INSTANCE`. It is a description of the shipped stream, not a
lever — naming that set is exactly what `tba1` §5 closed at +9.45 B.

---

## 3. Why M2 is closed: the conservation argument, with a measured price

`ddm_tri1` §3.4 states the general form, and it is the sharpest sentence in the corpus on this
question:

> **D3's byte credit *is* the lane information. Any leg that repairs its distortion must re-supply
> that information, and re-supplying it returns the bytes. A lossy token leg composes with nothing.**

The charter's hypothesis is exactly the repair leg that argument forbids: fold Lane out, then
re-supply it from a geometric curve. `tri1` predicted the outcome structurally. **`ws1` had already
measured it three days earlier, at optimal form, on the real n600 object.**

### What ws1 actually built and raced

`ws1`:104–106 — "Parametric lane: the existing **`LBND2` coherent-slot/openpilot polynomial
receiver**, not raster-run coding. Its decoded polynomial raster is followed by counted exact curve
exceptions." That is the charter's mechanism, with the charter's named geometry source, with the
free rasterizer at decode and only the coefficients counted.

`ws1`:59, the Road↔Lane stratum:

> `DEAD`: LBND2 polynomial **534,763 / 544,331** innovation+model B lost to generic
> **513,011 / 495,603** B

The lane polynomial lost **its own stratum** to a generic coder. It did not merely fail to reach the
demand; it failed to beat the thing it was replacing.

### Why it lost — the held-out diagnostic

`ws1`:79–86 runs a **real spatial holdout**, not an in-sample prefix: every n600 lane is fit on even
scorer rows and tested on odd rows, across **345,200 held-out pixels**.

| quantity | MEASURED |
|---|---:|
| median absolute lateral residual | **0.04340 m** |
| p95 lateral residual | 0.26105 m |
| **decoded LBND2 lane mask coverage of source lane pixels** | **mean 0.54494** |

The geometry fits. **The mask does not.** A 4.3 cm median lateral residual still recovers only
**54.494%** of the lane pixels the scorer counts, because the scored object is not the centerline —
it is the **dashed, thin, high-frequency pixel set** whose measured temporal IoU is **0.2526**
(`tba1`:74), the lowest of all five classes by 3.4×. A smooth curve does not determine dash phase,
dash length, gap, or width. Those must be transmitted, and transmitting them is where the 38,650 B
lives.

This is the mechanism behind the charter's own stated risk. The charter wrote: *"A geometric prior
that fits Lane well enough to replace 38,650 B of learned representation is a strong claim."* It is,
and it was measured false — at 0.0434 m of fit and 54% of coverage.

### The routing table pre-registered this exact fork

`ddm_ig1`:183, the routing table for NR1, written **before** ws1 landed:

> `Lane / Road↔Lane` | **IMPLICIT pending WS1; HYBRID if WS1 wins its stratum**

WS1 lost its stratum. The routing resolves to **IMPLICIT** — keep Lane in the learned coded field.
The fork was declared in advance, the measurement fired, and it resolved against the charter's
hypothesis.

**Vocabulary inversion — flagged, because it is how this arm could have been mis-chartered.** The
charter uses "implicit" to mean *regenerated at decode from a curve*. In `ig1`'s measured routing
vocabulary that is **OPTIMAL-GRAMMAR**, and `ig1`'s **IMPLICIT** means the opposite: *stay in the
learned coded field*. Under the corpus's own vocabulary the charter proposed the route `ws1`
measured dead, using the word for the route `ws1` confirmed. `ig1`:16 also records that
"**implicit beats explicit** as a universal law is **refuted**" — the defensible rule is the
per-stream routing table, and Lane's row is now measured.

### Honest limit on the price comparison

`ws0`/`ws1` price a **complete lossless boundary representation of the partition**, not a drop-in
Lane-token replacement. The 144,092 B and 538,668 B are therefore directionally right but not
strictly apples-to-apples with a 38,650 B token-class cost. Two things keep the conclusion standing:
`g1`'s **180,701 B** row-run price (`ws0`:89) is a **Lane-only** object and is still **4.675×** the
credit; and the LBND2 row is a like-for-like race **within** ws1's own stratum, where the polynomial
lost to the generic coder it was meant to beat. Every explicit form measured on this object costs
more than Lane's tokens cost.

---

## 4. What a build would have cost, had the recall gate not closed it

The charter framed this as "one retrain plus one authority row" (inherited from `af1` §9, "cheap on
the axis that matters"). Source says otherwise. `NUM_CLASSES` / `F26_ALPHABET` is a **hardcoded
constant in two languages across the dx2 runtime**, not a parameter:

| file | sites |
|---|---:|
| `f26_corrector_native.c` | 64 |
| `f26_hpac_native.c` | 29 |
| `fx2_model_axis_corrector.py` | 20 |
| `fx1_logistic_mixer_corrector.py` | 17 |
| `free_corrector.py` / `residual_archive.py` / `native_free_corrector.py` / `rr4_free_corrector.py` / `f26_inflate.py` / `ihs2.py` | 8 / 6 / 5 / 3 / 1 / 1 |

≈**154 sites across 10 files**, plus the encoder-side mirror at
`experiments/ddm_jg2_tail_reencode.py:154`. `FIXED_STATES` drops 25 → 16, changing the packed
fixed-table wire layout. `fx1_logistic_mixer_corrector.py:414–456` builds ~15 mixer context-table
dimensions as products of `NUM_CLASSES`, so every table resizes. `f26_hpac_native.py:159` refuses
outright. And the fit is `jf1`'s sealed **60-epoch** reference schedule under a 48-hour wall cap —
`jf1`'s own epoch-2 instrument was **7,554 stream bytes behind the shipped fit on the unmodified
field**, so a matched comparison needs the terminal fit, not a short one.

That is a multi-day build in two languages, not an hour. It would have been spent on a cell already
refused at 21.62× on its seg leg alone.

**One datum worth banking from `jf1`, because it is the only measured estimate of what a refit
recovers.** `jf1`:94 records that k060000's refit stream is **4,672 B below** the matched epoch-2
null refit stream — the merged field is genuinely easier for the same model. Against the
**27,319.051 B** of incumbent cost that rung moved (`ld1`:80), that is a **17.10%** realization
fraction. Extrapolated to Lane's full 38,649.8 B it gives **≈6,609 B**, or **15.6% of demand** —
not 91.19%. The extrapolation is weak (epoch 2 of 60; a within-run difference, not an absolute win;
5 symbols, not 4; 8.68% of Lane, not all of it), and a terminal fit should realize *more*. I state
it as a **bracket**: the realizable credit sits somewhere in **[6,609 B, 38,650 B]**, a 5.8× band.
**The published 38,649.8 B is a ceiling that assumes Lane codes for exactly zero and the merged
positions cost nothing. Both are false.** No arm should quote it as the credit.

---

## 5. Corrections to the charter — the most valuable return

1. **The 20× disambiguation is a false dichotomy.** The token alphabet *is* the class partition on
   this vehicle. Resolving it needed three source lines, not a measurement. The charter's instinct
   to disambiguate first was right; the premise that two readings exist was not.
2. **The charter's own reading-(a) arithmetic was correct and already measured.** `tri1` §3.4 puts
   it at 0.556381 S = 19.72× the gap. The charter computed ≈19× and then treated it as the branch to
   *avoid* rather than as the answer.
3. **The arm was closed one and three days before the charter was written, by `ddm_tri1` §3.4 and
   `ddm_ws1`. Neither is cited in the charter.** Both are in `.omx/research/`; `ws1`'s verdict is
   also in `.omx/state/main_hot_state.md` in the words "**Lane polynomials LOST their race**."
4. **`ddm_af1` contradicts itself on this cell, and the charter inherited only one side.** §2.4
   records `tri1`'s "the seg leg alone refuses at ≥21.62×"; §9 then nominates the same cell as "the
   single cheapest deciding measurement." A cell refused at 21.62× on one leg is not the cheapest
   deciding purchase. The charter carried §9's recommendation without §2.4's refusal.
5. **38,649.8 B and 38,183 B are different index sets, not adjacent measurements** (§2, exact).
   `af1` §6's phrasing should be corrected. The credit is the decoded-axis figure; the damage is
   counted on the GT axis; 466.8 B moves between them.
6. **38,649.8 B is a ceiling, not a credit**, and the only measured realization datum brackets the
   achievable value at [6,609 B, 38,650 B] (§4).
7. **The build estimate was wrong by orders of magnitude** — ≈154 hardcoded sites in two languages
   plus a 60-epoch retrain (§4), not "one retrain plus one authority row."
8. **Bar (c) was unreachable by construction.** This arm does not hold the exclusive n600 scorer
   lane; `ld1` and `jf1` both queued their scorer joins for the same reason. A charter that requires
   an authority-lane `d_pose` must first transfer the lane or name who will fire it.
9. **The charter's word "implicit" inverts the corpus's measured vocabulary** (§3). Under `ig1`'s
   routing table the charter proposed OPTIMAL-GRAMMAR while calling it IMPLICIT — and Lane's row
   resolves to IMPLICIT, the opposite route.

---

## 6. Disposition and fire order

- **CLOSED — no successor charter.** M1 and M2 are both closed on inherited, measured evidence.
  Do not charter alphabet reduction, Lane class-merge, or Lane geometric regeneration on the dx2
  token body without new evidence that overturns `tri1` §3.4 or `ws1`'s Road↔Lane stratum.
- **FIRED AND RETAINED:** the index-set reconciliation, owner `ddm_ma2`, consumer any arm quoting a
  Lane byte figure. Store
  `/Volumes/APDataStore/pact/ddm_ma2_lane_index_set/measurement_v1/LANE_INDEX_SET_RECONCILIATION.json`,
  stable content digest **`411e57f4daf3c5d5217ffc4f0ebd8b23fcad33138fa89e28b636bf9bafef313a`**
  (SHA-256 over the report with the wall-clock field removed; reproduced identically on a
  determinism repeat). The raw file sha is deliberately **not** the citable identity — it carries
  `elapsed_s` and therefore changes on every run even when every measured number is bit-identical.
  Payload is the JSON itself: the run materialized no field it discarded, and every input is an
  existing retained field cited by digest.
- **OWED TO MAIN (documentation, not measurement):** `af1` §6's "same concentration" phrasing and
  §9's recommendation-versus-§2.4-refusal contradiction. Both are `af1`'s to amend; recorded here
  per APPEND-ONLY, not edited into its memo.
- **WHAT STAYS LIVE, unchanged by this arm:** `tri1` §3.5 names **R3 — lossless traversal reorder
  *with* model refit** as "the last live cell" (`to2`/`ad2` never refit; `grep -ci "refit"` over both
  → 0). It is lossless, so it is the one route that does not face the 1.25e-4 pose budget at all.
  That is where `af1` §9's own "cheaper still, and worth firing first" pointer leads, and this arm
  did not touch it.

---

## 7. What I did not do

- **I ran no encoder and no scorer.** No re-encode, no `d_seg`, no `d_pose`, no `S`. The one
  measurement is a sum over a retained cost field. Per `tba1` §8, **the bit map is a MAP, not a
  PRICE** — nothing here is a rate claim.
- **I did not build the 5→4 alphabet.** The recall gate closed the arm first, which is the charter's
  stated preferred outcome.
- **I did not verify `ws1`'s or `tri1`'s raw receipts.** Their numbers are taken as published, with
  their own scopes attached. `ws1` is `FAMILY` for the registered explicit-worldsheet reference
  form; `ws0` is `FORMULATION`; `tri1` §3.4 is `FORMULATION` on the `tokens × HPAC` R2 cell. None is
  a global lower bound over all possible Lane representations, and I do not present them as one.
- **I did not measure `Δd_pose`** against the 1.25e-4 budget (§0). Unreachable, and moot given the
  seg and rate refusals.
- **I did not re-derive λ_B or the demand** — CITED from `ddm_tx1_toolbox_crosswalk_20260819.md` §0.
- **The 17.10% realization fraction (§4) is an extrapolation from one undertrained rung**, labelled
  as a bracket, not a prediction. No arm should quote 6,609 B as a measured credit.
- **I did not measure a noise floor** on any ratio quoted here. Per `ds1`, nobody on this campaign
  has. The 21.62× and 3.728× are far above any plausible floor; the 10.170× per-position ratio
  in §2 is a raw sum over 2,248 positions and is unprotected.
- **The `ws0`/`ws1` price comparison is not strictly apples-to-apples** with a token-class cost, and
  §3 says so rather than burying it.

---

**Own-vehicle frontier: dx2 — S 0.14821987563243377 @ 180,368 B `[contest-CUDA T4, n600]`, archive
sha `976f706d…f6de6674` — UNMOVED by this arm.** Gap to 0.12 = 0.028220 ⇒ 42,382 B at fixed
distortion, or 150 B at zero distortion. This arm bought no bytes, closed one direction on inherited
evidence, and corrected two committed memos for $0.
