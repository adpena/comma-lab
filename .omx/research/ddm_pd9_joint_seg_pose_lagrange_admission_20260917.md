# ddm_pd9 — replacing the seg SCREEN with a JOINT seg+pose admission, on pd8's price-first pool

Tokens: `[no-triality] [p0-ledger-ok]`

Lane `ddm_pd9_joint_seg_pose_admission_20260917`. Base at bind time: **move 54**,
S 0.13605599532783202 @ 179,266 B, archive sha
`5c6bf403b4cb4554fe24a46bdf5b46d62876854764a10a22d8c90d76a7292ee6`, re-derived from
`reports/latest.md` and `.omx/state/canonical_frontier_pointer.json` and re-checked against the
live pointer inside `bind54`. **The pointer moved to move 55 while this arm's wave ran** (§6).
Axis `[macOS-CPU advisory, frozen CPU-torch PoseNet + SegNet, DALI GT lineage, n600]` for the
distortion legs; charges EXACT through the shipped RLC1 coder. **No score is claimed. Only
`upstream/evaluate.py` on shipped bytes is a score, and MAIN fires.** No Modal, no authorization,
no fire, no completion, no packet, no pointer write.

Pose solver, stated once: every per-pair credit comes from
`ddm_jg5_pose_resolve_on_edited_renders.refine_pair` at PoseNet batch 1 on the moved render, with
`outer_rounds=40, max_gn_iterations=400`; every n600 number comes from
`ddm_up2_shipping_pose_solve.measure_pose` at batch 8.

---

## 1. THE VERDICT — the joint rule is RIGHT and it is worth 0.14 bars

**F3 FIRES.** On the identical 4,291-proposal pool, at the identical real sheet charges, the JOINT
admission (no seg screen; seg is a term in the score's one currency) admits **52 pairs / 61 tokens
for −3.147209e-05 S**, and the SCREEN admission admits **44 pairs / 51 tokens for −2.875289e-05 S**.
The gain is **−2.719198e-06 S** against a pre-registered bar of −1e-05. The joint set is a strict
SUPERSET: it drops nothing and adds exactly **eight pairs — 20, 22, 197, 234, 288, 438, 456, 587**.

−2.719e-06 S is **0.136 bars** of the −2e-05 admit bar and **0.117×** the 34.8 B container-break
standard deviation (2.318e-05 S). The two admissions are **not distinguishable on shipped bytes**
at this operating point.

**And the mechanism is not the one the charter assumed.** The charter's example was a proposal that
costs one argmax cell and buys −3e-06 S of resolved pose. That is not what the screen was hiding.
Of the eight pairs the joint rule adds, **seven cost exactly one cell (+8.483e-07 S) and pay for it
with NEGATIVE BITS** (−0.81e-06 to −1.32e-06 S); their pose legs run from −2.7e-07 down to
**+7.4e-08, and three of them make pose WORSE**. Only pair 22 is a real pose buy.

> **The screen's defect is that it is a TWO-TERM test inside a THREE-TERM score.** Its budget,
> `floor(base[pair]·pose_unit / seg_cell)`, is computed from pose alone. A proposal that SAVES
> 13.8 bits and costs one SegNet cell was invisible to it, even though the bits pay for the cell
> twice over.

The cure is a drop-in one-line change and it is **strictly dominant** — more complete *and* cheaper
(§7).

## 2. The binding, and the controls that reproduce a known answer

pd9 re-points pd8's move-54 constants at pd9's own copies process-locally
(`experiments/ddm_pd9_joint_admission.py::rebind_pd8_to_pd9`) and leaves pd8's and pd4's own
verification paths in charge. No pd4, pd6, pd7, pd8 or shipped-runtime source was edited.

| control | outcome |
|---|---|
| **move 54's cold decode, rebuilt in this arm's store** | **3,662,409,600 B sha `ff43a9c97c72d0917ac4c2b856315648eddf3c0d3f69717eca132bfecf37a324`** — BIT-IDENTICAL to the decode pd7 and pd8 each reported for these bytes; **1,107.4 s** on CPU at the contest thread count (4), `decoded_field_matches_admitted: true`. A third arm, a third decode, the same bytes |
| **bind54** | archive / argmax / field / raw all verified by sha; the pointer's score re-derived from its three components = **0.13605599532783202**, equal to the pointer's field to all digits |
| **operating point** (EXPIRED at move 55) | S per archive byte **6.658589531221714e-07**, S per seg cell **8.482724499051702e-07**, S per unit of one pair's d_pose **1.3015705930547656**, base pose mean **4.099227789357173e-06** — all four identical to pd8's, as they must be on the same field |
| **F1 — this arm's own control encode re-packs move 54's archive** | **179,266 B sha `5c6bf403b4cb…`**, twins byte-identical in-process, stream 119,014 B (ideal 119,013.279), decoded plane sha `ed69d961fe0b98c3…`. **F1 does not fire** |
| **the INHERITANCE control** | this arm's control encode and pd8's charge move 54's field **IDENTICALLY, frame for frame: max absolute gap 0.0 bits over all 600 frames**, total 952,106.2347452887 bits on both. Two independently compiled rc64 backends in two stores are the same instrument, which is what makes inheriting pd8's sheet charges sound |
| **F2 — the reuse holdout** | 21 of pd8's realized rows were deliberately withheld from the seed so the wave re-rendered, re-argmaxed and re-refined them. **21/21 reproduced, d_cells identical on all 21, max relative difference in resolved d_pose 0.0.** **F2 does not fire** |
| **the base band, re-measured over ALL 600 pairs** | max gap **5.464407685599634e-09**, median 2.929e-10, p99 3.414e-09, band 1.0928815371199267e-08 — identical to pd8's |
| **the screen pool IS pd8's pool** | `split-pool --screen-cap 2` over this arm's enriched rows returns **1,115 rows / 398 pairs** — exactly pd8's pool, with **133 rows only in the joint pool**. The control is therefore pd8's own set, not a re-derivation of it |

Receipts: `BIND54.json`, `BASEBAND.json`, `CONTROL_IDENTITY.json`, `VERIFY_HOLDOUT.json`,
`SPLIT_POOL.json`, `parseback_base/PARSEBACK_RESULT.json`.

## 3. What the screen refused, and the ceiling that says how much of it mattered

MEASURED from pd8's own screen rows: **3,176 of 4,291 proposals were refused before any credit was
read.** The histogram of their SegNet cost:

| refused seg cost (cells) | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 22 | 28 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| proposals | 696 | 732 | 694 | 464 | 260 | 153 | 72 | 38 | 31 | 8 | 8 | 6 | 6 | 1 | 1 | 2 | 1 | 1 | 1 | 1 |

**2,122 refused proposals over 528 pairs cost three cells or fewer** — the charter's scope. By pd8's
own per-pair budget: 2,573 were refused on a budget of 0, 251 on 1, 352 on 2.

**DERIVED before the wave** (`REFUSED_CENSUS.json`, written before any credit existed): a row can
never net if `bits/8·S_per_byte + d_cells·seg_cell − base[pair]·pose_unit ≥ 0`, because the pose
half of that bound is exact (credit ≥ −base[pair]). Of the 2,122, **1,989 can never net and 133 over
87 pairs can** — and **111 of those 133 cost exactly three cells on pairs whose DERIVED budget was
already ≥ 2**: they were refused by pd8's flat CAP of 2, not by the budget arithmetic. The other 21
cost one cell on budget-0 pairs and are reachable only because their real charge is negative. The
bits half of the bound is the SHEET charge, so "can never net" is DERIVED, not MEASURED; §9 tests it.

The pre-wave prediction was recorded twice and the second corrected the first.
`PREREGISTRATION.json` predicted F3 would fire from the halving paying population;
`PREDICTION_REFINED.json`, written before the wave, measured −5.0168e-04 S of reachable ceiling over
79 pairs pd8 did not admit, resampled pd8's own per-pair best recovered-pose fraction (positive on
248 of 398 pairs; median(>0) 0.2813, p90 0.7066) and projected a median gain of **−1.082e-04 S with
P(gain ≤ −1e-05) = 1.000** — while naming the three reasons that projection is optimistic. **The
measured gain came in 40× below that projection.** The projection's first named optimism — it
resampled a per-pair BEST drawn from ~2.8 rows per pair onto reachable pairs carrying 1–2 rows — is
the one that did the damage.

## 4. The wave

The screen was disabled without touching a line of pd6: its budget is
`min(cap, floor(base·pose_unit·frac / seg_cell))`, so `--cell-budget-cap 3 --cell-budget-frac 1e9`
makes the budget a **flat three cells for every pair**. Confirmed in the run's own screen rows: a
3-cell proposal was not refused.

pd8's 1,115 realized rows were INHERITED (they are measurements of the same proposals on the same
field by the same instrument, and §2 proves the instruments agree), minus the 21-row holdout the
wave re-measured. The new work was sharded in two waves:

| wave | rows | pairs | what |
|---|--:|--:|---|
| **A** | 154 | 100 | the 21-row reproduction holdout + **all 133 reachable refused rows** |
| **B** | 436 | 100 | a declared random PAIR-sample of the 1,989 unreachable refused rows |

**Wave A carries every row that can change the set**, which is provable, not assumed: a row whose
ceiling is ≥ 0 has a negative `net_S_estimate` at any credit, and pd5's admission keeps a pair only
when its best row's net is positive. Wave B is realized for the charter's scope and to TEST that
argument (F7); it is a SCOPE reduction, declared with its seed (`WAVE_B_SAMPLE.json`, seed
20260917, uniform over PAIRS, never a prefix — m88), taken because the per-pair render/argmax/
base-pose cost dominates this wave and a pair-sample buys far more realized rows per second.

After wave A the pool is **1,248 realized rows over 409 pairs**: 540 rows improve their pair (43.3 %),
260 pairs carry positive resolved credit, **52 pairs whose best row pays at its own real price**
(pd8: 44), best single credit −2.987846692880755e-06 d_pose.

## 5. THE JOINT-vs-SCREEN CONTROL

Both sides are the same code (`ddm_pd5_multitoken.setprice`, whose admission is already the score's
one currency: `benefit_S = −credit·pose_unit − d_cells·seg_cell`, keep iff `benefit_S − bits/8·S_per_byte > 0`),
on the same rows, at the same real sheet charges. Only membership differs.

| set | pairs | tokens | ledger bits | **net S** | seg-costing members | cells | seg S |
|---|--:|--:|--:|--:|--:|--:|--:|
| **JOINT** (no screen) | **52** | **61** | −163.32 | **−3.147209e-05** | **15 of 52** | **+17** | +1.4421e-05 |
| **SCREEN** (pd8's own iteration 0) | 44 | 51 | −70.06 | −2.875289e-05 | 7 of 44 | +7 | +5.9379e-06 |
| **gain** | +8 | +10 | −93.26 | **−2.719198e-06** | +8 | +10 | +8.483e-06 |

`d_cells` histograms: the SCREEN set is **−1 on 4 members (they REPAIR a cell), 0 on 33, +1 on 7**;
the JOINT set is **−1 on 4, 0 on 33, +1 on 14, +3 on 1**. The joint rule changes nothing below zero
and nothing at zero — it adds exactly seven one-cell members and one three-cell member.

### The eight pairs the joint rule adds, leg by leg

| pair | cells | bits | ΔS seg | ΔS rate | ΔS pose | **net S** |
|--:|--:|--:|--:|--:|--:|--:|
| 20 | +1 | −13.80 | +8.483e-07 | −1.149e-06 | −2.675e-07 | **−5.679e-07** |
| 22 | +3 | −1.93 | +2.545e-06 | −1.605e-07 | −2.738e-06 | **−3.535e-07** |
| 197 | +1 | −14.41 | +8.483e-07 | −1.200e-06 | −1.667e-07 | **−5.182e-07** |
| 234 | +1 | −15.83 | +8.483e-07 | −1.318e-06 | **+7.401e-08** | **−3.956e-07** |
| 288 | +1 | −11.51 | +8.483e-07 | −9.581e-07 | −7.153e-08 | **−1.813e-07** |
| 438 | +1 | −12.50 | +8.483e-07 | −1.040e-06 | **+6.058e-08** | **−1.316e-07** |
| 456 | +1 | −9.71 | +8.483e-07 | −8.081e-07 | −3.531e-07 | **−3.129e-07** |
| 587 | +1 | −13.56 | +8.483e-07 | −1.129e-06 | **+2.251e-08** | **−2.583e-07** |
| **sum** | | | **+8.482724e-06** | **−7.762395e-06** | **−3.439527e-06** | **−2.719198e-06** |

**All eight are paid for by bits. Five have any pose benefit at all; three make pose worse.** The
whole prize of removing the seg screen is **+8.4827e-06 S of extra seg cost** bought with
**−7.7624e-06 S of saved bytes** and **−3.4395e-06 S of pose** — the bytes alone very nearly pay
for the cells, and the pose is the smaller half of the change.

**The gap is EXACTLY the eight added pairs, and that makes F3's firing airtight.**
−3.147209e-05 + 2.719198e-06 = −2.875289e-05, the screen's own net to all digits. So the wider pool
did not change a single one of the 44 shared pairs' choices: not one of the 133 newly realized rows
beat the incumbent row on a pair the screen already admitted. The joint rule is purely additive
here. A set-price fixed point is a SUBSET of iteration 0, so the joint fixed point is the screen
fixed point plus whichever of the eight survive the absorb — and the gain can only ERODE from
−2.719198e-06, never grow past the −1e-05 bar.

Receipt: `JOINT_VS_SCREEN.json`.

## 6. Why the closed-archive ladder is not in this memo

`ddm_sj1_rlc1_price.guard()` **REFUSED** to register the iteration-0 set field:
`POINTER_MOVED: refuse; the price must be measured against the live row`. While this arm's wave ran,
pd8's own candidate landed as **pointer move 55** — S 0.13603403441098336 @ 179,255 B, sha
`ddadf998ddacab9b356b9b6a01a78c845ab3d6f643dd1e3cf8f37840ae550b8a` (commit `e4de53eff`). The gate is
correct and this arm did not go around it: the set-price fixed point and the closed-archive ladder
are simply not available for a superseded base, and **no byte claim is made here**.

The comparison in §5 survives that, for three stated reasons: both sides are priced in the same
currency by the same code; the charges are real 600-frame sheet encodes, not a model; and the two
sets share 44 of their 52 members, so the set-level interaction largely cancels in the difference.
What the missing ladder costs is the *absolute* net of each side, not their gap.

**The absorb would not have rescued F3.** Every added pair depends on a NEGATIVE bits estimate
(−9.71 to −15.83 bits) and each nets at most −5.679e-07 S. pd8 measured the iteration-0 ledger
OVER-crediting the real encode by −149 % on this exact field, and the absorb has shrunk the set in
every pass (pd8: 44 → 28 → 26). The eight added pairs ARE the most marginal members of the joint
set, and the most marginal members are the first the absorb drops.

**The pointer move also means the joint set's eight extra pairs are not a candidate on the live
row**: at −2.719e-06 S they are 0.14 bars, and move 55 already banked the screen's own set
(realized −2.196e-05 S against move 54).

## 7. THE CURE — a rate-aware ceiling is strictly dominant

`experiments/ddm_pd6_price_first.py::cmd_realize` screens on

```python
budget = min(cap, floor(base[pair] * pose_unit * frac / seg_cell))   # POSE only
```

Replace it with the full-score ceiling — realize iff

```python
bits/8 * S_PER_BYTE + d_cells * seg_cell - base[pair] * pose_unit < 0
```

MEASURED on the same 4,291-proposal pool at move 54's operating point:

| screen | refines spent | reachable rows covered |
|---|--:|--:|
| pd8's pose-only budget | **1,115** | misses **133** |
| the rate-aware ceiling | **798** (665 of pd8's + the 133 it missed) | **all of them** |

**28 % fewer refines AND complete coverage.** 450 of pd8's own 1,115 refines (40.4 %) were spent on
rows the ceiling can prove dead before paying for them. The change needs the proposal's real charge
at screen time, which the price-first generator already has — the charge is measured before the
wave by construction. Receipt: `SCREEN_CURE.json`.

## 8. The price of a re-rendered pair, and the reversion trap

A decomposition of pd8's own sheet charges that pd8 did not publish. Split each sheet by whether
the pair was RE-RENDERED in the move 53 → 54 step (53 of 588 pairs):

| sheet | all | on the 53 EDITED pairs | on the 535 others |
|---|--:|--:|--:|
| **00 (rank 0)** | +1,939.8 bits, **3.299**/tok | **−150.9 bits, −2.847/tok** | +2,090.7 bits, **+3.908**/tok |
| 01 (rank 1) | 6.504/tok | 5.452/tok | 6.621/tok |
| 02 | 7.619/tok | 7.435/tok | 7.639/tok |
| 03 | 7.777/tok | 7.398/tok | 7.819/tok |

**64 % of edited pairs carry a NEGATIVE rank-0 charge against 19 % of unedited ones (2.79×.)**

**But half of that is the coder's own undo button.** Tested against pd7's retained move-53 field:
**18 of the 53 rank-0 proposals on edited pairs simply REVERT a move-54 token to its move-53 value,
and 16 of those 18 price negative.** The effect survives the correction — over the whole pool,
non-reversion proposals on edited pairs price **4.469 bits/token against 5.892** on unedited ones and
are negative **13 %** of the time against **6 %** — but any claim about "cheap changes on re-rendered
pairs" that does not remove reversions is counting an undo. Receipts:
`EDITED_PAIR_PRICE_SPLIT.json` (whose first reading is corrected in place, not deleted) and
`REVERSION_SPLIT.json`.

**Consequence for the decay curve the charter asked about (2.163 → 3.075 → 3.279 → ?):** there is
**no fourth point**, because this arm ran on move 54's own field and the rank-0 sheet on that field
is pd8's 3.279 by identity. What the decomposition adds instead is *where the decay lives*: the
dearness is a property of the **unedited population**, not exhaustion on the pairs the previous pass
spent.

## 9. WAVE B — the ceiling test and the seg-cost / pose-credit coupling

Wave B realized a declared random PAIR-sample of the refused pool — **436 rows
over 100 of the 462 pairs** that carry a ≤ 3-cell
refused proposal (seed 20260917, uniform over pairs, never a prefix). Every sampled row was
realized exactly as an admitted one: re-rendered from the shipped field, frozen-argmax seg census,
per-pair carrier re-solve. The full pool is **1684 realized rows over
440 pairs**.

**F7 — the pre-wave ceiling holds.** Of the realized rows, **569
carry a pre-wave ceiling**, and **0 of them net**. Not one row the
ceiling called arithmetically dead came back alive. **F7 does not fire**, which is what licenses
§4's claim that wave A carried every row that could change the set.

**The coupling, MEASURED, with the selection controlled.** Does a proposal that costs more SegNet
cells buy more pose? The pooled answer is biased: the REACHABLE rows (wave A) were selected for a
negative charge, and a proposal with a strongly negative charge is a different animal. So the two
populations are reported apart. **The UNREACHABLE sample is the one that answers the question** —
wave B's rows were sampled at random over pairs and selected on nothing:

| refused seg cost (cells) | rows | improves its pair | median credit (d_pose) | best credit |
|---|--:|--:|--:|--:|
| **1** | 140 | **41.4 %** | +2.1701e-08 | -3.2270e-07 |
| **2** | 161 | **39.1 %** | +3.7162e-08 | -6.9717e-07 |
| **3** | 135 | **41.5 %** | +2.4958e-08 | -8.0068e-07 |

and the REACHABLE rows, selected on bits, for contrast:

| refused seg cost (cells) | rows | improves its pair | median credit (d_pose) | best credit |
|---|--:|--:|--:|--:|
| **1** | 21 | **28.6 %** | +1.6245e-07 | -2.7129e-07 |
| **2** | 1 | **100.0 %** | -3.8787e-08 | -3.8787e-08 |
| **3** | 111 | **52.3 %** | -4.5230e-08 | -2.1035e-06 |

**There is NO coupling in the unselected population.** The improve-fraction is FLAT —
41.4 %, 39.1 %,
41.5 % at one, two and three cells — and **the median proposal at
every cost makes pose WORSE** (+2.1701e-08,
+3.7162e-08, +2.4958e-08 d_pose). Moving more
argmax cells does not buy more pose. About two in five proposals help their pair whatever they cost,
which is the same rate the screen-admitted rows show at ZERO cells
(38.9 %).

**The pooled table LOOKS monotone and that is the selection talking.** The REACHABLE stratum reaches
52.3 % improving at three cells with a median of
-4.5230e-08 — the only negative median anywhere in this measurement —
and it is the one stratum that was CHOSEN for a strongly negative charge. Mixing it into the pool
manufactures a trend that the controlled sample does not have. An interim read of this arm's own
partial wave, before the sample was complete, showed 40.4 → 41.9 → 44.2 % and looked monotone; the
completed sample says flat. **Two points of a noisy fraction are not a trend.**

**This kills the last defence of the screen's premise.** The proposals the seg screen refused are
pose-INDISTINGUISHABLE from the ones it admitted. They differ in their BYTE price, which is exactly
what §5 measured from the other side: the eight pairs the joint rule adds are paid for by bits.

For completeness, the pooled table over all 1684 realized rows (both populations,
plus the screen-admitted rows that carry no ceiling):

| seg cost (cells) | rows | improves its pair | median credit (d_pose) | best credit |
|---|--:|--:|--:|--:|
| **-2** | 1 | **0.0 %** | +2.6812e-06 | +2.6812e-06 |
| **-1** | 28 | **25.0 %** | +1.6578e-07 | -5.3906e-07 |
| **0** | 627 | **38.9 %** | +3.9586e-08 | -1.9057e-06 |
| **1** | 414 | **45.9 %** | +2.5614e-08 | -2.9878e-06 |
| **2** | 368 | **44.0 %** | +3.5326e-08 | -1.7259e-06 |
| **3** | 246 | **46.3 %** | +1.4323e-08 | -2.1035e-06 |

Receipts: `CEILING_TEST.json`, `CREDIT_FULL.json`, `WAVE_B_SAMPLE.json`,
`search/wave/realized_*.jsonl` and `screen_*.jsonl` (every realized row and every screened proposal).

## 9b. What is left on this object after move 55 — 0.30 bars

The joint set's 52 pairs split cleanly against what move 55 actually banked: **26 are already in
move 55** (pd8's ladder winner) and **26 are unspent**. On move 54's own — now EXPIRED — prices the
whole unspent remainder is worth

> **−6.027519e-06 S = 0.30 bars of the −2e-05 admit bar**, of which −2.719198e-06 S is the eight
> pairs the screen would never have taken.

Unspent pairs: 6, 9, 19, 20, 22, 45, 197, 234, 242, 247, 248, 268, 288, 305, 311, 333, 348, 373,
385, 401, 438, 456, 584, 587, 588, 597. This SIZES the successor's pool; it does not price it — move
55 changed 26 pairs' tokens and re-solved the carrier over all 600, so every charge and every credit
above expires with it (`UNSPENT_AFTER_MOVE55.json`).

### The B/H decomposition of move 55, which m132 owes and nobody had taken

m132 ([[seg_mechanisms_die_on_collateral_not_targeting_20260821]]) requires a seg mechanism's
admission to report BENEFICIAL against HARMFUL, never the net alone. pd6's realize records only the
net `d_cells` per proposal, and the net is what the score charges — so the admission is priced
correctly — but the diagnostic was owed. Taken here on move 55's own cold-decode argmax against move
54's, on the DALI GT lineage:

| | cells |
|---|--:|
| argmax cells that MOVED | **19** |
| **BENEFICIAL** (move 54 wrong → move 55 right) | **8** |
| **HARMFUL** (move 54 right → move 55 wrong) | **10** |
| wrong → wrong, relabelled | 1 |
| **net (H − B)** | **+2**, and 12,127 → 12,129 flips |

Collateral ratio H/B = **1.25**. The identity net = H − B holds. The check reproduces pd8's own
published counts (12,127 / 12,129 / 19 moved) before adding the split, which is why the split is
trusted. Receipt: `BH_CENSUS_pd8_control.json`, tool `bh_census.py`.

pd8 predicted from the halving paying population (128 → 72 → 44) that "pass 4 is the one that has to
change something, not just run again." **That is now measured rather than projected, and the
admission rule was the last cheap thing left to change.** A fourth pass of this generator — with or
without the seg screen — has about a third of a bar in it.

## 10. Falsifiers, as pre-registered

| falsifier | outcome |
|---|---|
| **F1** the pricer does not re-pack move 54's own archive | **does not fire** — 179,266 B sha `5c6bf403…`, twins identical |
| **F2** the inherited pd8 rows are not reproduced on this instrument | **does not fire** — 21/21 rows, d_cells identical, resolved d_pose relative difference **0.0** |
| **F3** the JOINT set's net is not better than the SCREEN set's by ≥ 1e-05 S | **FIRES — gain −2.719198e-06 S, 3.7× short of the bar.** `verdict_scope: formulation` (§14) |
| **F4** the admitted set nets worse than −2e-05 S on the shipped bytes' decode | **not evaluated** — F3 fired first and the pointer moved; no candidate was built, no decode was owed, none is claimed |
| **F5** the SET re-price differs from the ledger by more than 10 % after absorbed iteration | **not evaluated** — the pricer correctly refused to register a field against a superseded pointer (§6) |
| **F6** the projection's pose leg and the decode's differ by more than 1.5× | **not evaluated** — no candidate decode |
| **F7** a row the pre-wave ceiling called unreachable actually nets | **does not fire** — **0 violations over 569 realized rows that carry a pre-wave ceiling** (§9) |

Per the charter, **a fired first falsifier CLOSES the joint-admission formulation on this object.**

## 11. What this does NOT claim

1. **No score of any kind.** Every S here is a projection on measured legs at move 54's operating
   point, which EXPIRED at move 55. Only `upstream/evaluate.py` on shipped bytes is a score.
2. **No byte claim.** No archive was closed by this arm. The set-price fixed point was not reached.
3. **The distortion legs are `[macOS-CPU advisory]`** — frozen CPU-torch PoseNet and SegNet against
   DALI-lineage GT.
4. **The charges are INHERITED** from pd8's real 600-frame sheet encodes on the unchanged move-54
   field, pinned by file sha and justified by the frame-for-frame identity of the two roots' control
   encodes (§2). They are not re-measured here, and `price-merge` is wired to REFUSE in this arm so
   a re-charge cannot silently land under a `pd8sheet` name.
5. **pp1's twelve floor pairs stay excluded** on pp1's measurement; every number is drawn on the
   remaining 588 pairs, which hold 43.3 % of the n600 pose mass.
6. **The ceiling is DERIVED, not MEASURED**, on its bits half: `real_delta_bits` is the sheet charge
   and a SET re-encode can move it. §9 is the empirical test.
7. **The edited-pair price split is one field and one edit set.** The sign and magnitude are exact;
   the generality is INFERRED until a second field reproduces it.
8. **Wave B is a declared random pair-sample** (100 of 462 pairs, 436 of 1,989 rows), not the
   population.
9. **The B/H split is NOT available per proposal.** pd6's realize records the net `d_cells`
   per row, and the net is what the score charges, so the admission is priced correctly; but
   the m132 collateral diagnostic needs both argmax planes and realize keeps neither. The
   decomposition in §9b is therefore taken on the shipped object (move 55's own decode), not
   on the 1,684 proposals this arm realized.

## 12. Custody

Store **`/Volumes/APDataStore/pact/ddm_pd9/`**. **Vertigo was never opened for writing.** Nothing
under `ddm_pd1`–`ddm_pd8`, `ddm_sj1`, `ddm_jr*`, `ddm_psa*` or `ddm_mrs*` was written; those stores
were read only, and pd7's and pd8's files were copied out, never modified in place.

Retained **0.15 GiB** against the charter's 2 GiB cap —
**313 files**, 159,342,219 B hashed in `RETENTION_MANIFEST.json`
(plus 153 KiB of launcher logs and
116 MiB of rebuildable pricer u8 bulk, each recorded
but not hashed twice). **Every realized search row and every screened proposal is kept, not only
the rows that would have shipped.**

APDataStore free space MEASURED at every heavy step: **22 GiB at start**, 18 GiB through the base
decode and the control encode, 18 GiB through both waves, **22Gi after the certified prune**.

Every bulk payload removed was hashed first, with the exact command that rebuilds it, in
`BULK_CERTIFICATE.json` — the record lands on disk BEFORE the bytes leave it:

| payload | bytes | sha256 |
|---|--:|---|
| `parseback_base/0.raw` | 3,662,409,600 | `ff43a9c97c72d0917ac4c2b856315648eddf3c0d3f69717eca132bfecf37a324` |
| `parseback_base/.f26_decode_checkpoints` | 117,968,488 | `(directory, per-file)` |

| path | what |
|---|---|
| `PREREGISTRATION.json` · `PREDICTION_REFINED.json` · `REFUSED_CENSUS.json` | the falsifiers, the refined pre-wave prediction with its three named optimisms, and the DERIVED ceiling — all written before any credit was read |
| `BIND54.json` · `BASEBAND.json` · `CONTROL_IDENTITY.json` · `VERIFY_HOLDOUT.json` | the binding and the four reproduction controls |
| `SEED_WAVE.json` · `WAVE_B_SAMPLE.json` · `plan/WAVE_A.jsonl` · `plan/WAVE_B*.jsonl` · `plan/refused_within_cap.jsonl` | what was inherited, what was held out, what was sampled and with which seed |
| `search/wave/realized_*.jsonl` · `screen_*.jsonl` · `HOLDOUT.jsonl` | every realized row and every screened proposal, winners and losers |
| `sheets/priced_realized*.jsonl` · `SPLIT_POOL.json` · `CREDIT.json` · `CREDIT_FULL.json` | the enriched pool, the joint/screen split and both credit censuses |
| `setprice/STATE.json` · `setprice/set_00.npz` · `JOINT_VS_SCREEN.json` · `UNSPENT_AFTER_MOVE55.json` | the joint admission's iteration-0 set, the control table and the successor's pool size |
| `SCREEN_CURE.json` · `CEILING_TEST.json` · `EDITED_PAIR_PRICE_SPLIT.json` · `REVERSION_SPLIT.json` · `BH_CENSUS_pd8_control.json` | the cure, the F7 test, the edited-pair price split with its reversion correction, and move 55's B/H decomposition |
| `rlc1_price/encode/control/primary/` | this arm's own control encode of move 54's field — the F1 re-pack and the inheritance control |
| `BULK_CERTIFICATE.json` · `RETENTION_MANIFEST.json` | the custody record |

Producer: `experiments/ddm_pd9_joint_admission.py` — a thin process-local rebinding of pd8's
move-54 constants onto pd9's own copies, plus `refused-census`, `seed-wave`, `verify-holdout`,
`split-pool` and a `price-merge` that REFUSES. Every stage that measures anything is pd6's
(through pd7 and pd8), imported and called unchanged; downstream, pd4's merge/carry, pd5's
setprice and `ddm_sj1_rlc1_price` are unchanged. Store-side helpers are pd8's, copied with their
paths and one import line changed.

## 13. What this hands the next arm

1. **Adopt the rate-aware ceiling. It is free and strictly dominant.** One expression replaces
   pd6's pose-only budget; on the same pool it costs 28 % fewer refines and stops missing rows that
   can net. There is no trade-off to weigh. (§7, `SCREEN_CURE.json`.)
2. **Stop expecting the seg screen to be hiding pose.** It was hiding BYTES. Seven of the eight
   pairs it cost us are one-cell edits that SAVE 10–16 bits; three of them make pose slightly worse
   and are admitted anyway, correctly, on the rate leg. Any future "the screen is the binding gate"
   argument has to be an argument about the rate term.
3. **On a re-rendered pair, a third of the time the coder's cheapest next change is an UNDO.**
   18 of 53 rank-0 proposals on pd7's edited pairs revert a move-54 token to its move-53 value, and
   16 of those price negative. A generator that ranks by first-order price on a freshly edited field
   will spend its cheapest slot on its predecessor's own reversal unless it filters for it. The
   filter is one comparison against the previous field. (§8.)
4. **The dearness lives in the UNEDITED population.** The rank-0 sheet costs −2.847 bits/token on
   the pairs the previous pass re-rendered and +3.908 on the ones it did not. The pass-over-pass
   rank-0 series (2.163 → 3.075 → 3.279) is therefore not exhaustion of the pairs we spent.
5. **Seg cost and pose credit are NOT coupled — control for the selection and the trend vanishes.**
   (§9.) In the unselected sample the improve-fraction is flat at 41.4 / 39.1 / 41.5 % over one,
   two and three cells, and the median proposal at every cost makes pose WORSE. Moving more argmax
   cells does not buy more pose. The pooled table looks monotone only because the rows selected for
   a strongly negative charge sit inside it. **Stop treating the seg screen as a pose gate.** An
   interim read of this arm's own half-finished wave showed 40.4 → 41.9 → 44.2 % and looked like a
   trend; the completed sample says flat. Two points of a noisy fraction are not a trend.
6. **This generator has about 0.30 bars left on this object.** 26 of the joint set's 52 pairs are
   unspent after move 55, worth −6.027519e-06 S on move 54's expired prices (§9b). pd8 predicted
   pass 4 would have to change something; the admission rule was the last cheap thing to change, and
   it is worth 0.14 bars. **The next unit should change the GENERATOR or the OBJECT, not the
   admission.**
7. **What is still unmeasured:** whether the eight added pairs survive a real set re-encode (the
   pointer moved before the ladder could run); whether the reversion enrichment reproduces on a
   second field; and whether the seg/pose coupling in §9 is strong enough anywhere in the plane —
   this arm measured it only on the cheap half of the price list.

## 14. verdict_scope

**verdict_scope: FORMULATION — the joint seg+pose Lagrange admission, on this object, at this
operating point.** What is MEASURED is that replacing pd6's per-pair seg screen with the score's own
three-term arithmetic on the identical pool adds eight pairs and **−2.719198e-06 S**, which is 3.7×
below the pre-registered 1e-05 bar and 0.117× the container-break standard deviation. This closes
the *admission rule* as a lever worth a pass of its own on this generator. It does NOT close: the
price-first generator itself (pd8's own pass 3 became move 55 while this arm ran); the rate-aware
ceiling, which is a strictly dominant drop-in that costs nothing and should simply be adopted (§7);
or the same question on a different pool where byte-saving-but-cell-costing proposals are less rare
than 8 in 4,291.

**verdict_scope: n/a** for the reversion finding and the screen cure — both are positive
measurements, not negatives.

<!-- # FORMALIZATION_PENDING: a measurement pass with a fired falsifier and no candidate; the score
arithmetic used throughout is the registered S = 100*d_seg + sqrt(10*d_pose) + 25*B/37,545,489. -->

Own-vehicle frontier (unchanged by this arm):
**S 0.13603403441098336 @ 179,255 B [contest-CUDA T4 n600]** (move 55, lane
`ddm_pd8_price_first_pass3_on_move54_20260916`, archive sha `ddadf998ddacab9b…`).
