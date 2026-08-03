---
arm: ddm_gp1
title: "Selective GT / micro-student pricing — the whole selectivity ladder, $0, real coders"
utc: 2026-08-03
research_only: true
score_claim: false
promotion_eligible: false
pointer_moved: false
pointer: "0.1910828242 [contest-CPU] UNMOVED"
axis: "[macOS-CPU cache-derived advisory] NON-PROMOTABLE"
scorer_forwards: 0
baseline_named: "live best S = 0.7910689 at 353,805 B (pu2, sha c72ef357); seg leg 0.4311790; pose leg 0.1597310; gap to PR130 floor 0.172141 = 0.6189279; 1% of gap = 0.0061893 S = 9,295 B"
verdict_scope: "FORMULATION for every negative below; INSTANCE for the cx1/v4d base"
---

# ddm_gp1 — pricing the selectivity ladder

## §0 ANSWER FIRST

The operator asked four nested selectivity questions: ship GT for **certain pairs**, within
them **certain classes**, within those **certain entities**, within those **certain
dimensions**. Priced end to end on free caches with real coders, **the ladder's rungs are
not equally real, and the productive axis is none of the four.**

1. **The address collapses, and it collapses for FREE.** The band of pixels within 1 px of
   the *receiver's own label boundary* is **3.96% of the field and contains 97.26% of all
   508,640 flips** (24.6× enrichment). The frozen scorer's margin ranking — which the
   receiver cannot compute — reaches 98.81% at 5.62%. **The free band is 98.4% as good as
   the oracle band at matched capture.** Independently reproduces `rz1`'s band geometry
   (3.971%) to 0.3%.
2. **Selecting *within* the free band is NOT free, and this corrects a live corpus number.**
   `rz1` §2.5 concluded *"addressing is free; only the nudge value costs."* Band
   **membership** is free; **selection within it** is not. MEASURED (LZMA1-raw, n600,
   150-pair like-for-like coder calibration): **0.5129 bits per band-pixel = 299,427 B**,
   which consumes **85.5% of rz1's entire 0.60 bits/band-px budget** before a single nudge
   value is sent. Any attack budgeted against 0.60 bits/px has ~0.087 bits/px of actual room.
3. **Payload is already nearly free, and the class filter — rung R2 — is what makes it
   exactly free.** H(gt | rendered) = **1.1012 bits/flip** measured. Restricted to the
   Road↔Lane edge it is **EXACTLY 0.0000 bits/flip** (a 2-class restriction: the receiver's
   own class determines the answer). If the receiver can name its own runner-up it is
   **0.1056 bits/flip** (GT is the runner-up **98.89%** of the time — reproduces `ru1`'s 98.8%).
4. **The pose rung is inverted: GT should never be shipped there at all.** `pu2` MEASURED the
   realized cost of a per-pair pose correction at **−0.5 B/pair** — correcting 6 pairs made
   the archive **3 bytes smaller**. GT is a *compress-time oracle*, not a payload. Shipping
   the whole GT pose bank (6,650 B brotli) is **strictly dominated** by solving against it and
   shipping the solved knobs (−300 B for all 600).
5. **The micro-student is not a competitor to GT-shipping — it is an address-cost enabler,
   and it is economically wide open.** Its entire job is ordering-within-band, worth
   **106,954 B = 0.07122 S**. Every candidate student is orders of magnitude under that
   ceiling (a built 20-param head: **2,674× under**; Rudin's ~1 KB named-feature head:
   **104× under**). **The P6 compliance question is not economically close — it is a fidelity
   question, and fidelity is unmeasured.**
6. **`gt2` §8's named win condition is MET.** Beating the 410,584 B implicit whole-corpus
   `L*` bound: free-band addressing **367,523 B (−10.5%)**, oracle-band **260,569 B (−36.5%)**.

**The binding caveat, stated once and applying to every S column below: these are
DESCRIPTION costs, not REALIZATION costs.** Every ΔS is BOUND-IF-REALIZED — an upper bound
on gain assuming the shipped description perfectly zeroes the addressed flips. Nothing here
builds the mechanism that turns "this band pixel should be class *c*" into an RGB change the
frozen SegNet actually argmaxes differently. `dd1` reached the same wall from the carrier
side: *"453× is the best-priced description in the corpus… It is also still only a
description."* **The description side of the seg axis is now priced and it fits. The
realization side is the wall, and it is where the next unit belongs.**

---

## §1 What was measured, and the controls

$0. **Zero SegNet/PoseNet forwards.** Inputs: cached argmax arrays (`gt_argmax_n600.npy`,
`cx1_argmax_n600.npy`), cached frozen-scorer margin fields (600 × `pair-*.npz`), and
`ddm_pz1_dpose_paired_n600_cx1_20260803.json`. Coders actually run: **brotli Q11**,
**LZMA1 FORMAT_RAW** (`lc=0,lp=0,pb=0`, PR95-family L24 form). Exact combinatorial bounds
via `log2 C(n,k)`.

**Three independent positive controls, all PASS:**

| control | this arm | prior receipt | agreement |
|---|---:|---:|---|
| total flips / d_seg (cx1 base) | 508,640 / 0.004311794704861111 | `pu2` 508,640 / 0.00431179 | **EXACT** |
| GT is the render's runner-up class | 98.89% | `ru1` 98.8% | 0.09 pp |
| receiver label-boundary band, r=1 | 3.959% of field | `rz1` 3.971% | 0.3% |
| Road↔Lane flip count | 235,148 | `pu2` §10.3 235,148 | **EXACT** |

**A bug I caught and fixed before it reached this table.** The first free-band run
accumulated real-coder bits over 150 pairs but divided by the full 600-pair band, so
`min(real, bound)` selected a 4×-understated number and reported 143,298 B where the bound
alone is ~351 kB. Fixed to compare the coder against the *same subset's* bound and scale the
population estimate by the measured ratio (**1.0520**). All bytes below are post-fix and
hand-reconciled.

**#875 prefix control (the coder subset is a contiguous prefix, so this is mandatory).**
Governing quantity = flips/pair. Prefix n=150 mean **857.35** vs population **847.73**,
**ratio 1.0113** — representative, so scaling the population bound by the subset's
real/bound ratio is sound. (The seg axis is only mildly skewed, mean/median **1.050**; the
pose axis is **4.6×** skewed and a prefix there would *not* be representative — see §4.4.)
The n=40 subset used for the pass-2 coder table is 0.9144 of population mean, but that table
reports a *ratio of two quantities on the same subset*, which is robust to subset difficulty.

**Band-source honesty.** The margin field is the **frozen scorer's** (`distill_logit_margin`,
`qa75_solve` render, 99.9% GT-agreeing) — *not* receiver-legal (73 MB scorer). ORACLE rows are
therefore bounds. The FREE band is computed from that render's **argmax**, a label field of
exactly the kind a receiver holds after decoding — a *proxy* for the decoder's own `L*`, not
`L*` itself. `rz1` measured the real thing and got the same band size; the capture numbers
below are still labelled BOUND for that reason.

---

## §2 THE UNIFIED LADDER

All rows: `net ΔS = rate_cost − gross_gain`; **negative = score goes down = worth doing**.
`rate = 25·B/37,545,489`. `%gap` against 0.6189279. Every gross column BOUND-IF-REALIZED.

| # | rung | addressing mode | bytes | rate S | gross S | **net S** | **%gap** | realized? |
|---|---|---|---:|---:|---:|---:|---:|---|
| **A1** | R2 seg, ALL classes | ORACLE margin rank q=0.05 | 307,893 | 0.20501 | 0.42914 | **−0.22412** | **36.21%** | needs student + realization |
| **A2** | R2 seg, ALL classes | ORACLE margin rank q=0.02 | 260,569 | 0.17350 | 0.39314 | **−0.21964** | **35.49%** | needs student + realization |
| **A3** | R2 seg, ALL classes | **FREE** label-boundary r=1 | 367,523 | 0.24472 | 0.41938 | **−0.17466** | **28.22%** | **address legal**; realization unbuilt |
| A4 | R2 seg, ALL classes | FREE label-boundary r=2 | 408,244 | 0.27183 | 0.42604 | −0.15421 | 24.92% | as A3 |
| **B1** | R2 seg, Road↔Lane only | ORACLE q=0.02 (payload **0 bits**) | 130,089 | 0.08662 | 0.18279 | **−0.09617** | **15.54%** | needs student + realization |
| **B2** | R2 seg, Road↔Lane only | **FREE** r=1 (payload **0 bits**) | 174,120 | 0.11594 | 0.19526 | **−0.07932** | **12.82%** | **address legal**; realization unbuilt |
| B3 | R2 seg, Road↔Lane only | ORACLE q=0.05 | 177,704 | 0.11833 | 0.19862 | −0.08030 | 12.97% | as B1 |
| B4 | *comparator* — gc16 explicit edge fix at `W` | realization-priced | 299,369 | 0.19933 | 0.19900 | ≈0 | ~0% | measured, break-even |
| C1 | R2 seg, Movable only | ORACLE q=0.02 | 80,714 | 0.05374 | 0.08668 | −0.03294 | 5.32% | as B1 |
| C2 | R2 seg, Movable only | ORACLE q=0.05 | 110,534 | 0.07360 | 0.10081 | −0.02721 | 4.40% | as B1 |
| **D1** | **R1 pose, top-6 pairs** | pair index (`hs1` 21 B for 30) | **−3** | −2.0e−6 | 0.06129 | **−0.06129** | **9.90%** | **REALIZED (pu2 banked −0.0354283)** |
| D2 | R1 pose, top-10 pairs | pair index | −5 | −3.3e−6 | 0.06831 | −0.06831 | 11.04% | idealized-to-zero bound |
| D3 | R1 pose, top-112 pairs | pair index | −56 | −3.7e−5 | 0.09516 | −0.09516 | 15.38% | idealized-to-zero bound |
| D4 | R1 pose, all 600 | pair index | −300 | −2.0e−4 | 0.15973 | −0.15993 | 25.84% | idealized-to-zero bound |
| **D5** | **R1 pose — pu2's MEASURED realizable total** | — | ~0 | ~0 | — | **−0.03916** | **5.99%** | **−0.0354283 already banked** |
| E1 | R4a pose, ship whole GT bank (600×6 fp16) | none needed | 6,650 | 0.00443 | 0.15973 | −0.15530 | 25.09% | **DOMINATED by D4** |
| **F1** | **R5 micro-student** | — | **≤106,954** | ≤0.07122 | *enables A2 over A3* | **break-even ceiling** | — | fidelity UNMEASURED |
| G1 | *reference* — gt2 implicit whole-corpus `L*` | one coder context | 410,584 | 0.27342 | 0.43118 | −0.15776 | 25.49% | the bound A3/A2 beat |

**R3 (per-entity) is deliberately not given a byte row here** — codex arm `sg3x` owns surgical
per-island/interface GT-artifact shipping. This arm contributes only the **address-overhead
scaling** so the rung is placed (§3.3).

---

## §3 Rung by rung

### 3.1 R1 — per-pair selection

**Seg: per-pair selectivity buys essentially nothing.** Flip mass is nearly uniform across
pairs — top-6 pairs hold **2.10%**, top-25 **7.05%**, top-112 **25.07%**, top-200 **41.02%**
(a third of the pairs holding 41% of the mass = 1.23× uniform). Selecting pairs moves you
*along* the same RD line, not to a better one: cost and gain scale together. This
independently confirms `ru1`'s conclusion — *"cell selection is the strategy"*, not pair
selection — and `hs1`'s §3 measurement that per-pair heterogeneity is the quantity that is
**absent** on the seg axis.

**Pose: per-pair selectivity is everything, and the bytes are negative.** Pair 74 alone is
**30.92%** of the pose axis; 10 pairs are 67.24%. And `pu2` MEASURED that correcting 6 pairs
made the archive **3 bytes smaller** (−0.5 B/pair): the correct pose sits nearer the `dim0`
offset (32.1875) so its fp16 residual entropy-codes better. **There is no rate budget to
clear on the pose rung.** The binding cost is solver wall-clock (~30 s/pair single-start,
~250 s/pair 6-start; n600 6-start ≈ 42 h, per-pair independent and parallelisable).

Rows D2–D4 are **idealized-to-zero bounds** and `pu2` has already refuted that idealization:
its measured floors reach only 0.4581× (pair 74) and 0.9231× (pair 523) of shipped, giving a
realization efficiency of **0.0354261 / 0.0612850 = 57.8%** on the measured tail, and its
control group showed the multi-start defect is **tail-specific, not population-wide**. **D5
is the honest row: −0.0391625 total (5.99% of gap), of which −0.0354283 is already banked.**
Only ~0.6% of gap remains on that route.

### 3.2 R2 — per-class within selected pairs

**The class filter's real value is not byte reduction — it is that it drives PAYLOAD to
exactly zero.** Measured H(gt | rendered):

| restriction | flips | share | **payload bits/flip** |
|---|---:|---:|---:|
| ALL | 508,640 | 100% | 1.1012 |
| **Road↔Lane edge** | **235,148** | 46.23% | **0.0000** |
| Lane involved | 236,816 | 46.56% | 0.0184 |
| Movable involved | 119,933 | 23.58% | 0.3757 |

A 2-class restriction is *self-describing*: the receiver knows its own class, so the target
is determined. This is a genuinely free lunch and it makes **rung R4b redundant** (§3.4).

**The class-filter address tax is zero when the band is shared.** The band is derived from
the receiver's label field regardless of which classes we correct, so a class filter adds no
address cost — it only shrinks the selected subset. Restricting to Road↔Lane at the same free
band takes 367,523 B → 174,120 B (−52.6%) for 46.5% of the gain: **very slightly
sub-proportional**, i.e. the filter is close to RD-neutral and is worth taking only for the
payload-zeroing and for footprint control.

**Against the operator's named comparator:** gc16's explicit Road↔Lane fix is **299,369 B for
0.199 S** (235,148 flips × W=1.2731 B/flip), which is break-even and dead. On the *identical*
235,148-flip object, cheap addressing gives **174,120 B free-band (0.582×)** and **130,089 B
oracle-band (0.435×)**. **Cheap addressing turns a dead break-even row into −0.079 S (free)
or −0.096 S (oracle).** This is the single clearest instance of the operator's steer: the
comparator was priced at realization cost `W`, and the description costs 2.3× less.

### 3.3 R3 — per-entity (address-overhead scaling only; `sg3x` owns the rest)

From `dd1`'s census (components/frame: Lane 27.64, Movable 3.68, Road 2.11, Undrivable 1.08,
MyCar 1.00 = **35.51 total**):

| granularity | address cost | vs pixel coords |
|---|---:|---:|
| pixel (naive) | 17.585 bits | 1.00× |
| entity-within-frame | **5.15 bits** | **3.41×** cheaper |
| Lane-word + arc-length offset (`dd1` aspect 25.48) | **7.27 bits** | 2.42× cheaper |
| Lane word only (2,813 ≥64px comps ⇒ 4.69/frame) | 2.23 bits | 7.9× cheaper |

**But the address is the cheap part and the *extent* is not.** `gt2`'s lossless Lane polygon
program is 130,960 B over 16,581 components = **7.90 B = 63.2 bits per component** — an order
of magnitude more than any address above. **Entity granularity buys ~3.4× on a term that is
already ≤15% of the cost; it does not touch the term that dominates.** That is why the
entity rung is worth pursuing as *shape carriers* (`dd1`'s 703 B Lane perpendicular-offset
carrier, 453× vs W; `sg3x`'s surgical islands) and **not** as an addressing device.

### 3.4 R4 — per-dimension

**R4a (pose, 6 scored dims): NOT PRICEABLE from cached data. Blocker named.** The per-dim
decomposition of scored `d_pose` does not exist anywhere on disk, and the naive
reconstruction from `(shipped − GT)²` is **invalid** — it disagrees with authoritative
per-pair `d_pose` by **0.59×–94×**, because the shipped 6-vector is a *warp-knob* vector, not
PoseNet's output. `pu2` §4.4 additionally measured that per-dim deviations have **no**
predictive power (`|r| ≤ 0.095`, dim0 `r² ≈ 0.000`). Getting a real per-dim answer requires
frozen PoseNet on decoded frames — not cached, not $0.

What *is* known per-coordinate is the **search** headroom (`mq1`): p2 **0.8743%**, p1
**0.4694%**, beta **0.3358%**, p0 0.1412% of gap, total ≥1.82% — against a **format** ceiling
of **≤0.056%**. So the per-dimension rung is real but lives on the *search* axis, where the
byte cost is already negative (§3.1).

**R4b (seg, rank-limited correction field): SUBSUMED by R2, and its premise was
mis-stated.** My charter called #583 a "rank-4 head bank"; it is the **J-bank / corrected-J**
(Fisher/Jacobian) surface, and the DAG records the `rank-4-feature→RGB` pullback as
**absent** and the #583 candidate as **non-executable**. Independently, the thing a rank-4
correction field would buy — cheaper class naming — is already achieved: payload is 1.1012
bits/flip, **0.1056** if the receiver names its runner-up, and **exactly 0** under the R2
class filter. **There is ≤1.1 bits/flip of payload in the whole system; a low-rank logit
field cannot recover more than that, and R2 already recovers all of it on the dominant edge.**

### 3.5 R5 — micro-student / teacher

The student's job in this ladder is **ordering within the free band** (band *membership* is
already free). Its value is exactly the free→oracle byte gap:

**367,523 B (free) − 260,569 B (oracle) = 106,954 B = 0.07122 S.**

A student that *perfectly* reproduced the frozen margin ordering pays for itself iff it costs
less than 106,954 B. Candidates against that ceiling:

| candidate | bytes | margin under break-even | status |
|---|---:|---:|---|
| `LearnableConv1x1StudentHead` (20 params, RGB→5, fp16) | ~40 | **2,674×** | BUILT (cascade_b receipt, 42% KL reduction) — but as a *training* surrogate, never priced as payload |
| Rudin named-feature logistic head (gc16 estimate) | ~1,024 | **104×** | proposed, unbuilt |
| 10K params @ FP4 + 50% sparse + brotli (a1 estimate) | 5,000–15,000 | 7–21× | ESTIMATE, "interpretively risky" |

**The compliance question is not economically close.** Even a 15 KB student sits 7× under the
ceiling. What is entirely **unmeasured** is *fidelity*: how much of the 106,954 B ordering
gain survives when an approximate student produces the ranking. That single number decides
the rung, it costs one training run plus a scorer-free re-price on this exact harness, and it
is the pre-registered falsifier in §5.

---

## §4 CROSSOVERS — where the ranking flips

1. **GT-shipping vs student-shipping is a false dichotomy on the seg axis.** They are
   complements: GT supplies the *content*, the student supplies the *ordering* that makes the
   content cheap to address. The crossover is not "which one" but "is the student's fidelity
   worth 106,954 B" — and at 40 B–15 KB of cost, it needs to capture only **0.04%–14%** of
   the ordering gain to break even.
2. **On the pose axis the crossover has already happened and GT-shipping LOST.** Shipping the
   GT pose bank costs 6,650 B for a −0.1553 S bound (E1); solving against GT at compress time
   and shipping the knobs costs **−300 B** for a −0.1599 S bound (D4). **GT as oracle strictly
   dominates GT as payload — by 6,950 bytes and a better bound.** The operator's "selective
   shipping of GT" framing is the wrong frame for pose specifically.
3. **Cheap addressing flips the Road↔Lane row from dead to live.** At realization price `W`
   it is 299,369 B for 0.199 S = break-even (B4, dead). At description price it is
   174,120–130,089 B = **−0.079 to −0.096 S**. **This is the row where the operator's steer
   changes the verdict.**
4. **Per-pair selection flips sign between axes.** It is worthless on seg (2.10% in the top-6)
   and decisive on pose (30.92% in a single pair). Any ladder that prices "top-k pairs" as one
   rung across both axes is averaging two opposite populations — the `#875` trap.
5. **The free band beats the oracle band once the student's cost exceeds 106,954 B.** No
   candidate is near that, so **the free band is currently the fallback, not the plan** — but
   it is the row that is *legal today* with zero compliance ruling (A3/B2).

---

## §5 THE REALIZATION WALL — and the pre-registered falsifiers

Everything in §2 prices **descriptions**. The seg axis has no shipped mechanism that consumes
a per-pixel class correction: the receiver changes RGB, and the frozen SegNet decides argmax
from *regions*, not pixels (CLAUDE.md). Define realization efficiency

  **η = (flips actually fixed) / (flips described)**

η is **UNMEASURED on the seg axis**. Every net-ΔS in §2 scales by η. The measured anchor from
the neighbouring axis is `pu2`'s pose tail: **η_pose = 57.8%**. If η_seg were comparable,
row A3 would be −0.101 S (16.3% of gap) rather than −0.175 S; if η_seg ≤ 0.583, row A3 goes
net-positive and the rung dies.

**Pre-registered falsifiers (each kills a specific row, not the family):**

| # | falsifier | kills |
|---|---|---|
| F1 | η_seg ≤ **0.583** measured through the real receiver on ≥32 pairs | A3 (free-band ALL) |
| F2 | η_seg ≤ **0.594** on the Road↔Lane subset | B2 |
| F3 | a student under 106,954 B recovers **<14%** of the ordering gain (measured by re-running this exact harness with the student's ranking substituted for `distill_margin`) | F1 / the whole R5 rung |
| F4 | the decoder's **actual** `L*` band (not this arm's near-GT proxy) captures **<90%** of flips at r=1 | A3, B2, and rz1's 93.53% jointly |
| F5 | per-pair pose realization on non-tail pairs yields **<0.3%** of gap beyond pu2's banked row | D2–D4 (already ~refuted by pu2's control) |

**Verdict scope on every negative here: FORMULATION.** Nothing licenses a family-level
negative on GT-shipping, on students, or on entity carriers.

---

## §6 What the operator's P6 ruling unlocks, at each price point

| ruling | unlocks | value |
|---|---|---|
| **No ruling needed** | A3 / B2 — free-band addressing uses only the decoder's own label field; **zero scorer weights, zero compliance question** | −0.175 S / −0.079 S bounds, legal today |
| **Student permitted at any size ≤107 KB** | A2 / B1 — oracle-grade ordering | +0.045 S over A3 (7.3% of gap), and B1 at −0.096 S |
| **Student permitted at ~1 KB (Rudin named-feature head)** | same as above at 104× margin | the economically obvious point; **fidelity is the only open variable** |
| **Ruling declines a student entirely** | ladder is unharmed — falls back to A3/B2, which are 84% and 82% as good | the rung is **not load-bearing** for the ladder's headline |

**The ruling is worth at most 0.07122 S (11.5% of gap) and the ladder does not depend on
it.** That is the practical answer to the escalation: it is a real but bounded prize, and it
should not block A3/B2, which need no ruling at all.

---

## §7 Corrections this arm makes to the corpus

1. **`rz1` §2.5 "addressing is free; only the nudge value costs"** — refined. Band membership
   is free; **selection within the band costs 0.5129 bits/band-px = 85.5% of the stated 0.60
   bits/px budget.** (And that budget itself inherits `sx1`'s 253,341 B estimate, which `gt2`
   refuted as a coder cost at 1.6207×.)
2. **Charter premise "#583 rank-4 head bank"** — #583 is the J-bank/corrected-J surface; the
   rank-4 head is a separate named surface whose `rank-4-feature→RGB` pullback is recorded
   absent.
3. **Charter premise "QA43's ~120 B/pair counterfactual"** — that figure is `ng1`'s, not
   `qa43`'s, and it has been explicitly invalidated for the free-frame0 content class
   (`uh1`, `su2` §4, `pu1` §7). Not used as a comparator anywhere above.
4. **`gt2` §8 win condition (beat 410,584 B)** — MET, twice (367,523 B free / 260,569 B
   oracle), by exactly the boundary-conditioned addressing route gt2 named.

---

## §8 NEXT-IF-RESUMED

1. **MEASURE η_seg (F1/F2).** The single number that converts this entire table from bounds
   to expectations. Cheapest form: take the free-band correction set for ~32 pairs, realize it
   through the actual receiver, re-score. This is the arm's top follow-on and it is the only
   one that needs a scorer.
2. **F4 first, because it is $0**: recompute the free band from the decoder's **actual** `L*`
   (not this arm's near-GT proxy). Inputs are on the SSD; `rz1`'s 93.53% suggests it holds.
3. **F3 student fidelity, $0 given a student**: substitute any candidate ranking for
   `distill_margin` in `experiments/ddm_gp1_ladder_pricing.py` and re-read the free→oracle gap.
   The harness is written to accept a drop-in ranking.
4. **Do NOT re-derive**: per-pair pose economics (pu2, settled, negative bytes); per-class
   flicker floors (fl1); entity shape carriers (dd1 + sg3x).
5. **Carry the trap**: never average `rz1`'s 0.60 bits/px (budget, dilated band) with
   `sx1`/`dd1`'s 0.6984 bits/px (entropy, strict band) — different numerator *and*
   denominator.

---

## §9 Artifacts

Scripts (committed): `experiments/ddm_gp1_selective_gt_pricing.py` (enrichment + payload
entropy), `experiments/ddm_gp1_ladder_pricing.py` (RD curve, per-class, real coders),
`experiments/ddm_gp1_free_band_and_net.py` (free band, runner-up, net S).

Receipts (SSD, certify-or-block; rebuildable from the committed scripts + existing caches):
`/Volumes/VertigoDataTier/pact/ddm_gp1_20260803/` — `gp1_pass1.json`, `gp1_pass2.json`,
`gp1_pass3.json`, `gp1_per_pair_flips.npy`, `gp1_in_band.npy`, `pass3.log`.

Consumed (read-only): `ddm_pu2_20260803/argmax_cache/`, `ddm_b2b_qa75_field_20260730/`,
`.omx/research/ddm_pz1_dpose_paired_n600_cx1_20260803.json`.

**pointer 0.1910828242 [contest-CPU] UNMOVED. No score claim. No promotion.**
