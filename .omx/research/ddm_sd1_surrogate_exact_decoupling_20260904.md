# DDM SD1 — where the expected-flip surrogate mis-prices the exact argmax

**Date:** 2026-09-04
**Arm:** `ddm_sd1_surrogate_exact_decoupling`
**Axis:** `[macOS-CPU advisory; retained EMA-shadow scorer logits; frozen CPU-torch SegNet argmax; not contest authority]`
**Disposition:** **MEASURED / PRE-REGISTERED PREDICTION FALSIFIED IN DIRECTION / THE DEFECT IS NOT ON vr1's LIST**

## Result first

**The QBR1 loss did not mis-price the argmax. It mis-priced the CLOCK.** The whole −37.7% fall in
`seg_expected_flip_realized` decomposes EXACTLY into a **−40.54% τ-schedule leg** and a **+4.85%
field leg**, and the two multiply back to the observed number with residual **0.000e+00**:

```
(1 − 0.405364) × (1 + 0.048453) − 1 = −0.376553      observed own-τ change = −0.376553
```

The field got **worse** over the same window (+8.56% exact flips) and the loss still fell, because
`tau_for_step` walked τ from 0.15 to 0.05 and `sigmoid(-m/τ)` deflates by 40.5% on a **frozen field**
when it does. Both cells reproduce it to four decimals (control −40.5364/+4.8453, treatment
−40.5364/+4.8307).

Hold τ fixed and the surrogate is a **faithful** directional proxy: at τ = 0.05 it **peaks at step
2,000 — the same milestone as the exact term** — its net sign matches the exact term in **5 of 5**
windows, and per-edge sign agreement covers **97.2–99.98%** of the excursion mass (control;
95.2–100.0% treatment). Under the anneal
the net surrogate delta is **negative in all 5 windows**, including the two where the exact error
rose by +3,223 and +911 flips.

So the charter's framing — "the surrogate mis-prices the argmax, and vr1 rows 1/3/4 are the
calibration cures" — is **half right and half wrong, and the wrong half is the load-bearing half.**
The surrogate's *functional form* is sound. Its *schedule* is the defect, and **none of vr1 rows
1, 3 or 4 touches it**: all three re-weight the same annealed sigmoid.

Two further MEASURED facts that do rank the vr1 rows:

* **67–85% of the seg gradient is spent on pixels that are already correct** (85.1% at τ = 0.15,
  67.2% at τ = 0.05). The loss is mostly defending, not repairing.
* **The excursion is rare-class over-paint.** Lane predicted-area/GT-area went 1.0334 → **1.0929**
  and Movable 1.0259 → **1.0580**, both peaking at **step 2,000** — the exact milestone of the d_seg
  peak — with Road absorbing the loss (0.99828 → 0.99593, trough also at 2,000). qbt1's dual ascent
  is one-sided by construction and has nothing that caps this.

## Verified at source (every premise carries `path:line`)

| claim | evidence | label |
|---|---|---|
| the surrogate is `sigmoid(-(z_target − max_{c≠target} z_c)/τ)`, pixel-mean, HT-weighted over pairs | `experiments/ddm_qbt1_qbflow_trainer.py:527-548` | MEASURED |
| τ is a single global scalar annealed linearly 0.15 → 0.05 over 5,000 updates | `ddm_qbt1_qbflow_trainer.py:626`; applied `ddm_qbr1_born_fairform_burn_prep.py:545-550` | MEASURED |
| the exact per-pair `d_seg` is `mean(argmax != target)` on the retained arrays | `ddm_qbt1_qbflow_trainer.py:1933` | MEASURED |
| the milestone retains `segnet_logits_f16 (5,384,512)`, `segnet_argmax_u8`, `target_argmax_u8`, `camera_pair_u8` per pair — the exact argmax is RETAINED, not merely reproducible | `ddm_qbt1_qbflow_trainer.py:1907-1936` (`_retain_eval_outputs`) | MEASURED |
| the milestone forward runs under `ema_scope` — it is the **EMA shadow**, while the training objective is the **live-weights** forward | `ddm_qbr1_born_fairform_burn_prep.py:420-439` vs `ddm_qbt1_qbflow_trainer.py:643` | MEASURED |
| a milestone materializes **only** `qbt.SELECTION_IDS` (n32); the 568 unfitted pairs are not retained at any milestone | `ddm_qbr1_born_fairform_burn_prep.py:433` (`qbt.pair_chunks(qbt.SELECTION_IDS, 16)`); `ls realized/*.npz` = 32 at all 6 milestones | MEASURED |
| the vehicle's target is the **PyAV** cache `gt_n600.npz` | `ddm_qbt1_qbflow_trainer.py:123` `GT_CACHE`; `_target_arrays` :1865-1871 | MEASURED |
| the retained `target_argmax_u8` is byte-equal to `gt_n600.npz['lstars'][pair_id]` at all 32 pairs × 6 milestones × 2 cells | this arm, `vehicle_target_matches_gt_cache: true` in every milestone record | MEASURED |
| DALI authority `gt_cache_dali.pt` sha `a91d9825…`, lineage `DALI_NVDEC`; PyAV `gt_n600.npz` sha `cf8d8360…`, lineage `PYAV_YUV420_TO_RGB`; 20,671 argmax sites differ over n600 | `tac.gt_lineage.assert_gt_lineage`, both reads | MEASURED |
| `δ_R = 0.021881818771362305` (n600 R-chain margin noise floor) | `.omx/research/arm_final_messages/ddm_dr1_final_20260903T221804Z.md:3` | TRANSFERRED (dr1, n600, same frozen R chain) |
| vr1 row 1 = `_live_margin_weight`; row 3 = Chan-Vese one-sided area cap; row 4 = per-class-pair margin normalization from the rank-4 head (2.185× flipdist spread) | `.omx/research/ddm_vr1_v7_v11_signal_recall_20260903.md:137,139,140` | TRANSFERRED |
| qbt1's live constraint set is RECALL-ONLY dual ascent on Lane and Movable — no area cap exists | `ddm_qbt1_qbflow_trainer.py:593-618` `dual_ascent_margin_constraints`; `MARGIN_CONSTRAINT_LANE_MOVABLE` bounds | MEASURED |

## Calibration receipts (the instrument, before any finding)

1. **Arithmetic, exact.** Recomputing HT-weighted `d_seg` from the retained argmax reproduces the
   milestone's own `d_seg_hat` with difference **0.00e+00** at all 6 milestones in both cells, and
   per-pair `recorded − recomputed = 0.0` at all 32 × 6 × 2 = 384 pair-milestones.
2. **Differential against the objective itself.** `test_surrogate_matches_the_trainers_own_loss_exactly`
   calls `qbt.expected_flip_margin_loss` and asserts this instrument's surrogate matches it to
   `rel=1e-6` at τ ∈ {0.15, 0.11, 0.05}. The instrument cannot drift from the loss it decomposes.
3. **float16 storage leg, MEASURED not assumed.** The retained logits are cast to `<f2`
   (`ddm_qbt1_qbflow_trainer.py:1918`). Re-deriving the argmax from the stored f16 disagrees with the
   retained f32 argmax at **744 sites over 6 control milestones = 1.97e-05 of pixels**, and the cast
   manufactures exact ties that make `margin < 0` undercount flips by **~0.7% of the flip count**.
   **Cure applied:** the flip indicator and the edge label are taken from the RETAINED argmax, so
   per-edge exact counts sum to the recorded `d_seg` bit-for-bit (asserted). `margin < 0` is kept as
   the diagnostic. Every price ratio below therefore has an exact denominator.
4. **Both GT lineages, never mixed.** DALI (authority) and PyAV (what the loss actually saw) are
   computed separately. On this n32 selection DALI/PyAV `d_seg` = **1.011–1.017×** — a small offset,
   not a confound: the excursion has the same shape and the same step-2,000 peak on both.

## 1. The τ-schedule leg is the whole decoupling (MEASURED)

Control cell, authority DALI target, n32 HT-weighted where marked.

| step | τ_eval | exact flips | surrogate @ own τ | price ratio | phantom % of surrogate | grad on already-correct px |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.15000 | 16,551 | 32,313.3 | 1.9523 | 62.17% | **85.10%** |
| 1,000 | 0.13000 | 19,774 | 30,430.9 | 1.5389 | 50.75% | 79.41% |
| 2,000 | 0.10999 | **20,685** | 28,232.5 | 1.3649 | 42.70% | 75.75% |
| 3,000 | 0.08999 | 20,093 | 25,456.6 | 1.2669 | 36.66% | 72.95% |
| 4,000 | 0.06998 | 19,020 | 22,712.0 | 1.1941 | 30.70% | 70.55% |
| 5,000 | 0.05000 | 17,967 | **20,145.7** | 1.1213 | 23.39% | 67.18% |

The exact term peaks at 2,000 and ends +8.56% above start. The surrogate falls monotonically and
ends at its minimum. Now hold τ fixed (same bins, same field, only the reported temperature changes):

| step | Δ exact | Δ surrogate @ τ=0.05 | Δ surrogate @ τ=0.10 | Δ surrogate @ τ=0.15 | Δ surrogate @ own τ |
|---:|---:|---:|---:|---:|---:|
| 1,000 | +19.47% | **+13.11%** | +6.28% | +2.49% | −5.83% |
| 2,000 | **+24.98%** | **+16.94%** | +7.90% | +2.79% | −12.63% |
| 3,000 | +21.40% | +14.05% | +5.65% | +0.99% | −21.22% |
| 4,000 | +14.92% | +9.39% | +3.00% | −0.59% | −29.71% |
| 5,000 | +8.56% | +4.85% | +0.53% | −1.82% | **−37.66%** |

At τ = 0.05 the surrogate has the **right sign at every milestone** and **peaks at step 2,000**, the
same peak as the exact term; its magnitude is compressed to ≈0.57× the exact relative move. At
τ = 0.15 the coupling collapses to ≈0.11× and inverts by step 4,000. **A large τ is blind and a
falling τ is actively misleading.** DERIVED corollary: the surrogate's *sensitivity* to the field is
itself a function of τ, so annealing τ changes both the level AND the gain of the reported loss — a
single scalar doing two jobs.

**τ in physical units.** δ_R = 0.02188 logits is the roundtrip noise floor. τ = 0.15 is **6.86 δ_R**
and its gradient half-max sits at **12.08 δ_R**; τ = 0.05 is 2.29 δ_R with half-max at **4.03 δ_R**.
The loss's soft band is 4–12× wider than the band in which the classification is actually undecided.

## 2. The mis-pricing map, per edge (MEASURED)

`price_ratio = surrogate mass / exact flips` on the (GT class → competitor class) edge. Control,
step 0, τ = 0.15, authority DALI, edges with ≥50 flips:

| edge | exact flips | % of d_seg | price ratio |
|---|---:|---:|---:|
| MyCar→Road | 536 | 3.24% | **3.767** |
| Road→MyCar | 650 | 3.93% | **3.528** |
| Movable→Road | 295 | 1.78% | 3.154 |
| Road→Undrivable | 912 | 5.51% | 2.685 |
| Undrivable→Road | 1,097 | 6.63% | 2.530 |
| Movable→Undrivable | 474 | 2.86% | 2.072 |
| Lane→Road | 4,225 | 25.53% | 1.826 |
| **Road→Lane** | **5,511** | **33.30%** | **1.638** |
| **Undrivable→Movable** | **1,191** | **7.20%** | **1.441** |
| Road→Movable | 1,587 | 9.59% | 1.419 |

**The map's shape:** the price ratio is essentially *(near-boundary pixel count) / (flip count)* per
edge. The long smooth majority boundaries — the ego-hood contour Road↔MyCar and the horizon
Road↔Undrivable — carry enormous near-boundary pixel mass and almost no error, so the scalar τ
over-charges them 2.5–3.8×. The genuinely hard edges (Road↔Lane, Road↔Movable, Undrivable↔Movable)
are the **best**-calibrated, at 1.42–1.83×.

The spread is a function of τ and **not** of training progress (control, ≥1000-flip support floor):

| step | spread @ own τ | @ τ=0.15 | @ τ=0.10 | @ τ=0.05 |
|---:|---:|---:|---:|---:|
| 0 | 1.7836 | 1.7836 | 1.5259 | 1.2050 |
| 2,000 | 1.6204 | 1.9194 | 1.5426 | 1.1858 |
| 5,000 | **1.1895** | 1.8534 | 1.5370 | 1.1895 |

At fixed τ the spread is flat across the whole run (1.78–2.02 at 0.15; 1.19–1.23 at 0.05). As τ → 0
the sigmoid approaches the step function and every edge's ratio converges to 1 regardless of its
logit scale.

### The pre-registered falsifier, read out

> *Prediction (from vr1 row 4): the Undrivable↔Movable and Road↔Lane edges are mis-scaled ≥2× by the
> scalar τ. Falsifier: the per-edge ratio spread across edges < 1.3× (then τ is not the defect and
> row 1's spatial weight is the next race).*

**Verdict: the prediction is FALSIFIED IN DIRECTION; the magnitude falsifier is τ-dependent and
fires at the terminal τ.**

* **Direction — FALSIFIED.** The two named edges are the **least** mis-priced of the ten supported
  edges (Road→Lane 1.638, Undrivable→Movable 1.441), not ≥2×. The ≥2× edges are Road↔MyCar
  (3.77 / 3.53), Movable→Road (3.15) and Road↔Undrivable (2.69 / 2.53) — majority-vs-majority
  boundaries carrying 3–7% of d_seg each. The prediction inverted the rank.
* **Magnitude — τ-dependent, which the pre-registration did not anticipate.** Spread is
  **1.78–2.02× at τ = 0.15** (does NOT fire) and **1.19–1.23× at τ = 0.05** (FIRES, below 1.3).
  The anneal walks the falsifier from not-fired to fired without the field changing.
* Where the prediction was RIGHT: a scalar τ *is* mis-scaled across edges, and row 4's normalization
  is **directionally correct** — equalizing the ratios strips phantom mass preferentially from the
  four over-priced majority edges (17.3% of d_seg) and thereby shifts relative gradient weight
  toward Road↔Lane / Lane↔Road / Road↔Movable / Undrivable↔Movable, which carry **75.6%** of d_seg.
  It is the right direction with 1.19–2.0× of headroom, shrinking as the anneal proceeds.

## 3. Bands: the excursion is decided error, not roundtrip noise (MEASURED)

|margin| capture curve, control, authority DALI, band half-width `k · δ_R`:

| k | pixel share | flip capture @ step 0 | enrichment | **share of the 0→2,000 net excursion** |
|---:|---:|---:|---:|---:|
| 1 | 0.0741% | 13.04% | 176× | **1.50%** |
| 2 | 0.1517% | 24.59% | 162× | 3.36% |
| 5 | 0.3859% | 47.56% | 123× | 23.90% |
| 10 | 0.8029% | 65.45% | 82× | 57.01% |
| **25** | **2.12%** | **76.44%** | **36×** | **91.92%** |
| 50 | 4.39% | 80.01% | 18× | 96.49% |
| 100 | 12.48% | 86.17% | 6.9× | 98.67% |

The **δ_R annulus itself (k = 1) contains only 1.5% of the excursion.** The damage is not in the
roundtrip's noise band — it is at **|margin| between 2 δ_R and 25 δ_R**, i.e. errors the field
decided, by a margin 2–25× larger than uint8 noise could undo. Reproduced in the treatment cell
(k=25 captures 95.31% of its 0→2,000 excursion and 97.82% of its 2,000→5,000 recovery).

**Self-correction, recorded because it would have mis-ranked row 1 by ~7×:** my first read used the
k = 1 band alone and scored row 1's coverage at 8.9%. That is the narrowest possible setting of a
margin-magnitude allocator, i.e. a LOWER BOUND, not the answer. The reach curve is the honest
reading and it puts row 1 at 91.9% on 2% of the pixels.

## 4. The generating mechanism: rare-class over-paint (MEASURED)

Predicted-area / GT-area per class, control, authority DALI, over the 32 pairs:

| step | Road | Lane | Undrivable | Movable | MyCar |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.99828 | 1.03339 | 0.99972 | 1.02591 | 1.00007 |
| 1,000 | 0.99633 | 1.08721 | 0.99942 | 1.04939 | 1.00001 |
| **2,000** | **0.99593** | **1.09291** | 0.99932 | **1.05801** | 1.00001 |
| 5,000 | 0.99694 | 1.06288 | 0.99958 | 1.04261 | 1.00003 |

Lane and Movable over-paint, peaking at **step 2,000**; Road is the sink, troughing at **step
2,000**. Mass conserves exactly. Change in predicted area 0 → 2,000, as a fraction of the frame
(MEASURED on this n32 selection, whose GT areas are Road 23.12% / Lane 0.597% / Undrivable 49.58% /
Movable 1.246% / MyCar 25.45%):

| | Road | Lane | Undrivable | Movable | MyCar | sum |
|---|---:|---:|---:|---:|---:|---:|
| Δ predicted area | −5.430e-04 | **+3.551e-04** | −1.966e-04 | **+3.999e-04** | −0.154e-04 | −9.1e-20 |

The rare classes gain **+7.550e-04** and the majority classes lose **−7.550e-04** — identical to the
last digit, because every pixel carries exactly one label. The treatment cell repeats the shape
(Lane 1.03339 → 1.11199 peak at 2,000; Road trough at 2,000).

This is exactly the configuration vr1 row 3 describes: `dual_ascent_margin_constraints`
(`ddm_qbt1_qbflow_trainer.py:593`) is **recall-only** — it pushes Lane and Movable within-class error
DOWN and nothing caps the area they take. **The excursion is that uncapped pressure, measured on this
vehicle.** By the like-for-like per-edge metric, **75.9–83.5%** of the excursion mass sits on edges
whose *winning* class is Lane or Movable (control; treatment 72.0–83.8%).

## 5. The DERIVED ranking of vr1 rows 1 / 3 / 4

Coverage fractions on the common denominator Σ_e |Δ exact flips_e| (control, authority DALI):

| window | row 4 (sign-agreeing, own τ) | row 4 (fixed τ) | row 1 (k=1 band) | row 1 (k=25, net) | row 3 (rare competitor) |
|---|---:|---:|---:|---:|---:|
| 0 → 1,000 | 0.9374 | 0.9998 | 0.0888 | — | 0.8260 |
| 1,000 → 2,000 | 0.6617 | 0.9722 | 0.1121 | — | 0.7590 |
| 0 → 2,000 (net) | — | — | — | **0.9192** | — |

**Ranking, and the caveat that outranks all three.**

**0. Not on the list — make the reported objective τ-invariant.** No row 1/3/4 variant removes a
−40.54% multiplicative deflation applied to the loss by its own schedule; all three re-weight the
same annealed sigmoid. The cheapest cure is free and is not a training change at all: **report
`seg_expected_flip_realized` at a fixed reference τ alongside the annealed training value.** At
τ = 0.05 the reported number would have peaked at step 2,000 with the exact term instead of falling
monotonically through it, and the excursion would have been visible in `history.jsonl` in real time.

**1. Row 3 (Chan-Vese one-sided area cap) — highest, and the only one that is causal.** It covers
75.9–83.5% of the excursion mass, its precondition is MEASURED present on this vehicle (Lane 1.093×,
Movable 1.058× at peak), and its peak is **co-located with the d_seg peak at step 2,000**. It acts on
the mechanism that generated the excursion rather than on the pricing of the result. It also closes
a structural asymmetry that is visible in the source: the dual is one-sided.

**2. Row 1 (per-pixel margin weight) — highest raw reach, but parameterized.** At k = 25 it sees
91.9% (control) / 95.3% (treatment) of the excursion on 2.0% of pixels at 36× enrichment. Two
honest deductions: its coverage is set by a free band width (8.9% at k = 1), and it re-allocates the
gradient rather than removing the pressure that creates the flips. Note it does address the
"85% of gradient on already-correct pixels" finding directly, which row 3 does not.

**3. Row 4 (per-edge τ from the rank-4 head) — real, directionally correct, and smallest.** Measured
headroom is a 1.78–2.02× ratio spread at τ = 0.15 collapsing to **1.19× at τ = 0.05, below the
charter's own 1.3× threshold**. Its value is concentrated in early training and it is worth least
exactly where d_seg mass is greatest. Cost is one frozen 5×5 constant matrix, so the EV is not zero
— but it is a second-order fix to a first-order schedule problem.

Coverage caveats that travel with these numbers: the three fractions **overlap** (a pixel can be
both near-margin and rare-class over-paint) — they are not a partition. The row 4 figure is an
**UPPER BOUND**: it asks only whether a positive per-edge scale *could* preserve the correct sign
per edge; a single static matrix cannot know which edge to boost at which step.

## Scope and limits (these travel with the numbers)

* **verdict_scope = FORMULATION.** The τ-schedule finding is scoped to the sealed QBR1 objective
  (`expected_flip_margin_loss` + linear `tau_for_step` 0.15→0.05 over 5,000 updates) on the QBF1 born
  vehicle, seed 20260902, n32 sealed selection. It is a statement about a temperature-annealed
  sigmoid surrogate over a piecewise-constant argmax field; it does not close the surrogate family.
* **Axis.** `[macOS-CPU advisory]`. Every input is the burn's own `[macOS-MPS n32 stratified
  advisory]` retained payload. **No score claim. Nothing here is promotable.** The pointer is untouched.
* **The OBJECT gap is real and is NOT absorbed.** My two quantities come from the **same** retained
  logits array (the EMA shadow), so the calibration finding is confound-free. But the run's own
  headline decoupling additionally straddles a live-weights loss and an EMA-shadow milestone. This
  arm bounds the calibration half exactly and leaves the live-vs-shadow half **UNMEASURED** — it
  would need the live-weights logits, which the milestone does not retain.
* **n = 2 cells, 1 seed.** Control and treatment of seed 20260902 agree to four decimals on the
  schedule leg. That is a same-seed repeat, not a seed replication.
* **Unfitted pairs: STRUCTURALLY UNAVAILABLE, not skipped.** `_evaluate_milestone` materializes only
  `qbt.SELECTION_IDS` (`ddm_qbr1_born_fairform_burn_prep.py:433`), so the 568 unfitted pairs have no
  retained logits at any milestone. Reaching them needs a re-render through the frozen CPU SegNet
  from a milestone-aligned checkpoint, and checkpoints exist only at multiples of 16 — **step 2,000
  has one; steps 0, 1,000 and 5,000 do not.** Every number here is the TRAINED n32 population and is
  labelled as such; it is never mixed with an unfitted read.
* **float16 storage** perturbs the argmax at 1.97e-05 of sites and the `margin<0` flip count by ~0.7%.
  Cured for the denominators (§Calibration 3); it remains a ~0.7% caveat on the phantom/recovered split.
* **δ_R is TRANSFERRED** from dr1's n600 measurement on the same frozen R chain, not re-measured here.
* **The band and edge tables are read at n32**, whose DALI-vs-PyAV `d_seg` offset (1.011–1.017×) is
  much smaller than the n600 population figure (1.4425×) quoted in `tac.gt_lineage`. That is a
  selection-composition fact and a reason not to transfer these absolute levels to n600.

## Equations leg (`tac.canonical_equations`)

**`scalar_top1_top2_margin_is_exact_distance_to_flip_v1` — CONSUMED, IN-DOMAIN on its scorer half.**
The law ("gap13 ≥ gap12 at all 118M pixels ⇒ the scalar top1−top2 margin IS the exact distance to
flip") is the premise that makes this entire decomposition exact: it is why `1[margin < 0]` is the
exact flip term and why the surrogate and the exact cost are two functionals of ONE scalar field.
Confirmed on this vehicle: the margin sign reproduced the recorded `d_seg` bit-for-bit at 384
pair-milestones. **REFINEMENT owed to the law, MEASURED here:** the identity is exact in float32 but
**breaks at 1.97e-05 of sites under float16 storage**, which manufactures exact ties — a numerical
caveat on the law's realization, not on the law.

**No anchor appended.** The law's `domain_of_validity.vehicle` is
`softmax_of_sdf_levelset_witness`; QBF1-born is a different vehicle sharing only the frozen scorer.
Appending a QBR1 anchor would be the cross-vehicle transfer the campaign has extincted
(`[[m21]]` constants→laws, `[[m143]]` cross-regime transfer, `[[L18]]` ancestor lessons-not-numbers).

**`segnet_head_rank4_linear_flipdist_v1` — TESTED AS A PREDICTION, not consumed.** Its downstream
inference (vr1 row 4: the 2.185× flipdist spread ⇒ the scalar τ mis-scales edges ≥2×) is measured
here as a realized price-ratio spread of **1.78–2.02× at τ = 0.15 and 1.19× at τ = 0.05**, with the
rank of the named edges **inverted**. The law itself is untouched — it describes the head's geometry
in feature space; what is refined is the transfer from feature-space flip distance to realized
surrogate mis-pricing, which the anneal dominates. No anchor appended, same reason.

**FORMALIZATION_PENDING** — the law this finding needs does not exist:

> *A temperature-annealed sigmoid surrogate `sigmoid(-m/τ)` over a piecewise-constant argmax field
> reports a loss whose change factorizes exactly as (schedule leg × field leg), where the schedule
> leg on a frozen field is a monotone deflation determined by τ and the realized margin distribution
> alone. Whenever |schedule leg| exceeds |field leg|, the reported loss is sign-decoupled from the
> exact error, and the surrogate's gain with respect to the field is itself τ-dependent — so one
> scalar sets both the level and the sensitivity of the reported objective.*

It should be registered once a τ-invariant reporting cell has been burned, so it is anchored on a
measurement of the cure rather than on this diagnosis. Its callable is already implicit in
`experiments/ddm_sd1_surrogate_exact_map.py::global_split` + `accumulate_pair`.

## GESTALT-DELTA

The campaign's gestalt held that the born trainer's loss was *mis-calibrated across the argmax
boundary* — the wrong pixels weighted, the wrong edges scaled. **That is not what is broken.** At any
fixed temperature the surrogate tracks the exact argmax faithfully in sign (5/5 windows, 97–100% of
edge mass) and peaks at the same milestone. What is broken is that the objective's *units* move
during training: the schedule deflates the loss by 40.5% on a frozen field, which is 8.4× the field's
own +4.85% signal over the same window, so a monotone-falling loss is compatible with a monotone-worsening
field. This is the `[[m157]]` "solve the objective on the shipping axis" genus at the TIME axis
rather than the metric axis: the reported quantity was not comparable to its own earlier values.
Add to the gestalt: **any annealed surrogate must be reported at a fixed reference temperature, or its
time series is not a time series of anything.** Second delta: `[[m131]]` (Lane = 0.59% of area but
the dominant share of the demand) gains a mechanism — on this vehicle the Lane/Movable pressure is
one-sided by construction, and the measured over-paint peak is co-located with the d_seg peak, so
the rare-class axis and the excursion axis are the same axis.

## NEXT_IF_RESUMED

* **`TAU-INVARIANT-REPORTING-CELL`** — owner MAIN. **Zero training change**: emit
  `seg_expected_flip_realized` at a fixed reference τ (0.05) beside the annealed value in
  `history.jsonl`. Cost ~1 extra sigmoid per update. Fire trigger: the next QBR1-lineage burn.
  This is the highest-value item in this memo and it is nearly free. It makes falsifier 2 of the ng1
  design executable in real time instead of post-hoc.
* **`ROW-3-AREA-CAP-RACE`** — owner MAIN; the DERIVED first race. Add the one-sided
  `relu(A_c − A_c^GT)²` cap beside the existing recall-only dual at
  `ddm_qbt1_qbflow_trainer.py:593`. Pre-registerable falsifier from this arm: if the cap works, Lane
  part_frac stays below **1.0929** and Movable below **1.0580** at step 2,000, and the exact peak
  moves off step 2,000 or falls below 20,685 flips.
* **`ROW-1-BAND-WIDTH-IS-THE-KNOB`** — if row 1 is raced, the band width is the lever, not the
  allocator shape: k = 1 sees 1.5% of the excursion, k = 25 sees 91.9%. A row-1 race that does not
  sweep the width is measuring the wrong parameter.
* **`LIVE-VS-SHADOW-RESIDUAL`** — UNMEASURED. Needs the live-weights logits at a milestone, which the
  sealed loop does not retain. Cheapest form: log the live-weights `d_seg` beside the EMA one at each
  milestone (one extra forward), which also converts this arm's one confessed gap into a receipt.
* **`UNFITTED-POPULATION-AT-STEP-2000`** — the only milestone with an exactly aligned checkpoint
  (`periodic_002000.pt`). A 568-pair CPU re-render there would test whether the rare-class over-paint
  is a fitted-selection artefact or a field property. Estimated cost: hours of CPU, so it needs the
  Metal or a budget decision — it is NOT the "$0 if cheap" the charter hoped for.

## DEAD-ENDS

* **"The surrogate's functional form mis-prices the argmax" is CLOSED for this objective.** At fixed
  τ it has the correct sign in 5/5 windows, peaks at the correct milestone, and agrees per-edge on
  97.2–100.0% of the excursion mass. Do not re-open the sigmoid-vs-step question.
* **"Per-edge τ is the defect" is CLOSED as a first-order claim** — the spread is 1.19× at the
  terminal τ, below the charter's own 1.3× falsifier, and the two edges the prediction named are the
  best-calibrated of the ten supported edges. It survives as a second-order lever with 1.19–2.0×
  headroom, concentrated early.
* **"The excursion lives in the δ_R roundtrip annulus" is CLOSED** — the k=1 band holds 1.5% of it.
  The damage is decided error at 2–25 δ_R.
* **"Read row 1's reach off the δ_R band" is CLOSED as a method** — it understates the family ~7×.
* **Re-rendering the milestone argmax is CLOSED as unnecessary** — `_retain_eval_outputs` already
  retains it, plus the full 5-class logits. ar1's instrument stays the render reference and supplied
  the GT-lineage loader; nothing was rebuilt.

## Custody (ALWAYS KEEP THE PAYLOAD)

Store root `/Volumes/APDataStore/pact/ddm_sd1_surrogate_decoupling/` (43 MB total; nothing written
under the live chain's `runs/`, `authorized_configs/` or `CHAIN_LEDGER.jsonl`, and the claims ledger
was not touched).

| artifact | path |
|---|---|
| control report (all tables, 1.7 MB) | `measure/control_native100/SD1_REPORT.json` |
| control per-pair rows (192 rows, sha `29bf0813…`) | `measure/control_native100/pair_rows.jsonl` |
| treatment report + rows | `measure/treatment_zero_native/{SD1_REPORT.json,pair_rows.jsonl}` |
| retained fields — margin f32, competitor, realized+retained argmax, both targets — 4 pairs × 6 milestones × 2 cells = **48 npz**, sha256 each in the report | `measure/<cell>/fields/step_*_pair_*_fields.npz` |
| launcher manifests + `safe_run` receipts (exit 0, 56.4 s, peak RSS 1,531 MiB) | `launch_measure_v2/` |
| instrument + 34 tests | `experiments/ddm_sd1_surrogate_exact_map.py`, `src/tac/tests/test_ddm_sd1_surrogate_exact_map.py` (commit `038f2d81c`) |

Every measurement ran CPU-only under `tools/launch_detached_process.py` at `nice 10` with
`torch.set_num_threads(4)`; the Metal lane stayed with the live QBR1 chain throughout and the
seed-20260902 run directory was opened read-only.

**Own-vehicle frontier: NOT MOVED** — this arm measures a decomposition; it cannot move one.
`afr1 S 0.14797617125559104 @ 180,002 B [contest-CUDA T4 n600] — UNMOVED`.
