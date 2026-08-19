# ddm_na10 — re-grading the negative corpus against the five laws of 2026-08-18/19

- **arm** `ddm_na10` (task #1140 — operator directive 2026-08-19: *"Audit all negative and
  mixed signal given our modern understanding"*)
- **date** 2026-08-19
- **axis** This arm measures nothing. It ADJUDICATES. Every number quoted below is
  attributed to the artifact that measured it, with that artifact's own axis label
  preserved. `score_claim=false` · `promotable=false` · no Modal · $0.
- **verdict_scope** `corpus-adjudication`. A re-grade here is a statement about whether a
  prior verdict's EVIDENCE still supports its LABEL under the five laws — never a claim
  that the re-graded family works. REOPENED means *the negative no longer binds*, not
  *the positive is proven*. Each REOPENED row carries the measurement that would settle it.
- **status** IN PROGRESS — written incrementally, committed at every stage boundary.

---

## §0 THE FIVE LAWS, VERIFIED AT SOURCE (not quoted from the charter)

My charter handed me five laws. Per the charter-recall law
([[charter_recall_validation_is_apparatus_not_volition_20260816]]) I re-derived each from
its source artifact before applying it. All five verify. Two carry refinements the charter
compressed away, and those refinements change how the audit runs.

| # | law | source, verified | refinement the charter compressed |
|---|---|---|---|
| **L1** | **GT-lineage fork.** contest-CUDA scores DALI GT; local advisory scores PyAV GT. Pose factor median **19.09×**; seg factor **1.43×**; rate identical. | memory `pose_gap_was_gt_cache_lineage_not_cuda_20260819` UPDATE 2 §1, UPDATE 3 §1; structurally confirmed at `evaluate.py:31-42`, asserted at `frame_utils.py:113/:188` | The pose factor is a **population MEDIAN**. Per-pair span **0.887 → 1,627** (fo2h §4.0c). Multiplying an advisory per-pair number by 19× is wrong on nearly every pair. **Same-lineage DIFFERENCES stay sound — lineage cancels.** |
| **L2** | **Composition law.** Seg token edits destroy pose ×387 through the photometric frame; re-running the pose carrier's own solver against the EDITED frame recovers `d_pose` to **1.073×** at ~0 bytes. | jg1 ANSWER-5/6, §S2 (the negative) + §S2b (the reversal) | The recovery is measured on **3 pairs** (1.01× / 1.34× / 0.87×), 9–12 of 12 coefficients moved, inside up2's "±4 on all 7,200 coeffs = +5 B" envelope. It is a REVERSAL, not yet a rate-closed result. |
| **L3** | **Move-class law.** On quantized stored values, **realized-acceptance lattice coordinate descent** works; gradient-step and blind-search negatives do not bind it. | up2 §5 (every LM step left the linear regime and lost; basis demands 366–10,141 code units vs int12 rail 2048) + jg1 ANSWER-4 (block/dilate realizes −55% at r=1, −351% at r=2; single-cell coordinate moves repair 1.55 cells/token, additive) | The law has now fired on **two different axes with two different actuators** (pose coefficients; seg tokens). That is what promotes it from an observation to a law. |
| **L4** | **Seg debt composition.** 95.9% of the seg leg is render/re-segment loss; stored labels are **99.9985%** correct vs DALI GT (1,714 of 117,964,800 cells). | jg1 §S1a, whole-field array comparison, both lineages | Also establishes **the shipped tokens were fit against DALI** — the encoder already targets the shipping lineage. Negatives premised on "store better labels" aimed at ≤5% of the debt. The live actuator is **pre-distortion**. |
| **L5** | **Pose variance.** Pose's estimate band is a median **13.4×** seg's at equal n — wider in **100.0%** of 2,000 shuffles (p10 5.9×, p90 27.9×). Implies ~2 orders of magnitude more pairs for equal precision. | fo2h §4.3 | fo2h ALSO records the failure it corrected: one shuffle said 21.8×. The law is the 2,000-shuffle median, not the single draw. Older sister: prefix bias inverts by axis — pose 2.5–4.2× HARDER on prefixes, seg ~0.96× ([[m96]]). |

**The two refinements that change the audit's method.** (a) L1's per-pair span means I may
NOT re-price a closed per-pair negative by multiplying by 19× — that would repeat the
error the law names. A PyAV per-pair verdict is *unmeasured on the shipping axis*, and the
only cure is to re-score it there. (b) L1's "differences stay sound" clause is the main
brake on this audit: a great many negatives are candidate-vs-baseline DIFFERENCES measured
end-to-end on ONE lineage, and those **STAND**. Only verdicts that consumed an advisory
ABSOLUTE — or transferred an advisory number across the fork by a constant — are confounded.

---

## §1 METHOD

Store-seeded, never from working memory ([[m44]]). Surfaces swept:

1. `.omx/state/canonical_task_status.jsonl` — REFUSED / CLOSED / DEAD / NO_VERDICT /
   FORMULATION rows.
2. The probe-outcome / verdict ledgers under `.omx/state/`.
3. `.omx/research/**` verdict memos — read at the RECEIPT, not the headline
   ([[corrections_land_in_bodies_headlines_keep_the_stale_number_20260805]]).
4. `tools/graph_memory_recall.py` verdict clusters.
5. The charter's nine high-prize seeds, each verified at source first.

Classification vocabulary:

- **STANDS** — the verdict's evidence still supports its label under all five laws. First
  class ([[m48]]); a rubber-stamp audit that reopens everything is worthless.
- **REOPENED** — a law removes the evidentiary basis for the label. Re-graded UNMEASURED
  (not POSITIVE). Carries: resolving measurement · consumer · cost.
- **SHARPENED** — the label survives but its SCOPE narrows, or its mechanism was
  misattributed and the correct mechanism changes what it forbids.

---

## §2 A REFINEMENT OF L1 THE AUDIT NEEDED, AND FOUND: THE LINEAGE GAP IS **ADDITIVE**, NOT MULTIPLICATIVE

I did not go looking for this. It fell out of trying to apply L1 to a verdict the charter
did not name (`wd2`, §3), and it changes how the whole corpus can be re-graded — cheaply.

**The problem with the law as stated.** L1 gives the pose lineage gap as a factor: median
19.09×, per-pair span 0.887–1,627. A factor with a 1,834× span is not a transfer rule; it
is a warning label. It tells every closed negative "your pose number is wrong" without
telling any of them *what the right number is*. Under that form, every pose negative in the
corpus becomes UNMEASURED and needs a fresh DALI-lineage run. That is a very expensive
audit outcome, and I distrusted it.

**The test.** If instead the two lineages differ by an *additive* phantom term — the GT
decode difference itself, which is a property of the GT pair and not of the candidate —
then `advisory = shipping + C`, the ratio `advisory/shipping = 1 + C/shipping` is not a
physical constant at all, and its apparent variability is fully explained by dividing a
near-fixed numerator by a varying denominator. Three bodies in the corpus carry BOTH an
advisory and a shipping `d_pose` at n600:

| body | advisory (PyAV) | shipping (DALI/T4) | **ratio** | **offset (adv − ship)** | source |
|---|---:|---:|---:|---:|---|
| up3 pointer, before solve | 1.482928e-04 | 7.769500e-06 | 19.087 | **1.405233e-04** | up2 §4c table |
| up3 pointer, after solve | 1.483534e-04 | 7.649247e-06 | 19.395 | **1.407042e-04** | up2 §4c table |
| hv1 ep0634 @182,759 B | 1.474700e-04 | 6.880000e-06 | 21.435 | **1.405900e-04** | wd2 memo (advisory) + hv1 memo line 6 (T4 pose contribution 0.0082945765) |

**The ratio spreads 12.3% across the three. The offset spreads 0.13%.** Mean
`C = 1.406058e-04`. The additive form is ~95× tighter than the multiplicative one on the
same three rows, and the two bodies are independent objects (different archives, different
byte counts, different arms).

**What this fixes.**

1. **The correct transfer is SUBTRACT `C`, never multiply or divide by ~19–21×.** Any
   closed negative that moved a pose number across the fork by a factor mis-transferred it,
   and the error grows precisely as the candidate gets *better* (the ratio inflates as the
   denominator shrinks). The campaign's own history shows this: `ddm_pi2` went looking for
   what "the 21.4× pose offset actually is" — it is `C/d_pose_true`, an artifact of the
   division, and there was never a 21.4× physical effect to explain.
2. **It explains fo2h's 0.887–1,627 per-pair span exactly**, with no new mechanism: the
   per-pair ratio is `1 + C_pair/true_pair`, so pairs with tiny true pose error show
   enormous ratios and pairs with large true error show ratios near 1. L1's per-pair
   warning is *correct* and now has its cause.
3. **It hands the audit a $0 shipping-level estimator.** For any negative that retained an
   advisory n600 pose row — and many did — the shipping-axis level is
   `d_pose_ship ≈ d_pose_adv − 1.406058e-04`, good to about ±2e-7 on these rows. **Closed
   negatives can be re-graded from bytes already on disk, with no new run.** That is the
   difference between an audit that can finish and one that just issues IOUs.

**What this does NOT fix — and I checked, because it would have been the convenient answer.**
It does **not** rescue advisory *delta* gating at the frontier. If `C` were exactly constant
then `Δadvisory = Δshipping` and the advisory would be a perfectly good gate. It is not
exactly constant: between up3's before and after rows the offset drifted `+1.81e-07`, while
the shipping delta the solve actually bought was `−1.20e-07`. **The offset drift is ~1.5× the
signal, and of the opposite sign** — which is precisely the opposite-sign result up2 §4c
measured and could not otherwise explain. So:

> **Additive in LEVEL across bodies (±0.13%); NOT additive in DELTA under a fine pose
> actuator.** `up2 §4d` — "the local advisory eval is not a valid gate for a CUDA-axis pose
> candidate" — **STANDS**, and now has its mechanism: the gate fails because the phantom
> term drifts under actuation by more than the signal, not because the axes are unrelated.

**Scope.** n=3 bodies (two independent), n600 aggregate, `d_pose` only. `verdict_scope:
formulation`. The estimator is a re-grading instrument for adjudication — it is **not** a
score, and no verdict below is *promoted* on it. It is used only to ask whether a closed
negative's margin was large enough that the lineage question cannot reach it. Where the
estimator says a margin is thin, the row is REOPENED for a real DALI-lineage measurement,
never resolved by the estimator. Falsifier: any fourth body whose offset departs from
1.406e-04 by more than ~1% breaks it, and jg1's $0 DALI seg instrument plus a DALI pose run
can produce that fourth body cheaply.

---

## §3 STANDS — the honest non-reactivations

These are first-class ([[m48]]). An audit that reopens everything has measured nothing.

### 3.1 `wd2` ep60 nested-width student — REFUSED **STANDS**, and the refusal was *understated*

Not in my charter's seed list; the memory graph surfaced it
(`tools/graph_memory_recall.py --tool keywords "pose refusal advisory lineage"`). It is the
single most consequential negative in the corpus — it killed a route projected to reach
**≈0.1480**, i.e. through the sub-0.15 goal — and its entire verdict was drawn on
`[macOS-CPU advisory]` n600 rows. Exactly the shape L1 threatens. Receipt:
`.omx/research/ddm_wd2_ep60_advisory_refusal_verdict_20260815.md`, same-instrument
confirmation block.

| leg | measured (advisory, same-instrument) | under the audit |
|---|---|---|
| seg | Δd_seg +7.4963e-04 = **7.01×** the 1.07e-04 admission bar | L1 seg factor is 1.43×; even conceding the whole factor the delta is ~4.9× over bar. **Refusal survives on the seg leg alone.** |
| pose | Δd_pose +9.1839e-02 → memo quotes pose ΔS **+0.9207** | §2 estimator: student shipping `d_pose ≈ 0.0918456`, base 6.880e-06 → pose ΔS **+0.9501**. The refusal was **understated by +0.0294 S**. |
| rate | ΔS −0.01157 (the prize) | unchanged; rate is lineage-identical (L1) |

**STANDS.** The margin is three orders of magnitude wider than any lineage effect. Applying
the sharper instrument makes the negative *stronger*, not weaker — which is the outcome that
distinguishes an audit from a rescue mission.

**Correction to my own §2 — it is a REDISCOVERY, and that is the audit's headline finding.**
See §4. `ddm_pi2` measured the additive floor on 2026-08-16, three days before the laws I
was sent to apply, to ten significant figures. My §2 arithmetic reproduces it independently
(1.406058e-04 vs pi2's 1.4061324889e-04, 0.005% apart) which is a useful *confirmation* —
but the law was already ours, and I found it by re-deriving rather than by recalling. I am
leaving §2 standing rather than deleting it, because the rediscovery **is** the evidence for
§4, and because an audit that quietly overwrote its own error would be the exact failure it
is auditing ([[save-memories-not-apologies-anti-forgetfulness]]).

**One thing in it IS corrected.** The memo priced the pose leg by importing a `21×` axis
constant ("21× CUDA degradation per #1054") — the exact multiplicative transfer §2 shows is
the wrong functional form. The verdict does not depend on it (the margin is 624×), but the
*method* was the one now known to be unsound, and the memo's own puzzled axis note ("base
CPU-vs-T4 gaps here are d_seg 1.443× / d_pose 1.778×") records the confusion it caused. The
1.443× seg figure, measured independently on this body in August, is an unlooked-for
**independent confirmation of L1's 1.43× seg lineage factor** — two arms, two bodies, two
weeks apart, agreeing to 1%.



---

## §4 THE HEADLINE — THE "MODERN UNDERSTANDING" I WAS SENT TO APPLY IS ITSELF A REDISCOVERY, IN A WORSE FORM

The operator asked me to re-grade the negative corpus *given our modern understanding*. The
audit's largest finding is about that understanding: **its pose half was already measured on
2026-08-16 by `ddm_pi2`, more completely and in a strictly better functional form, and the
2026-08-19 line re-derived it without citing it — restating as a multiplicative factor the
exact thing pi2 had explicitly refuted.**

Receipt: `.omx/research/ddm_pi2_pose_axis_attribution_20260816.md`, frontmatter + §ANSWER +
§0.1–§0.5.

### 4.1 What pi2 already had, three days earlier

| pi2 (2026-08-16) | the 08-19 line |
|---|---|
| §ANSWER decomposes the offset **exhaustively**: (A) GT-decode path `1.4061324889e-04` = **99.9960%**; (B) scorer-forward + platform `3.572e-12` = 0.0000025%; (C) our inflate's device dependence **≤ 5.576e-09 = ≤0.0040%** | up1 announces the GT-cache lineage as an "INSTRUMENT EUREKA" and localizes device-dependence to `cpr1/inflate.py:312/:335` as a live mechanism |
| §0.1 names the retained **DALI GT pose table** (sha `a91d9825…`, 117,980,732 B, `(600,6) float32`) and measures it reproducing contest-CUDA at **1.00081×** | up1 reports the $0 DALI-GT local instrument at 0.9999× as new |
| §0.3 the seg cache is **already** DALI (`d_seg` ratio 1.00021); PyAV GT costs **1.4425×** | up3 reports the seg lineage factor 1.43× as new |
| §0.2 **"Never rescale an advisory `dS_pose`. Never quote an advisory `d_pose` ratio."** Gives the exact conversion `dS_pose = sqrt(10(6.88e-06 + Δd_pose_abs)) − sqrt(10·6.88e-06)` | **L1 is stated as a ratio**: median 19.09×, per-pair span 0.887–1,627 |
| §0.4 **"Why a single multiplier would have been wrong"** — a flat multiplier is wrong by up to **52%** across four rn1 rows; the conversion matches rn1's independent absolute cross-check to **0.2%** | fo2h §4.0c presents the 1,834× per-pair span as a new discovery about the factor |

**Citation audit** (`grep -c "pi2"`): `up2` 1 (and only as *"its unbuilt …"*, i.e. dismissed
as unfinished rather than consumed as law) · `jg1` **0** · `fo2h` **0** · `jg2` **0** ·
the memory node that carries L1 into every session, `pose_gap_was_gt_cache_lineage_not_cuda_20260819`,
**0**. Occurrences of "additive floor" or `1.4061` in all five: **0**.

### 4.2 Why the form regression is not cosmetic

pi2 §0.4's arithmetic is the proof: the multiplicative form is wrong by up to **52%** on
real rows, and its error is *systematic in the direction that matters* — the ratio inflates
as a candidate improves, so the better the candidate, the worse the multiplicative transfer
mis-prices it. The additive form has no such bias; the floor cancels exactly in an absolute
delta. pi2 §0.4 also shows the practical cost: `rn1`'s ratio-based worked-through gave
`+0.0217` where the correct conversion gives `+0.027545`, and pi2's strengthened reading is
that a one-LSB dither is **2.87× the entire remaining gap**, not "25% of it."

The 08-19 line's own difficulties are downstream of the regression. fo2h §4.0c had to
discover, painfully and across 48 pairs, that the factor "is not a constant — it spans
1,834×." Under pi2's form that span needs no discovery: the per-pair ratio is
`1 + C_pair/true_pair`, so it *must* diverge wherever `true_pair` is small. A law stated in
the right variables would have predicted fo2h's headline instead of being surprised by it.

### 4.3 The correction, and what it costs

`L1` should be restated in pi2's form, with pi2 as its source, and the two artifacts
reconciled rather than one silently superseding the other:

> **L1′ (restated).** The advisory/contest split is a **GT-lineage split inside our own
> tooling**. In `d_pose` it is an **additive floor** `C = 1.4061e-04` (99.996% of the total
> offset; scorer-forward and platform terms are ≤0.004%). Quote advisory pose only as an
> **absolute Δd_pose** and convert with pi2 §0.2. **Never** a ratio, **never** a rescaled
> `dS_pose`. In `d_seg` the split is multiplicative, **1.4425×** (pi2 §0.3) / 1.43× (up3) —
> but only because our seg cache was *already* the authority decode. The fix, not the
> caveat, is to point advisory pose at the retained DALI table (pi2 §0.1).

Cost: an edit to the memory node and a DAG FEED line. **$0, no measurement.** The
measurements were all done on 08-16 and again on 08-19.

### 4.4 The class, and the honest part

This is [[m18]] (writes outrank reads) and the anti-forgetfulness backbone, firing on a
three-day horizon — the shortest-latency instance the campaign has recorded, and inside the
window where the artifact was still the newest thing in `.omx/research`. Both the 08-19 arms
and **I** hit it: my §2 re-derived pi2's constant from scratch rather than recalling it.
Three independent agents reached the same number by measurement and none by memory. That is
not a discipline failure by any one arm; it is an apparatus gap — the corpus had the law and
no surface put it in front of the arms that needed it. Sisters:
[[charter_recall_validation_is_apparatus_not_volition_20260816]] (the cure is at the spawn
site, not in anyone's volition) · [[m44]] (never recall from working memory alone) ·
[[m36]] (defer-at-source into one canonical ledger).

**The one thing the 08-19 line adds that pi2 did not have** — and it is real, so the
reconciliation is a merge and not a revert: pi2 bounded the inflate device-dependence at
≤0.0040% of the offset but did not resolve it; up2 §6 *falsified half of it* by measurement
(`pose_batch` 1-vs-64 is bit-identical over 1.83e9 pixels; only `semantic_batch` differs, at
1,326 pixels, max |Δ|=1). And jg1's DALI-scored seg instrument (0.99995×) plus the L2/L3/L4
results are new work that pi2 does not contain.

---

## §5 REOPENED — ordered by projected S value

### 5.1 `ddm_rc4` rung-4 token drop — **REOPENED**. Its own named door has since been measured OPEN.

**This is the audit's highest-value row, and it is the cleanest kind: rc4 did everything
right, stated exactly what would overturn it, and was overturned three days later by an arm
that did not know it was answering rc4.**

Receipt: `.omx/research/ddm_rc4_rung4_token_drop_verdict_20260816.md` (2026-08-16), §VERDICT.

rc4 is *not* lineage-confounded, and I want that on the record because the inventory pass
flagged it as an L1 candidate and the source refutes that. rc4 scored its pose leg against
`gt_cache_dali.pt["pose"]` **per pi2's fix**, cross-checked it against PyAV (1.7% different),
and noted the drop is a *paired differential* so the offset cancels. rc4 is the one arm in
this corpus that had already absorbed pi2. **L1 does not touch it.**

What refused it was the pose leg, and rc4 named its single open door precisely:

> "The charter's own composition — qs5's **in-compile frame-0 Schur compensation** — is
> untested at this amplitude and is the one door left. It must cancel **99.807%** of the pose
> perturbation to make the rung net-negative … What is unmeasured is its reach at this
> amplitude and the carrier's re-coding cost across all 600 pairs."

rc4 even derived the structural case that jg1 later confirmed — *"6 pose equations, 12 free
coefficients per pair."* It was right, and it correctly refused to claim it without measuring.

**jg1 §S2b measured that reach on 2026-08-19.** Same mechanism (token edit perturbs frame 1;
the 12-coefficient carrier re-solves against the perturbed frame), realized acceptance,
DALI targets:

| pair | orig `d_pose` | damaged | re-solved | cancellation |
|---|---:|---:|---:|---:|
| 283 | 1.0989e-05 | 9.0402e-03 | 1.1098e-05 | 99.9988% |
| 468 | 4.2551e-06 | 4.4506e-04 | 5.7074e-06 | 99.6705% |
| 513 | 2.3061e-06 | 5.4207e-04 | 2.0027e-06 | 100.0562% |
| **aggregate** (the `avg_posenet_dist` quantity) | | | | **99.9874%** |

**99.9874% measured vs 99.807% required — the door clears by +0.18 pp.** And the amplitude
objection rc4 raised is answered by coincidence rather than by design: rc4's damage is
`3.3279e-03`; jg1's mean per-pair damage is `3.3366e-03`. **Ratio 1.00×.** jg1 tested the
compensation at, to three figures, exactly the amplitude rc4 said was untested.

**Classification: REOPENED — UNMEASURED, not positive.** The prize is real: rc4's rate+seg
leg is `−3.243e-3 S` with the *rate leg exact*, which is 34% of hv1's gap to 0.15 and **50%
of the live jg2 base's 0.006526 gap**.

**What is honestly still open, and it is not small:**

1. **n = 3.** Pair 468 alone reaches only 99.6705% — *below* rc4's bar. The aggregate clears
   because pair 283 dominates the sum. Under **L5** (pose band 13.4× seg's; ~100× the pairs
   for equal precision) a 3-pair aggregate on the pose axis is exactly where estimates
   wander, and the margin is 0.18 pp.
2. **Two different bodies.** rc4 prices hv1 @182,759 B; jg1 measures on the up3 pointer
   @176,420 B. Same carrier architecture, different archive.
3. **rc4's second unmeasured item is still unmeasured** — "the carrier's re-coding cost
   across all 600 pairs." jg1 bounds it by up2's envelope (±4 on all 7,200 coefficients =
   +5 B) but did not encode it.
4. **Wall-clock.** jg1's re-solve is 39–64 s/pair ⇒ **6.5–10.7 h for n600**. That is a real
   budget item, not a footnote.

**Resolving measurement:** run the jg1 carrier re-solve against rc4's retained rung-4
token-drop deltas, on ≥60 seeded-random pairs (not 3, not a prefix), aggregate by
ratio-of-sums, and report the realized cancellation with its band. **Consumer:** the jg2
chain (§6) — it is already building this exact actuator, so the marginal cost is the pair
count, not new code. **Cost:** $0, ~40–65 min of local CPU at n=60; rc4's deltas are retained
at `/Volumes/APDataStore/pact/ddm_rc4_rung4_token_drop_20260816/`.

### 5.2 The already-registered re-measures that never fired — **REOPENED as a queue, not as verdicts**

The inventory pass surfaced something worth more than any single re-grade: **at least eight
prior negative audits already exist** (`na1` 08-02 · `na2` 08-03 · `na5` 08-09 · `na7` 08-14 ·
`nx1`/`ns1` 08-16 · `ns2` 08-17 · `na9` 08-18, plus `rv1` 07-28 and the 07-07/07-10 re-audits),
and several of their outputs are **registered re-measures that were never fired**:

| row | registered | state |
|---|---|---|
| `#ddm_na2_strided_rerun_four_pose_family_verdicts_20260803` — 4 pose-family verdicts resting on n8/n24 **contiguous prefix** docs | 08-03 | pending, never fired |
| `ddm_na5_pose_stratified_texture_representative_n120_blocked` | 08-09 | DEFER — harness absent |
| `ddm_na5_pose_mladder_representative_n120_blocked` | 08-09 | DEFER — `pose_mladder.py` + A2 logs absent |
| `ddm_na5_pose_l2_truedepth_representative_n120_blocked` | 08-09 | DEFER — depth cache absent |
| `ddm_na5_pose_carrier_arms_representative_n120_blocked` | 08-09 | DEFER — source parity |
| `na5_prefix_bias_ratio_rederivation_and_pose_source_parity_block` | 08-09 | DEFER — `source_parity_not_slot_availability` |

Every one is a **pose** re-measure blocked on **source parity or absent harness** — and L1′
plus jg1's $0 DALI instruments (pose table at 1.00081×, seg at 0.99995×) are precisely the
missing apparatus. **The blocker class these rows were parked on has dissolved.**

I am explicitly *not* re-grading the four na2 pose-family verdicts here. na2 already graded
them correctly (STANDING on weak evidence, reopenable by a seeded random n≥120) and the
honest state is that the re-run was owed and never happened. **Adding a ninth audit on top of
eight is the wrong move; draining this queue is the right one.** Consumer: whichever arm owns
the pose axis next. Cost: the n≥120 DALI-lineage re-runs are now $0 local, gated only on
rebuilding the absent harnesses.

### 5.3 Candidates carried forward, not adjudicated

The inventory ranked 30; I verified two at source within this arm's budget (`wd2` → STANDS,
`rc4` → REOPENED). The remainder are **carried as candidates with their triggering evidence
recorded, not as re-grades** — asserting a verdict I did not verify at source would be the
failure this audit exists to catch. Highest-ranked unverified, all **L2** (pose damage from a
token/row edit priced with no carrier re-solve), all now answerable by the same §5.1
measurement: `qw1/mp2 deep prune` (pose leg UNMEASURED, projected +0.027) · `hv2 rank-3`
(REFUSED at +0.0362 S) · `b2e` edit-replay ×3 (n=50, **L5**) · `ra2c` rank-4 (its own memo
finds the damage law over-predicts 9.23×) · `td1` token-drop Schur (explicitly "modeled
−4.4e-4 *assumes no* compensation").

### 5.4 `ps135b` / `ps1u` pose-carrier terminals — **REOPENED, instrument-confounded** — and this collapses `pu3`'s convergence claim

**The mechanism, settled by content-hash not by argument.** The constant `qs1.GT_POSE` — the
GT pose table the whole pose-solve chain minimizes against — resolves to sha
`82ed61ce6a11a6612502527fbb6864a22fe6c6099312e637d971214ab660fb27`, which
`src/tac/gt_lineage_registry.json` classifies as **`PYAV_YUV420_TO_RGB`**, evidence
`EMPIRICAL_NEAREST_RULER_POSE_MSE`, measurement *"pose: MSE **1.406151e-04** vs DALI ruler,
**4.889832e-12** vs AV ruler."* It is the AV table to eleven decimal places. The registry
also records exactly one artifact named `gt_first6_n600.npy` and it is `DALI_NVDEC` (sha
`1f2fe6d1…`) — the basename collision `gl1` was built to catch.

**A fourth independent route to `C`.** That registry line measures the PyAV↔DALI GT distance
*directly*, table against table: `1.406151e-04`. Compare §2's 3-body offset `1.406058e-04`,
pi2's `1.4061324889e-04`, and the sister sweep's independent `MSE(GT_PyAV, GT_DALI) =
1.406149e-04`. **Four routes, four methods, agreeing to 0.007%.** This is no longer an
empirical fit: `C` **is** the mean-squared distance between our two GT pose tables, which is
why it is additive in level and a property of the clip rather than of any candidate. L1′ is
now mechanically closed.

**What that did to `ps135` pass 4.** The receipt is
`.omx/research/ddm_hc1_hy1_container_push_20260812.md:198` — *"pass 4's `d_pose=9.67e-06`
CPU advisory state realizes as `1.4674e-04` on CUDA."* Read it against the base (PyAV
`1.4747e-04` / DALI `6.886e-06`): the carrier drove the **PyAV** residual down 15× and landed
on the shipping axis at 104.4% of the table distance. **It swapped the two residuals.** The
verdict recorded — *"ps135 pass4 pose carrier REFUTED"*, a genuine `[contest-CUDA T4, n600]`
row at `S 0.192632768165` — is a true measurement of a carrier **aimed at the wrong target**.
This is not a carrier that lacks pose authority; it is one with ample authority, pointed 19×
away from the objective that ships.

**`ps1u` carries a second, independent defect.** Its `top_mass_pairs` selector
(`experiments/ddm_ps1u_uncapped_pose_solve.py:131-141`) ranks pairs by **PyAV** residual
mass. Measured rank correlation against the DALI ranking: **Spearman 0.1222, top-30 overlap
1 of 30 (3.3%)**. The solve spent its whole budget on a nearly disjoint pair set from the one
carrying shipping-axis debt. Its T4 row (`d_pose 6.146e-05`) sits 40.8% of the way from base
toward the wrong target — the signature of a solver working correctly on the wrong problem.

**The consequence that matters most — `pu3`'s 6/6 collapses to 4/6 + one instance.** `pu3` §6
concluded *"no measured carrier-addressable pose headroom exists on the shipping object"* from
**"six distinct mechanisms, all negative,"** and explicitly discharged the same-defect law:
*"these do not share a defect — refit, overlay, carrier swap, GN solve, basis, and iteration
budget are different mechanisms."* **The actuators differ; the objective does not.** `pk4`,
`ps135b` and `ps1u` all minimize against `qs1.GT_POSE`. Under pu3's own cited law
([[same_defect_negatives_masquerade_as_family_convergence_20260805]]) those three are **one
instance, not three**. `ra3` **STANDS** — it priced its 35.5× refusal on the authority GT
(`6.88559506e-06`, 1.00081× contest-CUDA) and is clean. So the convergence that licensed the
campaign's belief that pose is closed rests on a materially thinner base than it claimed.

**Blast radius, and it is the [[m56]] shape — the cure is built but unwired.** `pi2` landed
the lineage finding on **2026-08-16** (`ed153d0203`).
`experiments/ddm_qs1_frame0_schur_coupled_solve.py` was last touched **2026-08-13** and has
never been repointed. `gt_cache_dali.pt` *is* wired into the diagnostic and rescore tools
(pi2, rc4, ra3-rescore, sg2) and into the successor solver `ddm_up2_shipping_pose_solve.py` —
but **not into the solve chain that drew these verdicts**. Files loading the PyAV constant:
`pk3 · pk4 · ps1u · qs1 · qs2 · qs5 · mc35`, plus `ddm_mt1_978_multitoken_screen.py:79` and
`ddm_mt1_modal_multitoken_sign_gate.py:100` which hardcode the PyAV path. **Note `qs5` is on
that list** — and `qs5` is the compensation route `rc4` §5.1 named as its one open door, and
that `pk4`'s routing block calls "qs5-proven."

**Classification: REOPENED — UNMEASURED.** I am claiming the negatives do not bind, not that
headroom exists. The sister sweep re-scored `pk4`'s retained heldout vectors against the DALI
table at $0 and the **sign did not change** — so the *tested, PyAV-fit* candidates fail on
the shipping axis too (that is why `pk3/pk4` below are SHARPENED, not reopened). A
DALI-*targeted* solve is a different, unmeasured object.

**Resolving measurement, two rungs.** (1) **$0 diagnosis:** `d_pose` is a pure function of
(generated pose6, GT pose6), so re-scoring `ps135` pass-4's retained pose6 against
`gt_cache_dali.pt["pose"]` is arithmetic on retained bytes — it confirms the floor and
recovers nothing. (2) **The real answer:** re-run the pass-4 solve with **both** the objective
and the pair selection pointed at the DALI table, then one T4 row. Local solve $0; **one T4
row ≈ $0.16.** **Consumer:** the `pu3` §6 verdict, the `pk4` routing block, and the js1/js8
joint line that every pose refusal currently reinforces.

### 5.5 Three verdicts that **STAND** on re-examination

- **`et1` address-band closure — STANDS.** The pose-viability question the charter raised
  does not bite: η is a **ratio** on one field (lineage cancels), the `0.61491` bar is
  **DERIVED** from the band's own bytes (no lineage, no n), and et1 had already quarantined
  its pose leg as subset-scoped after its own self-catch. The shortfall is 2.04× / 12.7σ and
  break-even *rises* monotonically with radius, so it is settled from the rate side without
  the realizer. [[m96]] makes it *more* secure, not less: et1 records these pairs as 0.2692×
  of population, i.e. 3.7× **easier** on pose, so a representative sample lowers the
  pose-viable η further.
- **`pk3` / `pk4` frame-0 linear overlays — STANDS, SHARPENED.** Re-scored on DALI at $0:
  every rung still hurts or does nothing (rung 42 −7.57e-06, rung 250 0.0, rung 1000
  −3.93e-05). **What narrows:** the fit target was 95.4% artifact (heldout base MSE 5.22e-05
  PyAV vs 6.83e-06 DALI, 7.6× inflated), and "differences cancel" does **not** rescue it —
  the gate metric carries `g` linearly and the candidate was *produced* by solving against
  `g`. Read the ceiling as scoped to *linear overlays fit against a PyAV-contaminated
  residual*, not to the formulation. Settling re-run: repoint `qs1.GT_POSE`, recompute stages
  20–40 (the 64-pair Jacobian bank is retained and GT-independent) — **$0**.
- **`sa1` family closure — STANDS, but the charter's premise about it is REFUTED.** My
  charter believed sa1 closed on a T4 row. It did not: all three candidate rows are
  `[macOS-CPU advisory n600, env-mismatch grade]`; the only T4 anchor is the *base*. It
  stands anyway, and for the right reason — sa1 bought a same-instrument base leg and
  adjudicates **deltas** against it. Deflating each pose leg to the shipping operating point
  (the √ makes the inflation exactly √21.416 = **4.63×**) gives net ΔS ≈ +0.0328 / +0.0124 /
  +0.0096 against a −3.5e-6 bar: refusals survive by 3–4 orders of magnitude. **Sharpening
  that matters operationally:** the carried-forward damage law "68–512× the rate credit"
  becomes **~15–110×** on the shipping axis, so sa1's reactivation criterion (~250×
  sub-linear damage for keep87) currently holds successors to a bar **4.6× stricter than the
  shipping axis requires**. Second sharpening: sa1 line 19 attributes its 21.4× to
  "CPU-vs-T4 drift on the SAME bytes"; the sister sweep reproduced 21.416× by swapping
  **only the GT table** with pose vectors held fixed, and pi2 bounded true forward drift at
  3.572e-12. Mechanism misattributed; verdict unaffected (sa1 predates up1 by one day).

---

## §6 THE CONSOLIDATED RE-GRADE TABLE

Every row was verified at its **receipt**. Rows marked ⚠ are ones where a charter premise (mine
or the sweep's) was **wrong at source** and the audit had to refuse the reopening it expected.

| # | verdict | original scope | law(s) | class | resolving measurement | consumer | cost |
|---|---|---|---|---|---|---|---|
| 1 | **`rc4` rung-4 token drop** REFUSED on pose 517× | FORMULATION, uncompensated | **L2** | **REOPENED** | jg1 re-solve vs rc4's retained deltas, ≥60 seeded-random pairs, ratio-of-sums | jg2 §S2 | $0, 40–65 min |
| 2 | **`ps135b` pass-4 carrier** REFUTED (real T4 row) | INSTANCE | **L1′** | **REOPENED** | re-run solve w/ objective **and** pair-selection on DALI, then 1 T4 row | pu3 §6, pk4 routing, js1/js8 | $0 + **$0.16** |
| 3 | **`ps1u` r2** REFUSED +1.686e-02 | INSTANCE | **L1′+L3** | **REOPENED** | same; selector `top_mass_pairs` is PyAV-ranked (Spearman 0.122 vs DALI, 1/30 overlap) | pu3 §6 | $0 + $0.16 |
| 4 | **`sq2` R8 pose guard** BLOCKED, 113.1× damage | INSTANCE | **L2** | **REOPENED** | re-run R8 with carrier re-solve, same 32 pairs | sq-line | $0, ~4 h |
| 5 | **`js6b`** 200/200 held, 0 survivors | FORMULATION, *unprojected* bank | **L2+L5** | **REOPENED** | re-screen with compensated pose envelope + qs2's measured 5.667 B/pair | js-line | **$0, minutes** |
| 6 | **`qs3`** post-mortem ×3 | INSTANCE, "one missing authority field" | **L1′** | **REOPENED** | verify local DALI seg argmax vs retained T4 field, then run attribution | qs2 step-2, cb1 re-scores | **$0** |
| 7 | **`qs2` dead-zone** steps 2–4 | INSTANCE | **L1′** | **REOPENED** | 1 T4 dual-axis row on the step-2 lattice | qs-line | $0.16 (or $0 after #6) |
| 8 | **`cb1` Lane band** REJECTED +22.7 d_pose | INSTANCE | **L1′+L2** | **REOPENED** | re-take with carrier re-solve in the loop; Lane = 53% of enemy | cb1 | $0 |
| 9 | **`cb1` MyCar hood** ADMITTED −0.051646 | INSTANCE | **L1′+L18** | **REOPENED** ⚠ *a reopened **win**, the uncomfortable direction* — 98% of the ΔS is an ancestor-vehicle pose absolute (implied base `d_pose ≈ 31`) | re-score on DALI | cb1 | $0, ~140 s |
| 10 | **`pg1`-GN celldrop50** 3.06× over break-even | instance (a table row, not a verdict) | **L1′+L3+L5** | **REOPENED** — margin 3.06× **smaller than the fork**; double-inflated (AV + hardest-first) | re-score 113 pairs on DALI | pg1 | **$0, ~140 s** |
| 11 | `wd2` ep60 student REFUSED | instance | L1 | **STANDS** — margin 624×; estimator makes it **worse** (+0.9501 vs +0.9207) | — | — | — |
| 12 | `et1` address-band DEAD | FORMULATION | L1/L5 | **STANDS** — η is a ratio; bar is DERIVED; m96 makes it *more* secure | — | — | — |
| 13 | `sa1` family closure REFUSED 3/3 | family | L1′ | **STANDS** ⚠ charter premise refuted: **not** a T4 row, all advisory — stands anyway on same-instrument deltas (√ deflation 4.63× ⇒ +0.0328/+0.0124/+0.0096 vs −3.5e-6 bar) | — | — | — |
| 14 | `pk3`/`pk4` frame-0 overlays GATE_FAIL | FORMULATION | L1′+L5 | **SHARPENED** — re-scored on DALI at $0, **sign unchanged**; but fit target was 95.4% artifact ⇒ scope narrows to *PyAV-fit* overlays | optional DALI re-fit | pk-line | $0 |
| 15 | `qs1` headline REFUSED +2.4257e-5 | instance | L1 (cancels) | **STANDS** — matched-base T4 dual-axis, all three legs one instrument | — | — | — |
| 16 | `qs2` fired candidate / `qs2` R2 ADMITTED | component | L1 (cancels) | **STANDS** ×2 | — | — | — |
| 17 | `fp1` f′ ≥ 0.008305 INSTANCE-DEAD | FORMULATION | L1 (cancels) | **STANDS** — floor and 0.0051 threshold are the *same* AV cache; rider: never quote against a contest-axis seg number | — | — | — |
| 18 | `pj1` f_photometric CONFOUNDED | FORMULATION | L1 (rider) | **STANDS** — an internal-invalidity finding; no cross-lineage comparison exists | — | — | — |
| 19 | `lc1` per-edge labels −12,884 | FORMULATION/ROUTING | **L4 corroborates** | **STANDS** ⚠ both charter premises wrong: n32 was **seeded stratified, not a prefix** (m88 1.0100×), and there is **no pose leg at all** — so L2 and L5 cannot apply. 32/32 pairs worsened (p≈2.3e-10) | — | — | — |
| 20 | `cb1` per-class carrier NO_VERDICT | — | none | **STANDS** ⚠ charter premise wrong: the blocker is **byte-close / receiver-rate custody**, not pose measurement. **No law builds an encoder.** | — | — | — |
| 21 | `sq1` F4 r=1 capture 0.8668 < 0.90 | FORMULATION | **L3+L4 corroborate** | **STANDS** — the cap is *move-class-independent by construction* (an upper bound on any r=1 realizer, "independent of eta") | — | — | — |
| 22 | `pg1`-GN ep854 723× over | formulation | L1 (magnitude only) | **STANDS** — no correction reaches 723×; rider: the magnitude is an AV absolute and may **not** be divided by 19.09× | — | — | — |
| 23 | `ra3` 35.5× refusal | instance | — | **STANDS** — already priced on the authority GT (1.00081×). The one clean row in pu3's six. | — | — | — |
| 24 | `pg1` P7 yuv6-null projection cure | QUEUED, never fired | L2 | **RE-ROUTE** — `js6b`/JS4 measured 8.836e-4 residual leakage after first-order nulling; `pz1` measured null-space membership not surviving a lattice change (attenuation 1.662×). L2's re-solve is the *measured-working* alternative. Prefer it before spending the MLX-host run. | pg1 | — |
| 25 | `na2` 4 pose-family verdicts + `na5` ×5 blocked re-measures | STANDING-on-weak-evidence | L1′+L5 | **REOPENED as a QUEUE** — every one is a pose re-measure parked on *source parity / absent harness*; that blocker class has dissolved | whichever arm owns pose next | $0 local, gated on rebuilding harnesses |

**Tally: 24 verdicts adjudicated at receipt — 13 STANDS · 10 REOPENED · 1 SHARPENED-only · 1
RE-ROUTE**, plus one reopened queue of 6 registered-but-never-fired re-measures, drawn from an
inventory of **207 negative probe-ledger rows · 93 keyword-negative task rows · ~57 deferral
D-rows · 158 strict verdict-negative memos**. Four charter premises (⚠) were refuted at source
and the expected reopenings refused.

**The audit is not a rubber stamp in either direction.** More than half the adjudicated rows
stand, two stand *more strongly* under the sharper instrument (`wd2` +0.0294 worse; `et1`
tighter under m96), and three of the campaign's own arms — `sq1`, `rc4`, `qs2` — turn out to
have discovered or correctly flagged these laws **before** they were named. `sq1` (08-03)
measured L3's move-class contrast and L4's 100%-render-loss decomposition sixteen days early;
`rc4` (08-16) had already absorbed pi2 and stated its own overturning condition to three
decimals; `qs2` (08-13) observed the 21.4× fork empirically and correctly refused to act on it.

---

## §7 WHAT SHOULD CHANGE THE LIVE `jg2` CHAIN'S COURSE — IMMEDIATELY

`jg2` (task #1139) is live at S1, base `S = 0.15652626435208142`, gap **0.006526**.

**First, the reassurance, because it is the thing most at risk.** I checked
`experiments/ddm_jg1_seg_solve.py` at source: it loads `gt_cache_dali.pt` (`:86`) with the AV
cache present only as the labelled comparison (`:89`). **jg1's actuator and jg1's L2 carrier
re-solve are DALI-targeted. The chain jg2 inherits is clean.** That is not luck — jg1 read the
receiver at source rather than borrowing constants.

Four items, in order of urgency:

1. **⚠ Do NOT borrow `qs5` machinery without repointing it.** `rc4` named "qs5's in-compile
   frame-0 Schur compensation" as its open door, and `pk4`'s routing block calls the route
   "qs5-proven" — but `experiments/ddm_qs5_resolve_compensation.py:183,783` targets
   **`qs1.GT_POSE`, the PyAV table**. Any jg2 reach for qs5 code inherits a 19× misaimed
   objective. jg2's own re-solve (jg1's, DALI) is the correct actuator; keep it.
2. **Fold `rc4` into S2 — it is a 50%-of-gap rate lever that jg2's own actuator unlocks.**
   rc4's rung-4 rate leg is **exact** and its rate+seg gain is **−3.243e-3 S = 50% of jg2's
   0.006526 gap**. It was refused only on the uncompensated pose leg, and §5.1 shows jg1's
   re-solve clears rc4's stated 99.807% bar (99.987%) at 1.00× the amplitude. The deltas are
   retained. jg2 is already building the exact machinery; the marginal cost is pair count, not
   code.
3. **Correct one leg of jg1's §S2 retro-explanation — and note it *reinforces* jg2's plan.**
   jg1 wrote that `qs1`'s refusal was "read as the token actuator being weak" when really "the
   pose coupling is what refuses it." At the qs1 receipt the legs are seg `−2.712674e-5`, pose
   `+1.126177e-7`, rate `+5.127114e-5` — **the rate leg is 455× the pose leg.** qs1's pose was
   already cured by Schur compensation; qs1 refused **on rate**, exactly as its own text says.
   The claim holds for `js8`/`vd1`, not for `qs1`. **Consequence for jg2: rate is the binding
   leg on this axis, which is precisely why S1 (measure the real rate) before S2 is the right
   ordering.** The audit endorses jg2's stage order.
   *And a positive closure jg2 may cite:* qs1 pre-registered "get compensation ≤6.8 B/pair" as
   its reactivation lever. It paid 12.83 B/pair; qs2 got 5.667; jg1's re-solve moves 9–12
   **already-shipped** coefficients, which up2 measured the Rice stream absorbing at +5 B for
   all 7,200 ≈ **0.83 B/pair** — satisfying qs1's own lever by an order of magnitude.
4. **Print the POSE recovery ratio with its band at every rung, not just seg realized-vs-projected.**
   jg2's honesty rail currently prints the seg leg. The pose leg is the underpowered one:
   the 1.073× recovery is **n=3**, and pair 468 alone reaches only 99.6705% — *below* rc4's
   bar. Under **L5** (pose band 13.4× seg's; ~100× the pairs for equal precision) this is the
   leg that wanders. Use **seeded-random pairs**, aggregate by **ratio-of-sums** (not mean of
   ratios), and report the band. The asymmetry matters: at a 1.5× recovery instead of 1.073×
   the pose cost rises to ~+0.00197 S ≈ **25% of the gap**; at 2× it is ~55%.
   **Budget note:** the re-solve is 39–64 s/pair ⇒ **6.5–10.7 h for n600.** That is an S2
   schedule item, not a footnote.

---

## §8 THE ONE APPARATUS CHANGE THIS AUDIT ASKS FOR

Not a ninth audit. **§4 shows the corpus already held the pose law and no surface put it in
front of the arms that needed it** — pi2 landed 08-16, `qs1_frame0_schur_coupled_solve.py` was
last touched 08-13 and was never repointed, and three independent agents (two 08-19 arms and
me) re-derived the same constant by measurement rather than by recall. `src/tac/gt_lineage_registry.json`
already exists and already answers the question **by content hash**. What is missing is that
nothing *consults* it at the moment an arm loads a GT table.

**Ask:** a fail-closed GT-lineage assertion at the load site — any experiment that loads a GT
pose or seg table resolves its sha against the registry and refuses (or loudly labels) a
lineage that does not match the objective the arm declares. `up1` already named "build/keep the
fail-closed GT-lineage gate" as owed. This audit supplies its business case: **five of the ten
reopened rows above exist only because that gate does not.**

Sisters: [[m56]] (unwired-but-built — the cure existed for three days and reached nothing) ·
[[charter_recall_validation_is_apparatus_not_volition_20260816]] · [[m18]].
