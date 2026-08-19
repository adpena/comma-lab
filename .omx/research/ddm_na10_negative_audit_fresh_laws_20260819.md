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

**One thing in it IS corrected.** The memo priced the pose leg by importing a `21×` axis
constant ("21× CUDA degradation per #1054") — the exact multiplicative transfer §2 shows is
the wrong functional form. The verdict does not depend on it (the margin is 624×), but the
*method* was the one now known to be unsound, and the memo's own puzzled axis note ("base
CPU-vs-T4 gaps here are d_seg 1.443× / d_pose 1.778×") records the confusion it caused. The
1.443× seg figure, measured independently on this body in August, is an unlooked-for
**independent confirmation of L1's 1.43× seg lineage factor** — two arms, two bodies, two
weeks apart, agreeing to 1%.


