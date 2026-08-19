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

*(sections filled incrementally below)*
