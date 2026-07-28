# DDM-OC1 — the coherent-order codec, the ξ-temporal PREDICT stage MEASURED, and the crux it relocates

**Date:** 2026-07-27 · **Arm:** `ddm_oc1_composed_pipeline_20260727` (branched from r6cal)
**Evidence axis:** `[macOS-CPU advisory — measured on real 0.mkv pixels; description-byte prices, NOT a scorer/S claim]`
**`score_claim=false · promotion_eligible=false · rank_or_kill_eligible=false`.** Pointer UNMOVED.

The three operator reframes (compose → joint-solve → coherent-order coding-tree) are CONFIRMED and
they NEST: the coherent codec order `PREDICT → TRANSFORM → QUANTIZE → ENTROPY-CODE` applied
recursively over a coding tree (clip→frame→stratum→region→cell→value) IS the recursive structure of
the joint differentiable solve, whose stage-1 is ξ-temporal prediction. This unit built + ran the
**stage-1 (PREDICT) measurement** on the real video, and it relocates the crux precisely.

---

## Verdict first

1. **ξ-temporal PREDICT, in its 2D-homography (MPEG-GMC/sprite) realization, does NOT reduce the
   per-pixel residual.** MEASURED on all 600 pairs of the real `0.mkv` at the 384×512 description
   resolution: the homography-motion-compensated residual is **4.6% LARGER** than the shipped fixed
   1-2-1 blur (`homography/blur = 1.0458`, n600, 0 fit failures, mean warp cover 0.978). All three
   simple predictors sit on a ~**360 KB/pair floor** (blur 362,379 · copy 374,795 · homography
   378,985 B/pair at brotli-q9). The predictor does not matter at the L2-lossless pixel level.

2. **The cross-frame "one static scene × ξ(t)" mosaic-atlas does NOT compress the sequence at the
   pixel level either.** MEASURED: cumulative-homography residual predicting frame(t+d) from a
   keyframe GROWS monotonically with temporal distance — 385 KB @d1 → 416 @d8 → 432 @d32 → ~477 @d85
   (L1 error 3.8→23) — and is already larger than the intra-pair blur at d1. Forward-driving motion is
   **parallax-dominated (3D)**, exactly the case a 2D homography cannot model, so "one keyframe reaches
   far" is FALSE for this scene under homography.

3. **The TASK-domain measurement (rung 1, done) makes it WORSE, not better, and relocates the
   compounding lever off PREDICT entirely.** The coherent order is TASK-LOSSY: stage-3 QUANTIZE zeros
   the residual everywhere the frozen SegNet argmax does NOT flip. So the PREDICT stage's real cost is
   the argmax-flip SUPPORT it leaves. MEASURED through the frozen SegNet on all 600 predicted last
   frames vs the cached GT argmax (`lstars`), d_seg with ZERO residual:
   **copy 0.008642 · blur 0.008648 · homography 0.018672**. The 2D-homography PREDICT is **2.16× WORSE
   than trivial frame-copy** in the task domain too (16.1× the shipped codec's 0.00116 vs copy's 7.45×).
   ξ-temporal-via-homography is measured NEGATIVE on BOTH axes (L2 +4.6%, task +116%). The falsification
   threshold (homography flip-support < 0.5× blur ⇒ task lever) is decisively FAILED (2.16× blur).
   **verdict_scope: FORMULATION** (2D-homography PREDICT); true-3D-depth prediction is the only PREDICT
   reformulation left untested, and the compounding lever is NOT the predictor (below).

3b. **THE compounding lever, located and sized: the SPARSE auth-weighted residual (stage-3), not
   PREDICT.** Trivial copy-PREDICT already leaves only **0.864% of sites flipped** (1,019,467 of
   117,964,800). The shipped codec drives that to 0.116% (the box) but pays a **DENSE 210 MB residual
   (89% nonzero)** to do it. The escape is a residual RESTRICTED to the ~0.86% flip support (+ its
   SegNet receptive-field dilation), which is 60–100× sparser than dense — exactly the operator's
   stage-3 auth-weighted QUANTIZE, and exactly what the shipped codec does NOT do. The synergy chain
   fires from stage-3, not stage-1.

4. **The composed real evaluator row on the coding-tree codec AS IT EXISTS is r6cal's S = 194.42556**
   (inherited, real `upstream/evaluate.py` n600 CPU on the exact 291,205,400 B bytes; 99.731% rate).
   That archive IS a coding tree with a **DEAD PREDICT node** (`descriptor_len = 0` on all 1,200
   records, mode uniformly `SPATIAL_SMOOTH_121`). This unit measured that even a *real* 2D-homography
   PREDICT node would not revive it at L2 — so the dead-predict is not "a bad predictor," it is
   **"per-pixel L2 prediction of these frames is inherently ~360 KB/frame regardless of predictor."**

---

## The measurement is validated against the real archive

My blur baseline at q9 = **362,379 B/pair** ⇒ ≈ **335 KB/pair at the codec's q11** (q9 is ~8% larger;
measured q11/q9 factor on a residual = 0.94). r6cal's exact walk of the SHIPPED archive gives the
residual at **350,049 B/pair** (210,029,373 B / 600). Match within ~4%. The independent measurement
reproduces the real archive's residual cost — the ~360 KB/pair floor is real, not a proxy artifact.

Tool: `experiments/ddm_oc1_xi_temporal_measure.py` (ruff-clean; self-checking: 0 homography failures,
cover recorded). Coder: brotli-q9 for the predictor COMPARISON (ratios faithful; absolute ~8% above
the codec's q11). Frames decoded from `0.mkv` via BT.601 YUV420→RGB then area-downsample to 384×512,
the exact plane the V10 archive encodes.

Artifacts (SSD, small JSON): `/Volumes/VertigoDataTier/pact/ddm_oc1_20260727/xi_n1200_q9.json`
(L2, n600, full atlas curves) + `xi_smoke40.json` (n20, q11, same ratios) +
`flip_support_n600_aggregate.json` (task-domain d_seg, n600, from `fs_chunk0..4.json`). Tools:
`experiments/ddm_oc1_xi_temporal_measure.py` (L2) + `experiments/ddm_oc1_flip_support_measure.py`
(task, self-validated against `lstars`). Both ruff-clean.

---

## The coherent codec order, and where the bytes are (answering "which SCALE, which STAGE")

The operator's synergistic order — each stage makes the next cheap — mapped onto the shipped V10 codec,
with the MEASURED byte carrier at each stage (from r6cal's exact walk + this unit's PREDICT measurement):

| Stage (order) | Shipped V10 realization | State | Byte carrier |
|---|---|---|---|
| 1. PREDICT (intra sprite / inter ξ-advection) | fixed 1-2-1 blur, `descriptor_len=0` | **DEAD** (0 video-derived params) | — (its failure *creates* the residual) |
| 2. TRANSFORM (boundary-aligned sparse basis) | **absent** — residual coded directly | MISSING | — |
| 3. QUANTIZE (auth-weighted, E-cell) | box-tolerance q4/q8 select (seg ERROR box, not S) | present but mis-tuned (208 q4 score-negative by ΔS −30.32, r6cal) | — |
| 4. ENTROPY-CODE | brotli-q11 internal | **saturated** (RAW wins 50/50, r6cal) | — |

**Bytes are carried at the FRAME scale by the RESIDUAL (72.12%, 210 MB) + the frame-0 BOOTSTRAP
(27.85%, 81 MB)** — i.e. by the *absence* of a working PREDICT node and TRANSFORM node. The recursive
tree is real (clip→frame→pair→plane records), but its PREDICT node is empty at every scale. This unit's
new datum: a *real* 2D-homography PREDICT node does not move the residual at L2 — so reviving PREDICT
requires either **true-3D-depth prediction** (parallax-aware) OR operating the whole order in the
**task/argmax domain** (stage-3 first-class), not the L2 domain the shipped residual lives in.

**Break-the-order confirmation (r6cal, exact):** the shipped codec quantized to a seg-ERROR box, not to
the S-objective (the dynamical auth-weighted λ). At this rate operating point that box is 319× tighter
than the score-optimal 1.2731 B/error dual, so 208 q4 selections are score-NEGATIVE by ΔS = −30.32.
This is a mis-tuned stage-3, exactly the synergy-break the operator names — but it is a −30 correction
on a +194 archive; the dominant carrier is still the missing PREDICT/TRANSFORM.

---

## The synergy chain, re-read through the measurement

- Good PREDICT ⇒ small residual ⇒ few TRANSFORM coefficients ⇒ cheap QUANTIZE ⇒ near-free CODE.
- **Measured:** the 2D-homography PREDICT does NOT make the residual small *at L2*. So the L2 chain
  does not fire. The shipped codec proves the same thing from the other side: a task-lossy (box)
  QUANTIZE on a dead-PREDICT residual is still 72% dense.
- **The open, unmeasured link:** whether ξ-PREDICT makes the residual small *in the argmax-flip support*
  (stage-3's actual currency). A homography that registers the static scene need not lower L2 bytes, yet
  could still collapse the argmax-flip support to the codim-1 boundary annulus (which r6cal sizes at the
  0.0035-bits/plane-value box). That is the crux the whole codec now rests on, and it is measurable.

---

## First rungs — each names its next measurement

1. **DONE (this unit): argmax-flip support after ξ-PREDICT.** Result above (§3): 2D-homography is
   task-NEGATIVE (2.16× copy); the falsifier fired; the PREDICT node is not the lever. The measurement
   is self-validated (SegNet reproduces the cached `lstars` to 1 site in 117.9M). Tool:
   `experiments/ddm_oc1_flip_support_measure.py`. Artifact: `flip_support_n600_aggregate.json`.
2. **THE next build (READY, high-EV): the SPARSE auth-weighted residual (stage-3 first-class).** Store
   residual only on the copy-PREDICT flip support (~0.86% of sites) dilated by the SegNet receptive
   field, brotli-code it, byte-close through the r6cal tool → the first real evaluator row that could
   move rate by ~60–100× while holding d_seg. *Falsification:* if the receptive-field dilation to hold
   argmax needs > ~10% of sites, the sparse path does not beat dense and the crux moves to the pose leg /
   scorer-native description. This is the composed-S candidate the coherent order actually points at.
3. **Re-solve stage-3 under S, not the error box** (r6cal rung 2): the exact q4→q8 ΔS = −30.32 is a
   free, in-family correction toward the S-objective; byte-close the all-q8 S-optimum through the r6cal
   tool → a real evaluator row banking −30. Deprioritized per operator ("keep the compress-the-solved-
   object path ONLY as a control"): it lands ~S 164, still rate-dead; validates the machinery, not the goal.
4. **PREDICT reformulation, only if rung 2 walls: true-3D-depth prediction** (MPI / per-pixel depth +
   6-DOF ego), NOT a 2D warp — the parallax the homography cannot model. Gate on rung 2; a 2D warp is
   measured dead on both axes, so do not re-attempt homography/affine sprites.

---

## What this unit did NOT do, and why (honest boundaries)

- **Did not build the full recursive-tree auth-weighted codec, and did not byte-close a new archive.**
  It is the capstone; a wrap-up cannot fake it. This unit built + ran BOTH the L2 (stage-1) and the
  task-domain (argmax-flip support) measurements and relocated the crux to stage-3 (rung 2, sparse
  residual), the composed-S candidate.
- **Did not claim ξ-temporal is a dead FAMILY.** The 2D-homography realization is measured NEGATIVE
  (both axes, n600); true-3D-depth prediction is untested. verdict_scope: FORMULATION. But the crux moved
  OFF PREDICT regardless — the lever is the sparse residual, so depth-PREDICT is now a low-priority rung 4.
- **Did not move the pointer.** No new evaluator row was produced beyond r6cal's inherited 194.42556.
- **The n600 task measurement was run in 5 chunks of 120 pairs (harness kills any single call >3 min);
  self-validation (SegNet reproduces `lstars` to 1 site/117.9M) was run once then skipped in the chunks.
  Aggregated exactly over all 600 pairs (117,964,800 sites) — this is n600 evidence, not a subset.**

## FORK

Composed S is NOT ≤ 0.17 @ ≤ 200 KB (it is r6cal's 194.42556, rate-dead). → **ELSE branch.** The exact
binding stage is **QUANTIZE (stage-3), not PREDICT** — corrected by this unit's rung-1 measurement. The
2D-homography PREDICT (MPEG-GMC/sprite ξ-advection) is measured NEGATIVE on both axes (L2 +4.6%, task
+116%), so PREDICT is not the lever; the compounding lever is the **SPARSE auth-weighted residual**: the
shipped codec stores a DENSE 210 MB residual where only 0.864% of sites flip after trivial copy-PREDICT.
The reformulation that compounds: restrict the residual to the flip support (+ receptive-field dilation)
— rung 2, the composed-S candidate. Not an R6 exact-row candidate this unit (no byte-closed archive was
produced; the composed real row remains r6cal's inherited 194.42556).

## STORES CONSULTED

`CLAUDE.md` (NO-FAKE; measured-scored-quantity axis; THE GOAL bar; §7.1 no-naive-static / auth-weighted;
inflate-is-free / compile-the-generator; SSD-first; pointer-only) · `MEMORY.md` current-state incl.
`ms2r_r3_solved_seg_is_box_solve_not_q1_and_description_compression_is_dead_20260727`,
`dont_compose_on_weak_pricing_base_byteclose_the_solved_objects_20260727`,
`codec_archetype_mpeg4_object_x_netflix_percontent_x_robotics_worldmodel_task_lossy_20260728`,
`pantheon_synergy_is_task_lossy_ego_scene_codec_crux_is_realization_in_image_chart_20260727`,
`objective_is_min_S_over_solution_set_not_box_or_point_20260724`,
`frozen_scorer_exact_factorization_20260715` (resize-first, A_seg≡A_pose),
`opportunity_pools_non_additive_rate_distortion_reachable_20260718` (KKT waterfill) ·
`.omx/research/r6cal_solved_object_byteclose_eval_20260727.md` (the inherited row + byte map + duals) ·
`upstream/{evaluate.py,frame_utils.py,modules.py}` (seq_len=2 non-overlapping; SegNet argmax; 384×512).
