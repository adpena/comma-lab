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

3. **BUT this is the wrong distortion metric for the operator's codec, and that is the load-bearing
   re-scope.** The coherent order is **TASK-LOSSY**: stage-3 QUANTIZE is auth-weighted — it zeros the
   residual everywhere the scorer's argmax does NOT flip, and distortion is E-cell violation, NOT L2.
   This measurement priced the **L2-lossless** residual; it does NOT test whether ξ-PREDICT shrinks the
   **argmax-flip SUPPORT** (the only residual the task codec pays for). **verdict_scope: FORMULATION**
   (2D-homography PREDICT, L2-lossless residual) — NOT the ξ-temporal family, NOT the task codec.

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
(n600, full atlas curves) + `xi_smoke40.json` (n20 smoke, q11, confirms the same ratios).

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

## First rungs — each names its next measurement (the crux is now sharp + READY)

1. **THE decisive next measurement (READY, $0, first rung): argmax-flip support after ξ-PREDICT.**
   For each pair, run the frozen SegNet (`upstream/models/segnet.safetensors`) on the homography-,
   blur-, and copy-predicted last frame and compare argmax to the cached GT argmax
   (`experiments/results/mlx_fleet_gt_cache/gt_n600.npz['lstars']`). The flip fraction per predictor =
   the residual support the task codec must pay for after auth-weighted QUANTIZE. If ξ-PREDICT collapses
   the support far below blur/copy, ξ-temporal IS the dominant task lever (operator right) and the whole
   coherent order fires; if it does not, the PREDICT node must be true-3D-depth, not a 2D warp.
   *Falsification threshold:* ξ-PREDICT flip-support < 0.5 × blur flip-support ⇒ ξ-temporal is a task
   lever; ≥ blur ⇒ 2D warp is neutral for the task codec too (escalate PREDICT to depth/MPI).
2. **Re-solve stage-3 under S, not the error box** (r6cal rung 2): the exact q4→q8 ΔS = −30.32 is a
   free, in-family correction; re-run the DP with the 1.2731 B/error dual as the stopping rule and
   byte-close the all-q8 S-optimum through the r6cal tool → a real evaluator row banking −30. Deprioritized
   per operator ("keep the compress-the-solved-object path ONLY as a control") — it lands ~S 164, still
   rate-dead; it validates the machinery, not the goal.
3. **If rung 1 is positive, build the TRANSFORM node** (boundary-aligned/curvelet on the small
   support) then the tree recursion with the shared dynamical λ waterfill (Ortega-Ramchandran) — the
   full recursive-fractal codec. Gate it on rung 1; do not build the tree before PREDICT is shown to
   collapse the task support.

---

## What this unit did NOT do, and why (honest boundaries)

- **Did not build the full recursive-tree auth-weighted codec.** It is the capstone; a wrap-up cannot
  fake it. This unit built + ran its stage-1 (PREDICT) measurement and relocated the crux to a single
  READY $0 measurement (rung 1).
- **Did not run the frozen SegNet argmax-flip measurement here.** The code path is identified and it is
  rung 1; doing it faithfully (camera-res, lattice embedding, exact preprocess) is real scorer plumbing,
  not a wrap-up line — and a subtly-wrong scorer measurement would be worse than none (NO-FAKE).
- **Did not claim ξ-temporal is dead.** It is measured-neutral only for the **L2-lossless** codec; for
  the task codec it is UNMEASURED (rung 1). One failed formulation is not a dead family.
- **Did not move the pointer.** No new evaluator row was produced beyond r6cal's inherited 194.42556.

## FORK

Composed S is NOT ≤ 0.17 @ ≤ 200 KB (it is r6cal's 194.42556, rate-dead). → **ELSE branch.** The exact
binding stage: **PREDICT** — the coding-tree's PREDICT node is dead at every scale, and its natural 2D
realization (homography ξ-advection) is measured-NEUTRAL at the L2 level (rung-1 test pending for the
task level). The reformulation that can compound: a PREDICT that collapses the **argmax-flip support**
(not the L2 residual), realized as either true-3D-depth prediction OR a task-domain PREDICT→auth-quantize
solve — measured first via rung 1. Not an R6 exact-row candidate this unit.

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
