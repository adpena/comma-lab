# ddm_gf2 — static/dynamic-factorized generator form: CEILING-REFUSED at the static ceiling (MAIN adjudication; the arm died to the codex usage limit with its receipts complete)

Owner: MAIN (2026-09-03). Axis: `[macOS-CPU advisory / scorer-free exact count and real-coder rate, n600]`.
Score claim: false. Pointer moved: false. Charter: `.omx/research/charters/ddm_gf2_static_dynamic_generator_form_20260903.md`.
Runner: `experiments/ddm_gf2_static_dynamic_generator_form.py` (6ca377a04 + the working-tree bytes landed with this memo).
Receipts (retained, sha-manifested): `/Volumes/VertigoDataTier/pact/ddm_gf2_static_dynamic_generator_form/{RESULT.json,converged_v2/,converged_v3/}`.

## Result first

| run | alignment | static mismatches (of 117,964,800) | fraction | largest count plausibly repairable at 0.2909 B/site with the ENTIRE 71,404.5 B packet as dynamic | excess | typed |
|---|---|---:|---:|---:|---:|---|
| v1 | 4 coordinate-descent iters, not converged | 3,077,812 | 2.609% | 292,264 (target 46,804) | 2,785,548 | CEILING-INCONCLUSIVE-ALIGNMENT |
| v3 (fixed-point continuation) | **converged, authoritative** (per-pair integer translation dy,dx ∈ [−12,+12]) | **3,072,488** | **2.605%** | 292,264 | **2,780,224** | **CEILING-REFUSED** |

Byte split at the ceiling (v3): static field 258 B + dynamic geometry 193 B = 451 B (the shared field is nearly
free) — and the domain-matched residual for the exact field is **335,096 B** (generic raster residual 462,076 B)
against the **85,020 B** replacement cap. Unaligned mismatches were 3,126,748: the best rigid alignment removes
only 1.7% of them. The exact empirical residual entropy given the aligned static field is 1.57 MB per-site /
2.89 MB class-conditional — the static field is a poor predictor of 600 pairs of driving.

## Why this closes the formulation (and the per-GOP variant without a build)

The mismatch mass is the SCENE'S BOUNDARY MOTION between pairs (road edge, horizon, lanes, cars), not sparse
"events": 2.6% of sites change class per pair relative to one scene field even after the best rigid alignment.
A sparse dynamic term (Lane/Movable events) cannot repair 2.78M sites within 71 KB at any price ≥ 0.026 B/site.
The charter's per-GOP fallback (a few shared fields) reduces the static horizon but not the per-pair boundary
motion: consecutive-pair class IoUs of 0.95–0.99 (Road/Undrivable) still mean ~1–3% of sites move per pair-step,
the same order as measured here — DERIVED, not built; the falsifier would be a GOP field leaving < 292,264
mismatches, which would require per-pair motion < 0.25% of sites, contradicting the measured IoUs.

verdict_scope: **FORMULATION** — one shared static field (or a few GOP fields) + rigid per-pair geometry + sparse
dynamic events, under the generic optimistic repair arithmetic. It does not close a generator that MODELS the
boundary motion parametrically (curve-domain atoms with per-pair deltas) — but that family's Lane member was
priced exactly at 233,262 B by ltg1 (6.47× its bar) and its predictor member at 60,191 B weights (blp1).

## Consequence for the gestalt (fold into `ddm_gs3`)

Both "class-matched form" contenders of the day are closed on measured ceilings: gc1 (capacity on the existing
form: 9.62× at the crossing; residual priced by hard sites) and gf2 (static/dynamic factorization: 10.5× at the
static ceiling). **The only door left to sub-0.12 on the small-body route is OPTIMIZATION of the born field —
the QBR1 six-cell burn now running under the chain driver.** If its adjudication is OPTIMIZATION_CLOSED, the
small-body route is closed on measured evidence and the Pareto-shelf conjecture (0.147–0.155 ≈ task R(D) under
decodable conditioning, gp2/gp3) is strengthened to a measured family verdict.

## NEXT_IF_RESUMED
- FOLDED — owner: MAIN; consumer store: `ddm_gs3_gestalt_after_submission_20260903.md` addendum 2; fire trigger: none (this memo).
- QUEUED-CONDITIONAL — owner: a future form builder; consumer store: `/Volumes/VertigoDataTier/pact/ddm_gf2_static_dynamic_generator_form/`;
  fire trigger: a parametric boundary-motion generator whose closed-form arithmetic (not a fit) predicts ≤ 292,264
  mismatches at ≤ 71,404.5 B — cite ltg1/blp1 before building.

## DEAD-ENDS
- One shared static field + rigid alignment: 3,072,488 mismatches, CLOSED. Per-GOP static fields: closed by the
  measured per-pair motion (DERIVED). Sparse dynamic events on top of a static scene: cannot reach the cap.

Own-vehicle frontier: **afr1 S 0.14797617125559104 @ 180,002 B [contest-CUDA T4 n600]** — unmoved.

---

## ADDENDUM (ddm_eq1, 2026-09-04) — the equations leg

**Law:** `decoder_derivable_ideal_savings_ceiling_v1` — `tac.canonical_equations.ddm_lv3_current_arc_laws_20260901` (`tac.canonical_equations`). **Relation:** IN-DOMAIN ANCHOR (ceiling-first refusal).

GF2 is that law's method executed on a new object: an optimistic repair ceiling (292,264 repairable sites at 0.2909 B/site with the ENTIRE 71,404.5 B packet as dynamic) used to REFUSE, never as a build target. Measured excess 2,780,224 sites. Sister law for the inherited-form leg: `generator_form_fit_error_entanglement_v1` (the 2.178x ratio does not transfer without its ~1.12% fit error).
