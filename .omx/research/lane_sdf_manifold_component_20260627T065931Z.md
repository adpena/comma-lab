# Manifold-aware lane-SDF level-set component — PRECISE + CONTAINED, supersedes the crude lane-edge loss for the lane SHAPE — FEED-dm

**UTC:** 2026-06-27T06:59:31Z
**Lane:** `lane_sdf_manifold_component_FEED_dm`
**Authority:** `[macOS-CPU advisory]` research-signal — `score_claim=false`, `promotable=false`,
`ready_for_exact_eval_dispatch=false`. $0 CPU-only. NOT a byte-closed row, NOT a trained-witness
output. Frozen CPU-torch SegNet argmax partition (cached `lstars` in `gt_n96.npz`, bit-exact per
FEED-db; NO surrogate, NO new scorer pass, NO GPU/MPS). **GPU UNTOUCHED** (levelset descent pid
72600/72602 + byte-close builder a5021f9c ran concurrently; this task read ONLY `lstars` ~150 MB).

Operator refinement 2026-06-27 verbatim: *"the lane edge stuff can probably be more precisely and
optimally targeted and engineered and contained and manifold across level set dimensions."* This
UPGRADES the crude SEALED lane-edge pixel-margin lever (LEVER-3, FEED-de/df) into a manifold-aware
lane-SDF component for the NEXT iteration (after the first exact row via the current SEALED lever +
builder a5021f9c — does NOT disrupt the critical path; additive, default-off).

<!-- FORMALIZATION_PENDING: the canonical equation "lane class-1 = SDF-to-(ground-frame
centerline_poly + image-halfwidth_poly) level-set field, contained by SDF local support" is queued
for tac.canonical_equations registration once a byte-closed witness row confirms the predicted d_seg
benefit. -->

---

## 0. TL;DR (the decisive answer)

**YES — the manifold-aware lane-SDF gives the lane STRUCTURALLY (precise) WHILE contained, and in the
SDF/level-set framing it does so WITHOUT needing an explicit dash model.** Measured n48, $0, frozen
CPU-torch L*, injecting a structured `phi_1 = SDF-to-lane-polynomial-band` into the K=5 level set with
the OTHER classes' ideal SDFs (the isolation test the operator asked for):

| variant (phi_1 =) | total d_seg | lane_fn (SHAPE) | class0 d_seg (CONTAINMENT) | post-R d_seg |
|---|---|---|---|---|
| **ideal SDF** (baseline; argmax==L*) | 0.000000 | 0.000000 | 0.000000 | 0.000013 |
| **continuous poly band** (no dash) | **0.000415** | **0.000128** | **0.000193** | **0.000797** |
| 2-param ground dash gate | 0.001865 | 0.001762 | 0.000059 | 0.001980 |

- **PRECISE: YES.** Continuous band gives the lane to total **0.000415** (shape FN **0.000128**) —
  **below the witness target 0.00087** and at/under FEED-dj's structural floor 0.00046.
- **CONTAINED: YES.** class-0 leak **0.000193** (below target). This is **~20× SMALLER than FEED-dj's
  hard-mask dash FP 0.00396** — the SDF/argmax competition ITSELF supplies the containment (the deep
  road SDF `phi_0` dominates the locally-supported lane SDF `phi_1` in dash gaps). Containment is
  STRUCTURAL (SDF local support), not a loss-tuning artifact.
- **The dash model is NOT needed here** (refines FEED-dj): the dash-gap FP that dominates the *mask*
  reconstruction is already suppressed to 0.000193 by the SDF argmax framing. A naive 2-param dash gate
  OVER-gates (FN 0.000128 → 0.001762) and HURTS net → **do NOT use as default**.
- **R-SURVIVES:** continuous structured lane SDF post-R **0.000797** (sub-target), vs ideal 0.000013 —
  a mild R-cost from the thin band (the Eikonal/length regularizers already in the witness tighten it
  further when LEARNED rather than EDT-rasterized).
- **SUPERSEDES the crude lane-edge LOSS lever for the lane SHAPE** (verdict §4). DOF = **~30
  floats/frame** (~5 lines × centerline-deg3 + halfwidth-deg1) = the lane manifold coords = the COUNTED
  byte payload (~1–2 KB); the IPM+EDT rasterizer is FREE inflate-time generic algorithm (rule 118).

---

## 1. The design — phi_1 as a signed-distance field on the ~8-dim lane manifold

The level-set witness (`lever_b_levelset_generator.py`, `train_levelset_witness_realized_through_R_mlx.py`)
represents the SegNet argmax partition as `argmax_k phi_k` of K=5 SDF fields; partition = `argmax`,
boundary = the flat zero-crossing `{phi_i = phi_j}`. Class 1 = lane markings (comma10k, CONFIRMED).

The crude lane-edge lever (LEVER-3) is `relu(target − margin)·lane_mask` over ALL GT-lane pixels — it
is (i) **diffuse** (the whole region, not the flip-prone boundary band), (ii) **uncontained** (operates
on the SHARED argmax decision margin `gt_logit − runner_up`, so pushing the lane head up can trade the
50%-majority class-0 = the R4 risk), and (iii) **2D-pixel** (ignores that the lane is a ~8-dim
ground-plane-polynomial manifold, FEED-dj).

**OPTIMAL FORM (this memo):** make the lane a STRUCTURED component of the level set —
`phi_1 = signed-distance-to-(ground-frame lane-polynomial band)`:

- **centerline** `lateral = poly_D(forward)` (deg≤3, 2–4 floats/line) in the openpilot road frame
  (`x→forward, y→left`; CONFIRMED openpilot `common/transformations`),
- **image half-width** `hw = poly_1(v)` (2 floats/line; perspective: wider near, thinner far),
- rasterized via the in-tree small-angle flat-ground IPM (`forward = H·fy/(v−v_h)`,
  `lateral = −(u−cx)·forward/fx`; `H=1.2, fx=400.3, fy=399.5, cx=256, v_h=174`),
- `phi_1 = +EDT inside the band, −EDT outside` (scipy EDT, the SAME construction as
  `signed_distance_fields`; 1-Lipschitz).

Why this is the optimal form (the three properties, now MEASURED, not asserted):

1. **PRECISE** — the SDF is 1-Lipschitz, so the decision margin `m = phi_top1 − phi_top2` is ~linear
   through zero → sub-pixel boundary placement that R-survives the uint8 knife-edge (the SDF R-survival
   cure that motivated the whole level-set vehicle). Measured shape FN **0.000128** < target.
2. **CONTAINED** — an SDF is **locally supported**: `phi_1 > 0` ONLY inside the band, decaying linearly
   negative outside. Far from the lane manifold `phi_1 ≪ 0` → `phi_0` (road) wins → road preserved
   **by construction**. Widening the lane margin geometrically (a local field) cannot globally inflate
   the lane head and trade class-0 (the crude lever's R4 failure mode). Measured class-0 leak
   **0.000193** vs ideal 0 — and the residual is the dash gaps, which the SDF competition already
   suppresses 20× vs the hard mask.
3. **MANIFOLD** — DOF = the ~30 floats/frame polynomial coords, NOT H·W pixel weights. Same object as
   the rate payload (FEED-dj).

This UNIFIES FEED-de's lane-edge LOSS and FEED-dj's lane-geometry PRIOR into ONE manifold-aware
component: the polynomial captures the lane shape to FN < target → the lane is SOLVED STRUCTURALLY,
freeing witness capacity for the all-class edges (81% of flips, FEED-dj).

---

## 2. The $0 mechanism (decisive isolation test) — what was actually measured

Per-frame (n48, frozen CPU-torch L* = `lstars[i]`, 384×512):
1. `phi_ideal = signed_distance_fields(L, 5)` (the K=5 ideal SDF; `argmax == L` EXACTLY → containment
   baseline 0, verified).
2. `phi_1_lane, meta = build_structured_lane_sdf(L)` — cluster class-1 px by ground lateral into ~5
   lines, fit centerline+halfwidth(+optional dash), rasterize band, signed-distance it.
3. `pred = inject_lane_sdf(phi_ideal, phi_1_lane, mode="replace").argmax(-1)` — substitute ONLY phi_1,
   keep the other classes ideal (isolates the lane component).
4. `decompose_argmax_disagreement(pred, L)` → lane FN (shape) / class-0 d_seg (containment) / other.

This is the operator's exact test: *"INJECT it into the level-set partition (as phi_1) and re-measure
the realized argmax d_seg — does the lane component give the lane region to ~0.00046 AND leave
class-0/other-class d_seg UNCHANGED (containment)?"* Answer in §0.

**The key new finding vs FEED-dj** (which measured the *mask* reconstruction): the SDF/argmax framing
changes the containment arithmetic. FEED-dj's continuous-band FP was 0.00396 because EVERY band pixel
in a dash gap counts against the hard mask. In the SDF level set, a dash-gap band pixel is a
false-positive ONLY if `phi_1_lane > phi_0` there — and `phi_0` (the road SDF) is DEEP in the road
interior, so it dominates the shallow (≤ half-width) `phi_1_lane`. Result: dash-gap FP collapses
0.00396 → **0.000193** with NO dash model. The SDF *is* the dash-containment mechanism.

NO-FAKE: real scipy EDT on the real rasterized band of the real polynomial fit to the real class-1
pixels; real argmax recomputed; real disagreement vs the real cached L* (bit-exact). No stub, no
surrogate. Scripts: `experiments/measure_lane_sdf_containment.py` (delegates to
`src/tac/boundary_math/lane_sdf_component.py`); JSON
`experiments/results/lane_sdf_containment_FEED-dm/n48.json`.

---

## 3. Integration spec into the witness + byte-close (for the NEXT iteration)

Additive / default-off. New flags (proposed) on
`experiments/train_levelset_witness_realized_through_R_mlx.py`:
`--lane-sdf-component` (bool, default off), `--lane-sdf-mode {bias,replace}` (default `bias`),
`--lane-sdf-bias-scale` (float, default 1.0).

How `phi_1_lane` plugs in (consume `tac.boundary_math.lane_sdf_component`):

- **Train time:** precompute `phi_1_lane[pi]` once per pair from `gt.lstars[pi]` via
  `build_structured_lane_sdf` (CPU, cached like `lstar_cache`). In `total_loss_fn`, after
  `phi = model.sdf(cf, c0)`, add the deterministic lane field to the class-1 channel:
  - **`mode="bias"` (recommended start, lighter):** `phi[...,1] += scale · phi_1_lane` — keeps the
    LEARNED lane head and PULLS it toward the manifold (a structural prior). The witness still renders
    all classes; the bias shapes the lane head and supplies the deep margin the containment relies on.
  - **`mode="replace"` (stronger reallocation, FEED-dj option c):** render lane class-1 DIRECTLY from
    `phi_1_lane` (FN 0.000128 < target) and let the witness model ONLY non-lane classes + the all-class
    boundary annulus residual. This DIRECTLY reallocates bytes/capacity off the lane shape.
  - In BOTH modes, FiLM-condition the witness on the per-frame coeffs `θ_lane(t)` so the learned
    residual is *given the shape*.
- **Inflate time:** `θ_lane(t)` (the stored coeffs) → ground centerline poly → IPM reproject → band of
  width `hw(v)` → `phi_1_lane` via EDT → injected. The IPM + EDT rasterizer is GENERIC algorithm =
  **FREE** (rule 118 + CLAUDE.md "inflate.py is a FREE interpreter").

**Byte-close composition (NO-FAKE / rule-118 boundary):**
- **COUNTED (archive.zip):** `θ_lane(t)` = ~30 floats/frame (centerline deg3 + halfwidth deg1 × ~5
  lines), the irreducible VIDEO-DERIVED statistic. Temporal-delta (lanes move slowly) + AR/brotli →
  est. **~1–2 KB** → rate-score ≈ `25·1500/37.5M ≈ 0.001`. Consistent with FEED-dd "rate NOT binding."
- **FREE (inflate.py):** the IPM reprojection + band rasterizer + EDT (generic deterministic compute).
- The coeffs are FIT FROM L* (the GT SegNet argmax) = legitimately video-derived/counted; you may NOT
  smuggle a per-frame learned table into inflate.py disguised as code (the hide-data-in-code fake).

**Containment dependency (honest caveat):** the measured class-0 containment (0.000193) assumes `phi_0`
is the DEEP ideal road field. With a LEARNED, shallower `phi_0`, the locally-supported `phi_1_lane`
could leak more. The **Eikonal regularizer already in the witness** (`--eikonal-weight 0.01`, drives
`|grad phi|→1`) is exactly what maintains the deep-field property the containment relies on — so the
integration should keep Eikonal ON (it now has a second, containment, justification). `mode="bias"` is
the safer first integration because it does not depend on a perfect learned `phi_0`.

---

## 4. Supersede verdict — does it replace the crude lane-edge LOSS lever?

**YES, for the LANE SHAPE — the structured lane-SDF SUPERSEDES the crude lane-edge margin-hinge
(LEVER-3) and they should NOT be stacked redundantly.** Rationale:

- The crude lever is diffuse + uncontained + 2D-pixel (§1). The structured SDF gives the lane to
  total 0.000415 (< target) with class-0 contained to 0.000193 (vs the crude lever's shared-argmax
  margin that can trade class-0) and at ~30 floats/frame (the manifold coords = the rate payload).
- The crude lever's only *residual* value is gradient pressure on the **all-class boundary annulus**
  (the 81% NON-lane edges, FEED-dj) — but that is a DIFFERENT lever (all-class, not lane-class). If
  kept, re-scope it to all-class edges (a margin hinge over the union boundary band), NOT the lane
  region. The lane region is now handled structurally.
- **Net plan for the capstone:** REPLACE LEVER-3's lane-specific role with the lane-SDF component;
  optionally retain an all-class boundary-band margin loss as a separate lever for the 81% residual.

This keeps the witness OPTIMAL-FORM (per "implementations must reach OPTIMAL FORM before a verdict"):
the lane is parametrized on its native manifold, contained by SDF support, R-survivable, and rate-cheap.

---

## 5. Honest caveats / NO-FAKE
- **Isolation test** (other classes IDEAL). The real witness LEARNS the other classes; containment then
  depends on `phi_0` staying deep (Eikonal). `mode="bias"` mitigates; a byte-closed trained-witness row
  is required to confirm the predicted plateau drop (0.0032 → ~0.002–0.0025, FEED-dj) — NOT claimed here.
- **n48** (not n600); the comma2k19 RAV4 segment is mostly straight (FEED-dj). On curves the ground-frame
  centerline advantage GROWS (the polynomial stays low-order), so the recommendation is conservative.
- **Dash model:** the naive 2-param matched-filter dash gate OVER-gates on this data (FN explosion) and
  is NOT recommended; the SDF framing makes it largely unnecessary for containment. A better-fit dash
  could shave the last ~0.0001 of class-0 leak but it is NOT the binding term (lane already sub-target).
- **R-cost:** structured-lane post-R 0.000797 vs ideal 0.000013 — the thin EDT band is more R-fragile
  than the smooth ideal field; a LEARNED phi_1 under Eikonal/length reg should close most of this gap.
- This is a PRIOR / structural lever measured at the representation level — the predicted exact-eval
  benefit requires a byte-closed witness row (the next step, not this memo). **Pointer UNMOVED 0.19110.**

## 6. Observability surface
- Per-variant decomposition (`ContainmentDecomp`): total / lane_fn / lane_fp←road / lane_fp←other /
  class0_dseg / other_dseg — diff-able across variants, decomposable per term, queryable from the JSON.
- Per-frame `meta`: n_lines, total_floats, n_dash_modeled, band_px (the manifold coords + cost).
- Reproducible CPU $0: `experiments/measure_lane_sdf_containment.py --n 48 --r-survival`; inputs:
  `gt_n96.npz` `lstars` member only. Module: `src/tac/boundary_math/lane_sdf_component.py` (19 tests).

## 7. Wire-in (6-hook)
1. **sensitivity-map:** lane class-1 = the binding d_seg orbit (FEED-dd/dj); the lane-SDF is its
   structural solver. 2. **Pareto:** lane payload ~1–2 KB (rate non-binding). 3. **bit-allocator:**
   `θ_lane` coeffs = a new COUNTED section (temporal-delta + AR). 4. **cathedral:** the inflate.py
   IPM+EDT lane rasterizer = the FREE generic interpreter path. 5. **continual-learning:** this memo +
   DAG FEED-dm + `lane_sdf_containment_FEED-dm/n48.json`. 6. **probe-disambiguator:** N/A (single
   measured verdict; continuous-vs-dash arbitrated by the measurement: continuous wins in the SDF
   framing).

## 8. Borrowed-substrate accounting
- **BORROWED (cited):** openpilot road-frame IPM (commaai/openpilot `common/transformations`); scipy
  EDT (same primitive as `signed_distance_fields`); numpy polyfit; comma10k class-1 = lane markings.
- **OURS-ORIGINAL:** parametrizing the SegNet argmax LANE class as a signed-distance field to a
  ground-frame polynomial+halfwidth band, injected as phi_1 of the softmax-of-SDF level set so the lane
  is given STRUCTURALLY (contained by SDF local support) and witness capacity reallocates to the
  all-class boundary annulus; the SDF-framing-suppresses-dash-FP finding (20× vs hard mask).
