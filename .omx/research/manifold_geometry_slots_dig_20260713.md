# Manifold/space selection for every geometric witness slot — 2026-07-13

**Outcome:** one geometry changes the next build: S1 should use a **measured flip-risk companding
metric**, not a decorative constant-curvature label. S3 closes a second law — a marked world-sheet
can save rate only by reducing conditional entropy — but the codec A/B is still unmeasured. S2 is
already locally flat on the flip band, S4 has no missed hard constraint outside the sibling-owned
FiLM/Stiefel block, and S5's cached trajectory cannot identify intrinsic curvature.

**Authority:** `$0` local-only research; `[macOS-CPU numpy advisory]`; n600 wherever a measured claim
is load-bearing. `score_claim=false`, `promotion_eligible=false`, `pointer_moved=false`. No SegNet or
PoseNet call, `upstream/evaluate.py`, provider, paid dispatch, training launch, or run-directory
mutation occurred. Branch `main`; deliverables intentionally uncommitted for main review.

**STORES CONSULTED:** `CLAUDE.md`; `AGENTS.md`; `docs/operating_manual_craft_handoff.md`; top-10
operator memory; latest Codex finding/session summary, T3 council, and V9 design; recent operator
directives; `.omx/state/{lane_registry,subagent_progress}`; `sub015_DAG...`; canonical equations
`lane_groundframe_xi_transport_no_collapse`, `deepmath_amortizing_argmax_laws`, `rate_law_ladder`,
`cgauge_parametrization_optima`, and `witness_modular_norm_assignment`; primary n600 artifacts named
below. Recall-before-decide was completed before derivation.

## Verdicts and EV rank

| EV | slot | natural geometry | disposition | direct program effect |
|---:|---|---|---|---|
| **1** | **S1 input chart** | data-adaptive 1-D Riemannian compander over a projective ground chart | **LEVER: PROCEED-with-gates** | reallocates coordinate resolution/steps toward measured d_seg flip rows; counted chart params |
| **2** | **S3 temporal spacetime** | Newton–Cartan/Galilean transport; acoustic Lorentzian envelope only with bounded residual speed | **LEVER: build identical-decode world-sheet rate probe** | removes per-frame curve entropy only when marked conditional entropy is smaller |
| **3** | **S2 head simplex** | Fisher–Rao radius-two sphere; locally Euclidean at the decision wall | **NO-GO new head now** | ETF/additive-margin incumbent should fire first; new geometry cannot change argmax cells |
| **4** | **S4 optimizer blocks** | source-specific Euclidean/Lie/pullback product; Stiefel only for sibling-owned FiLM map | **NO-GO remaining hard constraints** | no new current-module lever; possible future gain-decoupled code chart |
| **5** | **S5 witness manifold** | unknown stratified manifold; cache samples temporal curves, not curvature neighborhoods | **NO-GO curvature actuation** | no mod-dim change; future atlas sizing only after identifiable curvature sampling |

The EV ordering is toward `d_seg/rate/steps`, not aesthetic novelty. A geometry that does not alter a
loss, chart, code, or counted bytes is decoration.

---

## S1 — input chart / perspective ground plane

### Natural geometry derived from the measured structure

Let `w(v)` be measured flip probability per image row and let a monotone chart `s(v)` receive `N`
uniform bins. Its local image-row width is `Delta v ~= 1/(N rho(v))`, where
`rho(v)=ds/dv=sqrt(g_vv)`. Equalizing **first-order flip mass per bin** gives

```text
w(v) Delta v = constant
        => rho*(v) = sqrt(g_vv(v)) = w(v) / integral w(r)dr.          (S1-LAW)
```

This is a Riemannian companding metric, but not generally a constant-curvature hyperbolic metric.
If a different local error model is chosen, e.g. `D_q=integral w rho^{-q}dv`, KKT gives
`rho proportional to w^(1/(q+1))`; therefore **flip density alone does not select an interpolation
order**. S1-LAW is the exact max-risk/equal-mass first-order statement and is the law registered.

Flat-ground perspective gives depth `Z(v)=f_y H/(v-v_h)`. Two familiar parameterizations are:

```text
log depth:       |d log Z/dv| = 1/(v-v_h)       (hyperbolic/log chart)
raw depth:       |d Z/dv|     = f_y H/(v-v_h)^2 (projective inverse-depth chart)
```

They are candidate approximations to `rho*`; neither is optimal merely because perspective exists.

### $0 n600 probe

Source: `.omx/research/dseg_reducibility_gt_margin_n600_20260623.json`, 600/600 pairs,
117,964,800 pixels and 250,519 flips. Probe receipt:
`.omx/research/manifold_geometry_slots_probe_s1_s2_20260713.json`.

- **MEASURED:** 92.9259% of all flip mass is in rows `v>174`; within this support the peak is row
  193 and rows 175–210 carry 65.6581% of the mass.
- **MEASURED:** JS divergence to the normalized flip profile is `0.160440` for unshifted log-depth,
  `0.248188` for uniform, and `0.460261` for unshifted inverse-depth. Thus **log-depth is the best
  parameter-free candidate** and raw projective depth is too singular at the horizon.
- **MEASURED:** a softened inverse-depth family `rho proportional to (v-v_h+delta)^-2` fits
  `delta=32.5258 px`, JS `0.069943`; softened log-depth fits `delta=7.8761 px`, JS `0.101338`.
- **MEASURED:** either monotone perspective family puts its top 10% row allocation on rows carrying
  37.8909% of measured flip mass; uniform captures 5.9344%.
- **VERDICT-SCOPE:** the row ledger is all-class. It is restricted to the ground-support rows but is
  not a ground-class-pair flip ledger. It can rank chart families; it cannot close per-class routing.

### Prior-art delta and lever

Current source calls the built implementation **`GroundFrameChart #194 / §17.1`**. It is a
per-frame xi-homography input precomposition; it canonicalizes temporal ego motion and is held as the
default-off `GroundFrameChart` DSL lever. It is **not a row-density resampler**. The earlier A10/#185
IPM+foveal composite is still described as TIER-3/unbuilt in the June config audit. I therefore do
not launder the full #185 composite into the built #194 surface.

**LEVER — `margin_companded_ground_chart`, PROCEED-with-gates.** Compose a monotone row compander
with #194's temporal chart:

```text
s(v) = integral[v_h,v] w_ground,class-pair(r)dr / integral[v_h,H] w_ground,class-pair(r)dr,
```

with an inverse lookup at receiver render. First build the n600 ground-class-pair row ledger; then
A/B `{identity, log-depth, shifted-projective, measured-CDF}` with the same parameter/step/byte
budget. The fitted `delta` or CDF is video-derived and therefore **COUNTED**; only the generic inverse
compander is free. Admission requires receiver-close and n600 d_seg/per-class/rate, not proxy loss.

**VERDICT:** `PROCEED`, verdict_scope = default-off chart family after class-pair density custody.
The present all-class result does not promote a specific chart into a launch config.

**Triality:** DSL = extend/compose the existing `GroundFrameChart` only after build; equations =
`flip_density_chart_metric_v1`; DAG = FEED-manifold-S1; pool row = `margin_companded_ground_chart`.

---

## S2 — head / five-class simplex

### Natural geometry and exact decision-wall distance

For categorical probabilities `p in Delta^4`, the square-root map `u_i=sqrt(p_i)` places the open
simplex in the positive octant of the unit sphere. With the conventional Fisher metric the sphere
has radius two:

```text
ds_FR^2 = sum_i dp_i^2/p_i,        d_FR(p,q)=2 arccos(sum_i sqrt(p_i q_i)).
```

For top classes `a,b`, the decision wall is the great-sphere section `u_a=u_b`. Its exact distance is

```text
d_wall(p) = 2 asin(|sqrt(p_a)-sqrt(p_b)|/sqrt(2)).                (S2-LAW)
```

A top1–top2 logit gap `delta` fixes `p_a/p_b=e^delta` but not `p_a+p_b`; therefore the full K=5
distance is **not identifiable from a gap alone**. Renormalizing the pair supplies an exact upper
envelope:

```text
d_FR,2(delta)=2 asin(|sqrt(sigmoid(delta))-sqrt(sigmoid(-delta))|/sqrt(2))
             = |delta|/2 + O(|delta|^3).
```

Near a full-simplex wall with `p_a=p_b=q`, the local form is
`d_wall ~= sqrt(q/2)|delta|`; the omitted top-two mass is an adaptive scalar, not a new decision cell.

### $0 n600 probe and ETF delta

- **MEASURED:** at flip median `delta=0.121649`, the pairwise sphere distance differs from
  `delta/2` by `-0.0616%`; at flip p90 `delta=0.439997`, by `-0.7970%`.
- **MEASURED:** all 250,519 flip margins enter the settled histogram; a through-origin fit over its
  nonempty bins has slope `0.486015` and relative RMSE `5.348%`. The larger histogram residual is
  driven by the sparse high-margin tail; the load-bearing p90 band remains below 0.8% deviation.
- **DERIVED:** `d_FR,2(delta)` is strictly monotone, so replacing logit margin by this geodesic cannot
  change argmax cells. It can only rescale training gradients.
- **RECALLED MEASURED:** the in-tree Fisher-curvature versus negative-margin Pearson correlation is
  0.978. This probe does not remeasure it.

The built `HeadGeometry` #218 ETF + additive-margin arm is therefore the correct incumbent: ETF is an
angular, locally-flat shadow of the spherical simplex geometry. It is not the exact Fisher metric,
but the only material missing factor is full-simplex top-two mass `p_a+p_b`, and its value has not
been shown to change d_seg enough to justify a second head implementation.

**VERDICT:** `NO-GO`, verdict_scope = building a new pairwise Fisher-geodesic head before firing the
already-built ETF/additive-margin arm. This does not kill full-simplex natural-gradient weighting.
**Reopen:** cache full K=5 logits, measure conditional variation of `p_top1+p_top2` inside the flip
band, and require a matched n600 gradient/d_seg A/B against `HeadGeometry`.

**Triality:** DSL = existing `HeadGeometry`; equations = `fisher_pairwise_decision_wall_v1`; DAG =
FEED-manifold-S2; no new pool row because no new lever is admitted.

---

## S3 — temporal spacetime / separatrix world-sheets

### Natural causal structure

**DERIVED from the source coordinates:** the observed field lives on `(t,x,y)`, hence **2+1
dimensions**, and xi supplies a preferred flow
`u(t,x)`. The native transport equation is

```text
partial_t phi + u dot grad phi = source/events.
```

Pure transport has absolute time, a Euclidean spatial metric, and a preferred advection connection:
a **Newton–Cartan/Galilean** structure. A deterministic flow line is not a causal cone.

If the residual/interface dynamics additionally impose an isotropic finite speed `c>0` — for
example an explicit CFL-bounded normal velocity — the admissible perturbations
`||dx/dt-u|| <= c` are null/timelike directions of the acoustic Lorentzian metric

```text
ds^2 = ||dx-u dt||^2 - c^2 dt^2,
g = [[|u|^2-c^2, -u_x, -u_y], [-u_x,1,0], [-u_y,0,1]],  det g=-c^2.  (S3-METRIC)
```

Spatial/temporal variation in `u,c` can make this metric curved; constant `u,c` makes it flat after a
comoving coordinate change. Curvature is therefore a measured/model-derived field, not a label.

### World-sheet rate law

Let the separatrix world-sheet be `Sigma={(t,Gamma_t(s))}`. If xi flow deterministically predicts the
non-event continuation, #468's marked chain rule gives

```text
R_sheet = H(Gamma_0|C)
        + sum_t [H(E_t|X_t,C) + H(Phi_t|E_t,X_t,C)
                 + H(Delta_t^E|Phi_t,E_t,X_t,C)] + R_model/receiver,

Delta R = sum_t H(Gamma_t|C) - R_sheet.                              (S3-RATE)
```

This is a **DERIVED** rate advantage criterion, not an automatic saving. It pays precisely when
conditional innovations/events are cheaper than independent curves, with identical decoded
world-sheets. Topology-change marks, receiver phase, and lattice/flicker residuals cannot be dropped.
The L87 covariance statement and #468 are thus causal/conditional factorization laws, not evidence
for a particular constant-curvature spacetime.

### Where de Sitter specifically lands

**NO-GO.** De Sitter means constant positive Lorentzian sectional curvature and maximal symmetry.
For the actual 2+1 witness volume the analog is `dS_3` with isometry group `SO(3,1)`, not `SO(4,1)`;
`SO(4,1)` belongs 3+1-dimensional `dS_4`. More importantly, the road video has preferred absolute
time, a preferred ego flow, boundaries, class strata, and event defects. These break de Sitter
homogeneity/maximal symmetry, and no constant positive curvature has been measured. The useful
structure is the advective cone/conditional chain, not dS curvature.

**LEVER — `marked_worldsheet_separatrix_codec`, needs-build.** Hold the quantized curve family fixed;
encode it both independently and as `{initial, xi, marks, phase, event residual}`; require exact
decoded equality and real archive bytes. This refines the existing v8 edge carriers and #468 rather
than inventing a new field theory.

**VERDICT:** `PROCEED-to-probe`, verdict_scope = an identical-decode codec A/B. No byte or score
advantage is claimed today.

**Triality:** DSL = N/A with rationale (receiver codec, not a trainer flag); equations = existing
`rate_law_ladder_v1` plus `advective_worldsheet_rate_v1`; DAG = FEED-manifold-S3; pool row =
`marked_worldsheet_separatrix_codec`.

---

## S4 — optimizer parameter manifolds, excluding sibling-owned FiLM/Stiefel build

**DERIVED from source inventory:** `MODULE_NORM_ASSIGNMENTS` has 12 complete current-V9 blocks and
reconciles 87,575 trainable parameters. The only current hard manifold lever is `film.weight = QH_0`,
`Q in St(768,19)`, owned by `muon_round2_wire`; this audit did not edit or duplicate it.

| block | candidate geometry | source-derived verdict |
|---|---|---|
| `palette (5x3)` | simplex / sphere? | **NO:** entries are RGB logits, not class probabilities. After sigmoid they lie in a color cube; evaluator pullback is natural. |
| `code (1200x19)` | product sphere? | **NO hard constraint:** radial magnitude changes FiLM amplitude. `R_+ x S^18` is valid only after an explicit gain-direction factorization preserving `Wc`. |
| `out_sdf` rows | product sphere / ETF? | row directions and gains separate, but gains set margins. Sphere-only deletes a load-bearing DOF; #218 owns class angular geometry. |
| `pose_carrier.dxi` | `SE(3)` / Lie group | already represented in `se(3)^600` with task-pullback metric; not missed. |
| trunk matrices | Stiefel/Grassmann/flag? | ordinary unconstrained linear maps; spectral/RMS induced norms are natural. Constraining column spaces lacks a source invariant. |
| FiLM map | Stiefel / Grassmann | **YES, already owned:** polar `QH_0`; the column subspace is Grassmannian only after quotienting the basis gauge. |

There is a joint gauge `c -> A c`, `W -> W A^{-1}`. The sibling polar chart fixes part of it. A future
gain-decoupled code chart may split `c=r u`, but it must store/learn `r`, prove exact reconstruction,
and beat tuned AdamW/Muon. Merely normalizing codes to a sphere is destructive, not geometric rigor.

**VERDICT:** `NO-GO`, verdict_scope = new simplex/sphere/Grassmann constraints on remaining current
module classes. `REOPEN` only for an equivalence-preserving gauge-fixed radial-direction chart with
matched steps-to-d_seg evidence. No pool row is added because no present lever is admitted.

**Triality:** DSL = unchanged; equations = reuse `witness_modular_norm_assignment_v1`; DAG =
FEED-manifold-S4; Stiefel build remains sibling-owned.

---

## S5 — curvature of the approximately eight-dimensional witness manifold

### What the $0 probe actually samples

The probe reran the committed lane fitter on `gt_n600.npz['lstars']` into its own receipts; no live
run directory was touched. It produced 2,948 observations in eight coefficient coordinates across
29 tracked lane curves.

- **MEASURED:** pooled participation rank `2.801`, pooled rank90 `4`.
- **MEASURED:** track-window participation-rank medians are `1.429` (5 observations), `1.741` (11),
  and `1.974` (21); rank90 medians are 2, 2, and 2.
- **MEASURED extrinsic-only:** normalized Menger-curvature median `1.717` and five-observation
  path/chord median `3.824`. These describe bending/noise in the chosen embedding after robust
  coordinate scaling.

### Ruthless identifiability verdict

Each track is a one-parameter temporal curve. Every one-dimensional Riemannian manifold has
identically zero intrinsic Riemann curvature; Menger curvature of its image is **extrinsic**.
Pooling 29 curves from different lane strata does not create local samples spanning independent
tangent 2-planes, so sectional/Ricci/scalar curvature of the proposed eight-dimensional manifold is
**NOT IDENTIFIABLE** from this cache. A graph-Ricci number here would be estimator decoration.

Whitney's generic embedding bound depends on intrinsic/topological dimension, not curvature. Thus
the settled `d=8 -> 2d+1=17`, plus the existing two-coordinate gauge margin `->19`, is unchanged.
Curvature could later inform atlas chart radius, reach, or covering number — not mod-dim — after a
probe samples controlled multi-direction neighborhoods around the same witness state (at least two
independent perturbation directions, exact evaluator pullback distances, and repeated neighborhoods).

**VERDICT:** `NO-GO`, verdict_scope = using the current lane-orbit cache to claim intrinsic curvature
or modify mod-17/19. **Reopen** with multi-direction local interventions and a preregistered
sectional-curvature estimator; until then, curvature is decoration.

**Triality:** DSL = N/A (measurement audit); equations = reuse `cgauge_parametrization_optima` rather
than duplicate Whitney; DAG = FEED-manifold-S5; receipt =
`.omx/research/manifold_geometry_slots_probe_s5_20260713.json`.

---

## Canonical equations, pool routing, and follow-ups

New equation module: `src/tac/canonical_equations/manifold_geometry_slots_20260713.py`:

1. `flip_density_chart_metric_v1` — empirical n600 S1 law.
2. `fisher_pairwise_decision_wall_v1` — exact geometry plus n600 flat-shadow anchor.
3. `advective_worldsheet_rate_v1` — derived S3 metric/rate law, explicitly unmeasured in bytes.

The S5 Whitney conclusion reuses `cgauge_parametrization_optima`; S4 reuses
`witness_modular_norm_assignment`. Refine-don't-duplicate is deliberate.

Pool candidates, written only through `tac.witness_dsl.curriculum_candidate_pool.record_candidate`:

- `margin_companded_ground_chart`: `needs-build`, architecture-growth, d_seg axis, class-pair ledger
  and counted-parameter receiver-close gate.
- `marked_worldsheet_separatrix_codec`: `needs-build`, state-evolution, rate axis, identical-decode
  entropy/byte A/B gate.

Named follow-up `MANIFOLD-S1-PAIR`: generate the n600 row density by ground class-pair and compare the
same four charts. Named follow-up `MANIFOLD-S3-IDDECODE`: build the fixed-representation world-sheet
codec disambiguator. Named follow-up `MANIFOLD-S5-INTERVENTION`: only if chart covering number becomes
binding, sample multi-direction evaluator-pullback neighborhoods; do not rerun curvature on a time
curve.

## External mathematical sources checked

- Zhu & Müller, *Spherical Autoregressive Models* — square-root compositional data on the Fisher–Rao
  sphere: https://arxiv.org/abs/2203.12783
- Visser, *Acoustic propagation in fluids: an unexpected example of Lorentzian geometry* — Newtonian
  flow inducing a Lorentzian acoustic metric for finite-speed perturbations:
  https://arxiv.org/abs/gr-qc/9311028
- Randono, *de Sitter Spaces* — constant-positive-curvature de Sitter geometry and `Spin(4,1)` gauge
  structure: https://arxiv.org/abs/0909.5435
- Whitney, *Differentiable Manifolds* (1936) — the original generic embedding theorem:
  https://www.math.ucdavis.edu/~saito/data/high-dimensions/whitney-diffmanifolds.pdf

## Pointer delta honesty

`pointer_delta = NONE`. No exact archive bytes were produced, no evaluator axis was exercised, and no
score row moved. The durable output is a top-ranked chart lever with a measured n600 allocation
receipt, a conditional world-sheet rate law, three canonical equations, two pool candidates, and
three scoped NO-GOs that prevent geometry decoration.
