# P1 SEAT S1 — DEEP-MATH / TROPICAL-ARGMAX ENERGY (independent position, crucible-3 v8)

**Author:** S1 (deep-math / tropical-argmax / Laguerre power-diagram energy — Daubechies/Mallat/Shannon
lens). **Date:** 2026-07-09. **Axis:** every number `[macOS advisory · research-signal · NON-PROMOTABLE]`.
**Pointer 0.19110 UNMOVED — MEANS.** Independent: NO cross-read of sibling seats S2–S6 (this crucible).
Cites `docs/operating_manual_craft_handoff.md` (§4 re-derive-from-primary, §5 label-by-provenance, §6
attack-own, §8 mistakes). Works from `DELTA_GROUNDING_20260709.md` + the primary deep-math artifacts.

STORES CONSULTED: crucible-3 {CONVENING, DELTA_GROUNDING, ORCHESTRATION_LEDGER}_20260709 · SPEC_v8
§1–§7 (I am S1, not the structure-blind seat — SPEC_v8 architecture IS in my scope) · registered eqs
`v8_geometric_rate_decomposition_v1` (read the module source, re-derived the ledger from it) +
`maslov_dequantization_bound_v1` + `shearlet_nterm_upper_bounds_task_rate_v1` +
`multiphase_modica_mortola_perimeter_gamma_limit_v1` + `mcf_minority_erasure_inevitability_v1` +
`laguerre_ot_head_offset` (N-1) + `lane_groundframe_xi_transport_no_collapse_v1` (N-2) · DAG
FEED-03y/03z/04a/04b (#284 EUREKA chapters 1/2/3/5) + FEED-v8-{voronoi, roadlane, rollup,
bytecost-sharpened} · `v8_movable_residual_rollup_20260709.md` (§A/B/C, re-read the coder tables) ·
crucible-2 `position_S1_deepmath` (my own pattern — the area-Lagrange missing-term method). **NOT
consulted (independence):** sibling S2–S6 positions this crucible.

**Method (the area-Lagrange method, §4):** write the tropical-argmax composite energy the v8 decode
DEMANDS, then check each carrier / coder / bias against the piece of that energy it is supposed to
realize. A term the energy demands but the design lacks (or realizes WRONG) is a MISSING/MIS-SPEC'd
force — this is how N-1's mass-matching b_c and the residual-coder frame are caught from FIRST
PRINCIPLES, not from the memo. Relative-significance throughout: **remaining gap to sub-0.15 = 0.0411 S**
(0.19110 − 0.15); every ΔS is quoted as a fraction of it.

---

## 0. THE ENERGY (the one object every claim is derived from)

The frozen SegNet argmax partition is (PROVEN, `maslov_dequantization_bound_v1` + Ch.1 Laguerre law):

```
P(x) = argmax_c ( φ_c(x) + b_c )                       # the HARD partition
      = additively-weighted Voronoi = LAGUERRE / POWER DIAGRAM       (sites = φ_c, weights = b_c)
      = complement of the TROPICAL hypersurface  Σ = { top1 = top2 }  (the separatrix, DERIVED never stored)
softmax_τ → argmax as τ→0 is the (max,+) semiring limit; error ∈ [0, τ·ln5]     (Maslov, PROVEN)
```

The v8 rate law (REGISTERED `v8_geometric_rate_decomposition_v1`, re-derived from the module source):

```
S_edge  =  S_gen( dominant, ~k coeffs/frame )  +  S_resid( uncovered px )        # per adjacency EDGE
S_rate  =  Σ_{edges e}  S_edge                                                    # whole-scene, de-shared
```

Two PROVEN facts that do all the work below:
- **d_seg = HAMMING (symmetric-difference) mismatch of two Laguerre labelings — NOT a Wasserstein/mass
  cost** (Ch.1 Laguerre law). ⇒ the frozen scorer contributes only the DUAL (margin-saliency #141 + a
  class-mass budget); we cannot run OT to *design* it, only to place ties.
- **the SegNet argmax partition is a CARTOON** (piecewise-C² regions with C² boundary) ⇒ the optimal
  sparse chart of the separatrix is a curvelet/shearlet, N-term boundary error **O(N⁻²(logN)³)**, and the
  Fisher/margin-weighted N-term count is an **UPPER BOUND on the task-space rate R_X(D_Y)**
  (`shearlet_nterm_upper_bounds_task_rate_v1`, PROVEN upper bound; tightness conjectured).

Everything below is read off these. The v8 carrier stack is a RATE problem on the separatrix Σ of a
Laguerre diagram; d_seg is the HAMMING placement error of Σ.

---

## 1. THE PARSIMONY-NOT-DUAL LAW (DERIVED — the carrier-representation law, §C formalized)

**Claim (DERIVED from the shearlet N-term bound):** for each separatrix curve Σ_e, the rate is set by
the **N-term decay of the sparsest chart of Σ_e**, and the *choice of dual*
(curvelet-boundary ≡ medial-axis-centerline ≡ power-diagram-sites) changes only the CONSTANT (basis +
bits/coeff), **never the N(D) decay exponent** — because all three are sparse charts of the SAME C²
boundary and all achieve the same O(N⁻²) cartoon rate. What breaks the rate is picking a
**non-sparse** representation (dense medial axis, boundary bitmap): those have N ~ boundary-LENGTH (no
decay), so they cost ~the bitmap.

**Re-derived receipts (MEASURED, §C):**
- dense medial axis (444 ridge-pts/frame) = 0.352 S ≈ boundary bitmap 0.384 S — **1.09×, NO WIN.**
  verdict_scope: **FORMULATION** (the *dense* generator; the few-coefficient fit is the lever). This is
  the parsimony law's negative control: same object, N ~ length ⇒ no decay ⇒ ≈ bitmap.
- horizon deg-3 poly (4 coeffs) = 14.6× (0.0032 S). This is the parsimony law's POSITIVE control: the
  horizon is a single smooth low-order arc ⇒ N₀ ≈ 4 curvelet atoms capture it ⇒ O(N⁻²) collapse. In
  relative terms 0.047 → 0.0032 is a 0.044-S rate saving on that one edge = **1.07× the whole remaining
  gap** from a 4-coefficient fit.

**Design consequence (the law, binding on every carrier):** *build each carrier as the PARAMETRIC
GENERATOR whose coefficient count = the number of C²-features (inflections/corners) of its separatrix
— never the boundary bitmap, never the dense medial axis. The rate lives in the coefficient count, not
the dual choice.* This is why the carrier table is heterogeneous by GEOMETRY not by taste: horizon =
1 low-order poly (1 smooth arc); Road/Lane = few centerline polys (few near-parallel smooth arcs);
Movable = sparse per-object SITES (the boundary is many disconnected short closed curves — the sparsest
chart of a disconnected boundary is per-component sites, NOT a global centerline; this is the SAME law
choosing the natural dual per topology, not an ad-hoc switch). MyCar/hood = one static curve (rigid).

---

## 2. WHICH CARRIERS THE GEOMETRY DEMANDS — the edge-centric decomposition, and the 41-vs-5 resolution

**The structural derivation (DERIVED, the headline of this seat).** Because d_seg = Hamming and a pixel
flips ONLY on a separatrix between two ADJACENT classes, d_seg factorizes over the region-adjacency
graph:

```
d_seg  =  Σ_{edges e ∈ ADJACENCY GRAPH}  (Hamming flip mass on Σ_e)                (Laguerre factorization)
```

The tie locus Σ_{cc'} = {φ_c+b_c = φ_{c'}+b_{c'}} is **SHARED** by regions c and c'. Store a field per
REGION and you pay for each curve TWICE (once from each side) — this is risk-1 (edge-duplication), and
it is not a heuristic worry, it is the double-cover of a shared codim-1 set. **The energy DEMANDS one
generator per EDGE of the adjacency graph, not per region.**

**Now the 41-vs-5 tension (the CONVENING's prime cargo-cult target), resolved from the MEASURED graph.**
SPEC_v8 §1 says "one field per adjacency EDGE" and cites up to 41 spatial-adjacency edges; the carrier
table has 5 fields. These are NOT in conflict — they are reconciled by the MEASURED adjacency being a
**STAR, not a complete graph**:

- P-A (MEASURED, n600, probe bf1ee1fa8): **Road = hub; every class flips ONLY at its Road separatrix;
  ZERO interior flips at the oracle floor** (§SETTLED). The active adjacency is therefore
  `{Road–Lane, Road–Undriv, Road–Movable, Road–MyCar}` + the one non-Road active edge
  `Undriv–Movable` (cars break the horizon, MEASURED 1.6–2.0 crossings/row, §E).
- A star with L leaves has exactly L edges. Here each NON-Road class is a LEAF with exactly ONE active
  edge (to Road) ⇒ **"one field per active edge" COINCIDES with "one carrier per non-Road class"** for
  the four leaves. The 41 (or C(5,2)=10) potential edges are the geometric-MAXIMAL adjacency; the
  ACTIVE adjacency (where Hamming flip mass lives) is a **5-edge Road star**.
- The ONLY place edge-centric ≠ class-naive is **Road↔Undriv**: both endpoints are BACKGROUND-like
  (Road = the hub, Undriv = sky/background sharing the horizon curve), so neither is a leaf and the
  shared horizon curve would be double-paid by a per-region split. **This is exactly the ONE new build
  increment-1 specifies (the shared Road+Undriv bulk-boundary field).**

**Verdict (DERIVED, vindication-not-divergence per the blind-derivation contract):** the "5 fields" and
the "edge-centric" framing AGREE, and the deep reason is sharper than "5 because 5 classes" — it is **"5
because the measured adjacency is a Road-hub star with 5 active edges, and each leaf-class owns exactly
one edge."** The "41 edges / edge-centric" label is aspirational ONLY in that it names the
geometric-maximal graph; on the MEASURED graph the label is real, and increment-1's single new field IS
the correct edge-centric build (it de-shares the unique edge whose two endpoints are both non-leaf). The
aspirational-label residue is therefore SEMANTIC (the "41" over-counts inactive edges), not STRUCTURAL
(the 5-field build is right). **The carrier COUNT the geometry demands for increment-1 = 5 (four leaf
carriers, reused; one shared-edge field, new).**

**One caveat the energy forces (multi-component Road):** Road is multi-blob in 37.2% of frames (§D
lift-verify) ⇒ the shared Road+Undriv field's zero-set is MULTIPLE closed curves, and the signed-SDF
lift must be multi-component-Road-aware. This is not a decomposition change — a single signed field
represents multi-blob Road correctly (zero-set = several curves) — but it pushes that edge's coefficient
count toward the HIGH end (each Road blob is its own arc-set). DERIVED consequence for §5's floor.

---

## 3. THE b_c FLIP-WEIGHTED TIE CALIBRATION (N-1 reformulation, DERIVED closed-form)

**b_c is the Laguerre ADDITIVE WEIGHT** — raising b_c enlarges cell c, pushing all its separatrices
outward. It is a ~0-byte lever (5 scalars). N-1 measured that OT/area-mass-matching offsets **HURT both
arms** (no_offset 0.00272 < menon 0.00293 < ot_newton 0.00487; the exact Newton solve hit KKT to
1e-11 — the SOLVER is right, the OBJECTIVE is wrong). verdict_scope: **FORMULATION** (mass-matching as a
d_seg surrogate).

**Why it HURTS, from the energy (DERIVED):** OT matches cell MASS (area) to GT class frequency. But
**d_seg is HAMMING, not Wasserstein** (Ch.1). Matching the mass of the rare Lane cell (0.59% area)
drives b_Lane ≈ +28.7 to inflate its area → the tie locus over-shoots → Lane over-predicted → SegNet
Hamming-penalizes. Mass-matching optimizes the WRONG functional.

**The DERIVED correct form (the OPEN reformulation, formalized here).** For each edge, project onto the
margin coordinate `m = φ_c − φ_{c'}`; the tie locus sits at `m = b_{c'} − b_c`. The Hamming flip mass
mis-placed by an offset t is `∫ flip-density(m) · 1[wrong side] dm`. Minimizing an L1/Hamming loss over
a threshold placement is solved by the **MEDIAN of the flip distribution**, not the mean/mass:

```
b_c − b_{c'}  =  argmin_t  ∫ flipdensity_e(m) · |sign(m−t)−sign(m)|/2 dm
              =  MEDIAN_{flip-weighted}( m along Σ_e )                    (Hamming-optimal threshold = median)
```

**⇒ the Laguerre additive-weight difference should place each tie locus at the FLIP-DENSITY MEDIAN along
its margin coordinate, per edge — NOT at the mass-matching point (the N-1 arm that hurt).** This is:
(a) closed-form, per-edge, 1-D — a weighted-quantile placement; (b) ~0-byte (5 scalars, or the 5-edge
star's 5 offsets); (c) calibrated OUTSIDE the scorer-gradient loop (risk-2 guard — a data-side quantile,
never joint-vs-d_seg); (d) reuses the damped-Newton machinery (`laguerre_logit_offset` /
`damped_newton_ot_offsets`) with the flip-weighted target substituted for the mass target — a ~1-line
objective swap in a BUILT solver.

**Relative-significance:** the N-1 span no_offset(0.00272)→ot_newton(0.00487) is 0.00215 d_seg = **0.215 S
= 5.2× the remaining gap 0.0411**. Getting b_c RIGHT (or merely not-wrong) is one of the highest-leverage
~0-byte moves in the whole stack — but the MEASURED story is that *area-matching* offsets are net-harm, so
the flip-weighted median is the ONLY b_c form with a derived path to a POSITIVE contribution. LABEL:
DERIVED-form, ΔS ASSUMED (owed a flip-weighted-target A/B — the N-1 probe re-run with the median target,
$0 on the label cache).

---

## 4. THE RESIDUAL CODER AS AN R(D) PROBLEM — the named enemy (0.079 S = 1.92× the remaining gap)

The residual = the uncovered separatrix px the parsimonious generator does not capture = the boundary
DISPLACEMENT between the true Σ_e and the generator's rasterized band. Re-derived from the coder tables
(§B): per-edge residual S = Movable 0.01741 · horizon 0.01892 · Road/Lane 0.04203; **complete − dominant
= 0.079 S** and it is the ENTIRE gap between the thesis-confirmed 0.061 and the frontier-tying 0.140.
**Relative-significance: 0.079 S = 1.92× the whole remaining gap to sub-0.15.** This is the correct
target of the seat's depth (§3 of the manual — blast radius, not line count).

**The R(D) framing (DERIVED — the deep-math of why chain-code failed and offset-coding is the headroom).**
d_seg is a **boundary-displacement functional** (PROVEN, Ch.5). The generic residual coder (§B) pays for
the residual as a **2-D scattered point set**: rate ≈ 0.4–0.6 B/px ≈ its 2-D coordinate entropy
(chain-code buys only **−13%** — MEASURED, Movable 22.3 KB vs 25.5 KB; verdict_scope: FORMULATION, the
*absolute-coordinate 2-D walk* — chain-code still codes a 2-D walk, capturing connectivity but NOT
low-order smoothness). **But the residual is NOT intrinsically 2-D.** It is a signed NORMAL OFFSET
`δ(s)` from the generator curve, as a function of arc-length `s` — a **1-D signal**:

```
residual entropy (absolute)  ≈  H( 2-D point set )        ≈  area · log(area)     ← what §B pays
residual entropy (curve-rel) ≈  H( 1-D offset δ(s) )      ≈  arclength · (bits/offset-sample)
                                 |δ| ≤ band-halfwidth (a few px ≈ 2–3 bits) AND δ(s) is C²-smooth in s
                                 ⇒ by the SAME cartoon N-term bound, N-term(δ) ≪ arclength
```

**This is a DIMENSIONAL REDUCTION (2-D → 1-D), and it is exactly what chain-code does NOT do.** The
generator already removed the low-order trend of Σ_e; the residual offset δ(s) is what's LEFT — the
high-curvature wiggle — which, coded as a bandlimited 1-D signal along the generator's arc-length, has
the offset alphabet (±few px) AND the C² smoothness that make it N-term-sparse. **Prediction (DERIVED,
CONJECTURED value): a curve-relative offset coder converts the residual from "near its 2-D coordinate
entropy" (where chain-code stalls at −13%) to "the N-term count of a bounded 1-D C² signal."** This is
the theory-optimal residual coder the shearlet framework NAMES; the generic 0.4–0.6 B/px is an UPPER
BOUND, never the floor (§E, DELTA agrees — I derive WHY).

**Order-of-magnitude ESTIMATE (labeled CONJECTURED — owed a P4 $0 measurement on the cache).** Road/Lane
residual = 228 px/frame at ~0.4 B/px = 0.04203 S. As a 1-D offset δ(s): ~228 arc-samples × ~2–3
bits/sample, and if δ(s) is ~4× wavelet-compressible (C² ⇒ energy in low bands), ≈ 228·2.5/4 ≈ 143
bits/frame ≈ 18 B/frame vs the current ~0.42·228 ≈ 96 B/frame → **≈ 5×** → 0.04203 → **≈ 0.008–0.011 S**.
Applied across the three residual rows (0.079 → ≈ 0.016–0.022) this would move COMPLETE from 0.140 to
**≈ 0.077–0.083** — still ABOVE 0.118? No: 0.061 + 0.016..0.022 ≈ **0.077–0.083, i.e. 1.4–1.5× BELOW the
0.118 frontier.** LABEL: **CONJECTURED / DOUBLY-CONDITIONAL** (presupposes δ(s)'s 4× compressibility,
which is UNMEASURED — the offset-signal spectrum has never been measured; and presupposes the generator's
normal-offset map is well-defined where Σ_e is multi-valued, which fails at horizon secondary-arc
crossings — see the de-share lever). This is the seat's single most consequential CONJECTURE and it is
flagged as owed, not asserted.

**The de-share lever (headroom lever 1) is EXACT, buildable in increment-1, and I recommend it FIRST.**
The horizon residual's "secondary arcs" ARE objects breaking the horizon = Movable/Undriv px the Movable
sites ALREADY carry (MEASURED 1.6–2.0 crossings/row, §E). Attributing them to the Movable carrier removes
them from the horizon residual for ZERO new bytes — a set-subtraction, exact, no coder change. This is the
Laguerre factorization enforced correctly (each px on exactly ONE edge's residual). LABEL: DERIVED-exact;
the MAGNITUDE (how much of the 0.01892 horizon residual is secondary-arc double-count) is UNMEASURED —
owed a $0 attribution on the cache (P4). It is strictly a DEFLATION (the double-count currently INFLATES
the complete number), so it is a free monotone win regardless of magnitude.

**THE RESIDUAL-CODER DECISION (my position on Q2, from the energy):** build **BOTH** headroom levers,
in this order: (1) DE-SHARE (exact, ~0-cost, monotone, buildable in increment-1) → then (2) CURVE-RELATIVE
OFFSET δ(s) coder (the derived theory-optimal, DOUBLY-CONDITIONAL, its own build). Do NOT ship a lossy
operating point on the lane edge (**N-4 MEASURED: the RD knee sits at the lossless point; every
quantization step RAISES S_total — the distortion is a fixed ~0.52-S band-geometry floor, not real
quantization headroom; verdict_scope: FORMULATION, filled-band-vs-thin-lines artifact**). Lossy is
dominated on the binding edge; the two lossless headroom levers are the only principled closers. If
neither is built for increment-1, ship **dominant-only 0.061 with the residual as a measured-owed
sidecar** — the thesis (0.061 < 0.118, 1.9×) is CONFIRMED and load-bearing on its own; the complete
number is the stretch.

---

## 5. THE EDGE-CENTRIC FLOOR + THE PER-PAIR σ IS SUPPLIED FOR FREE (v8's d_seg mechanism, principled)

**The floor (DERIVED lower bound).** From the shearlet N-term bound, the edge-centric lossless rate is
bounded BELOW by the number of C²-INFLECTIONS (not the length) of each separatrix:

```
S_rate  ≥  Σ_{edges e}  [ #inflections(Σ_e) · (bits/atom) ] / N · 25          (cartoon N-term floor)
```

The horizon (1 smooth arc, ~1 inflection) → ~4 coeffs (MEASURED 14.6×, at the floor). Road/Lane (dashed,
many features) → higher. **The residual's irreducibility (chain-code −13%) is the MEASURED evidence that
the uncovered px are near their inflection-count entropy in ABSOLUTE coordinates — but §4's curve-relative
map is precisely the change of chart that lowers the inflection count** (the generator removes the
low-order trend, so δ(s) has fewer inflections than Σ_e). So the floor is chart-dependent: **0.061 IS
reachable as the complete number IFF the offset chart reaches the inflection floor; it stays 0.140 if we
code Σ_e in absolute coordinates.** This is the crisp deep-math statement of the named enemy — the 0.079
gap is a CHANGE-OF-CHART problem, not a fundamental-entropy problem.

**The σ_{cc'} that v7.5 was missing is STRUCTURALLY SUPPLIED by v8's edge-centric decomposition (DERIVED
— a v8 d_seg advantage, my crucible-2 headline catch resolved).** The multiphase Modica-Mortola Γ-limit
(`multiphase_modica_mortola_perimeter_gamma_limit_v1`, Baldo/Sternberg, K=5 wells) demands a PER-PAIR
anisotropic surface tension σ_{cc'} (interface stiffness). v7.5's single-trunk length term is a SCALAR
isotropic perimeter — it under-stiffens the thin-lane and Road↔Lane interfaces and lets MCF erase them
(`mcf_minority_erasure_inevitability_v1`; the un-nucleated-lane creep 0.00475→0.00657 is the receipt). In
crucible-2 I flagged σ_{cc'} as v7.5's #1 unconsumed missing force. **In v8 it is not a bolt-on: each
edge-field carries its OWN eikonal / length / τ_c / β_c anneal, so the per-pair anisotropic stiffness is
the decomposition itself.** The Road↔Lane field can be stiff (anti-erosion) while the Road↔Undriv field
is soft — WITHOUT a shared scalar length term forcing one σ on all pairs. **⇒ v8's edge-centric
decomposition IS the anisotropic-surface-tension cure made structural — this is a PRINCIPLED reason v8's
d_seg mechanism should hold, beyond the ∂φ_c/∂θ_{c'}=0 theft-decoupling bet.** LABEL: DERIVED; the
through-R d_seg gain is UNPROVEN until increment-1 (the bet is still a bet — §7 of the manual: a number
from another vehicle is a hypothesis here), but the MECHANISM is now derived, not asserted.

---

## 6. MISSING-TERM ANALYSIS (tropical composite energy demanded vs v8 design present)

Walk the v8 decode energy `S_rate = Σ_e S_gen + S_resid`, `d_seg = Σ_e Hamming(Σ_e)`, piece by piece.

| # | energy piece the tropical/Laguerre object demands | in the v8 design? | verdict |
|---|---|---|---|
| 1 | one parsimonious generator per ACTIVE edge (parsimony law §1) | ✓ (5-edge star, §2) | present, DERIVED-correct |
| 2 | 1-Lipschitz eikonal SDF gauge (all φ_c in px units ⇒ b_c comparable) | ✓ (SDF gauge, §D) | present |
| 3 | **b_c placed at the flip-density MEDIAN per edge (Hamming-optimal)** | **✗ (design uses OT/Menon mass-match — N-1 HURT)** | **MIS-SPEC'd — §3 the derived fix** |
| 4 | **residual coded in CURVE-RELATIVE offset δ(s) (boundary-displacement chart)** | **✗ (design uses generic 2-D sparse-coord)** | **MIS-CHARTED — §4 the derived fix** |
| 5 | de-share: each px on EXACTLY ONE edge's residual (Laguerre factorization) | ✗ (horizon residual ⊃ Movable px, double-count) | MISSING — §4 lever-1 (exact, monotone) |
| 6 | per-pair σ_{cc'} anisotropic stiffness (multiphase Modica-Mortola) | ✓ STRUCTURAL (per-edge field anneal, §5) | **present-by-construction — the v8 advantage** |
| 7 | MERGE = tropical argmax (idempotent in (max,+) ⇒ seam-free at the LABEL level) | ✓ (argmax composite) | present, DERIVED-free (see note) |
| 8 | ξ-transport per carrier (rate/pose) — image-frame ONLY | ✓ horizon/pose; ✗ ground-lane (N-2 NO-GO) | correct — N-2 forbids ξ on ground-canonicalized carriers |

**Note on #7 (a deep-math gift for the reconciliation — MERGE is FREE at the label level):** the composite
partition IS `argmax_c(φ_c+b_c)` — a single tropical (max,+) sum of the edge-fields. argmax is idempotent
and associative in the (max,+) semiring ⇒ **there is NO blending seam at the LABEL level** (the composite
labeling is exact, not an interpolation of independent fields). Seams appear ONLY at the PAINT (RGB)
level, which is precisely why the reconciliation DIFFs `R(paint)` against the frozen scorer, NOT the
labels. This bounds the reconciliation's job: the label composite is exact-by-construction; the only
thing that can go wrong is the paint→R→argmax readout — a MUCH smaller surface than "compose 5 fields."
(Handing this to S3/S4 as the reconciliation's deep-math floor.)

**The two catches (#3, #4) are the seat's load-bearing findings** and they are INDEPENDENT of the memo's
framing — they fall out of "d_seg is Hamming (⇒ median not mass)" and "d_seg is a displacement functional
(⇒ offset not coordinate)". Both are BUILDABLE (a solver objective swap; a coder chart change), ~0-to-low
byte, and each has a $0 owed measurement.

---

## 7. CONFIG-SHAPED RECOMMENDATION (every knob a VALUE with provenance)

Provenance ladder: **DERIVED** (from the energy) · **MEASURED** (n600/probe anchor) · **DERIVED-LIVE**
(computed at config from loaded GT) · **CONJECTURED** (owed a measurement) · **ASSUMED** (owed an A/B).
This is a CONTROL-LAW shape for the v8 increment-1 carrier stack, NOT a launchable argv (P2 owns the DSL
`Lever` factory + `lever_registry.completeness()==0`); the values below are the ones the tropical energy
demands.

### 7.1 CARRIER STACK — 5-edge Road star (§2), each a parsimonious generator (§1)
```
Road↔Undriv (horizon)  = deg-3 boundary poly + ξ-intercept   # DERIVED (§1); MEASURED 14.6× / 0.0032 S
    THE ONE NEW BUILD (road_undriv_bulk_field): multi-component-Road-aware (37.2% multi-blob, MEASURED §2)
    KNOWN BUG to fix: bulk_boundary_byte_cost must measure the Road↔Undriv shared edge (426 px), NOT the
      full Road perimeter (2228 px) — mode="horizon_poly_xi"  # MEASURED scope bug, §D "707B is naive"
Road↔Lane              = centerline deg-3 poly + deg-1 halfwidth, ~5 lines×11 coeffs  # MEASURED 7.4× / 0.0275 S
    NO ξ transport (N-2 MEASURED NO-GO: ground-canonicalized chart, ‖innov_ξ‖₁ ≥ ‖ΔQ‖₁ ∀ξ) — use 0.0275
Movable (Road/Mov ∪ Undriv/Mov) = sparse bbox SITES, Hungarian ξ-slot-track  # MEASURED 70% cover / 0.00344 S
    ONE carrier de-shares BOTH edges (47.3%/51.9% split, MEASURED); the "sites" dual is DERIVED (§1, disconnected boundary)
MyCar/hood             = static curve frame0 + rigid ξ-shift  # MEASURED IoU 0.994 (#139) / 0.0202 complete — REUSE
Lane/* (3 rows)        = already tiny 0.007 S — untouched      # MEASURED
```

### 7.2 THE b_c TIE CALIBRATION — flip-weighted median, closed-form, OUT of loop (§3, the derived N-1 fix)
```
b_c − b_{c'} = flip-density MEDIAN of m=φ_c−φ_{c'} per edge   # DERIVED (Hamming-optimal threshold = median)
    solver = laguerre_logit_offset / damped_newton_ot_offsets with FLIP-WEIGHTED target (not mass)  # BUILT solver, objective swap
    calibrated OUTSIDE the scorer-gradient loop (risk-2 guard) ; ~0 byte
    # N-1 MEASURED: mass-match HURT (no_offset 0.00272 < ot_newton 0.00487); flip-weighted ΔS ASSUMED (owed $0 A/B)
```

### 7.3 RESIDUAL CODER — the named enemy (§4); build BOTH levers, de-share FIRST
```
LEVER-1 de-share (build in increment-1)   # DERIVED-EXACT: each residual px on ONE edge only (Laguerre factorization)
    attribute horizon secondary-arcs to the Movable carrier ; monotone DEFLATION ; magnitude UNMEASURED (owed $0 P4)
LEVER-2 curve-relative offset δ(s) coder   # DERIVED theory-optimal (2-D→1-D dim-reduction, §4); its own build
    code signed normal offset from the generator vs arc-length ; UPPER-BOUND on today's 0.4–0.6 B/px
    projected COMPLETE ≈ 0.077–0.083 S (1.4–1.5× < 0.118)   # CONJECTURED / DOUBLY-CONDITIONAL (δ(s) 4× compressibility UNMEASURED)
NO lossy operating point on the lane edge   # N-4 MEASURED: knee at lossless, every step RAISES S (verdict_scope FORMULATION)
FALLBACK if neither built for increment-1: ship dominant-only 0.061 (1.9× < frontier, thesis CONFIRMED) + residual as measured-owed sidecar
```

### 7.4 STAGED TRAINING (risk-2, from the energy — hand to S3/S4)
```
Stage A: each edge-field vs EXACT signed-SDF target (argmax(sdf)==labels) — NO cross-class gradient path  # DERIVED (∂φ_c/∂θ_{c'}=0)
Stage B: paint solved SEPARATELY vs the frozen scorer     # FORBIDDEN: end-to-end paint→R→SegNet (re-opens theft)
MERGE = tropical argmax = seam-free at LABEL level (#7) ⇒ reconciliation DIFFs R(paint) only, a bounded surface
per-edge σ via per-field eikonal/length/τ_c anneal        # DERIVED (§5) = the multiphase-Modica-Mortola σ_{cc'} made structural
```

---

## 8. SELF-ATTACK (§6 of the manual) — where this position is weakest

- **§4 curve-relative coder ΔS is CONJECTURED, not MEASURED.** The 2-D→1-D dimensional-reduction argument
  is derived, but the offset signal δ(s)'s actual compressibility (the assumed ~4×) is UNMEASURED — the
  δ(s) spectrum has never been computed. The projected complete ≈ 0.077–0.083 is DOUBLY-CONDITIONAL and
  could be worse if δ(s) is not C²-smooth where Σ_e is dashed (the dashes are exactly the high-curvature
  fragments the generator misses — δ(s) may inherit their roughness). **This is the single claim most
  likely to over-promise; it is flagged CONJECTURED and owed a $0 P4 measurement (compute δ(s) on the
  cache, measure its wavelet N-term).** The de-share lever (exact, monotone) is the safe recommendation;
  the curve-relative coder is the stretch.
- **The offset chart is ill-defined where Σ_e is multi-valued.** The normal-offset δ(s) presumes a
  single-valued normal from the generator; at horizon secondary-arc crossings and near Road multi-blob
  junctions the normal is multi-valued. So §4 and §5's "change of chart lowers the inflection count" holds
  on the SMOOTH arcs and DEGRADES exactly at the junctions — which is where the residual concentrates.
  Mitigation: the de-share lever removes the worst multi-valued case (secondary arcs → Movable). Residual
  honesty: the junction px may stay near their 2-D entropy even under the offset chart.
- **The flip-density MEDIAN b_c (§3) is DERIVED under a 1-D-margin reduction.** The flip mass is projected
  onto a single margin coordinate per edge; true flips have a 2-D annulus structure (#333) and the median
  is optimal for the projected 1-D problem, not the full 2-D placement. It should IMPROVE on mass-matching
  (which optimizes the wrong functional entirely) but the exact-optimal b_c is a harder 2-D problem. LABEL:
  DERIVED-form, superior-to-N-1-BY-CONSTRUCTION, exact-ΔS ASSUMED.
- **The 5-edge-star derivation (§2) rests on P-A's "zero interior flips at the ORACLE floor."** P-A is the
  UPPER bound (oracle paint); the generated-paint floor (P-C, UNRUN) could birth interior flips that add
  active edges not in the star. verdict_scope of "5-edge star": **CONDITIONAL on P-C** (the counted-seed
  paint floor). If P-C shows interior flips, the adjacency is denser than a star and the carrier count
  rises. **The 5-carrier claim is the increment-1 build; the n600 verdict rides increment-1, not the
  oracle.** (§8 mistake-5 — the oracle number is a hypothesis for the generated vehicle.)
- **σ-supplied-for-free (§5) is a MECHANISM claim, not a d_seg measurement.** v8's per-edge fields CAN be
  independently stiff, but whether that actually stops the MCF erasure through R at n600 is UNPROVEN until
  increment-1. The v7.5 baseline's Road floored at ~0.40 from cross-class theft; v8's decoupling is a BET
  (DELTA §J) and §5 makes the mechanism principled, not measured. Do not let "structurally supplies σ"
  read as "d_seg solved."

**Relative-significance ledger (all vs the remaining gap 0.0411 S to sub-0.15):** rate dominant-headroom
0.057 S = 1.39× · residual-coder enemy 0.079 S = 1.92× · N-1 b_c span 0.215 S = 5.2× · horizon parsimony
0.044 S = 1.07×. The rate levers are individually gap-sized-or-larger, which is WHY v8's rate story
matters — but **d_seg (Hamming placement) is the shared true blocker for BOTH vehicles, and every rate
win is CONDITIONAL on it holding through R.** Pointer 0.19110 UNMOVED — every line here is MEANS until a
byte-closed `upstream/evaluate.py` n600 row < 0.19110.
