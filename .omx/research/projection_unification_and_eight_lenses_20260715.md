# The projection unification + the eight lenses (durable framing) — 2026-07-15

**Source:** operator 2026-07-15 — *"All about projection since we've solved the deep math and geometry"* +
*"Pursue the unification as p0 of all p0"* + *"Can't we determine what is necessary for realization by
inverse of the flattened factorization?"* + *"per edge and saddle"* + *"v9 cgauge is v8 but with even more
optimal carriers per class"* + *"What other questions or framings or perspectives or lenses..."* + *"Register
those lenses ... durable framing memo too no signal loss."* This memo is the durable framing so the
consolidation survives compaction. Pointer 0.19108 UNMOVED — everything here is MEANS to the exact row;
the exact byte-closed `evaluate.py` row is the only authority (lens 8). Supreme P0:
`p0_UNIFICATION_projection_preimage_SUPREME_20260715`; facets: `p0_all_lenses_facets_of_unification_20260715`.

## 0. The claim
The deep math + geometry are solved: (a) the SegNet head is **rank-4 linear** (`segnet_head_rank4_linear_flipdist_v1`),
so the argmax is a **fixed hyperplane arrangement**; (b) the argmax partition is a **Laguerre power diagram =
Morse-Smale complex** (v8 #284); (c) the **optimal metric is Fisher** `g=∇²F(θ)` (#500 `optimal_metric_unification_v1`;
margin IS the Fisher surrogate, Pearson 0.978), not Euclidean; (d) **d_seg is realization-limited, not
gradient-limited** (the closed-form flip is sub-LSB in pixel space → the target is reachable). Given all four,
there is no "training" left — **only PROJECTION**. Those four are exactly the ingredients a projection needs: a
linear map to project through (rank-4), a convex cell structure to project onto (Laguerre), the right inner
product to project in (Fisher), and a reachable target so the projection lands (realization).

## 1. The primal — the witness IS a projection
The witness = **Dykstra alternating projection onto the feasible intersection**
`{argmax-correct} ∩ {pose-tube} ∩ {uint8-reachable} ∩ {cheap}`, **in the Fisher metric**. Solve-don't-train
(#342). Closed-form where the sets decouple; alternating-projection where they couple. In-tree operators:
Dykstra/ADMM (`joint_admm_coordinator`, `constrained_gen`), Fisher-natural (`fisher_natural_solver_policy`,
`bregman_dual_metric_guard`), quotient/preimage/null (`unified_action`, `scorer_exploits`, `precompute_corrections`,
`resize_null_preimage_compiler` #49), semi-discrete OT / Laguerre (`laguerre_logit_offset`, #288), PCGrad (`se3`).

## 2. The inverse — NECESSITY by preimage (the crown)
The dual of "project the witness onto the feasible set" is "**compute the feasible set** = the preimage of the
target." Invert the flattened chain `score = (decision) ∘ (N) ∘ (A)` (see `frozen_scorer_exact_factorization_20260715.md`):
- **decision⁻¹** = the rank-4 argmax polytope `P_c* = {f : ⟨w_c*−w_c, f⟩ ≥ 0 ∀c}`; the 10 boundary normals are the
  known head rows. Necessary = polytope membership only.
- **N⁻¹** = local `J_f⁺·(Π_P(f)−f)` (feature-Jacobian pseudoinverse; atlas #36). Free = `ker J_f` + in-polytope dirs.
- **A⁻¹** = the exact resize preimage `x_particular + ker(A)` (#49/#391). Free = `ker(A)` (blind complement).
- **∩ uint8 lattice** = the realization-limited intersection (may be tight — the sub-LSB tail).

**NECESSARY for realization = the min-description preimage; FREE = ker(A) ⊕ ker(J_f)|in-polytope ⊕ Fisher-flat
interior ⊕ sub-top-1-logit (= §8 blind complement). RATE FLOOR = |necessary preimage| = the quotient-codec
sufficient statistic (#155) computed by INVERSION, not search.** Computed **per Morse-Smale stratum**
(operator "per edge and saddle"): CELL (per-class palette) + EDGE (per-pair separatrix δ(s)) + SADDLE (triple-junction
tie-locus, #360 — likely dominates the necessary rate). Live arm: `p0_UNIFICATION` necessity solver.

## 3. V9·CGauge = v8++ (the vehicle for the above)
v8 = per-**class** Morse-Smale carriers (2-cells only). **V9·CGauge = the FULL Morse-Smale stratification
(cell + edge + saddle), each carrier's CONTENT = the necessity-solver's minimal preimage, free complement dropped.**
That is "even more optimal carriers per class" — not a bigger net, the **completed complex** (v8 carried interiors +
half the boundaries; v9 adds the separatrices' exact content and the saddle 0-cells). Carriers: cell→`decoupled_field`/
`road_undriv_bulk_field`; edge→`curve_relative_offset_coder` δ(s); saddle→a per-saddle point-code. Chroma finding
(rgb_at_boundaries_derivation_20260715): per-cell palette is TRUNK (6.2× worth); per-separatrix chroma is the finisher.

## 4. THE EIGHT LENSES — the seven faces of one projection + the honesty gate
One operator, eight views. Each = a probe → a V9·CGauge per-stratum carrier.
| # | Lens | What it is / asks | In-tree surface / arm |
|---|------|-------------------|------------------------|
| 1 | **DUAL / co-projection** | the rate side = `ker` complement of the distortion projection; code in the Bregman/CGauge MIRROR where the sufficient statistic is linear | #504 `bregman_dual_metric_guard`; the necessity solver's FREE set |
| 2 | **INVARIANCE / gauge** | the scorer's full SYMMETRY GROUP = free bytes (blind-coord, argmax-interior, sub-top-1-logit, uint8-sublattice + permutation/gauge) | Weyl #464, `witness_dsl/gauge.py`; §8 blind complement |
| 3 | **ADVERSARIAL / evasion** | Yousfi/UNIWARD DUAL of projection: hide the uint8-UNREACHABLE residual in the Fisher-FLAT null (minimize detection, not error) | #141 margin-saliency, `msal_uni`, #500 Fisher; live arm (evasion) |
| 4 | **FIXED-POINT / contraction** | is the Dykstra feasible-intersection point UNIQUE / convergent (closed-form) or non-convex multi-fixed-point (which is min-byte)? | Dykstra/ADMM surfaces; where "solve" is one-shot vs iterated |
| 5 | **TEMPORAL / advection** | project the Morse-Smale partition ONCE, transport by the ego-screw ξ∈se(3); ξ-redundant strata = FREE (trajectory rate amortization) | #194 `se3.py`/`tac.lie`, #424/#425 phase-carrier, #365; live arm (temporal) |
| 6 | **SCALE / persistence** | project coarse-cell-palette → fine-separatrix in Morse-Smale PERSISTENCE order = the curriculum; renormalization-group (stride-2 stem, ERF r90≈300px) | #284 persistence, curvelet #502, the curriculum |
| 7 | **MDL / generator** | the shortest PROGRAM (not data) whose fixed point is the witness; compile the projection into inflate.py (rule 118), store only the irreducible projected coords | rule-118 generator discipline; `unified_action`; #155 quotient |
| 8 | **FALSIFICATION (honesty gate)** | "it's all projection" is a CLAIM; the exact byte-closed `evaluate.py` row confirms/refutes each projection prediction; any projection↔exact gap IS the next finding | NO-FAKE; `tools/levelset_byte_close_and_eval.py`; every arm ends here |

Lenses 1/2/6/7 are largely SUBSUMED by the necessity solver (they name parts of the preimage/free split); lenses
3 (evasion) and 5 (advection) are distinct live arms; lens 4 (fixed-point) is a structural question on the
preimage; lens 8 is the standing authority every arm terminates in.

## 5. Cross-refs (no signal loss)
`frozen_scorer_exact_factorization_20260715.md` (§6 seen / §8 blind) · `segnet_recursive_fractal_factorization_20260715.md`
(rank-4 head, realization-limited) · `rgb_at_boundaries_derivation_20260715.md` (chroma=palette trunk) ·
`necessity_solver_inverse_factorization_20260715.md` (crown, in flight) · quotient codec #155 · resize preimage #49/#391 ·
Laguerre/Morse-Smale #284/#180 · Fisher metric #500 · Bregman #504 · se(3) #194 · phase-carrier #424/#425 · tie-locus #360 ·
SPEC_v8 per-class carriers #359/#380/#386. The projection frame does not replace any of these — it NAMES the one
operator they are all facets of.
