# `tac.lie` — MLX-native differentiable se(3)/SE(3) Lie-group library LANDED

**UTC** 2026-06-30T22:39:27Z · **authority** `[macOS-MLX research-signal / give-back component — advisory]`
**pointer UNMOVED 0.19110** · score_claim **false** · promotable **false** · ready_for_exact_eval_dispatch **false**

**means≠ends.** This library is an enabling MEANS for the v2 Stratified Screw-Warped Level-Set witness
(the screw warp, pose=screw, ξ_ego(t) trajectory, ground-frame canonicalization). It is ALSO a
community/science give-back (the MLX ecosystem has no Lie library). It does NOT move the pointer; the
pointer moves only on a byte-closed exact row below 0.19110.

## What was built (spec: `screw_twist_se3_literature_enrichment_20260630T220000Z.md` §1, build order)

A standalone, MLX-native, autodiff-clean, **parity-gated** Lie-group package at `src/tac/lie/` (MIT;
clean-room from textbook math — Solà micro-Lie 1812.01537, Barfoot 2017, Sommer et al. CVPR 2020, Kavan
et al. ACM TOG 2008; NO GPL/LGPL/MPL copied — basalt(MPL)/dqrobotics(LGPL) implemented FRESH).

| Module | Lines | Contents |
|---|---|---|
| `_se3_numpy.py` | 410 | **NumPy reference oracle (the bit-identical authority)** — dtype-faithful (fp32 authority / fp64 golden). SO(3)/SE(3) exp/log/Adjoint/ad/J_l/J_r(+inverses), Barfoot Q-matrix + an INDEPENDENT `Σ adⁿ/(n+1)!` series oracle. |
| `so3.py` | 160 | MLX SO(3): hat/vee, exp(Rodrigues)/log(atan2 form), J_l/J_r/J_l⁻¹/J_r⁻¹, small-angle Taylor + grad-safe branches. |
| `se3.py` | 164 | MLX SE(3): make_T/rotation_of/translation_of, exp/log, compose, analytic inverse, Adjoint, ad, full 6×6 left Jacobian (incl. Q). |
| `screw_blend.py` | 191 | Dual-quaternion DLB (Kavan seam-fix) + ScLERP (screw geodesic), se3↔dq, quat mul — numpy oracle + MLX. |
| `se3_bspline.py` | 139 | Cumulative cubic SE(3) B-spline (Sommer–Usenko) for ξ_ego(t) — numpy oracle + MLX, domain-clamped. |
| `__init__.py` | 91 | Clean typed public API + convention/standalone docstring. |

Convention (fixed + asserted): twist `ξ=(ρ,ω)` **translation-first**; SE(3) = `(...,4,4)`; quaternions
scalar-first `[w,x,y,z]`; dual quats `(...,8)`. Batched via broadcasting. Device-agnostic (no device pin;
GPU fast path stays legal for the future per-pixel warp).

## Tests — 49, all green (`src/tac/tests/test_lie_library.py`)

Three-tier authority (numpy-fp64 golden → numpy-fp32 authority → MLX-fp32 fast), CPU-pinned (MLX-GPU uses
fast-math transcendentals, ~1e-3 drift; cf. MLX #2205). Covers: per-op parity across a log-spaced θ grid
[1e-9, π-1e-3]+0; algebraic identities (exp∘log, Ad(AB)=AdA·AdB, `T exp(ξ)T⁻¹=exp(Ad·ξ)`, J·J⁻¹=I,
J_r(ξ)=J_l(−ξ), exp(−ξ)=inv); external **scipy** cross-check (exp/log/dual-quat); the Q-matrix two ways;
differentiability incl. the where-NaN trap; a **gradcheck** (analytic vs numpy-FD, catches wrong-but-finite
grads); screw-blend (roundtrip/unit-DQ/DLB endpoints+antipodal/ScLERP geodesic); B-spline
(constant/on-manifold/C0/domain-clamp/autodiff velocity); standalone-import guard.

## Returned answers (the asks)

- **Modules built:** 6 files, 1155 LOC lib + 1 test file. ✅ all three build-order tiers (core SO(3)/SE(3);
  dual-quat screw-blend; cumulative SE(3) B-spline).
- **Test count + green:** **49 tests, all passing** (`pytest src/tac/tests/test_lie_library.py`).
- **Q-matrix finite-diff verdict:** **VERIFIED.** Closed-form vs central finite-diff of `log_se3` = 5.7e-11;
  closed-form vs the independent `Σ adⁿ/(n+1)!` series = 1.6e-15; both hold across θ∈[1e-6, ~π]. The flagged
  formula is trustworthy (Barfoot 7.86b, liegroups coefficient form).
- **External-oracle cross-check:** jaxlie/liegroups/spatialmath/pytransform3d/sophuspy **NOT installed**;
  used **`scipy.spatial.transform.RigidTransform`** (exp/log/dual-quat, block-swapped rotation↔translation
  ordering) → exp 1.1e-10, log 1.2e-13, dual-quat 1e-16 (sign-aligned); plus an INDEPENDENT
  `scipy.linalg.expm/logm` 4×4 oracle → exp/log 2e-15. Beyond scipy: numpy oracle + algebraic identities.
- **Recursive-adversarial-review clean-pass count:** **3 consecutive clean passes (SEALED)** after Round 1.
  - **Round 1 — 2 findings, both FIXED + regression-tested:** (1) `se3_bspline_eval` silently wild-extrapolated
    for out-of-domain `t` (clamped the segment index but not `u`) → now clamps `u∈[0,1]` (boundary pose).
    (2) `left_jacobian_se3` (Q) had a **NaN gradient at exactly ξ=0** — the O(θ⁵) m4 term needs the
    **double-where** idiom (evaluate the exact branch at a benign θ where unused); denominator-clamping
    alone was insufficient.
  - **Round 2/3/4 — CLEAN:** exhaustive grad sweep (every op, at 0 / random / near-π) all finite;
    large-ρ stable; ad = d/dt Ad(exp tξ); Ad(T⁻¹)=Ad(T)⁻¹; associativity; ScLERP near-π; B-spline C1;
    DLB 3-way/antipodal/batched; branch-boundary continuity; `mx.compile` clean; gradcheck (analytic vs
    numpy-FD) = 2e-7 in fp32 AND fp64. Two apparent Round-4 anomalies (fit loss increase; an ad-hoc
    gradcheck64 helper reading 0.26) were investigated and proven **probe-script artifacts** (lr=0.5
    oscillation; buggy helper) — NOT library defects (the fit converges to 1e-15 at lr=0.1).
- **Standalone confirmation:** `tac.lie` imports **NOTHING** from the witness residual pipeline / scorer /
  trainer (subprocess-isolated import test asserts no `scorer/witness/residual/compose_witness/v2_compose/
  renderer/trainer/train_` module is pulled). It depends only on `numpy` + `mlx`. ✅

## Wire-in (Catalog #125)

1. sensitivity-map: N/A here (enabling primitive; the boundary screw-blend sensitivity row lands when #3 is
   measured through R). 2. Pareto: N/A (no rate↔distortion arm yet). 3. bit-allocator: N/A. 4. cathedral
   autopilot: N/A (not archive-deployable). 5. continual-learning: this memo + DAG FEED. 6.
   probe-disambiguator: the existing `tools/measure_screw_warp_through_R.py` is the downstream consumer.

**Consumes-next (NOT done here — gated design-refine step):** the witness canonicalize-to-ground-frame +
per-class screw warp + ξ_ego(t) spline-fit will import `tac.lie`. Metal-acceleration of the per-pixel
warp/blend is deferred (profile-then-accelerate per spec §1.7).
