---
schema: canonical_equations_draft_note.v1
date_utc: 2026-07-22
lane_id: lane_brenier_polar_factorization_crosswalk_20260722
registry_action: none
formalization_status: FORMALIZATION_PENDING
research_only: true
score_claim: false
main_landing_review_required: true
---

# Canonical-equations draft — Brenier polar factorization codec guards

No registry row is authorized by this note. Both source papers are inspected, but the codec-rate
law has no receiver-closed empirical anchor, the smooth Monge-Ampere law has no declared live
consumer, and no live negative-edge graph has a measured SSSP bottleneck. MAIN should keep all
four guards as `FORMALIZATION_PENDING` unless the stated gates land.

## Candidate 1 — `polar_factor_codec_conditional_rate_guard_v1`

### Source theorem

Under Gangbo's stated bounded-domain and `N^-1` assumptions, normalized factors satisfy

`u = grad(psi) o s`,

with convex `psi`, measure-preserving `s`, uniqueness of the factors, and
`s = grad(psi*) o u` almost everywhere.

Evidence label: **VERIFIED_VIA_SOURCE_INSPECTION** for the theorem. The local 18-page PDF has
SHA-256 `c5f34dd3fd8ad2ec7ce0830ec18d2bce29074131eaaff4a53393da9e196190cf`.

### Codec guard

Let `Y` be legal decoder-side information and `G` the final deterministic receiver/scored-output
map. A polar representation has honest counted length

`B_polar = B(grad(psi)) + B(s | grad(psi), Y) + B_framing`.

A psi-only representation is exact only if at least one receiver-checked condition holds:

1. `s` is deterministically derivable from `(grad(psi),Y)`, equivalently zero conditional coding
   debt for the declared grammar; or
2. `G(grad(psi) o s) = G_tilde(grad(psi),Y)` for every admitted `s` in the declared equivalence
   class.

Uniqueness of `(psi,s)` does **not** imply either condition. Gangbo's construction of `s` uses `u`,
which is unavailable to a decoder that is trying to reconstruct `u` from psi alone.

Rate/quality admission is

`Delta S = 100 Delta d_seg + sqrt(10 d_pose_new) - sqrt(10 d_pose_old) + 25 Delta B/37,545,489 < 0`,

on exact final ZIP bytes after strict parse/re-encode and deterministic receiver replay.

Evidence label: **INFERRED_FROM_DOMAIN_LITERATURE** for the conditional-rate guard;
**ASSUMED_AWAITING_VERIFICATION** for any proposed consumer.

### Registration gate

Register only after a named live vector-map stream provides:

- a verified `N^-1` or explicitly weaker theorem scope;
- exact source/vector-map and factor custody;
- a counted `s` or a receiver proof that `s` is derivable/invisible;
- baseline/polar exact final ZIP bytes and receiver-identical output;
- nonworse n600 through-`R` Seg/Pose and separate authority-axis labels.

Current verdict: `FORMALIZATION_PENDING_NO_PSI_ONLY_CONSUMER`.

## Candidate 2 — `monge_ampere_density_transport_scope_guard_v1`

### Smooth density form

If `T = grad(psi)` is sufficiently smooth and invertible, `psi` is convex, and `T` pushes a
positive absolutely-continuous density `rho_0` to `rho_1`, then change of variables gives

`rho_0(x) = rho_1(grad(psi(x))) det(D^2 psi(x))`,

or

`det(D^2 psi(x)) = rho_0(x) / rho_1(grad(psi(x)))`,

with the appropriate transport boundary condition.

Evidence label: **INFERRED_FROM_DOMAIN_LITERATURE**. This PDE is a stronger smooth corollary, not a
formula stated or proved in the inspected Gangbo companion.

### Semi-discrete scope guard

For a target `nu = sum_k nu_k delta_{p_k}`, the density ratio above is singular. The applicable
finite condition is instead

`mu(L_k(b)) = nu_k for every Laguerre cell L_k(b)`.

That condition is already implemented by `#288` damped-Newton semi-discrete OT. Its exact numerical
convergence does not imply pointwise scorer agreement: the recorded n600 formulation moves realized
`d_seg` from `0.0031436` to `0.0048921` despite mass residual `2.82e-11`.

Evidence label: **VERIFIED_VIA_EMPIRICAL_ANCHOR** for the named #288 formulation; negative scope is
raw global class-frequency matching at that checkpoint/tau, not OT or Monge-Ampere as families.

### Registration gate

Do not register a new Monge-Ampere runtime law until `#611` or another named owner declares:

- positive a.c. source/target densities and a coordinate chart;
- convex-potential, regularity, boundary, and nonfolding predicates;
- a callable residual whose zero changes a real inverse-solve decision;
- comparison against the existing exact-head QP, semi-discrete solver, and corrected factorized
  adjoint;
- real receiver/through-`R` evidence, not only PDE residual.

Current verdict: `FORMALIZATION_PENDING_NO_LIVE_SMOOTH_DENSITY_CONSUMER`.

## Candidate 3 — compander probe identity, not a registered law

For stratum `c`, a pointwise monotone compander has the form

`q_i = Q_c(F_c(v_i))`, `vhat_i = F_c^-1(q_i)`,

with semantic position `i` unchanged. Its complete bytes include `F_c`/levels, quantizer state,
symbols, backend model/tag, framing, and residuals. A positional sort has the different form
`v_sorted = v o s` and requires a counted inverse permutation unless the receiver proves invariance.

This identity exists only to prevent the rank-1 probe from laundering `s`. It should not enter the
canonical registry until an exact final-ZIP/receiver anchor beats a named incumbent.

## Candidate 4 — `negative_sssp_live_graph_applicability_guard_v1`

### Source theorem

Hair--Li--Li--Zhang Theorem 1.1 supplies a Las Vegas randomized algorithm for single-source
shortest paths on directed graphs with real, possibly negative, edge weights in
`m^(1+o(1))` time, with high probability.

Evidence label: **VERIFIED_VIA_SOURCE_INSPECTION**. The local 38-page arXiv:2607.19346v1 PDF has
SHA-256 `297a5b1779631c550ba67608abe46f879d875ae96cbd9565c912525ecd7ea75e`.

The theorem is not a minimum-cost-flow implementation and makes no small-instance constant-factor
claim. A discrete-OT or tracking formulation may call SSSP inside a flow solver, but that separate
reduction, algorithm, and implementation remain owed.

### Applicability guard

Let `G_C=(V_C,E_C,w_C)` be the explicitly constructed graph for named consumer `C`. Admit this
algorithm family only if all of the following hold:

1. `G_C` is directed and contains real negative weights while excluding reachable negative cycles;
2. SSSP is a measured encoder-side wall under the incumbent exact solver;
3. graph construction, flow iterations, reconstruction, and receiver verification are separately
   costed;
4. an implementation beats the incumbent on the exact n600 graph while returning the same optimum;
5. no runtime improvement is claimed as a byte or score improvement without exact final-ZIP evidence.

Current consumers fail gate 1 or gate 2. Lane/Movable association uses exact Hungarian/LAP on at
most six/ten items per frame; #307 knot/event problems are chains/DAGs; #586 is bounded
Diophantine enumeration plus a nonlinear hard oracle.

Evidence label: **VERIFIED_VIA_REPOSITORY_INSPECTION** for current graph absence and consumer forms;
**ASSUMED_AWAITING_VERIFICATION** for any future min-cost-flow graph.

Current verdict: `FORMALIZATION_PENDING_NO_LIVE_NEGATIVE_SSSP_CONSUMER`.

## Candidate 5 — `graph_node_potential_fixed_endpoint_gauge_v1`

For node potential `phi:V->R`, define

`w_phi(u,v) = w(u,v) + phi(u) - phi(v)`.

For any directed path `P=(v_0,...,v_k)`, telescoping gives

`w_phi(P) = w(P) + phi(v_0) - phi(v_k)`.

Therefore every path between the same endpoints receives the same additive shift, the shortest-path
argmin set is invariant, and every cycle weight is invariant. In a declared discrete-OT/min-cost-flow
graph, the related reduced-cost node prices are a discrete Kantorovich-potential gauge. This does
not identify `phi` with Brenier's convex `psi`, and the identity alone removes no counted flow,
assignment, event, or receiver state.

Evidence label: **VERIFIED_VIA_SOURCE_INSPECTION** for the reweighting definition and fixed-endpoint
property; **INFERRED_FROM_DOMAIN_LITERATURE** for the discrete-OT dual-price interpretation.

### Registration gate

Register only when a named live graph consumer exposes the potential in a callable solver contract,
proves exact optimum invariance before/after reweighting, and records whether `phi` is purely
encoder-internal or counted. No rate credit is allowed without a separately measured payload change.

Current verdict: `FORMALIZATION_PENDING_NO_LIVE_GRAPH_GAUGE_CONSUMER`.

## Triality and stores

- DAG: `brenier_polar_factorization_crosswalk_DAG_FEED_20260722.md`.
- DSL: N-A; no lever or raw flag is admitted by a draft.
- Equations: four scoped guards plus one probe identity above; registry append count is exactly zero.

STORES CONSULTED: delegated authority; Gangbo local PDF; paired crosswalk memo; `#288` equation and
solver source; `#539/#553/#574/#535/#542` artifacts; pending `#611` repository search; PRIMARY DDM
spec and coder survey; Hair--Li--Li--Zhang local PDF; live Lane/Movable Hungarian consumers; #307
contour-string source/receipt; #586 uint8-lattice source/spec; delegation supplement;
`docs/operating_manual_craft_handoff.md`; v7.5/v8 contracts; current canonical-equations registry.
Pointer `0.1910828242 [contest-CPU]` unchanged. MAIN landing review required.
