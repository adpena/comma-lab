---
schema: brenier_polar_factorization_crosswalk.v1
date_utc: 2026-07-22
lane_id: lane_brenier_polar_factorization_crosswalk_20260722
research_only: true
execution_allowed: false
score_claim: false
promotion_eligible: false
main_landing_review_required: true
---

# Brenier polar factorization and almost-linear Bellman-Ford against the live v10 describe line

## Outcome first

**One narrow probe is worth adopting; the proposed `store only grad(psi)` codec is not.** Gangbo's
source-inspected theorem factors a nondegenerate vector map as the **composition**
`u = grad(psi) o s`, with `s` measure-preserving. It is not an additive Helmholtz decomposition,
and uniqueness does not make `s` free. In fact, the proof constructs
`s = grad(psi*) o u`, while Remark 4.6 warns that a measure-preserving `s` need not be one-to-one.
A decoder that lacks `u` cannot recover `s` from `psi` alone.

The live counted representations already avoid dense vector fields: they use one `se(3)` twist,
static/chart coefficients, sparse events, explicit Pose6 residual ownership, and unique-home
carrier composition. The measured `#574` attempt to reuse counted `xi` for Lane temporal coding
made the logical stream **8,508 B larger** and the deterministic ZIP **8,977 B larger**. The
`#553` PDW2 packet already performs the relevant affine gauge quotient in **138/134 raw bytes**;
its missing object is a spatial rank-4 field, not a non-unique coefficient gauge. Brenier does not
remove either debt.

The only adoption is a **pointwise one-dimensional monotone compander probe** for scalar values
inside the PRIMARY direct-description strata, followed by the already-measured stream-specific
coders. It must preserve semantic position and count the compander/codebook. Sorting a stream and
silently dropping the inverse permutation is explicitly forbidden.

Hair--Li--Li--Zhang's source-inspected result is a **Las Vegas randomized**
`m^(1+o(1))` algorithm for real-weight directed single-source shortest paths, with high probability.
Its node-potential identity is a genuine discrete gauge: fixed-endpoint path costs change by one
endpoint constant. But it is not itself a min-cost-flow implementation, a practical small-graph
speed result, or a rate theorem. Pact's live association problems already use exact Hungarian/LAP
on at most six lane slots or ten movable sites per frame; the measured lossless lane benefit is only
0.5%. The #307 segmentation/knot graphs are small DAG/one-dimensional DPs, and #586 is a disjoint
four-variable Diophantine solve plus a nonlinear hard oracle. None needs negative-edge SSSP.

Pointer delta: **none**. `0.1910828242 [contest-CPU Linux x86_64]` is unchanged.

## Ranked crosswalk

`ADOPT` means build only the named `$0` probe. `ALREADY-HAVE-BETTER` means the live representation
or exact solver is more direct and has stronger custody. `N-A` is scoped to the named consumer and
one-line reason; it is not a family kill.

| Rank | Verdict | Brenier leg | Named consumer | Evidence and pay-rent test |
|---:|---|---|---|---|
| 1 | **ADOPT (probe only)** | 1-D monotone rearrangement as pointwise companding, not positional sorting | `DirectDescriptionOpsGrammarMinimizerV1`: scalar values in `static_ground_coefficients`, `xi_curve_knots`, actual Pose6/`dxi`, and `exceptions`; backend baselines from `truly_optimal_coder_survey_603_613_20260722.md` | **MEASURED baselines:** static `610 B` Brotli, extracted xi `204 B` LZMA, Pose proxy `3,509 B` Rice, exceptions `80,478 B` global Brotli. **$0 exit:** on the same SHA-bound source objects, preserve indices/order, count CDF/levels/tag/framing, strict parse/re-encode and deterministic receive, then accept only if exact final-ZIP `Delta S < 0` with nonworse through-`R` Seg/Pose. Reject if an inverse permutation is needed or if isolated payload savings disappear in ZIP. |
| 2 | **ALREADY-HAVE-BETTER (live association)** | exact discrete assignment / min-cost-flow precursor | `movable_site_coder.track_sites`, `lane_track_and_smooth.coherent_slot_pack`, and `predictor_r3_causal` event matching | **MEASURED/live shape:** exact SciPy Hungarian/LAP already handles at most `6` lane slots or `1–10` movable sites per frame over n600; movable tracking is `5,161 B`, and lossless coherent Lane slotting improves only `41,526 -> 41,303 B` (0.5%). No executable global min-cost-flow or negative-edge graph exists. Hair's asymptotic SSSP changes no affordability boundary here; a future whole-clip flow remains the prior unmeasured formulation, not a new adoption. |
| 3 | **N-A** | almost-linear negative SSSP for contour segmentation and knot/event placement | `#307` contour-string coder plus PRIMARY `xi_curve_knots` / `sparse_events` | One-line reason: #307's connected components average only `3.1 px` and anchor bytes dominate, while exact 1-D Potts/trend knot placement and time-ordered event segmentation are DAG/chain DPs solvable by topological or classic dynamic programming; general cyclic negative-edge SSSP adds no capability. |
| 4 | **N-A** | negative SSSP for tie-tight uint8 preimage enumeration | `#586` / `uint8_lattice_feasibility.py` and the exact integer-realizable prefix consumer | One-line reason: each affine resize cell is an independent four-variable bounded Diophantine equation with suffix-range/gcd pruning, and the remaining winner-cell test is a nonlinear frozen-CNN hard oracle without shortest-path optimal substructure; no sound negative-SSSP reduction is present. |
| 5 | **N-A as a byte lever** | Johnson-style node-potential reweighting as a graph gauge | future whole-clip Lane/Movable assignment graph; current `(xi,R)` CGauge | One-line reason: `w_phi(P)=w(P)+phi(source)-phi(target)` preserves fixed-endpoint path order and cycle costs, so the gauge is mathematically real but encoder-internal; it deletes no edge, flow, event, or receiver byte. If a global flow graph lands, use its solver-native reduced costs without confusing `phi` with Brenier's convex `psi`. |
| 6 | **ALREADY-HAVE-BETTER** | uniqueness / temporal canonicalization | `#574` xi-keyed temporal coder on coherent-slot LBND2 | **MEASURED:** settled LBND2 `35,393 B`; identity plus counted xi `42,413 B`; planar-3 xi predictor `43,901 B`; ZIP `451,191 -> 460,168 B`. The chart had already absorbed ground motion. `s` would carry the temporal correspondence being predicted, so dropping it is not lossless. Keep LBND2; verdict scope is this Lane formulation only. |
| 7 | **ALREADY-HAVE-BETTER** | polar split as a codec decomposition | live `ddm_v9_carrier_compose_byteclose` / PRIMARY `DirectDescriptionOpsGrammarMinimizerV1` | **SOURCE-INSPECTED:** the counted grammar is static ground coefficients + xi curve + Pose6 residual + sparse events + entropy state + exceptions, under `fullstack_unique_home_assignment_v1`. It stores no dense vector field. A second `psi,s` factorization would add a representation boundary without eliminating an owned stream. |
| 8 | **ALREADY-HAVE-BETTER** | gauge fixing / monotone transport | `#553` PDW2 and `#576` scorer-free spatial receiver | **MEASURED:** PDW2 is already common-affine-gauge invariant at `138 B` margin / `134 B` partition-only, with strict float32 parse-back. **MEASURED blocker:** identical packet bytes admit distinct spatial partitions without the explicit quotient feature field (`PDW2_COEFFICIENT_ONLY_SPATIAL_NONIDENTIFIABILITY`). Brenier uniqueness of a different transport problem cannot infer that missing field. |
| 9 | **ALREADY-HAVE-BETTER** | semi-discrete Brenier / Monge-Kantorovich | `#539` exact rank-4 power-diagram target and `#288` damped-Newton head-offset solver | **MEASURED:** the five-class affine head is an exact Laguerre/Bregman-Voronoi diagram; the mass solver converged to max error `2.82e-11` in eight iterations, but realized n600 `d_seg` worsened `0.0031436 -> 0.0048921`. The solver is already present; its raw global-mass objective is formulation-negative. A smooth Monge-Ampere PDE is not a better solver for a discrete target. |
| 10 | **ALREADY-HAVE-BETTER** | Helmholtz-flavored gauge separation | `(xi,R)` CGauge, `#580` resize-nullity, and `argmax_native_vjp_fidelity_v1` | The live quotient is defined by the frozen scorer, `R`, Fisher/margin geometry, and `ker(A)`; it is not the Euclidean quadratic-transport equivalence class. The scorer-specific quotient and reverse-waterfill are the byte-relevant gauge. Brenier's unique quadratic-cost map would select a different canonical representative without proving score or byte preservation. |
| 11 | **N-A** | scalar potential versus rearrangement for a full displacement field | `#535` channel-space collateral / FiLM flicker sidecar | One-line reason: the named carrier is a low-rank chart/channel actuator with sparse gates, not a counted full vector field; no `u` stream exists for a polar split, and the missing n600 object is the renderer-to-scorer inner Jacobian. |
| 12 | **N-A** | drop rotational/measure-preserving motion | `advected_screw6_chartlevel` worldsheet/warp formulation | One-line reason: the `se(3)` rotation/transport is precisely information that would live in `s`; dropping it changes the frames. The n16 full-screw treatment already measured `d_seg` worsening `2.31x` despite a `3.17%` Pose improvement, so no byte credit is available. Verdict scope is that chart/framing formulation. |
| 13 | **N-A** | Monge-Ampere as an end-to-end inverse | `#542` alternative-forms conv-preimage line | One-line reason: the live inverse is through a mixed SiLU/ReLU scorer, shared resize, uint8 lattice, Pose, overlap, and archive grammar; it is not transport between two smooth positive densities. The existing exact-head QP plus corrected factorized adjoint is the applicable solve form. |
| 14 | **N-A (today)** | a future smooth-density Monge-Ampere regularity certificate | pending closed-scorer variational-DE `#611` | One-line reason: no `#611` task row, callable consumer, or declared smooth source/target density exists in this worktree, so registering or wiring an equation would invent a consumer. The draft equation note states the conditional scope for a future owner. |
| 15 | **N-A as live evidence** | existing code named “Brenier” | `brenier_quantile_quantize_1d` and `composition/wbce_mera.py::BrenierOTQuantizer` | One-line reason: repository search finds only tests/package-local use; the orphan audit independently marks `wbce_mera.py` unconsumed. These are scalar quantizers, not a live multivariate polar-factor codec, and cannot be cited as existing v10 adoption. Their behavior may inform rank-1's probe only after exact stream/receiver custody. |

## What the theorem actually licenses

The local Gangbo PDF (SHA-256
`c5f34dd3fd8ad2ec7ce0830ec18d2bce29074131eaaff4a53393da9e196190cf`) was read in full and visually
checked. For bounded `Omega` with negligible boundary and `u in L^p(Omega,R^d)` satisfying the
`N^-1` property, it establishes a normalized, unique factorization

`u(x) = grad(psi)(s(x))`,

where `psi` has a convex extension and `s` preserves Lebesgue measure. It also identifies `s` with
the optimizer of the projection/duality problem over measure-preserving maps and proves continuity
of the factors away from maps that fail `N^-1`.

Three scope guards bind the codec reading:

1. **SOURCE-INSPECTED:** `u = grad(psi) o s` is compositional. Calling `grad(psi)` a stored
   “curl-free part” and `s` a discardable residual would be an additive-Helmholtz category error.
2. **DERIVED:** uniqueness says which factors represent `u`; it does not say the conditional
   description length of `s` is zero. A psi-only decoder needs `s` from decoder side information or
   must prove that the final receiver is invariant to it.
3. **SOURCE-INSPECTED / live-scope:** many codec maps are quantized, piecewise constant, discrete,
   or lower-dimensional and therefore do not satisfy Gangbo's `N^-1` premise. The exact Laguerre
   head is Brenier-adjacent semi-discrete geometry, but the class-label map is not a direct instance
   of Gangbo's nondegenerate vector-map theorem.

These guards answer the uniqueness question: **none of the inspected counted streams can discard a
measure-preserving/rotational factor today.** The live system already spends bytes only on compact
generators, one xi chart, sparse critical events, and receiver-positive residuals.

## Discrete potential and negative-SSSP disposition

The local Hair--Li--Li--Zhang PDF (SHA-256
`297a5b1779631c550ba67608abe46f879d875ae96cbd9565c912525ecd7ea75e`) was read in full and visually
checked. Its Theorem 1.1 gives a Las Vegas randomized `m^(1+o(1))` algorithm, with high probability,
for single-source shortest paths in directed graphs with real, possibly negative, weights. The
construction recursively reduces negative-hop count using valid potentials, betweenness reduction,
shortcut edges, and an unfolding lemma that prevents iterative edge blow-up. It does not supply a
minimum-cost-flow implementation, a deterministic runtime, or practical constants for Pact's tiny
graphs.

For `w_phi(u,v) = w(u,v) + phi(u) - phi(v)`, telescoping gives the **SOURCE-INSPECTED / DERIVED**
fixed-endpoint identity

`w_phi(P from a to b) = w(P) + phi(a) - phi(b)`.

Thus node potentials preserve the ordering of all `a -> b` paths and preserve every cycle cost. This
is the honest discrete gauge connection: in a declared min-cost-flow/discrete-OT graph, dual prices
appear as reduced-cost potentials. It is not the same object as Brenier's convex `psi`, and it carries
no codec rate credit by itself.

The consumer audit closes four tempting overextensions:

1. Lane and Movable association are already exact LAPs on small matrices. A future global flow could
   improve a formulation, but its affordability was never blocked at n600 and Hair's asymptotic does
   not create the graph, objective, or byte win.
2. #307 contour traversal is dominated by 142,270 component anchors, not by a slow shortest-path
   solve. Time-ordered knot/event placement is a chain/DAG problem with simpler exact algorithms.
3. #586 separates into bounded four-coordinate integer equations; the nonlinear scorer repair has no
   shortest-path optimal substructure.
4. Johnson-style reweighting can make reduced costs algorithmically convenient, but it cannot delete
   a counted event or reconstruct a missing receiver-visible factor.

## Monge-Ampere disposition

For a stronger smooth setting where `T = grad(psi)` is a diffeomorphism pushing a positive density
`rho_0` to `rho_1`, change of variables gives the **DERIVED** equation

`det(D^2 psi(x)) = rho_0(x) / rho_1(grad(psi(x)))`.

That statement needs convexity, sufficient differentiability, positive absolutely-continuous
densities, and a transport boundary condition. None is currently declared by `#542/#611`. For the
actual five-Dirac semi-discrete target, the density ratio is singular; the correct condition is
the Laguerre cell-mass system `mu(L_k(b)) = nu_k`, already implemented by `#288`. Its n600 negative
shows that exact mass matching can still select the wrong pointwise partition.

Therefore Monge-Ampere adds no live inverse-solve actuator. The equations draft keeps a scoped
future guard so a later `#611` owner cannot mistake a smooth-density PDE for the mixed scorer/R
preimage.

## Rearrangement probe contract

The rank-1 adoption deliberately avoids positional sorting. For scalar stratum `c`, fit a monotone
CDF/compander `F_c`, apply it pointwise, quantize or integerize in the transformed coordinate, and
encode with the survey winner for that stream. Semantic index, persistent ID, time, and stratum
remain unchanged. The candidate pays for every level/CDF knot, tag, framing byte, and any residual.

Required controls:

- incumbent exact stream with its measured winner;
- identity compander through the same new framing, to expose container overhead;
- monotone pointwise compander with the same semantic order;
- forbidden-sort canary proving that deleting/reordering values without a counted inverse
  permutation changes receiver output or fails parse-back.

Admission is the shared reverse-waterfill rule on exact final ZIP bytes:

`Delta S = 100 Delta d_seg + [sqrt(10 d_pose_new)-sqrt(10 d_pose_old)] + 25 Delta B/37,545,489 < 0`.

This is a `$0` local probe proposal, not authority to execute, launch, or register a lever. It must
use Fisher/margin ranking, corrected inner-Jacobian realization, and a curvelet/shearlet basis for
any boundary residual; it does not reopen Fourier.

## Triality, no-orphan routing, and blocker delta

- **DAG:** `.omx/research/brenier_polar_factorization_crosswalk_DAG_FEED_20260722.md` routes the
  theorem through premise, receiver, and byte gates.
- **Equations:**
  `.omx/research/brenier_polar_factorization_crosswalk_canonical_equations_DRAFT_20260722.md`
  records four `FORMALIZATION_PENDING` guards plus the nonregistered compander identity. No registry
  row is appended because no empirical psi-only/compander anchor or live negative-SSSP graph exists.
- **DSL:** N-A with rationale. This research-only crosswalk admits no launch lever. If rank-1 passes,
  its owner must add one typed, default-OFF, receipt-bound grammar policy through the PRIMARY
  consumer; no raw flag.
- **Sensitivity/Pareto/bit allocator:** reuse per-stratum necessity, Fisher/margin, exact nonlinear
  Pose delta, and final-ZIP marginal. Allocate zero bytes until a receiver-closed winning row exists.
- **Cathedral/autopilot:** no dispatch candidate; retain one optional `$0` probe and explicit
  do-not-route states for psi-only and smooth-MA substitution.
- **Continual learning:** this memo plus the FEED is the durable negative/conditional signal. No
  empirical posterior or score row is emitted.
- **Probe disambiguator:** identity versus pointwise companding versus forbidden positional sort is
  the three-arm arbitration; it separates value normalization from an uncounted `s`.

**Blocker delta: none expected.** This crosswalk does not add a prerequisite to the live v9 compose
or PRIMARY direct-description line. It closes only the unsupported shortcut “uniqueness lets us
drop `s`,” preserves the existing `#553/#574/#535/#542` gates, and queues one optional `$0` byte
probe.

## STORES CONSULTED

- Delegated authority file (verified SHA-256
  `0ebd889c6a732487598fa4de480e1a91e2a03d9640ebe3c8d1f2bcc713594cf5`); `CLAUDE.md`;
  `AGENTS.md`; `PROGRAM.md`; `docs/operating_manual_craft_handoff.md`.
- Gangbo, *An elementary proof of the polar factorization of vector-valued functions* (local PDF,
  full 18-page file, source SHA above); Brenier 1991 abstract supplied by the operator.
- Hair, Li, Li, and Zhang, *Bellman-Ford in Almost-Linear Time* (arXiv:2607.19346v1, local PDF,
  full 38-page file, SHA-256
  `297a5b1779631c550ba67608abe46f879d875ae96cbd9565c912525ecd7ea75e`).
- v7.5 operating-contract spec; v8 per-class-carrier spec; v10 integer-plane spec;
  `reports/latest.md`.
- `direct_description_minimizer_PRIMARY_SPEC_20260721T214800Z.md`; current FEED-603 v8/v9 routing;
  DDM v8 memo/findings/session summary; truly-optimal-coder survey.
- `xi_temporal_delta_coder_574` memo/FEED/equations; `advected_screw6_chartlevel`; recursive-fractal
  #503 build spec/FEED; `collateral_coupling_geometry_and_film_flicker_sidecar`; `#542`
  alternative-forms memo.
- `pdw2_gauge_packet_probe_20260719_codex.md`; `pdw2_spatial_receiver_576` spec and receiver source;
  `deepmath_lens_tropical_ot_powerdiagram`; `laguerre_ot_head_offset_20260709.py`;
  `laguerre_logit_offset.py`.
- Existing scalar Brenier helpers and their import sites; orphan-signal audit; lane/task/subagent
  state; delegation inbox and operator broadcasts through `2026-07-21T13:15:53Z`.
- `lane_coeff_tracking_denoising_optimal_survey_20260702.md`; measured Wave-F coherent-tracking row;
  `movable_site_coder.py`; `lane_track_and_smooth.py`; #307 contour-string memo/source; V10 bounded
  uint8 lattice spec/findings/source; `predictor_r3_causal.py`; per-arm supplement at
  `2026-07-22T12:43:46Z`.

This memo follows `docs/operating_manual_craft_handoff.md`: verdict first, primary-artifact
re-derivation, explicit evidence labels, narrow negative scopes, and pointer honesty. MAIN must
review the complete branch diff before landing.
