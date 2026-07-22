---
schema: brenier_polar_factorization_crosswalk_dag_feed.v1
date_utc: 2026-07-22
lane_id: lane_brenier_polar_factorization_crosswalk_20260722
research_only: true
execution_allowed: false
score_claim: false
main_landing_review_required: true
---

# FEED-BRENIER-POLAR-20260722

## DAG delta

```text
[SOURCE-INSPECTED GANGBO THEOREM]
  u in Lp, N^-1, bounded domain, negligible boundary
  u = grad(psi) o s; psi convex; s measure-preserving
  normalized factors unique; s = grad(psi*) o u
  s need not be one-to-one
                 |
                 v
[CODEC PREMISE GATE]
  Is the named live object a nondegenerate continuous vector map?
       | no ------------------------------------------------------+
       |                                                          |
       v yes                                                      v
  Can receiver derive s from psi + legal side information,   [NO PSI-ONLY CREDIT]
  or is final scored output invariant to s?                       |
       | no ------------------------------------------------------+
       |
       v yes
  Is B(psi)+B(s|psi,side)+framing smaller than incumbent,
  on exact final ZIP with nonworse through-R Seg/Pose?
       | no --> [NO ADOPTION]
       | yes
       v
  [ADMIT ONLY RECEIVER-CLOSED MEASURED ROW]

[SOURCE-INSPECTED HAIR--LI--LI--ZHANG THEOREM]
  directed real-weight graph, possibly negative edges
  Las Vegas randomized SSSP in m^(1+o(1)), high probability
  w_phi(u,v) = w(u,v) + phi(u) - phi(v)
  fixed-endpoint path order and cycle costs invariant
                 |
                 v
[GRAPH APPLICABILITY GATE]
  Does a named live consumer expose a cyclic negative-edge graph?
       | no --> [USE SMALL LAP / DAG DP / DIOPHANTINE SOLVER]
       | yes
       v
  Is SSSP the measured encoder bottleneck, not graph construction,
  receiver verification, or final-ZIP rate?
       | no --> [NO ALGORITHM ADOPTION]
       | yes
       v
  Compare a real implementation with the incumbent on the same graph;
  potentials are solver-internal and earn zero receiver-byte credit

[NAMED CONSUMER ROUTING]
  #574 LBND2 + xi predictor
    -> MEASURED +8,508 logical / +8,977 ZIP
    -> ALREADY-HAVE-BETTER; keep LBND2

  live v9 compose / PRIMARY DDM grammar
    -> static coeffs + xi curve + Pose6 + sparse events + exceptions
    -> no dense vector field; unique-home already binds
    -> ALREADY-HAVE-BETTER

  #553 PDW2 / #576 receiver
    -> common-affine gauge already quotiented to 138/134 B
    -> missing spatial rank-4 field is nonidentifiable from packet
    -> ALREADY-HAVE-BETTER; Brenier does not close receiver

  #535 FiLM/collateral
    -> low-rank chart actuator, not dense vector payload
    -> N-A

  worldsheet/full-screw warp
    -> rotation/temporal correspondence lives in s
    -> dropping s changes frames; N-A

  #539/#288 semi-discrete OT
    -> exact Laguerre solver already present
    -> n600 global-mass objective worsens d_seg
    -> ALREADY-HAVE-BETTER, formulation-negative objective

  #542 / pending #611
    -> mixed scorer + R + uint8 + archive is not smooth density OT
    -> no live #611 consumer in worktree
    -> N-A today; conditional scope note only

  Lane/Movable/event assignment
    -> exact Hungarian/LAP already runs on <=6 lane or <=10 site rows/frame
    -> no executable global min-cost-flow/negative-edge graph
    -> ALREADY-HAVE-BETTER for the live association instances

  #307 contour string + xi knot/event placement
    -> anchor fragmentation is the measured rate wall
    -> components/temporal states are small chains or DAGs
    -> N-A for general negative-edge SSSP

  #586 bounded uint8 lattice
    -> independent four-variable Diophantine blocks + nonlinear hard oracle
    -> no shortest-path optimal substructure; N-A

  Johnson-style graph potential gauge
    -> fixed-endpoint argmin invariant; cycle weights invariant
    -> solver-internal only, no byte deletion; N-A as codec lever

[ONE ADOPTED PROBE]
  PRIMARY scalar stratum, semantic order fixed
    -> pointwise monotone CDF compander
    -> survey-winner backend + counted CDF/levels/tag/framing
    -> exact parse/re-encode + deterministic receiver + final ZIP
    -> Fisher/margin + corrected-inner-Jacobian + nonlinear Pose pricing
    -> ADOPT iff Delta S < 0
    -> positional sort without counted inverse permutation = REFUSE
```

## Decisions encoded

1. **Uniqueness is not a rate theorem.** It identifies factors under the theorem's premises; it
   does not prove `B(s | psi, decoder_side_info)=0`.
2. **Polar factorization is composition, not additive Helmholtz.** No `curl-free payload + free
   residual` claim is admitted.
3. **The live line has no full dense vector stream.** It already factorizes geometry into compact
   xi/chart/event owners, so no psi-only replacement enters the critical path.
4. **Smooth Monge-Ampere is scope-guarded.** Semi-discrete cell masses stay with `#288`; the full
   scorer preimage stays with exact-head QP plus corrected factorized adjoint.
5. **Rearrangement earns one byte probe only.** Pointwise monotone companding is distinct from
   sorting positions; exact final ZIP and receiver semantics decide.
6. **Almost-linear SSSP is not almost-linear min-cost flow by implication.** The inspected paper
   proves the former. A live flow consumer still owes its graph, objective, implementation, and
   measured bottleneck.
7. **Graph potentials are a gauge, not a codec.** They preserve fixed-endpoint path ordering by a
   telescoping endpoint constant; they do not remove counted payload.

## Six-hook routing

- Sensitivity: consume `realization_necessity_preimage_per_stratum_v1` and Fisher/margin; no new map.
- Pareto: one same-artifact `(Delta bytes, Delta d_seg, Delta d_pose)` tuple.
- Bit allocator: zero allocation until the compander beats its exact stream incumbent.
- Cathedral/autopilot: `research_only=true`; no dispatch or promotion edge.
- Continual learning: retain the psi-only refusal and conditional probe in this FEED; no empirical
  posterior row.
- Probe disambiguator: identity framing / pointwise compander / forbidden positional sort canary.

## Triality

- DAG: this file.
- Equations: `brenier_polar_factorization_crosswalk_canonical_equations_DRAFT_20260722.md`, both
  continuous and both discrete guards `FORMALIZATION_PENDING`, no registry append.
- DSL: N-A with rationale until a receiver-closed winning probe exists; eventual integration must
  be one typed PRIMARY grammar policy, default OFF and receipt-bound.

## Verdict scope and pointer

The negative applies only to storing `grad(psi)` while omitting an underived, receiver-visible `s`,
and to substituting smooth Monge-Ampere for the named live inverse. It does not close Brenier maps,
optimal transport, monotone companding, smooth density transport, or future consumers satisfying
the premises.

Blocker delta: none expected. Pointer `0.1910828242 [contest-CPU]` unchanged. MAIN landing review
required.

## STORES CONSULTED

See the paired crosswalk memo. This FEED additionally rechecked both local source PDFs, the live
Lane/Movable Hungarian consumers, #307 contour-string coder, #586 uint8 lattice solver, current
lane/task/subagent state, and delegation inbox immediately before checkpointing. It follows
`docs/operating_manual_craft_handoff.md` and preserves the operator's 2026-07-19 Fisher/margin,
corrected-inner-Jacobian, curvelet/shearlet, xi-factorization, and reverse-waterfill directives.
