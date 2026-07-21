# DAG FEED — ACME Foundations crosswalk (Task #593)

```yaml
feed_id: acme_foundations_crosswalk_20260721T014505Z
lane_id: acme_foundations_crosswalk
research_only: true
execution_authority: false
score_claim: false
pointer:
  value: 0.19108
  axis: contest-CPU
  delta: UNMOVED
verdict_scope: >-
  Public BYU ACME lab manuals and cited primary estimator papers crosswalked
  to existing Pact source/research artifacts. No scorer, replay, dispatch,
  config mutation, archive selection, or promotion.
stores_consulted:
  - CLAUDE.md
  - AGENTS.md
  - docs/operating_manual_craft_handoff.md
  - .omx/research/t5_crucible/SPEC_v75_optimal_single_trunk_20260708.md
  - .omx/research/SPEC_v8_perclass_decomposition_20260708.md
  - reports/latest.md
  - .omx/state/lane_registry.json
  - .omx/state/subagent_progress.jsonl
  - .omx/state/master_gradient_anchors.jsonl
  - .omx/state/modal_call_id_ledger.jsonl
  - .omx/state/cost_band_posterior.jsonl
  - .omx/state/continual_learning_posterior.jsonl
  - .omx/state/probe_outcomes.jsonl
  - latest Codex findings/session memo and latest T3/design memo
  - four official public ACME lab-manual PDFs
  - primary Gap, Neyman-allocation, and Good-Turing papers
```

## Nodes

| Node | Verdict | Producer/evidence | Consumer | Gate / `FORMALIZATION_PENDING` |
|---|---|---|---|---|
| `A1_rank_custody` | `ADOPT` | ACME Vol. 1 Labs 4/7/11: pivoted Householder QR, direct SVD, conditioning/backward stability | `tools/build_r1b3_producers.py`; `src/tac/boundary_math/prereq_surfaces.py` | `$0`: frozen-head pivot/`R`/SVD/condition/reconstruction receipt. `FORMALIZATION_PENDING`: nonlinear pullback and receiver Jacobian. |
| `A2_pmp_conformance` | `ADOPT` | ACME Vol. 4 Labs 20/23/24: PMP forward/backward adjoint, bounded pointwise control, Riccati stability | `src/tac/witness_control/factorized_adjoint.py`; shadow controller; costate digest/introspection | `$0`: analytic LQ fixture, finite-difference Hamiltonian, projected-control and closed-loop-eigenvalue checks. `FORMALIZATION_PENDING`: typed curriculum Hamiltonian/control/terminal state. |
| `A3_kkt_diagnostics` | `ADOPT` | ACME Vol. 2 Labs 18/19: primal/dual residuals, complementarity, duality measure, fraction-to-boundary | `dual_solver_phase_2.py`; `pareto_dual.py`; `joint_seg_pose_rate.py`; #536/#549 | `$0`: synthetic/frozen curves plus brute-force finite oracle. `FORMALIZATION_PENDING`: measured three-axis curves and integer feasible-set custody. |
| `N1_gap` | `N-A` | Primary Gap paper; absent from ACME | proposed observer clustering only | `FORMALIZATION_PENDING`: clustering metric, null distribution, reference-replicate count, typed use of selected `k`. Use smallest-`k` one-SE rule, never global argmax. |
| `N2_neyman` | `N-A` | Neyman 1934; absent from ACME | proposed observer random background only | `FORMALIZATION_PENDING`: one scalar linear estimand, random strata, costs, integer/cap rounding. Never claim optimality for the hard tail or multi-facet maximum. |
| `N3_good_turing` | `N-A` | Good 1953; absent from ACME | none under current finite census/cohort | Refuse current stopping claim: exhaustive `n600` bootstrap plus deterministic hardest cohort is not an unseen-species exchangeable stream. |
| `B1_uint8_preimage` | `ALREADY-BETTER` | `bounded_uint8_resize_preimage_cell_feasibility_20260718.py`; r1b7 findings | #532/#586, r1b7 | Preserve exact Diophantine custody and scoped `NOT_FOUND_BUDGET`; individual receiver-composed prefixes remain open. |
| `B2_mc_finisher` | `ALREADY-BETTER` | exact `(1+1)` through-R gate plus `erm_margin_topk_v1` design | #396/#400 | Exact objective alone accepts; full-`K` fallback on rank failure. ACME uniform MC/MCMC is not terminal optimization. |
| `B3_directional_basis` | `ALREADY-BETTER` | genuine curvelet/shearlet artifacts plus governed legacy Fourier A/B control | #497/#502 | `FORMALIZATION_PENDING`: matched archive bytes and receiver survival. No Fourier-control retirement by proxy. |
| `B4_resize_exactness` | `ALREADY-BETTER` | `resize_full_kernel.py`; full-kernel build spec | #580 | Exact integer/range projector remains authority. ACME barycentric interpolation is diagnostics-only and cannot prove preimages/nullity. |

## Edges and admission logic

```text
A1_rank_custody
  -> existing segnet_head_rank4_linear_flipdist_v1 receipt
  -> sensitivity confidence only
  -X-> nonlinear pixel/camera flip authority until receiver Jacobian exists

A2_pmp_conformance
  -> existing hybrid_exact_factorized_costate_adjoint_v1 test surface
  -> shadow-controller advisory diagnostics
  -X-> live curriculum actuation until bounded typed control and terminal law exist

A3_kkt_diagnostics
  -> per-axis residual telemetry
  -> Pareto/bit-allocator advisory proposal
  -> brute-force finite oracle
  -X-> archive/pointer selection unless exact discrete through-R objective accepts

N1_gap + N2_neyman + N3_good_turing
  -> observer design review only
  -X-> “classically optimal panel budget” claim under the current mixed objectives

B1_uint8_preimage + B4_resize_exactness
  -> exact receiver/preimage custody

B2_mc_finisher
  -> exact top-k calibration and full-K fallback

B3_directional_basis
  -> matched-byte curvelet/shearlet versus legacy-control measurement
```

## Six-hook disposition

| Hook | Disposition |
|---|---|
| Sensitivity map | `A1` may attach a numerical-confidence receipt to the existing rank-4 sensitivity. No new sensitivity constant. |
| Pareto constraint | `A3` exposes separated continuous residuals; exact discrete feasibility and score remain binding. |
| Bit allocator | May consume `A3` only after measured/custody-complete curves exist. |
| Cathedral/autopilot | No dispatch hook activated; only `$0` local conformance probes are proposed. |
| Continual-learning posterior | No empirical anchor was produced; update is inadmissible. |
| Probe disambiguator | Rank receipt versus current custody, analytic LQ versus adjoint, and continuous KKT versus finite enumeration. Observer estimators remain unadmitted pending objective formalization. |

## Triality and MAIN boundary

- **DSL:** no new lever; any later estimator/control/tolerance must be typed.
- **DAG:** this file is the research-only feed.
- **Equations:** no equation registration; three conformance guards around existing laws only.
- **MAIN review required:** verify the public-corpus limitation, inspect the read-only sibling-observer assumption against the version actually proposed for landing, preserve exact-objective and legacy-control boundaries, and reject any score/promotion inference.

`0.19108 [contest-CPU]` **UNMOVED**.
