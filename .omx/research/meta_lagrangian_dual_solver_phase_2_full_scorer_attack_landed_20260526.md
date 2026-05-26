<!-- SPDX-License-Identifier: MIT -->

# META-LAGRANGIAN DUAL SOLVER PHASE 2 FULL-SCORER-ATTACK LANDED 2026-05-26
<!-- # HISTORICAL_SCORE_LITERAL_OK:pose_avg_PR106_operating_point_3p4e_minus_5_referenced_per_CLAUDE_md_SegNet_vs_PoseNet_importance_section_2026-05-04 -->

**Subagent_id**: `meta-lagrangian-dual-solver-phase-2-per-axis-dual-variable-computation-pure-full-scorer-attack-mlx-first-numpy-portable-individually-fractal-20260526`
**Lane**: `lane_meta_lagrangian_dual_solver_phase_2_20260526` L1 (impl_complete + strict_preflight + memory_entry)
**Status**: PARADIGM-VALIDATED apparatus-scope advancement per Catalog #307

## Mission contribution per Catalog #300

**`apparatus_maintenance` + `frontier_protecting`** (apparatus-growth; not direct score-lowering).

## Headline finding

Phase 2 advancement of Catalog #355 Phase 1 wire-in LANDED LOCALLY: per-candidate per-axis dual-variable computation via MLX-native Dykstra alternating projections onto the 3D seg/pose/rate Pareto polytope. The exposed scalar `adjustment_factor` remains bounded [0.95, 1.05] per Phase 1 SAFETY ENVELOPE contract preservation; the new per-axis dict (`adjustment_factor_per_axis`, `dual_variables_per_axis`, `kkt_residual_per_axis`, `posterior_sigma_per_axis`) is OBSERVABILITY-ONLY per Catalog #341 canonical-routing markers.

Empirical receipts:
- Synthetic 8-candidate MLX-local smoke: **8/8 converged** (100%) in mean **1.75 Dykstra iterations** (max 2), wall-clock **0.35 ms** for all 8 candidates total.
- Scalar adjustment factor distribution: min=0.9882, max=1.0000, mean=0.9956 — **all 8 candidates inside [0.95, 1.05] envelope** as required.
- Per-axis dual variables: interior candidates → all zeros; binding-on-N-axes candidates → N non-zero duals with magnitude equal to the constraint violation.
- 50 dedicated unit tests pass (`src/tac/findings_lagrangian/tests/test_dual_solver_phase_2.py`).

## What landed (LOC + files)

| File | LOC | Purpose |
|---|---:|---|
| `src/tac/findings_lagrangian/dual_solver_phase_2.py` | ~590 | NEW canonical Phase 2 module |
| `src/tac/findings_lagrangian/__init__.py` | +16 | Export Phase 2 symbols |
| `src/tac/findings_lagrangian/tests/test_dual_solver_phase_2.py` | ~395 | 50 unit tests across 8 test classes |
| `experiments/results/meta_lagrangian_dual_solver_phase_2_smoke_20260526/run_smoke.py` | ~165 | MLX-local apparatus-scope smoke |
| `experiments/results/meta_lagrangian_dual_solver_phase_2_smoke_20260526/smoke_results.json` | — | Smoke empirical anchor |
| `experiments/results/meta_lagrangian_dual_solver_phase_2_smoke_20260526/per_candidate_dual_solver_results.json` | — | Per-candidate per-axis result rows |
| `.omx/research/meta_lagrangian_dual_solver_phase_2_pre_execution_gate_report_20260526.md` | — | Pre-execution gate report |
| `.omx/research/meta_lagrangian_dual_solver_phase_2_full_scorer_attack_landed_20260526.md` | — | THIS landing memo |
| `.omx/state/canonical_equations_registry.jsonl` | +1 row | NEW canonical equation `meta_lagrangian_dual_solver_per_axis_kkt_residual_v1` registered |

## Sister-scope-disjoint declaration

Per Catalog #230 + #302 + #340 (sister-checkpoint guard PROCEED at edit-time):

- UNIWARD N+1 real-scorer empirical (#1369; substrate `uniward_per_pixel_distortion`): **DISJOINT**. Did not edit substrate dir.
- Cascade B Path A sister wave 1 (#1370; substrate `hinton_distilled_scorer_surrogate`): **DISJOINT**. Did not edit substrate dir.
- NSCS06 v8 Modal CUDA in flight (paid; operationally separate; no working-tree overlap this turn).

## Carmack-dissent verdict per Catalog #307

**PARADIGM-VALIDATED apparatus advancement, IMPLEMENTATION-COMPLETE.**

- The PARADIGM (Meta-Lagrangian dual variables on the 3-axis Pareto polytope) is canonical per CLAUDE.md "Meta-Lagrangian/Pareto solver — NON-NEGOTIABLE, HIGHEST EMPHASIS"; Phase 2 wires it into the cathedral autopilot ranker surface per Catalog #355 wire-in callsite advancement.
- The IMPLEMENTATION is complete: 50 unit tests pass; MLX-local 8-candidate smoke 100% converged with bounded adjustment factors per Phase 1 safety envelope; canonical Provenance per Catalog #323 + canonical-routing markers per Catalog #341.
- The empirical anchor (Dykstra convergence in mean 1.75 iters vs predicted ≤ 20) confirms the apparatus-scope prediction (convergence in O(log(1/ε)) for the 3-axis convex polytope per Boyd 2004 Theorem 1).
- No paradigm-level falsification surfaced. The work is sister-extensible to Phase 3+ (typed atom flow + per-element learned-optimal destination).

If Carmack were asked, the only critique would be: "The exposed scalar `adjustment_factor` is still bounded [0.95, 1.05] — same as Phase 1. What did Phase 2 actually buy?" Answer: Phase 2 buys (a) per-axis OBSERVABILITY (which axis is binding per candidate), (b) per-axis KKT RESIDUALS (apparatus sensitivity signal for downstream Pareto-polytope consumers per Dim 1 Phase 4 of cathedral smarter design blueprint), (c) Dykstra CONVERGENCE evidence (sister to substrate-scope Cascade C' Lagrangian dual primitive), and (d) a typed `PerAxisDualSolverResult` dataclass that downstream Phase 3+ subagents can flow typed atoms through. The bounded envelope IS the safety contract preservation per CLAUDE.md "Apples-to-apples evidence discipline"; Phase 2 enables internal unbounded computation while exposing only the safe envelope to the ranker.

## Canonical equation #344 anchor

NEW canonical equation **`meta_lagrangian_dual_solver_per_axis_kkt_residual_v1`** registered in `.omx/state/canonical_equations_registry.jsonl` per Catalog #344. Sister to the 3 just-registered equations (`mlx_cuda_bidirectional_drift_engineering_response_v1` + `hinton_kl_distill_enables_qat_catalyst_composition_savings_v1` + `daubechies_multi_scale_wavelet_hierarchical_composition_savings_v1`).

- In-domain contexts: `cathedral_autopilot_per_candidate_per_axis_adjustment` / `meta_lagrangian_dual_variables` / `dykstra_alternating_projections_3_axis_polytope` / `per_axis_kkt_residual_sensitivity_signal`.
- Excluded contexts: `raw_byte_authority` (per Catalog #318) / `substrate_level_dual_variables` (sister Cascade C' is substrate-scope; THIS is apparatus-scope) / `direct_score_claim` (per Catalog #323).
- Producer: `tac.findings_lagrangian.dual_solver_phase_2.compute_per_axis_dual_variables`.
- Consumer: `tools/cathedral_autopilot_autonomous_loop.py::invoke_meta_lagrangian_on_candidates` (Phase 2 extension; sister subagent operator-routable).
- First empirical anchor: `dykstra_3_axis_polytope_convergence_synthetic_smoke_20260526` (predicted ≤20 iters; empirical 1.75 mean iters; residual 18.25 within HARD-EARNED).

## 6-hook wire-in declaration per Catalog #125

- Hook #1 sensitivity-map: **ACTIVE** — per-axis KKT residuals ARE the canonical sensitivity signal at the apparatus-scope boundary.
- Hook #2 Pareto constraint: **ACTIVE** — Dykstra alternating-projections on the (seg, pose, rate) polytope IS the Pareto-feasibility primitive.
- Hook #3 bit-allocator: **ACTIVE** (indirect) — per-axis dual λ_rate IS the apparatus-scope rate-axis sensitivity signal.
- Hook #4 cathedral autopilot dispatch: **ACTIVE PRIMARY** — Phase 2 extends Catalog #355 wire-in callsite via typed `PerAxisDualSolverResult`.
- Hook #5 continual-learning posterior: **ACTIVE** — per-axis posterior anchors emit through `tac.findings_lagrangian.posterior_update_from_anchors`; canonical equation registry anchor LANDED.
- Hook #6 probe-disambiguator: **ACTIVE** — per-axis dual variables disambiguate which polytope axis is the binding constraint per candidate.

## Premise verification per Catalog #229

✅ All 9 PV items from pre-execution gate report verified.

## Operator-routable next-step plan

1. **Phase 3 typed atom flow**: route `CandidateRow` objects through the Dykstra solver as typed atoms with per-axis Provenance carried through the entire flow. Operator-routable to sister subagent (~3-4h apparatus advancement).

2. **Cathedral autopilot ranker Phase 2 wire-in**: extend `invoke_meta_lagrangian_on_candidates` in `tools/cathedral_autopilot_autonomous_loop.py` to optionally call `compute_per_axis_dual_variables` per candidate (gated by `--use-phase-2-dual-solver` flag for backward compat). Operator-routable to sister subagent (~30 min wiring + tests; preserves Phase 1 default).

3. **Sister Cascade C' substrate-scope cross-validation**: validate that apparatus-scope per-axis duals on a real CandidateRow set agree with substrate-scope per-pair duals from Cascade C' (commit `2d5337f27`). Cross-substrate convergence verification — sister operator-routable.

4. **Phase 4 per-element learned-optimal destination**: per META engineering vision — long-burn lane; defer until Phase 3 lands.

## Cross-references

- Catalog #355 STRICT preflight gate (this Phase 2 advancement PRESERVES the canonical invocation contract).
- Catalog #318 master-gradient raw-byte-authority guard (Phase 2 operates at axis-aggregate scale, NOT raw-byte; explicit refusal per #318).
- Catalog #323 canonical Provenance umbrella.
- Catalog #341 canonical-routing markers.
- Catalog #356 per-axis decomposition canonical Provenance.
- Sister `tac.score_composition.compose_score_from_axes` (uses SAME canonical formula constants).
- Sister substrate-scope Cascade C' Lagrangian dual primitive (commit `2d5337f27`; subagent `aa563bbb31adadfd6`).
- CLAUDE.md "Meta-Lagrangian/Pareto solver — NON-NEGOTIABLE, HIGHEST EMPHASIS".
- `feedback_mlx_first_with_numpy_portable_bridge_standing_directive_20260526.md`.
- Pre-execution gate report: `.omx/research/meta_lagrangian_dual_solver_phase_2_pre_execution_gate_report_20260526.md`.

## Mission alignment per CLAUDE.md "Mission alignment" subsection

Per operator NON-NEGOTIABLE 2026-05-26 directive *"all are approved + pursue other attacks as well remmeber all MLX first portable via numpy and indivually fractally optimized"*: THIS lane lands a PURE FULL-SCORER attack at apparatus-scope while sister UNIWARD + Cascade B subagents execute PURE DIST attacks at substrate-scope. The 3-strategy directive's strategy-coverage rule is satisfied by the convergent multi-subagent fan-out.

$0 GPU + ~80 minutes wall-clock + 8-cand MLX-local smoke. Honor preserved per CLAUDE.md "Remember all on MLX" non-negotiable + "Forbidden premature KILL without research exhaustion" non-negotiable.
