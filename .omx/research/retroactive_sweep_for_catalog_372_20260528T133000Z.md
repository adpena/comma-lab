# Retroactive sweep for Catalog #372 — Dykstra Pareto polytope solver invoker

Per Catalog #348 EVENT-DRIVEN RETROACTIVE VERDICT-TAINT SWEEP discipline.
Catalog #372 lands 2026-05-28T13:30:00Z as a new STRICT preflight gate enforcing
that `tools/cathedral_autopilot_autonomous_loop.py::main()` invokes the
canonical Dykstra Pareto polytope solver.

## Bug-class symptom signature

The bug class Catalog #372 prevents:

> A canonical Pareto polytope solver implementation exists at
> `tac.findings_lagrangian.dual_solver_phase_2` (landed 2026-05-26 per
> META-LAGRANGIAN-WIRE-1 Phase 2 advancement) BUT is invoked CONDITIONALLY via
> the `--use-phase-2-dual-solver` opt-in flag with default=False. The Pareto/KKT
> prune step canonical to CLAUDE.md "Meta-Lagrangian/Pareto solver — NON-
> NEGOTIABLE, HIGHEST EMPHASIS" loop description does NOT fire by default per
> cathedral autopilot iteration.

Symptom signature in source text:
- Existence of helper module `tac.findings_lagrangian.dual_solver_phase_2`
  with `dykstra_alternating_projections_3_axis` + `compute_per_axis_dual_variables`
  callable;
- ZERO Calls in `tools/cathedral_autopilot_autonomous_loop.py::main()` body
  to `invoke_dykstra_pareto_solver_on_candidates` OR
  `solve_pareto_polytope_intersection` (the canonical default-on invokers);
- OPTIONAL invocation via `use_phase_2_dual_solver=args.use_phase_2_dual_solver`
  kwarg threaded through `invoke_meta_lagrangian_on_candidates` defaults
  to `False`.

## Pre-fix window

2026-05-26 (Phase 2 sister landing) through 2026-05-28T13:30:00Z (Phase 4 wire-in
landing). Approximately 36 hours where the canonical Dykstra Pareto polytope solver
was implemented but DEFAULT-OFF per iteration.

Specifically Phase 1 (2026-05-20 META-LAGRANGIAN-WIRE-1 canonical invocation at
commit `c8d51ebb`) wired the meta-Lagrangian scalar Lagrangian invocation; Phase
2 (2026-05-26) added the Dykstra alternating-projections sub-helper but gated
it behind an opt-in flag for safety during the Phase 2 advancement. Phase 4
(THIS landing) flips the default ON and adds the canonical typed Polytope +
ParetoSolverVerdict contracts + STRICT preflight enforcement.

## Historical-KILL/DEFER/FALSIFY search

Searched the canonical `.omx/state/probe_outcomes.jsonl` ledger and recent
`.omx/research/*KILL*.md` / `*falsified*.md` / `*deferred*.md` memos for any
Dykstra Pareto polytope solver or per-axis dual variable verdict that may
need re-evaluation given the Catalog #372 invoker now fires by default.

**Verdict**: zero historical verdicts depend on the Dykstra solver being
opt-in. The opt-in default was a Phase 2 safety-rail introduced 2026-05-26
without empirical anchors. No historical KILL / DEFER / FALSIFY verdict
depends on the opt-in behavior.

## Per-finding RE-EVAL-priority assignment

Per Catalog #348 contract:

| Finding | RE-EVAL Priority | Rationale |
|---------|------------------|-----------|
| Phase 2 META-LAGRANGIAN-WIRE-1 opt-in default | RE-EVAL NOT NEEDED | The opt-in default was a Phase 2 safety-rail; Phase 4 lands DEFAULT-ON per the canonical CLAUDE.md "Meta-Lagrangian/Pareto solver" loop description. No empirical anchor depends on the opt-in behavior. |
| Existing cathedral autopilot ranker outputs | RE-EVAL NOT NEEDED | The bounded [0.95, 1.05] adjustment factor preserves the Phase 1 safety envelope per Catalog #355 sister discipline. Pre-Phase-4 ranker outputs are exactly correct; Phase 4 ADDS per-axis observability without mutating the predicted_score_delta computation. |
| Sister Slot 2 PACT-NeRV V3 int8 decoder compression dispatch | RE-EVAL NOT NEEDED | Slot 2 work is in flight at landing time and is disjoint from Catalog #372 (touches V3 files; Catalog #372 touches the canonical autopilot ranker surface). Per the mandate's explicit `DO NOT touch PACT-NeRV-V3 files` directive. |
| Future Compound C/D/F/G empirical work | RE-EVAL PRIORITY=HIGH (NEW WORK) | THIS is the compounding-mechanism foundation per CATHEDRAL-SMARTER-DESIGN-MEMO Dim 1 Phase 4. Future empirical work compounds through the per-axis Pareto polytope intersection mechanism. The canonical equation `dykstra_pareto_polytope_intersection_compounding_v1` accumulates anchors as Compound C/D/F/G land. |

## Sister-extinction architecture

Per Catalog #299 quota brake decision: NEW gate (NOT scope extension of
Catalog #355) because Catalog #355's surface is meta-Lagrangian scalar
invocation (Phase 1) while Catalog #372's surface is the canonical
Dykstra Pareto polytope solver invocation (Phase 4 advancement). Sister
gates at orthogonal sub-surfaces:

- Catalog #355 — META-LAGRANGIAN-WIRE-1 Phase 1 invoker (scalar
  Lagrangian).
- Catalog #372 — DYKSTRA-PARETO-SOLVER-WIRE-IN Phase 4 invoker (per-axis
  polytope intersection).
- Catalog #336 — cathedral consumer discovery invoker.
- Catalog #337 — master-gradient rerank invoker.

Together they extinct the orphan-canonical-helper-without-invoker bug class
at 4 orthogonal cathedral autopilot sub-surfaces.

## Current count: 372

Well under Catalog #299 quota brake at 400.

## 6-hook wire-in declaration

Per Catalog #125:

- hook #1 sensitivity-map = ACTIVE (per-axis duals are canonical sensitivity)
- hook #2 Pareto constraint = ACTIVE PRIMARY (Dykstra IS hook #2)
- hook #3 bit-allocator = ACTIVE (per-axis duals → bit allocation)
- hook #4 cathedral autopilot dispatch = ACTIVE
- hook #5 continual-learning posterior = ACTIVE (canonical equation anchors)
- hook #6 probe-disambiguator = ACTIVE (per-axis tight axis identification)

## Cross-references

- Canonical equation: `dykstra_pareto_polytope_intersection_compounding_v1`
- Landing memo: `.omx/research/dykstra_pareto_polytope_solver_wire_in_dim1_phase4_landed_20260528.md`
- CATHEDRAL-SMARTER-DESIGN-MEMO Dim 1 Phase 4:
  `.omx/research/cathedral_autopilot_smarter_design_blueprint_20260520T130325Z.md`
- Sister Phase 2 solver: `tac.findings_lagrangian.dual_solver_phase_2`
