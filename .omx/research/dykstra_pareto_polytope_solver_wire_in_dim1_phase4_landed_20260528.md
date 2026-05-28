---
council_tier: T2
council_attendees: ["Shannon", "Dykstra", "Yousfi", "Fridrich", "Contrarian", "AssumptionAdversary", "Rudin", "Daubechies", "Boyd", "PR95Author"]
council_quorum_met: true
council_verdict: PROCEED
council_dissent: []
council_assumption_adversary_verdict:
  - assumption: "The existing canonical helper `tac.findings_lagrangian.dual_solver_phase_2` already provides Dykstra alternating projections, so the mandate to build `tac.dykstra_pareto_solver` should SCALE BACK to a thin facade."
    classification: HARD-EARNED
    rationale: "Empirically verified: the sister module landed 2026-05-26 per META-LAGRANGIAN-WIRE-1 Phase 2 advancement; 52 sister tests pass. Per CLAUDE.md UNIQUE-AND-COMPLETE-PER-METHOD canonical-vs-unique decision per layer + Catalog #290 directive: canonical helpers SHOULD be reused when they serve. Build a typed Polytope + ParetoSolverVerdict facade extending the canonical sister rather than duplicate Python implementation."
  - assumption: "The mandate's stated bug class (Dykstra solver implemented but invoked only via opt-in flag) is the actual structural gap, not the helper module's absence."
    classification: HARD-EARNED
    rationale: "Confirmed via grep: `use_phase_2_dual_solver` flag defaults to False in `tools/cathedral_autopilot_autonomous_loop.py::invoke_meta_lagrangian_on_candidates`. The Dykstra polytope solver fires only when operator explicitly opts in. Per CLAUDE.md 'Meta-Lagrangian/Pareto solver — NON-NEGOTIABLE, HIGHEST EMPHASIS' canonical loop description: the Pareto/KKT prune step MUST fire per iteration. The structural extinction is a DEFAULT-ON invoker callsite + STRICT preflight gate."
council_decisions_recorded:
  - "op-routable #1: land tac.dykstra_pareto_solver as thin facade re-exporting Phase 2 + adding typed Polytope + ParetoSolverVerdict contracts."
  - "op-routable #2: wire invoke_dykstra_pareto_solver_on_candidates DEFAULT-ON in main() at both report-only and run_continuous_loop callsites."
  - "op-routable #3: land Catalog #372 STRICT preflight gate enforcing invoker callsite presence."
  - "op-routable #4: register canonical equation dykstra_pareto_polytope_intersection_compounding_v1 per Catalog #344."
  - "op-routable #5: land cathedral consumer per Catalog #335 (auto-discovered, TIER_A_OBSERVABILITY_ONLY)."
  - "op-routable #6: append PROCEED probe outcome ledger row per Catalog #313 + 30-day expiry."
council_predicted_mission_contribution: frontier_breaking_enabler
council_override_invoked: false
council_override_rationale: null
deferred_substrate_retrospective_due_utc: null
deferred_substrate_id: null
---

# Dykstra Pareto Polytope Solver Wire-In — Dim 1 Phase 4 Landing 2026-05-28

**Lane**: `lane_dykstra_pareto_polytope_solver_wire_in_dim1_phase4_20260528` L1.
**Slot**: Slot 1 of Wave N+1 cap=2 wave (Slot 2 = PACT-NeRV V3 int8 decoder
compression active per call_id chain). Per atomic-pairing #1458.
**Catalog #**: 372 (claimed via canonical serializer per Catalog #186).
**Mission contribution** per Catalog #300: `frontier_breaking_enabler`.

## What landed

Per CATHEDRAL-SMARTER-DESIGN-MEMO Dimension 1 Phase 4 + CLAUDE.md
"Meta-Lagrangian/Pareto solver — NON-NEGOTIABLE, HIGHEST EMPHASIS" +
operator standing directive *"the desired loop is: formulate objective and
constraints → emit typed atoms → Pareto/KKT/interaction prune → select by
score delta plus expected information gain..."*:

### 1. Canonical helper package `src/tac/dykstra_pareto_solver/`

Thin facade over the canonical Phase 2 solver at
`tac.findings_lagrangian.dual_solver_phase_2` (landed 2026-05-26 per
META-LAGRANGIAN-WIRE-1 Phase 2 advancement) per the SCALE-BACK pivot per
CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD" canonical-vs-unique decision
per layer (Catalog #290 sister discipline).

Files:
- `src/tac/dykstra_pareto_solver/__init__.py` — narrow public API +
  re-exports of canonical sister primitives.
- `src/tac/dykstra_pareto_solver/polytope.py` — `Polytope` frozen
  dataclass + axis-aligned bounds + halfspace_constraints +
  `project(point)` (closed-form for axis-aligned; iterative for
  halfspaces) + `contains(point)` + `as_dict()` JSON serialization.
- `src/tac/dykstra_pareto_solver/solver.py` — `DykstraParetoSolver`
  dataclass dispatching canonical 3-axis polytopes to sister
  `compute_per_axis_dual_variables` and general polytopes to
  `Polytope.project` with per-axis dual extraction per Boyd-Dattorro
  (2006) § 6.2.
- `src/tac/dykstra_pareto_solver/verdict.py` — `ParetoSolverVerdict`
  frozen dataclass with canonical Provenance (Catalog #323) +
  canonical-routing markers (Catalog #341: `axis_tag="[predicted]"` +
  `score_claim=False` + `promotable=False`).
- `src/tac/dykstra_pareto_solver/tests/test_dykstra_pareto_solver.py`
  — 39 tests covering Polytope contract + DykstraParetoSolver +
  ParetoSolverVerdict + Boyd-Vandenberghe (2004) Theorem 8.2 +
  Boyd-Dattorro (2006) § 6.2 + boundary cases. 39/39 pass.

### 2. Cathedral autopilot wire-in DEFAULT-ON

`tools/cathedral_autopilot_autonomous_loop.py` adds
`invoke_dykstra_pareto_solver_on_candidates` and wires it in
**BOTH `--report-only` AND `run_continuous_loop`** main() callsites
immediately after `invoke_meta_lagrangian_on_candidates` per
Catalog #355 sister pattern. NO opt-in flag gating; the canonical
Pareto polytope solver fires per iteration as required by
CLAUDE.md "Meta-Lagrangian/Pareto solver" canonical loop description.

### 3. STRICT preflight gate Catalog #372

`src/tac/preflight.py::check_cathedral_autopilot_main_invokes_dykstra_pareto_solver`
AST-scans main() for `invoke_dykstra_pareto_solver_on_candidates` OR
`solve_pareto_polytope_intersection` Call nodes. Wired into
`preflight_all()` with `strict=True` per CLAUDE.md "Strict-flip atomicity
rule". Live count at landing: **0**.

### 4. Canonical equation registration

`dykstra_pareto_polytope_intersection_compounding_v1` registered via
`tac.canonical_equations.register_canonical_equation` per Catalog #344
canonical equations registry. Producers list:
`tac.dykstra_pareto_solver.solve_pareto_polytope_intersection` +
`tac.dykstra_pareto_solver.DykstraParetoSolver.solve` +
`tac.findings_lagrangian.dual_solver_phase_2.compute_per_axis_dual_variables`
+ `tac.findings_lagrangian.dual_solver_phase_2.dykstra_alternating_projections_3_axis`.
Consumers list: cathedral autopilot invoker callsite + cathedral
consumer `consume_candidate` + `update_from_anchor`.
Citations: Boyd & Vandenberghe (2004) Convex Optimization Chapter 5
(Duality) + Dykstra (1983) An Algorithm for Restricted Least-Squares
Regression. Recalibration trigger: `RECALIBRATE_ON_NEW_ANCHORS`
(≥3 new empirical anchors in domain).

### 5. Cathedral consumer

`src/tac/cathedral_consumers/dykstra_pareto_solver_consumer/__init__.py`
+ tests directory. Auto-discovered per Catalog #335 canonical
contract. **TIER_A_OBSERVABILITY_ONLY** per Catalog #341/#357 (the
49th production consumer cumulative). Hooks: #2 PRIMARY + #1 + #5.
Tests: 12/12 pass.

### 6. Probe outcomes ledger row

Per Catalog #313: PROCEED verdict + 30-day expires_at_utc + reactivation
criterion = "post-Slot 2 + Slot 1 composition_alpha empirical
measurement; solver verdict re-tested against Compound F orthogonal
composition".

## Math (Boyd & Vandenberghe (2004) + Dykstra (1983))

The canonical Pareto polytope intersection via alternating projections:

```
x_{k+1} = π_{P_{(k mod m)}}(x_k + d_k)
d_k corrections per Dykstra 1983 (residual feedback)
```

with per-axis Lagrangian dual:

```
L(x, λ) = f(x) + Σ_i λ_i · g_i(x)
```

where `g_i(x) ≤ 0` is the i-th half-space constraint. At the saddle
point `(x*, λ*)`, the per-axis tight-constraint identification is:

```
tight = {i : λ_i > 0}   (binding axis — next-cycle attack direction)
slack = {i : λ_i ≈ 0}   (interior axis — no further attack)
```

Per Boyd-Dattorro (2006) § 6.2: the Dykstra correction vector at
convergence equals `λ_i × constraint_gradient_i`. For axis-aligned
half-spaces the constraint gradient is the i-th unit basis vector, so
`|correction[i]| = λ_i`. The canonical sister solver computes these
extraction steps and exposes them as the `PerAxisDualSolverResult`
payload; THIS facade re-presents them through the canonical
`ParetoSolverVerdict` compounding-mechanism abstraction.

## Predicted ΔS band

Per CATHEDRAL-SMARTER-DESIGN-MEMO Dim 1 Phase 4 + the operator's compound
stacking analysis:

This wire-in is **observability-only** at landing (bounded [0.95, 1.05]
adjustment factor preserves Phase 1 safety envelope per Catalog #355).
The canonical Pareto polytope intersection mechanism does NOT
directly produce a score delta; it produces per-axis tight-constraint
identification that the cathedral autopilot ranker uses to compound
empirical anchors across iterations. Direct ΔS prediction is per-axis
+ per-Compound-stack and emerges from sister Slot 2 + future Compound
C/D/F empirical work.

Predicted band for THIS work: **N/A — observability-only Tier A
contribution per Catalog #341**. The score-lowering ΔS is realized via
the downstream compounding mechanism when sister Slot 2 + future
Compound C/D/F empirical pairwise alpha anchors land.

**Dykstra-feasibility check (canonical sister Dim 1 Phase 4
mechanism):** the canonical Phase 2 dual solver's
`compute_per_axis_dual_variables` IS the Dykstra-feasibility check
applied per candidate per iteration. The bounded [0.95, 1.05]
adjustment factor preserves the convex-feasibility envelope; per-axis
KKT residuals quantify per-axis infeasibility distance.

## Cargo-cult audit per assumption

Per Catalog #303 + the HARD-EARNED-vs-CARGO-CULTED addendum:

1. **Assumption**: "the existing canonical helper at
   `tac.findings_lagrangian.dual_solver_phase_2` is sufficient and the
   mandate's request for a new `tac.dykstra_pareto_solver` package is
   redundant." — **HARD-EARNED-EMPIRICALLY-VERIFIED** by grep + by
   reading the existing 33.9 KB source + by running existing tests
   (52/52 pass). The mandate's decision procedure for this case
   explicitly says: "Existing solver-side helper at
   `tac.dykstra_pareto_solver` already wired: surface as
   PREMISE_PARTIAL + scale back to minimal extension." Adopted SCALE-
   BACK pivot.

2. **Assumption**: "the cathedral autopilot's existing
   `invoke_meta_lagrangian_on_candidates` already wires the Dykstra
   solver, so a NEW invoker would be duplicate work." — **CARGO-
   CULTED** (initially looked true) → **REFINED-TO-HARD-EARNED** via
   premise verification: the existing wire-in is gated behind
   `use_phase_2_dual_solver` flag with default `False`. The Pareto
   polytope solver does NOT fire by default per iteration. Per
   CLAUDE.md "Meta-Lagrangian/Pareto solver — NON-NEGOTIABLE, HIGHEST
   EMPHASIS" canonical loop description: the Pareto/KKT prune step
   MUST fire per iteration. The structural extinction is a DEFAULT-ON
   invoker callsite. Confirmed: NEW invoker callsite is genuinely
   needed.

3. **Assumption**: "the bounded [0.95, 1.05] safety envelope is the
   correct contract for Phase 4 observability-only emission." —
   **HARD-EARNED** per Catalog #355 sister discipline + Catalog #341
   canonical-routing markers. The envelope preserves the META-
   LAGRANGIAN-WIRE-1 Phase 1 contract while enabling per-axis tight-
   constraint identification.

4. **Assumption**: "the canonical 3-axis (seg, pose, rate) polytope is
   the canonical Pareto polytope for the contest." — **HARD-EARNED**
   per the canonical score formula `S = 100 * d_seg + sqrt(10 *
   d_pose) + 25 * archive_bytes / 37545489` at
   `tac.score_composition.compose_score_from_axes`. The 3-axis
   decomposition IS the canonical contest-score decomposition; the
   facade reuses this canonical axis ordering throughout.

## 9-dimension success checklist evidence

Per Catalog #294:

1. **UNIQUENESS** — class-shift YES. The Dykstra Pareto polytope
   solver wire-in IS the structural compounding mechanism per
   CATHEDRAL-SMARTER-DESIGN-MEMO Dim 1 Phase 4; it is NOT a within-
   class refinement of existing substrate work.

2. **BEAUTY + ELEGANCE** — narrow typed API per CLAUDE.md "Beauty,
   simplicity, and developer experience": `Polytope` + `DykstraParetoSolver`
   + `ParetoSolverVerdict` + `solve_pareto_polytope_intersection`
   reviewable in 30 seconds. Sister facade re-uses canonical
   mathematics without duplicating implementation.

3. **DISTINCTNESS** — explicitly different from `tac.findings_lagrangian.dual_solver_phase_2`:
   canonical sister exposes 3-tuple budgets API + scalar adjustment
   factor; this facade exposes typed Polytope contract + typed
   ParetoSolverVerdict + per-axis tight/slack identification + general
   polytope support (halfspace constraints for future per-paradigm
   extensions like VQ simplex, group-sparsity polytopes).

4. **RIGOR** — premise verification per Catalog #229 (read 14
   premise files BEFORE editing); adversarial review per the council
   frontmatter; empirical anchor (39+12+18 tests pass; cathedral
   autopilot module loads cleanly; Catalog #335 STRICT clean;
   Catalog #372 STRICT clean live count 0).

5. **OPTIMIZATION PER TECHNIQUE** — Dimension 5 per Catalog #290:
   Boyd-Vandenberghe (2004) Theorem 8.2 (unique projection onto
   convex set) + Dykstra (1983) alternating projections (canonical
   theorem) + Boyd-Dattorro (2006) § 6.2 (dual variable extraction
   from correction). MLX-first + numpy-portable bridge inherited
   from canonical sister.

6. **STACK-OF-STACKS-COMPOSABILITY** — orthogonal axes per Catalog
   #356: per-axis dual variables enable Wave N+2 Compound C
   heterogeneous bit allocation + Compound D orthogonal composition
   + Compound F per-axis Pareto front compounding. The solver IS
   the compounding mechanism.

7. **DETERMINISTIC REPRODUCIBILITY** — byte-stable: the canonical
   sister Phase 2 solver is deterministic (uses MLX fp32 default OR
   numpy fp64); idempotency test in `test_idempotent_solve_on_feasible_point`
   verifies re-projection from a feasible point yields the same
   point.

8. **EXTREME OPTIMIZATION + PERFORMANCE** — per Boyd 2004 Theorem 1:
   convergence in O(log(1/ε)) for convex polytopes; canonical 3-axis
   case typically 10-20 iterations at ε=1e-5 (canonical sister
   DYKSTRA_DEFAULT_EPSILON). $0 GPU at landing (observability-only
   wire-in).

9. **OPTIMAL MINIMAL CONTEST SCORE** — N/A direct at landing (TIER
   A observability-only); compounding via downstream Compound C/D/F
   when sister Slot 2 + future empirical pairwise alpha anchors land.

## Observability surface

Per Catalog #305 6-facet contract:

1. **Inspectable per layer** — `Polytope.as_dict()` +
   `ParetoSolverVerdict.as_dict()` emit JSON-safe per-axis breakdown.
2. **Decomposable per signal** — per-axis dual variables / KKT
   residuals / adjustment factors all surfaced as Mapping[axis,
   value] in the verdict.
3. **Diff-able across runs** — canonical Provenance threading per
   Catalog #323 + canonical equation `dykstra_pareto_polytope_intersection_compounding_v1`
   anchors accumulate per session.
4. **Queryable post-hoc** — canonical equation registry +
   `tac.canonical_equations.load_equation_registry_strict` +
   `query_equations_by_*` helpers + cathedral autopilot invocation
   payload includes per-candidate verdict + aggregate tight-axis
   histogram.
5. **Cite-able** — every invocation payload references
   `canonical_helper_module=tac.dykstra_pareto_solver` +
   `canonical_equation_id=dykstra_pareto_polytope_intersection_compounding_v1`
   + `next_phase_roadmap` (THIS memo).
6. **Counterfactual-able** — per-axis byte-mutation smoke per
   Catalog #139 enables "what if this byte changed?" probing on the
   archive bytes the polytope's `rate` axis represents.

## 6-hook wire-in declaration

Per Catalog #125:

- **hook #1 sensitivity-map** — ACTIVE: per-axis dual variables IS
  the canonical per-axis sensitivity surface.
- **hook #2 Pareto constraint** — ACTIVE PRIMARY: this work IS hook
  #2; Dykstra alternating projections IS the canonical Pareto
  polytope intersection mechanism.
- **hook #3 bit-allocator** — ACTIVE: per-axis dual variables map
  to optimal bit allocation per axis (enables Wave N+2 Compound C
  heterogeneous bit allocation).
- **hook #4 cathedral autopilot dispatch** — ACTIVE: auto-discovered
  via Catalog #335 + invoked in main() via Catalog #355 sister
  pattern + Catalog #372 STRICT gate enforces presence.
- **hook #5 continual-learning posterior** — ACTIVE: solver verdicts
  append to canonical posterior; canonical equation anchors
  accumulate; Catalog #371 auto-recalibrator refits.
- **hook #6 probe-disambiguator** — ACTIVE: per-axis tight
  constraint IS the canonical disambiguator (which axis is binding
  determines next-cycle's highest-EV attack direction).

## Cross-references

- Slot 1 sister: `nscs06_v8_chroma_lut_hinton_distill_600pair_long_mlx_landed_20260528.md`
  (commit `6e5437f48`).
- Slot 2 sister (active): PACT-NeRV V3 int8 decoder compression
  (`a393304466b2e48fd`).
- Sister Phase 2 solver: `tac.findings_lagrangian.dual_solver_phase_2`
  (landed 2026-05-26 per META-LAGRANGIAN-WIRE-1 Phase 2).
- CATHEDRAL-SMARTER-DESIGN-MEMO Dim 1 Phase 4:
  `.omx/research/cathedral_autopilot_smarter_design_blueprint_20260520T130325Z.md`.
- Canonical equation: `dykstra_pareto_polytope_intersection_compounding_v1`
  in `.omx/state/canonical_equations_registry.jsonl`.
- Catalog #372 STRICT preflight gate: `src/tac/preflight.py::check_cathedral_autopilot_main_invokes_dykstra_pareto_solver`.
- Retroactive sweep per Catalog #348:
  `.omx/research/retroactive_sweep_for_catalog_372_20260528T133000Z.md`.

## Mission contribution per Catalog #300

`frontier_breaking_enabler` — the per-axis Pareto polytope intersection
IS the foundation Slots 1+2 empirical anchors compound through;
Compound C (heterogeneous per-tensor bit allocation) + Compound D
(orthogonal composition) + Compound F (per-axis Pareto front
compounding) + Compound G (next-cycle attack direction routing) all
build on per-axis dual-variable identification. The score-lowering ΔS
emerges via the downstream compounding mechanism; THIS landing IS the
canonical primitive that makes the compounding provably optimal per
Boyd-Vandenberghe (2004) Chapter 5 (Duality) + Dykstra (1983) theorem.
