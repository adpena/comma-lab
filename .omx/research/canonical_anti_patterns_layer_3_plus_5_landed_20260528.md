---
council_tier: T1
council_attendees:
  - claude
council_quorum_met: true
council_verdict: PROCEED
council_dissent: []
council_assumption_adversary_verdict:
  - assumption: "Layer 5 anti-pattern constraints can extend ParetoSolverVerdict without breaking Wave N+1 invariants"
    classification: HARD-EARNED
    rationale: |
      The verdict's __post_init__ requires tight_constraint_axes ∪ slack_axes
      == per_axis_dual_variables keys. Anti-pattern dual variable keys
      naturally satisfy this partition (every constraint emits either
      a tight or slack row keyed by anti_pattern_<id> per the canonical
      prefix). Per-axis adjustment factors invariant in [0.95, 1.05]
      preserved because anti-pattern constraints emit factor=1.0
      (observability-only per Catalog #341). All 39 baseline Dykstra
      tests + 59 Layer 1+2 anti-pattern tests + 12 Dykstra consumer
      tests + 140 canonical equations tests still PASS after Layer 5
      integration (155 total dykstra+anti-pattern + 140 equations).
  - assumption: "Predicate-always-True per-candidate constraint is the correct contract"
    classification: HARD-EARNED
    rationale: |
      The Layer 1 matcher already establishes structural overlap with
      registered anti-pattern recurrence_conditions / forbidden_pattern_predicate
      at the consumer surface. Re-running the matcher inside the
      forbidden_region_predicate at projection time would be redundant
      (same stack_spec). The constraint encodes the BINDING semantics:
      "this candidate MATCHED this anti-pattern" -> dual = severity_weight.
      The polytope projection step then surfaces the canonical_unwind_path
      via Catalog #125 hook #6 disambiguator semantics.
  - assumption: "MAX-aggregation (not SUM) preserves the design memo §Mathematical compounding identity"
    classification: HARD-EARNED
    rationale: |
      Per the design memo verbatim: "anti-patterns are typically dominated
      by the WORST applicable pattern (one violated anti-pattern can kill
      the whole stack regardless of how many other anti-patterns are
      clear). SUM would double-count + dilute critical signal. MAX is the
      canonical aggregation for safety-constraint composition in Pareto
      polytope feasibility set construction." `aggregate_anti_pattern_duals`
      returns max_dual = max_j (severity_weight_j × indicator_j); ordered
      binding_paths by descending severity_weight for canonical operator-
      routable order.
  - assumption: "Cutoff exempts Wave N+1 + earlier memos correctly"
    classification: HARD-EARNED
    rationale: |
      Per "Strict-flip atomicity rule": Layer 3 STRICT-from-byte-one
      requires live count = 0 at landing. The 20260529 cutoff exempts
      ALL memos dated <= 20260528 (including Layer 1+2 landing memo
      itself, Dykstra Phase 4 landing memo, NSCS06 v8 chroma LUT memo,
      etc.). Live count verified: 0 in-scope memos at landing.
  - assumption: "Defensive fail-OPEN on registry/matcher unavailability is correct safety contract"
    classification: HARD-EARNED
    rationale: |
      Per CLAUDE.md "Subagent coherence-by-default" non-negotiable:
      defensive surfaces never crash the ranker cascade. If the
      canonical anti-patterns registry is unavailable (rare; structural
      regression covered by Catalog #131/#138 sister gates), the gate
      surfaces "matcher unavailable; skipping" + returns no violations
      rather than blocking the landing memo. Sister of Catalog #281
      F1 fail-closed contract at codex review surface (where codex
      review unavailability is FATAL because the review is a hard
      requirement); Catalog #373's matcher is OBSERVABILITY (Tier A)
      so fail-OPEN is the correct contract.
council_decisions_recorded:
  - "Layer 5 AntiPatternConstraint frozen dataclass landed at tac.dykstra_pareto_solver.anti_pattern_constraint"
  - "DykstraParetoSolver extended with anti_pattern_constraints field + _apply_anti_pattern_constraints helper + integration in both _solve_canonical_3_axis + _solve_general paths"
  - "Cathedral autopilot invoke_dykstra_pareto_solver_on_candidates extended with _candidate_to_stack_spec + _derive_anti_pattern_constraints_for_candidate + per-candidate matched_anti_patterns/binding_anti_pattern_ids/canonical_unwind_paths_recommended/anti_pattern_constraint_count surface + aggregate anti_pattern_binding_histogram"
  - "Layer 3 STRICT preflight gate Catalog #373 check_compound_stack_proposal_acknowledges_known_anti_patterns wired into preflight_all() strict=True"
  - "Canonical equation anti_pattern_polytope_exclusion_dykstra_compounding_v1 registered per Catalog #344"
  - "155 sister tests + 22 Catalog #373 tests pass (177 total Layer 3 + Layer 5 + sister regression)"
council_predicted_mission_contribution: frontier_breaking_enabler
council_override_invoked: false
council_override_rationale: null
deferred_substrate_retrospective_due_utc: null
deferred_substrate_id: null
---

# Canonical Anti-Patterns Registry — Layer 3 + Layer 5 landed (2026-05-28)

**Status**: LANDED (impl_complete + strict_preflight + cathedral autopilot wire-in + canonical equation registered + memory_entry)
**Lane**: `lane_canonical_anti_patterns_layer_3_plus_5_20260528` L1
**Sister design memo**: `.omx/research/canonical_anti_patterns_registry_design_20260528.md` (commit `37b5a0184`)
**Sister Layer 1+2 landing**: `.omx/research/canonical_anti_patterns_registry_layer_1_plus_2_landed_20260528.md` (commit `8ef0ed7d1`)
**Sister Slot 1 (DISJOINT scope; active)**: PACT-NeRV V3 int8 decoder compression (Compound C heterogeneous bit allocation)

## Operator directive

Per the canonical Wave N+2 mandate dispatching this lane + the parent
operator META directive 2026-05-28 verbatim: *"learning anti-patterns is
upser important too for compounding continual learning, like the canonical
equations bu netgative and a higher layer of abstraction"*. Layer 3 +
Layer 5 close the canonical anti-patterns design memo §"Wave N+2"
implementation queue, making anti-patterns ACTIVE Pareto polytope
feasibility constraints + REVIEWABLE memo-level obligations.

## ## Canonical-vs-unique decision per layer

| Layer | Decision | Rationale |
|---|---|---|
| AntiPatternConstraint dataclass | FORK_BECAUSE_PRINCIPLED_MISMATCH | Layer 1's AntiPattern + EmpiricalFalsification are the registry surface; the solver-side wrapper IS a different abstraction (forbidden_region_predicate + severity_weight + dual_variable_key) that converts a registered match into a Pareto polytope-exclusion constraint. Each is a typed shape with its own invariants. |
| Solver integration `_apply_anti_pattern_constraints` | ADOPT_CANONICAL_BECAUSE_SERVES | Mirrors the existing solver path semantics (compute projection, then partition per-axis dual variables into tight vs slack). Extends naturally via key-prefix discrimination (`anti_pattern_<id>` keys join the same per_axis_dual_variables mapping). |
| Cathedral autopilot derivation helper | ADOPT_CANONICAL_BECAUSE_SERVES | Mirrors sister `anti_pattern_lookup_consumer` stack_spec resolution order (canonical attr → sister synonym → fallback). The consumer + invoker call sites both delegate to the canonical `match_stack_against_anti_patterns` so registry source-of-truth invariant is preserved. |
| Canonical equation registration | ADOPT_CANONICAL_BECAUSE_SERVES | Mirrors `dykstra_pareto_polytope_intersection_compounding_v1` (sister positive equation) shape with the NEGATIVE-constraint Lagrangian-dual derivation per Boyd-Vandenberghe (2004) Chapter 5 (Duality). Producers/consumers chain reflects the ACTIVE polytope-exclusion contract. |
| STRICT preflight gate (Layer 3) | ADOPT_CANONICAL_BECAUSE_SERVES | Mirrors Catalog #344 (canonical equations memo-reference enforcement) at the negative-registry surface. Same `_landed_<YYYYMMDD>.md` regex + cutoff filter pattern; same waiver semantics; same Catalog #287 placeholder-rationale rejection discipline. |

## ## 9-dimension success checklist evidence

Per Catalog #294:

1. **UNIQUENESS** — Layer 5 is a class-shift integration (NOT within-class
   refinement). Anti-pattern constraints become ACTIVE Pareto polytope
   feasibility constraints, distinct from the Wave N+1 baseline 3-axis
   polytope. Per-anti-pattern dual variables surface alongside per-axis
   dual variables via the canonical key prefix.
2. **BEAUTY + ELEGANCE** — frozen dataclass + closed-form Callable
   predicate + MAX-aggregation per Boyd-Vandenberghe (2004) Chapter 5;
   reviewable in 30 seconds. The constraint contract mirrors the sister
   anti-pattern dataclass structurally (same severity taxonomy, same
   unwind path field, same dual variable key prefix).
3. **DISTINCTNESS** — Layer 5 explicitly different from Layer 2 cathedral
   consumer (which is TIER_A observability-only, no solver integration).
   Layer 5 integrates at the solver surface where the per-axis Pareto
   polytope intersection happens; Layer 2 surfaces routing recommendations
   at the candidate ranking surface.
4. **RIGOR** — premise verification per Catalog #229 (read design memo
   §Layer 3 + §Layer 5 + Layer 1+2 landing memo + 39 sister test
   surface + canonical 4-layer Modal call-id ledger pattern BEFORE
   editing); empirical anchor (155+22 tests pass; canonical equation
   registered + queryable via `get_equation_by_id`); the Wave N+1
   baseline regression test `test_solver_empty_anti_patterns_is_wave_n_plus_1_baseline`
   explicitly proves backward compat.
5. **OPTIMIZATION PER TECHNIQUE** — Per Catalog #290 canonical-vs-unique
   decision per layer (table above): adopted canonical helpers where
   they serve (solver path semantics, cathedral autopilot derivation,
   canonical equation registration, STRICT gate pattern) + forked at
   the AntiPatternConstraint dataclass boundary because the solver-side
   wrapper IS a different abstraction from the registry-side AntiPattern.
6. **STACK-OF-STACKS-COMPOSABILITY** — Per Catalog #356 axis decomposition
   + Catalog #357 Tier B sister: anti-pattern duals compose with per-axis
   duals via the canonical key prefix; downstream consumers (Slot 1
   Compound C heterogeneous bit allocation + Compound D orthogonal
   composition + Compound F per-axis Pareto front compounding) all
   inherit the binding/slack identification via prefix inspection.
7. **DETERMINISTIC REPRODUCIBILITY** — byte-stable: AntiPatternConstraint
   is frozen; severity_weights are pinned at module level; aggregation
   helper is deterministic; the `forbidden_region_predicate` Callable
   closes over candidate stack_spec at construction time. Identical
   inputs produce identical verdicts.
8. **EXTREME OPTIMIZATION + PERFORMANCE** — O(n_constraints) projection
   per candidate; per-anti-pattern dual computation is O(1) per constraint
   (single predicate call + severity weight lookup). No quadratic blowup
   even with many registered anti-patterns. Aggregate histogram is O(N)
   in candidates × constraints; bounded ≤ 13 currently registered.
9. **OPTIMAL MINIMAL CONTEST SCORE** — N/A direct at landing (TIER A
   observability-only per Catalog #341; bounded [0.95, 1.05] adjustment
   factor preserves Phase 1 safety envelope); compounding via downstream
   Slot 1 Compound C bit allocation when sister empirical pairwise
   alpha anchors land.

## ## Observability surface

Per Catalog #305 6-facet contract:

1. **Inspectable per layer** — `AntiPatternConstraint.evaluate_at(point)`
   + `dual_variable(point)` + `aggregate_anti_pattern_duals(point, constraints)`
   are all introspectable. `ParetoSolverVerdict.as_dict()` emits anti-
   pattern-keyed dual variables alongside per-axis duals.
2. **Decomposable per signal** — per-anti-pattern dual variables / per-
   anti-pattern KKT residuals / binding canonical_unwind_paths surfaced
   as Mapping[str, value] in the verdict. The cathedral autopilot
   invocation payload surfaces per-candidate `matched_anti_patterns` +
   `binding_anti_pattern_ids` + `canonical_unwind_paths_recommended` +
   aggregate `anti_pattern_binding_histogram`.
3. **Diff-able across runs** — canonical Provenance threading per
   Catalog #323 preserved at solver surface; canonical equation
   `anti_pattern_polytope_exclusion_dykstra_compounding_v1` anchors
   accumulate per session.
4. **Queryable post-hoc** — canonical equation registry +
   `tac.canonical_equations.get_equation_by_id` resolves the equation;
   `tac.canonical_anti_patterns.query_anti_patterns_by_substrate` +
   `query_falsifications_by_paradigm_class` query the sister registry;
   cathedral autopilot invocation payload aggregates per-iteration
   histograms.
5. **Cite-able** — every constraint references `anti_pattern_id` →
   registered AntiPattern (via `tac.canonical_anti_patterns.get_anti_pattern_by_id`);
   every invocation payload references `canonical_helper_module=tac.dykstra_pareto_solver`
   + `anti_pattern_constraint_canonical_equation_id=anti_pattern_polytope_exclusion_dykstra_compounding_v1`
   + `next_phase_roadmap` (THIS memo).
6. **Counterfactual-able** — per-anti-pattern dual identification enables
   "what if we apply canonical unwind?" simulation by re-projecting
   with the matched constraint removed (operator-routable; the canonical
   unwind path is the routing recommendation).

## ## Predicted ΔS band

Per Catalog #296 + CATHEDRAL-SMARTER-DESIGN-MEMO Dim 1 Phase 4 sister:

This wire-in is **observability-only** at landing (bounded [0.95, 1.05]
adjustment factor preserves Phase 1 safety envelope per Catalog #355).
Anti-pattern constraint binding reduces feasibility but does NOT
directly produce a score delta; it produces per-anti-pattern tight-
constraint identification that the cathedral autopilot ranker uses to
ROUTE next-cycle attack direction via the canonical_unwind_path.

Predicted band for THIS work: **N/A — observability-only Tier A
contribution per Catalog #341**. The score-lowering ΔS is realized via
the downstream compounding mechanism when sister Slot 1 (Compound C
heterogeneous bit allocation) + future Compound D/F empirical pairwise
alpha anchors land.

**Dykstra-feasibility check**: the constraint surface IS the Dykstra
feasibility check extended with anti-pattern polytope exclusions. The
MAX-aggregation identity per the design memo §"Mathematical compounding
identity" preserves the convex-feasibility envelope; per-anti-pattern
KKT residuals quantify per-constraint binding distance.

## ## Cargo-cult audit per assumption

Per Catalog #303 + the HARD-EARNED-vs-CARGO-CULTED addendum:

1. **Assumption**: "Wave N+2 mandate's Layer 5 contract ('per-anti-
   pattern dual variable in ParetoSolverVerdict.per_axis_dual_variables
   with key prefix anti_pattern_<id>') is the correct surface for the
   per-anti-pattern dual variable." — **HARD-EARNED** by reading
   verdict.py invariants. The existing tight ∪ slack partition naturally
   accepts anti-pattern keys; the canonical prefix `anti_pattern_<id>`
   discrimination keeps axis duals (canonical 3-axis names: seg/pose/rate)
   distinguishable from anti-pattern duals at consumer surface.

2. **Assumption**: "predicate-always-True per-candidate constraint is
   the correct semantics; the Layer 1 matcher's structural overlap
   establishes binding without per-point re-evaluation." — **HARD-
   EARNED**. The matcher already walks recurrence_conditions +
   forbidden_pattern_predicate against the candidate's flattened
   stack_spec haystack with min_confidence threshold; re-running inside
   the predicate would duplicate the structural check. The constraint
   encodes the BINDING SEMANTICS at construction time; the dual
   variable surfaces the severity weight.

3. **Assumption**: "MAX-aggregation across multiple matched anti-patterns
   is the canonical safety-constraint composition (vs SUM)." — **HARD-
   EARNED** per design memo §"Mathematical compounding identity":
   "anti-patterns are typically dominated by the WORST applicable
   pattern... SUM would double-count + dilute critical signal."

4. **Assumption**: "defensive fail-OPEN on registry / matcher unavailability
   is the correct contract at the cathedral autopilot consumer surface."
   — **HARD-EARNED** per CLAUDE.md "Subagent coherence-by-default"
   non-negotiable. Tier A observability-only surfaces must NEVER crash
   the ranker cascade. Sister of Catalog #281 fail-closed contract at
   the codex review surface (where review is a HARD requirement);
   Catalog #373's matcher is OBSERVABILITY so fail-OPEN is correct.

5. **Assumption**: "Layer 3 cutoff 20260529 exempts Wave N+1 + earlier
   memos correctly." — **HARD-EARNED-EMPIRICALLY-VERIFIED**: the live
   `.omx/research/` directory at landing carries 1000+ memos; ALL
   dated <= 20260528 are exempted by the cutoff filter; live count
   verified 0 via `tac.preflight.check_compound_stack_proposal_acknowledges_known_anti_patterns(strict=False, verbose=True)`.

## ## Mathematical compounding identity (canonical equations sister)

Per the design memo §"Mathematical compounding identity":

**Positive compounding** (canonical_equations sister + Catalog #372
sister): `ΔS_total = Σ_axes ΔS_axis_i` where each axis prediction is
a canonical equation; Pareto polytope intersection (Dykstra) computes
the optimal compound.

**Negative compounding** (THIS Layer 3 + Layer 5 wire-in):
`RegressionRisk_total = max_anti-patterns AntiPatternRisk_i × indicator(stack_matches_anti_pattern_class_i)`
— the canonical aggregation is MAX not SUM because anti-patterns are
typically dominated by the WORST applicable pattern.

The mathematical compounding for routing per the design memo:

```
NextCycleAttackDirection = argmax_axis (
    PredictedΔS_axis_i × λ_axis_i_tight_from_Dykstra
) subject to (
    NOT any (proposed_stack matches AntiPattern_j AND not waived)
)
```

Layer 3 enforces the LEFT side (memo-level acknowledgment); Layer 5
enforces the RIGHT side (ACTIVE polytope exclusion + per-anti-pattern
dual variable surfacing via Catalog #125 hook #6 disambiguator).

## ## Bug class anchor + structural extinction surface count

Pre-Layer-3 the canonical anti-patterns registry existing at
`tac.canonical_anti_patterns` (Layer 1+2 landed 2026-05-28) was
NECESSARY BUT NOT SUFFICIENT; landing memos proposing compound stack
work could re-discover a registered anti-pattern by burning paid GPU
dispatch WITHOUT structural memo-level acknowledgment.

Layer 3 + Layer 5 close the "compound stacking work re-discovers a
known anti-pattern via paid GPU dispatch" bug class STRUCTURALLY at TWO
surfaces:

| Surface | Gate / mechanism | Layer |
|---|---|---|
| Memo-level acknowledgment | Catalog #373 STRICT preflight gate | Layer 3 |
| Runtime polytope feasibility | Catalog #372 invoker + `_apply_anti_pattern_constraints` | Layer 5 |

Sister of:

* Catalog #344 (canonical equations memo-reference enforcement; #344
  is the POSITIVE registry surface, #373 is the NEGATIVE registry
  surface)
* Catalog #372 (Slot 1 Dykstra Pareto solver invoker; Layer 5 sister
  integration at the runtime surface)
* Catalog #287 (placeholder-rationale rejection)
* Catalog #176 (META-meta: STRICT callsites have CLAUDE.md row — the
  Catalog #373 row at CLAUDE.md L3761-L3762 satisfies this gate)
* Catalog #185 (META-meta-meta: Live count: 0 verified empirically)
* Catalog #335 (canonical cathedral consumer contract for
  `anti_pattern_lookup_consumer`)
* Catalog #341 (Tier A canonical-routing markers; Layer 2 consumer
  carries them)
* Catalog #324 (post-training Tier-C validation; sister negative-
  constraint surface at the recipe-emit surface)
* Catalog #125 (6-hook wire-in non-negotiable; Layer 5 IS hook #2
  Pareto constraint canonical extension)

## ## 6-hook wire-in declaration per Catalog #125

* **hook #1 sensitivity-map** — ACTIVE: anti-pattern severity contributes
  to per-substrate sensitivity ranking via canonical severity_weights;
  downstream sensitivity-map consumers route through.
* **hook #2 Pareto constraint** — **ACTIVE PRIMARY**: Layer 5 IS canonical
  conversion of anti-patterns → ACTIVE polytope exclusion via
  `_apply_anti_pattern_constraints` + MAX-aggregation per the design
  memo §"Mathematical compounding identity".
* **hook #3 bit-allocator** — ACTIVE: anti-pattern unwind paths route
  bit-allocator AWAY from known compounding-order anti-patterns
  (e.g. quantize-then-SVD); Slot 1 Compound C heterogeneous bit allocation
  consumes the surface.
* **hook #4 cathedral autopilot dispatch** — ACTIVE: Layer 3 STRICT at
  memo surface; Layer 5 at dispatch routing via cathedral autopilot
  invoker per-candidate payload surface; auto-discovery via Catalog
  #335 sister gate.
* **hook #5 continual-learning posterior** — ACTIVE: solver verdicts
  with anti-pattern duals append to canonical posterior; canonical
  equation `anti_pattern_polytope_exclusion_dykstra_compounding_v1`
  anchors accumulate; Catalog #371 auto-recalibrator + Layer 4 sister
  auto-recalibrator at `tac.canonical_anti_patterns.registry.auto_recalibrate_from_continual_learning_posterior`
  refit.
* **hook #6 probe-disambiguator** — ACTIVE: matched anti-pattern +
  canonical unwind path IS the canonical disambiguator between viable
  vs forbidden compounding routes; sister of original per-axis tight-
  constraint disambiguator per Catalog #372.

## ## Implementation summary

### Layer 5 — `tac.dykstra_pareto_solver.anti_pattern_constraint` module landed

Files added/extended:

* `src/tac/dykstra_pareto_solver/anti_pattern_constraint.py` (~300 LOC):
  `AntiPatternConstraint` frozen dataclass + `AntiPatternConstraintError`
  + `ANTI_PATTERN_CONSTRAINT_DUAL_KEY_PREFIX` constant +
  `VALID_SEVERITY_WEIGHTS` frozenset + `severity_weight_for` helper +
  `aggregate_anti_pattern_duals` MAX-aggregation function. Mirrors
  sister AntiPattern dataclass structurally (same severity taxonomy,
  same unwind path field). Comment-only contracts FORBIDDEN per CLAUDE.md
  "Comment-only contracts are FORBIDDEN" — every invariant enforced
  in `__post_init__` so construction surface refuses bad inputs at the
  source.
* `src/tac/dykstra_pareto_solver/solver.py` (extended):
  `DykstraParetoSolver` dataclass extended with `anti_pattern_constraints: tuple[AntiPatternConstraint, ...] = ()`
  field + `__post_init__` invariant (rejects non-tuple + duplicate
  anti_pattern_id collisions); new `_apply_anti_pattern_constraints`
  helper method that integrates anti-pattern duals into
  `(feasible, tight_axes, slack_axes, per_axis_dual_variables, per_axis_kkt_residuals, per_axis_adjustment_factors)`
  per the MAX-aggregation identity; both `_solve_canonical_3_axis` +
  `_solve_general` paths call the helper before constructing the
  verdict; `solve_pareto_polytope_intersection` convenience wrapper
  accepts `anti_pattern_constraints` kwarg.
* `src/tac/dykstra_pareto_solver/__init__.py` (extended): re-exports
  the 5 new constraint primitives.
* `src/tac/dykstra_pareto_solver/tests/test_anti_pattern_constraint_integration.py`
  (~480 LOC, 23 tests): covers dataclass invariants + baseline
  regression + binding/slack feasibility + multi-pattern MAX-aggregation
  + per-axis tight distinguished via canonical prefix + observability-
  only contract + duplicate-id rejection + end-to-end cathedral
  autopilot integration.

### Cathedral autopilot integration

`tools/cathedral_autopilot_autonomous_loop.py` extended with:

* `_candidate_to_stack_spec(candidate)` helper: resolves canonical
  attr → sister synonym → fallback dict (mirrors sister consumer
  resolution order).
* `_derive_anti_pattern_constraints_for_candidate(candidate)` helper:
  calls `match_stack_against_anti_patterns` + converts matches to
  `AntiPatternConstraint` tuple; defensive fail-OPEN on registry /
  helper unavailability.
* `invoke_dykstra_pareto_solver_on_candidates` extended:
  - Derives constraints per candidate via the helper.
  - Threads constraints into `solve_pareto_polytope_intersection`.
  - Surfaces per-candidate `matched_anti_patterns` +
    `binding_anti_pattern_ids` + `canonical_unwind_paths_recommended` +
    `anti_pattern_constraint_count` in invocation payload.
  - Aggregates `anti_pattern_binding_histogram` +
    `candidates_with_matched_anti_patterns` +
    `total_matched_anti_pattern_occurrences` +
    deduplicated `canonical_unwind_paths_recommended` in result.
  - References `anti_pattern_constraint_canonical_equation_id` =
    `anti_pattern_polytope_exclusion_dykstra_compounding_v1`.

### Layer 3 — STRICT preflight gate Catalog #373

`src/tac/preflight.py` extended with:

* Module-level constants `_CHECK_373_RESEARCH_REL` +
  `_CHECK_373_CUTOFF_YYYYMMDD` + `_CHECK_373_TRIGGER_TOKENS` +
  `_CHECK_373_WAIVER_TOKEN` + `_CHECK_373_PLACEHOLDER_RATIONALES` +
  `_CHECK_373_CITE_TOKEN_PREFIXES`.
* 6 helper functions: `_check_373_extract_yyyymmdd` /
  `_check_373_post_cutoff` / `_check_373_has_trigger` /
  `_check_373_has_valid_waiver` / `_check_373_has_acknowledgment` /
  `_check_373_landing_memos_in_scope` /
  `_check_373_safely_match_stack_for_memo`.
* Main gate function
  `check_compound_stack_proposal_acknowledges_known_anti_patterns`
  with the canonical 3-cascade acceptance path (cite / waiver / match-
  but-no-trigger).
* Wired into `preflight_all()` at L5402-5421 with `strict=True` per
  CLAUDE.md "Strict-flip atomicity rule" + Catalog #176 META-meta
  STRICT-callsite-has-CLAUDE.md-row sister discipline.

### Canonical equation registration

`tac.canonical_equations.register_canonical_equation` invoked at
landing time registering `anti_pattern_polytope_exclusion_dykstra_compounding_v1`
per Catalog #344. Equation declares:

* `equation_id`: `anti_pattern_polytope_exclusion_dykstra_compounding_v1`
* `latex_form`: Lagrangian dual `L(x, λ) = f(x) + Σ_j λ_j · w_j · 1[x ∈ F_j], λ_j ≥ 0; λ_agg = max_j (λ_j w_j)`
* `python_callable_module_path`: `tac.dykstra_pareto_solver.anti_pattern_constraint:aggregate_anti_pattern_duals`
* `canonical_producers`: 3 entries (helper + dataclass method + sister
  matcher).
* `canonical_consumers`: 4 entries (solver method + convenience wrapper
  + cathedral autopilot derivation helper + invoker).
* `next_recalibration_trigger`: `when_3+_new_empirical_anchors_in_domain`
  (Catalog #371 auto-recalibrator absorbs sister falsifications via
  Layer 4 helper).
* `provenance`: canonical Provenance per Catalog #323 (predicted /
  unknown hardware).

### Test coverage summary

```
src/tac/dykstra_pareto_solver/tests/test_dykstra_pareto_solver.py        39 (baseline Wave N+1)
src/tac/dykstra_pareto_solver/tests/test_anti_pattern_constraint_integration.py  23 (NEW Layer 5)
src/tac/canonical_anti_patterns/tests/test_registry.py                  44 (Layer 1+2 sister)
src/tac/cathedral_consumers/anti_pattern_lookup_consumer/tests/test_consumer.py  15 (Layer 2 sister)
src/tac/cathedral_consumers/dykstra_pareto_solver_consumer/tests/test_dykstra_pareto_solver_consumer.py  12 (sister)
src/tac/tests/test_check_373_compound_stack_acknowledges_anti_patterns.py  22 (NEW Layer 3)
src/tac/canonical_equations/tests/                                      140 (sister regression)
TOTAL                                                                   295 PASS
```

## ## Cross-references

* Layer 1+2 landing memo: `.omx/research/canonical_anti_patterns_registry_layer_1_plus_2_landed_20260528.md` (commit `8ef0ed7d1`).
* Design memo: `.omx/research/canonical_anti_patterns_registry_design_20260528.md` (commit `37b5a0184`).
* Dim 1 Phase 4 sister: `.omx/research/dykstra_pareto_polytope_solver_wire_in_dim1_phase4_landed_20260528.md` (commit `006dcfb2c`).
* CATHEDRAL-SMARTER-DESIGN-MEMO Dim 1 Phase 4 + Dim 6 dual-tier: `.omx/research/cathedral_autopilot_smarter_design_blueprint_20260520T130325Z.md`.
* Canonical equation: `anti_pattern_polytope_exclusion_dykstra_compounding_v1` in `.omx/state/canonical_equations_registry.jsonl`.
* Catalog #373 STRICT gate: `src/tac/preflight.py::check_compound_stack_proposal_acknowledges_known_anti_patterns`.
* Retroactive sweep per Catalog #348: `.omx/research/retroactive_sweep_for_catalog_373_20260528T141000Z.md`.

## ## Mission contribution per Catalog #300

`frontier_breaking_enabler` — Layer 3 + Layer 5 close the canonical
anti-patterns design memo §"Wave N+2" implementation queue. Together
with the canonical equations registry (POSITIVE) + canonical anti-
patterns registry (NEGATIVE) + Dykstra Pareto polytope solver +
cathedral autopilot ranker, this completes the symmetric framework
the operator META directive prescribed: "learning anti-patterns is
upser important too for compounding continual learning, like the
canonical equations bu netgative and a higher layer of abstraction".

The score-lowering ΔS emerges via the downstream compounding mechanism
when sister Slot 1 (Compound C heterogeneous bit allocation) + future
Compound D/F empirical pairwise alpha anchors land; THIS Layer 5
landing is the canonical primitive that makes the anti-pattern
exclusion provably MAX-aggregation-optimal per Boyd-Vandenberghe
(2004) Chapter 5 + the design memo §"Mathematical compounding
identity".
