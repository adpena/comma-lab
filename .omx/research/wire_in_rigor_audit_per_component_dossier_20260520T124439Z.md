# Wire-In Rigor Audit — Per-Component Dossier (Resume Wave)

**Subagent**: `wire-in-rigor-audit-resume-20260520` (resumed from crashed predecessor `wire-in-rigor-audit-20260520`)
**Scope**: 20+ Tier 1-5 components per parent dispatch brief
**Methodology**: source-file read + AST/grep call-chain trace + empirical runtime invocation where feasible
**Discipline**: Catalog #229 PV; #287 placeholder-rejection; #323 canonical Provenance; #292 per-component assumption surfacing
**Date**: 2026-05-20

## Verdict legend

- **FACADE** — surface exists, IMPORTS clean, TESTS pass, but downstream consumer never actually exercises the work
- **ORPHAN_SIGNAL** — producer fires, but no consumer ingests the output (signal lost)
- **PARTIALLY_WIRED** — runtime-active by design, but contribution is observability-only / does not influence the dispatch decision
- **FULLY_WIRED** — runtime-active, downstream consumer ingests output, output influences dispatch / promotion / score
- **SCAFFOLD_ONLY** — package exists, public API defined, but zero production callsites and self-tests only

## Tier 1: Decision-engine components (sanity-checked from predecessor)

### 1. Cathedral autopilot ranker (`tools/cathedral_autopilot_autonomous_loop.py`)
- **Verdict**: PARTIALLY_WIRED
- **Evidence**: `main()` invokes `invoke_cathedral_consumers_on_candidates` at lines 7043 (`--report-only`) and 7172 (`run_continuous_loop`). Per Catalog #336+#337 STRICT gates this is structurally enforced. Empirical runtime: 44 consumers discovered + invoked, each emits annotation row into output JSON.
- **Gap**: 44/44 consumers return `predicted_delta_adjustment=0.0` (observability-only by design per Catalog #341). The IN-MAIN-LINE ranker uses ~10 `adjust_predicted_delta_for_*` adjusters that DO mutate score; cathedral consumers are parallel observability surface.
- **Assumption surface**: "auto-discovery + observability annotations sufficient to extinct orphan-signal class" — empirically PARTIAL: discovery fires, but consumers do not feed score path.

### 2. Meta-Lagrangian / unified-action solver (`src/tac/unified_action.py`)
- **Verdict**: SCAFFOLD_ONLY / ORPHAN_SIGNAL
- **Evidence**: `Action.S_total`, `Action.gradient`, `Action.step`, `choose_solver`, `evaluate_with_admm`, `evaluate_with_magic_codec` all defined + tested. `experiments/train_unified_action_phase1.py` exists.
- **Gap**: ZERO production callsites in `tools/` / `experiments/` / `src/` actually invoke `evaluate_with_admm` or `choose_solver`. Cathedral consumer `unified_action_consumer` is STUB-by-design (zero adjustment). The "Meta-Lagrangian/Pareto solver — NON-NEGOTIABLE" rule's mandate is structurally unsatisfied at the dispatch surface.
- **File:line**: `src/tac/cathedral_consumers/unified_action_consumer/__init__.py:42-61` (stub returns observability annotation only).
- **Assumption surface**: "solver-as-library availability sufficient for dispatch" — FALSE; no caller exists.

### 3. Master-gradient extractor (`tools/extract_master_gradient.py`)
- **Verdict**: PARTIALLY_WIRED
- **Evidence**: 10 anchors in `.omx/state/master_gradient_anchors.jsonl`. Per-pair extraction tested + per-byte extraction tested.
- **Gap**: ALL 10 anchors are `[macOS-CPU advisory]` or pre-axis-correction `[contest-CPU]` on M5 Max (NOT 1:1 contest-compliant per CLAUDE.md "Submission auth eval"). Catalog #327 enforces fail-closed routing for non-authoritative axes; consumers refuse non-authoritative anchors by design. 0 paired `[contest-CUDA T4]` + `[contest-CPU Linux x86_64]` anchors exist.
- **Assumption surface**: "macOS-CPU + scorer surrogate sufficient to extract score-relevant master gradient" — UNCERTAIN; empirically unvalidated against contest hardware.

### 4. Cathedral consumers (`src/tac/cathedral_consumers/`)
- **Verdict**: FACADE (in aggregate)
- **Evidence**: 46 packages, 44 production (excluding `_example_consumer`). All canonical-contract-compliant per Catalog #335. All discoverable + loadable + invocable per Catalog #336+#337. Empirical: 44 consumers fire successfully on cathedral autopilot `--report-only`.
- **Gap**: **EVERY SINGLE production consumer returns `"predicted_delta_adjustment": 0.0`** (verified via grep). 0 of 44 mutate score. The cathedral consumer package is observability-annotation-only by design per Catalog #341 (`predicted_delta_adjustment=0.0` / `promotable=False` / `axis_tag="[predicted]"`). The "FACADE" label is design-correct: consumers fire but do not influence dispatch ranking.
- **Assumption surface**: "observability annotations on per-candidate axis = sufficient signal extinction of orphan-signal class" — DESIGN-CORRECT but mission-functionally LOW IMPACT.

## Tier 2: Signal / observability components

### 5. xray observability primitive (`src/tac/xray/`)
- **Verdict**: PARTIALLY_WIRED
- **Evidence**: 18 xray primitives registered. Consumed by `tac.unified_action.optimizer_analytical_boundaries:594-619` (lines `wire_in_for_hook` + `discover_primitives_by_hook` + `aggregate_hook_evidence_grade`).
- **Gap**: `optimizer_analytical_boundaries` output (`xray_hook_bundles`) is consumed by cathedral consumer `unified_action_consumer` which is STUB-by-design (returns 0.0). No downstream score path.
- **File:line**: `src/tac/unified_action.py:589-619` (xray bundle assembly); `src/tac/cathedral_consumers/unified_action_consumer/__init__.py:42` (consumer stub).
- **Assumption surface**: "xray inspection sufficient as documentation surface" — CORRECT for observability; INCORRECT for dispatch influence.

### 6. Atoms primitive (`src/tac/atom/`)
- **Verdict**: PARTIALLY_WIRED
- **Evidence**: 10 modules under `tac.analytical_solve_extinctions/*` use atoms via `emit_arbitrariness_atom=True`. Atom emits JSONL rows to canonical store. Per CLAUDE.md "Meta-Lagrangian/Pareto solver" the atom-row contract IS canonical (per `pr101_pose_filler_stc_anchor.py:259`).
- **Gap**: Cathedral consumer `atom_consumer` is STUB-by-design. Atom rows DO feed `tac.analytical_solve_extinctions` namespace, which feeds CLI tools, but the dispatch path doesn't query atoms per-candidate.
- **Assumption surface**: "atom emission sufficient as audit trail" — CORRECT; not in dispatch decision.

### 7. Sensitivity-map module (`src/tac/sensitivity_map/`)
- **Verdict**: PARTIALLY_WIRED
- **Evidence**: `axis_weights.default_axis_weights` + `wyner_ziv_reweight.axis_level_reweight` consumed in `tac.unified_action.optimizer_analytical_boundaries:573` AND in `tac.optimization.bit_allocator_end_to_end:1010`. ~12 production consumers.
- **Gap**: Final-consumer chain ends at cathedral consumer (stub) OR at `OptimizerAnalyticalBoundaries.bit_allocation_envelope`. Bit allocator IS produced; whether dispatch consumes it depends on per-candidate `OptimalPerPairTreatmentPlan` sidecars (currently sparse).
- **Assumption surface**: "sensitivity weights flow into bit allocator → influence per-pair coding budget → influence rate term" — STRUCTURALLY CORRECT, but EMPIRICALLY UNGROUNDED (no contest-CUDA anchor consumed end-to-end through this chain yet).

### 8. Canonical equations registry (`src/tac/canonical_equations/`)
- **Verdict**: PARTIALLY_WIRED
- **Evidence**: 11 equations registered (`brotli_cascade_bounded_per_stream_v1`, `mps_drift_architecture_class_dependent_v1`, `per_byte_leverage_uniformly_distributed_v1`, `per_pair_master_gradient_score_impact_taylor_v1`, etc.). 2 operator-facing CLIs. 7 cathedral consumers cite them.
- **Gap**: 7 cathedral consumers citing equations are ALL STUB-by-design (0.0 adjustment). No runtime path mutates per-candidate score via equation lookup. Producer-only with operator audit; not in dispatch decision loop.
- **Assumption surface**: "equation citation via cathedral consumer sufficient to formalize tribal knowledge" — DESIGN-CORRECT per Catalog #344 (formalization-pending waiver); functional-influence not yet on dispatch path.

## Tier 3: Canonical-state ledgers

### 9. Continual-learning posterior (`tac.continual_learning`)
- **Verdict**: FULLY_WIRED
- **Evidence**: 118 accepted anchors / 31 refused / last updated 2026-05-19T15:49Z. Consumed by `cathedral_autopilot_autonomous_loop.run_continuous_loop` via `--load-continual-posterior` flag.
- **Producer-consumer chain**: archive empirical anchors → `posterior_update_locked` → cathedral autopilot ranker via `_load_continual_posterior_into_candidates` → influences dispatch ranking via Z1 revision adjusters.

### 10. Probe-outcomes ledger (Catalog #313)
- **Verdict**: FULLY_WIRED
- **Evidence**: 59 outcomes (30 DEFER / 20 PROCEED / 6 PARTIAL / 3 INDEPENDENT). Catalog #313 STRICT gate enforces consumption by dispatch wrappers; runtime gate in `tools/operator_authorize.py::_check_predecessor_probe_outcome` refuses dispatch with blocking predecessor.
- **Producer-consumer chain**: probe disambiguator → `register_probe_outcome` → operator_authorize gate → dispatch blocked OR allowed.

### 11. Modal call-id ledger (Catalog #245)
- **Verdict**: FULLY_WIRED
- **Evidence**: 393 rows (148 dispatched / 163 failed / 81 harvested / 1 stale). Catalog #245 + #339 enforce. Consumed by `tools/harvest_modal_calls.py` + `tools/parallel_harvest_actuator.py`.
- **Producer-consumer chain**: `fn.spawn()` → `register_dispatched_call_id_fail_closed` → harvester polls + appends `harvested` event.

### 12. Canonical Provenance umbrella (Catalog #323)
- **Verdict**: FULLY_WIRED
- **Evidence**: `audit_score_claim_dict` callable; 10+ production consumers (audit tool + 4 advisory smokes + master_gradient surfaces).
- **Producer-consumer chain**: any score-claim row → `validate_provenance` → STRICT preflight refuses if missing canonical Provenance.

## Tier 4: Composition + substrate-class

### 13. Substrate composition matrix (Catalog #319)
- **Verdict**: PARTIALLY_WIRED
- **Evidence**: 56 canonical inventory rows; 2 entries in `.omx/state/substrate_composition_matrix.json`. Autopilot reweight v2 (`adjust_predicted_delta_for_composition_alpha_v2`) consumes via Cascade 3.
- **Gap**: Only 2 persisted entries; Cascade 1 (Lagrangian optimal-plan) requires per-archive `optimal_plan_*.json` sidecars (currently 0 found). Cascade 3 falls through to 1.0× passthrough. Per Catalog #322 fec6 NOT_DELIVERABLE classification = no fake reward.
- **Assumption surface**: "composition reward will fire when paired-alpha probes populate matrix" — CURRENTLY EMPIRICALLY UNFIRED.

### 14. Wyner-Ziv deliverability proof builder (Q1+Q2+Q3)
- **Verdict**: PARTIALLY_WIRED
- **Evidence**: Canonical helper at `src/tac/wyner_ziv_deliverability/proof_builder.py`. Catalog #319 STRICT gate enforces consumption. Cathedral autopilot `adjust_predicted_delta_for_venn_classification_v2` consumes via 3-cascade.
- **Gap**: 1 sidecar exists (`probe_f174192aeadf_*.json`) with `deliverability_verdict=NOT_DELIVERABLE`. 0 archives have positive deliverable savings currently.
- **Assumption surface**: "WZ reward branch can fire" — STRUCTURALLY YES, EMPIRICALLY NO (only NOT_DELIVERABLE anchor exists).

### 15. Master-gradient consumers (Task #890 47% under-wired anchor)
- **Verdict**: PARTIALLY_WIRED (per Catalog #354 + canonical-path Q3)
- **Evidence**: Catalog #354 bundle complete — 8/8 required exploit consumers loadable. `load_optimal_plan_for_archive` IS consumed in `adjust_predicted_delta_for_venn_classification_v2:1473` (Cascade 1 — REAL score mutation when sidecar exists).
- **Gap**: 0 `optimal_plan_*.json` sidecars currently in `.omx/state/master_gradient_consumers/`. Cascade 1 is structurally wired, empirically unfired. 15/15 master-gradient cathedral consumers loadable but observability-only by design.

## Tier 5: Helper namespaces

### 16. `tac.boosting` namespace
- **Verdict**: ORPHAN_SIGNAL
- **Evidence**: Rich public API (`BoostStageContract`, `BoostingPerPairWireInOutcome`, ledger constants).
- **Gap**: Only 1 production consumer (`src/tac/master_gradient_consumers.py`). No tool / experiment / dispatch wrapper exercises the helper namespace end-to-end.

### 17. `tac.compress_time_optimization` namespace
- **Verdict**: SCAFFOLD_ONLY
- **Evidence**: Rich public API (`ComposableCompressPipeline`, `CompressTimePassContract`, persistence layer).
- **Gap**: Consumed ONLY by its own sub-modules + tests. No tool / experiment / training script outside the namespace itself.

### 18. Canonical frontier pointer (`tac.canonical_frontier_pointer`)
- **Verdict**: FULLY_WIRED
- **Evidence**: Recently refreshed (2026-05-20T11:57Z). CPU frontier 0.1920513 (PR101 fec6) + CUDA frontier 0.2053300 (PR106 format0d). Used by `tools/refresh_canonical_frontier.py` + autopilot dispatch.

### 19. Cargo-cult-unwind methodology (Catalog #303)
- **Verdict**: FULLY_WIRED (operationally; methodology not a code surface)
- **Evidence**: Catalog #303 STRICT gate enforces `## Cargo-cult audit per assumption` section in every substrate design memo dated >= 2026-05-16. Empirical anchor: NSCS06 v6→v7 = 44% improvement via cargo-cult-unwind in ONE iteration.
- **Producer-consumer chain**: design-memo → catalog-#303-gate → council deliberation → next design iteration.

### 20. Per-substrate symposium discipline (Catalog #325)
- **Verdict**: FULLY_WIRED (operationally)
- **Evidence**: Catalog #325 STRICT gate enforces 14-day symposium memo + posterior anchor for every dispatchable substrate recipe. C6 IBPS first canonical instance landed 2026-05-18.
- **Producer-consumer chain**: substrate recipe → catalog-#325-gate → symposium memo + council anchor → dispatch eligibility.

## Per-verdict totals (20 components)

| Verdict | Count | Components |
|---|---|---|
| FULLY_WIRED | 7 | continual-learning posterior / probe-outcomes / modal call-id / canonical Provenance / canonical frontier pointer / cargo-cult discipline / per-substrate symposium discipline |
| PARTIALLY_WIRED | 9 | cathedral autopilot / master-gradient extractor / xray / atoms / sensitivity-map / canonical equations / substrate composition matrix / WZ deliverability / master-gradient consumers |
| FACADE | 1 | cathedral consumers (in aggregate; 44/44 production stubs at 0.0) |
| ORPHAN_SIGNAL | 1 | tac.boosting |
| SCAFFOLD_ONLY | 2 | meta-Lagrangian solver / tac.compress_time_optimization |

## Cross-cutting findings

1. **Cathedral consumer 44/44 STUB-by-design**: Per Catalog #341 canonical contract, ALL production cathedral consumers return `predicted_delta_adjustment=0.0` / `promotable=False` / `axis_tag="[predicted]"`. This is correct per design (observability-only annotation) but means the cathedral autopilot's "44 consumers fire" claim corresponds to ZERO score mutation. The real adjusters (`adjust_predicted_delta_for_*`) are 10 in-main-line functions, not cathedral consumers.

2. **Meta-Lagrangian solver ZERO callers**: The canonical "Meta-Lagrangian/Pareto solver — NON-NEGOTIABLE" rule's solver surface (`evaluate_with_admm`, `choose_solver`, `Action.S_total`) has ZERO production callsites. The training scaffold exists but is not exercised in any dispatch wrapper / ranker / autopilot path. The solver is SCAFFOLD_ONLY at the highest-tier component the CLAUDE.md mandates as canonical.

3. **Master-gradient anchors non-authoritative**: 10/10 anchors are `[macOS-CPU advisory]` or pre-axis-correction `[contest-CPU]` on M5 Max. Catalog #327 fail-closed routing correctly refuses them as non-authoritative. The producer fires, the consumers gate correctly, but the empirical signal is UNVALIDATED on contest hardware.

4. **Composition matrix empirically unfired**: 56 inventory rows declared; 2 persisted entries; 0 positive deliverable WZ proofs; 0 Lagrangian optimal-plan sidecars. The Catalog #319 Q3 cascade is structurally wired but Cascade 1 (REAL mutation) has 0 archives meeting precondition.

5. **Helper namespaces (`tac.boosting`, `tac.compress_time_optimization`) orphan**: Rich APIs, no production consumers outside the namespace itself.


<!-- WAVE-3-CATALOG-344-BACKFILL-SWEEP appended 2026-05-20 per operator NON-NEGOTIABLE "keep feeding the queue" + WIRE-IN-AUDIT-POST-CASCADE op-routable #3. -->
<!-- # FORMALIZATION_PENDING:wire-in-rigor-audit-per-component-dossier-trigger-tokens-describe-audited-components-not-new-equation -->
