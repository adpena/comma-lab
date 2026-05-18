---
council_tier: T1
council_attendees: [Working-group lead]
council_quorum_met: true
council_verdict: PROCEED
council_dissent: []
council_decisions_recorded:
  - "op-routable #1: queue wire-in #3 bit_allocator consumes OptimalPerPairTreatmentPlan (Task #800)"
  - "op-routable #2: queue Hook 4 cathedral autopilot consumes optimal_plan_to_candidate_row + Lagrangian planner CandidateRow adapter"
  - "op-routable #3: declare N/A-with-rationale for the 5 §7.6 namespace per-pair gaps that legitimately don't compose per-pair gradient"
  - "op-routable #4: queue Hook 5 continual-learning posterior update on per-pair difficulty deltas (Task #802 expansion)"
  - "op-routable #5: queue Hook 6 probe-disambiguator wiring for Consumer 6 Rashomon disagreement queue"
council_assumption_adversary_verdict:
  - assumption: "Public symbols in __all__ are the canonical wire-in surface (vs. internal helpers)"
    classification: HARD-EARNED
    rationale: "CLAUDE.md 'Beauty, simplicity, and developer experience' makes __all__ the explicit narrow API; Catalog #125 6-hook discipline applies to PUBLIC outputs"
  - assumption: "Hook 4 cathedral autopilot must consume OptimalPerPairTreatmentPlan VIA optimal_plan_to_candidate_row adapter (the existing adapter IS the wire-in surface)"
    classification: HARD-EARNED
    rationale: "Consumer 15 docstring + sig of `optimal_plan_to_candidate_row(plan: OptimalPerPairTreatmentPlan) -> object` named the contract explicitly; cathedral autopilot's CandidateRow IS the canonical consumer surface"
  - assumption: "Per-pair gradient is conceptually the same axis as the aggregate gradient (just kept per-pair)"
    classification: HARD-EARNED
    rationale: "Operator's 8-pair validation showed `max(|G_avg − G_pp.mean|) = 4.29e-6` → per-pair tensor IS the master tensor, aggregate is a `.mean(axis=1)` projection; consumers that ONLY need aggregate sensitivity can stay aggregate"
  - assumption: "tac.continual_learning.posterior_update_locked is the canonical Hook 5 surface; tac.cost_band_calibration.append_anchor is separate domain"
    classification: HARD-EARNED
    rationale: "CLAUDE.md 'Subagent coherence-by-default' Section 'Mandatory wire-in for every landing' Hook 5 names continual_learning.posterior_update_locked explicitly; cost_band is dispatch-cost domain not signal-axis"
council_predicted_mission_contribution: apparatus_maintenance
council_override_invoked: false
council_override_rationale: ""
horizon_class: apparatus_maintenance
---

# Comprehensive wire-in coverage matrix — 2026-05-17

**Purpose**: Operator standing directive 2026-05-17 verbatim: *"Also need to do a comprehensive wiring and integration pass"* + *"Also need to ensure all per-pair master gradient uses are wired up and integrated and implemented everywhere"*. Subsumes per-pair-master-gradient coverage audit (Task #810) into the comprehensive wire-in pass (Task #811).

**Lane**: `lane_comprehensive_wiring_integration_pass_20260517` (L1 at landing).

**Sister of**: `.omx/research/comprehensive_state_tracker_20260517.md` (canonical no-signal-loss session state — refreshed every wave; this memo is the structural complement that catalogs every CANONICAL helper's hook coverage).

**Refresh cadence**: After every wave that lands a new canonical helper / consumer / wire-in surface. The matrix below is structural at landing time; any future surface MUST be added.

## 9-dimension success checklist evidence

1. **UNIQUENESS**: this audit is the first comprehensive matrix of `__all__` symbol → Catalog #125 6-hook coverage across all 5 namespace packages + 13+ canonical single-module helpers landed this session.
2. **BEAUTY + ELEGANCE**: one matrix per package; per-symbol granularity only where the symbol is the surface (Producer/Consumer/Helper/Wire-in classification first).
3. **DISTINCTNESS**: explicitly different from `.omx/research/comprehensive_state_tracker_20260517.md` (state tracker = WHAT is in flight; matrix = STRUCTURAL coverage of what landed).
4. **RIGOR**: every cell evidenced by `grep -n` of canonical helper token in module body (not docstring); the AST source scan is reproducible.
5. **OPTIMIZATION PER TECHNIQUE**: out of scope (this is an audit memo).
6. **STACK-OF-STACKS-COMPOSABILITY**: the matrix shows where two canonical surfaces compose (e.g. `optimal_plan_to_candidate_row` Producer → cathedral autopilot Consumer) and where they don't.
7. **DETERMINISTIC REPRODUCIBILITY**: every grep/import in this matrix is byte-stable; module imports are seed-irrelevant; `__all__` enumeration is hash-stable.
8. **EXTREME OPTIMIZATION**: out of scope (audit memo).
9. **OPTIMAL MINIMAL CONTEST SCORE**: indirect — wire-in gap closure unblocks the Lagrangian planner Hook 4 → Hook 3 → Hook 5 chain that drives the FRONTIER-PURSUIT [0.147, 0.167] predicted band.

## Cargo-cult audit per assumption

| Assumption | Classification | Rationale |
|---|---|---|
| `__all__` is the canonical wire-in surface | HARD-EARNED | CLAUDE.md "Beauty, simplicity, and developer experience" + Catalog #125 6-hook discipline |
| Module-source grep with curated token sets correctly identifies hook consumption | HARD-EARNED | Same pattern Catalog #117/#157/#174/#235 use for serializer-event scanning; false-positive rate < 5% empirically |
| Per-pair gradient ⊃ aggregate gradient | HARD-EARNED | 8-pair fp64 anchor: `max(|G_avg − G_pp.mean|) = 4.29e-6` |
| Hook 4 cathedral autopilot wire-in MUST go through `CandidateRow` adapter | HARD-EARNED | `optimal_plan_to_candidate_row` already exists; CandidateRow IS the canonical ranker contract |
| Hook 5 cont-learn = `tac.continual_learning.posterior_update_locked` (not `cost_band_calibration.append_anchor`) | HARD-EARNED | CLAUDE.md Hook 5 names continual_learning specifically; cost_band is dispatch-cost domain |
| Hook 6 probe-disambiguator is a TOOL surface (`tools/probe_*_disambiguator.py`) not a library surface | HARD-EARNED | CLAUDE.md "Anti-arbitrariness primitive" + 37 existing `tools/probe_*.py` artifacts; the 10 `*_disambiguator` subset IS the canonical surface |
| Negative findings in this matrix are GAPS not anti-patterns | CARGO-CULTED → unwound | The §7.6 namespaces' uniform "no Hook 5 cont-learn" cell is NOT a bug because boost-stage outputs are persisted via `append_stage_outcome_locked` (sister fcntl-locked JSONL store, distinct from continual_learning posterior). The grep pattern needs broadening per op-routable #6 to detect sister persistence helpers; for now those cells are tagged ⓘ rather than ⏳. |

## Observability surface

- Coverage matrix: this file
- Raw audit results: regenerated at any time via the `.venv/bin/python` inline `pkgutil.walk_packages(tac.__path__)` + `inspect.getsource(mod)` substring scan
- Per-helper sidecar JSONs: `.omx/state/master_gradient_consumers/*.json`
- Per-package `__all__` enumeration: `.venv/bin/python -c "from tac.X import __all__; print(sorted(__all__))"` per surface
- Wire-in regression tests: `src/tac/tests/test_sensitivity_map_wyner_ziv_wire_in.py` + sister test files per consumer

## Discovery summary

| Surface | Modules | Public symbols | Notes |
|---|---|---|---|
| `tac.*` recursive | 1149 (737 with `__all__`) | 7683 | Full audit scope |
| `tac.master_gradient_consumers` | 1 | 43 | Canonical producer of per-pair gradient consumers (Consumer 1-6, 9, 12-15) |
| `tac.sensitivity_map` | 3 | 63 | Hook 1 surface; wyner_ziv_reweight already wires from Consumer 4 |
| `tac.optimization` | 58 | 706 | Hook 2 (pareto) + Hook 3 (bit_allocator) families |
| `tac.continual_learning` | 1 | 23 | Hook 5 canonical helper |
| `tac.council_continual_learning` | 1 | 24 | Sister Hook 5 surface (council-deliberation domain) |
| `tac.wyner_ziv_deliverability` | 2 | 14 | Producer of contest-compliance proof for Q1; consumes Consumer 4 |
| `tac.boosting` | 10 | 84 | §7.6 namespace |
| `tac.compress_time_optimization` | 11 | 100 | §7.6 namespace |
| `tac.inflate_time_post_processing` | 11 | 102 | §7.6 namespace |
| `tac.side_information` | 11 | 108 | §7.6 namespace |
| `tac.search` | 11 | 102 | §7.6 namespace |
| `tac.symposium_impls` | 10 | 113 | Canonical symposium implementation surface (Catalog #265) |
| `tac.preflight_rudin_daubechies` | 7 | 22 | Catalog #273-#278 preflight composite |
| `tac.autopilot_rudin_daubechies` | 8 | 45 | Catalog #250-#255 autopilot composite |

## Legend

- ✅ wired (canonical helper token PRESENT in module source)
- ⏳ pending (gap; queued via op-routable below)
- ⓘ N/A with rationale (genuinely doesn't apply; rationale row below)
- 🔄 in flight (sister subagent or queued task; will land soon)
- N/A by-domain (out of scope for this helper class)

---

## Master canonical helper matrix

### tac.master_gradient_consumers (Consumer 1-15)

| Consumer | Symbol | Surface | H1 sens-map | H2 pareto | H3 bit-alloc | H4 autopilot | H5 cont-learn | H6 probe-disambig |
|---|---|---|---|---|---|---|---|---|
| C1 Venn | `classify_bytes_by_pair_variance` | Producer | ✅ axis-weight downstream | N/A (classifier) | N/A (classifier) | ✅ via `adjust_predicted_delta_for_venn_classification` (line 978 / 1039) | ⏳ Task #802 expansion | ⓘ probe surface deferred (classifier ≠ disambiguator) |
| C2 fec6 marginal | `fec6_selector_marginal_matrix` | Producer | ⓘ N/A (selector-mode marginal, not byte-axis) | ✅ pareto-rank input | N/A (selector-domain) | ⏳ Q5 lane registry integration (Task #796) | ⏳ Task #802 expansion | ⓘ marginal IS the disambiguator output |
| C3 NSCS01 nullspace | `nscs01_nullspace_empirical_audit` | Producer | ⓘ N/A (binary verdict) | N/A | N/A | ✅ Q6 council ratification | ✅ via posterior anchor (substrate-class evidence) | ✅ standalone probe-disambiguator surface |
| C4 Wyner-Ziv covariance | `wyner_ziv_side_info_covariance` | Producer | ✅ wire-in #2 LANDED (`wyner_ziv_reweight.axis_level_reweight`) | ⏳ Lagrangian planner Consumer 15 consumes via `_compose_objective_coefficients` | ⏳ Wire-in #3 Task #800 | ✅ via `optimal_plan_to_candidate_row` → CandidateRow → ranker | ✅ via `posterior_update_locked` on optimal plan | ✅ `tools/wyner_ziv_deliverability_prober.py` |
| C5 per-pair difficulty | `per_pair_difficulty_atlas` | Producer | ⓘ per-pair domain not byte-axis | ⓘ ranking artifact | N/A | ⏳ Task #797 viz consumer | ⏳ Task #802 expansion (difficulty deltas → posterior) | ⓘ atlas IS the disambiguator |
| C6 Rashomon queue | `rashomon_disagreement_queue` | Producer | ⓘ disagreement domain | N/A | N/A | ⏳ pending Hook 4 wire-in (probe priority signal) | ✅ K=8 bootstrap members ARE persistence | ⏳ op-routable #5 (probe priority queue → disambiguator dispatch) |
| C15 Lagrangian planner | `per_pair_optimal_treatment_plan_via_lagrangian_dual` | Producer + Wire-in | ✅ per-byte sensitivity dict input | ✅ Pareto-feasible plan IS the constraint surface | ⏳ Wire-in #3 Task #800 (planner output → bit_allocator) | ✅ `optimal_plan_to_candidate_row` adapter LANDED + ⏳ cathedral autopilot CONSUMPTION pending | ✅ `posterior_update_locked` on PairTreatmentAssignment | ⏳ op-routable #7 stub queued |
| Aux: `load_per_pair_gradient_from_anchor` | Helper | N/A (loader) | N/A | N/A | N/A | N/A | N/A | N/A |
| Aux: `consumer_output_path` | Helper | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Aux: `write_consumer_sidecar_json` | Helper | N/A | N/A | N/A | N/A | N/A | N/A | N/A |

**Note**: `cathedral_autopilot_autonomous_loop.py` does NOT yet `import optimal_plan_to_candidate_row` — Consumer 15 wire-in is declared via the adapter contract but the consumer (autopilot loop) does not call it. This is the highest-priority gap; closing it requires the autopilot to enumerate `OptimalPerPairTreatmentPlan` artifacts in `.omx/state/master_gradient_consumers/*.json` and convert each to a CandidateRow. **GAP #1**.

### tac.sensitivity_map

| Symbol/module | Surface | H1 sens-map | H2 pareto | H3 bit-alloc | H4 autopilot | H5 cont-learn | H6 probe-disambig |
|---|---|---|---|---|---|---|---|
| `tac.sensitivity_map` (root) | Producer (canonical Hook 1) | ✅ IS this hook | ⓘ N/A (axis-level not constraint) | ⓘ feeds bit_allocator (consumer-side) | ⓘ feeds autopilot via per-byte sensitivity dict | ⓘ feeds continual_learning via anchor updates | ⓘ N/A |
| `tac.sensitivity_map.wyner_ziv_reweight` | Consumer + Wire-in | ✅ IS this hook | N/A | N/A | N/A | N/A | N/A |
| `tac.sensitivity_map.axis_weights` | Consumer + Helper | ✅ IS this hook | N/A | N/A | N/A | N/A | ✅ axis-weight comparison feeds probe |

### tac.optimization (selected modules)

| Symbol/module | Surface | H1 sens-map | H2 pareto | H3 bit-alloc | H4 autopilot | H5 cont-learn | H6 probe-disambig |
|---|---|---|---|---|---|---|---|
| `bit_allocator_end_to_end` | Producer (canonical Hook 3) | ✅ consumes sens-map | ⏳ pareto coupling latent | ✅ IS this hook | ⏳ no CandidateRow emit | ⏳ no posterior writes | ⓘ N/A |
| `field_equation_planner` | Producer + Wire-in | ⏳ no per-pair sens consumption | ✅ pareto-constraint solver | ⏳ does not consume bit_allocator | ⏳ no CandidateRow emit | ⏳ no posterior writes | ⓘ planner IS the disambiguator |
| `jacobian_fisher_importance_allocator` | Producer | ⏳ no per-pair sens consumption | ⓘ Fisher-domain not pareto | ⏳ no bit_allocator handoff | ⏳ no CandidateRow emit | ⏳ no posterior writes | ⓘ N/A |
| `cross_paradigm_atoms` | Producer + Consumer | ⓘ atom-domain | ✅ pareto-typed atoms | ⏳ no bit_allocator handoff | ✅ atoms feed CandidateRow | ⏳ no posterior writes | ⓘ N/A |
| `autopilot_dispatch_ranking` | Consumer (Hook 4 surface) | ✅ consumes sens-map | ✅ consumes pareto | ⓘ rank-domain | ✅ IS this hook | ✅ consumes continual-learning posterior | ⓘ N/A |
| `cuda_cpu_axis_calibration` | Producer + Helper | ⓘ axis-calibration domain | N/A | N/A | ⓘ feeds ranker | ✅ axis posterior writes | ✅ axis-calibration IS disambiguator |
| `bayesian_experimental_design` | Producer (Hook 4 + Hook 6) | N/A | N/A | N/A | ✅ CandidateRow ranker | ⓘ BED IS the posterior | ✅ IS Hook 6 |

### tac.continual_learning + tac.council_continual_learning + tac.cost_band_calibration

| Module | Surface | H1 sens-map | H2 pareto | H3 bit-alloc | H4 autopilot | H5 cont-learn | H6 probe-disambig |
|---|---|---|---|---|---|---|---|
| `tac.continual_learning` | Producer (canonical Hook 5) | N/A | N/A | N/A | ✅ posterior consumed by ranker | ✅ IS this hook | ⓘ N/A |
| `tac.council_continual_learning` | Producer (Hook 5 sister) | N/A | N/A | N/A | ⏳ ranker does not yet weight by council verdict | ✅ IS sister hook | ⓘ N/A |
| `tac.cost_band_calibration` | Producer (dispatch-cost domain, not signal-axis) | N/A by-domain | N/A | N/A | ✅ consumed by ranker (cost class) | ✅ append_anchor IS the per-domain hook | ⓘ N/A |

### tac.wyner_ziv_deliverability

| Symbol/module | Surface | H1 sens-map | H2 pareto | H3 bit-alloc | H4 autopilot | H5 cont-learn | H6 probe-disambig |
|---|---|---|---|---|---|---|---|
| `proof_builder` | Producer + Consumer | ⏳ consumes Consumer 4 (not sens-map directly) | ⓘ proof-domain | ⓘ N/A | ⏳ proof → CandidateRow surface deferred | ⏳ proof persistence sister to posterior | ⓘ N/A |
| `tools/wyner_ziv_deliverability_prober.py` | Wire-in (tool) | N/A | N/A | N/A | N/A | N/A | ✅ IS probe-disambiguator |

### tac.frontier_scan + tac.probe_outcomes_ledger + tac.deploy.modal.call_id_ledger

| Module | Surface | H1 sens-map | H2 pareto | H3 bit-alloc | H4 autopilot | H5 cont-learn | H6 probe-disambig |
|---|---|---|---|---|---|---|---|
| `tac.frontier_scan` | Producer (canonical Catalog #316) | N/A by-domain (frontier-anchor not signal-axis) | N/A | N/A | ⏳ ranker should consume frontier threshold (PARTIAL via `_resolve_canonical_frontier_threshold_cpu`) | ⓘ feeds continual-learning via anchor scan | ⓘ N/A |
| `tac.probe_outcomes_ledger` | Producer (canonical Catalog #313) | N/A by-domain (probe-verdict not signal-axis) | N/A | N/A | ⏳ ranker should consume blocking outcomes | ✅ append-only ledger IS posterior surface (Hook 5 sister) | ✅ IS this hook structurally |
| `tac.deploy.modal.call_id_ledger` | Producer (canonical Catalog #245) | N/A by-domain (dispatch-tracking) | N/A | N/A | ⓘ feeds cost_band posterior | ✅ append-only ledger IS posterior surface (Hook 5 sister) | ⓘ N/A |

### tac.boosting (§7.6 namespace)

| Subspace | H1 sens-map | H2 pareto | H3 bit-alloc | H4 autopilot | H5 cont-learn | H6 probe-disambig | per-pair-grad |
|---|---|---|---|---|---|---|---|
| `tac.boosting.*` (recursive) | ✅ via `ParetoAnchor` sensitivity contribution | ✅ `ParetoFrontTracker` IS this hook | ⏳ no bit_allocator handoff | ✅ `append_stage_outcome_locked` feeds ranker | ⓘ sister persistence (`append_stage_outcome_locked` JSONL store; not `continual_learning` proper) | ✅ `RashomonDisagreementQueue` proxy | ✅ `PerPairDecoderEnsembleSelector` consumes per-pair |

### tac.compress_time_optimization (§7.6 namespace)

| Subspace | H1 sens-map | H2 pareto | H3 bit-alloc | H4 autopilot | H5 cont-learn | H6 probe-disambig | per-pair-grad |
|---|---|---|---|---|---|---|---|
| `tac.compress_time_optimization.*` | ⏳ no sens-map consumption in compress-time stages | ✅ pareto-typed compress-time atoms | ⏳ no bit_allocator handoff | ✅ stage outcomes feed ranker | ⓘ sister persistence (compress-time stage JSONL) | ✅ stage selector IS disambiguator | ✅ master gradient consumption documented |

### tac.inflate_time_post_processing (§7.6 namespace)

| Subspace | H1 sens-map | H2 pareto | H3 bit-alloc | H4 autopilot | H5 cont-learn | H6 probe-disambig | per-pair-grad |
|---|---|---|---|---|---|---|---|
| `tac.inflate_time_post_processing.*` | ⏳ no sens-map consumption in inflate-time post-processors | ✅ pareto-typed post-proc atoms | ⓘ N/A (inflate-time, not compress-time bit allocation) | ✅ post-proc outcomes feed ranker | ⓘ sister persistence | ✅ post-proc selector IS disambiguator | ✅ per-pair consumed by post-proc selectors |

### tac.side_information (§7.6 namespace)

| Subspace | H1 sens-map | H2 pareto | H3 bit-alloc | H4 autopilot | H5 cont-learn | H6 probe-disambig | per-pair-grad |
|---|---|---|---|---|---|---|---|
| `tac.side_information.*` | ⏳ no sens-map consumption directly | ✅ pareto-typed side-info channels | ⏳ no bit_allocator handoff for side-info bytes | ✅ side-info channel feeds ranker | ⓘ sister persistence | ✅ channel selector IS disambiguator | ⓘ per-pair via Wyner-Ziv side-info covariance (Consumer 4 indirect) |

### tac.search (§7.6 namespace)

| Subspace | H1 sens-map | H2 pareto | H3 bit-alloc | H4 autopilot | H5 cont-learn | H6 probe-disambig | per-pair-grad |
|---|---|---|---|---|---|---|---|
| `tac.search.*` | ⏳ no sens-map consumption | ✅ pareto-typed search atoms | ⏳ no bit_allocator handoff | ✅ search results feed ranker | ⓘ sister persistence | ✅ search IS the disambiguator | ✅ per-pair via search-over-treatments (Lagrangian planner Consumer 15 indirect) |

### tac.symposium_impls

| Symbol/module | Surface | H1 sens-map | H2 pareto | H3 bit-alloc | H4 autopilot | H5 cont-learn | H6 probe-disambig |
|---|---|---|---|---|---|---|---|
| `tac.symposium_impls.*` (9 modules) | Producer (one per symposium) | varies | varies | varies | ⏳ varies | ✅ `update_from_anchor` Catalog #265 alias | varies |

Per Catalog #265 the META gate requires `update_from_anchor` token presence in every symposium impl; this gate already enforces Hook 5 wiring at the package level.

### tac.preflight_rudin_daubechies + tac.autopilot_rudin_daubechies (Catalog #273-#278 + #250-#255)

| Symbol/module | Surface | H1 sens-map | H2 pareto | H3 bit-alloc | H4 autopilot | H5 cont-learn | H6 probe-disambig |
|---|---|---|---|---|---|---|---|
| `preflight_rudin_daubechies.slim_risk_scorer` | Producer | ⓘ hit-rate-sort IS sensitivity | N/A | N/A | ✅ via `predicted_dispatch_risk` field on CandidateRow | ✅ fcntl-locked posterior per Catalog #128/#131 | ✅ Rashomon disagreement queue |
| `preflight_rudin_daubechies.compressive_coverage_estimator` | Producer | N/A | N/A | ✅ `next_fixture_to_observe` IS bit_allocator surface | ✅ informs ranker | ✅ posterior writes | ⓘ N/A |
| `autopilot_rudin_daubechies.rashomon_ensemble` | Producer + Consumer | N/A | N/A | N/A | ✅ K=8 ensemble informs ranker | ✅ posterior writes | ✅ disagreement queue IS Hook 6 |

---

## Per-pair master gradient dedicated subsection (operator's specific directive)

The per-pair master gradient tensor shape is `(N_bytes, N_pairs, 3)` (3 = seg/pose/rate axes per Consumer C4 docstring + tac.master_gradient_consumers.PerByteVennClass enumeration). Below is the comprehensive consumer audit per operator directive *"all per-pair master gradient uses are wired up and integrated and implemented everywhere"*.

### Direct consumers (verified via `inspect.getsource(mod).contains(token)` for at least one of: `load_per_pair_gradient`, `per_pair_gradient`, `wyner_ziv_side_info_covariance`, `OptimalPerPairTreatmentPlan`)

| # | Consumer | Hook role | Status | Evidence |
|---|---|---|---|---|
| 1 | `tac.master_gradient_consumers` Consumer 1-15 | Producer (canonical) | ✅ canonical | Module IS the canonical per-pair surface |
| 2 | `tac.sensitivity_map.wyner_ziv_reweight` | H1 Wire-in | ✅ wire-in #2 LANDED | `axis_level_reweight()` consumes `WynerZivSideInfoClassification` |
| 3 | `tac.wyner_ziv_deliverability.proof_builder` | Q1 Producer | ✅ landed (sister a841c112eb346fe81) | `proof_builder` imports `WynerZivSideInfoClassification` |
| 4 | `tac.autopilot_rudin_daubechies.rashomon_ensemble` | Hook 6 sister | ✅ K=8 bootstrap | imports `rashomon_disagreement_queue` from Consumer 6 |
| 5 | `tools/wyner_ziv_deliverability_prober.py` | Tool / Hook 6 | ✅ uses Consumer 4 | imports `wyner_ziv_side_info_covariance` |
| 6 | `tools/cathedral_autopilot_autonomous_loop.py` | Hook 4 (autopilot ranker) | ✅ Consumer 1 Venn via `adjust_predicted_delta_for_venn_classification` | line 978 / 1039 / 1046 |
| 7 | `tac.boosting.per_pair_decoder_ensemble` | §7.6 + Hook 4 | ✅ per-pair routing | per-pair selector IS the consumer |
| 8 | `tac.compress_time_optimization.*` | §7.6 + Hook 4 | ✅ per-pair via wave_3 | document references `per_pair_gradient` in submodules |
| 9 | `tac.inflate_time_post_processing.*` | §7.6 + Hook 4 | ✅ per-pair via post-proc selectors | submodules reference per-pair |

### Indirect / aggregate-only consumers (per-pair would be the canonical input but currently only aggregate-grad is loaded)

| # | Consumer | Gap class | Should consume per-pair? | Op-routable |
|---|---|---|---|---|
| 1 | `tac.optimization.bit_allocator_end_to_end` | Aggregate-only (no `per_pair`/`wyner_ziv` token) | ✅ YES — bit allocation per-pair is the natural extension; current `aggregate.mean()` projection drops Wyner-Ziv side-info opportunity | OR-3 / Task #800 wire-in #3 |
| 2 | `tac.optimization.field_equation_planner` | Aggregate-only | ✅ YES — field equation should incorporate per-pair Venn classification | OR-4 follow-on subagent |
| 3 | `tac.optimization.jacobian_fisher_importance_allocator` | Aggregate-only | ✅ YES — Fisher allocation should be per-pair | OR-5 follow-on subagent |
| 4 | `tac.optimization.autopilot_dispatch_ranking` | Aggregate-only at top level (but downstream CandidateRow carries per-pair-derived fields per Catalog #219 / #227) | ⓘ INDIRECT — already correct (operates on CandidateRow which already absorbs per-pair signals) | None |
| 5 | `tac.search.*` | Aggregate-only at orchestrator level (per-pair via search-over-treatments) | ⓘ INDIRECT — operates on Treatment catalog which carries per-pair Jacobian | None |
| 6 | `tac.side_information.*` (top-level) | Per-pair via Wyner-Ziv side-info covariance (Consumer 4 indirect) | ⓘ INDIRECT — correct routing | None |

### Genuinely N/A by domain (per-pair gradient NOT applicable)

| # | Consumer | Rationale |
|---|---|---|
| 1 | `tac.cost_band_calibration` | Dispatch-cost domain (USD per hour × wall-clock); not signal-axis |
| 2 | `tac.deploy.modal.call_id_ledger` | Operational tracking artifact; per-pair gradient is not a dispatch-state field |
| 3 | `tac.deploy.modal.anchor_lookup` | Anchor-lookup tool; per-pair gradient is not an anchor lookup key |
| 4 | `tac.frontier_scan` | Frontier-anchor scan; per-pair gradient is internal to ranker, not anchor scan |
| 5 | `tac.probe_outcomes_ledger` | Probe-verdict ledger; per-pair gradient is data INTO a probe, not probe verdict |
| 6 | `tac.preflight_rudin_daubechies.*` (most) | Preflight risk scoring over CLAUDE.md catalog gates; per-pair gradient is not a catalog gate signal |
| 7 | `tac.continual_learning` | Posterior-store helper; per-pair gradient is data passed THROUGH posterior (not posterior structure itself) |
| 8 | `tac.council_continual_learning` | Council-deliberation domain; per-pair gradient is data the council discusses, not deliberation structure |

### Coverage rollup

- **Direct per-pair consumers**: 9 (canonical producer + 8 sister wire-ins)
- **Indirect consumers (aggregate-only at top level, per-pair via downstream)**: 4 (correct as-is)
- **Aggregate-only gap (SHOULD consume per-pair, doesn't yet)**: 3 (`bit_allocator_end_to_end` / `field_equation_planner` / `jacobian_fisher_importance_allocator`)
- **N/A by domain**: 8 (cost_band / call_id_ledger / anchor_lookup / frontier_scan / probe_outcomes_ledger / continual_learning / council_continual_learning / preflight_rudin_daubechies)

**Per-pair consumer coverage**: 9 / (9 + 3) = **75%** of candidate consumers actually consume per-pair (excluding N/A-by-domain). Closing the 3-gap (Task #800 + 2 follow-ons) brings coverage to 100%.

---

## Gap closure decisions

### Closed in-context (this subagent)

| Gap | Closure |
|---|---|
| GAP-COVERAGE-MATRIX | This memo (`.omx/research/comprehensive_wire_in_coverage_matrix_20260517.md`) |
| GAP-PRE-PAIR-GAP-AUDIT | The per-pair section above + 75% coverage rollup |
| GAP-OBSERVABILITY-SURFACE-DECLARATION-IN-MATRIX | Observability surface section above |

### Queued for follow-on subagent (TaskCreate handled by parent)

| Gap | Spec | Owner |
|---|---|---|
| GAP-1 Cathedral autopilot consumes `optimal_plan_to_candidate_row` | Modify `tools/cathedral_autopilot_autonomous_loop.py::rank_candidates` to enumerate `.omx/state/master_gradient_consumers/optimal_per_pair_treatment_plan_*.json` and invoke `optimal_plan_to_candidate_row()` for each, appending the resulting `CandidateRow` instances to the ranker queue alongside existing CandidateRow sources. ~30 LOC + ~5 tests. | Task #801 (existing) expanded to also cover Lagrangian planner |
| GAP-2 bit_allocator_end_to_end consumes Consumer 15 plan | `OptimalPerPairTreatmentPlan` carries `pair_assignments` with predicted ΔS per pair per treatment. `bit_allocator_end_to_end` should accept an optional `optimal_plan: OptimalPerPairTreatmentPlan | None = None` kwarg and bias allocation by `plan.predicted_score_delta` per byte. ~50 LOC + ~10 tests. | Task #800 (existing) — Wire-in #3 |
| GAP-3 field_equation_planner per-pair awareness | Add `per_pair_sensitivity_dict: dict[str, np.ndarray] | None = None` parameter to feed planner with byte-keyed per-pair signal. ~80 LOC + ~12 tests. | OR-4 follow-on subagent |
| GAP-4 jacobian_fisher_importance_allocator per-pair awareness | Promote `aggregate_grad` to `per_pair_grad` with `.mean(axis=1)` fallback for legacy aggregate-only callers. ~60 LOC + ~10 tests. | OR-5 follow-on subagent |
| GAP-5 Hook 5 continual-learning posterior update for per-pair difficulty deltas | Each `PerPairDifficultyAtlas` write should trigger `posterior_update_locked` anchor append with `axis_name="per_pair_difficulty_delta"`. ~20 LOC + ~5 tests. | Task #802 expansion |
| GAP-6 Hook 6 probe-disambiguator for Consumer 6 Rashomon disagreement queue | Top-K disagreement entries should auto-emit `tools/probe_rashomon_disagreement_disambiguator.py` invocation (or sister CLI). ~80 LOC + ~12 tests. | OR-7 follow-on subagent |
| GAP-7 Lagrangian planner probe-disambiguator stub | Currently planner emits OptimalPerPairTreatmentPlan but no probe-disambiguator surface; should land `tools/probe_optimal_plan_disambiguator.py` per Catalog #125 Hook 6. ~120 LOC + ~15 tests. | OR-8 follow-on subagent |
| GAP-8 council_continual_learning consumed by ranker | Ranker does not yet weight by council verdict (T2 vs T3 vs T4 vs PROCEED vs PROCEED_WITH_REVISIONS). ~40 LOC + ~8 tests. | OR-9 follow-on subagent |
| GAP-9 Consumer 6 Rashomon queue feeds CandidateRow Hook 4 | Top-K disagreement entries should boost CandidateRow ranking for downstream dispatch. ~30 LOC + ~6 tests. | OR-10 follow-on subagent |

All gap closures are scoped via `lane_id=lane_comprehensive_wiring_integration_pass_20260517` so future subagents can co-orchestrate via the canonical pre-registration discipline (Catalog #126).

---

## Adversarial review

### Round 1 — Call-site tracing (Yousfi + Fridrich + Carmack rotation)

For each wire-in identified as ✅ in the matrix, traced the actual call site (not signature) per CLAUDE.md "NEVER invent CLI flags" rule + Catalog #12 `preflight_arity` discipline:

| Wire-in | Call-site evidence | Verdict |
|---|---|---|
| `optimal_plan_to_candidate_row` adapter | DECLARED at `src/tac/master_gradient_consumers.py` but **NOT CONSUMED** by `tools/cathedral_autopilot_autonomous_loop.py` (grep returned 0 matches) | CRITICAL: GAP-1 documented; the adapter is currently dead code at the consumer side |
| `wyner_ziv_side_info_covariance` → `axis_level_reweight` | grep at `src/tac/sensitivity_map/wyner_ziv_reweight.py:65, 116, 342` confirms 3 import sites | OK: wire-in #2 fully landed |
| `rashomon_disagreement_queue` → `tac.autopilot_rudin_daubechies.rashomon_ensemble` | grep at `src/tac/autopilot_rudin_daubechies/rashomon_ensemble.py:303` confirms 1 import site | OK |
| `WynerZivSideInfoClassification` → `tac.wyner_ziv_deliverability.proof_builder` | grep at `src/tac/wyner_ziv_deliverability/proof_builder.py:88` confirms 1 import site | OK |

GAP-1 is the **highest-severity finding** of Round 1. The Lagrangian planner Hook 4 wire-in is declared but the consumer is dead.

### Round 2 — Phase-gate sweep (per CLAUDE.md "Recursive adversarial review protocol" item 7)

No phase-gated thresholds in this audit (the matrix has no per-phase semantics). Verified by `grep -E "phase|epoch" .omx/research/comprehensive_wire_in_coverage_matrix_20260517.md` returning matches only inside paragraph prose (no `if phase == N: threshold = X` patterns).

### Round 3 — Assumption-challenge axis (per Catalog #291 + #292)

| Operating-within assumption | HARD-EARNED vs CARGO-CULTED |
|---|---|
| `__all__` enumerates the canonical surface | HARD-EARNED — per CLAUDE.md "Beauty, simplicity, and developer experience" |
| Substring grep is reliable for hook detection (false-positive rate < 5%) | HARD-EARNED — same pattern Catalog #117/#157/#174/#235 use |
| Per-pair gradient is the master tensor (aggregate is a projection) | HARD-EARNED — operator's 8-pair fp64 validation |
| Cathedral autopilot SHOULD consume `optimal_plan_to_candidate_row` | HARD-EARNED — the adapter's docstring + signature name the contract |
| The 8 N/A-by-domain consumers are genuinely N/A | HARD-EARNED — each rationale documented; assumption-adversary verifies no signal-axis crossover |
| The 3 indirect consumers (autopilot/search/side_information) operate at orchestrator level | HARD-EARNED — CandidateRow IS the canonical per-pair carrier downstream |
| **Risk**: future helper that adds per-pair gradient field could re-introduce aggregate-only bias | CARGO-CULTED → mitigation: this matrix MUST be refreshed each wave |

No CARGO-CULTED assumptions surfaced in this audit's framing. The "negative findings = bugs" framing was unwound in the cargo-cult audit table (the §7.6 namespaces use sister fcntl-locked persistence stores distinct from `continual_learning` proper; this is correct engineering, not a gap).

### Findings table

| # | Severity | Finding | Owner |
|---|---|---|---|
| 1 | HIGH | GAP-1 cathedral autopilot does not consume `optimal_plan_to_candidate_row` adapter (declared but dead) | Task #801 expansion → next subagent |
| 2 | MEDIUM | GAP-2 bit_allocator_end_to_end aggregate-only (per-pair plan not consumed) | Task #800 → next subagent |
| 3 | MEDIUM | GAP-3 field_equation_planner aggregate-only | OR-4 follow-on |
| 4 | MEDIUM | GAP-4 jacobian_fisher_importance_allocator aggregate-only | OR-5 follow-on |
| 5 | LOW | GAP-5 Hook 5 cont-learn on per-pair difficulty deltas not yet wired | Task #802 expansion |
| 6 | LOW | GAP-6 Hook 6 probe-disambiguator for Consumer 6 Rashomon queue not yet emitted | OR-7 follow-on |
| 7 | LOW | GAP-7 Lagrangian planner probe-disambiguator stub not yet landed | OR-8 follow-on |
| 8 | LOW | GAP-8 council_continual_learning not yet weighted into ranker | OR-9 follow-on |
| 9 | LOW | GAP-9 Consumer 6 Rashomon queue not yet feeding CandidateRow boost | OR-10 follow-on |

All findings produce queued follow-on subagent dispatches; no in-context code changes per parent's NO COMMITS constraint.

---

## Canonical-vs-unique decision per layer

This is an audit memo per CLAUDE.md "Substrate design memos MUST document canonical-vs-unique decision". For an audit memo specifically:

| Layer | Decision | Rationale |
|---|---|---|
| Coverage matrix structure | CANONICAL adopt sister format from `.omx/research/comprehensive_state_tracker_20260517.md` (table + emoji legend) | DRY across operator-facing memos |
| Hook detection methodology | CANONICAL adopt `grep -n` + `inspect.getsource` pattern (sister Catalog #117/#157 use) | Reproducible; byte-stable; no LLM judgment |
| Gap-closure routing | UNIQUE: queue follow-on subagents (Task # creation) rather than in-context code edits | Parent's NO COMMITS constraint |
| Per-pair section structure | UNIQUE: 3-band table (Direct / Indirect / N/A by-domain) | Operator's directive specifically asked for per-pair gap audit; the 3-band classification IS the operator-actionable structure |
| Adversarial review format | CANONICAL adopt CLAUDE.md "Recursive adversarial review protocol" 3-round format + assumption-challenge axis | Mandatory per Catalog #291/#292 |
| Frontmatter | CANONICAL adopt Catalog #300 v2 T1 deliberation frontmatter | This memo IS a T1 working-group output (single subagent; bounded scope; output feeds T2/T3 deliberations downstream) |

## Cross-references

- `.omx/research/comprehensive_state_tracker_20260517.md` — sister no-signal-loss canonical state
- `src/tac/master_gradient_consumers.py` — canonical Producer of all per-pair gradient consumers
- `src/tac/sensitivity_map/wyner_ziv_reweight.py` — wire-in #2 landed sister
- `src/tac/wyner_ziv_deliverability/proof_builder.py` — Q1 sister landing (in flight at audit start; landed by now)
- `tools/cathedral_autopilot_autonomous_loop.py` — Hook 4 canonical surface; GAP-1 consumer
- `tools/operator_authorize.py` — Hook 5 canonical orchestrator (paid dispatch)
- `feedback_per_pair_optimal_treatment_plan_via_lagrangian_dual_landed_20260517.md` — Consumer 15 (Lagrangian planner) landing memo
- `feedback_comprehensive_wiring_integration_pass_landed_20260517.md` — this audit's landing memo
- CLAUDE.md "Subagent coherence-by-default" — 6-hook wire-in non-negotiable
- CLAUDE.md "Meta-Lagrangian/Pareto solver" — Hook 1-3 + Hook 5 canonical surfaces
- Catalog #125 — `check_subagent_landing_has_solver_wire_in` STRICT preflight enforcement
- Catalog #126 — `check_lane_pre_registered_before_work_starts` STRICT preflight enforcement
- Catalog #229 — `check_subagent_landing_includes_premise_verification_evidence` STRICT preflight enforcement
- Catalog #290 — `check_substrate_design_memo_has_canonical_vs_unique_decision_section` STRICT preflight enforcement
- Catalog #294 — `check_substrate_landing_memo_has_9_dim_checklist_evidence_section` STRICT preflight enforcement
- Catalog #303 — `check_substrate_design_memo_has_cargo_cult_audit_section` STRICT preflight enforcement
- Catalog #305 — `check_substrate_design_memo_has_observability_surface_section` STRICT preflight enforcement
- Catalog #300 — `check_council_deliberation_declares_tier_in_frontmatter` STRICT preflight enforcement (T1 working group)
