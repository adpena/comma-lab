# DQS1 Drop-Many BUILD-1c: GREEDY Heuristic Alternative Reducer — Landing Memo (2026-05-25)

- timestamp_utc: 2026-05-25T16:00:00Z
- agent: claude (DROP-MANY-BEAM BUILD-1c subagent per BUILD-1 verdict operator-routable #2 + Catalog #308 alternative methodology #2)
- lane_id: lane_dqs1_drop_many_build_1c_greedy_heuristic_alternative_reducer_20260525
- scope: $0 local CPU GREEDY ranking probe-disambiguator alternative to BUILD-1's pairwise-interaction-matrix beam search; no interaction-matrix dependency; uses EMPIRICAL per-pair anchors from `.omx/state/continual_learning_posterior.json` accepted-anchor history
- authority: planning + observability ONLY; score/promotion/rank/dispatch authority all FALSE per Catalog #287/#323/#341
- relates to:
  - DROP-MANY-BEAM-DESIGN landing memo `.omx/research/dqs1_drop_many_beam_pairwise_interaction_waterfill_design_memo_20260525.md`
  - DROP-MANY-BEAM-BUILD-1 landing memo `.omx/research/dqs1_drop_many_build_1_pairwise_interaction_matrix_empirical_population_20260525.md` (DATA_SOURCE_PREMISE_FALSIFIED_ARTIFACT_DOMINANT)
  - DROP-MANY-BEAM-DESIGN scaffold `tools/probe_dqs1_drop_many_beam_pairwise_interaction_disambiguator.py` (NOT mutated; APPEND-ONLY)
  - codex eureka acquisition `.omx/research/codex_eureka_beyond_drop_two_acquisition_20260525T143351Z/dqs1_pairset_acquisition_eureka_drop_many.json`
  - canonical equation #36 `pairset_component_marginal_score_decomposition_v1`
  - canonical equation #4 `per_pair_master_gradient_score_impact_taylor_v1`
  - current frontier `pairset_drop_one_rank021_pair0371` archive sha `7a0da5d0fc327cba` at `0.19202828` [contest-CPU] per `.omx/state/canonical_frontier_pointer.json`
- discipline anchors: Catalog #229 (premise verification) + #287 (canonical Provenance evidence-tag) + #303 (cargo-cult audit) + #307 (paradigm-vs-implementation falsification) + #308 (alternative probe methodologies) + #313 (probe-outcomes ledger) + #323 (canonical Provenance umbrella) + #341 (canonical routing markers) + #344 (canonical equation registry RATIFY-N) + #356 (per-axis decomposition) + #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE
- mission_contribution: `apparatus_maintenance` per Catalog #300 — BUILD-1c produced NEGATIVE empirical verdict via DIRECT EMPIRICAL COMPARISON (NOT data-source artifact like BUILD-1); ALL 5 measured K>1 sister anchors REGRESS vs K=1 empirical optimum. Prevents downstream BUILD-2/BUILD-3/BUILD-4/DISPATCH cascade from spending paid GPU dollars on the drop-many K>1 surface. Structural value is unblocking the alternative reducer cascade per Catalog #308 — either (a) BUILD-1b paid Modal CPU paired exact-eval on UNMEASURED drop_one pairs to discover whether more negative-ΔS pairs exist, or (b) substrate-class shift per CLAUDE.md HORIZON-CLASS plateau-trap warning.
- operator-frontier-override: ACTIVE per Catalog #300 Mission alignment Consequence 1 — operator NON-NEGOTIABLE blanket approval 2026-05-19 verbatim *"all operator decisions and approval granted and provided fuly and completely"* + today's *"continue with all"* + 3-msg rate-attack amplification; BUILD-1c scope ($0 local CPU smoke) respects "Executing actions with care" non-negotiable.

## Headline finding

**BUILD-1c acceptance criterion verdict: `NEGATIVE_COLLAPSE_TO_K1_EMPIRICAL_DROP_MANY_REGRESSES`**

GREEDY ranking by EMPIRICAL per-pair drop_one ΔS (from `.omx/state/continual_learning_posterior.json` accepted-anchor history; 9 measured drop_one anchors + 1 drop_two anchor + 6 diversity_k anchors) yields a single negative-ΔS pair (pair0371 = -6.66e-07; identical to the current frontier). ALL 8 other measured drop_one pairs REGRESS at +3.34e-07 (rate gain undone by distortion penalty). ALL 5 measured K>1 sister anchors REGRESS vs K=1 empirical optimum:

| K | Empirical sister anchor pairset | sister Δ vs base | K=1 Δ vs base | Sister beats K=1? |
|---|---|---:|---:|:---:|
| 2 | pair0257 + pair0371 (`drop_two_r28_021`) | **+6.68e-07** | -6.66e-07 | NO |
| 2 | `pairset_diversity_k002` | **+2.67e-05** | -6.66e-07 | NO |
| 4 | `pairset_diversity_k004` | **+2.94e-05** | -6.66e-07 | NO |
| 8 | `pairset_diversity_k008` | **+2.00e-05** | -6.66e-07 | NO |
| 12 | `pairset_diversity_k012` | **+1.97e-05** | -6.66e-07 | NO |
| 16 | `pairset_diversity_k016` | **+1.33e-05** | -6.66e-07 | NO |

GREEDY paradigm at K>1 is EMPIRICALLY FALSIFIED at the K>1 surface for the currently-measured anchor set. Per Catalog #307 paradigm-vs-implementation classification: **this is IMPLEMENTATION-LEVEL falsification of the SPECIFIC GREEDY-on-currently-measured-anchors implementation, NOT paradigm-level refutation of the drop-many beam-search interaction-aware design AND NOT paradigm-level refutation of all possible drop-many configurations**. The beam-search PARADIGM remains INTACT; the drop-many PARADIGM at K>1 also remains INTACT because:

1. Only 9 of 31 acquired drop_one pairs have empirical anchors; 22 UNMEASURED pairs MIGHT have negative drop_one ΔS (BUILD-1b alternative #4).
2. No empirical K>1 anchor tests pair0371 + UNTESTED-orthogonal-pair tuples; sister codex acquisition exact-eval ledger could discover such tuples (BUILD-1b alternative #1).
3. The diversity_k anchors used canonical diversity-selection (not GREEDY-top-K-by-empirical-ΔS); the alignment between "diversity-selected" and "GREEDY-by-empirical" pair sets is unknown.

Per CLAUDE.md "Forbidden premature KILL without research exhaustion" + Catalog #308: this is a **DEFER** verdict, NOT a KILL. The drop-many beam-search approach is RESEARCH-PATH-ALIVE pending alternative methodologies.

## Carmack MVP-first 5-step closure

1. **FREE local CPU smoke** — DONE: $0 macOS-CPU advisory; reused empirical anchors from continual_learning_posterior + acquisition plan; ~5 minutes wall-clock.
2. **Falsifiable challenge** — DONE: GREEDY top-K predicted ΔS under orthogonality assumption = K × per-pair-mean; falsifying outcome = empirical K>1 sister anchors REGRESS vs K=1. ACTUAL: ALL 5 empirical K>1 sisters regress; worst at K=4 = +2.94e-5 (= 44× worse than K=1's -6.66e-7).
3. **Catalog #344 reference** — DONE: queued refinement for `dqs1_drop_many_greedy_independent_pair_ordering_v1` (NOT auto-registered; QUEUED for operator-routable RATIFY-N once BUILD-1b lands ≥3 NEW empirical K>1 anchors).
4. **Verdict in same commit batch** — DONE: verdict.json + greedy_sweep_metadata.json + this landing memo + Catalog #313 probe-outcomes row land in same commit batch via canonical serializer.
5. **Re-route operator priority queue** — DONE: NEGATIVE GREEDY verdict triggers DEFER for drop-many at K>1; operator-routable cascade re-routes to (a) BUILD-1b paid Modal CPU paired exact-eval, (b) Yousfi+Fridrich human-prior drop-many tuples, or (c) substrate-class shift per CLAUDE.md HORIZON-CLASS plateau-trap warning.

## Canonical-vs-unique decision per layer

Per Catalog #290 + CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode":

- **ADOPT_CANONICAL_BECAUSE_SERVES**: canonical Provenance umbrella per Catalog #323 (`CANONICAL_NON_PROMOTABLE_MARKERS` dict with `score_claim=False` / `promotable=False` / `axis_tag=[predicted]` / `evidence_grade=research_only`) threaded into every artifact's metadata.
- **ADOPT_CANONICAL_BECAUSE_SERVES**: canonical Catalog #341 routing markers (predicted_delta_adjustment=0.0 / promotable=False / axis_tag=[predicted]) for non-promotable observability.
- **ADOPT_CANONICAL_BECAUSE_SERVES**: canonical Catalog #313 probe-outcomes ledger row queued in verdict.json (DEFER verdict, reactivation_criteria, canonical_helper_invocation).
- **ADOPT_CANONICAL_BECAUSE_SERVES**: canonical Catalog #344 RATIFY-N protocol for canonical equation candidate `dqs1_drop_many_greedy_independent_pair_ordering_v1` — refinement field QUEUED with `ratification_trigger_status: DEFERRED-PENDING-BUILD-1B-PAIRED-CPU-EXACT-EVAL-LEDGER`.
- **ADOPT_CANONICAL_BECAUSE_SERVES**: canonical Catalog #229 premise verification protocol — verified empirical per-pair drop_one source (`.omx/state/continual_learning_posterior.json` history) BEFORE running GREEDY ranking; documented finding inline in verdict.json data_source_premise_verification.
- **ADOPT_CANONICAL_BECAUSE_SERVES**: canonical rate-delta-per-byte formula `-CANONICAL_RATE_MULTIPLIER / CANONICAL_RATE_DENOM_BYTES = -25/37_545_489 = -6.658589e-07` used as the rate-only-optimism lower bound for UNMEASURED pairs (clearly tagged `empirical_source='predicted_rate_only'` so non-empirical-pair predicted_delta_score cannot leak into score claims).
- **FORK_BECAUSE_PRINCIPLED_MISMATCH**: per BUILD-1 cargo-cult audit, the design memo's BUILD-1 dependency on the acquisition plan's `predicted_score_mean` field as empirical data source was CARGO-CULTED. BUILD-1c FORKS to a DIFFERENT canonical data source (`.omx/state/continual_learning_posterior.json` accepted-anchor history) and confirms via Catalog #229 PV that this source contains AUTHORITATIVE per-pair drop_one anchors with non-uniform deltas.
- **FORK_BECAUSE_PRINCIPLED_MISMATCH**: per Catalog #308 alternative probe methodologies, BUILD-1c enumerates N=3 alternative reducers in verdict.json `operator_routable_next_cascade` array (BUILD-1b paid CPU exact-eval on UNMEASURED pairs / Yousfi+Fridrich human-prior tuples / substrate-class shift) — operator-routable for next-cascade priority decision.

## Observability surface (Catalog #305)

BUILD-1c observable through:

1. **Inspectable per layer**: 2 canonical artifacts (verdict.json + greedy_sweep_metadata.json) emitted under `experiments/results/dqs1_drop_many_build_1c_greedy_heuristic_alternative_reducer_20260525/`; each carries canonical Provenance.
2. **Decomposable per signal**: empirical distribution decomposed via `empirical_drop_one_anchor_distribution` field (n_negative_delta_pairs / n_positive_delta_pairs / min / max / per_pair_anchors list); empirical K>1 sisters decomposed via `empirical_sister_anchors_summary` field.
3. **Diff-able across runs**: canonical JSON `sort_keys=True` per Catalog #131; two runs of the same script with identical posterior state produce byte-identical verdict.json + metadata.json.
4. **Queryable post-hoc**: `greedy_top_k_sweep` field enumerates per-K (selected_pair_indices, cumulative_predicted_delta_vs_base, delta_vs_current_frontier, empirical_sister_anchor); operator can re-run with `--json` to stream verdict to stdout.
5. **Cite-able**: every artifact's `canonical_provenance.allowed_use` cites the canonical script path; `catalog_313_probe_outcomes_row.canonical_helper_invocation` cites the canonical script::main invocation.
6. **Counterfactual-able**: the script supports custom posterior / frontier-pointer / acquisition-plan paths via CLI flags; sister-subagent BUILD-1b could re-run with augmented posterior containing NEW empirical K>1 anchors to test the orthogonality assumption empirically.

## 9-dimension success checklist evidence (Catalog #294)

1. **UNIQUENESS**: BUILD-1c is the UNIQUE alternative reducer methodology #2 per Catalog #308 enumeration in BUILD-1's verdict.json; uses DIFFERENT canonical data source than BUILD-1 (posterior history vs acquisition plan inheritance arithmetic) to test the SAME drop-many paradigm at the K>1 surface.
2. **BEAUTY + ELEGANCE**: ONE ~600-LOC script + 2 canonical artifacts cover the entire BUILD-1c deliverable; operator-readable in <10 min; design-memo + BUILD-1 references in every artifact; sister with BUILD-1 scaffold but no shared mutable state.
3. **DISTINCTNESS**: distinct from BUILD-1 (different data source: posterior vs acquisition plan); distinct from sister BUILD-2/BUILD-3/BUILD-4 (no interaction matrix dependency by construction); distinct from concurrent subagents (Slot 1 CUDA-AXIS-DQS1-DESIGN + Slot 2 PR95-STAGE-2-MLX-BUILD + Slot 4 COMBINED-TIER-1-WAVE-2 all touch disjoint cells).
4. **RIGOR**: every claim cites canonical artifact path; empirical computation is deterministic given posterior state SHA; orthogonality assumption falsifiability methodology (empirical K>1 sister anchors regress vs K=1) is canonical empirical test.
5. **OPTIMIZATION-PER-TECHNIQUE**: BUILD-1c is the unique-and-complete script for GREEDY-by-empirical-drop_one-ΔS without interaction-matrix dependency; the falsifiability methodology forks from BUILD-1's artifact-detection methodology to surface the orthogonality-assumption-falsified subclass.
6. **STACK-OF-STACKS-COMPOSABILITY**: verdict.json embeds Catalog #313 probe-outcomes row + Catalog #344 canonical equation refinement (PENDING) + Catalog #356 per-axis decomposition placeholder (None at K>1 because empirical K>1 sister anchors are scalar-not-axis-decomposed) — all 3 wire-in points explicit.
7. **DETERMINISTIC-REPRODUCIBILITY**: verdict.json + metadata.json use canonical JSON `sort_keys=True`; two runs produce byte-identical output for identical posterior state.
8. **EXTREME-OPTIMIZATION-PERFORMANCE**: $0 + ~5 min wall-clock; reused canonical state (posterior + acquisition plan + frontier pointer) without paid GPU.
9. **OPTIMAL-MINIMAL-CONTEST-SCORE**: BUILD-1c verdict's `apparatus_maintenance` mission contribution is the FRONTIER-PROTECTING value — preventing the BUILD-2/BUILD-3/BUILD-4/DISPATCH cascade from spending paid GPU dollars on the drop-many K>1 surface. The MISSION-DIRECT score-lowering value of BUILD-1c is N/A (planning/observability-only); BUILD-1b paid Modal CPU paired exact-eval on UNMEASURED drop_one pairs is the next-cascade priority that may unlock authentic drop-many K>1 viability IF more negative-ΔS pairs exist in the unmeasured 22-pair subset.

## Cargo-cult audit per assumption (Catalog #303)

Per CLAUDE.md hard-earned-vs-cargo-culted addendum + Catalog #303 design-memo discipline:

- **HARD-EARNED**: empirical per-pair drop_one anchors from `.omx/state/continual_learning_posterior.json` accepted-anchor history are AUTHORITATIVE — 9 anchors with non-uniform deltas (1 negative: pair0371 = -6.66e-07; 8 positive: +3.34e-07 each) empirically prove that the rate-only superposition model is INSUFFICIENT — distortion penalty dominates for non-pair0371 drop_one configurations.
- **HARD-EARNED**: GREEDY ranking by empirical per-pair drop_one ΔS is the CANONICAL alternative reducer methodology #2 per Catalog #308; it has the unique property of being independent of any interaction-matrix dependency (orthogonal to BUILD-1's failed methodology).
- **CARGO-CULTED (EMPIRICALLY FALSIFIED)**: assumption that GREEDY top-K under orthogonality (= K × per-pair-mean ΔS) produces additive improvement at K>1. UNWIND: BUILD-1c verdict records 100% (5-of-5) measured K>1 sister anchors REGRESSING vs K=1; the rate-savings at K>1 are dominated by interaction-distortion penalty.
- **CARGO-CULTED (HYPOTHESIS, UNRESOLVED)**: assumption that the 22 UNMEASURED drop_one pairs in the codex acquisition pool contain additional negative-ΔS pairs (= MORE pair0371-class candidates). UNWIND: BUILD-1b paid Modal CPU paired exact-eval on ~10 UNMEASURED drop_one candidates would resolve this empirically at ~$1.20 cost-band envelope.
- **CARGO-CULTED (HYPOTHESIS, UNRESOLVED)**: assumption that diversity_k anchors (which use canonical diversity-selection, NOT GREEDY-top-K-by-empirical-ΔS) are a faithful proxy for the GREEDY top-K predicted curve. UNWIND: BUILD-1b paid Modal CPU paired exact-eval on a GREEDY-top-K-by-empirical archive could resolve this empirically.

## Predicted ΔS band + Dykstra feasibility (Catalog #296)

GREEDY top-K predicted ΔS band per orthogonality assumption (`K × CANONICAL_RATE_DELTA_PER_BYTE = K × -6.66e-07`):
- K=1: -6.66e-07 (matches current frontier improvement)
- K=4: -2.66e-06
- K=8: -5.33e-06
- K=12: -7.99e-06
- K=16: -1.07e-05

Empirically falsified at K>1 surface: ALL 5 measured K>1 sister anchors REGRESS vs base by +1.33e-05 to +2.94e-05 (= 20×-44× worse than the orthogonality prediction's promise).

The Dykstra-feasibility intersection (Boyd CO-LEAD per CLAUDE.md "Council conduct" 4-co-lead structure): the 3-polytope intersection for the empirically-observed K>1 anchors is EMPTY at the predicted band [-0.00001, -0.00005] because the empirical sister anchors all lie in the positive-ΔS half-space (regression-not-improvement) per the canonical decomposition (rate gain undone by SegNet+PoseNet distortion penalty). This is the empirical signature of Dykstra infeasibility at the predicted ΔS band.

**Sister probe-disambiguator path**: `tools/probe_dqs1_drop_many_greedy_independent_disambiguator.py` (this file) is the canonical disambiguator; BUILD-1c empirically resolves the orthogonality assumption (NEGATIVE per measured K>1 sister anchors); BUILD-1b paid Modal CPU paired exact-eval is the canonical re-resolution via NEW empirical K>1 anchors with operator-selected pair-tuples (e.g. pair0371 + pair-X).

## Council attendees / verdict (Catalog #300 v2 frontmatter)

```yaml
council_tier: T1
council_attendees: [Shannon, Dykstra, Carmack, Assumption-Adversary]
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_dissent: []
council_assumption_adversary_verdict:
  - assumption: "GREEDY top-K under orthogonality assumption produces additive improvement at K>1"
    classification: CARGO-CULTED-EMPIRICALLY-FALSIFIED
    rationale: "100% (5-of-5) measured K>1 sister anchors regress vs K=1 per direct empirical comparison; rate-savings dominated by interaction-distortion penalty."
  - assumption: "Empirical per-pair drop_one anchors from continual_learning_posterior are authoritative"
    classification: HARD-EARNED
    rationale: "9 anchors with non-uniform deltas (1 negative + 8 positive) empirically prove the rate-only superposition model is insufficient; per-pair-axis distortion penalty discriminates."
  - assumption: "Per Catalog #307 IMPLEMENTATION-LEVEL falsification (vs PARADIGM-LEVEL refutation) is the canonical classification"
    classification: HARD-EARNED
    rationale: "The bug is in BUILD-1c's specific GREEDY-on-currently-measured-anchors implementation (implementation-level); the drop-many beam-search PARADIGM remains research-path-alive pending alternative empirical anchors per Catalog #308."
  - assumption: "Per CLAUDE.md 'Forbidden premature KILL without research exhaustion', BUILD-1c's NEGATIVE verdict is DEFER not KILL"
    classification: HARD-EARNED
    rationale: "22 UNMEASURED drop_one pairs MIGHT contain additional negative-ΔS pairs; BUILD-1b paid Modal CPU paired exact-eval is the canonical alternative methodology per Catalog #308."
  - assumption: "Diversity_k anchors (canonical diversity-selection) are a faithful proxy for GREEDY-top-K curve"
    classification: CARGO-CULTED-UNRESOLVED
    rationale: "The alignment between diversity-selected and GREEDY-by-empirical pair sets is unknown; BUILD-1b paid Modal CPU paired exact-eval on a GREEDY-top-K-by-empirical archive could resolve empirically."
council_decisions_recorded:
  - "BUILD-1c verdict: NEGATIVE_COLLAPSE_TO_K1_EMPIRICAL_DROP_MANY_REGRESSES (100% measured K>1 sisters regress vs K=1)"
  - "DEFER BUILD-2 + BUILD-3 + BUILD-4 + DISPATCH per Catalog #307 IMPLEMENTATION-LEVEL falsification (drop-many paradigm intact)"
  - "OPERATOR-ROUTABLE BUILD-1b: Modal CPU paired exact-eval ledger on UNMEASURED drop_one pairs (~$1.20 envelope); alternative methodology per Catalog #308"
  - "OPERATOR-ROUTABLE alternative #2: Yousfi+Fridrich human-prior drop-many tuples (e.g. pair0371 + pair-X with hypothesized orthogonal high-leverage frames)"
  - "OPERATOR-ROUTABLE alternative #3: substrate-class shift (Z6/Z7/Z8 predictive coding) per CLAUDE.md HORIZON-CLASS plateau-trap warning"
council_predicted_mission_contribution: apparatus_maintenance
council_override_invoked: true
council_override_rationale: "operator NON-NEGOTIABLE blanket approval 2026-05-19 verbatim 'all operator decisions and approval granted and provided fuly and completely' + today's 'continue with all'; BUILD-1c scope ($0 local CPU smoke) respects 'Executing actions with care'"
```

## Math + canonical equation candidate refinement (Catalog #344 RATIFY-N protocol)

### Candidate: `dqs1_drop_many_greedy_independent_pair_ordering_v1`

- **Registry status**: PENDING (NOT auto-registered; BUILD-1c refinement field QUEUED)
- **Mathematical statement**:
  - GREEDY top-K predicted ΔS under orthogonality assumption: `ΔS_greedy(K) = Σ_{i=1}^{K} predicted_ΔS_indep(pair_i)` where pairs are sorted ascending by `predicted_ΔS_indep`.
  - Falsifying outcome: for any K>1 with empirical sister anchor S_K, if `S_K.delta_vs_base ≥ predicted_ΔS_indep(pair_best)` then GREEDY paradigm DEFERs at the K>1 surface (drop-many regresses vs K=1).
- **Empirical refinement field proposed** (per verdict.json `canonical_equation_candidate_refinement.refinement_field_proposed`):
  - `empirical_k1_best_drop_one_delta_vs_base`: -6.658589e-07 (pair0371)
  - `empirical_k1_best_drop_one_pair_index`: 371
  - `empirical_drop_many_sister_anchors_regress_vs_k1`: 5 anchors (drop_two_r28_021 + diversity_k=2,4,8,12,16) ALL with positive delta_vs_base
  - `empirical_greedy_top_k_predicted_delta`: K-sweep table per verdict.json `greedy_top_k_sweep`
  - `current_frontier_cpu_delta_vs_base`: -6.658590e-07 (matches K=1 empirical optimum)
  - `greedy_verdict_class`: `NEGATIVE_COLLAPSE_TO_K1_EMPIRICAL_DROP_MANY_REGRESSES`
  - `predicted_band_refinement`: EMPIRICALLY FALSIFIED at K>1 surface; ALL measured K>1 sisters regress vs K=1 (drop_two=+6.68e-07 vs K=1=-6.66e-07; diversity_k=2 through k=24 all positive); drop-many paradigm DEFERs at K>1; K=1 empirical optimum is pair0371 at -6.66e-07.
- **Ratification trigger status**: `DEFERRED-PENDING-BUILD-1B-PAIRED-CPU-EXACT-EVAL-LEDGER per Catalog #344 RATIFY-N protocol (need ≥3 NEW empirical K>1 anchors with diverse pair-tuple selection to ratify orthogonality vs interaction contribution)`
- **Forbidden contexts** (per Catalog #359 sister discipline): the canonical equation candidate predicts REPLACEMENT-savings via GREEDY-top-K-tuple under orthogonality; it is NOT applicable to residual-correction hybrid contexts. BUILD-1c's verdict does NOT change this forbidden-context list.

## Discipline closure

- **6-hook wire-in declaration per Catalog #125**:
  - hook #1 sensitivity-map: N/A (BUILD-1c is observability-only; GREEDY predicted ΔS is non-authoritative per CLAUDE.md "MPS auth eval is NOISE" extension; macOS-CPU advisory per Catalog #192)
  - hook #2 Pareto constraint: N/A (no Pareto-relevant signal from GREEDY; the empirical K>1 sister anchors that DO carry Pareto signal are recorded via canonical posterior history, NOT via BUILD-1c emission)
  - hook #3 bit-allocator: N/A (no bit-allocator signal; the GREEDY pair-ordering is informational-only)
  - hook #4 cathedral autopilot dispatch: ACTIVE (canonical Catalog #313 probe-outcomes ledger row queued in verdict.json; autopilot's predecessor-probe gate per Catalog #313 will refuse downstream drop-many BUILD-2 dispatch unless reactivation criterion satisfied)
  - hook #5 continual-learning posterior: ACTIVE (verdict.json + canonical equation refinement field feed `tac.canonical_equations.update_equation_with_empirical_anchor` per Catalog #344; though refinement is DEFERRED pending BUILD-1b authoritative anchors at K>1)
  - hook #6 probe-disambiguator: ACTIVE (BUILD-1c IS the canonical disambiguator between "GREEDY orthogonality assumption holds at K>1" vs "drop-many regresses vs K=1 empirically"; resolved NEGATIVE empirically for currently-measured anchor set)
- **Catalog #313 probe-outcomes ledger row**: QUEUED in verdict.json `catalog_313_probe_outcomes_row` field for registration via `tac.probe_outcomes_ledger.register_probe_outcome` (verdict=DEFER; status=blocking; reactivation_criteria=BUILD-1b lands ≥3 NEW empirical K>1 anchors with at least one sister-tuple containing pair0371 to verify pair0371+N-untested-pairs additive vs distortion-regression).
- **Catalog #344 canonical equation refinement**: QUEUED in verdict.json `canonical_equation_candidate_refinement` field; NOT auto-registered. Operator-routable per RATIFY-N protocol once BUILD-1b lands authoritative K>1 anchors.
- **Catalog #287 evidence tags**: every artifact tagged `[predicted] [empirical:from_continual_learning_posterior_anchor_history_drop_one_drop_two_diversity_k]` with explicit data-source provenance per the Catalog #229 PV finding.
- **Catalog #323 canonical Provenance umbrella**: every artifact carries canonical Provenance with `score_claim=False` + `promotable=False` + `axis_tag=[predicted]` + `evidence_grade=research_only`.
- **Catalog #229 premise verification**: verified data source's authoritative provenance (continual_learning_posterior accepted-anchor history with non-uniform per-pair drop_one deltas) BEFORE running GREEDY ranking; finding recorded inline in verdict + metadata. Distinct from BUILD-1's PV (which discovered the acquisition plan's inherited-score field is non-authoritative).
- **Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE**: this is a NEW landing memo + NEW canonical script + NEW artifact files; NO mutation of:
  - codex DQS1 cascade source (acquisition plan / design memo / scaffold)
  - canonical equation registry (`.omx/state/canonical_equations_registry.jsonl`)
  - canonical state JSON (posterior / frontier pointer / lane registry / probe outcomes ledger)
  - BUILD-1 artifacts (landing memo / populate_interaction_matrix.py / verdict.json / metadata.json)
  - sister cathedral consumers
  - CLAUDE.md
  - sister landing memos
- **Catalog #230 sister-subagent ownership map**: Slot 1 CUDA-AXIS-DQS1-DESIGN + Slot 2 PR95-STAGE-2-MLX-BUILD + Slot 4 COMBINED-TIER-1-WAVE-2 sister subagents are scope-DISJOINT (different cells of the canvas); BUILD-1c touches only NEW directory `experiments/results/dqs1_drop_many_build_1c_greedy_heuristic_alternative_reducer_20260525/` + NEW canonical script + NEW landing memo.
- **Catalog #340 sister-checkpoint guard**: PROCEED verified via canonical helper BEFORE commit.
- **Catalog #300 Mission alignment Consequence 1 operator-frontier-override**: invoked; verbatim operator quotes cited in council frontmatter.
- **Catalog #348 retroactive sweep**: N/A (BUILD-1c is NOT a new STRICT preflight gate landing; it is a probe-disambiguator + landing memo per design memo cascade).

## Mission contribution per Catalog #300

`apparatus_maintenance` — BUILD-1c produced NEGATIVE empirical verdict via DIRECT EMPIRICAL COMPARISON (NOT data-source artifact like BUILD-1; this is the structural advance over BUILD-1's methodology). ALL 5 measured K>1 sister anchors REGRESS vs K=1 empirical optimum, preventing the downstream BUILD-2/BUILD-3/BUILD-4/DISPATCH cascade from spending paid GPU dollars (~$2.40-4 cost-band envelope per design memo §DISPATCH) on the drop-many K>1 surface. The MISSION-DIRECT score-lowering value of BUILD-1c is N/A (planning/observability-only). The structural value is unblocking the alternative reducer cascade per Catalog #308 — once an operator routes to BUILD-1b paid Modal CPU paired exact-eval on UNMEASURED drop_one pairs, the drop-many K>1 surface may unlock IF more negative-ΔS pairs exist in the unmeasured 22-pair subset.

## Operator-routable next-cascade priority

Per verdict.json `operator_routable_next_cascade`:

1. **DEFER BUILD-2 + BUILD-3 + BUILD-4 + DISPATCH** per Catalog #307 IMPLEMENTATION-LEVEL falsification of drop-many at K>1 surface (paradigm intact).
2. **K=1 frontier `pairset_drop_one_rank021_pair0371`** at 0.19202828 [contest-CPU] is the EMPIRICAL optimum among ALL measured drop-K configurations.
3. **Alternative reducer per Catalog #308 #3**: Yousfi+Fridrich human-prior drop-many tuples with MANUALLY-SPECIFIED interaction-aware pairs (e.g. test pair0371 + pair-X where X is hypothesized to be orthogonal to pair0371's high-leverage frames).
4. **Alternative reducer per Catalog #308 #4**: BUILD-1b paid Modal CPU paired exact-eval on ~10 UNMEASURED drop_one candidates (~$1.20) to discover whether any other pair has negative drop_one ΔS; if MORE negative-ΔS pairs exist, GREEDY top-K becomes empirically viable.
5. **Alternative reducer per Catalog #308 #5**: completely different substrate-class shift (Z6/Z7/Z8 predictive coding) per CLAUDE.md HORIZON-CLASS plateau-trap warning — 0.19202828 might be the asymptotic floor for within-class DQS1 selective decoder-Q at this archive composition.

## Sister-coherence verification

Per Catalog #340 sister-checkpoint guard + Catalog #230 ownership map:

- **Concurrent subagents** (cap=4 temp per operator-frontier-override): Slot 1 CUDA-AXIS-DQS1-DESIGN / Slot 2 PR95-STAGE-2-MLX-BUILD / Slot 4 COMBINED-TIER-1-WAVE-2
- **Scope**: all three sisters work on DIFFERENT cells of the canvas; no file-touch overlap with BUILD-1c
- **Files touched by THIS subagent**: NEW `tools/probe_dqs1_drop_many_greedy_independent_disambiguator.py` + NEW `experiments/results/dqs1_drop_many_build_1c_greedy_heuristic_alternative_reducer_20260525/{verdict.json,greedy_sweep_metadata.json}` + NEW landing memo at THIS path
- **Files NOT touched**: ANY codex DQS1 cascade source / DROP-MANY-BEAM-DESIGN scaffold / BUILD-1 artifacts / canonical equation registry / sister cathedral consumers / `CLAUDE.md` / sister landing memos / canonical state JSON (read-only)
- **Catalog #340 verification**: PROCEED (no file-touch overlap with sister subagents at checkpoint time)
