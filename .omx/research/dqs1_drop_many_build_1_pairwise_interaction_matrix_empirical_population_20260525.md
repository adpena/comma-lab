# DQS1 Drop-Many BUILD-1: Empirical Pairwise Interaction Matrix Population — Landing Memo (2026-05-25)

- timestamp_utc: 2026-05-25T15:45:00Z
- agent: claude (DROP-MANY-BEAM BUILD-1 subagent per design memo §BUILD-1 operator-routable enumeration)
- lane_id: lane_dqs1_drop_many_build_1_pairwise_interaction_matrix_empirical_population_20260525
- scope: empirical population of I[P,P] pairwise interaction matrix from canonical DQS1 acquisition plan; emit artifact matrix + canonical-provenance metadata + verdict per Carmack MVP-first 5-step
- authority: planning + observability ONLY; score/promotion/rank/dispatch authority all FALSE per Catalog #287/#323/#341
- relates to: DROP-MANY-BEAM-DESIGN landing memo `.omx/research/dqs1_drop_many_beam_pairwise_interaction_waterfill_design_memo_20260525.md`; scaffold `tools/probe_dqs1_drop_many_beam_pairwise_interaction_disambiguator.py`; codex eureka acquisition `.omx/research/codex_eureka_beyond_drop_two_acquisition_20260525T143351Z/dqs1_pairset_acquisition_eureka_drop_many.json`
- discipline anchors: Catalog #229 (premise verification) + #287 (canonical Provenance evidence-tag) + #303 (cargo-cult audit) + #307 (paradigm-vs-implementation falsification) + #308 (alternative probe methodologies) + #313 (probe-outcomes ledger) + #323 (canonical Provenance umbrella) + #341 (canonical routing markers) + #344 (canonical equation registry RATIFY-N) + #356 (per-axis decomposition) + #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE
- mission_contribution: `apparatus_maintenance` per Catalog #300 — BUILD-1 produced NEGATIVE empirical verdict via PREMISE FALSIFICATION (Catalog #229 + #307), preventing downstream BUILD-2/BUILD-3/BUILD-4/DISPATCH cascade from spending paid GPU dollars on an artifact-dominant interaction matrix; structural value is unblocking BUILD-1b sister-subagent alternative methodology per Catalog #308.
- operator-frontier-override: ACTIVE per Catalog #300 Mission alignment Consequence 1 — operator NON-NEGOTIABLE blanket approval + today's "continue with all" + 3-msg rate-attack amplification; BUILD-1 scope ($0 local CPU smoke) respects "Executing actions with care" non-negotiable

## Headline finding

**BUILD-1 acceptance criterion verdict: `DATA_SOURCE_PREMISE_FALSIFIED_ARTIFACT_DOMINANT`**

The codex DQS1 acquisition plan's `predicted_score_mean` field is emitted as `source_selector_inherited_non_authoritative` for ALL child candidates (drop_one + drop_two + drop_many alike). EVERY child candidate inherits the SAME source-selector predicted score (`0.19202894881608987`). The only discriminating field is `rate_score_delta_vs_source_selector` which uses an EXACT LINEAR SUPERPOSITION model (`drop_two delta = 2 × drop_one delta`).

When the canonical Δ_ij formula `Δ_ij = predicted_score(drop {i,j}) − predicted_score(drop {i}) − predicted_score(drop {j}) + base` is computed, the result is a CONSTANT `|Δ_ij| = 6.658589e-07` across 100% of the 465 computable pairs — empirically proving the LINEAR SUPERPOSITION ARTIFACT hypothesis. **The acquisition plan CANNOT yield true empirical I[P,P]** for BUILD-1's stated purpose.

Per Catalog #307 paradigm-vs-implementation classification: **this is IMPLEMENTATION-LEVEL falsification of the DATA-SOURCE assumption** ("BUILD-1 can use the codex acquisition plan as empirical anchor source"), **NOT paradigm-level refutation of the beam-search interaction-aware ranking design**. The beam-search PARADIGM remains INTACT; the bug is that the canonical data source for BUILD-1 was misidentified in the design memo.

Per Catalog #308 alternative probe methodologies: **BUILD-1b** is the canonical alternative methodology (Modal CPU paired exact-eval ledger on ~10 drop_one + ~20 drop_two candidates via canonical operator-authorize chain at ~$1.20-2 cost-band envelope). This yields TRUE per-pair `predicted_score_mean` as child-candidate-empirical (NOT source-selector-inherited).

Per CLAUDE.md "Forbidden premature KILL without research exhaustion" + Catalog #308: this is a **DEFER** verdict, NOT a KILL. The beam-search interaction-aware approach is RESEARCH-PATH-ALIVE pending BUILD-1b alternative methodology.

## Canonical-vs-unique decision per layer

Per Catalog #290 + CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode":

- **ADOPT_CANONICAL_BECAUSE_SERVES**: canonical Provenance umbrella per Catalog #323 (`CANONICAL_NON_PROMOTABLE_MARKERS` dict with `score_claim=False` / `promotable=False` / `axis_tag=[predicted]` / `evidence_grade=research_only`) is threaded into every emitted artifact's metadata.
- **ADOPT_CANONICAL_BECAUSE_SERVES**: canonical Catalog #341 routing markers (predicted_delta_adjustment=0.0 / promotable=False / axis_tag=[predicted]) for non-promotable observability ensure the artifact is observability-only and cannot be promoted to a score signal.
- **ADOPT_CANONICAL_BECAUSE_SERVES**: canonical Catalog #313 probe-outcomes ledger row (DEFER verdict with reactivation_criteria + canonical_helper_invocation) queued in verdict.json for downstream registration via `tac.probe_outcomes_ledger.register_probe_outcome`.
- **ADOPT_CANONICAL_BECAUSE_SERVES**: canonical Catalog #344 RATIFY-N protocol for canonical equation candidate `dqs1_drop_many_pairwise_interaction_beam_search_v1` — refinement field QUEUED with `ratification_trigger_status: DEFERRED-PENDING-BUILD-1B-PAIRED-CPU-EXACT-EVAL-LEDGER`.
- **ADOPT_CANONICAL_BECAUSE_SERVES**: canonical Catalog #229 premise verification protocol — verified the acquisition plan's `predicted_score_source` field BEFORE relying on `predicted_score_mean` for Δ_ij computation; documented finding inline in verdict.json `final_verdict_reason`.
- **FORK_BECAUSE_PRINCIPLED_MISMATCH**: per Catalog #303 cargo-cult audit, the design memo's BUILD-1 dependency on the acquisition plan as empirical data source was CARGO-CULTED (untested). BUILD-1 unwinds the cargo-cult by surfacing the ARTIFACT DETECTION methodology (single-value concentration ≥80%) that empirically identifies this class of mis-data-source for future bug-class extinction.
- **FORK_BECAUSE_PRINCIPLED_MISMATCH**: per Catalog #308 alternative probe methodologies, BUILD-1 enumerates N=4 alternative reducers in the verdict.json `operator_routable_next_cascade` array (BUILD-1b paired CPU exact-eval ledger / BUILD-1c GREEDY heuristic / DEFER BUILD-2 / sister codex acquisition refresh) — operator-routable for next-cascade priority decision.

## Observability surface (Catalog #305)

BUILD-1 observable through:

1. **Inspectable per layer**: 3 canonical artifacts (matrix .npy / metadata.json / verdict.json) emitted under `experiments/results/dqs1_drop_many_build_1_pairwise_interaction_matrix_population_20260525/`; each carries canonical Provenance.
2. **Decomposable per signal**: empirical distribution decomposed via `empirical_distribution` field (n_total / n_pairs_abs_gt_1e_8 / n_pairs_abs_gt_1e_6 / max / min / median / mean); artifact detection via `artifact_detection` field (most_common_abs_delta_ij_top_20 / artifact_dominant flag).
3. **Diff-able across runs**: artifacts use canonical JSON `sort_keys=True` per Catalog #131; two runs of the same script produce byte-identical metadata.json + verdict.json.
4. **Queryable post-hoc**: verdict's `top_100_by_abs_delta_ij` field enumerates per-pair (i, j, Δ_ij) tuples; `most_common_abs_delta_ij_top_20` histogram queryable for artifact-concentration analysis.
5. **Cite-able**: every artifact's `canonical_provenance.source_url` cites the canonical acquisition plan path; `canonical_helper_invocation` cites the canonical script path.
6. **Counterfactual-able**: the script's design supports `--probe-pairwise-interaction <i> <j>` extension (sister-subagent BUILD-1b future enhancement) for synthetic "what if we had child-candidate-empirical data?" probes.

## 9-dimension success checklist evidence (Catalog #294)

1. **UNIQUENESS**: BUILD-1 is the UNIQUE first executable step of the DROP-MANY-BEAM-DESIGN cascade; uses existing canonical state (acquisition plan + frontier pointer) at $0; falsifies the design memo's data-source assumption empirically before paid GPU spend.
2. **BEAUTY + ELEGANCE**: ONE 30 KB script + 3 canonical artifacts (matrix .npy + metadata + verdict) cover the entire BUILD-1 deliverable; operator-readable in <10 min; design-memo references in every artifact.
3. **DISTINCTNESS**: distinct from sister BUILD-2/BUILD-3/BUILD-4 (those depend on BUILD-1 outcome); distinct from sister cathedral consumers (no Tier B registration in BUILD-1 per design memo §4-BUILD enumeration); distinct from concurrent subagents (MLX-ARCH-5 / PAIR-FRAME-LATTICE / COMBINED-TIER-1-CCC-EXT-PROBES all touch different cells of the canvas).
4. **RIGOR**: every claim cites canonical artifact path; empirical computation is deterministic given source acquisition plan SHA; artifact-detection methodology (single-value concentration ≥80%) is canonical falsifiable test for this class of mis-data-source bug.
5. **OPTIMIZATION-PER-TECHNIQUE**: BUILD-1 is the unique-and-complete script for empirical I[P,P] population from existing data sources; the artifact-detection methodology forks from canonical Catalog #287 docstring-overstatement gate to surface the data-source-misidentification subclass.
6. **STACK-OF-STACKS-COMPOSABILITY**: verdict.json embeds Catalog #313 probe-outcomes row + Catalog #344 canonical equation refinement (PENDING) + Catalog #356 per-axis decomposition placeholder (None until BUILD-1b paired CPU exact-eval) — all 3 wire-in points explicit.
7. **DETERMINISTIC-REPRODUCIBILITY**: artifact .npy is float64; metadata + verdict use canonical JSON `sort_keys=True`; two runs produce byte-identical output.
8. **EXTREME-OPTIMIZATION-PERFORMANCE**: $0 + ~7 min wall-clock; reused canonical state (acquisition plan + frontier pointer) without paid GPU.
9. **OPTIMAL-MINIMAL-CONTEST-SCORE**: BUILD-1 verdict's `apparatus_maintenance` mission contribution is the FRONTIER-PROTECTING value — preventing the BUILD-2/BUILD-3/BUILD-4/DISPATCH cascade from spending paid GPU dollars on an artifact-dominant interaction matrix. The MISSION-DIRECT score-lowering value is N/A for BUILD-1 (planning/observability-only); BUILD-1b is the next-cascade priority that may unlock authentic interaction-aware beam search per design memo predicted ΔS band [-0.00001, -0.00005].

## Cargo-cult audit per assumption (Catalog #303)

Per CLAUDE.md hard-earned-vs-cargo-culted addendum + Catalog #303 design-memo discipline:

- **CARGO-CULTED (EMPIRICALLY FALSIFIED)**: assumption that the codex acquisition plan's `predicted_score_mean` field is suitable as empirical anchor source for Δ_ij computation. The field is `source_selector_inherited_non_authoritative` per the plan's own `predicted_score_source` annotation. UNWIND: BUILD-1 verdict's `final_verdict=DATA_SOURCE_PREMISE_FALSIFIED_ARTIFACT_DOMINANT` records this empirically with 100% single-value concentration; ALTERNATIVE: BUILD-1b paired CPU exact-eval ledger via Modal CPU dispatch.
- **HARD-EARNED**: canonical rate-superposition model (`drop_two delta = 2 × drop_one delta`) for `rate_score_delta_vs_source_selector` is EMPIRICALLY VERIFIED in this data (max_abs match constant 6.658589e-07 = per-pair rate delta × inheritance arithmetic).
- **HARD-EARNED**: canonical Catalog #229 premise verification protocol — the artifact-detection methodology (single-value concentration ≥80%) caught the bug class structurally without paid GPU spend.
- **CARGO-CULTED (HYPOTHESIS)**: assumption that re-issuing the codex DQS1 acquisition with per-child-candidate `predicted_score_mean` (replacing source-selector-inherited with autograd-projected or Taylor-expanded per-pair estimate) would satisfy BUILD-1 data-source requirement at $0. UNWIND: sister codex subagent could test this without paid GPU; or BUILD-1b paid CPU dispatch ($1.20-2) provides authoritative ground truth.

## Predicted ΔS band + Dykstra feasibility (Catalog #296)

Per design memo §"Predicted ΔS band": the canonical predicted ΔS band `[-0.00001, -0.00005]` for K=4-8 drop-many beam search ASSUMED ≥10% non-orthogonal pairs. **BUILD-1 cannot empirically refine this band** because the data source is non-authoritative.

The Dykstra-feasibility intersection (Boyd CO-LEAD per CLAUDE.md "Council conduct" 4-co-lead structure) was design-memo-verified for codex's 34 drop_many candidates by inspection of their `distortion_repair_budget_from_rate_savings.score_budget` fields (positive 3.99e-6 for k=6 anchor). BUILD-1 does not re-verify; BUILD-2 implements the actual alternating-projection iteration.

**Sister probe-disambiguator path**: `tools/probe_dqs1_drop_many_beam_pairwise_interaction_disambiguator.py` (scaffold) is the canonical disambiguator; BUILD-1 empirically resolves the data-source assumption (NEGATIVE per artifact-dominant); BUILD-1b is the canonical re-empirical resolution via paired CPU exact-eval.

## Council attendees / verdict (Catalog #300 v2 frontmatter)

```yaml
council_tier: T1
council_attendees: [Shannon, Dykstra, Daubechies, Carmack, Assumption-Adversary]
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_dissent: []
council_assumption_adversary_verdict:
  - assumption: "Acquisition plan's predicted_score_mean is empirical per-child-candidate"
    classification: CARGO-CULTED
    rationale: "Empirically falsified at 100% single-value concentration; the field is source_selector_inherited_non_authoritative per the plan's own predicted_score_source annotation."
  - assumption: "BUILD-1 produces an empirical I[P,P] matrix that meaningfully informs BUILD-2"
    classification: CARGO-CULTED-EMPIRICALLY-FALSIFIED
    rationale: "Per artifact-detection at 100% single-value concentration; the I[P,P] is an arithmetic artifact of the inheritance arithmetic, not a true per-pair interaction measurement."
  - assumption: "Catalog #307 IMPLEMENTATION-LEVEL falsification (vs PARADIGM-LEVEL refutation) is the canonical classification"
    classification: HARD-EARNED
    rationale: "The bug is in BUILD-1's data-source assumption (implementation-level), not in the beam-search interaction-aware design (paradigm-level). The paradigm remains research-path-alive pending BUILD-1b alternative methodology."
  - assumption: "Per CLAUDE.md 'Forbidden premature KILL without research exhaustion', BUILD-1's NEGATIVE verdict is DEFER not KILL"
    classification: HARD-EARNED
    rationale: "BUILD-1b paired CPU exact-eval ledger is the canonical alternative methodology per Catalog #308; the design memo's beam-search paradigm has alternative data-source paths that remain unprobed."
council_decisions_recorded:
  - "BUILD-1 verdict: DATA_SOURCE_PREMISE_FALSIFIED_ARTIFACT_DOMINANT (100% single-value concentration empirically detected)"
  - "DEFER BUILD-2 + BUILD-3 + BUILD-4 + DISPATCH per Catalog #307 IMPLEMENTATION-LEVEL falsification"
  - "OPERATOR-ROUTABLE BUILD-1b: Modal CPU paired exact-eval ledger ~$1.20-2 cost-band; alternative methodology per Catalog #308"
  - "OPERATOR-ROUTABLE BUILD-1c: drop-many GREEDY heuristic without beam search (no interaction-matrix dependency); collapses to depth-1 cumulative independent drop-K"
  - "OPERATOR-ROUTABLE sister: re-issue codex DQS1 acquisition with per-child-candidate autograd-projected predicted_score_mean (no paid GPU)"
council_predicted_mission_contribution: apparatus_maintenance
council_override_invoked: true
council_override_rationale: "operator NON-NEGOTIABLE blanket approval 2026-05-19 verbatim 'all operator decisions and approval granted and provided fuly and completely' + today's 'continue with all'; BUILD-1 scope ($0 local CPU smoke) respects 'Executing actions with care'"
```

## Math + canonical equation candidate refinement (Catalog #344 RATIFY-N protocol)

### Candidate: `dqs1_drop_many_pairwise_interaction_beam_search_v1`

- **Registry status**: PENDING (NOT auto-registered; BUILD-1 refinement field QUEUED)
- **Empirical refinement field proposed** (per verdict.json `canonical_equation_candidate_refinement.refinement_field_proposed`):
  - `empirical_sparsity_ratio_at_1e_6`: 0.0 (0% non-orthogonal pairs at 1e-6 threshold)
  - `empirical_artifact_concentration_pct`: 100.0 (100% concentration at single value)
  - `empirical_data_source_provenance`: `acquisition_plan_source_selector_inherited_non_authoritative_NOT_paired_cpu_exact_eval`
  - `empirical_max_abs_delta_ij_band`: 6.658589e-07 (constant)
  - `empirical_data_source_falsified_for_authoritative_use`: TRUE
  - `predicted_band_refinement`: PENDING (BUILD-1b alternative methodology required)
- **Ratification trigger status**: `DEFERRED-PENDING-BUILD-1B-PAIRED-CPU-EXACT-EVAL-LEDGER` per Catalog #344 `when_3+_new_empirical_anchors_in_domain` protocol
- **Forbidden contexts** (per Catalog #359 sister discipline): the canonical equation candidate predicts REPLACEMENT-savings via cumulative drop-K-tuple with cross-term correction; it is NOT applicable to residual-correction hybrid contexts. BUILD-1's artifact verdict does NOT change this forbidden-context list.

## Discipline closure

- **6-hook wire-in declaration per Catalog #125**:
  - hook #1 sensitivity-map: N/A (BUILD-1 is observability-only; matrix is non-authoritative)
  - hook #2 Pareto constraint: N/A (no Pareto-relevant signal from non-authoritative matrix)
  - hook #3 bit-allocator: N/A (no bit-allocator signal; interaction matrix is artifact-dominant)
  - hook #4 cathedral autopilot dispatch: ACTIVE (canonical Catalog #313 probe-outcomes ledger row queued in verdict.json; autopilot's predecessor-probe gate per Catalog #313 will refuse downstream BUILD-2 dispatch unless reactivation criterion satisfied)
  - hook #5 continual-learning posterior: ACTIVE (verdict.json + canonical equation refinement field feed `tac.canonical_equations.update_equation_with_empirical_anchor` per Catalog #344; though refinement is DEFERRED pending BUILD-1b authoritative anchors)
  - hook #6 probe-disambiguator: ACTIVE (BUILD-1 IS the canonical disambiguator between "acquisition plan has empirical anchors" vs "acquisition plan has source-selector-inherited scores"; resolved NEGATIVE empirically)
- **Catalog #313 probe-outcomes ledger row**: QUEUED in verdict.json `catalog_313_probe_outcomes_row` field for registration via `tac.probe_outcomes_ledger.register_probe_outcome` (verdict=DEFER; status=blocking; reactivation_criteria=BUILD-1b paired CPU exact-eval ledger)
- **Catalog #344 canonical equation refinement**: QUEUED in verdict.json `canonical_equation_candidate_refinement` field; NOT auto-registered. Operator-routable per RATIFY-N protocol once BUILD-1b lands authoritative anchors.
- **Catalog #287 evidence tags**: every artifact tagged `[predicted] [empirical:from_codex_eureka_drop_many_acquisition_plan_inherited_score_field]` with explicit data-source provenance per the artifact-detection finding.
- **Catalog #323 canonical Provenance umbrella**: every artifact carries canonical Provenance with `score_claim=False` + `promotable=False` + `axis_tag=[predicted]` + `evidence_grade=research_only`.
- **Catalog #229 premise verification**: verified data source's `predicted_score_source` field BEFORE relying on `predicted_score_mean` for Δ_ij; finding (CARGO-CULTED data-source assumption) recorded inline in verdict + metadata.
- **Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE**: this is a NEW landing memo + NEW artifact files; NO mutation of sister codex artifacts (acquisition plan / design memo / scaffold / canonical equation registry / canonical state JSON all preserved).
- **Catalog #230 sister-subagent ownership map**: PAIR-FRAME-LATTICE + MLX-ARCH-5 + COMBINED-TIER-1-CCC-EXT-PROBES sister subagents are scope-DISJOINT (different cells of the canvas); BUILD-1 touches only NEW directory `experiments/results/dqs1_drop_many_build_1_pairwise_interaction_matrix_population_20260525/` + NEW landing memo.
- **Catalog #340 sister-checkpoint guard**: PROCEED verified via canonical helper BEFORE commit.
- **Catalog #300 Mission alignment Consequence 1 operator-frontier-override**: invoked; verbatim operator quotes cited in council frontmatter.

## Mission contribution per Catalog #300

`apparatus_maintenance` — BUILD-1 produced NEGATIVE empirical verdict via PREMISE FALSIFICATION (Catalog #229 + #307 + #308), preventing the downstream BUILD-2/BUILD-3/BUILD-4/DISPATCH cascade from spending paid GPU dollars (~$2.40-4 cost-band envelope per design memo §DISPATCH) on an artifact-dominant interaction matrix. The MISSION-DIRECT score-lowering value of BUILD-1 is N/A (planning/observability-only). The structural value is unblocking BUILD-1b sister-subagent alternative methodology per Catalog #308; once BUILD-1b lands authoritative anchors, BUILD-2 + BUILD-3 + BUILD-4 + DISPATCH cascade can proceed per design memo's predicted ΔS band [-0.00001, -0.00005] for K=4-8 drop-many beam search.

## Operator-routable next-cascade priority

Per verdict.json `operator_routable_next_cascade`:

1. **BUILD-1b ALTERNATIVE PATH** (Catalog #308 alternative methodology): Modal CPU paired exact-eval ledger on ~10 drop_one + ~20 drop_two candidates via canonical operator-authorize chain (~$1.20-2 cost-band envelope); yields true per-pair `predicted_score_mean` as child-candidate-empirical.
2. **BUILD-1c ALTERNATIVE METHODOLOGY** (Catalog #308 alternative reducer): drop-many GREEDY heuristic without beam search (no interaction-matrix dependency); collapses to depth-1 cumulative independent drop-K which the canonical rate-superposition predicts unambiguously.
3. **DEFER BUILD-2** executable beam search (Catalog #307 IMPLEMENTATION-LEVEL falsification per data source; PARADIGM intact; reactivation criterion: BUILD-1b lands authoritative empirical anchors).
4. **Sister alternative**: re-issue codex DQS1 acquisition refresh with per-child-candidate `predicted_score_mean` (replacing source_selector_inherited with autograd-projected or Taylor-expanded per-pair estimate) — this would satisfy BUILD-1 data-source requirement WITHOUT paid GPU dispatch.

## Sister-coherence verification

Per Catalog #340 sister-checkpoint guard + Catalog #230 ownership map:

- **Concurrent subagents** (cap=4 temp per operator-frontier-override): Slot 1 PAIR-FRAME-LATTICE / Slot 2 MLX-ARCH-5 / Slot 4 COMBINED-TIER-1-CCC-EXT-PROBES
- **Scope**: all three sisters work on DIFFERENT cells of the rate-attack canvas; no file-touch overlap
- **Files touched by THIS subagent**: NEW `experiments/results/dqs1_drop_many_build_1_pairwise_interaction_matrix_population_20260525/{populate_interaction_matrix.py,interaction_matrix_artifact.npy,interaction_matrix_metadata.json,verdict.json}` + NEW landing memo at THIS path
- **Files NOT touched**: ANY codex DQS1 cascade source / DROP-MANY-BEAM-DESIGN scaffold (`tools/probe_dqs1_drop_many_beam_pairwise_interaction_disambiguator.py`) / canonical equation registry (`.omx/state/canonical_equations_registry.jsonl`) / sister cathedral consumers / `CLAUDE.md` / sister landing memos
- **Catalog #340 verification**: PROCEED (no file-touch overlap with sister subagents at checkpoint time)
