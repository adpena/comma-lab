# Canonical Anti-Patterns Registry — Design Memo

**Status**: DESIGN_MEMO (queued for next-cycle implementation; $0 design work; disjoint from Slot 1 Dykstra solver + Slot 2 V3 int8 in flight)
**Date**: 2026-05-28
**Lane**: `lane_canonical_anti_patterns_registry_design_20260528`
**Operator directive**: 2026-05-28 verbatim *"learning anti-patterns is upser important too for compounding continual learning, like the canonical equations bu netgative and a higher layer of abstraction"*

## Canonical-vs-unique decision per layer

| Layer | Decision | Rationale |
|---|---|---|
| Dataclass contract | ADOPT_CANONICAL_BECAUSE_SERVES | Mirrors `tac.canonical_equations.equation.CanonicalEquation` shape (frozen + Provenance + APPEND-ONLY history) |
| JSONL persistence | ADOPT_CANONICAL_BECAUSE_SERVES | Mirrors `.omx/state/canonical_equations_registry.jsonl` fcntl-locked pattern per Catalog #131/#138/#245 sister discipline |
| Cathedral consumer | ADOPT_CANONICAL_BECAUSE_SERVES | Mirrors Catalog #335 CathedralConsumerContract; auto-discovered |
| Auto-recalibrator | ADOPT_CANONICAL_BECAUSE_SERVES | Mirrors Catalog #371 `tac.canonical_equations.registry.auto_recalibrate_from_continual_learning_posterior` |
| Anti-pattern semantics | FORK_BECAUSE_PRINCIPLED_MISMATCH | Equation `prediction_band` IS replaced by `falsification_band` (the band where the anti-pattern MANIFESTS empirically) — semantically inverted |
| Higher-layer abstraction | FORK_BECAUSE_PRINCIPLED_MISMATCH | Catalog gates extinct per-instance recurrences; this registry captures CLASS-level pattern that compounds across substrates/atoms/elements/components (operator's "higher layer of abstraction" directive) |

## The structural gap this fills

Per operator's META insight: `tac.canonical_equations` (Catalog #344) captures POSITIVE patterns at instance granularity (closed-form predictions + `EmpiricalAnchor` rows that RATIFY/FALSIFY). What is MISSING is the SYMMETRIC half:

- **POSITIVE registry (canonical_equations)**: "what works + at what predicted band; empirical anchors refine band"
- **NEGATIVE registry (canonical_anti_patterns)**: "what does NOT work + at what falsification band; empirical falsifications refine the recurrence-conditions"

The CLAUDE.md "FORBIDDEN PATTERNS" section already documents ~20+ anti-patterns as PROSE narrative (e.g. "Forbidden device-selection defaults (the MPS-fallback trap)" / "Forbidden closed-form-CDF-allocator-without-empirical-bit-spend-proof"). Catalog gates extinct per-instance source-text recurrences. But neither surface is:
- **Queryable** by cathedral consumer auto-discovery per Catalog #335
- **Compoundable** across substrates via canonical posterior anchors
- **Auto-recalibratable** by new empirical falsifications per Catalog #371-sister
- **Routable** by per-axis Pareto polytope solver (Slot 1 Dykstra in flight) to STEER next-cycle attack away from known anti-pattern paths

Per CLAUDE.md "Results must become system intelligence" non-negotiable + "Subagent coherence-by-default" — every empirical falsification IS feedback for the solver, not standalone prose. The current state leaks signal across sessions: each new subagent re-reads the 20+ FORBIDDEN PATTERNS section in prose form rather than the cathedral autopilot ranker programmatically steering them away from known anti-pattern compounding stacks.

## Mathematical compounding identity (canonical equations sister)

Per CLAUDE.md "Meta-Lagrangian/Pareto solver — NON-NEGOTIABLE, HIGHEST EMPHASIS" + the just-spawned Slot 1 Dykstra Pareto polytope solver wire-in:

**Positive compounding**: `ΔS_total = Σ_axes ΔS_axis_i` where each `ΔS_axis_i` is a canonical equation prediction; Pareto polytope intersection (Dykstra) computes optimal compound

**Negative compounding** (THIS framework's contribution): `RegressionRisk_total = max_anti-patterns AntiPatternRisk_i × indicator(stack_matches_anti_pattern_class_i)` — the canonical aggregation is MAX not SUM because anti-patterns are typically dominated by the WORST applicable pattern (one violated anti-pattern kills the whole stack); per-axis Dykstra polytope intersection EXCLUDES anti-pattern-matching regions from feasibility set

The mathematical compounding for routing:
```
NextCycleAttackDirection = argmax_axis (
    PredictedΔS_axis_i × λ_axis_i_tight_from_Dykstra
) subject to (
    NOT any (proposed_stack matches AntiPattern_j AND not waived)
)
```

This converts the anti-patterns from PROSE WARNINGS into ACTIVE CONSTRAINTS on the Pareto polytope solver's feasibility set — operationalizing the operator's "higher layer of abstraction" directive.

## Canonical contract

### `AntiPattern` frozen dataclass (mirrors `CanonicalEquation`)

```python
@dataclass(frozen=True)
class AntiPattern:
    anti_pattern_id: str  # e.g. "lzma_on_already_brotli_saturated_compounding_v1"
    description: str
    forbidden_pattern_predicate: str  # mathematical / source-text / artifact-level signature
    falsification_band: Mapping[str, float]  # the range where the anti-pattern empirically MANIFESTS (e.g. {"lzma_ratio_lo": 0.999, "lzma_ratio_hi": 1.005})
    recurrence_conditions: tuple[str, ...]  # situations that trigger the anti-pattern's manifestation
    canonical_source_anchor: str  # CLAUDE.md FORBIDDEN_PATTERNS section / Catalog # / sister memo path
    canonical_unwind_path: str  # the canonical correct alternative (e.g. "Quantize FIRST, entropy code SECOND")
    canonical_producers: tuple[str, ...]  # where the anti-pattern manifests in repo
    canonical_consumers: tuple[str, ...]  # who should consult this anti-pattern (cathedral autopilot ranker / Dykstra solver / sister gates)
    paradigm_class: Literal["compounding_order_anti_pattern", "quantization_anti_pattern", "diagnosis_anti_pattern", "provenance_anti_pattern", "data_source_anti_pattern", "observability_anti_pattern", "rigor_loss_anti_pattern", "premature_kill_anti_pattern"]
    severity: Literal["critical_paradigm_blocker", "high_compound_corruption", "medium_substrate_regression", "low_implementation_inefficiency"]
    provenance: Mapping[str, Any]  # canonical Provenance per Catalog #323
    empirical_falsifications: tuple[EmpiricalFalsification, ...]
    last_recalibration_utc: str
    next_recalibration_trigger: Literal["when_3+_new_empirical_falsifications_in_domain", "when_severity_drift_exceeds_2x", "when_operator_invokes", "never_auto"]
```

### `EmpiricalFalsification` frozen dataclass (mirrors `EmpiricalAnchor`)

```python
@dataclass(frozen=True)
class EmpiricalFalsification:
    anti_pattern_id: str  # references parent AntiPattern
    falsification_id: str
    measurement_method: str  # how the anti-pattern was empirically confirmed
    empirical_artifact_path: str  # canonical artifact citing the falsification
    empirical_output: Mapping[str, Any]  # measured signature (e.g. {"lzma_ratio": 1.001, "additional_bytes_saved": 0, "expected_savings_pct": 5.0})
    falsification_residual: float | None  # |empirical - expected| / expected
    captured_at_utc: str
    canonical_provenance: Mapping[str, Any]  # per Catalog #323; NON-PROMOTABLE if MLX/MPS/macOS-CPU advisory per Catalog #127/#192
    incident_classification: Literal["paradigm_level_falsification_of_anti_pattern", "implementation_level_confirmation_of_anti_pattern", "ratification_of_anti_pattern_at_new_substrate", "edge_case_partial_manifestation"]  # per Catalog #307
    severity_observed: Literal["critical", "high", "medium", "low"]
    operator_routable_unwind_path: str  # the canonical correct alternative actually executed (or queued)
```

## Persistence layer

`.omx/state/canonical_anti_patterns_registry.jsonl` — fcntl-locked APPEND-ONLY JSONL per Catalog #131/#138/#245 sister discipline.

Event types (sister of canonical_equations `EVENT_REGISTERED` / `EVENT_ANCHOR_APPENDED` / `EVENT_RECALIBRATED`):
- `EVENT_ANTI_PATTERN_REGISTERED` (new anti-pattern landed)
- `EVENT_FALSIFICATION_APPENDED` (new empirical falsification landed)
- `EVENT_ANTI_PATTERN_RECALIBRATED` (auto-recalibrator fired per `when_3+_new_empirical_falsifications_in_domain`)
- `EVENT_UNWIND_PATH_RATIFIED` (canonical unwind path empirically demonstrated to recover from the anti-pattern; sister of `EmpiricalAnchor` RATIFY)

## Initial canonical population (12 anti-patterns)

From CLAUDE.md FORBIDDEN_PATTERNS section + my Wave N+1 compound stacking analysis + just-landed verdicts:

### Compounding-order anti-patterns

**1. `lzma_on_already_brotli_saturated_compounding_v1`** — LZMA over already-brotli'd byte stream; ratio saturates at ~1.001 per CASCADE_SATURATION sister verdict (commit `d78401444`). Canonical unwind: do NOT chain LZMA after brotli; use ANS instead OR brotli q=11 vs q=9 standalone. Severity: medium_substrate_regression. Source: decoder compression analysis commit `44d12e75d` + V3 RE-RUN commit `b01232473`.

**2. `quantize_then_svd_corrupted_low_rank_v1`** — SVD applied to already-quantized tensor; quantization noise dominates low-rank residual. Canonical unwind: SVD FIRST (lossless rank reduction) → quantization SECOND (per-factor int8) → entropy coding THIRD. Severity: high_compound_corruption.

**3. `fp4_packed_without_qat_cos_collapse_v1`** — FP4 packed nibbles deployment without Quantization-Aware Training; cos similarity drops below 0.999 threshold. Canonical unwind: QAT pipeline per CLAUDE.md "QAT pipeline — non-negotiable for FP4 deployment". Severity: critical_paradigm_blocker. Source: decoder compression analysis Scenario A.

**4. `brotli_plus_lzma_chained_anti_pattern_v1`** — Compounding entropy coders that operate on similar redundancy domains saturate. Canonical unwind: choose ORTHOGONAL entropy coders (ANS + dictionary) OR single high-quality (brotli q=11) standalone.

### Diagnosis anti-patterns

**5. `cross_paradigm_test_without_per_axis_decomposition_v1`** — Cross-paradigm substrate dispatched without per-axis decomposition surfacing; seg-axis dominance attributed to substrate when it is SHARED-DECODER artifact. Canonical unwind: enable Catalog #356 per-axis decomposition GAP FIX (post-commit `92a39dc62`) BEFORE cross-paradigm test. Severity: high_compound_corruption. Source: V3 RE-RUN commit `b01232473` empirical anchor.

**6. `predicted_band_from_random_init_tier_c_v1`** — Predicted band derived from Tier-C density measurement on RANDOM_INIT (pre-training) archive; band misses post-training reality by 22× (C6 IBPS empirical). Canonical unwind: post-training Tier-C re-measurement OR pending_post_training validation_status + reactivation criteria per Catalog #324. Severity: critical_paradigm_blocker.

**7. `rank_1_problem_spec_synergy_tautology_v1`** — Multi-op synergy measurement built on RANK-1 operator-gradient matrix returns synergy ≈ 0 for ANY input (arithmetic tautology, NOT empirical property). Canonical unwind: derive each operator's per-axis gradient from its OWN per-pair footprint via Catalog #356 AxisDecomposition; operator-gradient matrix MUST be rank > 1; assert via regression test. Severity: high_compound_corruption. Source: paradox half 2 rigor review commit `21014faa7`.

### Provenance anti-patterns

**8. `phantom_score_directory_naming_lie_v1`** — Output file/directory name claims a device/scope/contract that the contents do NOT match (e.g. `contest_auth_eval_cuda.json` containing CPU eval). Canonical unwind: filename MUST match metadata that generated it OR be device-agnostic. Severity: critical_paradigm_blocker. Source: Z3 v2 FULL Modal A100 commit + Catalog #249 self-protection.

**9. `transient_tmp_path_in_persisted_artifact_v1`** — `/tmp/<path>` cited as durable evidence in lane registry / dispatch claim / commit message / build manifest. Canonical unwind: `experiments/results/<lane_id>_<timestamp>/` for build artifacts; `.omx/state/` for ledgers; `.omx/research/` for analyses; `.omx/tmp/` for explicitly ephemeral. Severity: high_compound_corruption. Source: lane_pr106_stacked + Catalog #220 + multiple sister incidents.

### Data-source anti-patterns

**10. `source_selector_inherited_predicted_score_mean_v1`** — Empirical interaction matrix populated from `predicted_score_mean` field that inherits from source selector rather than measuring per-pair empirically; 100% of computed values concentrate at single arithmetic artifact. Canonical unwind: paired CPU exact-eval ledger via Modal CPU dispatch (~$1.20-2 per Catalog #246) OR drop-many greedy heuristic without interaction matrix dependency. Severity: medium_substrate_regression. Source: DQS1 drop-many BUILD-1 verdict 2026-05-25.

### Observability anti-patterns

**11. `silent_no_spawn_modal_dispatch_v1`** — Modal dispatch sys.exit(...) FATAL path BEFORE `fn.spawn()` queues a task; no canonical ledger row + no recovery dump; harvester-invisible per CLAUDE.md "Modal `.spawn()` HARVEST OR LOSE". Canonical unwind: route through `register_pre_spawn_fatal(...)` per Catalog #360 helper which writes `pre_spawn_fatal` event to canonical ledger BEFORE sys.exit. Severity: high_compound_corruption (paid GPU meter started; rows orphaned). Source: STC v2 5th consecutive silent-no-spawn anchor + Catalog #360.

### Rigor-loss anti-patterns

**12. `docstring_overstatement_without_evidence_tag_v1`** — Empirical claim ("saves N%" / "improves N%" / "beats baseline" / "verified") in docstring/report/script without adjacent `[empirical:<artifact>]` / `[contest-CUDA]` / `[prediction]` axis tag. Lane PD docstring stated 49% savings; empirical regression test caught 18.5%. Canonical unwind: tag every claim with canonical axis token per Catalog #287/#323. Severity: medium_substrate_regression.

## Wire-in surfaces (next-cycle implementation queue)

### Layer 1 — Canonical helper package `src/tac/canonical_anti_patterns/`

Structure:
```
src/tac/canonical_anti_patterns/
├── __init__.py
├── anti_pattern.py            # AntiPattern + EmpiricalFalsification frozen dataclasses
├── registry.py                # register_anti_pattern + append_empirical_falsification + auto_recalibrate
├── pattern_matcher.py         # match_stack_against_anti_patterns(stack_spec) -> tuple[AntiPattern, ...]
├── compounding_order_validator.py  # validate_compound_stack_order(ops) -> ValidationResult
├── builtins.py                # initial population of 12 anti-patterns per CLAUDE.md FORBIDDEN_PATTERNS
└── tests/
    └── test_registry.py       # ≥30 tests
```

### Layer 2 — Cathedral consumer auto-discovery per Catalog #335

`src/tac/cathedral_consumers/anti_pattern_lookup_consumer/__init__.py`:
- `CONSUMER_NAME = "anti_pattern_lookup_consumer"`
- `CONSUMER_VERSION = "1.0.0"`
- `CONSUMER_HOOK_NUMBERS = (2, 6)` — hook #2 Pareto constraint (anti-patterns EXCLUDE feasibility regions); hook #6 probe-disambiguator (anti-pattern match IS the canonical disambiguator)
- `CONSUMER_TIER = "TIER_A_OBSERVABILITY_ONLY"` per Catalog #357
- `update_from_anchor` reads new EmpiricalFalsifications from continual-learning posterior
- `consume_candidate` matches candidate's proposed stack against registered anti-patterns; emits observability-only verdict (matches: bool / matched_anti_patterns: tuple / canonical_unwind_path_recommended: str / severity: enum)

### Layer 3 — STRICT preflight gate (claim canonical # via Catalog #186)

`check_compound_stack_proposal_acknowledges_known_anti_patterns`:
- Scans landing memos under `.omx/research/*_landed_<YYYYMMDD>.md` dated >= 2026-05-29 (post-design-memo-landing cutoff per "Strict-flip atomicity rule")
- For each memo proposing a compound stack (regex: `compound stack` / `compounding order` / `stacking` / `combined` / sister patterns), checks if the memo body references the canonical anti-patterns registry per `tac.canonical_anti_patterns.match_stack_against_anti_patterns(...)` output
- Refuses memos that propose stacks matching known anti-patterns WITHOUT either:
  - (a) Citing the matched anti-pattern + applying its canonical unwind path
  - (b) Same-line waiver `# ANTI_PATTERN_MATCH_INTENTIONAL_OK:<rationale>` with substantive non-placeholder rationale (≥4 chars; placeholder `<rationale>` / `<reason>` literals rejected per Catalog #287 sister discipline)
  - (c) Empirical falsification appended that explicitly RATIFIES the anti-pattern in this context (sister `EVENT_FALSIFICATION_APPENDED`)
- STRICT-from-byte-one per CLAUDE.md "Strict-flip atomicity rule"; live count at landing: 0 (anti-pattern registry not yet consulted; cutoff exempts pre-landing memos)
- Sister of Catalog #344 (canonical equations memo-reference enforcement); Catalog #287 (placeholder rationale rejection)

### Layer 4 — Auto-recalibrator (Catalog #371 sister)

`tac.canonical_anti_patterns.registry.auto_recalibrate_from_continual_learning_posterior`:
- For each anti-pattern with `next_recalibration_trigger=when_3+_new_empirical_falsifications_in_domain` AND falsification count >= 3
- Re-derives `falsification_band` directly from anti-pattern's own landed EmpiricalFalsifications (latest-wins per `measurement_method` axis)
- Appends canonical `EVENT_ANTI_PATTERN_RECALIBRATED` row if recomputed band differs from stored
- Sister of canonical equation #344 + Catalog #371 fix (NOT a no-op stub; idempotent on second run)

### Layer 5 — Pareto polytope solver integration (Slot 1 Dykstra in flight)

When Slot 1 Dykstra Pareto polytope solver lands, extend `invoke_dykstra_pareto_solver_on_candidates`:
- For each candidate's proposed compound stack, consult `tac.canonical_anti_patterns.match_stack_against_anti_patterns(stack_spec)`
- If matched anti-pattern WITHOUT waiver: EXCLUDE from feasibility set (constraint added to polytope: `indicator(stack_matches_anti_pattern_j) ≤ 0`)
- Surface excluded region in ranker observability per Catalog #305
- Per-axis dual variable interpretation: if anti-pattern constraint is TIGHT (λ_anti-pattern > 0), the canonical unwind path becomes the routed next-cycle attack direction

This is the canonical METHOD that converts anti-patterns from PROSE warnings into ACTIVE polytope constraints — operationalizing the operator's "higher layer of abstraction" directive.

## 6-Hook Wire-In Declaration per Catalog #125

- hook #1 sensitivity-map: anti-pattern severity contributes to per-substrate sensitivity ranking (high-severity anti-patterns dominate downstream consumer routing)
- hook #2 Pareto constraint: PRIMARY (anti-patterns EXCLUDE regions from Pareto polytope feasibility set; sister of canonical equation predictions ADDING constraints)
- hook #3 bit-allocator: anti-pattern unwind paths route bit-allocator AWAY from known compounding-order anti-patterns (e.g. quantize-then-SVD)
- hook #4 cathedral autopilot dispatch: auto-discovered via Catalog #335 sister gate
- hook #5 continual-learning posterior: EmpiricalFalsifications append to canonical posterior; canonical anti-pattern recalibrates per Catalog #371-sister auto-recalibrator
- hook #6 probe-disambiguator: matched anti-pattern + canonical unwind path IS the canonical disambiguator between viable vs forbidden compounding routes

## Cargo-cult audit per assumption (Catalog #303)

| Assumption | Classification | Rationale |
|---|---|---|
| Anti-pattern severity has consistent ranking across substrates | CARGO-CULTED-NEEDS-EMPIRICAL | severity may vary by substrate; per-substrate empirical falsifications refine |
| Compound aggregation is MAX over applicable anti-patterns | HARD-EARNED | one violated anti-pattern can corrupt entire stack (sister of softmax-vs-max in worst-case analysis) |
| Anti-patterns and canonical equations are SYMMETRIC frameworks | HARD-EARNED | operator's META insight; mirrors canonical_equations API at every layer |
| Initial 12 anti-patterns are exhaustive | CARGO-CULTED-EMPIRICALLY-WILL-EXPAND | the 20+ CLAUDE.md FORBIDDEN_PATTERNS section is the canonical extension queue; this design covers the highest-EV initial population |
| Anti-pattern unwind paths are deterministic | CARGO-CULTED-NEEDS-EMPIRICAL | unwind path may require substrate-specific adaptation; per-substrate falsification rows refine |

## 9-dimension success checklist evidence per Catalog #294

| Dimension | Evidence |
|---|---|
| UNIQUENESS | Symmetric of canonical_equations; novel apparatus extension per operator META directive |
| BEAUTY + ELEGANCE | Mirrors `tac.canonical_equations` API at every layer; PR101-style ≤500 LOC initial implementation |
| DISTINCTNESS | DIFFERENT from per-instance Catalog gates; HIGHER abstraction layer captures CLASS-level pattern |
| RIGOR | premise verification + 0 existing modules + 191 canonical equation rows analyzed + 20+ CLAUDE.md FORBIDDEN_PATTERNS enumerated |
| OPTIMIZATION-PER-TECHNIQUE | Per-anti-pattern severity + per-substrate empirical falsification refinement |
| STACK-OF-STACKS-COMPOSABILITY | Compounds with Slot 1 Dykstra Pareto polytope solver as ACTIVE constraints |
| DETERMINISTIC-REPRODUCIBILITY | fcntl-locked JSONL + canonical Provenance + APPEND-ONLY |
| EXTREME-OPTIMIZATION-PERFORMANCE | O(n_anti_patterns) match per candidate; pattern_matcher uses hashable signatures |
| OPTIMAL-MINIMAL-CONTEST-SCORE | Steers cathedral autopilot ranker AWAY from anti-pattern compounding stacks → preserves rate-axis/quality-axis/score-axis from regression |

## Observability surface per Catalog #305

- **Inspectable per layer**: each AntiPattern + EmpiricalFalsification queryable via registry; cathedral consumer surfaces per-candidate verdict
- **Decomposable per signal**: per-anti-pattern severity / per-substrate manifestation / per-axis regression contribution
- **Diff-able across runs**: canonical Provenance + APPEND-ONLY ledger
- **Queryable post-hoc**: `tac.canonical_anti_patterns.query_anti_patterns_by_substrate(substrate_id)` / `query_falsifications_by_paradigm_class(class)` / `query_recurrence_rate_by_severity(severity)`
- **Cite-able**: every anti-pattern carries `canonical_source_anchor` (CLAUDE.md section / Catalog # / sister memo path); every falsification carries `empirical_artifact_path`
- **Counterfactual-able**: per-anti-pattern unwind path enables "what if we apply canonical unwind?" simulation via Slot 1 Dykstra solver re-projection with anti-pattern constraint removed

## Predicted-band Dykstra-feasibility per Catalog #296

Per Slot 1 Dykstra Pareto polytope solver wire-in: this framework's anti-pattern constraints become ACTIVE per-axis polytope exclusions. Dykstra alternating projections compute optimal compound BOTH considering canonical equation predictions (positive constraints adding regions to feasibility set) AND anti-pattern matches (negative constraints excluding regions). Per-axis tight dual variables identify whether the constraining axis is positive (canonical equation prediction binding) or negative (anti-pattern match binding) — routing next-cycle attack accordingly.

## Mission contribution per Catalog #300

`apparatus_maintenance + frontier_breaking_enabler`:
- **apparatus_maintenance**: canonicalizes the 20+ CLAUDE.md FORBIDDEN_PATTERNS prose narrative into queryable system + symmetric of canonical equations + cathedral consumer auto-discovery; addresses operator's "higher layer of abstraction" directive
- **frontier_breaking_enabler**: when wired into Slot 1 Dykstra Pareto polytope solver as ACTIVE constraints + cathedral autopilot ranker routing, prevents future compound stacking work from re-discovering anti-patterns one paid GPU dispatch at a time

## Implementation queue (when Slot frees)

**Wave N+1 (post Slot 1 Dykstra + Slot 2 V3 int8 land)**:
- Spawn 1: implement `tac.canonical_anti_patterns` package (Layer 1) + cathedral consumer (Layer 2) + initial population of 12 anti-patterns + tests
- Spawn 2: continue empirical work per existing Wave N+1 queue (Compound C heterogeneous bit allocation OR Compound F orthogonal composition test)

**Wave N+2**:
- Spawn 1: STRICT preflight gate (Layer 3) + Slot 1 Dykstra solver integration (Layer 5) + auto-recalibrator (Layer 4)
- Spawn 2: extend initial population to ~20+ anti-patterns covering full CLAUDE.md FORBIDDEN_PATTERNS section

## Cross-references

- CLAUDE.md "Canonical equations + models registry — NON-NEGOTIABLE" (Catalog #344) — the POSITIVE sister registry this design mirrors
- CLAUDE.md "FORBIDDEN PATTERNS — NON-NEGOTIABLE, READ BEFORE WRITING ANY CODE" — the prose narrative this design canonicalizes
- CLAUDE.md "Meta-Lagrangian/Pareto solver — NON-NEGOTIABLE, HIGHEST EMPHASIS" — Slot 1 Dykstra solver consumes this design's constraints
- CLAUDE.md "Results must become system intelligence — NON-NEGOTIABLE, HIGHEST EMPHASIS" — anti-patterns must become system intelligence too
- CLAUDE.md "Subagent coherence-by-default — NON-NEGOTIABLE, HIGHEST EMPHASIS" — cross-session knowledge preservation via canonical registry
- Catalog #335 cathedral consumer canonical contract
- Catalog #371 auto-recalibrator sister pattern
- Catalog #245 canonical 4-layer pattern (canonical helper + CLI + STRICT preflight gate + runtime ledger integration)
- Catalog #131/#138 fcntl-locked + strict-load JSONL discipline
- Catalog #287 placeholder-rationale rejection
- Catalog #307 paradigm-vs-implementation classification
- Catalog #323 canonical Provenance umbrella
- Catalog #346 canonical council roster (the EmpiricalFalsification needs T2+ council deliberation per Catalog #300)
- Wave N+1 compound stacking analysis (in-conversation; identified 7 forbidden compounding anti-patterns)
- Slot 1 Dykstra Pareto polytope solver wire-in (in flight; consumer of this framework)
- Slot 2 V3 int8 decoder compression (in flight; potential first empirical anchor candidate of unwound anti-pattern)
- Sister landings: V3 RE-RUN commit `b01232473` (cross-paradigm-without-per-axis anti-pattern empirical anchor); NSCS06 v8 chroma_lut commit `6e5437f48` (CASCADE_SATURATION_REFUTED EMPIRICAL anchor); decoder compression analysis commit `44d12e75d` (LZMA-on-brotli SATURATED EMPIRICAL anchor)

## APPEND-ONLY footer per Catalog #110/#113 HISTORICAL_PROVENANCE

This design memo is a DESIGN_MEMO not a LANDING_MEMO. Implementation is queued for Wave N+1 when current Slot 1 (Dykstra `a0cfddc196d765c74`) + Slot 2 (V3 int8 `a393304466b2e48fd`) complete. The design here is canonical reference; any deviation during implementation requires explicit `# DESIGN_DEVIATION_OK:<rationale>` waiver in the landing memo per Catalog #287 sister discipline.

Per operator's standing directive 2026-05-28 verbatim: *"learning anti-patterns is upser important too for compounding continual learning, like the canonical equations bu netgative and a higher layer of abstraction"* — this design operationalizes the META insight via the canonical equations sister framework at the negative-pattern + higher-abstraction-layer surface.

Mission contribution per Catalog #300: `frontier_breaking_enabler` (canonical apparatus extension that compounds across substrate work + sessions + atom/element/component granularity per operator's full-stack lens).
