# Codex Routing Directive — Dynamic Per-Candidate Composition Helper Canonical Build

**Date**: 2026-05-18
**Routing destination**: Codex (canonical helper builder)
**Parent memo**: `.omx/research/dynamic_per_candidate_composition_framework_all_canonical_apparatus_composed_20260518.md` (2928 lines; T3 grand council PROCEED_WITH_REVISIONS)
**Lane**: `lane_dynamic_per_candidate_composition_framework_20260518`
**Mission alignment**: frontier_breaking per CLAUDE.md "Mission alignment — non-negotiable"
**Operator directive**: "all operator decisions approved" + "compose ALL canonical apparatus per-candidate per-asymptotic-approach with anti-arbitrariness"

---

## 0. Build target summary

Build the canonical helper package `src/tac/dynamic_per_candidate_composition/` per the parent memo §17.

**Total estimated**:
- 11 implementation files (~2300 LOC)
- 1 test file (~1000 LOC)
- 3 sister CLIs (~600 LOC each = 1800 LOC)
- Total: ~5100 LOC across ~15 files

## 1. Mandatory pre-flight reads

Per CLAUDE.md "Subagent coherence-by-default" non-negotiable + Catalog #229 premise verification:

1. `/Users/adpena/Projects/pact/CLAUDE.md` (FULL; every NON-NEGOTIABLE)
2. `/Users/adpena/Projects/pact/AGENTS.md`
3. `.omx/research/dynamic_per_candidate_composition_framework_all_canonical_apparatus_composed_20260518.md` (parent memo; 2928 lines)
4. `.omx/research/cross_stack_synthesis_9_design_landings_unified_framework_20260518.md` (1449 lines; 4-tier bilevel canonical action)
5. `.omx/research/meta_audit_conflate_declarative_with_physical_error_pattern_12_claim_self_audit_20260518.md` (CONFLATE_DECLARATIVE_WITH_PHYSICAL extinction)
6. `.omx/research/rate_attack_legal_receiver_path_audit_codex_f1_finding_relay_20260518.md` (Codex F1 finding + A1 RECLAIMABLE_VIA_PACKET_COMPILER)
7. `.omx/research/rate_attack_synthesis_v2_reconciliation_*.md` (SYNTHESIS-V2 reconciled TOP-5)

## 2. Build sequence (5 phases per parent memo §17.2)

### Phase 1: Layer 1 anti-arbitrariness foundation (~600 LOC)

Build files:
- `src/tac/dynamic_per_candidate_composition/__init__.py` (~50 LOC; public API surface)
- `src/tac/dynamic_per_candidate_composition/legal_receiver_path_classifier.py` (~200 LOC; 4-class taxonomy NO_RECEIVER_NEEDED / LEGAL_RECEIVER_IN_BUDGET / RECLAIMABLE_VIA_PACKET_COMPILER / STRICT_SCORER_RULE_VIOLATION per Catalog #6 + HNeRV parity L4)
- `src/tac/dynamic_per_candidate_composition/anti_arbitrariness_foundation.py` (~250 LOC; 5-form per-primitive evidence enforcer per parent memo §6)

**API contract**:
```python
def classify_legal_receiver_path(
    primitive: CompositionPrimitive,
    *,
    loc_budget: int = 200,
    dep_budget: int = 2,
) -> LegalReceiverPathClassification:
    """Per parent memo §6.1.(d) 4-class taxonomy."""

def validate_per_primitive_evidence(
    primitive: CompositionPrimitive,
    *,
    archive_sha256: str,
) -> PerPrimitiveEvidenceValidation:
    """Per parent memo §6.1 5-form evidence: (a) HARD-EARNED/CARGO-CULTED + (b) empirical anchor cite + (c) per-candidate adaptation + (d) legal-receiver-path + (e) Dykstra-feasibility."""
```

### Phase 2: Layer 2 primitives registry (~350 LOC)

Build files:
- `src/tac/dynamic_per_candidate_composition/composition_primitive.py` (~150 LOC; typed enum + per-primitive metadata for the 14 primitives per parent memo §3)
- `src/tac/dynamic_per_candidate_composition/primitives_registry.py` (~200 LOC; registry with per-primitive 5-form evidence reference)

**API contract**:
```python
class CompositionPrimitive(IntEnum):
    """14 canonical primitives per parent memo §3."""
    MASTER_GRADIENT = 1
    VENN_CLASSIFIER = 2
    GRANULARITY_LAYER = 3
    SENSITIVITY_MAP = 4
    COMPOSITION_ALPHA = 5
    WYNER_ZIV_DELIVERABILITY = 6
    PROBE_OUTCOMES_LEDGER = 7
    XRAY_OBSERVABILITY = 8
    CATHEDRAL_AUTOPILOT_V2 = 9
    NULL_SPACE_EXPLOITER = 10
    PROCEDURAL_CODEBOOK = 11
    FREEZING_EXPLOITS = 12
    A1_SPECIALIZED_BINARY = 13
    PER_AXIS_HARDWARE_MATRIX = 14

def get_primitive_metadata(primitive: CompositionPrimitive) -> PrimitiveMetadata:
    """Returns canonical helper location + API signature + per-primitive evidence per parent memo §3."""
```

### Phase 3: Layer 3 bilevel optimizer (~400 LOC)

Build files:
- `src/tac/dynamic_per_candidate_composition/bilevel_optimizer_4_piece.py` (~400 LOC; 4-piece composition per parent memo §4)

**API contract**:
```python
def run_outer_tier(
    archive_sha256: str,
    applicable_primitives: list[CompositionPrimitive],
    asymptotic_approach: AsymptoticApproach,
    budget_usd: float,
) -> OuterTierVerdict:
    """Per parent memo §4.2."""

def run_middle_tier(
    outer_verdict: OuterTierVerdict,
    asymptotic_approach: AsymptoticApproach,
    archive_sha256: str,
) -> MiddleTierVerdict:
    """Per parent memo §4.3."""

def run_inner_tier(
    middle_verdict: MiddleTierVerdict,
    archive_sha256: str,
    master_gradient: MasterGradientAnchor,
    venn_classification: NSetVennClassification,
) -> InnerTierVerdict:
    """Per parent memo §4.4."""

def run_innermost_tier(
    inner_verdict: InnerTierVerdict,
    archive_sha256: str,
    master_gradient: MasterGradientAnchor,
    venn_classification: NSetVennClassification,
) -> InnermostTierVerdict:
    """Per parent memo §4.5."""
```

### Phase 4: Layer 4 per-candidate orchestration (~700 LOC)

Build files:
- `src/tac/dynamic_per_candidate_composition/per_candidate_orchestration.py` (~300 LOC; `compose_optimal_per_candidate(...)` + caching)
- `src/tac/dynamic_per_candidate_composition/composition_plan.py` (~250 LOC; CompositionPlan dataclass + builders)
- `src/tac/dynamic_per_candidate_composition/composition_plan_persistence.py` (~150 LOC; fcntl-locked JSONL persistence per Catalog #131)

**API contract**:
```python
def compose_optimal_per_candidate(
    candidate_archive_sha256: str,
    candidate_lane_id: str,
    asymptotic_approach: AsymptoticApproach,
    budget_usd: float = 1.0,
    repo_root: Path | None = None,
) -> CompositionPlan:
    """Per parent memo §0 executive summary + §5 detailed specification."""

def persist_composition_plan_to_ledger(
    plan: CompositionPlan,
    *,
    repo_root: Path | None = None,
) -> None:
    """Persist per Catalog #131 fcntl-locked JSONL append-only sister to Catalog #245 Modal call_id ledger."""
```

### Phase 5: Public API + tests (~1000 LOC tests)

Build files:
- `src/tac/dynamic_per_candidate_composition/composition_plan_observability.py` (~200 LOC; §13 observability surface API per Catalog #305)
- `src/tac/dynamic_per_candidate_composition/composition_alpha_n_way.py` (~150 LOC; primitive 5 enforcement + Dykstra-feasibility check)
- `src/tac/tests/test_dynamic_per_candidate_composition.py` (~1000 LOC; per-primitive evidence validation + per-tier verdict construction + per-candidate plan caching + 4-process fcntl-locked concurrent persistence + 6-hook wire-in regression tests)

## 3. Tier-1 engineering primitives applicability per CLAUDE.md Catalog #270

Per parent memo §17.3: the canonical helper is pure compress-time analysis + orchestration; no Tier-1 training primitives apply.

- autocast_fp16: N/A
- TF32: N/A
- torch.compile: N/A
- no_grad-at-eval: N/A
- GTScorerCache F3: N/A
- canonical scorer-loss helper routing: N/A

Catalog #270 protocol IS satisfied (no training in this helper; all primitives apply vacuously).

## 4. Cross-helper integration points (per parent memo §17.4)

Build the helper to integrate with:

| Existing canonical helper | Used for | API call |
|---|---|---|
| `tac.master_gradient` | primitive 1 lookup | `latest_anchor_for_archive(archive_sha256, axis=...)` or `query_anchors_by_archive(archive_sha256)` |
| `tac.wyner_ziv_deliverability.proof_builder` | primitive 6 lookup | `load_deliverability_proof_for_archive(archive_sha256, repo_root)` |
| `tools.cathedral_autopilot_autonomous_loop` | primitive 9 v2 cascade | `adjust_predicted_delta_for_venn_classification_v2(...)` per Catalog #319 Q3 |
| `tac.null_space_exploiter` | primitive 10 | `build_null_space_basis(...)`, `plan_null_space_byte_reduction(...)`, `project_modifications_onto_null_space(...)` |
| `tac.procedural_codebook_generator` | primitive 11 | `emit_seed(...)`, `expand_seed_to_codebook(...)`, `derive_codebook_from_archive_bytes(...)`, `freeze_source_member_sha256(...)` |
| `tac.optimization.substrate_composition_matrix` | primitive 5 | composition_alpha N-way per Catalog #322 |
| `tac.sensitivity_map` | primitive 4 | per-cell aggregated gradient |
| `tac.xray` | primitive 8 | per-layer observability |
| `tac.probe_outcomes_ledger` | primitive 7 | `latest_blocking_outcome_by_substrate(...)` per Catalog #313 |
| `tac.council_continual_learning` | hook #5 | `append_council_anchor(record)` per Catalog #300 |
| `tools/operator_authorize.py` | Layer 4 runtime gate | dispatch routing |
| `tools/cathedral_autopilot_autonomous_loop.py` | primitive 9 | rerank consumer |

**API verification note (2026-05-18)**: these integration symbols were
grep-verified after the first draft. Treat any function names in sections 1-2
as build-target API sketches unless the implementation already exports them;
do not copy prose-only symbol names into code without `rg`/`inspect`
verification.

## 5. Catalog #229 premise verification (5 PVs before edit)

Per CLAUDE.md "Forbidden prompt premise verification" + Catalog #229:

**PV-1**: master_gradient canonical helper EXISTS at `src/tac/master_gradient.py` (verified: 28678 bytes per `ls -la`)

**PV-2**: wyner_ziv_deliverability canonical helper EXISTS at `src/tac/wyner_ziv_deliverability/proof_builder.py` (verified: 49407 bytes)

**PV-3**: null_space_exploiter canonical helper EXISTS at `src/tac/null_space_exploiter/` (verified: 16829 bytes core.py)

**PV-4**: procedural_codebook_generator canonical helper EXISTS at `src/tac/procedural_codebook_generator/` (verified: 6178 + 3515 bytes per generator)

**PV-5**: probe_outcomes_ledger canonical helper EXISTS at `src/tac/probe_outcomes_ledger.py` (verified per CLAUDE.md Catalog #313 row)

**All 5 PVs PASS**. Proceed with build.

## 6. Catalog #206 checkpoint discipline

Per CLAUDE.md "Mandatory crash-resume protocol" + Catalog #206:

- At start: `tools/subagent_checkpoint.py read --subagent-id <YOUR_ID>` to check predecessor; resume if exists
- Every ~10 tool uses (or after each major milestone): `tools/subagent_checkpoint.py --subagent-id <YOUR_ID> --step <N> --status in_progress --files-touched <...> --next-action <...>`
- On completion: `tools/subagent_checkpoint.py --subagent-id <YOUR_ID> --step complete --status complete --files-touched <...> --next-action ""`

Canonical store: `.omx/state/subagent_progress.jsonl` (fcntl-locked per Catalog #131).

## 7. Catalog #117/#157/#174 commit serializer discipline

Per CLAUDE.md "Subagent commits MUST use serializer" + Catalog #117 + Catalog #157 + Catalog #174:

For EVERY commit:

```bash
# After all edits:
SHA1=$(shasum -a 256 src/tac/dynamic_per_candidate_composition/__init__.py | awk '{print $1}')
SHA2=$(shasum -a 256 src/tac/dynamic_per_candidate_composition/legal_receiver_path_classifier.py | awk '{print $1}')
# ... per file

.venv/bin/python tools/subagent_commit_serializer.py \
    --message "dynamic-per-candidate-composition: Phase 1 land Layer 1 anti-arbitrariness foundation" \
    --files src/tac/dynamic_per_candidate_composition/__init__.py src/tac/dynamic_per_candidate_composition/legal_receiver_path_classifier.py \
    --expected-content-sha256 "src/tac/dynamic_per_candidate_composition/__init__.py=${SHA1}" \
    --expected-content-sha256 "src/tac/dynamic_per_candidate_composition/legal_receiver_path_classifier.py=${SHA2}"
```

**Per Catalog #174**: every commit MUST pass `--expected-content-sha256` for EVERY file. Bare commits are REFUSED.

## 8. 6-hook wire-in declaration per Catalog #125

The canonical helper itself emits a 6-hook wire-in per Catalog #125:

1. **Sensitivity-map contribution**: PRIMARY (helper consumes master_gradient + sensitivity_map; emits per-pair sensitivity-map evidence in every CompositionPlan)
2. **Pareto constraint**: PRIMARY (helper enforces Dykstra-feasibility intersection per Catalog #296; CompositionPlan carries DykstraFeasibilityVerdict)
3. **Bit-allocator hook**: ACTIVE (Layer 3 INNER + INNERMOST tiers allocate bits per Venn cell + per hard-pair atlas)
4. **Cathedral autopilot dispatch hook**: PRIMARY (helper IS the cascade consumer per Catalog #319 Q3 v2)
5. **Continual-learning posterior update**: ACTIVE (Layer 4 §5.9 emits council-deliberation anchor per Catalog #300)
6. **Probe-disambiguator**: ACTIVE (helper consults predecessor probe-outcomes per Catalog #313)

## 9. 9-dim checklist evidence per Catalog #294

Per CLAUDE.md "9-dimension success checklist evidence" + Catalog #294: the canonical helper's design memo (parent memo §12) carries the 9-dim checklist evidence. The helper inherits per the parent memo:

1. UNIQUENESS: framework-distinctness per per-candidate META-orchestration
2. BEAUTY+ELEGANCE: 4-layer apparatus stack
3. DISTINCTNESS: distinct from sister cross-stack synthesis + hypergraph + SYNTHESIS-V2
4. RIGOR: 5 binding revisions + 12 PVs cargo-cult audit
5. OPTIMIZATION-PER-TECHNIQUE: per parent memo §15 canonical-vs-unique decision per layer
6. STACK-OF-STACKS-COMPOSABILITY: composition_alpha N-way primitive 5
7. DETERMINISTIC-REPRODUCIBILITY: deterministic per fixed inputs + byte-stable persistence
8. EXTREME-OPTIMIZATION-PERFORMANCE: ~5-30s per plan construction
9. OPTIMAL-MINIMAL-CONTEST-SCORE: predicted aggregate ΔS per asymptotic_approach

## 10. Observability surface declaration per Catalog #305

Per CLAUDE.md "Max observability — non-negotiable" + Catalog #305: the canonical helper carries 6-facet observability per parent memo §13:

1. Inspectable per layer (Layer 1-4 each inspect-able via `inspect_*` API)
2. Decomposable per signal (per-primitive contributions)
3. Diff-able across runs (sister CLI `tools/diff_composition_plans.py`)
4. Queryable post-hoc (sister CLI `tools/query_composition_plans.py` + `.omx/state/composition_plans/_index.jsonl`)
5. Cite-able (every field carries source cite per Catalog #287)
6. Counterfactual-able (sister CLI `tools/counterfactual_composition_plan.py`)

## 11. Anti-CONFLATE_DECLARATIVE_WITH_PHYSICAL discipline

Per parent memo §7: the canonical helper's Layer 1 enforces 5 structural mechanisms that extinct the pattern:

1. Mechanism 1: Legal-receiver-path classifier per primitive (`classify_legal_receiver_path(...)`)
2. Mechanism 2: Per-primitive empirical anchor cite per Catalog #287
3. Mechanism 3: Dykstra-feasibility intersection per Catalog #296
4. Mechanism 4: HARD-EARNED vs CARGO-CULTED classification per Catalog #303
5. Mechanism 5: Sister STRICT preflight gate `check_composition_plan_carries_per_primitive_evidence` (queued per parent memo §16 OP-FRAMEWORK-8)

The canonical helper's `build_composition_plan(...)` REFUSES plans where any primitive lacks any of the 5 forms.

## 12. Operator-approval status

**Operator directive 2026-05-18 verbatim**: "all operator decisions approved".

This routing directive is operator-approved. Codex proceeds.

## 13. Sister deliverables (queued)

Per parent memo §16:

- OP-FRAMEWORK-7: observability sister CLIs (`tools/diff_composition_plans.py` + `tools/query_composition_plans.py` + `tools/counterfactual_composition_plan.py`) — queue post-canonical-helper landing
- OP-FRAMEWORK-8: STRICT preflight gate `check_composition_plan_carries_per_primitive_evidence` — queue post-canonical-helper landing
- A-3 Dykstra-feasibility-FIRST canonical helper (`src/tac/optimization/dykstra_feasibility_first.py`) — sister build per SYNTHESIS-V2 RECONCILED-5
- A-4 MI-min Wyner-Ziv canonical helper (`src/tac/rate_attack_mi_lower_bound/`) — sister build per SYNTHESIS-V2 RECONCILED-5

## 14. Expected harvest

When build completes:

- Canonical helper at `src/tac/dynamic_per_candidate_composition/` (~5100 LOC across ~15 files)
- Test suite passing at `src/tac/tests/test_dynamic_per_candidate_composition.py`
- 6-hook wire-in regression test passing
- 9-dim checklist evidence inherited from parent memo
- Observability surface verified via sister CLI smoke tests

## 15. Lane registry update

Per Catalog #126 + Catalog #220:
- `lane_dynamic_per_candidate_composition_framework_20260518` advances to L2 when canonical helper lands + first per-candidate plan empirically demonstrates aggregate predicted ΔS within band on paired Linux x86_64 [contest-CPU] anchor

— Routing directive for DYNAMIC-PER-CANDIDATE-COMPOSITION-FRAMEWORK-2026-05-18
