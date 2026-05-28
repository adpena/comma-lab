---
council_tier: T1
council_attendees:
  - claude
council_quorum_met: true
council_verdict: PROCEED
council_dissent: []
council_assumption_adversary_verdict:
  - assumption: "Mirroring tac.canonical_equations API surface exactly is the right abstraction"
    classification: HARD-EARNED
    rationale: |
      The canonical_equations package is the canonical positive registry surface
      that the canonical_anti_patterns is the symmetric negative half of (per
      operator META directive verbatim: "like the canonical equations bu netgative
      and a higher layer of abstraction"). Mirroring its dataclass shape +
      registry API + persistence layer + auto-recalibrator pattern gives future
      readers the positive/negative symmetry at a glance. ADOPT_CANONICAL_BECAUSE_SERVES
      per CLAUDE.md UNIQUE-AND-COMPLETE-PER-METHOD falling-rule.
  - assumption: "MAX aggregation (not SUM) is the correct compounding identity"
    classification: HARD-EARNED
    rationale: |
      Per design memo §"Mathematical compounding identity": anti-patterns are
      typically dominated by the WORST applicable pattern (one violated anti-
      pattern can kill the whole stack regardless of how many other anti-
      patterns are clear). SUM would double-count + dilute critical signal.
      MAX is the canonical aggregation for safety-constraint composition in
      Pareto polytope feasibility set construction.
  - assumption: "Tier A (observability-only) is the correct tier for the consumer"
    classification: HARD-EARNED
    rationale: |
      Per Catalog #341 + #357: a matched anti-pattern surfaces a ROUTING
      RECOMMENDATION (apply canonical unwind path), NOT a score signal.
      Score-mutating contributions require empirically-grounded contest
      evidence per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA".
      Tier A keeps the consumer safe-by-construction; Wave N+2 Layer 5
      Slot 1 Dykstra solver integration consumes the matches as ACTIVE
      polytope constraints (a different surface from score-signal promotion).
council_decisions_recorded:
  - "Layer 1 (canonical helper package) + Layer 2 (cathedral consumer) landed"
  - "12 initial anti-patterns registered to live .omx/state/canonical_anti_patterns_registry.jsonl"
  - "Consumer count 73 -> 74 (auto-discovery via Catalog #335 verified)"
  - "Canonical equation canonical_anti_patterns_compounding_aggregation_v1 registered"
  - "Probe outcome ledger row registered per Catalog #313 (30-day staleness window)"
  - "Wave N+2 deferred: Layer 3 STRICT gate + Layer 4 sister auto-recalibrator wiring + Layer 5 Slot 1 Dykstra integration"
council_predicted_mission_contribution: frontier_breaking_enabler
council_override_invoked: false
council_override_rationale: null
---

# Canonical Anti-Patterns Registry — Layer 1 + Layer 2 landed (2026-05-28)

**Status**: LANDED (impl_complete + strict_preflight_queued_for_wave_n_plus_2 + memory_entry)
**Lane**: `lane_canonical_anti_patterns_layer_1_plus_2_20260528` L1
**Sister design memo**: `.omx/research/canonical_anti_patterns_registry_design_20260528.md` (commit 37b5a0184)
**Sister Slot 1 in flight (DISJOINT scope)**: Dykstra Pareto polytope solver `a0cfddc196d765c74`

## Operator directive

Per operator META directive 2026-05-28 verbatim: *"learning anti-patterns is upser important too for compounding continual learning, like the canonical equations bu netgative and a higher layer of abstraction"*.

## Implementation summary

### Layer 1 — `src/tac/canonical_anti_patterns/` package landed

Structure (mirrors `src/tac/canonical_equations/`):

- `anti_pattern.py` (~530 LOC): `AntiPattern` + `EmpiricalFalsification` frozen dataclasses with `__post_init__` invariants (snake_case_vN id pattern; NaN/negative residual rejection; orphan refused; paradigm/severity enum validation; canonical Provenance required).
- `registry.py` (~530 LOC): fcntl-locked APPEND-ONLY JSONL persistence + `register_anti_pattern` / `append_empirical_falsification` / `load_anti_patterns_strict` / `query_*` / `auto_recalibrate_from_continual_learning_posterior` (NON-STUB per Catalog #371 lesson — actually re-derives `falsification_band` from landed falsifications when count >= 3; idempotent on 2nd run; emits `EVENT_ANTI_PATTERN_RECALIBRATED`).
- `pattern_matcher.py` (~290 LOC): `match_stack_against_anti_patterns(stack_spec) -> tuple[AntiPatternMatch, ...]` + `validate_compound_stack_order(ops) -> ValidationResult`. Matcher walks recurrence_conditions + forbidden_pattern_predicate against a flattened stack_spec haystack; ordered by severity (CRITICAL first) then descending confidence.
- `builtins.py` (~700 LOC): 12 canonical initial anti-patterns fully populated with description + forbidden_pattern_predicate + falsification_band + recurrence_conditions + canonical_source_anchor + canonical_unwind_path + canonical_producers + canonical_consumers + paradigm_class + severity + canonical Provenance per Catalog #323.
- `__init__.py` (~135 LOC): narrow public API export (all canonical names).
- `tests/test_registry.py` (~600 LOC): 44 dedicated tests covering dataclass invariants (rejections + happy path) + registry persistence + lifecycle + query helpers + auto-recalibrator (NON-STUB regression vs Catalog #371) + pattern matcher + order validator + 4-process concurrency stress.

### Layer 2 — `src/tac/cathedral_consumers/anti_pattern_lookup_consumer/` package landed

- `__init__.py` (~170 LOC): canonical Catalog #335 CathedralConsumerContract compliance (CONSUMER_NAME / CONSUMER_VERSION / CONSUMER_HOOK_NUMBERS=(PARETO_CONSTRAINT, PROBE_DISAMBIGUATOR) / CONSUMER_TIER=TIER_A_OBSERVABILITY_ONLY). `consume_candidate` resolves stack_spec from candidate (canonical key + sister synonym + fallback) + calls pattern matcher + returns canonical routing markers per Catalog #341 (predicted_delta_adjustment=0.0 / promotable=False / axis_tag="[predicted]") + matched_anti_patterns + canonical_unwind_paths_recommended + provenance.
- `tests/test_consumer.py` (~180 LOC): 15 dedicated tests covering Catalog #335 contract compliance + hook number coverage + Catalog #357 Tier A + Catalog #341 routing markers + auto-discovery + matching behavior + provenance shape.

### Auto-discovery verification

Pre-landing: 73 consumers (compliant).
Post-landing: **74 consumers (compliant)**. The new `anti_pattern_lookup_consumer` is auto-discovered via `tools/cathedral_autopilot_autonomous_loop.discover_and_register_consumers()` and registered in the cathedral autopilot ranker cascade WITHOUT any manual `cathedral_autopilot_autonomous_loop.py` edits (Catalog #335 paradigm-shift discipline).

### Test coverage summary

```
src/tac/canonical_anti_patterns/tests/test_registry.py        44 tests
src/tac/cathedral_consumers/anti_pattern_lookup_consumer/tests/test_consumer.py    15 tests
TOTAL                                                          59 tests, ALL PASS
```

## Initial 12 anti-patterns landing summary

| # | anti_pattern_id | paradigm_class | severity | canonical source |
|---|---|---|---|---|
| 1 | lzma_on_already_brotli_saturated_compounding_v1 | compounding_order | medium | commit:d78401444 + commit:44d12e75d |
| 2 | quantize_then_svd_corrupted_low_rank_v1 | compounding_order | high | Wave N+1 compound stacking analysis |
| 3 | fp4_packed_without_qat_cos_collapse_v1 | quantization | critical | decoder compression Scenario A + CLAUDE.md QAT pipeline non-negotiable |
| 4 | brotli_plus_lzma_chained_anti_pattern_v1 | compounding_order | medium | Sister of #1 broader scope |
| 5 | cross_paradigm_test_without_per_axis_decomposition_v1 | diagnosis | high | commit:b01232473 V3 RE-RUN |
| 6 | predicted_band_from_random_init_tier_c_v1 | diagnosis | critical | Catalog #324 + C6 IBPS 22x miss |
| 7 | rank_1_problem_spec_synergy_tautology_v1 | diagnosis | high | paradox half 2 rigor review commit:21014faa7 |
| 8 | phantom_score_directory_naming_lie_v1 | provenance | critical | Z3 v2 + Catalog #249 |
| 9 | transient_tmp_path_in_persisted_artifact_v1 | provenance | high | lane_pr106_stacked + Catalog #220 |
| 10 | source_selector_inherited_predicted_score_mean_v1 | data_source | medium | DQS1 BUILD-1 verdict 2026-05-25 |
| 11 | silent_no_spawn_modal_dispatch_v1 | observability | high | STC v2 5x consecutive + Catalog #360 |
| 12 | docstring_overstatement_without_evidence_tag_v1 | rigor_loss | medium | CLAUDE.md FORBIDDEN_PATTERNS + Lane PD 49% vs 18.5% |

All 12 register cleanly via `populate_initial_anti_patterns()` to `.omx/state/canonical_anti_patterns_registry.jsonl` (34.4K JSONL; 12 events). Each carries Provenance per Catalog #323 with `ProvenanceKind.PREDICTED_FROM_MODEL` + `evidence_grade="predicted"` + `score_claim_valid=False` + `promotion_eligible=False` (anti-patterns are CLASS-level predictions of future recurrences; never promotable score claims).

## Canonical equation registration

NEW canonical equation registered per design memo + Catalog #344:

- **equation_id**: `canonical_anti_patterns_compounding_aggregation_v1`
- **mathematical_predicate**: `RegressionRisk_total = MAX over applicable anti-patterns × indicator(stack_matches_anti_pattern_class_i)` (MAX not SUM per design memo §Mathematical compounding identity — anti-patterns are typically dominated by the worst applicable pattern)
- **canonical_producers**: `tac.canonical_anti_patterns.pattern_matcher.match_stack_against_anti_patterns` + `validate_compound_stack_order`
- **canonical_consumers**: `src/tac/cathedral_consumers/anti_pattern_lookup_consumer/` + `tools/cathedral_autopilot_autonomous_loop.py` + `src/tac/dykstra_pareto_solver/` (Slot 1 Wave N+2 integration target)
- **next_recalibration_trigger**: `when_3+_new_empirical_anchors_in_domain`

Sister to Slot 1's Dykstra-related canonical equation (different equation_id; safe APPEND-ONLY append per Catalog #110/#113).

## Probe outcomes ledger row per Catalog #313

- **probe_id**: `canonical_anti_patterns_registry_layer_1_plus_2_landed_20260528`
- **verdict**: PROCEED (canonical apparatus extension; operator-directive ratified)
- **30-day staleness window** (auto-expires 2026-06-27)
- **reactivation criteria**: Wave N+2 deliverables landing (Layer 3 STRICT gate + Layer 4 sister auto-recalibrator wiring + Layer 5 Slot 1 Dykstra solver integration)

## 6-hook wire-in declaration per Catalog #125

- **hook #1 sensitivity-map**: ACTIVE — anti-pattern severity contributes to per-substrate sensitivity ranking (CRITICAL anti-patterns dominate ranker priority).
- **hook #2 Pareto constraint**: **PRIMARY** — anti-patterns EXCLUDE regions from Pareto polytope feasibility set; Wave N+2 Slot 1 Dykstra solver integration consumes as ACTIVE constraints (this is the structural reason the consumer's primary hook is #2).
- **hook #3 bit-allocator**: ACTIVE — anti-pattern unwind paths route bit-allocator AWAY from known compounding-order anti-patterns (e.g. quantize-then-SVD canonical unwind = SVD FIRST).
- **hook #4 cathedral autopilot dispatch**: ACTIVE — auto-discovered via Catalog #335; Wave N+2 invoker wire-in queued in Layer 5.
- **hook #5 continual-learning posterior**: ACTIVE — EmpiricalFalsifications append via `append_empirical_falsification`; canonical anti-pattern recalibrates per Catalog #371-sister `auto_recalibrate_from_continual_learning_posterior` (NON-STUB per Catalog #371 lesson; actually re-derives band from landed falsifications when count >= 3; idempotent).
- **hook #6 probe-disambiguator**: ACTIVE — matched anti-pattern + canonical unwind path IS the canonical disambiguator between speculative recurrence vs measured falsification (this is the secondary hook on the cathedral consumer).

## Cargo-cult audit per Catalog #303

| Surface | Decision | Rationale |
|---|---|---|
| Dataclass shape | ADOPT_CANONICAL (mirror CanonicalEquation) | Operator directive verbatim: "like the canonical equations bu netgative"; the symmetry is the design |
| JSONL persistence | ADOPT_CANONICAL (mirror CanonicalEquationsRegistry) | Catalog #131/#138/#245 sister discipline; fcntl-locked APPEND-ONLY |
| Cathedral consumer auto-discovery | ADOPT_CANONICAL (Catalog #335) | Paradigm-shift discipline already canonical; mirror canonical_equation_lookup_consumer pattern |
| Auto-recalibrator | ADOPT_CANONICAL (Catalog #371 lesson) | NON-STUB required; sister has the canonical pattern + tests demonstrating Catalog #371 regression |
| Anti-pattern semantics | FORK_BECAUSE_PRINCIPLED_MISMATCH (band IS falsification not prediction) | Semantically inverted: `falsification_band` IS the range where the anti-pattern MANIFESTS empirically (not where prediction lands) |
| Higher-layer abstraction | FORK_BECAUSE_PRINCIPLED_MISMATCH (CLASS not instance) | Catalog gates extinct per-instance bugs; this registry captures CLASS-level patterns compounding across substrates |
| MAX aggregation (vs SUM) | HARD-EARNED via design memo verdict | Anti-patterns dominated by WORST applicable; SUM would double-count + dilute critical signal |
| Tier A observability-only consumer | HARD-EARNED via Catalog #341/#357 + #127/#192 | Matched anti-pattern surfaces ROUTING RECOMMENDATION, NOT score signal; promotion gated by empirical paired-axis evidence per CLAUDE.md "Submission auth eval" non-negotiable |

## 9-dimension success checklist evidence per Catalog #294

1. **UNIQUENESS**: NEGATIVE registry surface is novel (positive sister exists; this is the symmetric half); CLASS-level abstraction (Catalog gates are per-instance).
2. **BEAUTY + ELEGANCE**: Mirrors canonical_equations API exactly; 30-second-reviewable per dataclass + per public-API helper; narrow public API export.
3. **DISTINCTNESS**: Different from Catalog gates (per-instance extinction), canonical equations (positive predictions), probe outcomes ledger (per-probe adjudication), and Slot 1 Dykstra solver (different cathedral consumer at the polytope-construction surface).
4. **RIGOR**: 59 dedicated tests pass (44 + 15); Provenance per Catalog #323; PREDICTED grade enforced at construction; Catalog #371 NON-STUB regression guard; 4-process concurrency stress.
5. **PER-METHOD OPTIMIZATION**: Mirroring canonical_equations is the substrate-optimal engineering (per UNIQUE-AND-COMPLETE-PER-METHOD falling-rule + operator directive's explicit symmetry framing).
6. **STACK-OF-STACKS-COMPOSABILITY**: Sister of canonical_equations + cathedral consumers + canonical Provenance + canonical probe outcomes + Dykstra polytope (Wave N+2 integration target). Composes orthogonally with existing surfaces.
7. **DETERMINISTIC REPRODUCIBILITY**: fcntl-locked atomic JSONL writes (no .tmp leakage); idempotent `populate_initial_anti_patterns`; canonical UTC timestamps; bit-stable JSON serialization with `sort_keys=True`.
8. **EXTREME OPTIMIZATION + PERFORMANCE**: O(N) registry query (N = anti-pattern count); O(N*M) matcher (M = recurrence_conditions per pattern; typically <=4); no GPU dispatch; observability-only consumer adds <1ms per candidate per Catalog #357 Tier A discipline.
9. **OPTIMAL MINIMAL CONTEST SCORE**: $0 implementation; structural foundation for Wave N+2 Layer 5 Slot 1 Dykstra integration which IS the per-axis Pareto polytope solver that compounds positive + negative constraints to STEER next-cycle attack direction.

## Observability surface per Catalog #305

1. **Inspectable per layer**: every AntiPattern is a 14-field structured row; every EmpiricalFalsification is a 10-field structured row; both serialize to canonical JSON via `to_dict()`.
2. **Decomposable per signal**: `recurrence_conditions` enumerated; `forbidden_pattern_predicate` is a structured predicate string; `falsification_band` is a per-axis dict.
3. **Diff-able across runs**: APPEND-ONLY JSONL ledger preserves all events; `auto_recalibrate` diffs stored band vs refit band; emits `EVENT_ANTI_PATTERN_RECALIBRATED` only when changed.
4. **Queryable post-hoc**: `query_anti_patterns` + `query_anti_patterns_by_substrate` + `query_falsifications_by_paradigm_class` + `query_recurrence_rate_by_severity` + `get_anti_pattern_by_id`.
5. **Cite-able**: every anti-pattern carries `canonical_source_anchor` (commit / Catalog # / sister memo path); every falsification carries `empirical_artifact_path` + canonical Provenance.
6. **Counterfactual-able**: matcher returns `match_confidence` in [0, 1]; operator can simulate "what if this anti-pattern were NOT registered" by passing a tmp registry path; consumer's `consume_candidate` is pure (no side effects).

## Mission contribution per Catalog #300

**`frontier_breaking_enabler`**: this is canonical apparatus extension compounding across substrate work + sessions. The IMMEDIATE score-lowering value is N/A (defensive validator + auto-discovered observability consumer); the STRUCTURAL value is unblocking Wave N+2 Layer 5 Slot 1 Dykstra integration which IS the per-axis Pareto polytope solver that compounds positive + negative constraints to STEER next-cycle attack direction.

Per the operator's just-saved META directive "AUTOMATED + COMPOUNDING + OPTIMAL" (2026-05-26):
- **AUTOMATED**: Catalog #335 auto-discovery (no manual ranker edits); auto-recalibration triggers structurally
- **COMPOUNDING**: every new EmpiricalFalsification compounds across sessions via canonical posterior anchors; every new AntiPattern compounds across substrates via cathedral consumer auto-ingestion
- **OPTIMAL**: MAX aggregation (canonical safety-constraint composition); Tier A observability-only (safe-by-construction); $0 implementation

## Wave N+2 deferred deliverables

Per design memo §"Wire-in surfaces":

- **Layer 3** STRICT preflight gate `check_compound_stack_proposal_acknowledges_known_anti_patterns` (claim canonical # via Catalog #186 next session)
- **Layer 4** sister auto-recalibrator wiring (cron / scheduled invocation) — the Layer 1 helper `auto_recalibrate_from_continual_learning_posterior` exists + tested; needs a wire-in caller similar to canonical_equation_lookup_consumer's update_from_anchor
- **Layer 5** Slot 1 Dykstra Pareto polytope solver integration (extend `invoke_dykstra_pareto_solver_on_candidates` to consult `match_stack_against_anti_patterns` + EXCLUDE matched regions from feasibility set; landed when Slot 1 `a0cfddc196d765c74` lands)

## Cross-references

- **Sister design memo**: `.omx/research/canonical_anti_patterns_registry_design_20260528.md` (commit 37b5a0184; 290 lines)
- **Sister POSITIVE registry**: `tac.canonical_equations` (Catalog #344)
- **Sister cathedral consumer**: `src/tac/cathedral_consumers/canonical_equation_lookup_consumer/` (mirror at positive surface)
- **Sister auto-recalibrator lesson**: Catalog #371 (NON-STUB required; this registry's auto-recalibrator inherits the lesson)
- **Sister concurrency discipline**: Catalog #131/#138/#245 (fcntl-locked + strict-load + canonical 4-layer ledger pattern)
- **Sister cathedral auto-discovery**: Catalog #335 (consumer count 73 → 74 verified)
- **Sister Tier A routing markers**: Catalog #341 (consumer's `consume_candidate` returns canonical 3-marker tuple)
- **Sister dual-tier architecture**: Catalog #357 (TIER_A_OBSERVABILITY_ONLY declared)
- **Sister canonical Provenance**: Catalog #323 (all Provenance per `build_provenance_for_predicted`)
- **Sister placeholder rejection**: Catalog #287 (rationale validators inherited from Provenance contract)
- **Sister probe outcomes ledger**: Catalog #313 (probe outcome registered 30-day window)
- **Sister landings (just before this)**: NSCS06 v8 `6e5437f48` / Slot 2 V3 int8 `71590fbef` / V3 RE-RUN `b01232473` / decoder compression `44d12e75d`
- **Sister in-flight (DISJOINT scope)**: Slot 1 Dykstra Pareto polytope solver `a0cfddc196d765c74` — Wave N+2 Layer 5 integration target
- **CLAUDE.md "Meta-Lagrangian/Pareto solver" non-negotiable** — anti-patterns are NEGATIVE constraints on the Pareto polytope
- **CLAUDE.md "Results must become system intelligence"** — extincts orphan FORBIDDEN_PATTERNS prose narrative
- **CLAUDE.md "Subagent coherence-by-default" 6-hook wire-in** — every hook declared above
- **CLAUDE.md "Forbidden premature KILL"** — matched anti-pattern is NOT a KILL verdict; surfaces canonical_unwind_path for operator routing

## Empirical receipts

Verified live:

```
$ .venv/bin/python -c "from tac.canonical_anti_patterns import populate_initial_anti_patterns, query_anti_patterns; populate_initial_anti_patterns(); print(len(query_anti_patterns()))"
12
```

```
$ wc -l .omx/state/canonical_anti_patterns_registry.jsonl
12 .omx/state/canonical_anti_patterns_registry.jsonl
```

```
$ .venv/bin/python -c "from tools.cathedral_autopilot_autonomous_loop import discover_and_register_consumers; r = discover_and_register_consumers(); print(f'count={len(r)}; anti_pattern_lookup_consumer in {\"anti_pattern_lookup_consumer\" in {x[\"consumer_name\"] for x in r}}')"
count=74; anti_pattern_lookup_consumer in True
```

```
$ PYTHONPATH=src .venv/bin/python -m pytest src/tac/canonical_anti_patterns/tests/ src/tac/cathedral_consumers/anti_pattern_lookup_consumer/tests/ --no-header -q
...........................................................              [100%]
59 passed in 0.50s
```

## Premise verification per Catalog #229

All 15 PV files read BEFORE editing:

1. `.omx/research/canonical_anti_patterns_registry_design_20260528.md` ✓
2. `src/tac/canonical_equations/__init__.py` ✓ + `equation.py` ✓ + `registry.py` ✓ + `builtins.py` (header + populate_initial_equations) ✓
3. `src/tac/canonical_equations/tests/` (mirrored test patterns) ✓
4. `src/tac/cathedral/consumer_contract.py` ✓ (Protocol + AxisDecomposition + ConsumerTier + HookNumber)
5. `src/tac/cathedral_consumers/` directory listing ✓ (73 sister consumers verified)
6. `src/tac/cathedral_consumers/_example_consumer/__init__.py` ✓
7. `src/tac/cathedral_consumers/canonical_equation_lookup_consumer/__init__.py` ✓ (mirrored pattern)
8. CLAUDE.md FORBIDDEN_PATTERNS section ✓ (canonical initial population source)
9. CLAUDE.md catalog rows for #344 / #371 / #335 / #357 / #341 ✓
10. `src/tac/probe_outcomes_ledger.py` `register_probe_outcome` API ✓
11. `tools/cathedral_autopilot_autonomous_loop.py` `discover_*` callsites ✓
12. `src/tac/provenance/contract.py` `Provenance` + `ProvenanceKind` + `ProvenanceEvidenceGrade` ✓
13. `src/tac/provenance/builders.py` `build_provenance_for_predicted` signature ✓
14. `tools/subagent_checkpoint.py` (Catalog #206 checkpoint discipline) ✓
15. `tools/subagent_commit_serializer.py` (Catalog #117/#157/#174) ✓ (commit will use canonical serializer)

## Bulk-edit count + reproducer per Catalog #229

Bulk-edit count: **6 NEW files** (5 Layer 1 + 1 Layer 2 init) + **2 NEW test files** (1 per layer) + **2 NEW test __init__ shells** + **1 NEW landing memo** + **3 ledger appends** (anti-patterns × 12 + canonical equation × 1 + probe outcome × 1).

Reproducer:
```bash
# Verify imports + initial population
PYTHONPATH=src .venv/bin/python -c "from tac.canonical_anti_patterns import build_all_initial_anti_patterns; print(len(build_all_initial_anti_patterns()))"  # → 12

# Run all 59 tests
PYTHONPATH=src .venv/bin/python -m pytest src/tac/canonical_anti_patterns/tests/ src/tac/cathedral_consumers/anti_pattern_lookup_consumer/tests/ -q

# Verify consumer auto-discovery
.venv/bin/python -c "from tools.cathedral_autopilot_autonomous_loop import discover_and_register_consumers; print(sum(1 for r in discover_and_register_consumers() if r['contract_compliant']))"  # → 74

# Verify pattern matcher live
.venv/bin/python -c "from tac.canonical_anti_patterns import match_stack_against_anti_patterns; print([m.anti_pattern.anti_pattern_id for m in match_stack_against_anti_patterns({'compression_ops': ['brotli_q11', 'lzma_q9']})])"
```

## NOT-INCLUDED scope (explicitly DEFERRED to Wave N+2)

- **Slot 1 Dykstra Pareto polytope solver integration**: Slot 1 lane in flight (`a0cfddc196d765c74`); Wave N+2 Layer 5 work to extend `invoke_dykstra_pareto_solver_on_candidates` to consult `match_stack_against_anti_patterns` per design memo §Layer 5.
- **Layer 3 STRICT preflight gate** `check_compound_stack_proposal_acknowledges_known_anti_patterns`: queued for next session via canonical Catalog # claim (Catalog #186 transactional discipline) + sister tests.
- **Layer 4 sister auto-recalibrator wire-in** (cron / scheduled invocation): the helper exists + tested; needs a calling site similar to `canonical_equation_lookup_consumer`'s posterior-anchor consumption.
- **CLAUDE.md catalog row for new STRICT gate**: only added when Layer 3 lands (Catalog #176 sister discipline).
- **Bare-writes-to-shared-state gate (Catalog #131) registration**: the new ledger path `.omx/state/canonical_anti_patterns_registry.jsonl` should be added to `_SHARED_STATE_PATH_MARKERS` in `src/tac/preflight.py` — DEFERRED to avoid Slot 1 scope collision (preflight.py is Slot 1's territory).
