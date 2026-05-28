# Canonical anti-pattern matcher false-positive architectural fix — LANDED 2026-05-28

```yaml
---
council_tier: T1
council_attendees: [Implementer, Assumption-Adversary]
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_predicted_mission_contribution: frontier_protecting
council_override_invoked: false
council_dissent: []
council_assumption_adversary_verdict:
  - assumption: "token-overlap fallback at confidence 0.50 is a safe default when no structural override matches"
    classification: CARGO-CULTED
    rationale: "empirically falsified 3x this session (Compound C / Wave N+3 Slot 1 / Compound F); the fallback fires WITHOUT honoring explicit-guarantee fields"
  - assumption: "anti-pattern's recurrence_conditions are sufficient self-description for matching"
    classification: CARGO-CULTED
    rationale: "the conditions describe the forbidden state but contain enough sister-term overlap that strict prose-substring matching produces token-overlap false positives"
  - assumption: "stack_spec is open-shape; the matcher cannot rely on structured fields"
    classification: HARD-EARNED
    rationale: "by design; explicit-override predicates degrade gracefully (False, '') when structured fields absent and fall back to legitimate token-heuristic positives"
  - assumption: "the symptom-only filter at pact_nerv_selector_v3:901 is sufficient defense-in-depth"
    classification: CARGO-CULTED
    rationale: "filter only protects ONE caller (PACT-NeRV selector); cathedral autopilot ratking + sister substrates + preflight #373 all consume the matcher and inherit the false-positive class"
council_decisions_recorded:
  - "land per-anti-pattern explicit_override_predicates table in pattern_matcher.py"
  - "ratify 4 PARADIGM-LEVEL EmpiricalFalsification rows via canonical append_empirical_falsification helper"
  - "preserve symptom-only filter at pact_nerv_selector_v3:901 as defense-in-depth (NOT redundant; covers anti-patterns without explicit override predicates yet)"
council_predicted_delta_s_band: null
council_override_rationale: ""
council_dispatch_target_substrate_id: null
deferred_substrate_id: null
deferred_substrate_retrospective_due_utc: null
---
```

## Summary

Architectural fix in `src/tac/canonical_anti_patterns/pattern_matcher.py` extincts a recurring false-positive bug class structurally. Pre-fix, the matcher's token-overlap fallback fired `confidence=0.50` matches EVEN WHEN the proposed stack_spec carried explicit override flags (`quantization_aware_training=True`, no `lzma` in `compression_ops`, `per_axis_decomposition_active=True`) — empirically confirmed 3× this session.

The fix adds a per-anti-pattern `_EXPLICIT_OVERRIDE_PREDICATES` table (5 entries; one per affected canonical anti-pattern) consulted BEFORE the token-overlap heuristic. When `stack_spec` structurally proves the forbidden predicate IS NOT MET, the matcher REFUSES the match regardless of token-overlap confidence.

## Empirical anchors (3× false-positive events this session)

| Anchor commit | Anti-pattern | Pre-fix verdict | Post-fix verdict |
|---|---|---|---|
| `e61ea93b0` | `fp4_packed_without_qat_cos_collapse_v1` | FIRED (false positive) | REFUSED (explicit `quantization_aware_training=True`) |
| `4c1daf186` | `brotli_plus_lzma_chained_anti_pattern_v1` | FIRED (false positive) | REFUSED (`compression_ops=["int8_per_channel","brotli_q11"]` has NO `lzma`) |
| `4c1daf186` | `lzma_on_already_brotli_saturated_compounding_v1` | FIRED (false positive; sister of above) | REFUSED (sister explicit-absence-of-lzma predicate) |
| `e5467cf05` | `cross_paradigm_test_without_per_axis_decomposition_v1` | FIRED (false positive) | REFUSED (explicit `per_axis_decomposition_active=True`) |

Verified empirically against the canonical Compound C stack_spec (all 4 explicit guarantees present): bug-class match count drops from 1-4 pre-fix to **0** post-fix.

## Cargo-cult audit per assumption

(per Catalog #303)

| Assumption (matcher engineering, NOT anti-pattern engineering) | Hard-earned vs cargo-culted | Unwind path |
|---|---|---|
| Token-overlap fallback at 0.50 is a safe default | CARGO-CULTED (3 empirical false positives) | Add per-anti-pattern explicit_override_predicate consulted BEFORE token-heuristic |
| recurrence_conditions are sufficient for matching | CARGO-CULTED (sister-term overlap produces FPs) | Honor structured stack_spec fields (compression_ops as list, quantization_aware_training as bool) |
| stack_spec is open-shape; rely only on string flattening | HARD-EARNED for legitimate positives; CARGO-CULTED for explicit guarantees | Degrade gracefully: override returns (False, "") when structured field absent → token-fallback fires |
| Symptom-only filter at pact_nerv_selector_v3:901 is sufficient | CARGO-CULTED (covers ONE caller; matcher has 5+ consumers) | Architectural fix at the matcher surface protects all consumers |
| `quantization_aware_training=True` is a bool flag, not a typed sentinel | HARD-EARNED (explicit primitive intent) | Recognized as canonical override via `_explicit_override_fp4_packed_without_qat` |
| All 4 anti-patterns share the same "absent lzma" predicate | CARGO-CULTED (b) and (c) overlap, but quantize_then_svd is different | Per-anti-pattern predicate functions; only (b)+(c) share via sister call |

## 9-dimension success checklist evidence

(per Catalog #294; concise per-dimension)

| Dim | Evidence |
|---|---|
| 1 UNIQUENESS | Architectural fix is per-anti-pattern, not a global heuristic; each predicate is canonical to that AP's structural guarantee field. |
| 2 BEAUTY+ELEGANCE | 5 small pure predicate functions + 1 dispatch dict + 1 public evaluator + 1 matcher call. Reviewable in 30 seconds. |
| 3 DISTINCTNESS | Sister of `assert_no_critical_anti_pattern_matches` (preserved as defense-in-depth) but at a different surface (matcher vs caller). |
| 4 RIGOR | 39 dedicated false-positive guard tests + 4 sister positive-path regression guards + Compound C end-to-end empirical verification + auto-recalibrator sanity check. |
| 5 OPTIMIZATION-PER-TECHNIQUE | Per-anti-pattern override table is the canonical per-AP-engineering surface; sister anti-patterns may share predicate functions (b)+(c) share the lzma-absence predicate. |
| 6 STACK-OF-STACKS-COMPOSABILITY | The override is decoupled from `AntiPattern` dataclass per JSONL backward-compat; future APs co-land sister entries. |
| 7 DETERMINISTIC-REPRODUCIBILITY | All predicates are pure functions of `stack_spec` Mapping → `(bool, str)`. Idempotent; deterministic. |
| 8 EXTREME-OPTIMIZATION+PERFORMANCE | Dict lookup + early-return BEFORE token-heuristic; faster, not slower. |
| 9 OPTIMAL-MINIMAL-CONTEST-SCORE | Indirect: prevents cathedral autopilot ranker from absorbing false-positive anti-pattern matches that would mis-route next-cycle attack direction. |

## Predicted ΔS band

(per Catalog #296)

**N/A for this landing**: this is an apparatus self-protection fix, not a substrate dispatch. No predicted score band; no Dykstra feasibility check needed. The empirical effect is observability-only (matcher refuses false positives; cathedral autopilot consumer no longer absorbs phantom anti-pattern matches into ranker cascade). Per Catalog #305 observability surface declaration below.

## Observability surface

(per Catalog #305 — 6 facets)

1. **Inspectable per layer**: each predicate function returns `(is_structurally_inapplicable, human_readable_reason)`; the public API `evaluate_explicit_override_for_anti_pattern` exposes the override decision.
2. **Decomposable per signal**: per-anti-pattern entries in `_EXPLICIT_OVERRIDE_PREDICATES` dict are independently inspectable; sister cathedral consumer can surface which AP was structurally REFUSED vs which token-matched.
3. **Diff-able across runs**: matcher output is deterministic; before/after diff = the 4 refused matches per Compound C anchor.
4. **Queryable post-hoc**: 4 ratified `EmpiricalFalsification` rows queryable via `tac.canonical_anti_patterns.registry.query_anti_patterns` + `.empirical_falsifications` attribute.
5. **Cite-able**: each EmpiricalFalsification row carries canonical Provenance per Catalog #323 + `empirical_artifact_path=commit:e61ea93b0`/`commit:4c1daf186`/`commit:e5467cf05` cite chain.
6. **Counterfactual-able**: the 4 positive-path regression guard tests prove the matcher still fires LEGITIMATE matches (e.g. fp4 without QAT, brotli + lzma both present).

## Canonical-vs-unique decision per layer

(per Catalog #290)

| Layer | Canonical vs unique | Rationale |
|---|---|---|
| Matcher API signature (`match_stack_against_anti_patterns`) | CANONICAL | Preserved verbatim; new override path is transparent to consumers |
| `_EXPLICIT_OVERRIDE_PREDICATES` dispatch dict | UNIQUE (this fix) | Per-anti-pattern explicit-guarantee detection is matcher-engineering, not anti-pattern-engineering |
| Per-predicate function bodies (`_explicit_override_fp4_packed_without_qat`, etc.) | UNIQUE per AP | Each AP's structural guarantee field is unique (`quantization_aware_training` vs `compression_ops` vs `per_axis_decomposition_active`) |
| Public evaluator `evaluate_explicit_override_for_anti_pattern` | CANONICAL pattern | Mirrors sister registry helpers' "expose introspection" idiom |
| Sister-share between (b) and (c) | CANONICAL | Both require lzma's presence; absence refutes both — explicit one-line delegation |
| Symptom-only filter at pact_nerv_selector_v3:901 | PRESERVED as defense-in-depth | NOT removed; covers anti-patterns without override predicates yet |
| Canonical `append_empirical_falsification` helper | CANONICAL | Used for all 4 ratification rows per existing API |
| Canonical Provenance per Catalog #323 | CANONICAL via `build_provenance_for_predicted` builder | PREDICTED grade + non-promotable + `[predicted]` axis |

## Observed effect on cathedral autopilot ranker (hook #4 evidence)

Pre-fix: the `anti_pattern_lookup_consumer` cathedral consumer would absorb up to 4 false-positive anti-pattern matches into the candidate annotation surface per candidate that declared explicit-guarantee fields. The ranker's MAX-aggregation severity cascade would surface CRITICAL severity (`fp4_packed_without_qat_cos_collapse_v1`) as the dominant constraint, mis-steering next-cycle attack direction toward "fix the QAT pipeline" when QAT was already structurally active.

Post-fix: the consumer absorbs only legitimate token-matched anti-patterns; false-positive class extincted at the matcher surface; sister-substrate selectors (pact_nerv_selector_v3) inherit the protection without per-caller waivers.

## 6-hook wire-in declaration

(per Catalog #125)

| Hook | Active? | Evidence |
|---|---|---|
| #1 sensitivity-map | N/A | Defensive validator gate; no signal contribution |
| #2 Pareto constraint | ACTIVE (downstream) | Slot 1 Dykstra Pareto polytope solver (Wave N+2 integration target) consumes matcher output as ACTIVE constraints; this fix reduces false-positive constraints in the polytope feasibility set |
| #3 bit-allocator | N/A | Not bit-allocation-related |
| #4 cathedral autopilot dispatch | ACTIVE PRIMARY | `anti_pattern_lookup_consumer` cathedral consumer's `consume_candidate` no longer absorbs false-positive matches; ranker cascade routes accurately |
| #5 continual-learning posterior | ACTIVE | 4 PARADIGM-LEVEL EmpiricalFalsification rows landed via canonical `append_empirical_falsification`; auto-recalibrator inspects rows on next session (current count 1 per AP; below 3-trigger floor by design per Catalog #371 lesson) |
| #6 probe-disambiguator | ACTIVE | The explicit-override predicates ARE the canonical disambiguator between "anti-pattern structurally inapplicable (REFUSE)" vs "anti-pattern token-matched but structural status unknown (FIRE at 0.5 confidence)" |

## Canonical-roster validation

(per Catalog #346)

T1 (working group; UNBOUNDED cadence per CLAUDE.md "Council hierarchy: 4-tier protocol"): single Implementer + single Assumption-Adversary; quorum met by definition. No grand-council attendees required for T1 implementation-engineering fix that does NOT touch a CLAUDE.md non-negotiable.

## Files touched

```
src/tac/canonical_anti_patterns/pattern_matcher.py    +338 lines  (architectural fix + 5 predicate functions)
src/tac/canonical_anti_patterns/__init__.py            +2 lines    (re-export evaluate_explicit_override_for_anti_pattern)
src/tac/canonical_anti_patterns/tests/test_pattern_matcher_override_predicates.py    +611 lines  (39 tests)
.omx/state/canonical_anti_patterns_registry.jsonl      +4 rows     (gitignored; durable local state)
.omx/research/canonical_anti_patterns_matcher_false_positive_architectural_fix_landed_20260528.md    (this memo)
```

## Test results

| Suite | Result |
|---|---|
| `tests/test_pattern_matcher_override_predicates.py` (new) | 39 passed |
| `tests/test_registry.py` (existing; regression guard) | 44 passed |
| `cathedral_consumers/anti_pattern_lookup_consumer/tests/test_consumer.py` (sister consumer) | 15 passed |
| `substrates/pact_nerv_selector_v3/tests/` (sister substrate) | 25 passed |
| **TOTAL** | **123 passed** |

Empirical Compound C bug-class anchor verification: `1-4` false positives pre-fix → `0` post-fix.

## Operator-routable next steps

1. **Catalog #373 sister consideration**: this architectural fix may make the `assert_no_critical_anti_pattern_matches` symptom-only filter at `pact_nerv_selector_v3:901` redundant for the 4 covered anti-patterns. Preserved as defense-in-depth (anti-patterns without override predicates yet still need the filter).
2. **Wave N+2 Layer 5 integration**: Slot 1's Dykstra Pareto polytope solver should consume the cleaned matcher output as ACTIVE constraints (no more false-positive constraints in the feasibility set).
3. **Sister anti-pattern landings**: future anti-patterns added to `builtins.py` SHOULD co-land an explicit-override predicate entry in `_EXPLICIT_OVERRIDE_PREDICATES` table; the matcher inherits the protection structurally.
4. **Auto-recalibrator trigger**: each affected AP now has 1 falsification (below 3-trigger floor). The 3+-trigger threshold is per-AP, not session-cumulative. If 2 additional session false-positive incidents land per AP (extremely unlikely now that the architectural fix extincts the class), the auto-recalibrator would refit the `falsification_band`. By design, this is NOT a stub per Catalog #371 lesson.

## Cross-references

- Predecessor crash-resume: `acaacac46a5b84db2` → `canonical_anti_patterns_matcher_fix_resume_20260528_1779982305` → THIS subagent `canonical_anti_patterns_matcher_fix_resume_20260528T154639Z_86725` (Catalog #206 mandatory crash-resume protocol)
- Original architectural finding: CLAUDE.md "Bugs must be permanently fixed AND self-protected against" non-negotiable
- Empirical anchors:
  - `commit:e61ea93b0` — Compound C STAND_DOWN
  - `commit:4c1daf186` — Wave N+3 Slot 1 PyTorch sister
  - `commit:e5467cf05` — Compound F preflight
- Sister symptom-only filter (preserved as defense-in-depth): `tac.substrates.pact_nerv_selector_v3.heterogeneous_bit_allocation.assert_no_critical_anti_pattern_matches`
- Catalog cross-references: #117/#157/#174 (canonical serializer) + #206 (crash-resume) + #292/#300/#346 (council discipline) + #287 (placeholder-rationale rejection) + #303 (cargo-cult audit) + #294 (9-dim checklist) + #296 (predicted-band Dykstra) + #305 (observability surface) + #307 (paradigm-vs-implementation) + #323 (canonical Provenance) + #335 (cathedral consumer canonical contract) + #344 (canonical equations registry sister) + #371 (auto-recalibrator NOT-stub canonical lesson)
- Canonical frontier pointer at landing (Catalog #343): `our_local_frontier_contest_cpu=0.1920089730` archive `18e3155fbbbe...`; `our_local_frontier_contest_cuda=0.205...` archive `9cb989cef519...` (this fix is apparatus self-protection; no frontier movement)
