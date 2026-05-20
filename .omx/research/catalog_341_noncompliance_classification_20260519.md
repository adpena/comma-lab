# Catalog #341 cathedral-consumer routing-marker non-compliance classification — Slot II 2026-05-19

**Date:** 2026-05-20T01:58:00Z (slot II Catalog #229 PV pass complete)
**Lane:** `lane_catalog_341_noncompliance_classification_20260519` L1 (impl_complete + memory_entry)
**Authority:** sister EE landing memo `.omx/research/master_gradient_xray_viz_tool_797_landed_20260519.md` op-routable #1 ("highest-EV"): *"plot 3 surfaced 2 of 34 cathedral consumers with non-canonical Catalog #341 routing markers (5.9% non-compliance vs canonical 100%). per_consumer_verdicts field in metrics dict names them; operator-routable sister investigation should classify each as (i) intentionally diagnostic with custom markers, (ii) Catalog #341 regression requiring fix, or (iii) candidate for # CATHEDRAL_CONSUMER_DEFERRED_OK:<rationale> waiver"*.
**Scope:** READ-ONLY classification work per task instructions; no consumer file mutations (sister HH may be reading `src/tac/cathedral_consumers/` via Catalog #335 auto-discovery; mutating would race per Catalog #340 sister-checkpoint guard).
**Sister coordination per Catalog #230 ownership map:** DD (council symposiums — disjoint `.omx/research/council_t3_*` + `.omx/state/council_deliberation_posterior.jsonl`); GG (B1 E.7 remediate — disjoint `.omx/operator_authorize_recipes/`); HH (consumers solver wire-in — disjoint `src/tac/sensitivity_map*.py` + `src/tac/optimization/pareto*.py` + `src/tac/bit_allocator*.py`). My scope = NEW classification memo + READ-ONLY on 2 consumer `__init__.py` files. Catalog #340 sister-checkpoint guard PROCEED.

---

## Executive summary

Plot 3 (`consumer_verdict_matrix`) empirically surfaced **2 of 34** cathedral consumers (5.9%) with non-canonical Catalog #341 routing markers. **Both are Classification (i) — INTENTIONALLY DIAGNOSTIC with custom markers**; the "non-compliance" is a SCOPING TENSION between two CANONICAL disciplines (Catalog #341 marker-uniformity vs CLAUDE.md "MPS auth eval is NOISE" + "Forbidden MPS-derived strategic decision" non-negotiables), NOT a regression and NOT a waiver candidate.

The 2 consumers:

| Consumer | File | Failing marker | Other markers status |
|---|---|---|---|
| `mps_diagnostic_consumer` | `src/tac/cathedral_consumers/mps_diagnostic_consumer/__init__.py:52` | `axis_tag="[MPS-PROXY]"` (Catalog #341 strict expects `"[predicted]"`) | `predicted_delta_adjustment=0.0` ✓ ; `promotable=False` ✓ ; `non_vacuous=True` ✓ ; `error=None` ✓ |
| `mps_gap_experiment_consumer` | `src/tac/cathedral_consumers/mps_gap_experiment_consumer/__init__.py:51` | `axis_tag="[MPS-PROXY]"` (Catalog #341 strict expects `"[predicted]"`) | `predicted_delta_adjustment=0.0` ✓ ; `promotable=False` ✓ ; `non_vacuous=True` ✓ ; `error=None` ✓ |

**Aggregate verdict**: 2/2 → Classification (i) intentionally diagnostic; 0/2 → Classification (ii) regression; 0/2 → Classification (iii) waiver candidate.

**Recommended remediation per consumer (operator-routable)**: BOTH consumers should receive same-file `# CATHEDRAL_CONSUMER_DEFERRED_OK:<rationale>` waiver — NOT to suppress the gate, but to STRUCTURALLY DOCUMENT the canonical tension. See "Per-consumer detail" below for exact waiver text.

**Catalog #341 compliance status**:
- **CURRENT**: 32/34 (94.1%) via xray tool's strict equality check `axis_tag == "[predicted]"`.
- **PROJECTED post-remediation**: 34/34 (100%) via waiver-respecting compliance interpretation (the 2 MPS consumers carry canonical `# CATHEDRAL_CONSUMER_DEFERRED_OK:<rationale>` waivers documenting the intentional `[MPS-PROXY]` axis_tag per CLAUDE.md "MPS auth eval is NOISE" non-negotiable).
- **ALTERNATIVE PROJECTED**: 34/34 (100%) via xray tool widening its strict check to accept `axis_tag ∈ {"[predicted]", "[MPS-PROXY]"}` per the canonical `CANONICAL_MEASUREMENT_AXES` frozenset which already includes BOTH (src/tac/provenance/contract.py + src/tac/continual_learning.py:107-116 NON_PROMOTABLE_TAGS).

The decision between waiver vs xray-tool-widening is COUNCIL-GRADE per CLAUDE.md "Design decisions — non-negotiable"; both paths are defensible.

---

## Per-consumer detail

### Consumer #1 of 2: `mps_diagnostic_consumer`

**Path**: `src/tac/cathedral_consumers/mps_diagnostic_consumer/__init__.py` (55 LOC)

**Specific marker failing Catalog #341**: `axis_tag="[MPS-PROXY]"` (line 52). Catalog #341 strict check at `tools/master_gradient_xray.py:974` is `verdict_row["axis_tag"] == "[predicted]"` — exact string match, no widening for non-promotable proxy tags.

**Other Catalog #341 markers (PASSING)**:
- `predicted_delta_adjustment=0.0` (line 45) ✓
- `promotable=False` (line 53) ✓
- `non_vacuous=True` (non-empty rationale at line 46-51) ✓
- `error=None` (clean consume_candidate execution) ✓

**Purpose**: Per the docstring (lines 1-16) the consumer is OBSERVABILITY-only — wraps `tac.mps_diagnostic` (LayerDriftRecord / measure_layerwise_drift / identify_drift_cliff_layer / emit_drift_table_markdown). Per CLAUDE.md "MPS auth eval is NOISE" non-negotiable: MPS results are NEVER score truth. The consumer surfaces MPS drift evidence as a diagnostic signal for architecture/runtime mismatch analysis; it CANNOT adjust a candidate's predicted_score_delta and explicitly returns `predicted_delta_adjustment=0.0` to enforce that.

**Classification: (i) INTENTIONALLY DIAGNOSTIC WITH CUSTOM MARKERS**

**Rationale**: The `[MPS-PROXY]` axis_tag is REQUIRED by CLAUDE.md "MPS auth eval is NOISE" Rule 3 verbatim *"Score numbers measured on MPS may NOT be reported as 'auth' or 'contest-compliant' anywhere — in commits, run_log, BATTLE_PLAN, or summaries. Tag them `[MPS-PROXY]` and treat as advisory only."* + the FORBIDDEN PATTERN "Forbidden MPS-derived strategic decision (the MPS-falsification trap)" verbatim *"Internal CUDA promotion/kill decisions REQUIRE a [contest-CUDA] artifact in the same record/section."* — the canonical apparatus REQUIRES the `[MPS-PROXY]` tag specifically so downstream consumers can distinguish MPS proxy from generic prediction. Mislabeling as `[predicted]` would VIOLATE the MPS-falsification trap forbidden pattern by hiding the MPS source axis behind a generic prediction tag.

The canonical Provenance contract at `src/tac/provenance/contract.py` includes `[MPS-PROXY]` in `CANONICAL_MEASUREMENT_AXES` per `src/tac/tests/test_provenance_contract.py:39` (`assert "[MPS-PROXY]" in CANONICAL_MEASUREMENT_AXES`). The non-promotable tag frozenset at `src/tac/continual_learning.py:107-116 NON_PROMOTABLE_TAGS` ALSO includes `[MPS-PROXY]`. The downstream MPS-VIABLE prescreen consumer at `src/tac/cathedral_consumers/mps_viable_prescreen_consumer/__init__.py` reads `axis_tag == "[MPS-PROXY]"` as routing input (per `src/tac/tests/test_mps_viable_prescreen_consumer.py:108-109`) — so DOWNSTREAM routing depends on the consumer preserving the `[MPS-PROXY]` axis_tag verbatim.

**Recommended remediation**:

```python
# Insert as line 2 (after SPDX header) in src/tac/cathedral_consumers/mps_diagnostic_consumer/__init__.py
# CATHEDRAL_CONSUMER_DEFERRED_OK:axis_tag_is_intentionally_[MPS-PROXY]_per_CLAUDE_md_MPS_auth_eval_is_noise_rule_3_and_forbidden_MPS_derived_strategic_decision_falsification_trap_required_for_downstream_mps_viable_prescreen_consumer_routing_disambiguator
```

**Estimated LOC**: 1 line addition. **Tests required**: 0 (waiver discovery is exercised by the canonical `discover_waiver_in_init` helper at `src/tac/cathedral/consumer_contract.py:182-209`). **Operator routing**: low-priority — the gate's "non-compliant" verdict is APPARATUS-INTERNAL OBSERVABILITY (plot 3 surfaces it); no downstream consumer is silently broken; no score regression risk.

---

### Consumer #2 of 2: `mps_gap_experiment_consumer`

**Path**: `src/tac/cathedral_consumers/mps_gap_experiment_consumer/__init__.py` (54 LOC)

**Specific marker failing Catalog #341**: `axis_tag="[MPS-PROXY]"` (line 51). Same xray-tool strict check as Consumer #1.

**Other Catalog #341 markers (PASSING)**:
- `predicted_delta_adjustment=0.0` (line 44) ✓
- `promotable=False` (line 52) ✓
- `non_vacuous=True` (non-empty rationale at line 45-50) ✓
- `error=None` (clean consume_candidate execution) ✓

**Purpose**: Per the docstring (lines 1-16) the consumer is OBSERVABILITY-only — wraps `tac.mps_gap_experiment` (TinyRenderer / build_tiny_renderer / train_on_mps_real_frames / GapManifest / classify_verdict / compute_gap_components). The active sister subagent `mps_phase_b_fire_and_harvest_20260519` owns the per-experiment verdict harvest path; this consumer surfaces canonical helper availability so the autopilot ranker can SEE the helpers exist without consuming MPS-derived score predictions.

**Classification: (i) INTENTIONALLY DIAGNOSTIC WITH CUSTOM MARKERS**

**Rationale**: Same as Consumer #1. The `[MPS-PROXY]` tag is canonically-mandated by CLAUDE.md "MPS auth eval is NOISE" Rule 3 + the "Forbidden MPS-derived strategic decision" FORBIDDEN PATTERN. The recent (2026-05-19) `mps_viable_prescreen_consumer` MPS-VIABLE prescreen sister consumer reads `axis_tag == "[MPS-PROXY]"` as the canonical signal for routing a candidate through the local-MPS-prescreen path before paid CUDA dispatch (per `feedback_mps_prescreen_cathedral_consumer_wire_in_landed_20260519.md` + Catalog #341's own scope-narrowing per the new sister Catalog #341 surface). Mislabeling as `[predicted]` would break the canonical routing disambiguator.

**Recommended remediation**:

```python
# Insert as line 2 (after SPDX header) in src/tac/cathedral_consumers/mps_gap_experiment_consumer/__init__.py
# CATHEDRAL_CONSUMER_DEFERRED_OK:axis_tag_is_intentionally_[MPS-PROXY]_per_CLAUDE_md_MPS_auth_eval_is_noise_rule_3_and_forbidden_MPS_derived_strategic_decision_falsification_trap_active_sister_mps_phase_b_fire_and_harvest_20260519_owns_per_experiment_verdict_harvest
```

**Estimated LOC**: 1 line addition. **Tests required**: 0 (same as Consumer #1). **Operator routing**: low-priority (same as Consumer #1).

---

## Aggregate verdict

| Classification | Count | Consumers |
|---|---|---|
| (i) INTENTIONALLY DIAGNOSTIC WITH CUSTOM MARKERS | **2** | `mps_diagnostic_consumer` + `mps_gap_experiment_consumer` |
| (ii) CATALOG #341 REGRESSION REQUIRING FIX | 0 | (none) |
| (iii) WAIVER CANDIDATE (operator routing required) | 0 | (none) |

**Compliance status path**:
- **Path A (recommended, low-effort)**: add `# CATHEDRAL_CONSUMER_DEFERRED_OK:<rationale>` waivers to both consumer `__init__.py` files. The canonical `discover_waiver_in_init` helper at `src/tac/cathedral/consumer_contract.py:182-209` honors the waiver per the canonical Catalog #335 contract. Catalog #341 strict-mode compliance per the xray tool's narrow check stays at 32/34 (94.1%) but the waiver makes the apparatus EMPIRICALLY DOCUMENTED — downstream operator audit + plot 3 future runs can SEE the waiver as canonical rationale rather than as silent drift.
- **Path B (council-grade, broader)**: widen the xray tool's Catalog #341 marker check at `tools/master_gradient_xray.py:974` from `axis_tag == "[predicted]"` to `axis_tag in CANONICAL_NON_PROMOTABLE_AXES = frozenset({"[predicted]", "[MPS-PROXY]", "[MPS-research-signal]", "[advisory only]", "[diagnostic-CPU]", "[diagnostic-CUDA]", "[macOS-CPU advisory]", "[macOS-CPU advisory only]"})`. This brings the xray tool's compliance check in line with the canonical `NON_PROMOTABLE_TAGS` frozenset at `src/tac/continual_learning.py:107-116` AND the canonical `CANONICAL_MEASUREMENT_AXES` at `src/tac/provenance/contract.py`. After widening, Catalog #341 compliance per the xray tool's check would be 34/34 (100%).

**Operator routing recommendation**: Path A first (no council; ~5 minutes wall-clock; pure documentation work), then Path B as a follow-on subagent if the operator wants the xray tool's interpretation to align with the canonical contract surfaces. The two paths are NOT mutually exclusive — Path A documents the canonical tension at the consumer level; Path B widens the tool to RECOGNIZE the canonical tension. Both together = belt-and-suspenders.

---

## 6-hook wire-in declaration per Catalog #125

1. **SENSITIVITY MAP** = N/A (classification memo; pure observability artifact; no sensitivity contribution).
2. **PARETO CONSTRAINT** = N/A (no Pareto-relevant signal).
3. **BIT-ALLOCATOR** = N/A (no bit-allocator signal).
4. **CATHEDRAL AUTOPILOT DISPATCH** = N/A directly; ACTIVE INDIRECTLY (sister Catalog #341 + #335 + #336/#337 surfaces consumer compliance to autopilot via plot 3 metrics; this memo documents the 2 canonical-tension consumers so future autopilot maintainers don't mis-classify the surface as drift).
5. **CONTINUAL-LEARNING POSTERIOR** = N/A (no posterior anchor; classification is not an empirical anchor — it is a discipline-level observation).
6. **PROBE-DISAMBIGUATOR** = **ACTIVE PRIMARY** — THIS CLASSIFICATION MEMO IS THE CANONICAL DISAMBIGUATOR between (a) compliant Catalog #341 consumer, (b) intentionally-diagnostic-with-custom-markers consumer, (c) Catalog #341 regression requiring fix, and (d) deferred-waiver consumer. Per CLAUDE.md "Subagent coherence-by-default" anti-arbitrariness primitive: when a design choice has 2+ defensible interpretations, ship BOTH modes via callable interface + build a probe-disambiguator that returns the regime-conditional verdict. The MPS consumer non-compliance is exactly this case — 2 canonical disciplines (Catalog #341 marker-uniformity vs CLAUDE.md MPS-tag-mandation) collide; the canonical Provenance contract resolves by widening; the canonical Catalog #341 strict check rejects. This memo is the probe-disambiguator that documents the regime-conditional verdict.

---

## Canonical-vs-unique decision per layer

| Layer | Decision |
|---|---|
| Classification memo location | ADOPT_CANONICAL — `.omx/research/<topic>_<utc>.md` per sister Catalog #287 / #290 / #294 / #305 design-memo pattern. |
| Per-consumer classification taxonomy | ADOPT_CANONICAL — task prompt specifies the canonical 3-category taxonomy (i / ii / iii); no fork needed. |
| Recommended remediation citation chain | ADOPT_CANONICAL — every observation cites canonical source via file:line per Catalog #287 sister discipline. |
| Waiver text format | ADOPT_CANONICAL — `# CATHEDRAL_CONSUMER_DEFERRED_OK:<rationale>` per Catalog #335 + `discover_waiver_in_init` helper contract. |
| Operator-routable recommendation Path A vs Path B framing | UNIQUE — the canonical apparatus does not have a precedent for "2-path remediation for canonical-tension consumer marker drift"; the framing is novel to this classification. |
| Aggregate compliance status projection | UNIQUE — the canonical apparatus does not currently track per-classification compliance projections; this memo's "CURRENT 94.1% → PROJECTED 100%" framing is the first operator-facing artifact in this format. |

## 9-dimension success checklist evidence

- **UNIQUENESS**: this is the FIRST per-consumer classification memo for the Catalog #341 non-compliance surface that sister EE's plot 3 tool empirically surfaced. The 2 MPS consumers are the canonical tension instance; no prior classification existed.
- **BEAUTY+ELEGANCE**: 2 consumers × 1-line waiver remediation each = minimum-LOC closure. Per-consumer detail section is structured table + rationale + recommended waiver text, each reviewable in 30 seconds per CLAUDE.md "Beauty, simplicity, and developer experience" non-negotiable.
- **DISTINCTNESS**: explicitly distinguished from (a) sister EE's plot-3 tool which SURFACES the non-compliance but does not CLASSIFY it, and (b) Catalog #341 + #335 STRICT preflight gates which ENFORCE the canonical contract but do not RECONCILE the canonical tension. This memo IS the missing reconciliation layer.
- **RIGOR**: every observation cites canonical source (file:line); empirically reproducible via `python tools/master_gradient_xray.py --plot consumer_verdict_matrix --archive-sha 6bae0201 --output reports/master_gradient_xray/6bae0201_catalog_341_classification/consumer_verdict.png` per `reports/master_gradient_xray/6bae0201_catalog_341_classification/consumer_verdict.png` artifact landed in same commit batch.
- **OPTIMIZATION-PER-TECHNIQUE**: O(N) consumer-iteration with N=34; classification is purely string-comparison + canonical contract lookup; no GPU; no paid dispatch; ~30-minute wall-clock for full memo.
- **STACK-OF-STACKS-COMPOSABILITY**: this classification memo composes with (a) sister EE's plot 3 tool (the visual surface), (b) sister EE's operator-routable #1 (the queued investigation that THIS memo completes), (c) Catalog #335 + #341 STRICT preflight gates (the contract-enforcement surfaces), (d) the canonical Provenance contract (`src/tac/provenance/contract.py` CANONICAL_MEASUREMENT_AXES), and (e) the canonical NON_PROMOTABLE_TAGS frozenset (`src/tac/continual_learning.py:107-116`). All 5 surfaces remain LIVE and consistent post-classification.
- **DETERMINISTIC-REPRODUCIBILITY**: the classification was produced by direct invocation of `xray.plot_consumer_verdict_matrix(candidate, output)` with `candidate = {"candidate_id": "catalog_341_classification_probe", "archive_sha256": "6bae0201fb082457" + "0" * 48}`. Re-running produces byte-identical per_consumer_verdicts output (modulo runtime timestamp).
- **EXTREME-OPTIMIZATION-PERFORMANCE**: classification memo is < 350 LOC (bolt-on size budget per CLAUDE.md HNeRV parity discipline L7); no runtime dependencies introduced; pure observability artifact.
- **OPTIMAL-MINIMAL-CONTEST-SCORE**: classification is OBSERVABILITY infrastructure — does NOT directly lower contest score, but ENABLES the operator to make an informed decision about whether to apply Path A (waivers) or Path B (xray tool widening) or both, both of which would close the apparatus-internal observability surface. Frontier-protecting per CLAUDE.md horizon-class.

## Observability surface

Per Catalog #305 6-facet observability:

1. **Inspectable per layer** — the classification process is layered: (a) plot 3 invocation returns `per_consumer_verdicts` field; (b) per-consumer Python file reads expose docstring + return-dict markers; (c) canonical contract docstring in `src/tac/cathedral/consumer_contract.py:153-155` enumerates the legal axis_tag values; (d) canonical Provenance contract at `src/tac/provenance/contract.py` lists `CANONICAL_MEASUREMENT_AXES`; (e) canonical non-promotable tag frozenset at `src/tac/continual_learning.py:107-116`. Every layer is operator-inspectable via grep + cat.
2. **Decomposable per signal** — the classification is decomposable per consumer: 2 consumers × 4 markers each (predicted_delta_adjustment / promotable / axis_tag / non_vacuous) = 8 marker signals; only 2 of 8 fail (both `axis_tag != "[predicted]"`).
3. **Diff-able across runs** — re-running plot 3 with same candidate input produces byte-identical per_consumer_verdicts modulo runtime timestamp; the classification artifacts (consumer files) are version-controlled so future-vs-current diff is `git diff`-able.
4. **Queryable post-hoc** — this memo itself is the canonical queryable artifact; the plot 3 PNG at `reports/master_gradient_xray/6bae0201_catalog_341_classification/consumer_verdict.png` is the operator-facing visual queryable artifact.
5. **Cite-able** — every observation in this memo cites canonical source via file:line per Catalog #287 sister discipline; the canonical Provenance contract at `src/tac/provenance/contract.py` is the canonical source for `CANONICAL_MEASUREMENT_AXES`.
6. **Counterfactual-able** — operator can simulate "what if we applied Path A?" by adding waivers to both consumer files in a test branch and re-running plot 3; the canonical `discover_waiver_in_init` helper at `src/tac/cathedral/consumer_contract.py:182-209` provides the counterfactual evaluation surface.

## Cargo-cult audit per assumption

Per Catalog #303 sister discipline + the HARD-EARNED-vs-CARGO-CULTED addendum (`feedback_assumptions_classification_hard_earned_vs_cargo_culted_critical_addendum_20260515.md`):

| Assumption | Classification | Rationale |
|---|---|---|
| The xray tool's narrow `axis_tag == "[predicted]"` strict check IS the canonical Catalog #341 contract | **CARGO-CULTED** | The canonical apparatus has TWO sister canonical surfaces — `CANONICAL_MEASUREMENT_AXES` (Provenance contract) + `NON_PROMOTABLE_TAGS` (continual_learning) — BOTH of which INCLUDE `[MPS-PROXY]`. The xray tool's narrow check inherits a "uniformity == correctness" reflex without testing whether broader CANONICAL_NON_PROMOTABLE_AXES uniformity is the actual canonical contract. Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" non-negotiable: every canonical helper / META layer field / engineering pattern adoption per substrate MUST be paired with explicit canonical-vs-unique decision documenting whether canonical adoption serves measurably. The xray tool's narrow check is path-of-least-resistance canonical adoption; widening to recognize `[MPS-PROXY]` is the optimal per-method engineering. |
| Uniform Catalog #341 enforcement across all consumers IS hard-earned discipline | **CARGO-CULTED for diagnostic consumers; HARD-EARNED for promotable-routing consumers** | Per the sister Catalog #341 scope-narrowing at the dispatch-direct surface (`tools/operator_authorize.py::_dispatch_local_mps` / `_dispatch_local_cpu`) the 3 markers are STRUCTURAL protection that local-research-signal cannot leak into score/promotion signals. That is HARD-EARNED for the dispatch surface. The cathedral-consumer surface (this gate) has a DIFFERENT semantics: consumers are OBSERVABILITY annotations on the autopilot ranker — the `[predicted]` axis_tag exists to distinguish "predicted-by-canonical-helper-availability" from "measured empirically". A diagnostic consumer that wraps MPS evidence is canonically distinct from a predicted-helper-availability consumer; mandating uniform `[predicted]` axis_tag across BOTH classes erases the canonical distinction. |
| The 2 MPS consumers should be MUTATED to use `[predicted]` to satisfy the xray tool's narrow check | **CARGO-CULTED** | This is path-of-least-resistance to satisfy a tool's narrow check while VIOLATING CLAUDE.md "MPS auth eval is NOISE" Rule 3 + the "Forbidden MPS-derived strategic decision" FORBIDDEN PATTERN. The canonical apparatus has 2 protective wrappers for the MPS surface (`NON_PROMOTABLE_TAGS` + `CANONICAL_MEASUREMENT_AXES`) precisely because the `[MPS-PROXY]` tag IS the canonical disambiguator. Mutating to `[predicted]` would silently mis-classify MPS evidence and break the downstream `mps_viable_prescreen_consumer` routing (which reads `axis_tag == "[MPS-PROXY]"` per `src/tac/tests/test_mps_viable_prescreen_consumer.py:108-109`). |
| Applying waivers per Catalog #335 + the canonical `discover_waiver_in_init` helper IS the canonical closure path | **HARD-EARNED** | Per Catalog #335 the canonical contract explicitly provides the `# CATHEDRAL_CONSUMER_DEFERRED_OK:<rationale>` waiver mechanism specifically for cases where a consumer legitimately defers/diverges from the canonical contract. The 2 MPS consumers fit this exactly — they defer the "uniform `[predicted]`" expectation because the canonical `[MPS-PROXY]` tag is REQUIRED by sister CLAUDE.md non-negotiables. |
| Widening the xray tool's strict check is ALSO a canonical closure path | **HARD-EARNED** | The xray tool's narrow check is operationalized at `tools/master_gradient_xray.py:974` as a single string-equality comparison. Widening to `axis_tag in CANONICAL_NON_PROMOTABLE_AXES` is a 1-line change that brings the tool into alignment with the canonical Provenance contract. Both Path A (waivers per consumer) and Path B (widen tool) are HARD-EARNED canonical closures; they are NOT mutually exclusive. |

---

## Cross-references

- Sister EE landing memo: `.omx/research/master_gradient_xray_viz_tool_797_landed_20260519.md` (commit `ad49cbd90`)
- Sister Cable D wire-in batch: `.omx/research/cable_d_wire_in_batch_landed_20260519.md` (commit `3af31f709`)
- Sister Catalog #341 scope-narrowing landing: `feedback_catalog_317_scope_narrowing_plus_consumer_routing_sister_gate_landed_20260519.md` (paired with Catalog #317 dispatch surface)
- Canonical contract: `src/tac/cathedral/consumer_contract.py` (Catalog #335)
- Canonical waiver helper: `src/tac/cathedral/consumer_contract.py::discover_waiver_in_init` (lines 182-209)
- Canonical Provenance axes: `src/tac/provenance/contract.py::CANONICAL_MEASUREMENT_AXES` (includes `[MPS-PROXY]`)
- Canonical non-promotable tags: `src/tac/continual_learning.py:107-116::NON_PROMOTABLE_TAGS` (includes `[MPS-PROXY]`)
- xray tool narrow check: `tools/master_gradient_xray.py:974` (`verdict_row["axis_tag"] == "[predicted]"`)
- Downstream MPS-VIABLE prescreen consumer (reads `[MPS-PROXY]`): `src/tac/cathedral_consumers/mps_viable_prescreen_consumer/__init__.py` + `src/tac/tests/test_mps_viable_prescreen_consumer.py:108-109`
- Visual artifact landed in same commit batch: `reports/master_gradient_xray/6bae0201_catalog_341_classification/consumer_verdict.png`
- Sister catalog gates:
  - Catalog #125 (subagent landing 6-hook wire-in)
  - Catalog #229 (premise verification before edit)
  - Catalog #230 (sister-subagent ownership map)
  - Catalog #287 (canonical Provenance / placeholder-rationale rejection)
  - Catalog #290 (canonical-vs-unique decision per layer)
  - Catalog #294 (9-dimension success checklist evidence)
  - Catalog #303 (cargo-cult audit per assumption)
  - Catalog #305 (observability surface declaration)
  - Catalog #317 (one-arg local-MPS vs Modal dispatch switch — sister at the dispatch surface)
  - Catalog #335 (cathedral consumer directory contract — primary META gate)
  - Catalog #336/#337 (cathedral autopilot main invocation)
  - Catalog #341 (cathedral consumer routing markers — primary META gate this memo classifies non-compliance against)

---

## Operator-routable next steps

1. **PRIMARY (Path A)**: apply same-file `# CATHEDRAL_CONSUMER_DEFERRED_OK:<rationale>` waivers to both `mps_diagnostic_consumer/__init__.py` (line 2) and `mps_gap_experiment_consumer/__init__.py` (line 2). Exact waiver text provided in "Per-consumer detail" sections above. Estimated wall-clock: 5 minutes. Estimated risk: zero (pure documentation; canonical `discover_waiver_in_init` helper handles).
2. **SECONDARY (Path B)**: widen `tools/master_gradient_xray.py:974` strict check from `axis_tag == "[predicted]"` to `axis_tag in CANONICAL_NON_PROMOTABLE_AXES` (frozenset to be added at the top of the file). Estimated wall-clock: 15 minutes including test update. Estimated risk: low (compliance check semantics widen; no consumer behavior change).
3. **TERTIARY (council-grade)**: Catalog #341 contract clarification at the canonical contract level — should the per-consumer routing markers contract uniformly require `[predicted]` OR widen to any `CANONICAL_NON_PROMOTABLE_AXES` member? This is a council-grade decision per CLAUDE.md "Design decisions — non-negotiable" + the per-substrate-symposium discipline at Catalog #325. Defer to operator OR convene T2 sextet pact per CLAUDE.md "Council hierarchy: 4-tier protocol".

This classification memo CLOSES sister EE op-routable #1 ("classify each as ..."). The 3 next steps above are operator-routable AS A FOLLOW-ON; this memo's scope is READ-ONLY classification per task instructions.

---

## Lane status

- Lane `lane_catalog_341_noncompliance_classification_20260519` to be registered via `tools/lane_maturity.py add-lane` (post-commit)
- Gates landed in this commit batch:
  - `impl_complete` ✓ (classification memo + 2 consumers identified + recommended remediation paths)
  - `memory_entry` ✓ (this memo IS the memory entry per Catalog #229)
- Gates not applicable:
  - `real_archive_empirical` — N/A (classification artifact; no archive bytes generated)
  - `contest_cuda` — N/A (no score claims)
  - `strict_preflight` — N/A (no new STRICT preflight gate needed; the canonical Catalog #341 gate already protects the surface; this memo classifies the 2 non-compliance instances surfaced by sister EE's plot 3)
  - `three_clean_review` — adversarial review cycle deferred to operator routing (Path A vs Path B)
  - `deploy_runbook` — N/A (no runbook for classification artifact)

Expected lane level after this commit: **L1** (impl_complete + memory_entry).


<!-- WAVE-3-CATALOG-344-BACKFILL-SWEEP appended 2026-05-20 per operator NON-NEGOTIABLE "keep feeding the queue" + WIRE-IN-AUDIT-POST-CASCADE op-routable #3. -->
<!-- # FORMALIZATION_PENDING:catalog-341-cathedral-consumer-routing-marker-classification-audit-trigger-tokens-describe-gate-not-new-equation -->
