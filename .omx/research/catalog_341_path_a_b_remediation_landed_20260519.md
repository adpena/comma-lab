# Catalog #341 Path A + Path B remediation landed — Slot JJ 2026-05-19

**Date:** 2026-05-20T02:10:00Z (Slot JJ remediation complete; Catalog #229 PV pass + Path A waivers + Path B widening + 12 new tests all green)
**Lane:** `lane_catalog_341_path_a_b_remediation_20260519` L1 (impl_complete + memory_entry)
**Authority:** Sister II landing memo `.omx/research/catalog_341_noncompliance_classification_20260519.md` commit `7b8c56ae1` operator-routable #1 (Path A: 5min) + #2 (Path B: 15min).
**Sister coordination per Catalog #230 ownership map:** DD (`adab84c8aba6dbc5f`, B6 council symposiums under `.omx/research/council_t3_*` + `.omx/state/council_deliberation_posterior.jsonl`) disjoint; GG (`a866a992b00edc7f2`, B1 E.7 remediate under `.omx/operator_authorize_recipes/`) disjoint; HH (`a07ee4e638842066c`, consumers solver wire-in under `src/tac/sensitivity_map*.py` + sister) auto-discovers `cathedral_consumers/` but does NOT mutate (HH adds comment-line waivers post-SPDX preserving Catalog #335 contract). Catalog #340 sister-checkpoint guard PROCEED across all 4 active sisters.

---

## Executive summary

Sister II classification recommended Path A (waivers) FIRST and Path B (widen xray strict check) OPTIONALLY. Slot JJ executed BOTH (non-mutually-exclusive — Path A documents the canonical tension at consumer level; Path B widens the tool to RECOGNIZE the canonical tension). **Empirically verified: 32/34 (94.1%) → 34/34 (100%) Catalog #341 compliance post-Path-B widening.**

**12 new tests in `src/tac/tests/test_xray_strict_check_widened.py` all pass** (4 canonical-set discovery + 4 widened-check semantics + 3 Path A waiver regression + 1 live-repo regression guard at 100%). **54 existing xray tests still pass** (sister regression guards intact). **21 sister `test_mps_viable_prescreen_consumer.py` tests still pass** (the `[MPS-PROXY]` axis_tag is preserved verbatim per Slot II rationale — downstream routing depends on it).

---

## Path A diffs (waivers per consumer)

### Path A diff #1 of 2: `src/tac/cathedral_consumers/mps_diagnostic_consumer/__init__.py:2`

```diff
 # SPDX-License-Identifier: MIT
+# CATHEDRAL_CONSUMER_DEFERRED_OK:axis_tag_is_intentionally_[MPS-PROXY]_per_CLAUDE_md_MPS_auth_eval_is_noise_rule_3_and_forbidden_MPS_derived_strategic_decision_falsification_trap_required_for_downstream_mps_viable_prescreen_consumer_routing_disambiguator_per_slot_II_classification_memo_catalog_341_noncompliance_classification_20260519
 """Cathedral consumer for ``tac.mps_diagnostic`` layerwise MPS drift evidence.
```

Verification: `discover_waiver_in_init(...)` returns non-empty rationale + `valid=True`. Module imports cleanly. `CONSUMER_NAME=mps_diagnostic_consumer` + `CONSUMER_VERSION=0.1.0` + `CONSUMER_HOOK_NUMBERS=(HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH,)` all intact. `consume_candidate({})` returns canonical dict with `axis_tag=[MPS-PROXY]` + `predicted_delta_adjustment=0.0` + `promotable=False`.

### Path A diff #2 of 2: `src/tac/cathedral_consumers/mps_gap_experiment_consumer/__init__.py:2`

```diff
 # SPDX-License-Identifier: MIT
+# CATHEDRAL_CONSUMER_DEFERRED_OK:axis_tag_is_intentionally_[MPS-PROXY]_per_CLAUDE_md_MPS_auth_eval_is_noise_rule_3_and_forbidden_MPS_derived_strategic_decision_falsification_trap_active_sister_mps_phase_b_fire_and_harvest_20260519_owns_per_experiment_verdict_harvest_per_slot_II_classification_memo_catalog_341_noncompliance_classification_20260519
 """Cathedral consumer for ``tac.mps_gap_experiment`` MPS gap quantification.
```

Verification: identical to diff #1; module imports cleanly; canonical Catalog #335 contract preserved.

---

## Path B diff at xray tool (file:line)

### Path B diff #1 of 2: `tools/master_gradient_xray.py:97-120` (canonical set addition)

```diff
 from tac.provenance.builders import (  # noqa: E402
     build_provenance_for_predicted,
 )
+from tac.continual_learning import NON_PROMOTABLE_TAGS as _CANONICAL_NON_PROMOTABLE_TAGS  # noqa: E402
+
+
+# Catalog #341 widening per Slot JJ Path B (2026-05-19): xray tool's
+# verdict-matrix Catalog #341 strict check at consumer_verdict_matrix
+# accepts ANY axis_tag in the canonical non-promotable set, NOT only the
+# narrow "[predicted]" string. [...canonical citation block...]
+CANONICAL_NON_PROMOTABLE_AXES: frozenset[str] = frozenset(
+    {"[predicted]"} | set(_CANONICAL_NON_PROMOTABLE_TAGS)
+)
```

### Path B diff #2 of 2: `tools/master_gradient_xray.py:993-1004` (strict-check widening)

```diff
                 # Catalog #341 marker compliance check (3 canonical markers).
+                # Slot JJ Path B (2026-05-19): widened from narrow
+                # ``axis_tag == "[predicted]"`` to ``axis_tag in
+                # CANONICAL_NON_PROMOTABLE_AXES`` per Slot II classification
+                # memo + canonical Provenance/NON_PROMOTABLE_TAGS surfaces.
                 markers_compliant = (
                     verdict_row["predicted_delta_adjustment"] == 0.0
                     and verdict_row["promotable"] is False
-                    and verdict_row["axis_tag"] == "[predicted]"
+                    and verdict_row["axis_tag"] in CANONICAL_NON_PROMOTABLE_AXES
                 )
```

---

## Canonical CANONICAL_NON_PROMOTABLE_AXES set

**Location**: `tools/master_gradient_xray.py:118-120` (module-level frozenset). **Source**: union of `{"[predicted]"}` + canonical upstream `tac.continual_learning.NON_PROMOTABLE_TAGS` (the single source of truth per Catalog #287 discipline).

**Live contents (9 axes)**:
- `[predicted]` — narrow pre-Path-B default; preserved for backwards-compat
- `[MPS-PROXY]` — canonical per CLAUDE.md "MPS auth eval is NOISE" Rule 3 (THE Path B headline)
- `[MPS-research-signal]` — canonical per CLAUDE.md sister rule (MPS-as-research-signal-generator)
- `[advisory only]` — canonical per Catalog #127 + #192
- `[macOS-CPU advisory only]` — canonical per Catalog #192
- `[macOS-CPU calibrated]` — canonical per sister Catalog #192 calibration extension
- `[distortion-proxy:local]` — canonical per local-distortion-proxy non-promotability discipline
- `[byte-anchor]` — canonical per byte-anchor-vs-score-anchor discipline (per Catalog #110/#113)
- `[predicted; unified-action; closed-form weighted-sum]` — canonical per unified-Lagrangian-action predicted form

The set discovery + addition uses the **CANONICAL** path (import from `tac.continual_learning`) NOT re-derivation, satisfying Catalog #287 single-source-of-truth + Catalog #290 "ADOPT_CANONICAL when serves" decision. Adding `[predicted]` to the union is the per-this-tool unique addition (because `tac.continual_learning.NON_PROMOTABLE_TAGS` does NOT include `[predicted]` — that's the xray tool's narrow pre-Path-B default that is canonically non-promotable in this CONSUMER-marker context).

---

## Test coverage summary

| Test surface | File | Test count | Status |
|---|---|---|---|
| Existing xray Slot EE tests | `src/tac/tests/test_tools_master_gradient_xray.py` | 21 | PASS |
| Existing xray Slot legacy tests | `src/tac/tests/test_master_gradient_xray.py` | 33 | PASS |
| **NEW Slot JJ Path B tests** | `src/tac/tests/test_xray_strict_check_widened.py` | **12** | **PASS** |
| Sister MPS-VIABLE prescreen tests | `src/tac/tests/test_mps_viable_prescreen_consumer.py` | 21 | PASS |
| **TOTAL** | (4 files) | **87** | **87/87 PASS** |

**12 new Slot JJ tests breakdown:**
- 4 canonical-set discovery: `CANONICAL_NON_PROMOTABLE_AXES` exists as module attribute + is frozenset + includes `[predicted]` default + includes `[MPS-PROXY]` + unions canonical `NON_PROMOTABLE_TAGS` source.
- 4 widened-check semantics: accepts `[MPS-PROXY]` consumer + still rejects promotable `[contest-CPU]` (canonical safety invariant) + still accepts `[predicted]` (backwards-compat regression) + rejects unknown axis (canonical set is bounded).
- 3 Path A waiver regression: `mps_diagnostic_consumer` imports cleanly + `mps_gap_experiment_consumer` imports cleanly + both carry canonical waivers per `discover_waiver_in_init`.
- 1 live-repo regression: live cathedral consumer surface (~34) shows 100% Catalog #341 compliance post-Path-B widening.

---

## Post-remediation Catalog #341 compliance

**Empirically verified via `_collect_consumer_verdicts(candidate)` with canonical frontier archive sha `6bae0201fb082457`:**

| Metric | Pre-remediation (32/34) | Post-Slot-JJ (34/34) | Delta |
|---|---|---|---|
| Total contract-compliant consumers | 34 | 34 | 0 |
| Catalog #341 markers compliant | 32 | 34 | +2 |
| Compliance fraction | 94.1% | **100.0%** | +5.9% |
| Non-compliant consumers | 2 (`mps_diagnostic_consumer` + `mps_gap_experiment_consumer`) | 0 | -2 |

**Belt-and-suspenders structure**: Path A documents the canonical tension at consumer level (each MPS consumer's `__init__.py` carries explicit waiver with rationale); Path B widens the xray tool to recognize the canonical tension at tool level (the strict check now uses canonical `CANONICAL_NON_PROMOTABLE_AXES` instead of narrow `[predicted]` string equality). Either path alone would close the apparatus-internal observability surface; together they close it bidirectionally so a future regression at either surface is structurally caught by the OTHER.

---

## 6-hook wire-in declaration per Catalog #125

1. **SENSITIVITY MAP** = N/A — Path A waivers + Path B widening are pure documentation/observability infrastructure; no sensitivity-map contribution.
2. **PARETO CONSTRAINT** = N/A — no Pareto-relevant signal.
3. **BIT-ALLOCATOR** = N/A — no bit-allocator signal.
4. **CATHEDRAL AUTOPILOT DISPATCH** = N/A directly; ACTIVE INDIRECTLY — sister EE plot 3 now correctly reports 34/34 = 100% Catalog #341 compliance, removing the apparatus-internal observability false-positive that 2/34 consumers were "non-compliant" when they were actually intentionally-diagnostic-with-canonical-marker.
5. **CONTINUAL-LEARNING POSTERIOR** = N/A — no posterior anchor written (this is structural-protection landing, not empirical-anchor landing).
6. **PROBE-DISAMBIGUATOR** = **ACTIVE PRIMARY** — Path B widening IS the canonical disambiguator between intentionally-diagnostic-with-canonical-non-promotable-axis-tag consumers and Catalog #341 routing-marker regression. The widened `axis_tag in CANONICAL_NON_PROMOTABLE_AXES` check correctly distinguishes (a) canonically-non-promotable diagnostic consumers (PASS) from (b) PROMOTABLE-axis regression like `[contest-CPU]` / `[contest-CUDA]` (FAIL — the canonical safety invariant per CLAUDE.md "Submission auth eval" non-negotiable).

---

## Canonical-vs-unique decision per layer

| Layer | Decision | Rationale |
|---|---|---|
| xray tool strict check | **ADOPT_CANONICAL** | Widened to use canonical `tac.continual_learning.NON_PROMOTABLE_TAGS` upstream source per Catalog #287 single-source-of-truth. The narrow `[predicted]` pre-Path-B check was path-of-least-resistance canonical adoption (per Slot II cargo-cult audit: CARGO-CULTED — uniform `[predicted]` enforcement was a uniformity-correctness reflex that didn't test whether broader CANONICAL_NON_PROMOTABLE_AXES is the actual canonical contract). |
| Per-consumer waiver text | **UNIQUE-PER-CONSUMER** | Each waiver cites the consumer-specific rationale (mps_diagnostic_consumer cites the downstream mps_viable_prescreen_consumer routing disambiguator; mps_gap_experiment_consumer cites the active sister mps_phase_b_fire_and_harvest_20260519 verdict harvest). Forking the rationale per consumer satisfies CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" — the canonical Catalog #335 waiver SHAPE is shared, but the rationale is per-consumer-optimal. |
| `CANONICAL_NON_PROMOTABLE_AXES` location | **ADOPT_CANONICAL with per-tool union** | The set lives in `tools/master_gradient_xray.py` (per-tool) but UNIONS from the canonical source `tac.continual_learning.NON_PROMOTABLE_TAGS`. The unique addition of `{"[predicted]"}` is per-this-tool because `[predicted]` is canonically non-promotable in the consumer-marker context but is NOT in the canonical NON_PROMOTABLE_TAGS frozenset (which is about empirical evidence tags). |
| Test file location | **ADOPT_CANONICAL** | Sister test file pattern per `test_tools_master_gradient_xray.py` (Slot EE) + `test_master_gradient_xray.py` (legacy). New file `test_xray_strict_check_widened.py` follows the canonical naming pattern. |
| Test scope per Slot II recommendation (~5 tests) | **UNIQUE — landed 12 tests** | Slot II recommended ~5 tests; Slot JJ landed 12. Rationale: 4 canonical-set discovery tests pin the new module attribute; 4 widened-check semantics tests cover positive (accept `[MPS-PROXY]`) + negative (reject promotable + reject unknown axis) + regression (preserve `[predicted]`); 3 Path A waiver regression tests cover both consumers + canonical helper discovery; 1 live-repo regression guard pins 100% compliance. The expanded test count reflects the canonical CATEGORY space (3 dimensions: set discovery + check semantics + waiver regression + live-repo) rather than the minimum-LOC target. |

---

## Cargo-cult audit per assumption (per Catalog #303 + HARD-EARNED-vs-CARGO-CULTED addendum)

| Assumption | Classification | Rationale |
|---|---|---|
| Slot II's recommendation that BOTH Path A and Path B should land (non-mutually-exclusive belt-and-suspenders) | **HARD-EARNED** | Per Slot II memo's empirical analysis: Path A documents the canonical tension at consumer level; Path B widens the tool to recognize the canonical tension. Either alone closes the surface; together they close it bidirectionally so future regression at either surface is caught by the OTHER. The "uniformity correlates correctness" reflex (uniform `[predicted]` enforcement) was the bug class; the belt-and-suspenders fix prevents future regression at either surface. |
| The xray tool's narrow `axis_tag == "[predicted]"` strict check WAS the canonical Catalog #341 contract pre-Path-B | **CARGO-CULTED** (Slot II's verdict applied) | The canonical apparatus has TWO sister sources of truth (`CANONICAL_MEASUREMENT_AXES` + `NON_PROMOTABLE_TAGS`) BOTH of which include `[MPS-PROXY]`. The xray tool's narrow check inherited a "uniformity == correctness" reflex without testing whether broader canonical-axis uniformity was the actual contract. Path B widening is the structural correction. |
| Diagnostic consumers (MPS-tagged) and routing-recommendation consumers should share the same canonical marker discipline | **CARGO-CULTED for diagnostic; HARD-EARNED for routing-recommendation** | Per the sister Catalog #341 scope-narrowing at the dispatch-direct surface (`tools/operator_authorize.py::_dispatch_local_mps` / `_dispatch_local_cpu`) the 3 markers are STRUCTURAL protection that local-research-signal cannot leak into score/promotion signals. That is HARD-EARNED for the dispatch surface. The cathedral-consumer surface has DIFFERENT semantics: consumers are OBSERVABILITY annotations on the autopilot ranker — a diagnostic consumer that wraps MPS evidence is canonically distinct from a predicted-helper-availability consumer; mandating uniform `[predicted]` axis_tag across BOTH classes erases the canonical distinction. |
| Adding `{"[predicted]"}` to the canonical NON_PROMOTABLE_TAGS union creates a per-tool divergence from the canonical source | **HARD-EARNED** | `[predicted]` is canonically non-promotable in the CONSUMER-MARKER context (per Catalog #341 default + sister consumers like `mps_viable_prescreen_consumer`) but `tac.continual_learning.NON_PROMOTABLE_TAGS` is scoped to EMPIRICAL EVIDENCE TAGS (per its source-of-truth contract — see test_continual_learning.py:143-145 sanity test). Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode": the per-tool union is the optimal engineering for the xray tool's verdict-matrix context. |

---

## Observability surface (per Catalog #305 6-facet)

1. **Inspectable per layer** — Path A waivers visible at consumer `__init__.py:2`; Path B widening visible at `tools/master_gradient_xray.py:97-120` (canonical set) + `tools/master_gradient_xray.py:993-1004` (widened check). Each layer operator-inspectable via grep + cat.
2. **Decomposable per signal** — 9 axes in `CANONICAL_NON_PROMOTABLE_AXES` × 34 consumers = 306 (axis, consumer) decision cells per consumer matrix invocation; each consumer's `axis_tag in CANONICAL_NON_PROMOTABLE_AXES` decision is independently inspectable.
3. **Diff-able across runs** — re-running `_collect_consumer_verdicts(...)` with same candidate input produces byte-identical `per_consumer_verdicts` modulo runtime timestamp. The 12 new tests provide cross-run regression guards.
4. **Queryable post-hoc** — this memo + sister II classification memo + the 12 new tests + the canonical CANONICAL_NON_PROMOTABLE_AXES module attribute are all queryable post-hoc via grep / pytest / `_collect_consumer_verdicts(...)`.
5. **Cite-able** — every observation in this memo cites canonical source via file:line. Cite-chain: Slot II memo → Slot JJ memo → 12 tests → canonical NON_PROMOTABLE_TAGS source → CLAUDE.md "MPS auth eval is NOISE" non-negotiable.
6. **Counterfactual-able** — operator can simulate "what if we removed Path B widening?" by reverting the strict-check diff in a test branch and re-running the 12 new tests; tests `test_widened_check_accepts_mps_proxy_consumer` + `test_live_repo_catalog_341_compliance_at_100_percent` will fail, structurally proving the widening's necessity.

---

## Lane status

- Lane `lane_catalog_341_path_a_b_remediation_20260519` to be registered via `tools/lane_maturity.py add-lane` (post-commit).
- Gates landed in this commit batch:
  - `impl_complete` ✓ (Path A waivers + Path B widening + canonical CANONICAL_NON_PROMOTABLE_AXES + 12 new tests)
  - `strict_preflight` ✓ (Path B widening uses canonical upstream `NON_PROMOTABLE_TAGS` per Catalog #287 single-source-of-truth — no NEW strict gate landed, but the widened check semantically extends the existing Catalog #341 gate)
  - `memory_entry` ✓ (this memo IS the memory entry per Catalog #229)
  - `three_clean_review` — N/A (defensive remediation; sister II classification was the council-grade adjudication; Slot JJ executes per operator routing)
- Gates not applicable:
  - `real_archive_empirical` — N/A (observability infrastructure; no archive bytes generated)
  - `contest_cuda` — N/A (no score claims)
  - `deploy_runbook` — N/A (no runbook for observability remediation)

Expected lane level after this commit: **L1** (impl_complete + strict_preflight + memory_entry).

---

## Cross-references

- Sister II classification memo: `.omx/research/catalog_341_noncompliance_classification_20260519.md` (commit `7b8c56ae1`)
- Sister EE landing memo: `.omx/research/master_gradient_xray_viz_tool_797_landed_20260519.md` (commit `ad49cbd90`)
- Sister Cable D wire-in batch: `.omx/research/cable_d_wire_in_batch_landed_20260519.md` (commit `3af31f709`)
- Sister Catalog #341 scope-narrowing landing: `feedback_catalog_317_scope_narrowing_plus_consumer_routing_sister_gate_landed_20260519.md`
- Canonical contract: `src/tac/cathedral/consumer_contract.py` (Catalog #335)
- Canonical waiver helper: `src/tac/cathedral/consumer_contract.py::discover_waiver_in_init`
- Canonical Provenance axes: `src/tac/provenance/contract.py::CANONICAL_MEASUREMENT_AXES`
- Canonical non-promotable tags (upstream source): `src/tac/continual_learning.py:107-116::NON_PROMOTABLE_TAGS`
- xray tool widened check: `tools/master_gradient_xray.py:993-1004` (was line 974 pre-widening; +6 lines for canonical set + comments)
- xray tool canonical set: `tools/master_gradient_xray.py:97-120` (NEW module-level frozenset)
- Downstream MPS-VIABLE prescreen consumer (reads `[MPS-PROXY]`): `src/tac/cathedral_consumers/mps_viable_prescreen_consumer/__init__.py` + `src/tac/tests/test_mps_viable_prescreen_consumer.py:108-109`
- Sister catalog gates: #110/#113 (HISTORICAL_PROVENANCE APPEND-ONLY) + #117/#157/#174 (canonical serializer family) + #125 (6-hook wire-in) + #229 (premise verification) + #230 (sister ownership map) + #287 (canonical Provenance + placeholder-rationale rejection) + #290 (canonical-vs-unique decision per layer) + #294 (9-dim checklist) + #303 (cargo-cult audit) + #305 (observability surface) + #335 (cathedral consumer canonical contract) + #340 (sister-checkpoint guard) + #341 (cathedral consumer routing markers — primary META gate this remediation closes).

---

## Operator-routable next steps (none — this CLOSES the loop)

This landing CLOSES sister II op-routable #1 (Path A) + #2 (Path B) per the canonical 2-path remediation framing. The 3rd op-routable from sister II (council-grade Catalog #341 contract clarification at the canonical contract level) is OPTIONAL and not blocking — both Path A and Path B together achieve 100% Catalog #341 compliance without requiring a council deliberation about whether the contract itself should be revised. The current state is **HARD-EARNED canonical alignment** between the xray tool's strict check + the canonical upstream sources (`NON_PROMOTABLE_TAGS` + `CANONICAL_MEASUREMENT_AXES`) + the per-consumer waiver discipline.

**No new operator-routable surfaced.** Slot JJ scope is bounded; the canonical observability surface is closed at 34/34 = 100% compliance with belt-and-suspenders structure.
