# Cargo-cult unwind audit: `formula_extinctions_consumer` (Wave 2C)

- **Date**: 2026-05-19
- **Subagent**: WAVE-2C-CARGO-CULT-UNWIND-AUDIT (`wave_2c_cargo_cult_audit_20260519`)
- **Target**: `src/tac/cathedral_consumers/formula_extinctions_consumer/__init__.py` (56 LOC; commit `179dc2501`)
- **Canonical reference**: `src/tac/cathedral_consumers/_example_consumer/__init__.py` (64 LOC)
- **Upstream namespace audited**: `tac.formula_extinctions` (canonical formula-derived constants: warmup_steps / validation_split / qint_max_grid / inflate_device_pin_metadata / bayesian_aggregation_quorum / frontier_threshold_from_state / early_stopping_patience)

## Source-text differential vs `_example_consumer`

`formula_extinctions_consumer` differs from the template by ~10 lines: docstring naming `tac.formula_extinctions` + `CONSUMER_NAME` literal + rationale string enumerating 7 canonical formula outputs. Body of `update_from_anchor` and `consume_candidate` is byte-identical to the template (both discard input + return canonical zero-adjustment dict).

Notably: declares ONLY `HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH` (no hook #5 — better honesty than `atom_consumer`).

## Canonical-vs-unique decision per layer

Per Catalog #290 falling-rule list:

| Layer | Canonical decision | Verdict | Evidence |
|---|---|---|---|
| L1: `CONSUMER_NAME` literal | adopt `"formula_extinctions_consumer"` | **ADOPT_CANONICAL_BECAUSE_SERVES** | Auto-discovery registration requirement. |
| L2: `CONSUMER_HOOK_NUMBERS = (CATHEDRAL_AUTOPILOT_DISPATCH,)` | adopt single-hook | **ADOPT_CANONICAL_BECAUSE_SERVES** | HARD-EARNED — formula extinctions ARE deterministic constants (per docstring: *"Canonical formula outputs are deterministic and do not require posterior update"*). Hook #5 would be a lie. This consumer is the IDEAL contract-honesty model the other 22 should mirror. |
| L3: `update_from_anchor(anchor)` body NO-OP | adopt template | **HARD-EARNED-FIRST-PRINCIPLES** | Deterministic constants have no posterior to update. The NO-OP body is correct because the consumer also dropped hook #5 from the declaration. |
| L4: `consume_candidate(candidate)` discards via `_ = candidate` | adopt template | **CARGO-CULTED-PATH-OF-LEAST-RESISTANCE** | Same class as atom_consumer #1 — discards the candidate. BUT: see cargo-cult #1 below; formula extinctions have a UNIQUE opportunity that other consumers don't. |
| L5: `axis_tag = "[predicted]"` | adopt canonical | **ADOPT_CANONICAL_BECAUSE_SERVES** | HARD-EARNED per CLAUDE.md "Apples-to-apples evidence discipline". |
| L6: `promotable = False` | adopt canonical | **ADOPT_CANONICAL_BECAUSE_SERVES** | HARD-EARNED per CLAUDE.md "Submission auth eval". |
| L7: `confidence = 0.0` | adopt canonical | **CARGO-CULTED-INHERITED-DEFAULT** | Template literal. Formula extinctions are PROOF-grade (deterministic, citation-bearing); confidence should be 1.0 not 0.0. |
| L8: rationale enumerating 7 formula outputs | adopt canonical | **ADOPT_CANONICAL_BECAUSE_SERVES** | Operator-readable + accurate. |

## Cargo-cult audit per assumption

Per Catalog #303. Four inherited assumptions, each classified:

### Assumption #1: "Formula extinctions don't apply to specific candidates — they're global constants"

- **Classification**: CARGO-CULTED-PATH-OF-LEAST-RESISTANCE
- **Evidence**: The discard `_ = candidate` treats formula outputs as global. BUT `tac.formula_extinctions` exposes `frontier_threshold_from_state` (per Catalog #316) which IS candidate-axis-dependent — different for `[contest-CPU]` vs `[contest-CUDA]`. And `inflate_device_pin_metadata` is per-archive (different inflate.py inline device-fork status per Catalog #205). A consumer that knows the candidate's score_axis + archive_sha could surface the per-candidate applicable formula values.
- **Why apparatus suppressed**: the template treats all consumers as namespace-discovery annotations. The unique opportunity (candidate-axis-aware formula lookup) requires reading the upstream `tac.formula_extinctions` API surface + understanding which functions are candidate-axis-keyed vs global. That's substrate-engineering depth (per HNeRV parity L7) that the template-clone discharge skipped.
- **Empirical anchor**: `tac.formula_extinctions.frontier_threshold_from_state` is canonical per Catalog #316 wire-in at `tools/cathedral_autopilot_autonomous_loop.py::_resolve_canonical_frontier_threshold_cpu`. The cathedral autopilot already calls a sister axis-keyed helper; the consumer wraps a different surface.
- **Unwind hypothesis**: replace `_ = candidate` with: (a) extract `candidate.get("score_axis")` (default `"contest_cpu"`); (b) call `tac.formula_extinctions.frontier_threshold_from_state(repo_root, axis=axis)`; (c) surface the per-axis frontier threshold in `rationale` (e.g. "applicable frontier_threshold for [contest-CPU] = 0.19205 — [predicted]") with `confidence = 1.0` (deterministic).
- **Unwind cost**: ~$0, ~20 LOC + 3 tests (axis-keyed lookup + missing-axis fallback + frontier-helper-unavailable graceful).
- **Predicted signal contribution**: operator dispatch decisions currently re-resolve the frontier manually; per-candidate auto-surface reduces ranking-review cycle. Estimated **-0.001 to -0.003 ΔS/month** via faster decisions.
- **Reactivation criterion**: 3 alternative reducers if axis-keyed surface proves noisy = (a) `qint_max_grid` per-archive lookup / (b) `early_stopping_patience` per-substrate / (c) `bayesian_aggregation_quorum` per-cost-band.

### Assumption #2: "`confidence = 0.0` is appropriate for formula constants"

- **Classification**: CARGO-CULTED-INHERITED-DEFAULT
- **Evidence**: Per the consumer's own docstring, formula outputs are *"PROOF-grade engineering outputs"*. They are deterministic functions of inputs. A confidence of 0.0 means "no evidence" but the evidence is the formula derivation itself.
- **Why apparatus suppressed**: same template-clone defense as cargo-cult #3 in atom_consumer audit.
- **Unwind hypothesis**: `confidence = 1.0` when the rationale cites a specific deterministic value; `confidence = 0.5` when listing availability only.
- **Unwind cost**: ~$0, 2 LOC.
- **Predicted signal contribution**: subsidiary to cargo-cult #1.

### Assumption #3: "Dropping hook #5 from CONSUMER_HOOK_NUMBERS is correct because update_from_anchor is NO-OP"

- **Classification**: HARD-EARNED-FIRST-PRINCIPLES
- **Evidence**: Honest declaration. Deterministic formula outputs genuinely have no posterior to update. This consumer is the IDEAL model — `atom_consumer` should follow this pattern.
- **Verdict**: ADOPT. No unwind. RECOMMEND PROMOTING THIS CONSUMER'S PATTERN as the canonical template-honesty exemplar.

### Assumption #4: "All formula extinctions belong in ONE rationale-line enumeration"

- **Classification**: CARGO-CULTED-INHERITED-DEFAULT (cosmetic only)
- **Evidence**: The 7-formula enumeration in one rationale string is dense. Per CLAUDE.md "Max observability — non-negotiable" facet #2 (decomposable per signal), per-formula structured output would be more queryable.
- **Why apparatus suppressed**: template constrains rationale to a single string per Protocol contract (line 6194 `[:512]`).
- **Unwind hypothesis**: tightly coupled to cargo-cult #1 — once the consumer surfaces per-candidate-applicable values, rationale naturally narrows to the 1-2 formulas actually invoked.
- **Unwind cost**: subsidiary to #1.

## Observability surface

Per Catalog #305:

1. **Inspectable per layer**: rationale + hook declaration + module docstring. ✓
2. **Decomposable per signal**: today NO (monolithic rationale enumeration). Post-unwind: YES (per-candidate-applicable formula values).
3. **Diff-able across runs**: today NO (identical template invocation). Post-unwind: YES (axis-dependent values diff with the candidate's score_axis).
4. **Queryable post-hoc**: partial — `consumer_invocations` persisted. Formula values stable across runs so post-hoc lookup re-resolves cleanly.
5. **Cite-able**: today NO (no formula-identity field). Post-unwind: cite formula name + canonical helper module path.
6. **Counterfactual-able**: today NO. Post-unwind: changing the candidate's score_axis changes the surfaced threshold (counterfactual via input mutation).

**Conclusion**: 1-of-6 facets today. Cargo-cult #1 unwind would lift to 5-of-6.

## Unwind priority queue

Ranked by `|predicted ΔS-signal-contribution| / cost`:

| Rank | Unwind | Cargo-cult # | Cost | Predicted ΔS | EV ratio |
|---|---|---|---|---|---|
| 1 | Per-candidate axis-keyed `frontier_threshold_from_state` lookup + confidence=1.0 + rationale surfaces the actual numeric value | #1 + #2 (composite) | ~$0, 20 LOC, 3 tests | -0.001 to -0.003 ΔS/month indirect | MEDIUM-HIGH |
| 2 | (subsidiary to #1) Per-formula structured rationale + sister-helper lookups (qint_max_grid / early_stopping_patience) | #4 | ~$0, +10 LOC | subsidiary | LOW |

**Top-1 unwind**: cargo-cult #1+#2 composite. ~30 min implementation.

## Cross-references

- This consumer's `CONSUMER_HOOK_NUMBERS = (CATHEDRAL_AUTOPILOT_DISPATCH,)` honesty pattern should be PROMOTED as the canonical template-honesty exemplar; `atom_consumer` + sister Wave 2C consumers should follow.
- Sister of Catalog #316 (`check_reports_latest_md_not_stale_vs_canonical_frontier`) — Catalog #316 wires `tac.frontier_scan.scan_reports_latest_md` to the autopilot; `tac.formula_extinctions.frontier_threshold_from_state` is the canonical computation that wraps it.
- Anchor memo: `feedback_canonical_share_when_serves_unique_when_suppresses_standing_directive_20260515.md` (operator's "tools when serve" doctrine; formula extinctions ARE a canonical helper that SERVES).
- Empirical evidence: `tools/cathedral_autopilot_autonomous_loop.py:6236` *"observability-only ranker bypass"*.

## Verdict

`formula_extinctions_consumer` has the HEALTHIEST hook-declaration discipline of all 12 Wave 2C consumers (drops hook #5 honestly). Its CARGO-CULTED issue (#1) is the highest-value unwind candidate in the Wave 2C set because `tac.formula_extinctions` exposes axis-keyed helpers that genuinely vary per-candidate. Per CLAUDE.md "Forbidden premature KILL" the lane is DEFER + REQUEST-REINVESTIGATION; reactivation criteria documented above.


<!-- # FORMALIZATION_PENDING:pre_framework_memo_dated_2026-05-19_predates_canonical_equations_birthday_registry_population_in_progress_appended_by_strict_flip_enablers_per_operator_blanket_approval_per_claude_md_forbidden_premature_kill_without_research_exhaustion_this_is_DEFER_pending_canonical_equation_backfill_NOT_kill -->
