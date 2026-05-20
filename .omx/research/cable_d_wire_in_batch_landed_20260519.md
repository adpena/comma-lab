# Cable D wire-in batch landed — D2+D3 cathedral consumers + D1+D4 audit

**Date:** 2026-05-20T01:30:00Z
**Authority:** T3 council prioritization 2026-05-19 (`.omx/research/council_t3_tier_45_backlog_prioritization_20260519.md` commit `79bd5695d`) rank #5 — "Cable D wire-in batch unlocks bit-allocator + sensitivity-consumer wave"; operator blanket approval 2026-05-19 "approved on all"
**Lane:** `lane_cable_d_wire_in_batch_d2_d3_20260519` (L1: impl_complete + memory_entry)
**Sister coordination:** disjoint scope — preservation audit (`~/.claude/` memory; not touched); cathedral autopilot activation; B1 dispatch; codex CLI external. Catalog #340 sister-checkpoint guard PROCEED.

---

## Canonical-vs-unique decision per layer

| Layer | Decision |
|---|---|
| Consumer package shape | ADOPT_CANONICAL — mirror existing `per_pair_pareto_envelope_consumer` pattern (5-token contract + SPDX header) per Catalog #335 |
| HookNumber declarations | UNIQUE per consumer — per-pair has hooks #1+#4+#5; aggregate adds #3 (bit-allocator) per Cable D D3 design |
| Canonical loader routing | ADOPT_CANONICAL — `tac.master_gradient_consumers.load_per_pair_gradient_from_anchor` + `load_aggregate_gradient_from_anchor` (existing fcntl-locked authority-filter pattern per Catalog #327) |
| Return-value shape | ADOPT_CANONICAL — Catalog #341 non-promotable markers (`predicted_delta_adjustment=0.0` / `promotable=False` / `axis_tag="[predicted]"`) |
| Exception handling | ADOPT_CANONICAL — `(FileNotFoundError, ValueError, OSError)` graceful degradation; no exception propagation per `consume_candidate` Protocol |
| D1+D4 audit scope | UNIQUE — surface canonical-key-set gaps as operator-routable follow-on rather than silently extending canonical schemas (council-grade per "Design decisions — non-negotiable") |

## 9-dimension success checklist evidence

- **UNIQUENESS**: D2+D3 are the FIRST cathedral consumer wrappers that surface the per-pair AND aggregate master-gradient ANCHOR-PRESENCE annotations as cathedral observability signals. Sister wrappers (`per_pair_pareto_envelope_consumer` etc.) wrap downstream CONSUMED signals, not the producer surface itself.
- **BEAUTY+ELEGANCE**: each consumer is ~180 LOC + 33 dedicated tests; ALL canonical-pattern code; no orchestration layer added.
- **DISTINCTNESS**: per-pair consumer wires (sensitivity #1 + cathedral #4 + CL #5); aggregate consumer adds bit-allocator #3 per the canonical shape difference (aggregate gradient ranks bit budget; per-pair gradient feeds sister per_pair_* consumer chain).
- **RIGOR**: 33 dedicated tests; sister regression (88 Catalog #335/#336/#337 + Catalog #341 = 115 tests) ALL PASS. Empirical anchor: live auto-discovery shows 35 consumers (was 33) with both new consumers contract-compliant.
- **OPTIMIZATION-PER-TECHNIQUE**: each consumer reads anchor on-demand (STATELESS); no caching; `update_from_anchor` is no-op by design (Catalog #131 sister state lives in canonical JSONL ledger, not in-consumer state).
- **STACK-OF-STACKS-COMPOSABILITY**: per-pair consumer surfaces the gradient that downstream sister consumers (Cable D batch consumers 7-14) consume via canonical typed loaders; aggregate consumer surfaces the gradient that DuckDB per-byte sensitivity table + `engineered_correction_targeting_consumer` + `gradient_informed_decoder_pruning_consumer` consume.
- **DETERMINISTIC-REPRODUCIBILITY**: pure functions of `(candidate, canonical ledger state)`; same archive_sha256 + same ledger = same return value.
- **EXTREME-OPTIMIZATION-PERFORMANCE**: O(1) per candidate (canonical loader is O(N_anchors) but anchors count is bounded; per-candidate work is one lookup + one dict construction).
- **OPTIMAL-MINIMAL-CONTEST-SCORE**: enables hook #4 cathedral autopilot to surface master-gradient anchor presence as ranking signal; structural enabler for `frontier_breaking` work via sister consumers that consume the SAME canonical gradient artifacts.

## Observability surface

- **Per-consumer**: `consume_candidate` return value is a dict with 5 canonical fields (predicted_delta_adjustment / rationale / axis_tag / promotable / confidence) inspectable per-call.
- **Live auto-discovery**: `tools/cathedral_autopilot_autonomous_loop.discover_and_register_consumers()` returns the consumer's `ConsumerRegistration` with `contract_compliant` + `consumer_hook_numbers` for runtime audit.
- **Test traces**: pytest verbose `-vv` provides per-test signal across 33 dedicated + 88 sister regression tests.
- **Diff-able across runs**: each consumer is STATELESS so anchor-state evolution in `.omx/state/master_gradient_anchors.jsonl` is the only signal source; sister tools provide diff visualization.
- **Post-hoc queryable**: cathedral autopilot ranker logs every consumer's contribution at INFO level; consumer outputs are part of the candidate row.
- **Counterfactual-able**: per Catalog #139 byte-mutation discipline applies to upstream anchor sources, not to this consumer's return values directly (the consumer is a SURFACING wrapper).

## Cargo-cult audit per assumption

| Assumption | Classification |
|---|---|
| Per-pair vs aggregate gradient deserve SEPARATE cathedral consumer packages | HARD-EARNED — Cable D landing memo declares 8 sister consumers (consumers 7-14) all consume per-pair gradient; aggregate gradient is consumed by `per_byte_sensitivity_consumer` + DuckDB ext; different shape + different downstream surface justifies separation |
| Catalog #341 markers (predicted_delta_adjustment=0.0 / promotable=False / axis_tag=[predicted]) apply to ANCHOR-PRESENCE annotations | HARD-EARNED — per CLAUDE.md "Apples-to-apples evidence discipline" + Catalog #287/#341, the presence of an anchor is NOT a score claim; it's a planning signal |
| HookNumber declarations should reflect WHERE the gradient is CONSUMED, not where this consumer IS | HARD-EARNED — Catalog #125 hook semantics are about the wire-in surface; per-pair gradient wires sensitivity_map (#1) because downstream sister consumers feed `tac.sensitivity_map.axis_weights`; per_pair does NOT wire bit-allocator because per-pair gradient feeds per-pair PLAN consumers, not aggregate byte budget |
| `update_from_anchor` should be no-op for STATELESS consumers | HARD-EARNED — Catalog #131/#138 sister discipline keeps state in fcntl-locked canonical JSONL ledgers; no consumer should mirror state in process memory |
| D1+D4 canonical-key-set extensions are council-grade | HARD-EARNED — extending `_REAL_SENSITIVITY_TEXT_KEYS` or `MultiGranularitySensitivityRun` schema affects downstream consumers + tests; per "Design decisions — non-negotiable" this is council-grade not subagent-grade |

## Horizon class

`frontier_protecting` — the wire-in is hardening infrastructure that ENABLES cathedral autopilot to surface master-gradient anchor presence as a ranking signal. The signal produced here does not itself lower the contest score; it enables the optimization loop that lowers the contest score via sister consumer downstream chains.

---

## D2 outcome — `master_gradient_per_pair_consumer` (NEW)

**File**: `src/tac/cathedral_consumers/master_gradient_per_pair_consumer/__init__.py` (~180 LOC)

**Hook wiring** (3 hooks): SENSITIVITY_MAP (#1) + CATHEDRAL_AUTOPILOT_DISPATCH (#4) + CONTINUAL_LEARNING_POSTERIOR (#5).

**Producer routing**: reads via `tac.master_gradient_consumers.load_per_pair_gradient_from_anchor(archive_sha256=...)` which internally:
- Filters anchors by `_anchor_has_tensor_kind(a, PER_PAIR_GRADIENT_TENSOR_KIND)`
- Filters by `is_authoritative_axis_anchor` (per Catalog #327 contest-axis custody)
- Returns `(np.ndarray of shape (N_bytes, N_pairs, 3), anchor_dict)`

**Cathedral contribution** (per Catalog #341):
- `predicted_delta_adjustment=0.0` (observability-only)
- `promotable=False` (no score claim)
- `axis_tag="[predicted]"` (planning signal, not contest-axis)
- `rationale`: cites archive sha[:12] + N_bytes + N_pairs + N_axes + measurement_axis + measurement_hardware + names downstream sister consumers
- `confidence=0.0` (consumer is OBSERVABILITY, not prediction-source)

**Graceful degradation**:
- Missing archive_sha256 → canonical markers + "no anchor lookup attempted" rationale
- archive_sha256 too short (<8 chars) → canonical markers + "short sha" rationale
- ImportError on `tac.master_gradient_consumers` → canonical markers + "import unavailable" rationale
- (FileNotFoundError, ValueError, OSError) on anchor lookup → canonical markers + "no authoritative anchor" rationale

**Per Catalog #318 raw-byte-authority-guard**: NEVER returns raw byte tensors; the cathedral contribution is shape + axis presence only.

## D3 outcome — `master_gradient_aggregate_consumer` (NEW)

**File**: `src/tac/cathedral_consumers/master_gradient_aggregate_consumer/__init__.py` (~180 LOC)

**Hook wiring** (4 hooks): SENSITIVITY_MAP (#1) + BIT_ALLOCATOR (#3) + CATHEDRAL_AUTOPILOT_DISPATCH (#4) + CONTINUAL_LEARNING_POSTERIOR (#5).

**Producer routing**: reads via `tac.master_gradient_consumers.load_aggregate_gradient_from_anchor(archive_sha256=...)` which internally:
- Filters anchors by `_anchor_has_tensor_kind(a, AGGREGATE_GRADIENT_TENSOR_KIND)`
- Filters by `is_authoritative_axis_anchor`
- Returns `(np.ndarray of shape (N_bytes, 3), anchor_dict)`

**Differs from D2** in:
- Wires hook #3 BIT_ALLOCATOR (aggregate per-byte importance ranks bit budget; per-pair gradient feeds per-pair PLAN consumers, not budget)
- Loader returns 2D `(N_bytes, 3)` tensor vs 3D `(N_bytes, N_pairs, 3)` from per-pair

**Cathedral contribution + graceful degradation**: identical Catalog #341 pattern to D2.

## D1 outcome — sensitivity_map cite-chain audit (operator-routable follow-on)

**Audit verdict**: PARTIAL — `tac.sensitivity_map` already enforces SOURCE-archive + MODEL + STATE-DICT + CERTIFICATION-SUMMARY sha-chain provenance via 4 canonical key tuples; HOWEVER `commit_sha` / `call_id` / `upstream_snapshot_sha256` (the Catalog #305 6-facet cite-chain trio) are NOT in any current canonical key set:

- `_REAL_SENSITIVITY_TEXT_KEYS` = `{evidence_grade, kind, mode, notes, provenance, source, source_kind, status, tag}`
- `_REAL_SENSITIVITY_SOURCE_SHA_KEYS` = `(source_archive_sha256, source_archive_sha, baseline_archive_sha256)`
- `_REAL_SENSITIVITY_MODEL_SHA_KEYS` = `(model_sha256, checkpoint_sha256, state_dict_sha256, state_dict_source_sha256, decoder_state_dict_sha256, model_state_dict_sha256, baseline_model_sha256, baseline_checkpoint_sha256)`
- `_REAL_SENSITIVITY_CERTIFICATION_SUMMARY_SHA_KEYS` = `(certification_summary_sha256, ...)`

**Why DEFERRED to follow-on (not silently extended)**: extending `_REAL_SENSITIVITY_TEXT_KEYS` is a canonical-key-set mutation that affects downstream consumers + tests. Per CLAUDE.md "Design decisions — non-negotiable" this is council-grade. Per CLAUDE.md "Forbidden premature KILL without research exhaustion" I am NOT killing this — I am DEFERRING-pending-council-grade-extension. Per Catalog #110/#113 HISTORICAL_PROVENANCE this preserves the existing canonical surface APPEND-ONLY.

**Operator-routable**: spawn a sub-symposium subagent to (a) extend `_REAL_SENSITIVITY_TEXT_KEYS` with the 3 cite-chain tokens, (b) thread the trio through the canonical `real_sensitivity_metadata_blockers` validator, (c) update sister tests + downstream consumers. Estimated scope: ~80 LOC + ~15 sister test edits.

## D4 outcome — DuckDB sensitivity table provenance audit (operator-routable follow-on)

**Audit verdict**: GOOD — `tac.canonical_duckdb.per_byte_sensitivity_ext.MultiGranularitySensitivityRun` already carries comprehensive provenance per Catalog #323 canonical Provenance:

- `run_id` + `archive_sha256` + `source_anchor_utc` (cite-able)
- `gradient_tensor_kind` + `gradient_array_path` + `gradient_array_sha256` (counterfactual-able)
- `class_source_path` + `class_source_sha256` + `class_basis` (decomposable per signal)
- `source_measurement_axis` + `evidence_grade` + `source_anchor_authoritative` (cite-chain anchored)
- `score_claim` + `promotion_eligible` + `ready_for_exact_eval_dispatch` + `blocker_reason` (Catalog #127 custody validator routing)

**Gap**: `commit_sha` + `call_id` + `upstream_snapshot_sha256` are NOT in the schema (same gap as D1).

**Why DEFERRED**: schema migration of `MultiGranularitySensitivityRun` dataclass is council-grade. The current schema is sufficient for Catalog #327 authority filtering + Catalog #323 canonical Provenance binding; the 3 cite-chain tokens are enrichment, not protection.

**Operator-routable**: same sub-symposium as D1; the schema extension can land jointly via APPEND-ONLY column-addition pattern.

---

## Test summary

| File | Tests | Status |
|---|---|---|
| `test_cable_d_wire_in_master_gradient_consumers.py` | 33 | PASS |
| sister Catalog #335 `test_check_335_cathedral_consumer_directory_contract.py` | 18 | PASS |
| sister Catalog #336/#337 `test_check_336_337_cathedral_main_discovery_invoker.py` | 27 | PASS |
| sister Catalog #341 `test_check_341_cathedral_consumer_mps_prescreen_routing.py` | 27 | PASS |
| sister auto-discovery `test_cathedral_autopilot_auto_discovery.py` | 13 | PASS |
| sister contract `test_cathedral_consumer_contract.py` | 30 | PASS |
| Total new + sister regression | **148** | **PASS** |

## 6-hook wire-in declaration (per Catalog #125)

1. **SENSITIVITY MAP**: ACTIVE — both new consumers wire hook #1; per-pair anchor presence + aggregate anchor presence are signals downstream sister consumers (per_byte_sensitivity / engineered_correction / Pareto envelope / Lagrangian lambda / KKT) consume to feed `tac.sensitivity_map.axis_weights`
2. **PARETO CONSTRAINT**: N/A — Pareto consumers (`per_pair_pareto_envelope_consumer`, `per_pair_kkt_residuals_consumer`, `per_pair_volterra_cross_terms_consumer`) handle hook #2 directly via the canonical per-pair loader; these new consumers do NOT need to also declare hook #2
3. **BIT-ALLOCATOR**: ACTIVE for aggregate consumer (hook #3) — aggregate gradient ranks per-byte bit budget per `engineered_correction_targeting_consumer` + `gradient_informed_decoder_pruning_consumer`. NOT ACTIVE for per-pair consumer (per-pair gradient feeds per-pair PLAN consumers, not budget)
4. **CATHEDRAL AUTOPILOT DISPATCH**: ACTIVE — both new consumers are auto-discovered + invoked by `tools/cathedral_autopilot_autonomous_loop.discover_and_register_consumers` per Catalog #335/#336/#337
5. **CONTINUAL-LEARNING POSTERIOR**: ACTIVE — `update_from_anchor` hook declared (no-op by design; canonical state lives in fcntl-locked JSONL ledger per Catalog #131/#138/#245)
6. **PROBE-DISAMBIGUATOR**: N/A — these consumers SURFACE the master-gradient anchor presence; they do NOT disambiguate between competing producers (the producer surface IS canonical per `tac.master_gradient`)

## Cross-cable implications + sister handoff

- **Cathedral consumer count** went from 33 → 35 (+2). Catalog #335 auto-discovery validated; Catalog #336/#337 invocation regression preserved.
- **Sister Cable D batch consumers 7-14** (per `feedback_cable_d_master_gradient_extension_batch_landed_20260519T055121Z.md`) continue to operate via their canonical typed loaders; no change to their behavior.
- **D1 follow-on** (sensitivity_map cite-chain trio): operator-routable sub-symposium subagent.
- **D4 follow-on** (DuckDB sensitivity schema trio): same operator-routable sub-symposium subagent (joint with D1).
- **Highest-EV op-routable surfaced**: per the prompt's request, the highest-EV next step is wiring the `OptimalPerPairTreatmentPlan` consumer signal (the existing Q2+Q3 `adjust_predicted_delta_for_venn_classification_v2` cascade pattern at `tools/cathedral_autopilot_autonomous_loop.py:1373`) to additionally consume per-pair Pareto envelope + Lagrangian lambda + KKT residuals signals from Cable D consumers 7-14. The infrastructure is now READY; the ranker just needs to read.

## Cross-references

- Cable D batch landing 2026-05-19: `feedback_cable_d_master_gradient_extension_batch_landed_20260519T055121Z.md` (commit `6a1e94a63`)
- T3 council prioritization 2026-05-19: `.omx/research/council_t3_tier_45_backlog_prioritization_20260519.md` (commit `79bd5695d`)
- Battle plan 2026-05-19: `.omx/research/integrated_battle_plan_priority_queue_dag_cables_gates_20260519T052801Z.md` (commit `6a1e94a63`) — Cable D rows D2/D3/D4
- Canonical pattern reference: `src/tac/cathedral_consumers/per_pair_pareto_envelope_consumer/__init__.py`
- Canonical contract: `src/tac/cathedral/consumer_contract.py` (Catalog #335)
- Sister catalog gates:
  - Catalog #125 (subagent landing 6-hook wire-in)
  - Catalog #127 (per-call-site custody routing)
  - Catalog #131 (fcntl-locked bare-write discipline)
  - Catalog #138 (strict-load fail-closed)
  - Catalog #229 (premise verification before edit)
  - Catalog #230 (sister-subagent ownership map)
  - Catalog #245 (canonical Modal call_id ledger)
  - Catalog #265 (symposium impls canonical contract — sister pattern)
  - Catalog #287 (phantom-API / placeholder rejection)
  - Catalog #318 (master-gradient raw-byte-authority guard)
  - Catalog #323 (canonical Provenance umbrella)
  - Catalog #327 (master-gradient contest-axis authority)
  - Catalog #335 (cathedral consumer directory contract — THIS landing's primary META gate)
  - Catalog #336/#337 (cathedral autopilot main invocation)
  - Catalog #341 (cathedral consumer routing markers)
- Sister forbidden patterns honored:
  - "Forbidden empirical-claim-without-evidence-tag" — every return value carries explicit `axis_tag="[predicted]"`
  - "Forbidden phantom-score directory trap" — no device-named output paths; consumer is STATELESS so no output paths exist
  - "Forbidden /tmp paths in any persisted artifact" — consumer never writes
  - "Forbidden force-canonical-without-evaluation-of-suppression" — explicit canonical-vs-unique decision per layer above

## Lane status

- Lane `lane_cable_d_wire_in_batch_d2_d3_20260519` registered at L0 via `tools/lane_maturity.py add-lane`
- Gates landed in this commit batch:
  - `impl_complete` ✓ (2 new consumers + 33 dedicated tests + memory entry)
  - `memory_entry` ✓ (this memo)
- Gates pending sister-subagent or operator follow-on:
  - `real_archive_empirical` — N/A (consumer is OBSERVABILITY wrapper; no archive bytes generated)
  - `contest_cuda` — N/A (consumer does not produce score claims)
  - `strict_preflight` — no NEW strict preflight gate needed (Catalog #335 + #341 cover the consumer contract + routing markers)
  - `three_clean_review` — adversarial review cycle (next subagent slot)
  - `deploy_runbook` — N/A for editor-only consumer wire-in work

Expected lane level after this commit: **L1** (impl_complete + memory_entry).

— Cable D wire-in batch subagent 2026-05-19 (claude_slot_aa_cable_d_wirein_batch_20260519)


<!-- WAVE-3-CATALOG-344-BACKFILL-SWEEP appended 2026-05-20 per operator NON-NEGOTIABLE "keep feeding the queue" + WIRE-IN-AUDIT-POST-CASCADE op-routable #3. -->
<!-- # FORMALIZATION_PENDING:cable-D-D2-D3-cathedral-consumer-wire-in-batch-landing-memo-existing-equations-referenced-not-new -->
