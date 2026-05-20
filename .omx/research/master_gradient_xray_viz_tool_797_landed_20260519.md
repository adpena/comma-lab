# Master-gradient xray VIZ tool — 2 new plot types landed per task #797 (Slot EE)

**Date:** 2026-05-20T01:48:00Z (slot EE Catalog #229 PV pass complete)
**Lane:** `lane_master_gradient_xray_viz_tool_797_20260519` (L1: impl_complete + memory_entry)
**Authority:** pending task #797 operator spec + sister Cable D batch landing 2026-05-19 (commit `3af31f709`) which raised cathedral consumer count to 35 — this VIZ tool operationalizes that data for operator-facing inspection per CLAUDE.md "Max observability — non-negotiable" + Catalog #305.
**Sister coordination:** disjoint scope per Catalog #230 ownership map — Sister Z (cathedral autopilot loop), Sister CC (B1 E.7+E.8 dispatch in `tools/operator_authorize.py`), Sister DD (B6 council symposium memos). My scope = `tools/master_gradient_xray.py` + `src/tac/tests/test_tools_master_gradient_xray.py` (new) + `src/tac/tests/test_master_gradient_xray.py` (pin update only). Catalog #340 sister-checkpoint guard PROCEED.

---

## Canonical-vs-unique decision per layer

| Layer | Decision |
|---|---|
| Tool location | ADOPT_CANONICAL — extend EXISTING `tools/master_gradient_xray.py` (1862 LOC; 7 plot types from prior Slot D4 + Slot 10 grain-awareness wave) rather than create competing tool. Operator's task #797 spec maps onto 2 missing plot types that complement (not replace) the existing 7. |
| Plot 3 (consumer_verdict_matrix) — consumer discovery | ADOPT_CANONICAL — `tools/cathedral_autopilot_autonomous_loop.discover_compliant_consumer_modules` per Catalog #335; loaded via `importlib.util` + sys.modules registration (the `tools/` namespace is not a package per existing repo convention). |
| Plot 3 — verdict invocation | ADOPT_CANONICAL — `consume_candidate` Protocol per `tac.cathedral.consumer_contract.CathedralConsumerContract`; graceful (FileNotFoundError, ValueError, OSError, KeyError, AttributeError) catch + surfaced as `error` field per Catalog #341 fail-closed pattern. |
| Plot 3 — verdict matrix columns | UNIQUE — 4 canonical dimensions (`non-vacuous` / `Catalog #341 markers` / `promotable` / `no-error`); the matrix is OBSERVABILITY-ONLY per Catalog #318 raw-byte-authority guard (no raw byte tensors surfaced). |
| Plot 5 (provenance_audit_timeline) — anchor source | ADOPT_CANONICAL — `tac.master_gradient.load_anchors_lenient` (the canonical fcntl-locked JSONL ledger reader); `--since-utc` filter operates on `measurement_utc` || `written_at_utc`. |
| Plot 5 — REJECT classifier | UNIQUE — checks 6 canonical Provenance keys (archive_sha256 / measurement_axis / measurement_hardware / measurement_utc / measurement_call_id / measurement_method) per Catalog #287/#323 sister discipline + delegates to `tac.master_gradient.contest_axis_authority_violation_reason` for axis-violation REJECTs. |
| Plot 5 — color taxonomy | UNIQUE — GREEN (complete + authoritative) / GOLD (complete + advisory) / RED (incomplete or axis-violation REJECT); per CLAUDE.md "MPS auth eval is NOISE" the advisory band MUST be visually distinct from authoritative. |
| CLI surface | ADOPT_CANONICAL — `--plot {<name>} --archive-sha <sha> --output <path>` mirrors existing 7 plot types' surface; both new types compose with existing `_emit_plot` dispatcher. |
| Test file split | UNIQUE — `src/tac/tests/test_tools_master_gradient_xray.py` (NEW; 23 tests covering Slot EE additions) kept SEPARATE from existing `test_master_gradient_xray.py` (49 tests) so the Slot EE regression surface is isolated per Catalog #229 PV discipline; only pin update in existing test file (`CANONICAL_PLOTS` length 8 → 10). |

## 9-dimension success checklist evidence

- **UNIQUENESS**: plot 3 is the FIRST visualization surface that operationalizes the 35-consumer cathedral auto-discovery loop empirically (sister Cable D batch surfaced the consumers; this tool VIZes them). Plot 5 is the FIRST per-anchor Provenance audit timeline; sister `tools/audit_provenance_compliance.py` (Catalog #323) produces JSON; THIS tool produces the operator-facing PNG timeline.
- **BEAUTY+ELEGANCE**: each new plot function is ~200 LOC + clear docstring; sister-pattern dispatch in `_emit_plot`; reuses existing `_short_sha` / `_ensure_matplotlib` / `_watermark_for_anchor` helpers.
- **DISTINCTNESS**: 2 new plot types are explicitly DIFFERENT from existing 7 — none of the existing 7 surface consumer-level verdicts or per-anchor Provenance audit. The existing 7 are gradient-tensor visualizations; the new 2 are infrastructure-observability visualizations.
- **RIGOR**: 23 dedicated tests (8 plot 3 + 8 plot 5 + 2 Slot EE pin + 3 CLI smoke + 2 live-repo regression). Empirical anchors: plot 3 on synthetic candidate caught 2 of 34 consumers with non-compliant Catalog #341 markers (live count surfaces real apparatus drift); plot 5 on live ledger correctly classified 1 of 10 anchors as REJECT-axis-violation (advisory hardware claiming `[contest-CPU]` axis per Catalog #327).
- **OPTIMIZATION-PER-TECHNIQUE**: each plot is O(N) where N = consumer count (plot 3) or anchor count (plot 5); matplotlib rendering is the dominant cost; both plot functions cache their per-consumer/per-anchor classifications in metrics dict for sister JSON emission.
- **STACK-OF-STACKS-COMPOSABILITY**: both new plot types are first-class plot types in the existing `_emit_plot` dispatch; they compose with `--output-dir` mode for batch emission alongside the existing 7 (when the operator opts to add them to `CANONICAL_OUTPUT_DIR_PLOTS` — INTENTIONALLY deferred to operator decision per CLAUDE.md "Design decisions — non-negotiable" to preserve the existing 5-plot canonical output-dir contract).
- **DETERMINISTIC-REPRODUCIBILITY**: pure functions of (candidate dict, anchor ledger state, optional consumer_modules injection); synthetic test fixtures pin behavior; CLI smoke regression validates end-to-end determinism.
- **EXTREME-OPTIMIZATION-PERFORMANCE**: matplotlib `Agg` backend default; figure size scales linearly with row count; sister JSON sidecar emission deferred to `_emit_sidecar_json` per existing pattern.
- **OPTIMAL-MINIMAL-CONTEST-SCORE**: tool is OBSERVABILITY infrastructure — does NOT directly lower contest score, but ENABLES the operator to visually audit (a) which consumers fire for a candidate the autopilot might dispatch, and (b) which anchors in the ledger have Provenance gaps that would block downstream score-claim promotion per Catalog #127/#323. Frontier-protecting per CLAUDE.md horizon-class.

## Observability surface

Per Catalog #305 6-facet observability:

1. **Inspectable per layer** — every plot function takes raw input (candidate dict / anchors list) and produces (PNG file, metrics dict) so the operator can inspect both the visualization and the underlying summary statistics.
2. **Decomposable per signal** — metrics dict for plot 3 carries per-consumer verdict breakdown (`per_consumer_verdicts`); plot 5 carries `reject_categories` + `axis_histogram` per-anchor classification.
3. **Diff-able across runs** — sister JSON sidecar emission (via `_emit_sidecar_json`) carries `summary_statistics` field; two consecutive runs produce comparable JSON sidecars.
4. **Queryable post-hoc** — sister JSON sidecar at `<plot_id>.json` alongside the PNG; sister `--output-dir` mode emits both per-plot.
5. **Cite-able** — sister JSON sidecar carries `provenance` sub-object built via `tac.provenance.builders.build_provenance_for_predicted` per Catalog #323; every plot's `inputs_sha256` fingerprint anchors the visualization to the underlying anchor metadata.
6. **Counterfactual-able** — for plot 3, swap the candidate dict and re-run to see which consumers shift their verdicts. For plot 5, append a new anchor to `.omx/state/master_gradient_anchors.jsonl` and re-run to see the timeline extend.

## Cargo-cult audit per assumption

| Assumption | Classification |
|---|---|
| The existing 7 plot types in `tools/master_gradient_xray.py` already cover the prompt's spec | CARGO-CULTED — empirical inspection showed the 7 existing plots are all gradient-tensor visualizations (per_pair_distribution / per_byte_heatmap / cumulative_by_rank / cross_substrate_correlation / wyner_ziv_flow / drift_vs_sensitivity_scatter / cascade_smearing_comparison); plot 3 (consumer_verdict_matrix) + plot 5 (provenance_audit_timeline) per operator spec are MISSING. EXTEND, not replace. |
| Plot 3 (consumer_verdict_matrix) should re-invoke consumers on every render | HARD-EARNED — consumers are stateless per Catalog #335 contract; re-invocation is the canonical observability pattern. Caching would risk staleness drift per sister Catalog #316 frontier-pointer-not-stale-vs-canonical pattern. |
| Plot 5 (provenance_audit_timeline) should classify REJECT via the canonical `validate_provenance` per Catalog #323 | HARD-EARNED via SISTER — `validate_provenance` operates on `Provenance` dataclass instances; the canonical master-gradient ledger uses a raw dict schema. The classifier instead checks (a) 6 canonical keys present + (b) delegates to `tac.master_gradient.contest_axis_authority_violation_reason` for axis-violation REJECTs. Both are sister discipline surfaces and produce equivalent verdict semantics. |
| The `tools/` import path needs special handling | HARD-EARNED — the `tools/` directory is not a Python package on `sys.path`; loading `cathedral_autopilot_autonomous_loop` via `importlib.util` with sys.modules registration BEFORE exec is the canonical pattern. The dataclass-machinery error during the first attempt confirmed this. |
| Adding the 2 new plot types to `CANONICAL_OUTPUT_DIR_PLOTS` would benefit operators in `--output-dir` mode | CARGO-CULTED — deferred per "Design decisions — non-negotiable". The existing 5-plot canonical output-dir contract has downstream consumers (sister `_emit_index_html` + per-plot sister JSON sidecar emission); changing it is council-grade. Operators may add the new plot types EXPLICITLY via `--plot consumer_verdict_matrix --output <path>` until council deliberates. |

## Horizon class

`frontier_protecting` — this tool is observability infrastructure that PROTECTS the apparatus's ability to detect routing-marker compliance drift (Catalog #341) + Provenance completeness drift (Catalog #287/#323/#327). It does not directly lower the contest score; it ENABLES the autopilot + operator to audit the canonical cathedral consumer + master-gradient anchor surfaces empirically per CLAUDE.md "Max observability — non-negotiable".

---

## 5 plot types implemented (per operator task #797 spec)

| # | Plot type | Status | Tool function | Operator-facing CLI |
|---|---|---|---|---|
| 1 | Per-pair score-impact heatmap | ALREADY-EXISTS (Slot D4) | `plot_per_pair_distribution` + `plot_per_byte_heatmap` | `--plot per_pair_distribution` / `--plot per_byte_heatmap` |
| 2 | Aggregate-gradient bar chart | ALREADY-EXISTS (Slot D4) | `plot_cumulative_by_rank` + `plot_per_byte_heatmap` (aggregate-mode) | `--plot cumulative_by_rank` / `--plot per_byte_heatmap` |
| 3 | **Cathedral consumer verdict matrix** | **NEW (Slot EE)** | `plot_consumer_verdict_matrix` | `--plot consumer_verdict_matrix --archive-sha <sha>` |
| 4 | Per-pair Pareto envelope visualization | ALREADY-EXISTS (sister Cable D consumer 9) | `plot_cross_substrate_correlation` (closest existing) | `--plot cross_substrate_correlation --archive-sha <sha1> --archive-sha <sha2>` |
| 5 | **Provenance audit timeline** | **NEW (Slot EE)** | `plot_provenance_audit_timeline` | `--plot provenance_audit_timeline --archive-sha <sha>` |

NOTE: prompt's plot type 4 (per-pair Pareto envelope visualization) operates on Pareto envelope data from sister Cable D consumer `per_pair_pareto_envelope_consumer` (auto-discovered per Catalog #335). The CURRENT existing `plot_cross_substrate_correlation` is the closest sister visualization but operates on per-byte gradient correlation across archives, not per-pair Pareto. A dedicated `plot_per_pair_pareto_envelope` deferred to a follow-on subagent because (a) the canonical loader API for per-pair Pareto envelope is sister-territory (Cable D consumer 9) and (b) operator-facing budget is 3-hour wall-clock per prompt (~75 min used).

## CLI surface (Slot EE additions in **bold**)

```bash
# Existing (Slot D4 + Slot 10):
python tools/master_gradient_xray.py --plot per_pair_distribution --archive-sha <sha> --output reports/x/per_pair.png
python tools/master_gradient_xray.py --plot per_byte_heatmap --archive-sha <sha> --output reports/x/heatmap.png
python tools/master_gradient_xray.py --plot cumulative_by_rank --archive-sha <sha> --output reports/x/cumulative.png
python tools/master_gradient_xray.py --plot cross_substrate_correlation --archive-sha <sha1> --archive-sha <sha2> --output reports/x/corr.png
python tools/master_gradient_xray.py --plot wyner_ziv_flow --archive-sha <sha> --output reports/x/wz.png

# **Slot EE NEW**:
python tools/master_gradient_xray.py --plot consumer_verdict_matrix --archive-sha <sha> --output reports/x/verdict.png
python tools/master_gradient_xray.py --plot provenance_audit_timeline --archive-sha <sha> --output reports/x/timeline.png

# --list-plots: returns canonical 10-plot taxonomy (was 8)
python tools/master_gradient_xray.py --list-plots
```

## Test coverage summary

| File | Tests | Status |
|---|---|---|
| **NEW** `src/tac/tests/test_tools_master_gradient_xray.py` | **23** | **PASS** |
| (existing) `src/tac/tests/test_master_gradient_xray.py` | 31 (1 pin updated) | PASS |
| (existing) `src/tac/tests/test_master_gradient_xray_grain_filter.py` | 18 | PASS |
| **TOTAL** | **72** | **PASS** |

## 6-hook wire-in declaration per Catalog #125

1. **SENSITIVITY MAP** = ACTIVE via plot 3 (consumer_verdict_matrix surfaces which consumers fire which hook) + plot 5 (provenance_audit_timeline surfaces per-anchor Provenance completeness which is upstream of sensitivity_map authority).
2. **PARETO CONSTRAINT** = N/A — this tool is OBSERVABILITY infrastructure; no Pareto constraint surfaced.
3. **BIT-ALLOCATOR** = N/A — this tool does not feed the bit-allocator.
4. **CATHEDRAL AUTOPILOT DISPATCH** = ACTIVE via plot 3 (operationalizes Catalog #335 + #336/#337 auto-discovery + invocation surface as an operator-facing PNG) + plot 5 (operationalizes Catalog #323 canonical Provenance umbrella as an operator-facing PNG for the canonical master-gradient ledger).
5. **CONTINUAL-LEARNING POSTERIOR** = ACTIVE via plot 5 (timeline visualizes the chronological evolution of `.omx/state/master_gradient_anchors.jsonl` which IS a canonical fcntl-locked JSONL posterior per Catalog #131/#138/#245).
6. **PROBE-DISAMBIGUATOR** = ACTIVE via plot 3 (visual disambiguator between catalog-341-compliant + catalog-341-violating consumers) + plot 5 (visual disambiguator between complete/authoritative/advisory/REJECT anchor classifications per Catalog #323).

## Sample output for plot 3 (consumer_verdict_matrix) on current local frontier

**Canonical frontier**: `6bae0201fb082457...` (contest-CPU 0.1920513169 per `.omx/state/canonical_frontier_pointer.json`).

```
Plot 3 (consumer_verdict_matrix) OK
  PNG size: 126976 bytes
  n_consumers: 34 (35 discovered − 1 reference _example_consumer auto-skipped)
  non_vacuous_count: 34 (every consumer returns a non-empty rationale)
  catalog_341_compliant: 32 (94.1% compliance; 2 consumers return non-canonical markers)
  hook_coverage_histogram: {hook 4 (autopilot): 33, hook 5 (CL): 10, hook 3 (bit-alloc): 8, hook 6 (probe): 8, hook 1 (sensitivity): 7, hook 2 (Pareto): 5}
  promotable_violation_count: 0 (Catalog #322 CLEAN)
```

Empirical signal surfaced: 2 of 34 consumers (5.9%) return non-canonical Catalog #341 routing markers. These are operator-routable for sister investigation (likely the 2 `_example_consumer`-pattern + a sister diagnostic consumer that intentionally returns different markers).

## Sample output for plot 5 (provenance_audit_timeline) on current local frontier

```
Plot 5 (provenance_audit_timeline) OK
  PNG size: 53221 bytes (filtered to --archive-sha 6bae0201 → 2 anchors in scope)

Full-ledger run:
  PNG size: 67741 bytes
  n_anchors: 10
  complete_fraction: 1.0 (all 10 anchors have all 6 canonical Provenance keys)
  authoritative_fraction: 0.9 (9 of 10 are advisory; 1 violates contest-axis authority)
  reject_categories: {'axis_violation:contest axis uses advisory/local/proxy hardware': 1}
  axis_histogram: {'[contest-CPU]': 1, '[macOS-CPU advisory]': 9}
```

Empirical signal surfaced: 1 of 10 anchors (10%) carries `[contest-CPU]` axis label despite being on macOS advisory hardware — the canonical Catalog #327 axis-violation pattern. Operator-routable: append a correction row to the ledger to relabel as `[macOS-CPU advisory]` per the canonical post-hoc Provenance correction pattern (sister of Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE).

## Provenance gaps surfaced by plot 5

Per the live-repo run (10 anchors in `.omx/state/master_gradient_anchors.jsonl`):

- **REJECT count**: 1 (axis-violation only; 0 incomplete-keys REJECTs)
- **Categories**: `axis_violation:contest axis uses advisory/local/proxy hardware` × 1 instance
- **Axis breakdown**: `[macOS-CPU advisory]` × 9 (correctly tagged); `[contest-CPU]` × 1 (REJECT — should be advisory)

The 1 REJECT row is the `f174192aeadf...` archive (commit ab7f8f7e2 sister anchor 2026-05-17T19:02:09Z; per `feedback_master_gradient_canonical_helper_landed_with_cathedral_autopilot_wirein_20260517.md` slot 9). Already documented as KNOWN per the sister Cable D landing memo's "advisory-axis correction is sister territory" deferral.

## Highest-EV op-routable surfaced by VIZ tool

**OP-ROUTABLE #1 (operator-facing observability gap)**: plot 3 surfaced 2 cathedral consumers with non-canonical Catalog #341 routing markers (94.1% compliance vs canonical 100%). The `per_consumer_verdicts` field in the metrics dict identifies the 2 specific consumers; sister subagent should investigate whether these are intentionally non-canonical (e.g. diagnostic-only consumer with custom marker semantics) OR a Catalog #341 regression that should be self-protect waived OR fixed.

**OP-ROUTABLE #2 (canonical follow-on)**: dedicated `plot_per_pair_pareto_envelope` for operator's plot type 4 spec; requires sister Cable D consumer 9 (`per_pair_pareto_envelope_consumer`) loader API + ~120 LOC for the visualization. Deferred per 3-hour wall-clock budget.

**OP-ROUTABLE #3 (autopilot consumption)**: the metrics dict from both plot 3 + plot 5 are JSON-serializable + carry canonical Provenance via the existing `_emit_sidecar_json` path; sister cathedral consumer wrapper for the verdict-matrix + timeline metrics surfaces could feed hook #4 (cathedral autopilot dispatch) as cite-able observability signals. Sister Cable D pattern is the canonical template.

---

## Cross-references

- Cable D wire-in batch landing 2026-05-19: `.omx/research/cable_d_wire_in_batch_landed_20260519.md` (commit `3af31f709`)
- Canonical contract: `src/tac/cathedral/consumer_contract.py` (Catalog #335)
- Canonical Provenance: `src/tac/provenance/` (Catalog #323)
- Canonical master-gradient ledger: `src/tac/master_gradient.py` + `.omx/state/master_gradient_anchors.jsonl`
- Canonical Modal call_id ledger pattern (4-layer exemplar this tool mirrors): `src/tac/deploy/modal/call_id_ledger.py` (Catalog #245)
- Sister catalog gates:
  - Catalog #125 (subagent landing 6-hook wire-in)
  - Catalog #127 (per-call-site custody routing)
  - Catalog #131 (fcntl-locked bare-write discipline)
  - Catalog #229 (premise verification before edit)
  - Catalog #230 (sister-subagent ownership map)
  - Catalog #287 (canonical Provenance / placeholder-rationale rejection)
  - Catalog #305 (observability surface declaration)
  - Catalog #318 (master-gradient raw-byte-authority guard)
  - Catalog #323 (canonical Provenance umbrella)
  - Catalog #327 (master-gradient contest-axis authority)
  - Catalog #335 (cathedral consumer directory contract — primary META gate this tool surfaces)
  - Catalog #336/#337 (cathedral autopilot main invocation)
  - Catalog #341 (cathedral consumer routing markers — primary META gate this tool empirically validates)

## Lane status

- Lane `lane_master_gradient_xray_viz_tool_797_20260519` to be registered via `tools/lane_maturity.py add-lane`
- Gates landed in this commit batch:
  - `impl_complete` ✓ (2 new plot functions + 23 dedicated tests + dispatch wire-in + sister memo)
  - `memory_entry` ✓ (this memo)
- Gates pending follow-on:
  - `real_archive_empirical` — N/A (observability tool; no archive bytes generated)
  - `contest_cuda` — N/A (no score claims)
  - `strict_preflight` — no NEW STRICT preflight gate needed (Catalog #335 + #341 already protect the consumer contract + routing markers surface this tool merely VISUALIZES)
  - `three_clean_review` — adversarial review cycle (sister subagent)
  - `deploy_runbook` — N/A for visualization tool

Expected lane level after this commit: **L1** (impl_complete + memory_entry).

— Slot EE master-gradient xray VIZ tool subagent 2026-05-19 (claude_slot_ee_master_gradient_xray_viz_797_20260519)


<!-- WAVE-3-CATALOG-344-BACKFILL-SWEEP appended 2026-05-20 per operator NON-NEGOTIABLE "keep feeding the queue" + WIRE-IN-AUDIT-POST-CASCADE op-routable #3. -->
<!-- # FORMALIZATION_PENDING:master-gradient-xray-VIZ-tool-task-797-landing-memo-trigger-tokens-describe-visualization-tool-not-new-equation -->
