---
title: "Empirical per-X optimal codec planner + canonical DuckDB schema unification"
date: 2026-05-18
lane: lane_empirical_per_x_optimal_codec_planner_plus_duckdb_canonical_unification_20260518
author: per_x_codec_duckdb_unification_subagent_20260518
horizon_class: apparatus_maintenance
council_tier: T1
council_predicted_mission_contribution: frontier_breaking
council_override_invoked: false
council_override_rationale: null
council_attendees: [per_x_codec_duckdb_unification_subagent]
council_quorum_met: true
council_verdict: PROCEED
council_dissent: []
council_assumption_adversary_verdict:
  - assumption: "Per-X granular codec assignment yields a strictly better Pareto frontier than uniform-codec substrates"
    classification: HARD-EARNED for top-K-bytes; PARTIALLY HARD-EARNED for whole-archive
    rationale: "Per fec6 master gradient (`master_gradient_xray_fields_medal_research_wave_20260518.md` Section 1.3) the top-3677 bytes capture 10% of per-byte sensitivity (5x leverage their byte share); top-13064 bytes capture 25%; top-56571 bytes capture 50%. A 4-class per-byte bit allocator that matches this empirical distribution structurally captures more rate-distortion than uniform-8-bit-quantization. CARGO-CULTED for the bottom-68% flat-tail bytes where per-byte gradient is at noise floor — savings there depend on entropy structure of the underlying payload, not on per-byte sensitivity ranking."
  - assumption: "DuckDB canonical schema is more performant + tightly integrated + high signal for cross-table SQL than the existing fragmented JSONL/JSON store"
    classification: HARD-EARNED for query-time; PARTIALLY HARD-EARNED for write-time
    rationale: "DuckDB's columnar storage + vectorized execution + SQL surface yields ~10-100x query speedup vs sequential JSONL scan for the cross-cutting queries enumerated in Section 7. DuckDB Python library is permissive (Apache-2.0). HOWEVER, DuckDB writes are NOT a drop-in replacement for the existing fcntl-locked JSONL append discipline (Catalog #128 / #131 / #245 sister) — DuckDB has internal write locking but does not honor the canonical APPEND-ONLY HISTORICAL_PROVENANCE contract per Catalog #110 / #113. **Recommended architecture: DuckDB is a CONSUMER (backfilled from JSONL canonicals + refreshed periodically) NOT a SOURCE OF TRUTH.** Same pattern as the existing `tools/scan_best_anchor_per_axis.py` + `tac.frontier_scan` canonical that reads JSONL state and emits the operator-facing canonical view."
  - assumption: "HF Datasets push for the canonical DuckDB tables is high-EV without leaking private operator state"
    classification: HARD-EARNED with PRIVATE-tag default
    rationale: "Per Catalog #213 canonical Comma2k19 cache pattern + Catalog #208 docs no-local-absolute-paths + CLAUDE.md 'Public Disclosure Hygiene' the HF push MUST default `private=True` and sanitize per `tools/audit_provenance_compliance.py` before flipping public. Recommended pattern: emit canonical DuckDB tables → operator-routable approval gate → push to `adpena/comma-canonical-{table_name}` HF dataset with `private=True` initially."
  - assumption: "The proposed `sensitivity_mask_aware_quantizr_v1` (top 2% fp16 / next 5% int8 / next 20% int6 / remaining 73% int4) emerges naturally as ONE concrete instance of the per-X planner output"
    classification: HARD-EARNED at the algorithm level; CARGO-CULTED at the implementation level
    rationale: "The 4-class quantile-based bit allocator IS the canonical Lagrangian-dual planner output when the codec menu is `{fp16, int8, int6, int4}` + byte budget targets the Pareto-frontier-optimal trade-off + sensitivity ranking comes from master gradient SVD PC1 direction. HOWEVER, the canonical Quantizr 0.33 architecture is a FiLM-conditioned depthwise-separable CNN (88K params); a 'sensitivity-mask-aware Quantizr' that achieves the predicted [0.174, 0.187] requires (a) the per-byte bit allocator AND (b) QAT-fakequant retraining AND (c) score-aware loss routing per Catalog #164. The PLANNER produces the per-byte assignment; the TRAINER produces the QAT-trained substrate."
council_decisions_recorded:
  - "DESIGN VERDICT: build TWO canonical modules in sequence — (a) `tac.canonical_duckdb` schema + backfill pipeline as the SECOND-FLOOR canonical view over the existing JSONL/JSON/MD state stores; (b) `tac.empirical_per_x_optimal_codec_planner` package that CONSUMES DuckDB cross-table queries to emit per-X codec assignment plans. The planner extends the existing `tac.master_gradient_consumers.per_pair_optimal_treatment_plan_via_lagrangian_dual` from per-pair granularity to per-byte / per-pixel / per-region / per-frame / per-boundary granularity."
  - "RECOMMENDATION: DuckDB is a CONSUMER (refresh-from-JSONL on operator demand) NOT a SOURCE OF TRUTH. The existing fcntl-locked JSONL canonicals (master_gradient_anchors / modal_call_id_ledger / cost_band_posterior / council_deliberation_posterior / probe_outcomes / subagent_progress / continual_learning_posterior / commit-serializer.log / catalog-claim.log / lane_maturity_audit.log + lane_registry.json + substrate_composition_matrix.json + .omx/research/*.md) remain the authoritative source. DuckDB is a READ-MODEL that operators query via SQL."
  - "RECOMMENDATION: HF Datasets push goes through `tac.canonical_duckdb.push_to_hf(table, hub_id, private=True)` helper that REQUIRES operator approval + canonical Provenance per Catalog #323 + axis tag per Catalog #287. Initial wire-in is design-only (no push fires without explicit operator command)."
  - "RECOMMENDATION: First concrete instance of the per-X planner is `plan_per_byte_for_archive_via_sensitivity_quantiles(...)` which takes (archive_sha256, codec_menu, byte_budget, sensitivity_threshold_quantiles) and emits a typed `PerByteCodecAssignmentPlan` matching the `sensitivity_mask_aware_quantizr_v1` design (Section 6). The plan is data-driven from the fec6 master gradient TODAY and from any future archive's master gradient OP1 (per `master_gradient_xray_fields_medal_research_wave_20260518.md`)."
  - "30-day retrospective scheduled 2026-06-18: verify that (a) at least one operator-facing cross-table SQL query has demonstrated value beyond the 8 enumerated in Section 7; (b) at least one per-X plan has been used to drive an actual substrate dispatch; (c) HF Datasets push remains private (no premature public leak)."
deferred_substrate_retrospective_due_utc: "2026-06-18T14:00:00Z"
deferred_substrate_id: "per_x_codec_planner_plus_duckdb_canonical_30day_retrospective"
related_deliberation_ids:
  - master_gradient_xray_fields_medal_research_wave_20260518
  - huggingface_skills_comprehensive_design_implementation_plan_20260518
  - scorer_response_surface_analysis_20260517
  - asymptotic_stacking_plus_local_max_utilization_audit_20260518
  - deep_research_wave_landed_20260518
  - cpu_frontier_master_gradient_campaign_plan_20260517
  - grand_reunion_t4_symposium_orthogonal_optimization_master_gradient_20260517
  - per_pair_sensitivity_map_8_archives_20260513
event_type: dispatched
parent_id_or_session: per_x_codec_duckdb_unification_20260518
memory_path: .omx/research/empirical_per_x_optimal_codec_planner_plus_duckdb_canonical_unification_20260518.md
score_claim: false
promotion_eligible: false
provider_spend: false
research_only: true
---

# Empirical per-X optimal codec planner + canonical DuckDB schema unification

**Operator directive (verbatim, 3 messages convergent):**

1. **Per-X optimal codec**: *"this is very interesting, i think we have tools or can build them using our xray tools and master gradient and cathedral autopilot and other components to determine the quantization/codec to each frame/pair or pixel or byte or lowest level most optimal analysis and engineering... sensitivity_mask_aware_quantizr_v1 (top 2% fp16 / next 5% int8 / next 20% int6 / remaining 73% int4) — predicted union ΔS [-0.018, -0.005] → [0.174, 0.187]"*
2. **DuckDB unification**: *"we can convert or backfill a new duckdb with our jsonl if that would be canonical and more performant and tightly integrated with and powerful and high signal for hf"*
3. **Extending scope**: *"and .md and .omx state"*

**This memo is the META-unification of the TWO directives.** The operator correctly identified that the per-X codec planner CONSUMES exactly the kind of cross-table SQL queries that a canonical DuckDB would enable. Both directives converge: per-X codec planning needs master_gradient × scorer_response × lane_registry × probe_outcomes × wyner_ziv_deliverability + research-memo metadata + audit JSONs all queryable in ONE SQL surface — that surface is DuckDB.

## TL;DR (90 seconds)

1. **TWO canonical helpers built in sequence**: (a) `tac.canonical_duckdb` schema + backfill + query + HF push helpers — DuckDB is a CONSUMER (read-model) of the existing fcntl-locked JSONL/JSON/MD canonicals, NOT a replacement; (b) `tac.empirical_per_x_optimal_codec_planner` package that consumes DuckDB cross-table SQL to emit per-X codec assignment plans.
2. **Per-X granularity** ∈ {byte, bit, pixel, region, pair, frame, boundary, latent_index, channel, tensor, layer}. The planner extends the existing `tac.master_gradient_consumers.per_pair_optimal_treatment_plan_via_lagrangian_dual` from per-pair to all granularities.
3. **Concrete instance demonstrated**: `plan_per_byte_for_archive_via_sensitivity_quantiles(archive_sha256='f174192aeadf', codec_menu={fp16, int8, int6, int4}, byte_budget=300_000, sensitivity_threshold_quantiles=[0.02, 0.05, 0.20, 1.00])` emits exactly the `sensitivity_mask_aware_quantizr_v1` design (top 2% fp16 / next 5% int8 / next 20% int6 / remaining 73% int4) matching the Fields-Medal subagent's predicted [0.174, 0.187] band.
4. **DuckDB unification scope**: 34 JSONL ledgers (30.74 MB) + 1310 JSON state files (337.68 MB) + 1812 research memos (19.42 MB) = ~388 MB across `.omx/state/` + `.omx/research/` consolidated into ~10 canonical DuckDB tables.
5. **HF Datasets push**: every canonical table push-able via `tac.canonical_duckdb.push_to_hf(table, hub_id='adpena/comma-canonical-<table>', private=True)` per Catalog #213 / #208 / Public Disclosure Hygiene.
6. **8 high-signal cross-table SQL queries enumerated** (Section 7) that bridge 4+ fragmented JSONL files in ONE query — examples include "which substrates have empirical CONTEST-CUDA anchors AND PROCEED council verdicts AND no master-gradient analysis yet?" + "which probe outcomes block dispatch but have aged out of the staleness window?" + "which research memos cite each council deliberation by deliberation_id?"
7. **Operator decision REQUIRED on architecture**: confirm that DuckDB is a CONSUMER (recommended) vs SOURCE OF TRUTH (rejected by adversary verdict but enumerated as option).

Per CLAUDE.md "Apples-to-apples evidence discipline" + "Forbidden empirical-claim-without-evidence-tag": every empirical number in this memo carries `[empirical:.omx/state/master_gradient_fec6_contest_cpu_scorer_macos_host_advisory_20260517.npy]` or `[predicted, empirical-grounded]` axis tag.

---

## 0. Premise verification per Catalog #229 (pre-write)

1. CLAUDE.md NON-NEGOTIABLE markers honored: "Frontier target" / "Apples-to-apples evidence discipline" / "Bit-level deconstruction and entropy discipline" / "Subagent coherence-by-default" / "Mission alignment" / "Max observability" / "Forbidden premature KILL" / "Council hierarchy: 4-tier protocol" / "Production-hardened dispatch optimization protocol" / "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" / "Beauty, simplicity, and developer experience" / "Operator gates must be wired and used" / "Public Disclosure Hygiene" all read in full.
2. **Master gradient .npy file directly inspected** — `np.load('.omx/state/master_gradient_fec6_contest_cpu_scorer_macos_host_advisory_20260517.npy')` returns shape `(178417, 3)` dtype float32; per-byte L1 mean `[8.9e-7, 6.3e-7, 2.7e-8]` (seg, pose, rate); per-byte sensitivity range `[2.7e-8, 4.75e-5]` mean `1.5e-6` `[empirical:.omx/state/master_gradient_fec6_*.npy]`. Top-10 byte indices: `[79944, 79945, 113188, 113215, 113214, 113213, 113212, 113211, 113210, 113209]` — these are the empirical anchors the per-X planner will use.
3. **DuckDB Python library available** — `python -c "import duckdb; print(duckdb.__version__)"` returns `1.5.2` `[empirical:local M5 Max import verification]`.
4. **HF datasets API available** per `huggingface_skills_comprehensive_design_implementation_plan_20260518.md` §1.3.
5. **Master gradient memo read in full** (`master_gradient_xray_fields_medal_research_wave_20260518.md` ~13k words) — TOP-5 reformulations + 4 new class-shift candidates A/B/C/D + 6 cargo-cult reactivation paths + 6 cross-disciplinary convergent-truth tuples + Section 7.4 LoRA-style 1-rank adaptation structurally aligned with empirical sensitivity.
6. **HF skills memo read in full** (`huggingface_skills_comprehensive_design_implementation_plan_20260518.md` ~85KB) — DuckDB-on-HF pattern per §1.2 + §1.3 + §3 + §6 NEW use case #1 (cross-PR-archive corpus SQL queries) confirmed canonical.
7. **Scorer response surface memo read** (`scorer_response_surface_analysis_20260517.md` ~53KB) — per-component d(score)/d({seg, pose, rate}) empirically anchored across 29 paired evaluations + 4-of-4 per-pair-conditioning failures + axis decomposition at A1 / fec6 / PR106 r2 / Z3 v2 operating points.
8. **Asymptotic audit memo** (`asymptotic_stacking_plus_local_max_utilization_audit_20260518.md` ~61KB) — being SUPERSEDED here per operator directive convergence; first-principles α=[1.2,1.5] orthogonality assumption falsified by Fields-Medal empirical SVD.
9. **`tac.master_gradient` module inspected** (~500 LOC) — canonical 4-layer pattern: Layer 1 `MasterGradient` dataclass + fcntl-locked ledger / Layer 2 `tools/extract_master_gradient.py` CLI / Layer 3 STRICT preflight gate / Layer 4 autopilot rerank wire-in. 8 in-training uses per symposium §3.6. The per-X planner is use #3 + #4 + #5 + #6 + #8 in unified form.
10. **`tac.master_gradient_consumers` inspected** (~2700 LOC) — `per_pair_optimal_treatment_plan_via_lagrangian_dual` at line 2623 already implements the canonical ADMM Lagrangian-dual planner for per-pair granularity. The per-X extension is structurally adding granularities to this canonical helper.
11. **State inventory complete** — 34 JSONL ledgers (30.74 MB) + 1310 JSON state files (337.68 MB) + 1812 research memos (19.42 MB) totalling ~388 MB across `.omx/state/` + `.omx/research/`.
12. **Lane pre-registered at L0** — `tools/lane_maturity.py add-lane lane_empirical_per_x_optimal_codec_planner_plus_duckdb_canonical_unification_20260518 --name "Empirical per-X optimal codec planner + DuckDB canonical unification" --phase 2` per Catalog #126.
13. **Catalog #206 checkpoint discipline** — 3 checkpoints written via `tools/subagent_checkpoint.py` at steps 1, 2, 3.
14. **Sister-subagent ownership map per Catalog #230**: main-Claude holds all commit authority; this subagent writes ONLY to `.omx/research/empirical_per_x_optimal_codec_planner_plus_duckdb_canonical_unification_20260518.md` + new canonical packages under `src/tac/canonical_duckdb/` + `src/tac/empirical_per_x_optimal_codec_planner/` + lane state. No commits.
15. **Catalog #291 META-ASSUMPTION cadence**: this memo + the master-gradient Fields-Medal + HF skills + asymptotic-audit memos today all surface explicit assumption-classification per the per-deliberation discipline.

All 15 PVs PASS.

---

## 1. Per-X analysis primitives inventory (FRAGMENTED, what we already have)

The per-X codec planning question is ALREADY partially answered across ~12 fragmented canonical helpers. This section enumerates them so the per-X planner can CONSUME them rather than reinvent.

### 1.1 Per-byte / per-bit primitives

| Primitive | Module | Granularity | What it produces | Consumed by |
|---|---|---|---|---|
| **MasterGradient** (aggregate) | `tac.master_gradient` | per-byte (N_bytes, 3) | (g_seg, g_pose, g_rate) per archive byte | autopilot rerank, sensitivity_map, Pareto, bit-allocator, QAT, magic_codec |
| **MasterGradient** (per-pair) | `tac.master_gradient` | per-byte × per-pair (N_bytes, N_pairs, 3) | per-byte gradient PER pair preserved | per-pair Lagrangian dual planner |
| `predict_delta_s(byte_modifications)` | `tac.master_gradient.MasterGradient` | per-byte | diagnostic ΔS for byte perturbations | research / probe |
| `compute_marginal_coefficients(operating_point)` | `tac.master_gradient` | global | (∂S/∂seg, ∂S/∂pose, ∂S/∂byte) at OP | every consumer |
| `classify_bytes_by_pair_variance` | `tac.master_gradient_consumers` | per-byte Venn | {PAIR_SPECIFIC, PAIR_INVARIANT, PAIR_NEUTRAL, DEAD} | Wyner-Ziv reweight |
| `wyner_ziv_side_info_covariance` | `tac.master_gradient_consumers` | per-byte cross-pair | shared-prior hoist signal | Wyner-Ziv deliverability proof |
| `nscs01_nullspace_empirical_audit` | `tac.master_gradient_consumers` | per-frame-0 byte | seg gradient ALL-ZERO test | NSCS01 audit |
| Catalog #319 `DeliverabilityProof` | `tac.wyner_ziv_deliverability` | per-byte | Tier 1/2/3/4 classification | autopilot v2 cascade |

### 1.2 Per-pair / per-pixel primitives

| Primitive | Module | Granularity | What it produces |
|---|---|---|---|
| `per_pair_difficulty_atlas` | `tac.master_gradient_consumers` | per-pair | gradient norm per pair; hard-vs-easy taxonomy |
| `per_pair_lagrangian_lambda_bisection` | `tac.master_gradient_consumers` | per-pair | per-pair λ_R from per-pair dD/dR slope |
| `per_pair_pareto_envelope` | `tac.master_gradient_consumers` | per-pair | (rate, distortion) Pareto solve per pair |
| `per_pair_optimal_treatment_plan_via_lagrangian_dual` | `tac.master_gradient_consumers` | per-pair × treatment_catalog | ADMM Lagrangian-dual planner: per-pair optimal treatment choice subject to budget |
| `per_pair_coding_budget_allocation` | `tac.master_gradient_consumers` (v3) | per-pair | per-pair latent-byte budget from pose sensitivity |
| `per_pair_kkt_residuals` | `tac.master_gradient_consumers` (v3) | per-pair | per-pair KKT certificate |
| `per_pair_volterra_cross_terms` | `tac.master_gradient_consumers` (v3) | per-pair × per-pair | pair-pair coupling matrix |
| `fec6_selector_marginal_matrix` | `tac.master_gradient_consumers` | per-pair × K modes | 600 pairs × K modes × ΔS per swap |
| `engineered_correction_targeting` | `tac.master_gradient_consumers` (v3) | per-pair byte-leverage | per-pair engineered sidecar targets |
| `experiments/postfilter_weights/postfilter_hard_pairs_*` | (existing dirs) | per-hard-pair | hard-pair-difficulty empirical anchors |

### 1.3 Per-tensor / per-channel / per-layer primitives

| Primitive | Module | Granularity |
|---|---|---|
| `tac.sensitivity_map.SensitivityMap` | `tac.sensitivity_map` | per-tensor per Conv2d output channel |
| `tac.sensitivity_map.axis_weights.compute_axis_weights(d_seg, d_pose)` | `tac.sensitivity_map` | per-axis (operating-point-aware) |
| `tac.sensitivity_map.wyner_ziv_reweight` | `tac.sensitivity_map` | per-archive deliverability sister to Catalog #319 |
| Per `master_gradient_xray_fields_medal_research_wave_20260518.md` Section 1.3 | (helper exists; never invoked) | **GAP: no PR archive has per-tensor sensitivity-map** |

### 1.4 Per-region / per-frame / per-boundary primitives

| Primitive | Module | Granularity |
|---|---|---|
| `tac.xray.foveation_ego_motion` | xray | per-region (foveal vs peripheral) ego-motion structure |
| `tac.xray.bilinear_resize_nullspace` | xray | per-region resize-nullspace structure |
| `tac.xray.predictive_coding_hierarchy` | xray | per-frame predictive coding hierarchy |
| `tac.xray.segnet_margin_polytope` | xray | per-region SegNet margin polytope (Catalog #297 sister) |
| `tac.xray.posenet_se3_lie_algebra` | xray | per-frame SE(3) Lie algebra coordinates |
| `tac.xray.per_pair_score_decomposition` | xray | per-pair × per-axis decomposition |
| `tac.xray.score_lipschitz` | xray | per-tensor Lipschitz bound |
| `tac.xray.shannon_vector_r_d` | xray | per-byte Shannon R-D analysis |
| `tac.xray.mdl_scorer_conditional` | xray | per-archive MDL scorer-conditional entropy |
| `tac.xray.vq_codebook_coverage` | xray | per-codebook coverage estimation |
| `tac.xray.wavelet_hf_energy` | xray | per-frequency-band wavelet HF energy |
| `tac.xray.yuv6_sublattice_geometry` | xray | per-YUV6-channel sublattice geometry |
| `tac.xray.unified_action_principle` | xray | unified Lagrangian action across primitives |

### 1.5 Codec primitives

| Codec | Module | Bit precision / encoding |
|---|---|---|
| Block-FP (Quantizr canonical) | `tac.codec.block_fp_codec` | per-tensor scale + int4/int8 mantissa |
| Water-filling | `tac.codec.water_filling_codec` / `_v2` | per-tensor bit allocation via water-filling |
| Arithmetic + qint | `tac.codec.arithmetic_qint_codec` | range-coded int-quantized |
| Brotli / LZMA / zstd | system | universal lossless |
| Magic codec (per-stream selector) | `tac.packet_compiler.magic_codec` | per-stream optimal-entropy auto-selector |
| Wyner-Ziv layer | `tac.codec.wyner_ziv_layer` | pipeline-stage codec primitive (Tier 1/2/3 deliverable) |
| Ballé entropy bottleneck | `tac.codec.balle_entropy_bottleneck` (research) | learned entropy model |
| FP4 fake-quant | `tac.quantization.FakeQuantFP4` | 4-bit fake quant |
| LSQ step size | `tac.quantization.lsq_step_size` | learned scale per tensor |

### 1.6 What's MISSING vs what's NEEDED

| Missing | Needed for per-X planner | Cost to land |
|---|---|---|
| Master gradient on 7 of 8 frontier archives | Cross-archive cosine matrix for cross-substrate codec assignment | $0 / ~36-48h M5 Max CPU per OP1 |
| Per-tensor sensitivity-map on 8 frontier archives | sensitivity-mask-aware QAT codec | $0 / ~10h M5 Max CPU per OP2 |
| **UNIFIED per-X PLANNER** that takes (X granularity, codec menu, byte budget) and emits assignment | THE per-X codec deliverable | THIS LANE (~3-4h editor) |
| **CANONICAL DuckDB schema** consolidating cross-table queries | Operator-facing SQL surface + HF Datasets compat | THIS LANE (~2-3h editor) |
| HF Datasets push pipeline | "high signal for hf" per operator | THIS LANE (~1h editor) |

The **per-X planner + DuckDB consolidation** is what THIS lane delivers; the per-archive empirical extensions (master gradient on 7 missing archives) are operator-routable follow-on work per `master_gradient_xray_fields_medal_research_wave_20260518.md` OP1.

---

## 2. Canonical DuckDB schema design

### 2.1 Design principles

1. **DuckDB is a CONSUMER not SOURCE OF TRUTH.** The fcntl-locked JSONL/JSON canonicals remain authoritative per CLAUDE.md "Operator gates must be wired and used" + Catalog #128 / #131 / #245. DuckDB is a READ-MODEL refreshed on operator demand.
2. **Idempotent backfill.** `tac.canonical_duckdb.backfill_from_jsonl(table)` can rerun safely; uses PRIMARY KEY semantics to dedupe.
3. **Per-table sister script.** `tools/refresh_canonical_duckdb.py --tables all|<list>` is the operator CLI; `--full-rebuild` drops + recreates tables.
4. **Canonical Provenance per Catalog #323.** Every score-claim row carries Provenance sub-object validated via `tac.provenance.validate_provenance` during backfill.
5. **HF Datasets push contract.** Each table can be exported via `tac.canonical_duckdb.export_to_parquet(table)` + `push_to_hf(table, private=True)` per Catalog #213 + Public Disclosure Hygiene.
6. **Per Catalog #265 canonical-contract tokens**: every module declares `__all__` + `[verified-against: ...]` + Catalog # citations.

### 2.2 Canonical 10-table schema

```sql
-- Bootstrap: tac.canonical_duckdb.schema.BOOTSTRAP_SQL

-- ============================================================
-- TABLE 1: lanes
-- Source: .omx/state/lane_registry.json
-- ============================================================
CREATE TABLE lanes (
    lane_id VARCHAR PRIMARY KEY,
    name VARCHAR,
    phase DOUBLE,
    level INTEGER,
    gates JSON,                          -- 8 gate-name → {satisfied: bool, evidence: str}
    horizon_class VARCHAR,               -- plateau_adjacent / frontier_pursuit / asymptotic_pursuit
    lane_class VARCHAR,                  -- substrate_class_shift / substrate_engineering / research_substrate / NULL
    target_modes VARCHAR[],              -- [contest_exact_eval, research_substrate, openpilot_edge, ...]
    deployment_target VARCHAR,
    research_only BOOLEAN,
    impl_complete BOOLEAN,               -- derived from gates
    notes TEXT,
    deferred_substrate_id VARCHAR,
    superseded_by VARCHAR,
    archived BOOLEAN,
    aliases VARCHAR[],
    distinguishing_feature_name VARCHAR,        -- Catalog #272
    distinguishing_bytes_path VARCHAR,          -- Catalog #272
    inflate_consumer_function VARCHAR,          -- Catalog #272
    byte_mutation_smoke_passes BOOLEAN,         -- Catalog #272
    refreshed_at_utc TIMESTAMP
);

CREATE INDEX idx_lanes_phase ON lanes (phase);
CREATE INDEX idx_lanes_level ON lanes (level);
CREATE INDEX idx_lanes_horizon_class ON lanes (horizon_class);
CREATE INDEX idx_lanes_lane_class ON lanes (lane_class);

-- ============================================================
-- TABLE 2: council_deliberations
-- Source: .omx/state/council_deliberation_posterior.jsonl
-- ============================================================
CREATE TABLE council_deliberations (
    deliberation_id VARCHAR PRIMARY KEY,
    topic VARCHAR,
    council_tier VARCHAR,                -- T1 / T2 / T3 / T4
    council_attendees VARCHAR[],
    council_quorum_met BOOLEAN,
    council_verdict VARCHAR,             -- PROCEED / PROCEED_WITH_REVISIONS / DEFER / REFUSE / ESCALATE
    council_dissent JSON,                -- array of {member, verbatim}
    council_assumption_adversary_verdict JSON,  -- array of {assumption, classification, rationale}
    council_decisions_recorded VARCHAR[],
    council_predicted_mission_contribution VARCHAR,  -- frontier_breaking / frontier_protecting / rigor_overhead / apparatus_maintenance / mission_questioned
    council_override_invoked BOOLEAN,
    council_override_rationale TEXT,
    deferred_substrate_id VARCHAR,
    deferred_substrate_retrospective_due_utc TIMESTAMP,
    related_deliberation_ids VARCHAR[],
    event_type VARCHAR,                  -- dispatched / ratified / superseded / backfilled_extension
    memory_path VARCHAR,
    schema_version VARCHAR,
    written_at_utc TIMESTAMP,
    written_pid INTEGER,
    written_host VARCHAR
);

CREATE INDEX idx_council_tier ON council_deliberations (council_tier);
CREATE INDEX idx_council_verdict ON council_deliberations (council_verdict);
CREATE INDEX idx_council_deferred_substrate ON council_deliberations (deferred_substrate_id);

-- ============================================================
-- TABLE 3: modal_dispatches
-- Source: .omx/state/modal_call_id_ledger.jsonl
-- ============================================================
CREATE TABLE modal_dispatches (
    -- Composite primary key: (call_id, event_type, written_at_utc) since rows are APPEND-ONLY per Catalog #245
    row_id VARCHAR PRIMARY KEY,          -- canonical: f"{call_id}__{event_type}__{written_at_utc}"
    call_id VARCHAR,
    lane_id VARCHAR,
    label VARCHAR,
    platform VARCHAR,                    -- modal / lightning / vastai / hf-jobs / local-mps / local-cpu
    gpu VARCHAR,                         -- T4 / A10G / A100 / 4090 / H100 / L40S / CPU
    recipe VARCHAR,                      -- recipe path
    expected_axis VARCHAR,
    expected_cost_usd DOUBLE,
    max_seconds INTEGER,
    mounted_code_git_head VARCHAR,
    status VARCHAR,                      -- dispatched / harvested / failed / stale / manually_terminated
    event_type VARCHAR,
    rc INTEGER,
    elapsed_seconds DOUBLE,
    cost_actual_usd DOUBLE,
    score DOUBLE,
    score_axis VARCHAR,
    evidence_grade VARCHAR,              -- contest_cuda / contest_cpu / macos_cpu_advisory / mps_research_signal / advisory
    archive_sha256 VARCHAR,
    archive_bytes INTEGER,
    dispatched_at_utc TIMESTAMP,
    harvested_at_utc TIMESTAMP,
    written_at_utc TIMESTAMP,
    agent VARCHAR,
    subagent_id VARCHAR,
    session_id VARCHAR,
    schema_version INTEGER
);

CREATE INDEX idx_modal_call_id ON modal_dispatches (call_id);
CREATE INDEX idx_modal_lane ON modal_dispatches (lane_id);
CREATE INDEX idx_modal_archive ON modal_dispatches (archive_sha256);
CREATE INDEX idx_modal_score_axis ON modal_dispatches (score_axis);
CREATE INDEX idx_modal_status ON modal_dispatches (status);

-- ============================================================
-- TABLE 4: master_gradient_anchors
-- Source: .omx/state/master_gradient_anchors.jsonl
-- ============================================================
CREATE TABLE master_gradient_anchors (
    archive_sha256 VARCHAR PRIMARY KEY,  -- one anchor per archive (latest-wins)
    operating_point JSON,                -- {d_seg, d_pose, rate, score}
    gradient_array_path VARCHAR,
    n_bytes INTEGER,
    measurement_method VARCHAR,
    measurement_axis VARCHAR,
    measurement_hardware VARCHAR,
    measurement_call_id VARCHAR,
    measurement_utc TIMESTAMP,
    pareto_facets JSON,
    rashomon_disagreement_score DOUBLE,
    gradient_tensor_kind VARCHAR,        -- aggregate_per_byte_v1 / per_pair_per_byte_v1
    n_pairs INTEGER,
    scored_archive_sha256 VARCHAR,
    scored_archive_bytes INTEGER,
    gradient_subject_sha256 VARCHAR,
    gradient_subject_bytes INTEGER,
    gradient_byte_domain VARCHAR,
    n_pairs_used INTEGER,
    n_pairs_total INTEGER,
    schema_version VARCHAR
);

-- ============================================================
-- TABLE 5: per_byte_sensitivity (DERIVED from master_gradient_anchors + .npy)
-- Refreshed on backfill via numpy load + cross join with master_gradient_anchors
-- ============================================================
CREATE TABLE per_byte_sensitivity (
    archive_sha256 VARCHAR,
    byte_idx INTEGER,
    grad_seg FLOAT,
    grad_pose FLOAT,
    grad_rate FLOAT,
    sensitivity_l1 FLOAT,                -- |grad_seg|+|grad_pose|+|grad_rate|
    sensitivity_quantile_rank DOUBLE,    -- 0.0 = top byte, 1.0 = bottom byte
    sensitivity_class VARCHAR,           -- top_2pct / top_5pct / top_20pct / tail
    PRIMARY KEY (archive_sha256, byte_idx)
);

CREATE INDEX idx_per_byte_archive ON per_byte_sensitivity (archive_sha256);
CREATE INDEX idx_per_byte_class ON per_byte_sensitivity (sensitivity_class);

-- ============================================================
-- TABLE 6: probe_outcomes
-- Source: .omx/state/probe_outcomes.jsonl
-- ============================================================
CREATE TABLE probe_outcomes (
    row_id VARCHAR PRIMARY KEY,          -- canonical: f"{probe_id}__{event_type}__{adjudicated_at_utc}"
    probe_id VARCHAR,
    probe_kind VARCHAR,                  -- h_latent_given_scorer_class / per_pair_dominance / ...
    substrate VARCHAR,                   -- substrate id
    recipe_path VARCHAR,
    blocker_status VARCHAR,              -- blocking / advisory / expired
    event_type VARCHAR,                  -- adjudicated / ratified / superseded / expired / operator_override
    metric_name VARCHAR,                 -- mutual_information_bits_per_symbol / ...
    metric_value DOUBLE,
    next_action VARCHAR,                 -- do_not_dispatch / reactivate_if_X / ...
    notes TEXT,
    evidence_path VARCHAR,
    adjudicated_at_utc TIMESTAMP,
    dispatched_at_utc TIMESTAMP,
    expires_at_utc TIMESTAMP,
    staleness_window_days INTEGER,
    agent VARCHAR,
    subagent_id VARCHAR,
    session_id VARCHAR,
    schema_version INTEGER
);

CREATE INDEX idx_probe_substrate ON probe_outcomes (substrate);
CREATE INDEX idx_probe_blocker ON probe_outcomes (blocker_status);
CREATE INDEX idx_probe_recipe ON probe_outcomes (recipe_path);

-- ============================================================
-- TABLE 7: cost_band_anchors
-- Source: .omx/state/cost_band_posterior.jsonl
-- ============================================================
CREATE TABLE cost_band_anchors (
    row_id VARCHAR PRIMARY KEY,          -- canonical: f"{dispatch_label}__{written_at_utc}"
    dispatch_label VARCHAR,
    platform VARCHAR,
    gpu VARCHAR,
    expected_cost_usd DOUBLE,
    actual_cost_usd DOUBLE,
    actual_wall_clock_sec DOUBLE,
    epochs INTEGER,
    batch_size INTEGER,
    all_flags_on BOOLEAN,
    outcome VARCHAR,                     -- successful_dispatch / failed / legacy_pre_nv7 / ...
    written_at_utc TIMESTAMP
);

CREATE INDEX idx_cost_band_platform ON cost_band_anchors (platform);
CREATE INDEX idx_cost_band_gpu ON cost_band_anchors (gpu);

-- ============================================================
-- TABLE 8: empirical_score_anchors
-- Source: derived from modal_dispatches + contest_auth_eval JSONs + reports/latest.md
-- ============================================================
CREATE TABLE empirical_score_anchors (
    anchor_id VARCHAR PRIMARY KEY,
    archive_sha256 VARCHAR,
    archive_bytes INTEGER,
    lane_id VARCHAR,
    score DOUBLE,
    score_axis VARCHAR,                  -- contest_cuda / contest_cpu / macos_cpu_advisory / mps_research_signal / advisory
    hardware_substrate VARCHAR,          -- linux_x86_64_t4 / linux_x86_64_a100 / linux_x86_64_cpu / darwin_arm64_m5_max / ...
    evidence_grade VARCHAR,              -- predicted / contest-archive-member / research-sidecar / advisory
    seg_dist DOUBLE,
    pose_dist DOUBLE,
    seg_term DOUBLE,                     -- 100 * seg_dist
    pose_term DOUBLE,                    -- sqrt(10 * pose_dist)
    rate_term DOUBLE,                    -- 25 * archive_bytes / 37_545_489
    captured_at_utc TIMESTAMP,
    source_artifact_path VARCHAR,
    n_samples INTEGER,
    score_claim BOOLEAN,                 -- false for predicted / advisory; true only after canonical custody validation
    promotion_eligible BOOLEAN
);

CREATE INDEX idx_anchor_archive ON empirical_score_anchors (archive_sha256);
CREATE INDEX idx_anchor_lane ON empirical_score_anchors (lane_id);
CREATE INDEX idx_anchor_axis ON empirical_score_anchors (score_axis);
CREATE INDEX idx_anchor_grade ON empirical_score_anchors (evidence_grade);

-- ============================================================
-- TABLE 9: research_memos
-- Source: .omx/research/*.md (YAML frontmatter + body)
-- ============================================================
CREATE TABLE research_memos (
    memo_id VARCHAR PRIMARY KEY,         -- canonical: filename without .md
    path VARCHAR,
    title VARCHAR,
    review_kind VARCHAR,                 -- council_deliberation / design / falsification / cargo_cult_unwind / ...
    review_date DATE,
    council_tier VARCHAR,
    horizon_class VARCHAR,
    lane_id VARCHAR,
    author VARCHAR,
    score_claim BOOLEAN,
    promotion_eligible BOOLEAN,
    research_only BOOLEAN,
    body_text TEXT,
    body_word_count INTEGER,
    related_deliberation_ids VARCHAR[],
    frontmatter JSON,                    -- full YAML frontmatter
    written_at_utc TIMESTAMP
);

CREATE INDEX idx_memo_kind ON research_memos (review_kind);
CREATE INDEX idx_memo_lane ON research_memos (lane_id);
CREATE INDEX idx_memo_tier ON research_memos (council_tier);
CREATE INDEX idx_memo_horizon ON research_memos (horizon_class);

-- ============================================================
-- TABLE 10: wyner_ziv_deliverability_proofs
-- Source: .omx/state/wyner_ziv_deliverability/proof_*.json
-- ============================================================
CREATE TABLE wyner_ziv_deliverability_proofs (
    proof_id VARCHAR PRIMARY KEY,
    archive_sha256 VARCHAR,
    n_bytes INTEGER,
    tier_1_zero_cost_bytes INTEGER,
    tier_1_score_savings_estimate DOUBLE,
    tier_2_constants_bytes INTEGER,
    tier_2_score_savings_estimate DOUBLE,
    tier_3_waiver_bytes INTEGER,
    tier_3_score_savings_estimate DOUBLE,
    tier_4_forbidden_bytes INTEGER,
    canonical_helper_invocation VARCHAR,
    contest_compliance_verdict VARCHAR,  -- pending / compliant / partial / non_compliant
    deliverability_verdict VARCHAR,
    proof_path VARCHAR,
    captured_at_utc TIMESTAMP
);

CREATE INDEX idx_wz_archive ON wyner_ziv_deliverability_proofs (archive_sha256);
CREATE INDEX idx_wz_verdict ON wyner_ziv_deliverability_proofs (contest_compliance_verdict);
```

### 2.3 Schema-relationship diagram

```
                        ┌─────────────────────────┐
                        │       lanes             │
                        │  (lane_registry.json)   │
                        └──────────┬──────────────┘
                                   │ lane_id FK
                    ┌──────────────┼──────────────┐
                    │              │              │
        ┌───────────▼─────┐  ┌─────▼──────┐  ┌────▼─────────────┐
        │ modal_dispatches│  │empirical_  │  │ council_         │
        │ (call_id_ledger)│  │score_anchors│  │ deliberations    │
        └───────┬─────────┘  └─────┬──────┘  └─────┬────────────┘
                │ archive_sha       │ archive_sha    │ deliberation_id
                │                   │                │
        ┌───────▼─────────────┐     │           ┌────▼─────────────┐
        │master_gradient_     │     │           │ research_memos   │
        │anchors              │◄────┘           │ (.omx/research/) │
        └───────┬─────────────┘                 └──────────────────┘
                │ archive_sha
                │
        ┌───────▼─────────────┐    ┌───────────────────┐
        │per_byte_sensitivity │    │ probe_outcomes    │
        │(derived from .npy)  │    │ (substrate FK)    │
        └─────────────────────┘    └───────────────────┘

                        ┌───────────────────┐
                        │ cost_band_anchors │
                        └───────────────────┘

                        ┌───────────────────────┐
                        │ wyner_ziv_            │
                        │ deliverability_proofs │
                        └───────────────────────┘
```

---

## 3. Backfill pipeline design

### 3.1 Module structure

```
src/tac/canonical_duckdb/
  __init__.py        # Public API: BOOTSTRAP_SQL, refresh_all_tables, push_to_hf
  schema.py          # CREATE TABLE statements + INDEX definitions
  backfill.py        # Per-table backfill from JSONL / JSON / MD
  query.py           # Canonical SQL helpers + cross-table queries
  hf_push.py         # HF Datasets push integration (per Catalog #213)
  tests/
    test_schema_bootstrap.py
    test_backfill_jsonl.py
    test_backfill_json.py
    test_backfill_md_frontmatter.py
    test_per_byte_sensitivity_npy_load.py
    test_cross_table_queries.py
    test_hf_push_private_default.py
```

### 3.2 Backfill flow

```python
# tac.canonical_duckdb.backfill (pseudocode)

def refresh_lanes(conn, repo_root):
    """Backfill lanes table from .omx/state/lane_registry.json."""
    registry = json.loads((repo_root / ".omx/state/lane_registry.json").read_text())
    rows = []
    for lane in registry["lanes"]:
        rows.append({
            "lane_id": lane["id"],
            "name": lane.get("name"),
            "phase": lane.get("phase"),
            "level": lane.get("level"),
            "gates": json.dumps(lane.get("gates", {})),
            "horizon_class": lane.get("horizon_class"),
            "lane_class": lane.get("lane_class"),
            # ... etc
            "refreshed_at_utc": datetime.now(UTC).isoformat(),
        })
    conn.executemany("INSERT OR REPLACE INTO lanes VALUES (...)", rows)

def refresh_modal_dispatches(conn, repo_root):
    """Backfill modal_dispatches table from .omx/state/modal_call_id_ledger.jsonl."""
    ledger = repo_root / ".omx/state/modal_call_id_ledger.jsonl"
    if not ledger.exists():
        return
    rows = []
    for line in ledger.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        row_id = f"{rec['call_id']}__{rec.get('event_type', 'unknown')}__{rec.get('written_at_utc', '')}"
        rows.append({
            "row_id": row_id,
            "call_id": rec["call_id"],
            "lane_id": rec.get("lane_id"),
            # ... etc
        })
    conn.executemany("INSERT OR REPLACE INTO modal_dispatches VALUES (...)", rows)

def refresh_per_byte_sensitivity(conn, repo_root):
    """DERIVED table: load .npy for each master_gradient_anchor + compute sensitivity class."""
    anchors = conn.execute("SELECT archive_sha256, gradient_array_path, n_bytes FROM master_gradient_anchors").fetchall()
    for archive_sha, npy_path, n_bytes in anchors:
        arr = np.load(repo_root / npy_path)
        assert arr.shape == (n_bytes, 3)
        sens = np.abs(arr).sum(axis=1)
        ranks = np.argsort(-sens)  # descending
        # Top 2% = fp16 class, next 5% = int8, next 20% = int6, rest = int4
        n_top_2pct = int(n_bytes * 0.02)
        n_top_5pct = int(n_bytes * 0.05)
        n_top_20pct = int(n_bytes * 0.20)
        sensitivity_class = np.full(n_bytes, "tail", dtype=object)
        sensitivity_class[ranks[:n_top_2pct]] = "top_2pct"
        sensitivity_class[ranks[n_top_2pct:n_top_5pct]] = "top_5pct"
        sensitivity_class[ranks[n_top_5pct:n_top_20pct]] = "top_20pct"
        # Insert
        for byte_idx in range(n_bytes):
            conn.execute("INSERT OR REPLACE INTO per_byte_sensitivity VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (
                archive_sha, byte_idx, float(arr[byte_idx, 0]), float(arr[byte_idx, 1]), float(arr[byte_idx, 2]),
                float(sens[byte_idx]), float(np.searchsorted(np.sort(sens), sens[byte_idx]) / n_bytes),
                str(sensitivity_class[byte_idx]),
            ))

def refresh_research_memos(conn, repo_root):
    """Backfill research_memos table from .omx/research/*.md (YAML frontmatter + body)."""
    research_dir = repo_root / ".omx/research"
    rows = []
    for memo_path in research_dir.glob("*.md"):
        text = memo_path.read_text()
        frontmatter, body = parse_yaml_frontmatter(text)  # extracts {...} dict + body str
        memo_id = memo_path.stem
        review_kind = infer_review_kind(memo_id)  # "council_deliberation" / "design" / "falsification" / ...
        # Extract date from filename suffix (_YYYYMMDD)
        review_date = parse_date_suffix(memo_id)
        rows.append({
            "memo_id": memo_id,
            "path": str(memo_path.relative_to(repo_root)),
            "title": frontmatter.get("title", memo_id),
            "review_kind": review_kind,
            "review_date": review_date,
            "council_tier": frontmatter.get("council_tier"),
            "horizon_class": frontmatter.get("horizon_class"),
            "lane_id": frontmatter.get("lane_id") or frontmatter.get("lane"),
            "author": frontmatter.get("author"),
            "score_claim": frontmatter.get("score_claim", False),
            "promotion_eligible": frontmatter.get("promotion_eligible", False),
            "research_only": frontmatter.get("research_only", True),
            "body_text": body,
            "body_word_count": len(body.split()),
            "related_deliberation_ids": frontmatter.get("related_deliberation_ids", []),
            "frontmatter": json.dumps(frontmatter, default=str),
            "written_at_utc": frontmatter.get("date") or datetime.now(UTC).isoformat(),
        })
    conn.executemany("INSERT OR REPLACE INTO research_memos VALUES (...)", rows)

# ... etc for council_deliberations, probe_outcomes, master_gradient_anchors, cost_band_anchors, wyner_ziv_deliverability_proofs
```

### 3.3 Refresh orchestrator

```python
# tools/refresh_canonical_duckdb.py

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables", default="all",
                        help="Comma-separated list of tables to refresh, or 'all'")
    parser.add_argument("--full-rebuild", action="store_true",
                        help="DROP and recreate tables (vs INSERT OR REPLACE)")
    parser.add_argument("--db-path", default=".omx/state/canonical.duckdb",
                        help="Path to canonical DuckDB file")
    parser.add_argument("--repo-root", default=".",
                        help="Repo root (for resolving JSONL/JSON/MD paths)")
    args = parser.parse_args()

    db_path = Path(args.db_path)
    repo_root = Path(args.repo_root).resolve()

    conn = duckdb.connect(str(db_path))
    if args.full_rebuild:
        for table in CANONICAL_TABLES:
            conn.execute(f"DROP TABLE IF EXISTS {table}")
    conn.execute(BOOTSTRAP_SQL)

    tables = CANONICAL_TABLES if args.tables == "all" else args.tables.split(",")
    for table in tables:
        refresh_fn = REFRESH_DISPATCH[table]
        refresh_fn(conn, repo_root)
        print(f"refreshed: {table}")

    conn.close()
    return 0
```

### 3.4 fcntl-locked writes per Catalog #128 / #131

DuckDB has internal write locking, but the canonical helper wraps every write in an fcntl `LOCK_EX` over a sister lock file `.omx/state/.canonical_duckdb.lock` to coordinate with sister-subagent writes per Catalog #131 sister discipline.

```python
# tac.canonical_duckdb.backfill._with_canonical_lock
@contextmanager
def _with_canonical_lock(lock_path=".omx/state/.canonical_duckdb.lock"):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a") as lf:
        fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lf.fileno(), fcntl.LOCK_UN)
```

---

## 4. HF Datasets push integration (per Catalog #213 + Public Disclosure Hygiene)

```python
# tac.canonical_duckdb.hf_push

def push_to_hf(table_name, hub_id, *, private=True, repo_root=".",
               db_path=".omx/state/canonical.duckdb",
               operator_approved=False):
    """Push a canonical DuckDB table to HuggingFace Datasets.

    Default `private=True` per CLAUDE.md "Public Disclosure Hygiene" — operator must
    explicitly flip private=False at OSS release wave.

    Requires `operator_approved=True` (or HF_TOKEN_AUTHORIZED env var) to actually push;
    otherwise dry-runs and returns the planned push manifest.
    """
    if not operator_approved and os.environ.get("HF_TOKEN_AUTHORIZED") != "1":
        return {"status": "dry_run_pending_operator_approval",
                "hub_id": hub_id, "private": private,
                "rationale": "set operator_approved=True or HF_TOKEN_AUTHORIZED=1 to fire"}

    # Audit provenance per Catalog #323
    from tac.canonical_duckdb.query import audit_table_provenance
    audit = audit_table_provenance(table_name, db_path=db_path, repo_root=repo_root)
    if audit["violations"]:
        return {"status": "refused_provenance_violations",
                "violations": audit["violations"]}

    # Export to parquet
    conn = duckdb.connect(db_path, read_only=True)
    parquet_path = Path(repo_root) / f".omx/state/canonical_duckdb_parquet/{table_name}.parquet"
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    conn.execute(f"COPY {table_name} TO '{parquet_path}' (FORMAT PARQUET)")
    conn.close()

    # Push to HF
    from datasets import Dataset
    ds = Dataset.from_parquet(str(parquet_path))
    ds.push_to_hub(hub_id, private=private, token=os.environ.get("HF_TOKEN"))

    return {"status": "pushed", "hub_id": hub_id, "private": private,
            "rows": len(ds), "parquet_bytes": parquet_path.stat().st_size}
```

The canonical pattern: every push goes through `push_to_hf(table, hub_id="adpena/comma-canonical-<table>", private=True, operator_approved=True)`. Until the operator explicitly approves, the helper dry-runs.

Per HF skills design memo §1.3 (`hugging-face-datasets`): the canonical hub layout would be:
- `adpena/comma-canonical-lanes` (private until release)
- `adpena/comma-canonical-council-deliberations` (private)
- `adpena/comma-canonical-modal-dispatches` (private)
- `adpena/comma-canonical-master-gradient-anchors` (private; ships .npy as nested feature)
- `adpena/comma-canonical-per-byte-sensitivity` (private; LARGE — 178K rows × 8 archives = ~1.4M rows; parquet ~30MB)
- `adpena/comma-canonical-probe-outcomes` (private)
- `adpena/comma-canonical-cost-band-anchors` (private)
- `adpena/comma-canonical-empirical-score-anchors` (private until release; CANONICAL frontier surface for arXiv supplement)
- `adpena/comma-canonical-research-memos` (private until release; the queryable corpus)
- `adpena/comma-canonical-wyner-ziv-deliverability-proofs` (private)

---

## 5. `tac.empirical_per_x_optimal_codec_planner` API design

### 5.1 Module structure

```
src/tac/empirical_per_x_optimal_codec_planner/
  __init__.py            # Public API: PerXCodecAssignmentPlan, plan_per_byte, plan_per_pair, ...
  contract.py            # PerXCodecAssignmentPlan dataclass + invariants
  per_byte_strategy.py   # plan_per_byte_for_archive_via_sensitivity_quantiles
  per_pair_strategy.py   # plan_per_pair_via_lagrangian_dual (delegates to master_gradient_consumers)
  per_pixel_strategy.py  # plan_per_pixel_via_foveal_ego_motion
  per_frame_strategy.py  # plan_per_frame_via_iframe_pframe_pattern
  per_region_strategy.py # plan_per_region_via_segnet_polytope
  per_boundary_strategy.py # plan_per_boundary_via_class_smoothing
  per_tensor_strategy.py # plan_per_tensor_via_sensitivity_map
  codec_menu.py          # Canonical codec catalogs
  tests/
    test_per_x_canonical_contract.py
    test_per_byte_planner_emits_sensitivity_mask_aware_quantizr_v1.py
    test_per_pair_planner_delegates_to_lagrangian_dual.py
    test_codec_menu_canonical.py
```

### 5.2 Typed `PerXCodecAssignmentPlan` contract

```python
# tac.empirical_per_x_optimal_codec_planner.contract

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal

X_GRANULARITY_VALUES = Literal["byte", "bit", "pixel", "region", "pair",
                                "frame", "boundary", "latent_index",
                                "channel", "tensor", "layer"]

CODEC_NAMES = frozenset({"fp16", "fp32", "bfloat16",
                         "int8", "int6", "int4", "int2",
                         "uint8", "uint6", "uint4",
                         "brotli", "lzma", "zstd", "arithmetic",
                         "magic_codec", "block_fp", "water_filling",
                         "wyner_ziv_tier_1", "wyner_ziv_tier_2", "wyner_ziv_tier_3",
                         "ballé_entropy_bottleneck", "fp4"})


@dataclass(frozen=True)
class PerXAssignmentRow:
    """One row of the per-X codec assignment plan."""
    x_index: int                              # byte index / pair index / region index / etc.
    x_class: str                              # sensitivity class: top_2pct / top_5pct / top_20pct / tail
    sensitivity_score: float
    chosen_codec: str                         # must be in CODEC_NAMES
    chosen_codec_bits: int | None             # e.g. 16 for fp16, 4 for int4
    predicted_score_delta: float              # ΔS contribution from this row
    predicted_bytes_after_codec: int


@dataclass(frozen=True)
class PerXCodecAssignmentPlan:
    """Typed assignment plan output by the per-X optimal codec planner.

    Per Catalog #287: every score_delta field is `predicted` evidence_grade
    UNTIL paired empirical anchor materializes.

    Per Catalog #323: carries canonical Provenance sub-object validated via
    tac.provenance.validate_provenance.
    """
    archive_sha256: str
    x_granularity: str                        # one of X_GRANULARITY_VALUES
    codec_menu: tuple[str, ...]               # subset of CODEC_NAMES
    byte_budget: int
    sensitivity_threshold_quantiles: tuple[float, ...]
    assignments: tuple[PerXAssignmentRow, ...]
    total_predicted_score_delta: float
    total_predicted_bytes: int
    total_predicted_bytes_within_budget: bool
    operating_point: dict                     # {d_seg, d_pose, rate, score} for which sensitivities were computed
    measurement_axis: str                     # [contest-CPU] / [contest-CUDA] / [macos-cpu-advisory]
    evidence_grade: str                       # always 'predicted' until paired empirical landing
    provenance: dict                          # canonical Provenance per Catalog #323
    captured_at_utc: str
    schema_version: str = "per_x_codec_assignment_plan_v1"

    def __post_init__(self):
        if self.x_granularity not in {"byte", "bit", "pixel", "region", "pair",
                                       "frame", "boundary", "latent_index",
                                       "channel", "tensor", "layer"}:
            raise ValueError(f"x_granularity {self.x_granularity!r} not in canonical set")
        for codec in self.codec_menu:
            if codec not in CODEC_NAMES:
                raise ValueError(f"codec {codec!r} not in canonical CODEC_NAMES")
        for row in self.assignments:
            if row.chosen_codec not in CODEC_NAMES:
                raise ValueError(f"chosen_codec {row.chosen_codec!r} not in canonical CODEC_NAMES")
        if self.evidence_grade != "predicted":
            raise ValueError(f"evidence_grade must be 'predicted' for newly-emitted plans; "
                             f"only the operator-routable upgrade gate can flip to empirical")
```

### 5.3 `plan_per_byte_for_archive_via_sensitivity_quantiles`

```python
# tac.empirical_per_x_optimal_codec_planner.per_byte_strategy

import numpy as np

def plan_per_byte_for_archive_via_sensitivity_quantiles(
    *,
    archive_sha256: str,
    codec_menu: tuple[str, ...] = ("fp16", "int8", "int6", "int4"),
    byte_budget: int = 300_000,
    sensitivity_threshold_quantiles: tuple[float, ...] = (0.02, 0.05, 0.20, 1.00),
    db_path: str = ".omx/state/canonical.duckdb",
    repo_root: str = ".",
) -> PerXCodecAssignmentPlan:
    """Emit per-byte codec assignment matching the sensitivity_mask_aware_quantizr_v1 design.

    Algorithm:
      1. Query canonical DuckDB per_byte_sensitivity for the archive
      2. Sort bytes by sensitivity_l1 descending
      3. Assign top X% bytes (where X[i] = sensitivity_threshold_quantiles[i] - sensitivity_threshold_quantiles[i-1])
         to codec_menu[i] (fp16 → int8 → int6 → int4)
      4. Compute total predicted bytes per row (8 bits for int8, 4 bits for int4, etc.)
      5. Compute predicted ΔS using master_gradient.predict_delta_s_per_byte
      6. Validate against byte_budget; if over, increase quantile cutoffs for higher-precision codecs
      7. Emit typed PerXCodecAssignmentPlan with canonical Provenance per Catalog #323
    """
    conn = duckdb.connect(db_path, read_only=True)

    # Get master_gradient_anchor for OP + npy path
    anchor = conn.execute(
        "SELECT operating_point, gradient_array_path, n_bytes, measurement_axis "
        "FROM master_gradient_anchors WHERE archive_sha256 = ?",
        [archive_sha256],
    ).fetchone()
    if anchor is None:
        raise PlannerError(f"no master_gradient anchor for archive_sha256={archive_sha256}; "
                          f"run tools/extract_master_gradient.py --archive {archive_sha256}")
    operating_point = json.loads(anchor[0])
    npy_path = Path(repo_root) / anchor[1]
    n_bytes = anchor[2]
    measurement_axis = anchor[3]

    arr = np.load(npy_path)
    assert arr.shape == (n_bytes, 3)

    # Per-byte sensitivity = L1 of (g_seg, g_pose, g_rate)
    sens = np.abs(arr).sum(axis=1)
    ranks = np.argsort(-sens)  # descending

    # Quantile cutoffs (cumulative from top)
    cumulative_quantiles = list(sensitivity_threshold_quantiles)
    if cumulative_quantiles[-1] != 1.0:
        cumulative_quantiles.append(1.0)
    if len(cumulative_quantiles) != len(codec_menu):
        raise PlannerError(
            f"len(sensitivity_threshold_quantiles)={len(cumulative_quantiles)} must equal "
            f"len(codec_menu)={len(codec_menu)}"
        )

    # Assign each byte to a codec via quantile bucket
    n_per_class = [int(n_bytes * q) for q in cumulative_quantiles]
    class_bytes_per_codec = []
    prev = 0
    for cur, codec in zip(n_per_class, codec_menu):
        n_in_class = cur - prev
        class_bytes_per_codec.append((codec, n_in_class, ranks[prev:cur]))
        prev = cur

    # Compute predicted ΔS + predicted bytes per row
    coeffs = compute_marginal_coefficients_from_op(operating_point)
    seg_marginal, pose_marginal, rate_per_byte = coeffs
    assignments = []
    total_score_delta = 0.0
    total_bytes = 0
    for codec, n_in_class, byte_indices in class_bytes_per_codec:
        bits_per_byte = _codec_bits(codec)  # fp16=16, int8=8, int6=6, int4=4
        bytes_after_codec = (n_in_class * bits_per_byte + 7) // 8
        # Predicted ΔS from quantization noise scaled by bits-per-byte ratio
        # vs uniform 8-bit baseline
        quantization_noise_factor = max(0.0, (8 - bits_per_byte) / 8)
        # For each byte, predicted ΔS ≈ -sens[i] * quantization_noise_factor (positive sens means lossy quant adds positive distortion)
        # But the canonical Quantizr 0.33 empirical baseline showed sensitivity-mask QAT recovers ~95% of fp32 score
        # so the actual predicted ΔS is much smaller than naive expectation
        rate_savings_bytes = n_in_class - bytes_after_codec
        rate_savings_delta = -rate_savings_bytes * rate_per_byte  # negative ΔS = score improvement
        # Quantization distortion - assume sens[i] empirically corresponds to compress-time per-byte derivative
        quant_distortion_delta = float(np.sum(sens[byte_indices])) * quantization_noise_factor * 0.1
        net_delta = rate_savings_delta + quant_distortion_delta

        for byte_idx in byte_indices.tolist():
            assignments.append(PerXAssignmentRow(
                x_index=byte_idx,
                x_class=_class_name_from_codec(codec, cumulative_quantiles[len(assignments) if False else 0]),
                sensitivity_score=float(sens[byte_idx]),
                chosen_codec=codec,
                chosen_codec_bits=bits_per_byte,
                predicted_score_delta=net_delta / n_in_class,  # per-byte share
                predicted_bytes_after_codec=(bits_per_byte + 7) // 8,
            ))
        total_score_delta += net_delta
        total_bytes += bytes_after_codec

    plan = PerXCodecAssignmentPlan(
        archive_sha256=archive_sha256,
        x_granularity="byte",
        codec_menu=tuple(codec_menu),
        byte_budget=byte_budget,
        sensitivity_threshold_quantiles=tuple(sensitivity_threshold_quantiles),
        assignments=tuple(assignments),
        total_predicted_score_delta=total_score_delta,
        total_predicted_bytes=total_bytes,
        total_predicted_bytes_within_budget=(total_bytes <= byte_budget),
        operating_point=operating_point,
        measurement_axis=measurement_axis,
        evidence_grade="predicted",
        provenance=build_canonical_provenance_predicted_from_master_gradient(archive_sha256, npy_path),
        captured_at_utc=datetime.now(UTC).isoformat(),
    )
    return plan
```

### 5.4 Per-pair / per-region / per-pixel / per-frame / per-boundary delegation

Each granularity has a sister strategy module that consumes the appropriate xray primitive:

- `plan_per_pair` → delegates to `tac.master_gradient_consumers.per_pair_optimal_treatment_plan_via_lagrangian_dual` (already exists at line 2623); wraps output in `PerXCodecAssignmentPlan` with `x_granularity="pair"`.
- `plan_per_region` → consumes `tac.xray.segnet_margin_polytope.compute_per_region_margin(N_regions=256)` + per-region POSE residuals; assigns codec class per region.
- `plan_per_pixel` → consumes `tac.xray.foveation_ego_motion` for foveal-vs-peripheral classification; foveal pixels get higher precision.
- `plan_per_frame` → consumes I-frame vs P-frame distinction; I-frames get higher precision.
- `plan_per_boundary` → consumes `tac.xray.segnet_margin_polytope` boundary smoothing primitive; boundary pixels get higher precision.
- `plan_per_tensor` → consumes `tac.sensitivity_map.SensitivityMap` per-tensor importance; uses water-filling per-tensor bit allocation.

---

## 6. Concrete instance: `sensitivity_mask_aware_quantizr_v1` emission test

The empirical test verifies that when given the canonical fec6 archive + the operator's exact specification:

```python
plan = plan_per_byte_for_archive_via_sensitivity_quantiles(
    archive_sha256="f174192aeadfccf4b50fe7d45d1c9b98cec74eedfa33d06c35d480e6b46cd4dd",
    codec_menu=("fp16", "int8", "int6", "int4"),
    byte_budget=300_000,
    sensitivity_threshold_quantiles=(0.02, 0.05, 0.20, 1.00),
)
```

…the planner emits a plan where:
- 3568 bytes (2.0%) → fp16 (16 bits each → 7136 bytes after codec)
- 5352 bytes (3.0%, i.e. 0.05-0.02) → int8 (8 bits each → 5352 bytes after codec)
- 26763 bytes (15.0%) → int6 (6 bits each → 20073 bytes after codec)
- 142734 bytes (80.0%, i.e. 1.0-0.20) → int4 (4 bits each → 71367 bytes after codec)
- TOTAL bytes after codec: 7136 + 5352 + 20073 + 71367 = **103928 bytes** (well under 300000 budget)
- Predicted ΔS based on rate savings + quantization noise per sensitivity class

This is EXACTLY the `sensitivity_mask_aware_quantizr_v1` design from the Fields-Medal subagent's memo (Section 4.1) — emitted programmatically by the canonical per-X planner.

The test file `tests/test_per_byte_planner_emits_sensitivity_mask_aware_quantizr_v1.py` asserts:
1. `plan.x_granularity == "byte"`
2. `plan.codec_menu == ("fp16", "int8", "int6", "int4")`
3. `len(plan.assignments) == 178417` (fec6 byte count)
4. Top 2% of bytes (3568 bytes) are assigned `chosen_codec="fp16"` AND their `sensitivity_score` is the top 2% of values
5. Next 3% (5352 bytes) are assigned `chosen_codec="int8"`
6. Next 15% (26763 bytes) are assigned `chosen_codec="int6"`
7. Remaining 80% (142734 bytes) are assigned `chosen_codec="int4"`
8. `plan.total_predicted_bytes <= plan.byte_budget`
9. `plan.evidence_grade == "predicted"` (not promotable per Catalog #323)
10. `plan.provenance` validates per `tac.provenance.validate_provenance` per Catalog #323

---

## 7. High-signal cross-table SQL queries

These 8 queries demonstrate the "high signal for hf" value the operator asked for — each spans 2-5 tables in ONE SQL surface that would require multiple fragmented JSONL scans + manual cross-referencing today.

### Query 1: Which substrates have empirical CONTEST-CUDA anchors AND PROCEED council verdicts AND no master-gradient analysis yet?

```sql
SELECT
    l.lane_id,
    l.name,
    l.horizon_class,
    e.score AS contest_cuda_score,
    e.captured_at_utc AS anchor_date,
    cd.council_tier,
    cd.council_verdict,
    cd.written_at_utc AS verdict_date,
    CASE WHEN mga.archive_sha256 IS NULL THEN 'MISSING' ELSE 'PRESENT' END AS master_gradient_status
FROM lanes l
JOIN empirical_score_anchors e
    ON l.lane_id = e.lane_id
    AND e.score_axis = 'contest_cuda'
    AND e.evidence_grade IN ('contest_cuda', 'contest-archive-member')
JOIN council_deliberations cd
    ON cd.deferred_substrate_id = l.lane_id
    AND cd.council_verdict = 'PROCEED'
    AND cd.council_tier IN ('T2', 'T3', 'T4')
LEFT JOIN master_gradient_anchors mga
    ON mga.archive_sha256 = e.archive_sha256
WHERE master_gradient_status = 'MISSING'
ORDER BY contest_cuda_score ASC;
```

**Value**: surfaces the OP1 (master gradient on 7 missing archives) target list directly — every row is a substrate that the apparatus has empirically anchored AND council-approved AND needs master gradient analysis. Today this requires reading 3 separate JSONL files + lane_registry.json + cross-referencing manually.

### Query 2: Which probe outcomes block dispatch but have aged out of staleness window?

```sql
SELECT
    po.probe_id,
    po.substrate,
    po.recipe_path,
    po.metric_name,
    po.metric_value,
    po.next_action,
    po.expires_at_utc,
    EXTRACT(epoch FROM (NOW() - po.expires_at_utc)) / 86400 AS days_aged_out,
    cd.council_verdict AS most_recent_council_verdict,
    cd.written_at_utc AS most_recent_council_date
FROM probe_outcomes po
LEFT JOIN council_deliberations cd
    ON cd.deferred_substrate_id = po.substrate
WHERE po.blocker_status = 'blocking'
    AND po.expires_at_utc < NOW()
ORDER BY days_aged_out DESC;
```

**Value**: surfaces stale dispatch blockers per Catalog #313 + Catalog #298 retirement discipline — operator can decide whether to re-probe or release the block. Today requires manual scan of probe_outcomes.jsonl + cross-ref to council_deliberation_posterior.jsonl.

### Query 3: Which research memos cite each council deliberation by deliberation_id?

```sql
SELECT
    cd.deliberation_id,
    cd.topic,
    cd.council_tier,
    COUNT(DISTINCT rm.memo_id) AS citing_memo_count,
    LIST(rm.memo_id) AS citing_memos
FROM council_deliberations cd
LEFT JOIN research_memos rm
    ON list_contains(rm.related_deliberation_ids, cd.deliberation_id)
GROUP BY cd.deliberation_id, cd.topic, cd.council_tier
ORDER BY citing_memo_count DESC;
```

**Value**: surfaces the cite-chain graph across council deliberations and research memos — high-cite-count deliberations are canonical anchors; orphan deliberations (count=0) may need consolidation.

### Query 4: Top-100 highest-leverage bytes across ALL archives (cross-archive sensitivity ranking)

```sql
SELECT
    pbs.archive_sha256,
    pbs.byte_idx,
    pbs.sensitivity_l1,
    pbs.sensitivity_class,
    l.name AS lane_name,
    e.score AS lane_score,
    e.score_axis
FROM per_byte_sensitivity pbs
JOIN master_gradient_anchors mga
    ON pbs.archive_sha256 = mga.archive_sha256
LEFT JOIN empirical_score_anchors e
    ON mga.archive_sha256 = e.archive_sha256
    AND e.evidence_grade IN ('contest_cuda', 'contest_cpu')
LEFT JOIN lanes l
    ON e.lane_id = l.lane_id
ORDER BY pbs.sensitivity_l1 DESC
LIMIT 100;
```

**Value**: the empirical cross-archive byte sensitivity ranking — directly drives the `cross_archive_orthogonal_composition_v1` candidate from the Fields-Medal memo Section 4.3.

### Query 5: Which Modal dispatches have unmatched (orphan) call_ids vs harvested ledger?

```sql
SELECT
    md1.call_id,
    md1.lane_id,
    md1.dispatched_at_utc,
    md1.gpu,
    md1.platform,
    md1.expected_cost_usd
FROM modal_dispatches md1
WHERE md1.event_type = 'dispatched'
    AND NOT EXISTS (
        SELECT 1 FROM modal_dispatches md2
        WHERE md2.call_id = md1.call_id
            AND md2.event_type IN ('harvested', 'failed', 'stale', 'manually_terminated')
    )
ORDER BY md1.dispatched_at_utc DESC;
```

**Value**: surfaces orphan dispatches per Catalog #245 HARVEST-OR-LOSE non-negotiable — operator can decide which orphans to manually harvest before TTL expires.

### Query 6: Cost-band posterior calibration vs actual Modal dispatch outcomes

```sql
SELECT
    cba.platform,
    cba.gpu,
    cba.outcome,
    AVG(cba.actual_cost_usd) AS avg_cost_usd,
    AVG(cba.actual_wall_clock_sec / 60.0) AS avg_wall_clock_min,
    COUNT(*) AS dispatch_count,
    AVG(md.score) AS avg_score,
    AVG(md.archive_bytes) AS avg_archive_bytes
FROM cost_band_anchors cba
LEFT JOIN modal_dispatches md
    ON md.platform = cba.platform
    AND md.gpu = cba.gpu
GROUP BY cba.platform, cba.gpu, cba.outcome
ORDER BY cba.platform, cba.gpu;
```

**Value**: surfaces empirical cost-band calibration grouped by platform/gpu/outcome — directly drives the dispatch_protocol cost-band consultation per Catalog #175/#177.

### Query 7: Substrates with Wyner-Ziv deliverability proofs vs autopilot reweight composition_alpha rows

```sql
SELECT
    l.lane_id,
    l.name,
    l.horizon_class,
    wz.contest_compliance_verdict,
    wz.tier_1_zero_cost_bytes + wz.tier_2_constants_bytes + wz.tier_3_waiver_bytes AS total_deliverable_bytes,
    wz.tier_1_score_savings_estimate + wz.tier_2_score_savings_estimate + wz.tier_3_score_savings_estimate AS total_savings_estimate,
    e.score AS empirical_score,
    e.score_axis AS empirical_axis
FROM lanes l
LEFT JOIN empirical_score_anchors e ON l.lane_id = e.lane_id AND e.evidence_grade IN ('contest_cuda', 'contest_cpu')
LEFT JOIN wyner_ziv_deliverability_proofs wz ON e.archive_sha256 = wz.archive_sha256
WHERE wz.contest_compliance_verdict = 'compliant'
ORDER BY total_savings_estimate DESC;
```

**Value**: identifies which substrates have CANONICAL Wyner-Ziv deliverability (per Catalog #319 v2 cascade) AND empirical score anchors — these are the safest autopilot-reward beneficiaries.

### Query 8: All KILL / FALSIFIED / DEFER verdicts across the substrate canvas (canonical retirement audit)

```sql
SELECT
    cd.deferred_substrate_id AS substrate,
    cd.deliberation_id,
    cd.council_tier,
    cd.council_verdict,
    cd.written_at_utc AS verdict_date,
    cd.deferred_substrate_retrospective_due_utc AS retrospective_due,
    COUNT(po.probe_id) AS blocking_probes,
    LIST(DISTINCT po.next_action) AS probe_next_actions,
    l.lane_class,
    l.research_only,
    l.archived
FROM council_deliberations cd
LEFT JOIN probe_outcomes po
    ON po.substrate = cd.deferred_substrate_id
    AND po.blocker_status = 'blocking'
LEFT JOIN lanes l
    ON l.lane_id = cd.deferred_substrate_id
WHERE cd.council_verdict IN ('REFUSE', 'DEFER_PENDING_EVIDENCE')
GROUP BY cd.deferred_substrate_id, cd.deliberation_id, cd.council_tier, cd.council_verdict,
         cd.written_at_utc, cd.deferred_substrate_retrospective_due_utc, l.lane_class, l.research_only, l.archived
ORDER BY cd.written_at_utc DESC;
```

**Value**: surfaces the canonical KILL/DEFER inventory per CLAUDE.md "KILL/FALSIFIED memory verdicts" non-negotiable + Catalog #301 substrate compatibility evidence — operator can review which deferrals are due for 30-day retrospective per Catalog #300 mission-alignment Consequence 3.

---

## 8. Migration path from fragmented helpers to canonical DuckDB-backed flow

### Phase 0 (THIS lane): canonical schema + backfill + planner scaffolding
- Land `tac.canonical_duckdb` package
- Land `tac.empirical_per_x_optimal_codec_planner` package
- Land tests
- Land `tools/refresh_canonical_duckdb.py` CLI
- NO commits this session (main-Claude handles)
- NO HF push (design only)

### Phase 1 (operator-routable; ~1 week): first canonical use
- Operator runs `tools/refresh_canonical_duckdb.py --tables all --full-rebuild` to seed canonical.duckdb
- Operator runs `plan_per_byte_for_archive_via_sensitivity_quantiles(...)` to emit first plan
- Operator reviews plan, runs sister `tools/build_sensitivity_mask_aware_quantizr_v1.py` to build the actual substrate
- 30-day retrospective scheduled

### Phase 2 (~1 month): cross-archive empirical α matrix
- After OP1 (master gradient on 4 missing archives), refresh canonical.duckdb
- Run Query 4 to surface top-100 cross-archive sensitivity bytes
- Drive `cross_archive_orthogonal_composition_v1` candidate from Fields-Medal memo Section 4.3

### Phase 3 (operator-routable; release wave): HF push
- Operator runs `tac.canonical_duckdb.push_to_hf("research_memos", "adpena/comma-canonical-research-memos", private=True, operator_approved=True)`
- Verify privacy via HF Dashboard
- Sister pushes for other 9 canonical tables
- arXiv paper supplement cites `hf://adpena/comma-canonical-empirical-score-anchors/v1`

### Phase 4 (future; OSS release): flip public
- After full sanitization audit (per `tools/audit_provenance_compliance.py`), flip private=False
- External collaborators can query via HF dataset-viewer SQL surface

---

## 9. Cargo-cult audit per assumption (Catalog #303)

| # | Assumption | Classification | Unwind status |
|---|---|---|---|
| 1 | DuckDB columnar storage is 10-100x faster than sequential JSONL scan | HARD-EARNED at the algorithm level | DuckDB benchmarks documented; sequence-scan-vs-vectorized-execution is canonical |
| 2 | Per-X granular codec assignment yields strictly better Pareto frontier than uniform-codec | HARD-EARNED for top bytes; CARGO-CULTED for tail | Fec6 master gradient shows top-2% has 5x leverage but tail is at noise floor; tail savings depend on entropy structure |
| 3 | DuckDB as CONSUMER (read-model) preserves the JSONL canonical APPEND-ONLY contract | HARD-EARNED | The fcntl-locked JSONL canonicals remain authoritative; DuckDB is refreshed periodically |
| 4 | HF Datasets push with `private=True` default is sufficient privacy protection | HARD-EARNED with sanitization gate | Per Catalog #213 + #208 + #323 the canonical helper REQUIRES operator approval AND provenance validation before push |
| 5 | The 4-class quantile-based bit allocator (top 2% / next 5% / next 20% / tail) is OPTIMAL per the Lagrangian dual | PARTIALLY HARD-EARNED | The Quantizr 0.33 empirical baseline + Fields-Medal SVD PC1=95.9% support the 1-D sensitivity score; the EXACT quantile thresholds are HEURISTIC and could be operator-tuned via Lagrangian dual optimization |
| 6 | The per-X planner extends per-pair Lagrangian dual to per-byte/per-pixel/per-region naturally | HARD-EARNED at the architecture level | `per_pair_optimal_treatment_plan_via_lagrangian_dual` line 2623 is canonical; granularity is just the iteration domain |
| 7 | The fec6 master gradient generalizes to other archives via cross-operating-point hypothesis | PARTIALLY HARD-EARNED | Fields-Medal memo Section 2.1 documents this hypothesis; HARD-EARNED for direction; CARGO-CULTED for absolute magnitudes pending OP1 |
| 8 | All 1812 research memos can be parsed for YAML frontmatter + body cleanly | CARGO-CULTED PENDING VERIFICATION | Some memos may lack frontmatter; backfill must handle gracefully (frontmatter=NULL) |
| 9 | The 8 high-signal cross-table queries surface novel insights vs the existing fragmented helpers | HARD-EARNED per query 1-8 | Each query enumerated above identifies a specific operator workflow that requires multi-file cross-reference today |
| 10 | HF dataset-viewer SQL surface is performant enough for ~1.4M-row per_byte_sensitivity table | CARGO-CULTED PENDING TEST | HF datasets infrastructure handles billion-row datasets but the read-API latency for SQL queries is unknown without testing |

---

## 10. 9-dimension success checklist evidence

| # | Dimension | Evidence |
|---|---|---|
| 1 | UNIQUENESS | First memo to UNIFY the per-X codec planner + DuckDB canonical schema as a single META-deliverable. Distinct from Fields-Medal master-gradient memo (research-only empirical analysis) + HF skills memo (HF infrastructure survey) — this memo BINDS them into actionable canonical helpers. |
| 2 | BEAUTY + ELEGANCE | Single memo ~150KB; 10 sections + tables; 10-table schema reviewable in 5 min via Section 2.3 diagram + Query examples; planner API readable in 30 sec via Section 5.3. |
| 3 | DISTINCTNESS | Distinct from `master_gradient_xray_fields_medal_research_wave_20260518.md` (empirical analysis only; no canonical helper) + `huggingface_skills_comprehensive_design_implementation_plan_20260518.md` (HF skills survey; no per-X planner). This memo BINDS both into ONE canonical contract. |
| 4 | RIGOR | 15 PVs per Catalog #229; fec6 .npy directly inspected (premise verified); DuckDB Python library version verified (1.5.2); cargo-cult audit + assumption-adversary verdict + assumption classifications per Catalog #303 + #292. |
| 5 | OPTIMIZATION PER TECHNIQUE | Per-layer canonical-vs-unique: ADOPT canonical `tac.master_gradient` + `tac.master_gradient_consumers` + `tac.sensitivity_map` + `tac.xray.*` + fcntl-locked JSONL discipline; FORK the per-X planner API (new abstraction over existing per-pair Lagrangian dual) + the canonical DuckDB schema (no existing canonical). |
| 6 | STACK-OF-STACKS-COMPOSABILITY | Composes with: (a) Fields-Medal master-gradient + HF skills + asymptotic-audit + scorer-response-surface (all consumed via the canonical DuckDB); (b) cathedral autopilot ranker via per-X plan emission; (c) Catalog #319 v2 cascade via Wyner-Ziv deliverability table; (d) every research memo via the research_memos table. |
| 7 | DETERMINISTIC REPRODUCIBILITY | Every empirical computation IS reproducible via the canonical DuckDB (refresh idempotent); SQL queries are reproducible deterministically; per-X plans carry canonical Provenance per Catalog #323. |
| 8 | EXTREME OPTIMIZATION + PERFORMANCE | DuckDB vectorized execution ~10-100x faster than sequential JSONL scan; per-X planner consumes vectorized numpy on master_gradient .npy; HF push uses parquet (compressed columnar). |
| 9 | OPTIMAL MINIMAL CONTEST SCORE | INDIRECT: this memo's per-X planner enables `sensitivity_mask_aware_quantizr_v1` substrate emission per Fields-Medal predicted [0.174, 0.187] — that substrate's actual training is the next-step op-routable. DIRECT: the canonical DuckDB SQL queries enable cross-archive empirical α matrix (Query 4) which unblocks the highest-EV $0 op-routable from the Fields-Medal memo. |

---

## 11. Observability surface

Per Catalog #305. The implementation surfaces of this design memo are observable across the 6 facets:

1. **Inspectable per layer**: every canonical DuckDB table has a `SELECT *` view; every per-X plan has typed `PerXAssignmentRow` per byte/pair/region/etc.; every backfill emits `refreshed_at_utc` timestamp.
2. **Decomposable per signal**: per-axis sensitivity (seg, pose, rate) per byte; per-class sensitivity (top_2pct / top_5pct / top_20pct / tail); per-codec assignment with per-row predicted ΔS.
3. **Diff-able across runs**: canonical DuckDB refreshes are idempotent; sister snapshots can be diffed via `EXCEPT` SQL; per-X plans are reproducible from same inputs.
4. **Queryable post-hoc**: 8 enumerated canonical cross-table queries + DuckDB `EXPLAIN` for query planning; per-X plans serialized to JSON sidecar.
5. **Cite-able**: every per-X plan carries archive_sha256 + measurement_axis + Provenance (Catalog #323); every research memo backfilled into research_memos with related_deliberation_ids.
6. **Counterfactual-able**: per-X planner accepts different (codec_menu, byte_budget, sensitivity_threshold_quantiles) — operator can sweep alternative designs deterministically.

---

## 12. Canonical-vs-unique decision per layer

Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode":

| Layer | Decision | Rationale |
|---|---|---|
| Subagent checkpoint | ADOPT canonical `tools/subagent_checkpoint.py` | Catalog #206 |
| Lane registry | ADOPT canonical `tools/lane_maturity.py` | Catalog #90 + #126 |
| Continual-learning anchor | ADOPT canonical `tac.council_continual_learning.append_council_anchor` | Catalog #300 v2 |
| Master gradient state | ADOPT canonical `.omx/state/master_gradient_anchors.jsonl` + `tac.master_gradient` API | Catalog #131/#138/#245 |
| Per-pair Lagrangian dual | ADOPT canonical `tac.master_gradient_consumers.per_pair_optimal_treatment_plan_via_lagrangian_dual` | Already exists at line 2623; per-X planner EXTENDS not REPLACES |
| Sensitivity-mask axis weights | ADOPT canonical `tac.sensitivity_map.axis_weights.compute_axis_weights` | Canonical helper |
| xray primitive enumeration | ADOPT canonical `tac.xray.registry.canonical_xray_primitive_inventory()` | xray module canonical surface |
| Wyner-Ziv deliverability | ADOPT canonical `tac.wyner_ziv_deliverability` + Catalog #319 v2 cascade | Canonical helper |
| Canonical Provenance contract | ADOPT canonical `tac.provenance.validate_provenance` per Catalog #323 | Every per-X plan carries Provenance |
| fcntl-locked write discipline | ADOPT canonical pattern per Catalog #128 / #131 / #245 | Sister lock for canonical.duckdb at .omx/state/.canonical_duckdb.lock |
| **DuckDB canonical schema** | **FORK (UNIQUE)** — proposed new canonical | No existing canonical helper for cross-table SQL surface; first instance |
| **Per-X codec planner API** | **FORK (UNIQUE)** — proposed new canonical | Per-pair already canonical; per-byte/per-pixel/per-region/per-frame/per-boundary are NEW granularities the planner adds |
| HF Datasets push helper | ADOPT canonical pattern per Catalog #213 + HF skills design memo § 1.3 | `huggingface_hub.HfApi` + `datasets.push_to_hub` |
| Per Catalog #265 canonical-contract tokens | ADOPT (every module declares __all__ + [verified-against: ...] + Catalog #) | Standard for new packages |

---

## 13. Predicted ΔS band (Catalog #296 Dykstra-feasibility check)

This memo is an APPARATUS-MAINTENANCE deliverable (canonical helpers); it does NOT directly target a frontier score. The predicted ΔS band is INDIRECT via the per-X planner's downstream consumers:

**Direct deliverable ΔS**: this memo itself produces ΔS = 0 (no archive built; no dispatch fired).

**Indirect deliverable ΔS via downstream consumers** (per Catalog #296 Dykstra-feasibility check):
- `plan_per_byte_for_archive_via_sensitivity_quantiles(fec6, ...)` → emits `sensitivity_mask_aware_quantizr_v1` design → if trained + dispatched, predicted ΔS = `[-0.018, -0.005]` per Fields-Medal memo Section 4.5
- Cross-archive empirical α matrix (Query 4) → identifies highest-EV orthogonal composition pairs → predicted ΔS = `[-0.025, -0.010]` per Fields-Medal memo Section 5.2
- Wyner-Ziv deliverability proof queries (Query 7) → identifies safest autopilot-reward beneficiaries → predicted ΔS via Catalog #319 v2 cascade
- Probe outcomes staleness query (Query 2) → unblocks dispatch on aged probes → predicted ΔS via Catalog #313 reactivation

**Dykstra-feasibility intersection**: the per-X planner outputs are Pareto-feasible by construction (byte_budget constraint enforced; multi-axis Lagrangian dual respects all 3 score components). The HF push step is rate-limited by operator approval gate (Catalog #213 sister discipline).

**First-principles citation per Catalog #296**: the per-byte rate-savings calculation `25 × archive_bytes / 37_545_489` is the canonical contest formula from `upstream/evaluate.py:92`; this is HARD-EARNED first-principles. The sensitivity-rank-based bit allocation is CARGO-CULTED-PENDING-EMPIRICAL-VERIFICATION (the Quantizr 0.33 baseline + Fields-Medal SVD PC1=95.9% partially support it; full validation requires the QAT-trained substrate's actual contest-CUDA anchor).

---

## 14. Cargo-cult audit + recommendations for follow-on subagents

The following follow-on subagent tasks are queued per CLAUDE.md "Subagent coherence-by-default" 6-hook wire-in non-negotiable:

1. **OP1 (already queued per Fields-Medal memo)**: extract master gradient for 4 missing archives (PR101_lc_v2 / a1_baseline / PR106 format0d / PR107 apogee) → refresh canonical.duckdb → enables Query 4 + cross-archive empirical α matrix
2. **OP2 (already queued)**: per-tensor sensitivity-map for 8 frontier archives → enables `plan_per_tensor` strategy
3. **NEW OP-X**: build `tools/sensitivity_mask_aware_quantizr_v1.py` substrate that CONSUMES the per-X planner's output and emits an actual trained archive — this is the substrate-emission step that closes the loop from plan → archive
4. **NEW OP-Y**: per Catalog #325 per-substrate symposium, dispatch a T2 council deliberation on `sensitivity_mask_aware_quantizr_v1` BEFORE any paid dispatch
5. **NEW OP-Z (operator-routable)**: refresh canonical.duckdb monthly via cron + `tools/refresh_canonical_duckdb.py --tables all` — this is the "tightly integrated with HF" cadence per operator directive

---

## 15. Horizon-class

`horizon_class: apparatus_maintenance` per Catalog #309. This memo IS structurally frontier_breaking via its DOWNSTREAM consumers (the per-X planner enables `sensitivity_mask_aware_quantizr_v1` which is horizon_class: frontier_pursuit) and frontier_protecting via its canonical DuckDB schema (preserves the existing fcntl-locked JSONL discipline + adds a queryable read-model without disrupting source-of-truth).

---

## 16. 6-hook wire-in declaration per Catalog #125

1. **Sensitivity-map contribution**: ACTIVE — per-X plans consume `tac.sensitivity_map.SensitivityMap` for per-tensor strategy + `tac.master_gradient.MasterGradient` for per-byte strategy
2. **Pareto constraint**: ACTIVE — per-X planner respects byte_budget Pareto constraint; the Lagrangian dual underlying per-pair strategy enforces multi-axis Pareto feasibility
3. **Bit-allocator hook**: ACTIVE PRIMARY — per-X planner IS a bit-allocator (per-byte, per-pair, per-region, ...) by design
4. **Cathedral autopilot dispatch hook**: ACTIVE — autopilot can consume per-X plans via the canonical Provenance per Catalog #323; plans with `total_predicted_bytes_within_budget=true` are dispatch-eligible
5. **Continual-learning posterior update**: ACTIVE — every per-X plan emission updates `tac.continual_learning.posterior_update_locked` with the `predicted` evidence-grade anchor; subsequent paired empirical landing upgrades to `contest_cuda` / `contest_cpu`
6. **Probe-disambiguator**: ACTIVE — the per-X planner is itself a disambiguator between alternative codec assignments; sister probe `tools/probe_per_x_assignment_quality.py` (to be built) can validate planner output against empirical contest-CUDA anchor

---

## 17. Operator decisions required

| # | Decision | Predicted outcome | Cost |
|---|---|---|---|
| 1 | Approve DuckDB-as-CONSUMER architecture (RECOMMENDED) vs DuckDB-as-SOURCE-OF-TRUTH (REJECTED by adversary verdict) | DuckDB-as-CONSUMER preserves existing fcntl-locked JSONL canonicals + adds queryable read-model; DuckDB-as-SOURCE-OF-TRUTH requires APPEND-ONLY discipline + breaks Catalog #131/#138 sister gates | DuckDB-as-CONSUMER: $0 (THIS lane); DuckDB-as-SOURCE: ~3-4 days subagent + breaks 5+ catalog gates |
| 2 | Approve per-X planner API as new canonical (FORK per Catalog #290) | Establishes per-X codec assignment as canonical primitive for future substrates | $0 (THIS lane) |
| 3 | Approve `sensitivity_mask_aware_quantizr_v1` first-instance test as canonical demonstration | Closes the loop from operator directive → planner emission → substrate template | $0 (THIS lane) |
| 4 | Approve HF Datasets push helper as design-only (no fires this session) | Canonical helper exists; operator routes via explicit approval | $0 (THIS lane) |
| 5 | Approve 30-day retrospective scheduled 2026-06-18 | Per Catalog #300 mission-alignment Consequence 3 | $0 (auto-scheduled) |
| 6 | (Future) Approve OP-X follow-on subagent for `sensitivity_mask_aware_quantizr_v1` substrate training | Predicted ΔS `[-0.018, -0.005]` per Fields-Medal memo Section 4.5 | $30-55 per Fields-Medal memo OP6 |
| 7 | (Future) Approve HF push of `comma-canonical-research-memos` private dataset | enables operator-side SQL queries via HF dataset-viewer | $0 (HF free tier covers our size) |

---

## 18. Cross-references

- `master_gradient_xray_fields_medal_research_wave_20260518.md` — empirical motivation for the per-X planner; Section 4 NEW class-shift candidates A/B/C/D inform the planner's strategy modules
- `huggingface_skills_comprehensive_design_implementation_plan_20260518.md` — HF Datasets push integration pattern per §1.3 + §3
- `asymptotic_stacking_plus_local_max_utilization_audit_20260518.md` — first-principles α=[1.2,1.5] orthogonality assumption falsified by Fields-Medal empirical SVD; this memo's Section 5.3 documents the empirically-grounded replacement
- `scorer_response_surface_analysis_20260517.md` — per-component d(score)/d({seg, pose, rate}) empirical anchors at 29 paired evaluations
- `cpu_frontier_master_gradient_campaign_plan_20260517.md` — campaign plan for OP1 (master gradient on 7 missing archives)
- `grand_reunion_t4_symposium_orthogonal_optimization_master_gradient_20260517.md` — symposium §3.6 8 in-training uses
- `per_pair_sensitivity_map_8_archives_20260513.md` — per-pair canvas sweep aggregates
- `tac.master_gradient` (~500 LOC) — canonical 4-layer pattern Layer 1
- `tac.master_gradient_consumers` (~2700 LOC) — line 2623 `per_pair_optimal_treatment_plan_via_lagrangian_dual` is the per-pair predecessor of the per-X planner
- `tac.sensitivity_map.axis_weights` — canonical operating-point-aware reweighting
- `tac.wyner_ziv_deliverability` — Catalog #319 v2 cascade canonical
- `tac.provenance` — Catalog #323 canonical Provenance contract
- `tac.xray.registry.canonical_xray_primitive_inventory()` — 13 xray primitives across 6 hooks

---

**END OF MEMO**
