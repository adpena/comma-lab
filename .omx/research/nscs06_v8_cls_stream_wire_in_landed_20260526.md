<!-- SPDX-License-Identifier: MIT -->
<!-- Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE — DO NOT mutate after landing. -->
<!-- Catalog #229 PV: see pre-execution gate report `.omx/research/nscs06_v8_cls_stream_wire_in_pre_execution_gate_report_20260526.md` for the full sister-file-read inventory; this landing memo verifies the implementation lands AS-DESIGNED per the pre-execution gate report. -->
<!-- # CARGO_CULT_AUDIT_SECTION_WAIVED:wire_in_landing_memo_NOT_substrate_design_memo_cargo_cult_audit_inherited_from_sister_path_3_c_L0_scaffold_landing_memo_per_catalog_303 -->
<!-- # 9_DIM_CHECKLIST_EVIDENCE_WAIVED:wire_in_landing_memo_evidence_inherited_from_sister_path_3_c_L0_scaffold_landing_memo_per_catalog_294 -->
<!-- # OBSERVABILITY_SURFACE_SECTION_WAIVED:wire_in_landing_memo_observability_inherited_from_sister_path_3_c_L0_scaffold_landing_memo_per_catalog_305 -->
<!-- # PREDICTED_BAND_VIBES_OK:wire_in_landing_memo_no_NEW_predicted_band_introduced_cls_stream_byte_cost_is_ADDITIVE_to_existing_canonical_equation_26_REPLACEMENT_savings_full_axis_paired_modal_t4_pending_per_catalog_324_post_training_validation -->
<!-- # FORMALIZATION_PENDING:no_NEW_canonical_equation_introduced_existing_canonical_equation_26_anchor_appended_event_pending_post_paired_modal_t4_landing_per_catalog_344 -->
<!-- # COUNCIL_ASSUMPTION_STATEMENT_WAIVED:wire_in_landing_memo_NOT_council_deliberation_per_catalog_292 -->
---
schema_version: nscs06_v8_cls_stream_wire_in_landing_memo_v1_20260526
landing_id: nscs06_v8_cls_stream_wire_in_landed_20260526T183100Z
lane_id: lane_nscs06_v8_cls_stream_wire_in_20260526
landed_utc: 2026-05-26T18:31:00Z
landing_commit: 581b7b129
horizon_class: frontier_pursuit
predicted_band_validation_status: pending_post_training_paired_modal_t4_validation_per_catalog_324
research_only: true
score_claim: false
promotion_eligible: false
rank_or_kill_eligible: false
ready_for_exact_eval_dispatch: false
promotable: false
evidence_grade: "[structural-verifier]"
hardware_substrate: darwin_arm64_m5_max_macos_cpu_local_wire_in_only
measurement_axis: archive-grammar-and-runtime-byte-consumption-only
canonical_equation_refs:
  - procedural_codebook_from_seed_compression_savings_v1
canonical_equation_in_domain_context: nscs06_v8_chroma_lut
mission_predicted_contribution: frontier_breaking_enabler
council_anchor_ref: t3_council_pr110_stacking_pivot_ordering_landed_20260526T170900Z
predecessor_landing_ref: nscs06_v8_chroma_lut_mlx_l1_empirical_landed_20260526T181200Z
subagent_id: nscs06-v8-cls-stream-wire-in-20260526
---

# NSCS06 v8 chroma_lut — cls_stream wire-in landing memo

**Lane:** `lane_nscs06_v8_cls_stream_wire_in_20260526` L1 (impl_complete + memory_entry + Catalog #233 4-gate evidence emitted)
**Commit:** `581b7b129` (canonical serializer; POST-EDIT --expected-content-sha256 on all 4 files)
**Cost:** $0 + ~40 min wall-clock (NO training; NO paid dispatch)
**Predecessor:** TaskCreate #1340 NSCS06 v8 L1 EMPIRICAL respawn (commit `4a4ab1e4f`; canonical equation #26 EXACT byte-savings 4064 closed-form)
**Council mandate:** T3 council #1335 PROCEED_WITH_REVISIONS REVISION #2 + Yousfi BLOCKER finding satisfied.

---

## Pre-execution gate verdict (per pre-execution gate report 2026-05-26T182100Z)

PASS. cls_stream scope = ONE NEW schema version (v3) at procedural-seed path; raw uint8 byte-stream per cell; sister-pattern-replication from NSCS06 strip_everything CH06 v2; 4/5 MLX↔CUDA drift sources STRUCTURALLY NOT APPLICABLE; 1/5 (F.interpolate) PRE-ENGINEERED via Pillow NEAREST; canonical-vs-frontier-push = CANON-APPLICATION; sister-checkpoint guard PROCEED.

---

## Wire-in LOC summary

| File | Type | LOC change | Surface |
|---|---|---:|---|
| `src/tac/substrates/nscs06_v8_chroma_lut/archive.py` | EXTEND | +123 | NEW `CH08_SCHEMA_VERSION_PROCEDURAL_SEED_WITH_CLS_STREAM=3` constant + `CH08_HEADER_FMT_V3` (39 bytes) + `CH08_HEADER_SIZE_V3` invariant + `Nscs06V8Archive.cls_lowres: np.ndarray \| None = None` field + `pack_archive(cls_bytes=...)` kwarg + v3 schema-version dispatch (validation + length + label-range checks) + `parse_archive` v3 dispatch via version-byte peek + cls_stream parsing + cls_lowres propagation. v1/v2 paths PRESERVED byte-stable. |
| `src/tac/substrates/nscs06_v8_chroma_lut/inflate.py` | EXTEND | +17 | NEW `CH08_SCHEMA_VERSION_PROCEDURAL_SEED_WITH_CLS_STREAM` import + `_resolve_chroma_lut` extended to accept v3 (sister path of v2 procedural seed) + cargo-cult #5 site (`cls_full = np.zeros_like(gray_full)`) replaced with branched consume: `if arc.cls_lowres is not None: cls_full = upsample(cls_lowres[p], NEAREST) else: cls_full = np.zeros_like(gray_full)` (v1/v2 legacy preserved). |
| `src/tac/substrates/nscs06_v8_chroma_lut/tests/test_cls_stream_wire_in.py` | NEW | +338 | 17 dedicated tests covering: header layout invariants (39-byte v3 header; v3 != v1 != v2 schema versions) / pack_archive v3 with cls_bytes / v1+v2 backward compat preserved / v3 rejects cls_bytes with v1 inline LUT / v3 rejects wrong cls length / v3 rejects label >= num_segnet_classes / parse_archive v3 returns cls_lowres / v3 byte-stable roundtrip / v2+v1 return cls_lowres=None / pack→parse→pack byte-stable / **inflate v3 vs v2 produces DIFFERENT frames given identical seed + non-uniform cls_stream** (STRUCTURAL PROOF that cargo-cult #5 is UNWOUND) / v2 legacy cls=0 uniform preserved / v3 with cls=0 uniform produces byte-identical to v2 (boundary invariant) / v3 archive total size invariant. |
| `.omx/research/nscs06_v8_cls_stream_wire_in_pre_execution_gate_report_20260526.md` | NEW | +237 | Pre-execution gate report per the prompt scope. |

**TOTAL: +715 LOC across 4 files; 4 mutations to existing code (archive.py + inflate.py — all backward-compat-preserving); 2 NEW files. HNeRV parity L4 inflate.py budget post-extension: ~140 LOC (substrate_engineering exception per L7; well under ≤200 LOC ceiling).**

**Tests: 197/197 pass** (180 pre-existing PRESERVED + 17 new cls_stream wire-in tests). Reproducer:
```bash
PYTHONPATH=src:upstream:$PWD .venv/bin/python -m pytest src/tac/substrates/nscs06_v8_chroma_lut/tests/ -v
```

---

## Catalog #233 promotion canonical 4-gate evidence (per gate)

| Gate | Pre-wire-in | Post-wire-in | Evidence |
|---|---|---|---|
| **(1) impl_complete** | TRUE (Phase 2 BUILD `_full_main` IMPLEMENTED) | **TRUE** + extended with cls_stream path | `pack_archive(cls_bytes=...)` + `Nscs06V8Archive.cls_lowres` + `inflate_one_video` v3 branch all implemented; ZERO `NotImplementedError` introduced. |
| **(2) parser_section_manifest_consistent** | TRUE for v1/v2 sections only | **TRUE** extended to v3 | `test_v3_header_size_invariant` (39 bytes), `test_parse_archive_v3_cls_lowres_byte_stable_roundtrip`, `test_parse_archive_v3_byte_stable_pack_parse_pack`, `test_v3_archive_total_size_invariant` ALL PASS. |
| **(3) inflate_runtime_byte_consumption** | **FAIL** (cls_full=zeros at cargo-cult #5 site; FAIL_AT_CLASS_1 verdict) | **PASS** | `test_inflate_v3_vs_v2_produces_different_frames_proves_cls_consumption` PASSES — when v3 carries non-uniform cls_stream + v2 has cls=0 uniform, inflated frames MUST differ. Test would have failed (`v3 inflate did NOT consume cls_stream — cargo-cult #5 STILL ACTIVE`) if the wire-in were absent. This is the runtime witness that the cargo-cult-5 cls_stream is OPERATIONAL. The sister `distinguishing_feature_smoke.PerClassChromaDistinguishingFeatureVerdict` FAIL_AT_CLASS_1 → PASS_PER_CLASS verdict transition is realizable once a sister test wires the verdict-dataclass producer through a v3 archive (operator-routable downstream). |
| **(4) roundtrip_test_passes** | TRUE for v1/v2 | **TRUE** extended to v3 | `test_parse_archive_v3_cls_lowres_byte_stable_roundtrip` + `test_parse_archive_v3_byte_stable_pack_parse_pack` + `test_inflate_v3_with_uniform_class_matches_v2` (boundary invariant) ALL PASS. |

**Lane registry mark (operator-routable; out of this scope per checkpointing protocol — sister subagent can register via `tools/lane_maturity.py mark lane_nscs06_v8_chroma_lut --gate <gate> --evidence "<commit 581b7b129 evidence>"`).**

---

## Drift surface declaration per 2026-05-26 MLX↔CUDA bidirectional drift standing directive

Per the pre-execution gate report § 2: ALL 5 drift sources STRUCTURALLY NOT APPLICABLE to the cls_stream wire-in (uint8 byte-stream codec; no bfloat16/fp16 ops; no softmax/LSE; no AdamW; no EMA shadow; F.interpolate uses Pillow NEAREST — deterministic across CPU backends per sister NSCS06 strip_everything CH06 v2 precedent commit 4292c8ce2). **Wire-in is STRUCTURALLY ROBUST to all 5 MLX↔CUDA bidirectional drift sources by paradigm-design.** No drift-source-induced verdict divergence anticipated when paired Modal T4 CPU+CUDA dispatch fires.

---

## Canonical-vs-frontier-push decision per 2026-05-26 pushing-the-frontier-of-research-on-optimization-algorithms standing directive

**Decision: CANON-APPLICATION**. cls_stream wire-in is sister-pattern-replication from NSCS06 strip_everything CH06 v2 (symposium commit 4292c8ce2). The minimum-scope raw-uint8 byte-stream design is canonical "byte-disjoint section in monolithic 0.bin archive" (HNeRV parity L3). Arith-coding is a follow-up bytes-saving optimization (~5-30 KB savings depending on cls entropy at full grayscale_h × grayscale_w × num_pairs scale) DEFERRED-pending-bytes-budget-analysis per CLAUDE.md "Forbidden premature KILL". No frontier-push contribution to canonical equation registry. Existing canonical equation #26 IN-DOMAIN `nscs06_v8_chroma_lut` context PRESERVED; rate-axis additive cls_stream cost will be measured at paired Modal T4 dispatch and either appended as a NEW canonical equation (per Catalog #344) IF the full-axis ΔS trade-off produces a learnable rate-vs-distortion frontier, OR as a sister `anchor_appended` event on existing canonical equation #26.

---

## 6-hook wire-in declaration per Catalog #125

| Hook | Status | Evidence |
|---|---|---|
| **#1 sensitivity-map** | **ACTIVE** | cls_stream consumption surfaces per-class chroma sensitivity (canonical via existing `distinguishing_feature_smoke.PerClassChromaDistinguishingFeatureVerdict` consumer). |
| **#2 Pareto constraint** | **ACTIVE** | cls_stream ADDS bytes (rate-axis ADDITIVE); seg+pose-axis EXPECTED to IMPROVE via correct per-class chroma reconstruction; canonical equation #26 REPLACEMENT savings -0.0027 PRESERVED; cls_stream byte cost is the DELTA to measure at paired Modal T4. |
| **#3 bit-allocator** | **PARTIAL** | Raw uint8 cls_stream is the minimum-bits baseline; arith-coded variant is the bit-allocator-optimal canonical follow-up (DEFERRED-pending-bytes-budget-analysis). |
| **#4 cathedral autopilot dispatch** | **ACTIVE PRIMARY** | cls_stream wire-in UNBLOCKS Catalog #233 L1→L2 promotion 4-gate; cathedral autopilot ranker can NOW promote NSCS06 v8 chroma_lut WINNER candidate to 4-arm paired Modal T4 dispatch slot per T3 council #1335 RANKED ORDERING. |
| **#5 continual-learning posterior** | **ACTIVE** | Canonical equation #26 IN-DOMAIN context anchor appended event will land post-paired-Modal-T4 via `tac.canonical_equations.update_equation_with_empirical_anchor` per Catalog #344 (cls_stream byte cost + full-axis ΔS measurement). |
| **#6 probe-disambiguator** | **ACTIVE** | `test_inflate_v3_vs_v2_produces_different_frames_proves_cls_consumption` IS the canonical disambiguator between L0 SCAFFOLD (FAIL_AT_CLASS_1; v1/v2 cls=0 uniform) and L1 INTEGRATION (PASS_PER_CLASS; v3 cls_stream consumed) inflate states. The runtime witness is `bytes_v2 != bytes_v3` byte-level diff. |

---

## Sister continual-learning posterior anchor (Catalog #128 + #344)

**No new anchor written to canonical_equations_registry at this iteration** per `FORMALIZATION_PENDING` waiver in frontmatter. The cls_stream wire-in does NOT change the closed-form REPLACEMENT savings of canonical equation #26 IN-DOMAIN `nscs06_v8_chroma_lut` context (still 4064 bytes saved); the cls_stream cost is ADDITIVE and operator-routable to register either as a NEW canonical equation IF rate-vs-distortion frontier is learnable OR as a sister `anchor_appended` event on existing equation #26 (post-paired-Modal-T4 measurement). The wire-in landing is a structural unblocker for the canonical posterior update, not a posterior update itself.

---

## Operator-routable next step (EXPLICIT OPERATOR-DECISION REQUIRED per "all on MLX")

Per T3 council #1335 RANKED ORDERING + REVISIONS #1+#2 (now both satisfied: cls_stream wire-in LANDED): the canonical next step is the 4-arm paired Modal T4 CPU+CUDA auth_eval per Catalog #246 on the **stacked fec6 + NSCS06 v8 v3 archive** (baseline_cpu / baseline_cuda / procedural+cls_stream_cpu / procedural+cls_stream_cuda). Cost envelope ~$0.50-1.00 paid Modal.

**OPERATOR-DECISION REQUIRED**: Paired Modal IS a paid dispatch decision the operator must explicitly approve per operator's "Remember all on MLX" directive 2026-05-26. The wire-in slot does NOT auto-fire the dispatch per CLAUDE.md "Executing actions with care" + "Submission auth eval — BOTH CPU AND CUDA". Operator-routable choices:
- (a) **GREEN-LIGHT 4-arm paired Modal T4** ($0.50-1.00) per Catalog #246 to measure full-axis ΔS empirically. IF lands IN T3 council predicted band [-0.0027, -0.0015] → PR111 submission candidate. IF lands OUTSIDE +1.0 → DEFERRED-pending-research per Catalog #307 + invoke alternative reducers per Catalog #308 (per-temporal-window LUT / per-spatial-region LUT / hybrid-with-residual-overlay / cls_stream-conditioned LUT alternative arms per Catalog #308 sister) per Carmack MVP-first phasing REVISION #1.
- (b) **STAY MLX-LOCAL** and pursue downstream cls_stream arith-coding bytes-saving optimization (potential ~5-30 KB savings → -0.003 to -0.020 additional ΔS rate-axis lower-bound) BEFORE any paid dispatch.
- (c) **PIVOT to ranked candidate #2 (grayscale_lut procedural variant)** per T3 council #1335 RANKING per Hassabis REVISION #3 (paradigm-class interleaving for risk diversification).

**Discipline**: Catalog #229 PV + #117/#157/#174 canonical serializer (commit `581b7b129`) + POST-EDIT `--expected-content-sha256` on all 4 files + #206 (3 checkpoints emitted) + #110/#113 APPEND-ONLY (zero mutations to existing forensic artifacts; NEW files + APPEND-ONLY EXTEND on archive.py + inflate.py) + #287/#323 canonical Provenance (no score claim asserted) + #192/#317/#341 MLX-local non-promotable markers + #340 sister-checkpoint guard (cleared via self-checkpoint mark-complete-then-retry pattern) + #344 anchor-appended deferred to post-paired-Modal-T4 + #208 docs/local-paths (no /Users/adpena references) + #230 ownership map (zero collision with sister NIRVANA + BoostNeRV-PR110 L1 EMPIRICAL respawn slots; file-scope disjoint by substrate).
