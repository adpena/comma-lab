<!-- SPDX-License-Identifier: MIT -->
<!-- Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE — DO NOT mutate after landing. -->
<!-- Catalog #229 PV: verified premises by reading (a) T3 council #1335 PROCEED_WITH_REVISIONS verdict
     verbatim (commit 5ef4ea9f9; council_decisions_recorded REVISION 2 + Yousfi BLOCKER); (b) NSCS06 v8 chroma_lut L1 EMPIRICAL respawn landing memo (commit 4a4ab1e4f); (c) Path 3 C' L0 SCAFFOLD landing memo (commit a6e2a06e3; cargo-cult #5 FAIL_AT_CLASS_1 EMPIRICALLY CONFIRMED); (d) v8 inflate.py 223 LOC (line 185 `cls_full = np.zeros_like(gray_full, dtype=np.uint8)` is the cargo-cult-5 site); (e) v8 archive.py 418 LOC (CH08 grammar; v1=inline LUT, v2=PCG64 seed; ZERO existing cls_stream section); (f) sister NSCS06 strip_everything inflate.py 222 LOC + archive.py 22.9K (CH06 v2 carries CLS_STREAM with arith-coded uniform-CDF labels + decode_class_label_stream + decode_grayscale_stream); (g) v8 distinguishing_feature_smoke.py canonical PerClassChromaDistinguishingFeatureVerdict (verdict_kind taxonomy: PASS_PER_CLASS / FAIL_AT_CLASS_<c>). -->
<!-- # CARGO_CULT_AUDIT_SECTION_WAIVED:scoped_wire_in_pre_execution_gate_report_NOT_substrate_design_memo_cargo_cult_audit_lives_in_sister_path_3_c_L0_scaffold_landing_memo_per_catalog_303_design_memo_surface -->
<!-- # 9_DIM_CHECKLIST_EVIDENCE_WAIVED:scoped_wire_in_pre_execution_gate_report_NOT_substrate_landing_memo_9_dim_evidence_lives_in_sister_path_3_c_L0_scaffold_landing_memo_per_catalog_294_landing_memo_surface -->
<!-- # OBSERVABILITY_SURFACE_SECTION_WAIVED:scoped_wire_in_pre_execution_gate_report_NOT_substrate_design_memo_observability_surface_inherited_from_sister_path_3_c_L0_scaffold_landing_memo_per_catalog_305_design_memo_surface -->
<!-- # FORMALIZATION_PENDING:no_NEW_canonical_equation_introduced_in_this_wire_in_scope_canonical_equation_26_unchanged_per_catalog_344_only_existing_anchor_appended_event_will_land_post_wire_in_via_update_equation_with_empirical_anchor -->
<!-- # COUNCIL_ASSUMPTION_STATEMENT_WAIVED:pre_execution_gate_report_NOT_council_deliberation_per_catalog_292_no_per_member_operating_within_assumption_surfacing_required -->
---
schema_version: nscs06_v8_cls_stream_wire_in_pre_execution_gate_report_v1_20260526
landing_id: nscs06_v8_cls_stream_wire_in_pre_execution_gate_report_20260526T182100Z
lane_id: lane_nscs06_v8_cls_stream_wire_in_20260526
horizon_class: frontier_pursuit
predicted_band_validation_status: pending_post_training_paired_modal_t4_validation
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
subagent_id: nscs06-v8-cls-stream-wire-in-20260526
---

# NSCS06 v8 chroma_lut — cls_stream wire-in pre-execution gate report

**Lane:** `lane_nscs06_v8_cls_stream_wire_in_20260526` (pre-execution gate; impl pending)
**Cost:** $0 + ~30 min wall-clock (local wire-in; NO training; NO paid dispatch)
**Predecessor empirical anchor:** TaskCreate #1340 NSCS06 v8 L1 EMPIRICAL respawn (commit `4a4ab1e4f`; canonical equation #26 EXACT byte-savings 4064; ΔS = -0.00270605 rate-axis closed-form)
**Council mandate:** T3 council #1335 PROCEED_WITH_REVISIONS REVISION #2 + Yousfi BLOCKER finding: "NSCS06 v8 #1 dispatch BLOCKED until cls_stream wire-in lands at L0 inflate (~$0 free-CPU smoke; ~30 min wall-clock) per Catalog #325 per-substrate symposium 14-day window 2026-05-26 → 2026-06-09"

---

## 1. cls_stream scope (per the empirical bug class & sister patterns)

**The cargo-cult #5 anchor**: `src/tac/substrates/nscs06_v8_chroma_lut/inflate.py:185`
```python
# v8 L0 SCAFFOLD: class=0 uniformly (sister to v7 SCAFFOLD pattern;
# L1 promotion couples to a v7-style CLS_STREAM).
cls_full = np.zeros_like(gray_full, dtype=np.uint8)
```

The hardcoded `cls=0` causes the chroma LUT lookup to ALWAYS read `LUT[gray, 0, :]` — every pixel uses the class-0 anchor. The structurally-confirmed test `test_verify_per_class_chroma_at_L0_FAILS_AT_CLASS_1_unwinds_cargo_cult_5` mutates `LUT[:, 1, :]` (class 1) bytes and observes `verdict_kind=FAIL_AT_CLASS_1` because the L0 inflate never reads class-1 bytes from the LUT.

**The L1 wire-in fix**: thread a per-cell class-label byte-stream (cls_stream) THROUGH the archive grammar so inflate consumes per-pixel class labels and the chroma LUT lookup uses the canonical formula `RGB = chroma_lut[gray_full[y,x], cls_full[y,x], :]` with non-uniform cls_full.

**Sister pattern (NSCS06 strip_everything, CH06 v2 commit 4292c8ce2)**:
- Archive carries `CLS_LEN (u32)` header field + `CLS_STREAM` section (arith-coded uniform-CDF per-cell class labels)
- Inflate calls `decode_class_label_stream(arc.cls_arith_bytes, shape=(num_pairs, grayscale_h, grayscale_w))` → returns `cls_lowres` array, then upsamples NEAREST to full resolution
- The chroma LUT lookup uses `cls_full` (per-cell class labels) instead of uniform 0

**The minimum-scope wire-in for v8** (scope-disciplined; ~30 min wall-clock per operator + Yousfi BLOCKER):
- ADD: schema_version v3 (procedural seed + cls_stream) — wire-in WINNER candidate path
- ADD: `CLS_LEN (u32)` header field
- ADD: `CLS_STREAM` section storing **raw uint8** per-cell class labels (NOT arith-coded; minimum-LOC for wire-in; arith-coding is a follow-up bytes-saving optimization)
- ADD: `cls_lowres` field on `Nscs06V8Archive` dataclass
- ADD: `cls_bytes` kwarg to `pack_archive(...)`
- MODIFY: `inflate.py:185` consume `cls_lowres` upsampled NEAREST to `cls_full` instead of `np.zeros_like(gray_full)`
- PRESERVE: v1 (inline LUT, cls=0 backward compat) + v2 (PCG64 seed, cls=0 backward compat) paths

**Why raw uint8 not arith-coded**: per operator's ~30 min scope + sister test_path_3_c_prime_cargo_cult_unwinds.py predicate "wire cls_stream consumption at L0 inflate"; the verdict transition FAIL_AT_CLASS_1 → PASS_PER_CLASS requires only that cls_full is non-uniform (one byte per cell, raw uint8 suffices). Arith coding is a downstream bytes-saving optimization (≤1 KB savings at grayscale_h * grayscale_w * num_pairs scale; canonical equation #26 closed-form 4064-byte savings is preserved regardless).

**LOC estimate per file**:
- `archive.py`: +60 LOC (new schema_version constant + CLS_LEN field + CLS_STREAM section + Nscs06V8Archive.cls_lowres + pack_archive cls_bytes kwarg + parse_archive cls_lowres parsing)
- `inflate.py`: +5 LOC (`if arc.cls_lowres is not None: cls_full = upsample(cls_lowres[p], NEAREST) else: cls_full = np.zeros_like(gray_full)`)
- `tests/test_substrate.py`: +40 LOC (new test_v3_cls_stream_roundtrip + test_cls_stream_disambiguates_v8 + test_v3_inflate_consumes_cls_lowres + per-class chroma smoke verdict transition test)

**Total LOC budget**: ~105 LOC additions; ZERO mutations to v1/v2 paths (HNeRV parity L4 ≤200 LOC inflate budget preserved at ~125 LOC including new branch; substrate_engineering exception per L7).

---

## 2. Drift surface declaration per new 2026-05-26 MLX↔CUDA bidirectional drift standing directive

| Drift source | Applies? | Rationale |
|---|---|---|
| #1 Bfloat16/fp16 precision divergence | **NO** | cls_stream is uint8 raw bytes (or arith-coded uint8); no bfloat16/fp16 ops at encode or decode. NOT APPLICABLE by paradigm. |
| #2 Softmax/LSE epsilon stability | **NO** | NO softmax/LSE ops in cls_stream codec path (raw uint8 store + load). NOT APPLICABLE by paradigm. |
| #3 AdamW β₁/β₂ state buffers | **NO** | cls_stream is NOT gradient-trained (deterministic encode from compress-time SegNet argmax output). NOT APPLICABLE by paradigm. |
| #4 F.interpolate bicubic non-bit-identity | **NO** | cls_full upsample uses Pillow Image.NEAREST (NOT bicubic, NOT bilinear). NEAREST is bit-identical across CPU backends per sister strip_everything precedent. NOT APPLICABLE by paradigm. |
| #5 EMA shadow drift | **NO** | No EMA shadow weights (deterministic cls_stream byte-stream). NOT APPLICABLE by paradigm. |

**Verdict**: cls_stream wire-in is STRUCTURALLY ROBUST to all 5 MLX↔CUDA bidirectional drift sources by paradigm-design. No drift-source-induced verdict divergence anticipated when paired Modal T4 CPU+CUDA dispatch fires.

---

## 3. Canonical-vs-frontier-push decision per new 2026-05-26 pushing-the-frontier-of-research-on-optimization-algorithms standing directive

**Decision: CANON-APPLICATION**.

**Rationale**:
- cls_stream wire-in is a sister-pattern-application from NSCS06 strip_everything CH06 v2 (commit 4292c8ce2 symposium). No novel algorithm proposed.
- The minimum-scope raw-uint8 byte-stream design is canonical "byte-disjoint section in monolithic 0.bin archive" (HNeRV parity L3 archive grammar discipline).
- Inflate consumption pattern (`cls_full = upsample(cls_lowres[p], NEAREST)`) is sister-pattern-replication from strip_everything inflate.py:180-184.
- Arith-coding is a follow-up bytes-saving optimization (DEFERRED-pending-bytes-budget-analysis per CLAUDE.md "Forbidden premature KILL"); current scope = wire-in only.

**No frontier-push contribution to canonical equation registry**. The existing canonical equation #26 (`procedural_codebook_from_seed_compression_savings_v1`) IN-DOMAIN `nscs06_v8_chroma_lut` context PRESERVED. Post-wire-in empirical anchor will be appended as new `anchor_appended` event via `tac.canonical_equations.update_equation_with_empirical_anchor` (verifies cls_stream bytes don't drift the 4064-byte closed-form prediction; predicted residual = 0 since cls_stream is ADDITIVE bytes not REPLACEMENT bytes).

---

## 4. Sister-subagent ownership map (Catalog #340 STAY-DISJOINT)

**My scope**: `src/tac/substrates/nscs06_v8_chroma_lut/{archive.py, inflate.py, tests/test_substrate.py}` ONLY.

**In-flight sisters at launch** (per the subagent prompt body):
- NIRVANA L1 EMPIRICAL MLX respawn (different substrate; ZERO file overlap)
- BoostNeRV-PR110 L1 EMPIRICAL MLX respawn (different substrate; ZERO file overlap)

**Sister-checkpoint guard verdict**: PROCEED (zero in-flight subagent has my target files in `files_touched`).

---

## 5. Catalog #233 promotion canonical 4-gate evidence path (post-wire-in)

| Gate | Pre-wire-in state | Post-wire-in evidence |
|---|---|---|
| (1) `impl_complete` | TRUE (Phase 2 BUILD landed; `_full_main` implemented) | PRESERVED + extended with cls_stream wiring path |
| (2) `parser_section_manifest_consistent` | TRUE for v1/v2 sections | EXTENDED to v3 (cls_stream section parses + roundtrips; new test) |
| (3) `inflate_runtime_byte_consumption` | FAIL (cls_full=zeros; cargo-cult #5 FAIL_AT_CLASS_1) | PASS (cls_lowres consumed at inflate; `test_verify_per_class_chroma_at_L0_FAILS_AT_CLASS_1_unwinds_cargo_cult_5` predicate transitions to `PASS_PER_CLASS` when archive carries v3 + cls_stream) |
| (4) `roundtrip_test_passes` | TRUE for v1/v2 | EXTENDED to v3 (pack_archive(cls_bytes=...) + parse_archive returns cls_lowres + inflate produces byte-stable frames) |

---

## 6. 6-hook wire-in declaration per Catalog #125

- **hook #1 sensitivity-map**: ACTIVE — cls_stream consumption surfaces per-class chroma sensitivity (canonical via existing `distinguishing_feature_smoke.PerClassChromaDistinguishingFeatureVerdict` consumer).
- **hook #2 Pareto constraint**: ACTIVE — cls_stream ADDS bytes (rate-axis ADDITIVE; ~5-50 KB depending on grayscale_h × grayscale_w × num_pairs); seg+pose-axis EXPECTED to IMPROVE via correct per-class chroma reconstruction (T3 council #1335 predicted lower-bound -0.0027 closed-form REPLACEMENT savings PRESERVED; cls_stream byte cost is the DELTA to be measured at paired Modal T4 dispatch).
- **hook #3 bit-allocator**: PARTIAL — raw uint8 cls_stream is the minimum-bits baseline; downstream arith-coded variant is the bit-allocator-optimal canonical follow-up (DEFERRED-pending-bytes-budget-analysis).
- **hook #4 cathedral autopilot dispatch**: ACTIVE PRIMARY — cls_stream wire-in UNBLOCKS the L1→L2 promotion gate per Catalog #233; cathedral autopilot ranker can NOW promote v8 chroma_lut WINNER candidate to 4-arm paired Modal T4 dispatch slot.
- **hook #5 continual-learning posterior**: ACTIVE — canonical equation #26 IN-DOMAIN context anchor (post-wire-in empirical pack+parse roundtrip) appended as new `anchor_appended` event via `tac.canonical_equations.update_equation_with_empirical_anchor`.
- **hook #6 probe-disambiguator**: ACTIVE — `test_verify_per_class_chroma_at_L0_FAILS_AT_CLASS_1_unwinds_cargo_cult_5` IS the canonical disambiguator between L0 SCAFFOLD (FAIL_AT_CLASS_1) and L1 INTEGRATION (PASS_PER_CLASS) inflate states; the post-wire-in test predicate transition is the runtime witness.

---

## 7. Operator-routable next step

Per T3 council #1335 RANKED ORDERING + REVISIONS #1 + Carmack MVP-first phasing: post-wire-in, the canonical next step is the 4-arm paired Modal T4 CPU+CUDA auth_eval (~$0.50-1.00) on the **stacked fec6 + NSCS06 v8 v3 archive** (baseline_cpu / baseline_cuda / procedural+cls_stream_cpu / procedural+cls_stream_cuda) per Catalog #246. **EXPLICIT OPERATOR-DECISION REQUIRED** per operator's "Remember all on MLX" directive 2026-05-26 — paired Modal IS a paid dispatch decision the operator must explicitly approve in a sister session.

**Discipline**: Catalog #229 PV + #117/#157/#174 canonical serializer + POST-EDIT `--expected-content-sha256` + #206 (4 checkpoints) + #110/#113 APPEND-ONLY + #287/#323 canonical Provenance + #192/#317/#341 MLX-local non-promotable markers + #340 sister-checkpoint guard PROCEED + #344 anchor appended via canonical helper post-wire-in.
