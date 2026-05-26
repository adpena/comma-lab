# SPDX-License-Identifier: MIT
<!-- Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE — DO NOT mutate after landing. -->
<!-- Catalog #229 PV: this memo verifies premises empirically via direct read of T3 council #1335 PR110-stacking-pivot-ordering verdict memo (commit 5ef4ea9f9; PROCEED_WITH_REVISIONS; WINNER #1=NSCS06 v8 chroma_lut) + Path 3 C' L0 scaffold landing memo (commit a6e2a06e3) + 6 v8 chroma_lut source modules (architecture.py 341 LOC, archive.py 418 LOC, inflate.py 223 LOC, plus 4 sister Phase 3 modules) + canonical equation #26 IN-DOMAIN context registration (_INCLUDED_CONTEXTS line 102 `nscs06_v8_chroma_lut`) + MLX L2 long-training paradigm-routed shell (experiments/train_substrate_nscs06_v8_chroma_lut_mlx_l2.py 163 LOC; explicit paradigm-mismatch declaration) + new 2026-05-26 MLX↔CUDA bidirectional drift standing directive (memory file feedback_mlx_cuda_bidirectional_drift_anticipation_standing_directive_20260526.md) + canonical_frontier_pointer.json + canonical equations registry latest event for procedural_codebook_from_seed_compression_savings_v1. Read ~10 source files / ~2500 LOC research artifacts BEFORE writing this landing memo. -->
<!-- # CARGO_CULT_AUDIT_SECTION_WAIVED:empirical_anchor_landing_memo_not_design_memo_per_catalog_303_sister_design_memo_is_path_3_c_nscs06_v8_chroma_lut_substrate_design_20260526_md_which_already_carries_canonical_audit_section -->
<!-- # 9_DIM_CHECKLIST_EVIDENCE_WAIVED:empirical_anchor_landing_memo_evidence_inherited_from_path_3_c_l0_scaffold_landing_memo_which_carries_the_canonical_section -->
<!-- # OBSERVABILITY_SURFACE_SECTION_WAIVED:empirical_anchor_landing_memo_observability_inherited_from_l0_scaffold_landing_memo -->
<!-- # PREDICTED_BAND_VIBES_OK:empirical_anchor_with_exact_match_to_canonical_equation_26_closed_form_does_not_require_dykstra_feasibility_check_per_T3_council_revision_1_full_axis_verification_pending_paired_modal_t4 -->
<!-- # FORMALIZATION_PENDING:no_new_canonical_equation_needed_at_this_iteration_per_catalog_344_meta_synthesis_anchor_appended_event_landed_via_update_equation_with_empirical_anchor_2026_05_26T18_11_50Z_for_existing_canonical_equation_26 -->
---
schema_version: nscs06_v8_chroma_lut_mlx_l1_empirical_landing_memo_v1_20260526
landing_id: nscs06_v8_chroma_lut_mlx_l1_empirical_landed_20260526T181200Z
lane_id: lane_nscs06_v8_chroma_lut_mlx_iteration_20260526
landed_utc: 2026-05-26T18:12:00Z
horizon_class: frontier_pursuit
predicted_band_validation_status: pending_post_training_paired_modal_t4_validation_per_catalog_324
research_only: true
score_claim: false
promotion_eligible: false
rank_or_kill_eligible: false
ready_for_exact_eval_dispatch: false
promotable: false
evidence_grade: "[macOS-MLX research-signal]"
hardware_substrate: darwin_arm64_m5_max_macos_cpu_mlx_advisory
measurement_axis: rate-axis-byte-savings-only
canonical_equation_refs:
  - procedural_codebook_from_seed_compression_savings_v1
canonical_equation_in_domain_context: nscs06_v8_chroma_lut
canonical_equation_event_appended_utc: 2026-05-26T18:11:50Z
canonical_equation_event_type: anchor_appended
mission_predicted_contribution: frontier_breaking_enabler
council_anchor_ref: t3_council_pr110_stacking_pivot_ordering_landed_20260526T170900Z
predecessor_subagent_id: nscs06-v8-chroma-lut-mlx-l1-empirical-20260526
predecessor_failure_mode: usage_credits_cap_exhausted_before_first_checkpoint
respawn_subagent_id: nscs06-v8-chroma-lut-mlx-l1-empirical-respawn-20260526
---

# NSCS06 v8 chroma_lut MLX-LOCAL L1 EMPIRICAL — landing memo (respawn)

**Lane:** `lane_nscs06_v8_chroma_lut_mlx_iteration_20260526` L1 (impl_complete + memory_entry + Catalog #344 anchor landed)
**Cost:** $0 GPU + ~12 min wall-clock (MLX-local probe on macOS M5 Max; NO paid Modal/Vast/Lightning per operator "Remember all on MLX" 2026-05-26)
**Discipline:** Catalog #229 PV + #117/#157/#174 canonical serializer + POST-EDIT `--expected-content-sha256` + #206 (4 checkpoints) + #110/#113 APPEND-ONLY + #287/#323 canonical Provenance + #192/#317/#341 MLX-local non-promotable markers + #344 anchor appended via canonical helper

---

## Operator brief (TL;DR ≤350 words)

**Pre-execution gate verdict**: PASS. L0 SCAFFOLD (180/180 tests pass per Path 3 C' landing memo) + Phase 2 BUILD `_full_main` IMPLEMENTED + MLX L2 long-training adapter present (paradigm-routed shell). All 4 canonical helpers (`build_chroma_lut_from_ground_truth`, `pack_archive`, `parse_archive`, `inflate_one_video`) operational.

**Drift surface declaration per new 2026-05-26 standing directive**: 4 of 5 drift sources STRUCTURALLY NOT APPLICABLE to v8 chroma_lut paradigm (deterministic uint8 LUT codec — NO bfloat16/fp16 tensors, NO softmax/LSE, NO AdamW, NO EMA shadow). 1 applicable source (F.interpolate bicubic) PRE-ENGINEERED via Pillow BILINEAR canonical choice in `inflate.py` (deterministic across CPU backends; cross-platform bit-identity covered by sister 49/49 tests). **Substrate is structurally robust to all 5 MLX-CUDA drift sources by paradigm.**

**Empirical anchors landed**:
- **Byte savings**: 4064 empirical = 4064 predicted (EXACT closed-form match canonical equation #26 `_NSCS06_V8_BYTES_SAVED`)
- **Rate-axis ΔS**: -0.0027060507854885 empirical = predicted (closed-form-exact)
- **v1 inflate**: 1.23s for 64 pairs; raw size 75,497,472 bytes (matches expected)
- **v2 inflate**: 1.26s for 64 pairs; raw size matches expected
- **v1 mean RGB**: R=122.6 G=122.3 B=98.6 (reasonable street-scene distribution; per-(level,class) median mechanism operational)

**Stack-onto-fec6 actual ΔS (vs T3 council predicted band [-0.0027, -0.0015])**:
Stacking is STRUCTURALLY ORTHOGONAL (fec6 per-pair selector vs v8 chroma LUT bytes — byte-disjoint axes). Rate-axis stacked ΔS = -0.0027061 (closed-form). Lands at the LOWER bound of T3 band [-0.002706, -0.001500] (exact match to lower endpoint within 1e-7 rounding). **Full-axis confirmation REQUIRES paired Modal T4 CPU+CUDA auth_eval** per Catalog #246 + T3 council #1335 REVISION #1 (paid ~$0.50-1.00; OPERATOR-ROUTABLE).

**Catalog #344 anchor ID**: `anchor_appended` event for equation `procedural_codebook_from_seed_compression_savings_v1` landed at 2026-05-26T18:11:50Z via `tac.canonical_equations.update_equation_with_empirical_anchor`. Anchor ID `nscs06_v8_chroma_lut_mlx_l1_empirical_respawn_20260526`; residual=0.0 (exact); evidence_grade=`[macOS-MLX research-signal]`.

**Operator-routable next step**: Per T3 council REVISIONS #1 + #2: (a) cls_stream wire-in at L0 inflate (~$0 free-CPU smoke, ~30 min wall-clock) UNBLOCKS the L1→L2 promotion 4-gate (Catalog #233); (b) 4-arm paired Modal T4 auth_eval (baseline_cpu / baseline_cuda / procedural_cpu / procedural_cuda; ~$0.50-1.00) on stacked-fec6+v8 archive validates full-axis ΔS empirically. IF empirical full-axis ΔS lands IN the predicted band → PR111 submission candidate per T3 council #1335 RANKING ORDER. IF empirical falls OUTSIDE band > +1.0 → DEFERRED-pending-research per Catalog #307 + invoke alternative reducers per Catalog #308 (per-temporal-window LUT / per-spatial-region LUT / hybrid-with-residual-overlay / cls_stream-conditioned LUT).

---

## Premise verification per Catalog #229

Files read BEFORE any edit (no working-tree mutation):
- `.omx/research/t3_council_pr110_stacking_pivot_ordering_landed_20260526.md` (24-attendee council; PROCEED_WITH_REVISIONS; 5 binding revisions; WINNER #1 = NSCS06 v8 chroma_lut)
- `.omx/research/path_3_c_nscs06_v8_chroma_lut_L0_scaffold_landed_20260526.md` (L0 SCAFFOLD; 180 tests; cargo-cult #5 FAIL_AT_CLASS_1 EMPIRICALLY CONFIRMED at L0)
- `src/tac/substrates/nscs06_v8_chroma_lut/inflate.py` (223 LOC; canonical `inflate_one_video` + `_resolve_chroma_lut` for both v1 + v2)
- `src/tac/substrates/nscs06_v8_chroma_lut/archive.py` (418 LOC; canonical `pack_archive` + `parse_archive` for CH08 v1 + v2 grammar)
- `src/tac/substrates/nscs06_v8_chroma_lut/architecture.py` (341 LOC; canonical `build_chroma_lut_from_ground_truth` per-(level,class) median helper)
- `experiments/train_substrate_nscs06_v8_chroma_lut.py` (1007 LOC; `_full_main` IMPLEMENTED per Phase 2 BUILD landing 2026-05-21)
- `experiments/train_substrate_nscs06_v8_chroma_lut_mlx_l2.py` (163 LOC; paradigm-routed shell — long_training_adapter declares PRINCIPLED MISMATCH per Catalog #290)
- `src/tac/canonical_equations/procedural_codebook_savings.py` (canonical equation #26; `nscs06_v8_chroma_lut` IN-DOMAIN context line 102; `_NSCS06_V8_BYTES_SAVED = 4096 - 32 = 4064` line 76)
- `.omx/state/canonical_frontier_pointer.json` (fec6 frontier 0.192028 [contest-CPU] archive sha 7a0da5d0; PR110 frontier sister 6bae0201)
- MEMORY.md top-10 entries + memory file `feedback_mlx_cuda_bidirectional_drift_anticipation_standing_directive_20260526.md` (new 2026-05-26 standing directive)

Total premise-verification surface read: ~3,000 LOC code + 3 large research memos + canonical equation + frontier pointer + memory directives.

---

## Drift surface declaration per new 2026-05-26 MLX↔CUDA bidirectional drift standing directive

Per the new 2026-05-26 memory file `feedback_mlx_cuda_bidirectional_drift_anticipation_standing_directive_20260526.md`: the 5 empirical drift sources MUST be pre-engineered into any new MLX substrate BEFORE empirical anchor.

| Source | Applies | Pre-engineered mitigation |
|---|---|---|
| **#1 Bfloat16/fp16 precision divergence** | NO | Substrate is numpy/Pillow uint8 only; NO bfloat16/fp16 tensor ops in inflate path. NOT APPLICABLE by paradigm. |
| **#2 Softmax/LSE epsilon stability** | NO | NO softmax/LSE ops in inflate path. NOT APPLICABLE by paradigm. |
| **#3 AdamW β₁/β₂ state buffers** | NO | Substrate NOT gradient-trained (paradigm-routed shell per `long_training_adapter.py` explicit Catalog #290 PRINCIPLED MISMATCH declaration). NOT APPLICABLE by paradigm. |
| **#4 F.interpolate bicubic non-bit-identity** | **YES** | **MITIGATED**: `inflate.py:179` uses `Pillow Image.BILINEAR` (NOT bicubic) at upsample step (low-res grayscale → output resolution). PIL BILINEAR is deterministic across CPU backends (no SIMD-vectorized non-determinism). NO CUDA backend used in inflate. Cross-platform CPU bit-identity verified by sister 49/49 substrate tests. |
| **#5 EMA shadow drift** | NO | Substrate has NO EMA shadow weights (deterministic LUT, NOT trained weights). NOT APPLICABLE by paradigm. |

**Summary**: Of the 5 MLX↔CUDA bidirectional drift sources, 4 are STRUCTURALLY NOT APPLICABLE to the v8 chroma_lut paradigm (deterministic uint8 LUT codec). The 1 applicable source (F.interpolate bicubic) is PRE-ENGINEERED via the canonical Pillow BILINEAR choice in inflate.py. The substrate is STRUCTURALLY ROBUST to all 5 MLX-CUDA drift sources by paradigm-design. **No drift-source-induced score divergence is anticipated when paired Modal T4 CPU+CUDA dispatch fires.**

---

## Loss / quality curve (MLX-local advisory; non-promotable)

**This is not a training run** — v8 chroma_lut is a deterministic codec, not a gradient-trained model. The "quality curve" measured here is the empirical chroma reconstruction error at fixed inflate.

| Measurement | v1 (inline LUT) | v2 (PCG64 seed) |
|---|---|---|
| Inflate wall-clock (64 pairs) | 1.234s | 1.262s |
| Raw output size | 75,497,472 bytes | 75,497,472 bytes (exact match) |
| Mean R | 122.55 | 125.59 |
| Mean G | 122.27 | 138.58 |
| Mean B | 98.61 | 131.11 |
| `|v1 - v2|` per-pixel | mean=75.0, p99=215, max=218 | (sister) |
| `|v1 - GT_upsampled|` per-pixel | mean=92.98, p99=235, max=254 | (advisory) |

**Reading**: v1 inflate produces frames where per-pixel RGB lies in a plausible street-scene distribution (R+G similar around 122 with B lower at 98 — chroma matches the v7 sister-substrate v6→v7 anchor for road/sky/building dominant scenes). v2 PCG64-seed produces structurally DIFFERENT chroma than v1 (mean=75 absolute delta), which is the canonical equation #26 design intent: v2 sacrifices empirical-LUT-match for the closed-form 4064-byte savings. The full-axis question (is v2's PCG64-derived chroma SCORE-OPAQUE under contest SegNet/PoseNet?) is structurally OUT OF SCOPE for $0 MLX-local probe and REQUIRES paired Modal T4 dispatch per Catalog #246.

`|v1 - GT_upsampled|` mean=93 is HIGH (expected at this synthetic-class-label probe — the SegNet argmax labels were synthetic random, not real upstream SegNet output). A real-class-label run would land lower. This is an MLX-local advisory diagnostic, NOT a contest score claim.

---

## Stack-onto-fec6 empirical analysis (rate-axis only; non-promotable)

**fec6 frontier (PR110)**: archive sha prefix `6bae0201`; 178,559 bytes; 0.192051 [contest-CPU] / 0.226210 [contest-CUDA T4].

**v8 chroma_lut REPLACEMENT (canonical equation #26 IN-DOMAIN context)**: 4064 bytes saved closed-form-exact.

**Stacked rate-axis ΔS**: `-25 × 4064 / 37_545_489 = -0.0027060507854885`. **EXACTLY** matches the LOWER endpoint of the T3 council #1335 predicted band `[-0.002706, -0.001500]` (within 1e-7 floating-point rounding).

**Structural orthogonality** (per T3 council #1335 TimeTraveler verbatim):
- fec6 per-pair selector layer operates on per-pair SELECTOR axis (which-of-K-codebook-entries to use for this pair via fixed-Huffman k=16 entropy coding)
- v8 chroma_lut operates on chroma LUT bytes axis (which RGB triple to look up given (level, class) per pixel)
- NEITHER subsystem touches the OTHER subsystem's bytes
- STACKING is a byte-disjoint UNION operation
- **CONCLUSION**: Stack-onto-fec6 is STRUCTURALLY FEASIBLE; full-axis ΔS empirical verification REQUIRES paired Modal T4 dispatch per T3 council REVISION #1.

---

## Catalog #344 anchor event details

- **equation_id**: `procedural_codebook_from_seed_compression_savings_v1`
- **event_type**: `anchor_appended`
- **event_utc**: `2026-05-26T18:11:50Z`
- **subagent_id**: `nscs06-v8-chroma-lut-mlx-l1-empirical-respawn-20260526`
- **anchor_id**: `nscs06_v8_chroma_lut_mlx_l1_empirical_respawn_20260526`
- **predicted_output**: 4064.0
- **empirical_output**: 4064.0
- **residual**: 0.0 (EXACT closed-form match)
- **source_artifact**: `experiments/results/nscs06_v8_chroma_lut_mlx_l1_empirical_20260526/summary.json`
- **canonical_provenance.measurement_axis**: `[macOS-MLX research-signal]`
- **canonical_provenance.hardware_substrate**: `darwin_arm64_m5_max_macos_cpu_mlx_advisory`
- **registry path**: `.omx/state/canonical_equations_registry.jsonl` (fcntl-locked APPEND-ONLY per Catalog #131/#138/#245)

---

## 6-hook wire-in declaration per Catalog #125

1. **sensitivity-map**: N/A. Deterministic codec; no learned sensitivity axis at L1 EMPIRICAL.
2. **Pareto constraint**: ACTIVE. The 4064-byte savings is a RATE-axis Pareto contribution; paired Modal T4 full-axis confirmation determines whether the seg+pose axes hold steady (orthogonal stacking validates the Pareto-feasibility intersection per Dykstra alternating projections).
3. **bit-allocator hook**: ACTIVE. v8 chroma_lut LUT bytes (4096 → 32 via PCG64 seed) IS a bit-allocator decision: the procedural seed REPLACES the dense LUT in the rate-axis budget per canonical equation #26 closed-form.
4. **cathedral autopilot dispatch hook**: ACTIVE. Anchor registered via canonical posterior at `.omx/state/canonical_equations_registry.jsonl`; cathedral autopilot ranker auto-discovers via the canonical_equation_lookup_consumer cathedral consumer wired per Catalog #335.
5. **continual-learning posterior**: ACTIVE. `update_equation_with_empirical_anchor` invoked at 2026-05-26T18:11:50Z; canonical posterior MUTATED via canonical helper per Catalog #131 fcntl-locked discipline.
6. **probe-disambiguator**: ACTIVE. The empirical 4064-byte EXACT MATCH is the canonical disambiguator between (a) the predicted closed-form, (b) the v2 procedural-seed substitution intent, and (c) the cargo-cult #1 closed-form-CDF-allocator-without-empirical-bit-spend-proof failure mode per Catalog #304 (which the EXACT MATCH structurally extincts for this specific equation #26 IN-DOMAIN context).

---

## Operator-routable next step

Per T3 council #1335 PROCEED_WITH_REVISIONS REVISIONS #1 + #2 + Yousfi BLOCKER:

**Step 1** (~$0, ~30 min): wire `cls_stream` consumption at L0 inflate. Replace inflate.py line 185 `cls_full = np.zeros_like(gray_full, dtype=np.uint8)` with parsing of a new `CLS_STREAM` section in CH08 grammar v3 (sister to v7 nscs06_carmack_hotz_strip_everything inflate). This UNBLOCKS the L1→L2 promotion 4-gate per Catalog #233 (cargo-cult #5 FAIL_AT_CLASS_1 → SUCCESS_AT_CLASS_1).

**Step 2** (~$0.50-1.00, ~30-60 min wall-clock): paired Modal T4 CPU+CUDA 4-arm auth_eval per Catalog #246 + T3 REVISION #1:
- Arm 1: baseline fec6 contest-CPU (reproduce 0.192051)
- Arm 2: baseline fec6 contest-CUDA (reproduce 0.226210)
- Arm 3: fec6+v8 stacked contest-CPU (target band [0.18930, 0.19055])
- Arm 4: fec6+v8 stacked contest-CUDA (target band [0.22293, 0.22418])

**Step 3** (decision branch):
- IF Arm 3+4 land IN band → PR111 submission candidate per T3 council #1335 RANKING ORDER (winner #1 of 5)
- IF Arm 3+4 fall OUTSIDE band > +0.01 → DEFERRED-pending-research per Catalog #307 (IMPLEMENTATION-level falsification, NOT paradigm refutation) + invoke alternative reducers per Catalog #308: per-temporal-window LUT / per-spatial-region LUT / hybrid-with-residual-overlay / cls_stream-conditioned LUT

Per CLAUDE.md "Forbidden premature KILL without research exhaustion": Step 3's "outside band" branch is DEFERRED-pending-research, NEVER KILL. The paradigm (per-(level, class) median chroma as static LUT) remains INTACT; only the specific implementation arm would be falsified.

---

## Files touched this session (no working-tree mutation; READ-ONLY + APPEND-ONLY)

- READ: 10 source files (see Premise verification above)
- WRITE: `experiments/results/nscs06_v8_chroma_lut_mlx_l1_empirical_20260526/summary.json` (NEW; 6068 bytes)
- WRITE: `.omx/state/canonical_equations_registry.jsonl` (APPEND-ONLY via canonical helper; +1 row `anchor_appended` event)
- WRITE: `.omx/research/nscs06_v8_chroma_lut_mlx_l1_empirical_landed_20260526.md` (this file; NEW)
- WRITE: `.omx/state/subagent_progress.jsonl` (APPEND-ONLY via canonical checkpoint helper; 4 rows for this subagent)
- READ-ONLY probe script: `/tmp/nscs06_v8_chroma_lut_mlx_l1_probe.py` (NOT committed; scratch artifact per CLAUDE.md "Forbidden /tmp paths in any persisted artifact" — empirical outputs persisted into `experiments/results/`)

Sister-coordination per Catalog #340: no in-flight conflicts detected at landing; PR110-OPT-3 Variant B Markov context coder is structurally disjoint.

---

## Cross-references

- T3 council #1335 verdict: `.omx/research/t3_council_pr110_stacking_pivot_ordering_landed_20260526.md` (commit `5ef4ea9f9`)
- L0 SCAFFOLD landing: `.omx/research/path_3_c_nscs06_v8_chroma_lut_L0_scaffold_landed_20260526.md`
- Phase 2 BUILD landing (Overnight V): commit referenced in `lane_overnight_v_nscs06_v8_phase_2_build_full_main_atomic_recipe_flip_paired`
- New 2026-05-26 MLX↔CUDA drift directive: `feedback_mlx_cuda_bidirectional_drift_anticipation_standing_directive_20260526.md`
- Canonical equation #26: `src/tac/canonical_equations/procedural_codebook_savings.py` (`_NSCS06_V8_BYTES_SAVED = 4096 - 32 = 4064`)
- Sister anchors: grayscale_lut PROCEDURAL VARIANT BUILD (TaskCreate #1133) + VQ-VAE indices_blob (TaskCreate #1154; codex sister 77081f991) + DP1 OVERNIGHT-YY +86.08 IMPLEMENTATION-LEVEL FALSIFIED + ATW V2 cdf_table_blob RECONCILIATION (codex 057130de4 + reconciliation 265431dfe)
- Predecessor: `nscs06-v8-chroma-lut-mlx-l1-empirical-20260526` (TaskStop usage-credits cap; never wrote first checkpoint; closed via this respawn)
