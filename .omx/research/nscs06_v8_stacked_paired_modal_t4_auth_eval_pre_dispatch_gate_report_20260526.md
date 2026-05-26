<!-- SPDX-License-Identifier: MIT -->
<!-- Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE — DO NOT mutate after landing. -->
<!-- Catalog #229 PV: this report verifies pre-dispatch state empirically via direct read of v8 trainer + cls_stream wire-in landing memo + L1 EMPIRICAL landing memo + canonical operator-authorize recipe + paired dispatcher CLI before any paid Modal dispatch. -->
<!-- # CARGO_CULT_AUDIT_SECTION_WAIVED:pre_dispatch_gate_report_NOT_substrate_design_memo_per_catalog_303_sister_design_memo_chain_is_path_3_c_L0_scaffold_plus_phase_2_BUILD_landing_memo -->
<!-- # 9_DIM_CHECKLIST_EVIDENCE_WAIVED:pre_dispatch_gate_report_evidence_inherited_from_chain_path_3_c_L0_scaffold_landing_memo_per_catalog_294 -->
<!-- # OBSERVABILITY_SURFACE_SECTION_WAIVED:pre_dispatch_gate_report_observability_inherited_from_phase_2_build_landing_memo_per_catalog_305 -->
<!-- # PREDICTED_BAND_VIBES_OK:pre_dispatch_gate_report_no_NEW_predicted_band_introduced_T3_council_band_already_anchored_per_catalog_296 -->
<!-- # FORMALIZATION_PENDING:no_NEW_canonical_equation_needed_pre_dispatch_gate_report_canonical_equation_26_already_registered_per_catalog_344 -->
<!-- # COUNCIL_ASSUMPTION_STATEMENT_WAIVED:pre_dispatch_gate_report_NOT_council_deliberation_per_catalog_292 -->
---
schema_version: nscs06_v8_stacked_paired_modal_t4_auth_eval_pre_dispatch_gate_report_v1_20260526
report_id: nscs06_v8_stacked_paired_modal_t4_auth_eval_pre_dispatch_gate_report_20260526T183633Z
lane_id: lane_nscs06_v8_stacked_paired_modal_t4_auth_eval_20260526
report_utc: 2026-05-26T18:36:33Z
horizon_class: frontier_pursuit
predicted_band_validation_status: pending_post_training_paired_modal_t4_validation_per_catalog_324
research_only: true
score_claim: false
promotion_eligible: false
rank_or_kill_eligible: false
ready_for_exact_eval_dispatch: false
promotable: false
evidence_grade: "[structural-verifier]"
hardware_substrate: darwin_arm64_m5_max_macos_cpu_pre_dispatch_audit
measurement_axis: pre-dispatch-gate-verdict-only
canonical_equation_refs:
  - procedural_codebook_from_seed_compression_savings_v1
canonical_equation_in_domain_context: nscs06_v8_chroma_lut
mission_predicted_contribution: frontier_protecting
council_anchor_ref: t3_council_pr110_stacking_pivot_ordering_landed_20260526T170900Z
predecessor_landing_refs:
  - nscs06_v8_chroma_lut_mlx_l1_empirical_landed_20260526T181200Z
  - nscs06_v8_cls_stream_wire_in_landed_20260526T183100Z
subagent_id: nscs06-v8-stacked-paired-modal-t4-auth-eval-20260526
verdict: BLOCKED_OPERATOR_DECISION_REQUIRED
blocker_id: TRAINER_V3_WIRE_IN_MISSING
---

# NSCS06 v8 stacked-paired Modal T4 4-arm auth_eval — pre-dispatch gate report

**Lane:** `lane_nscs06_v8_stacked_paired_modal_t4_auth_eval_20260526` L1 (pre-dispatch audit; NO paid dispatch fired)
**Cost:** $0 GPU + ~12 min wall-clock (pre-dispatch audit only)
**Discipline:** Catalog #229 PV (read full pre-dispatch state) + #117/#157/#174 canonical serializer + POST-EDIT `--expected-content-sha256` + #206 (2 checkpoints emitted) + #110/#113 APPEND-ONLY + #287/#323 canonical Provenance (no score claim asserted) + CLAUDE.md "Executing actions with care" (NO paid dispatch fired pending operator decision on TRAINER_V3_WIRE_IN_MISSING blocker)

---

## Verdict

**`BLOCKED_OPERATOR_DECISION_REQUIRED`** — pre-dispatch gate FAILS with one BLOCKER: `TRAINER_V3_WIRE_IN_MISSING`. Dispatching the canonical operator-authorize recipe NOW would emit a CH08 v2 archive (`cls_full=zeros` at inflate per cargo-cult #5 — the L0 SCAFFOLD `FAIL_AT_CLASS_1` failure mode), structurally NOT the v3 stacked archive T3 council #1335 ranked as WINNER #1.

**Recommended operator-routable next step**: route a TRAINER-V3-WIRE-IN sister subagent to extend `experiments/train_substrate_nscs06_v8_chroma_lut.py::_full_main` (and optionally `_smoke_main`) to invoke `pack_archive(..., cls_bytes=...)` per the CH08 v3 schema BEFORE this slot proceeds to paid Modal T4 dispatch.

**Cost saved by HALT**: ~$0.50-1.00 paid Modal T4 4-arm dispatch + 30-60 min wall-clock that would have produced a v2 archive auth_eval anchor (cargo-cult #5 active; FAIL_AT_CLASS_1 from L0 SCAFFOLD EMPIRICAL CONFIRMATION).

---

## Pre-dispatch state audit (per Catalog #229 PV)

### Surface 1: cls_stream wire-in tests (Catalog #233 Gate 3 structural proof)

Reproducer: `PYTHONPATH=src:upstream:$PWD .venv/bin/python -m pytest src/tac/substrates/nscs06_v8_chroma_lut/tests/test_cls_stream_wire_in.py -v`

**Result**: **17/17 PASS** including:
- `test_inflate_v3_vs_v2_produces_different_frames_proves_cls_consumption` PASS
- `test_v3_header_size_invariant` PASS (39 bytes)
- `test_parse_archive_v3_cls_lowres_byte_stable_roundtrip` PASS
- `test_inflate_v3_with_uniform_class_matches_v2` PASS (boundary invariant)

**Verdict**: cls_stream wire-in at archive.py + inflate.py is **OPERATIONAL** per Catalog #233 Gate 3 (inflate_runtime_byte_consumption) structurally.

### Surface 2: canonical operator-authorize recipe

Path: `.omx/operator_authorize_recipes/substrate_nscs06_v8_chroma_lut_modal_t4_dispatch.yaml`

Key fields:
- `platform: modal`; `gpu: T4`; `min_vram_gb: 16`; `min_smoke_gpu: T4`
- `dispatch_enabled: true`; `research_only: false`
- `dispatch_blockers: []`
- `cost_band: {epochs: 1, hand_calibrated_fallback_p50_usd: 0.50}`
- `predicted_delta: -0.002706 [prediction; canonical-equation-26-grounded; per-substrate-symposium-pending]`
- `predicted_band_validation_status: pending_post_training`
- `lane_script: scripts/remote_lane_substrate_nscs06_v8_chroma_lut.sh`
- `trainer_path: experiments/train_substrate_nscs06_v8_chroma_lut.py`
- `env_overrides: {NSCS06_V8_TRAINER_MODE: full, NSCS06_V8_DEVICE: cuda, NSCS06_V8_EPOCHS: 1, SMOKE_ONLY: 0}`
- `per_substrate_symposium_memo: .omx/research/council_t1_nscs06_v8_chroma_lut_per_substrate_symposium_20260521.md`
- `per_substrate_symposium_verdict: PROCEED_WITH_REVISIONS`
- `per_substrate_symposium_window_end_utc: 2026-06-04T00:00:00Z` (still in window; today 2026-05-26)
- `canonical_equation_in_domain_context: nscs06_v8_chroma_lut` (Catalog #344 registered)

**Verdict**: Recipe is **STRUCTURALLY DISPATCH-READY** per Catalog #240 recipe-vs-trainer-state consistency + Catalog #325 per-substrate symposium within 14-day window + Catalog #324 predicted_band post-training validation declared.

### Surface 3: trainer archive emission path (THE BLOCKER)

Audit of `experiments/train_substrate_nscs06_v8_chroma_lut.py`:

- `_full_main` line 740: `pack_archive(... chroma_seed=seed)` — **v2 path** (no `cls_bytes` kwarg)
- `_full_main` line 754: `pack_archive(... chroma_lut=chroma_lut)` — **v1 path** (no `cls_bytes` kwarg)
- `_smoke_main` line 517: `pack_archive(... chroma_lut=lut)` — **v1 path** (no `cls_bytes` kwarg)
- `_smoke_main` line 525: `pack_archive(... chroma_seed=seed)` — **v2 path** (no `cls_bytes` kwarg)

`grep -n "cls_stream\|cls_bytes\|CH08_SCHEMA_VERSION_PROCEDURAL_SEED_WITH_CLS_STREAM\|v3_procedural" experiments/train_substrate_nscs06_v8_chroma_lut.py` returns ZERO MATCHES.

**Verdict**: Trainer is **NOT WIRED to the CH08 v3 schema** despite the archive.py + inflate.py extension landing in commit `581b7b129`. Dispatching the recipe NOW would emit a v2 procedural-seed archive whose `cls_full=zeros` at inflate triggers cargo-cult #5 (`FAIL_AT_CLASS_1`) per the Path 3 C' L0 SCAFFOLD landing memo's EMPIRICAL CONFIRMATION.

### Surface 4: Catalog #246 paired dispatcher CLI

`tools/dispatch_modal_paired_auth_eval.py` is the canonical 4-arm paired Modal CPU+CUDA dispatcher. It requires:
- `--archive <path>` (target archive bytes)
- `--submission-dir <path>` (runtime tree with inflate.sh)
- `--label <label>` + `--gpu T4` + `--execute`
- Honors `--skip-axis-if-promotable-anchor-exists` per Catalog #246 PAIRED-DISPATCH-SKIP-IF-ANCHOR-EXISTS-ENHANCEMENT 2026-05-15

The dispatcher expects an EXISTING archive to dispatch — it does NOT build the archive itself. The canonical archive-build path is invoking the trainer first (via `tools/operator_authorize.py --recipe substrate_nscs06_v8_chroma_lut_modal_t4_dispatch`), then dispatching the resulting archive paired.

### Surface 5: stacking semantics

Per the L1 EMPIRICAL landing memo paragraph 121 + T3 council #1335 STRUCTURAL ORTHOGONALITY claim:
- "Stacked fec6+v8" is **structural orthogonality** of two byte-disjoint sub-substrates within the FULL contest archive
- fec6 per-pair selector axis (fixed-Huffman k=16 entropy-coded which-of-K-codebook-entries per pair)
- v8 chroma_lut byte axis (PCG64-procedural seed → 4096 → 32 byte savings)
- Per the closed-form: stacked rate-axis ΔS = -25 × 4064 / 37_545_489 = -0.0027060507854885

**However**, the v8 trainer (`experiments/train_substrate_nscs06_v8_chroma_lut.py`) emits a STANDALONE v8 archive — NOT a byte-merged stacked archive that LITERALLY contains both fec6 selector bytes + v8 chroma bytes. The "stacking" claim is a **rate-axis closed-form prediction**, not a literal byte-merge operation.

The v8 trainer's archive emission produces ~187 bytes for the v2 procedural-seed variant (per recipe predicted_delta_basis). This is the FULL trainer output, NOT a stacking-onto-fec6 byte-union.

**Architectural reality check**: T3 council #1335 STRUCTURAL ORTHOGONALITY claim is satisfied IFF the v8 trainer is dispatched and the resulting archive produces an empirical ΔS that — when ADDED to the existing fec6 frontier contest-CPU score 0.192028 — lands in band [0.18930, 0.19055]. The "stack onto fec6" framing is a counterfactual modeling step on the rate-axis ΔS prediction, NOT a literal byte-stacking operation in the trainer or dispatcher.

This interpretation aligns with: (a) the recipe declaring `predicted_delta: -0.002706` (a SCALAR rate-axis delta); (b) the L1 EMPIRICAL landing memo declaring stacking as "STRUCTURALLY ORTHOGONAL" with closed-form rate-axis math; (c) the absence of any byte-stacking helper in `tools/` or `tac.composition.*`.

---

## Blocker enumeration

### BLOCKER 1: `TRAINER_V3_WIRE_IN_MISSING` (critical)

**Symptom**: v8 trainer's `_full_main` invokes `pack_archive` WITHOUT the `cls_bytes` kwarg that the cls_stream wire-in extended `pack_archive` to accept.

**Mechanism**:
- cls_stream wire-in landed at commit `581b7b129` (sister subagent `nscs06-v8-cls-stream-wire-in-20260526`)
- Wire-in scope: archive.py + inflate.py + 17 new tests — STRUCTURALLY OPERATIONAL per Catalog #233 Gate 3
- Wire-in scope EXCLUDED the trainer per the landing memo's explicit out-of-scope note (operator-routable downstream)
- Per cls_stream landing memo Catalog #125 hook #4 declaration: "cathedral autopilot ranker can NOW promote NSCS06 v8 chroma_lut WINNER candidate to 4-arm paired Modal T4 dispatch slot per T3 council #1335 RANKED ORDERING"
- BUT the wire-in landing memo did NOT mark Catalog #233 Gate 1 (`impl_complete`) as covering the TRAINER-V3-WIRE-IN — only the archive.py + inflate.py at the codec surface

**Empirical consequence**: dispatching the canonical recipe NOW invokes `_full_main` which emits a v2 archive. The v2 archive's CH08 v2 grammar has NO `CLS_STREAM` section. The inflate path (per cls_stream landing memo Gate 3 evidence) consumes cls_stream when present (v3) and falls back to `cls_full=np.zeros_like(gray_full)` when absent (v1/v2 — cargo-cult #5 active).

**Predicted outcome if dispatched now**:
- v2 archive ~187 bytes emitted (matches recipe `predicted_delta_basis`)
- Stacked rate-axis ΔS = -0.002706 (closed-form, predicted)
- But seg+pose axes operate on cls_full=zeros (no per-class chroma reconstruction) → SegNet sees uniform chroma → cargo-cult #5 FAIL_AT_CLASS_1 → seg+pose distortion DOMINATES → empirical score likely LANDS OUTSIDE T3 predicted band [0.18930, 0.19055] by significant margin (potentially > +1.0 ΔS — sister anchors from L0 SCAFFOLD `FAIL_AT_CLASS_1` empirical confirmation)
- Per Catalog #307 paradigm-vs-implementation: this would be IMPLEMENTATION-level falsification (cls_stream wire-in incomplete at trainer surface), NOT paradigm refutation
- Per Catalog #308 alternative-probe-methodologies: the canonical remediation is the trainer-v3-wire-in, NOT pivoting to alternative reducers

**Recommendation**: HALT this dispatch slot per CLAUDE.md "Executing actions with care" + "Submission auth eval — BOTH CPU AND CUDA" non-negotiable. Route a sister subagent to wire `_full_main` to invoke `pack_archive(..., cls_bytes=...)` per the v3 CH08 schema. After the trainer-v3-wire-in lands, RE-FIRE this slot to dispatch the v3 archive and measure the empirical 4-arm paired ΔS.

**Sister-coordination signal**: the trainer-v3-wire-in is a small (~20-50 LOC) extension to `_full_main` that:
1. Derives per-cell uint8 class labels at compress time via existing SegNet forward at grayscale resolution (h=48, w=64)
2. Packs `cls_bytes = cls_lowres.tobytes()` where `cls_lowres.shape = (n_pairs, h_g, w_g)` per the wire-in archive.py contract
3. Routes `pack_archive(..., chroma_seed=seed, cls_bytes=cls_bytes)` to emit CH08 v3 schema
4. Updates the v8 trainer's `archive_variant_tag` to `"v3_procedural_seed_with_cls_stream"` for traceability

This is a NON-TRIVIAL but BOUNDED extension. Per CLAUDE.md "Forbidden premature KILL": the cls_stream wire-in landing IS NOT FALSIFIED by this BLOCKER; only the trainer integration is incomplete.

### Non-blockers (all PASS)

- ✅ cls_stream wire-in 17/17 tests pass
- ✅ canonical operator-authorize recipe `dispatch_enabled: true` with `dispatch_blockers: []`
- ✅ Catalog #325 per-substrate symposium PROCEED_WITH_REVISIONS within window
- ✅ Catalog #324 predicted_band post-training validation status declared (`pending_post_training`)
- ✅ Catalog #344 canonical equation #26 IN-DOMAIN context registered
- ✅ Catalog #246 paired dispatcher CLI canonical helper exists + supports `--skip-axis-if-promotable-anchor-exists`
- ✅ fec6 frontier archive sha `6bae0201` exists in continual_learning posterior (PR110 frontier)
- ✅ canonical_frontier_pointer.json contest_cpu best 0.192028 / contest_cuda best 0.205330 (DIFFERENT archive; v8 is candidate WINNER #1 to potentially beat 0.192028)
- ✅ Catalog #270 dispatch optimization protocol — Tier 1+2+3 declared in recipe + trainer
- ✅ Catalog #244 canonical NVML/Modal/CUDA env hygiene block auto-emitted in lane_script
- ✅ Catalog #339 silent-no-spawn extinction self-protection active (canonical operator-authorize wrapper)
- ✅ Catalog #360 pre-spawn fatal observability active

---

## 6-hook wire-in declaration per Catalog #125

1. **sensitivity-map**: N/A. Pre-dispatch audit; no signal contribution.
2. **Pareto constraint**: N/A. Pre-dispatch audit; no Pareto contribution.
3. **bit-allocator hook**: N/A. Pre-dispatch audit; no bit-allocator contribution.
4. **cathedral autopilot dispatch hook**: ACTIVE. This BLOCKED verdict prevents cathedral autopilot ranker from promoting the v8 v2 archive (cargo-cult #5 active) into a paid Modal dispatch slot. Operator-routable trainer-v3-wire-in then re-fires the dispatch.
5. **continual-learning posterior**: N/A. No posterior anchor written; the empirical anchor will land only after trainer-v3-wire-in + paired Modal dispatch land.
6. **probe-disambiguator**: ACTIVE. This pre-dispatch gate report IS the canonical disambiguator between (a) "cls_stream wire-in COMPLETE at all surfaces" (FALSE per audit; trainer is the missing surface) and (b) "cls_stream wire-in COMPLETE at archive.py + inflate.py only" (TRUE; the bounded scope of commit `581b7b129`).

---

## Sister coordination (Catalog #340 + #230 ownership map)

In-flight sister subagents per active checkpoint state (read via `.omx/state/subagent_progress.jsonl`):
- PR110-OPT-3 Variant C variable-K escape mechanism (slot 2; different layer entirely; selector-stream codec): **DISJOINT** (touches `tools/` + `tac.codec.*` selector layer; does NOT touch v8 substrate or v8 trainer)
- BoostNeRV gain_clamp sweep (slot 3; different substrate): **DISJOINT** (touches `boostnerv_pr110` substrate; does NOT touch v8 substrate or v8 trainer)

No ownership-map collisions for this pre-dispatch audit. The v8 substrate trainer `experiments/train_substrate_nscs06_v8_chroma_lut.py` is read-only-by-this-slot (no edits attempted; bounded scope per CLAUDE.md "Executing actions with care" pending operator decision on BLOCKER 1).

---

## Operator-routable next steps (3 options enumerated; operator decides)

### Option A (recommended): route TRAINER-V3-WIRE-IN sister subagent

Scope: extend `experiments/train_substrate_nscs06_v8_chroma_lut.py::_full_main` (and `_smoke_main`) to emit CH08 v3 schema via `pack_archive(..., cls_bytes=...)`. Bounded ~20-50 LOC extension. After landing:
- Re-fire THIS slot (re-spawn `nscs06-v8-stacked-paired-modal-t4-auth-eval-20260526` via TaskCreate) to dispatch the v3 archive paired Modal T4
- Cost: $0 (sister wire-in) + $0.50-1.00 (paired dispatch after sister lands)
- Wall-clock: ~30-60 min (sister) + ~30-60 min (paired dispatch + harvest)
- Outcome on success: PR111 submission candidate per T3 council #1335 RANKING ORDER

### Option B: explicit operator-frontier-override

Per Catalog #300 §"Mission alignment" Consequence 1: operator can bypass the BLOCKER and dispatch the v2 archive NOW as a "measure-falsification-cost" diagnostic anchor. Per CLAUDE.md "Forbidden premature KILL without research exhaustion": this would produce a contest-CUDA + contest-CPU paired diagnostic anchor proving cargo-cult #5 FAIL_AT_CLASS_1 empirically AT MODAL SCALE (vs the synthetic L0 SCAFFOLD measurement). Diagnostic value: confirms the magnitude of the cargo-cult #5 score impact at production scale; informs whether trainer-v3-wire-in is high-EV vs alternative reducers per Catalog #308. Cost: $0.50-1.00 paid Modal for diagnostic-only anchor (non-promotable per Catalog #127 + #192 axis tag).

Operator-frontier-override invocation requires verbatim operator quote per Catalog #300 §"Mission alignment" Consequence 1 fields `council_override_invoked: true` + `council_override_rationale: "<verbatim operator quote ≥4 chars>"` recorded in the dispatch audit-trail.

### Option C: PIVOT to ranked candidate #2 (grayscale_lut)

Per T3 council #1335 RANKING ORDER WINNER #2 = grayscale_lut. The grayscale_lut substrate is already L2 promoted per sister TaskCreate work. Skip v8 entirely and dispatch grayscale_lut as the WINNER #1 candidate (instead of v8). Cost: $0.50-1.00 paid Modal grayscale_lut dispatch. Outcome: empirically validates grayscale_lut as PR111 candidate if it lands in band; falls back to alternative WINNERs (#3 VQ-VAE indices_blob / #4 ATW V2 REMOVAL) if outside-band.

---

## Files touched this audit session (no working-tree mutation; READ-ONLY + APPEND-ONLY this NEW report)

- READ:
  - `.omx/research/nscs06_v8_chroma_lut_mlx_l1_empirical_landed_20260526.md`
  - `.omx/research/nscs06_v8_cls_stream_wire_in_landed_20260526.md`
  - `.omx/operator_authorize_recipes/substrate_nscs06_v8_chroma_lut_modal_t4_dispatch.yaml`
  - `experiments/train_substrate_nscs06_v8_chroma_lut.py` (lines 280-490 + 700-860)
  - `tools/dispatch_modal_paired_auth_eval.py` (CLI surface)
  - `.omx/state/canonical_frontier_pointer.json`
- WRITE:
  - `.omx/research/nscs06_v8_stacked_paired_modal_t4_auth_eval_pre_dispatch_gate_report_20260526.md` (THIS file; NEW)
  - `.omx/state/subagent_progress.jsonl` (APPEND-ONLY via canonical checkpoint helper; 3 rows for this subagent)

NO paid Modal dispatch fired. NO trainer modifications attempted. NO recipe modifications attempted.

---

## Cross-references

- T3 council #1335 verdict: `.omx/research/t3_council_pr110_stacking_pivot_ordering_landed_20260526.md`
- NSCS06 v8 L1 EMPIRICAL landing: `.omx/research/nscs06_v8_chroma_lut_mlx_l1_empirical_landed_20260526.md` (commit `4a4ab1e4f`)
- NSCS06 v8 cls_stream wire-in landing: `.omx/research/nscs06_v8_cls_stream_wire_in_landed_20260526.md` (commits `581b7b129` + `545beb35c`)
- Canonical operator-authorize recipe: `.omx/operator_authorize_recipes/substrate_nscs06_v8_chroma_lut_modal_t4_dispatch.yaml`
- Canonical paired dispatcher: `tools/dispatch_modal_paired_auth_eval.py` (Catalog #246)
- Canonical equation #26: `src/tac/canonical_equations/procedural_codebook_savings.py`
- Path 3 C' L0 SCAFFOLD (cargo-cult #5 FAIL_AT_CLASS_1 empirical confirmation): `.omx/research/path_3_c_nscs06_v8_chroma_lut_L0_scaffold_landed_20260526.md`
- New 2026-05-26 MLX↔CUDA drift directive: `feedback_mlx_cuda_bidirectional_drift_anticipation_standing_directive_20260526.md`
- Sister candidate WINNER #2 (grayscale_lut): per T3 council #1335 RANKING ORDER

---

## Final operator-routable verdict

**BLOCKED_OPERATOR_DECISION_REQUIRED.** This slot HALTS pending operator selection of Option A / B / C above. Cost saved by HALT: ~$0.50-1.00 paid Modal + 30-60 min wall-clock that would have produced a v2 archive auth_eval anchor (cargo-cult #5 active; FAIL_AT_CLASS_1 from L0 SCAFFOLD EMPIRICAL CONFIRMATION). No PR111 candidacy decision possible without first satisfying BLOCKER 1.
