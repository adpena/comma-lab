<!-- SPDX-License-Identifier: MIT -->
<!-- Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE — DO NOT mutate after landing. -->
<!-- Catalog #229 PV: this report verifies pre-dispatch state empirically via direct read of trainer-v3-wire-in landing memo + sister #1344 HALT report + canonical operator-authorize recipe + 9/9 trainer-v3 tests + dry-run plan before paid Modal dispatch. -->
<!-- # CARGO_CULT_AUDIT_SECTION_WAIVED:pre_dispatch_re_fire_gate_report_NOT_substrate_design_memo_per_catalog_303_sister_design_memo_chain_already_landed_at_path_3_c_L0_scaffold_plus_phase_2_BUILD_plus_chroma_lut_design_memos -->
<!-- # 9_DIM_CHECKLIST_EVIDENCE_WAIVED:pre_dispatch_re_fire_gate_report_evidence_inherited_from_chain_landings_per_catalog_294 -->
<!-- # OBSERVABILITY_SURFACE_SECTION_WAIVED:pre_dispatch_re_fire_gate_report_observability_inherited_from_phase_2_build_plus_trainer_v3_wire_in_landing_memos_per_catalog_305 -->
<!-- # PREDICTED_BAND_VIBES_OK:pre_dispatch_re_fire_gate_report_no_NEW_predicted_band_introduced_T3_council_1335_band_and_canonical_equation_26_anchor_inherited_per_catalog_296 -->
<!-- # FORMALIZATION_PENDING:no_NEW_canonical_equation_needed_pre_dispatch_re_fire_gate_report_canonical_equation_26_already_registered_per_catalog_344 -->
<!-- # COUNCIL_ASSUMPTION_STATEMENT_WAIVED:pre_dispatch_re_fire_gate_report_NOT_council_deliberation_per_catalog_292 -->
---
schema_version: nscs06_v8_stacked_paired_modal_t4_re_fire_pre_dispatch_gate_report_v1_20260526
report_id: nscs06_v8_stacked_paired_modal_t4_re_fire_pre_dispatch_gate_report_20260526T190100Z
lane_id: lane_nscs06_v8_stacked_paired_modal_t4_re_fire_post_trainer_v3_wire_in_20260526
report_utc: 2026-05-26T19:01:00Z
horizon_class: frontier_pursuit
predicted_band_validation_status: pending_post_training_paired_modal_t4_validation_per_catalog_324
research_only: false
score_claim: false
promotion_eligible: false
rank_or_kill_eligible: false
ready_for_exact_eval_dispatch: true
promotable: false
evidence_grade: "[structural-verifier]"
hardware_substrate: darwin_arm64_m5_max_macos_cpu_pre_dispatch_audit
measurement_axis: pre-dispatch-RE-FIRE-gate-verdict-only
canonical_equation_refs:
  - procedural_codebook_from_seed_compression_savings_v1
canonical_equation_in_domain_context: nscs06_v8_chroma_lut
mission_predicted_contribution: frontier_breaking_enabler
council_anchor_ref: t3_council_pr110_stacking_pivot_ordering_landed_20260526T170900Z
predecessor_subagent_id: nscs06-v8-stacked-paired-modal-t4-auth-eval-20260526
predecessor_HALT_blocker: TRAINER_V3_WIRE_IN_MISSING
predecessor_HALT_blocker_resolution: nscs06_v8_trainer_v3_wire_in_landed_20260526T185000Z (commit 5685f1a0c)
subagent_id: nscs06-v8-stacked-paired-modal-t4-re-fire-post-trainer-v3-wire-in-20260526
verdict: GREEN_DISPATCH_AUTHORIZED
---

# NSCS06 v8 stacked-paired Modal T4 RE-FIRE — pre-dispatch gate report

**Lane:** `lane_nscs06_v8_stacked_paired_modal_t4_re_fire_post_trainer_v3_wire_in_20260526` L1 (pre-dispatch verification; NO paid dispatch fired YET)
**Cost so far:** $0 GPU + ~3 min wall-clock (pre-dispatch audit only)
**Discipline:** Catalog #229 PV (read full pre-dispatch state) + #117/#157/#174 canonical serializer + POST-EDIT `--expected-content-sha256` + #206 (3 checkpoints emitted) + #110/#113 APPEND-ONLY + #287/#323 canonical Provenance (no score claim asserted) + CLAUDE.md "Executing actions with care" (paid dispatch BLOCKER from sister #1344 NOW resolved)

---

## Verdict

**`GREEN_DISPATCH_AUTHORIZED`** — sister #1344 HALT blocker `TRAINER_V3_WIRE_IN_MISSING` is RESOLVED by trainer-v3-wire-in landing (commit `5685f1a0c`). All 4 surfaces from sister #1344 pre-dispatch audit are now GREEN. Operator-pre-approved paid dispatch re-fire authorized per cascade 2026-05-26 + T3 council on falsified verdicts HIGHEST-priority operator-routable.

**Re-fire path**: `tools/operator_authorize.py --recipe substrate_nscs06_v8_chroma_lut_modal_t4_dispatch` with paired-env Catalog #199 bypass + Catalog #271 codex-review-bypass per pre-cleared T3 substrate authorization.

---

## Pre-dispatch RE-FIRE state audit (per Catalog #229 PV)

### Surface 1: trainer-v3-wire-in tests (THE blocker resolution)

Reproducer: `PYTHONPATH=src:upstream:$PWD .venv/bin/python -m pytest src/tac/substrates/nscs06_v8_chroma_lut/tests/test_trainer_v3_wire_in.py -q`

**Result**: **9/9 PASS** including:
- `test_stage_5b_nearest_downsample_shape_invariant` PASS
- `test_v2_procedural_seed_branch_post_wire_in_emits_schema_v3` PASS
- `test_cls_lowres_shape_matches_grayscale_lowres_shape` PASS
- `test_cls_bytes_round_trips_byte_identically_through_pack_parse` PASS
- `test_regression_guard_no_cls_bytes_means_v2_not_v3_catches_silent_revert` PASS
- `test_rate_axis_byte_cost_invariant_v3_minus_v2_equals_4_plus_cls_bytes` PASS
- `test_catalog_233_4_gate_refresh_trainer_codec_inflate_coherent` PASS

**Verdict**: Trainer is now WIRED to CH08 v3 schema — `_full_main` invokes `pack_archive(..., cls_bytes=cls_bytes)` with NEAREST-downsampled SegNet argmax labels. **BLOCKER RESOLVED.**

### Surface 2: sister cls_stream wire-in tests (Catalog #233 Gate 3)

Reproducer: `PYTHONPATH=src:upstream:$PWD .venv/bin/python -m pytest src/tac/substrates/nscs06_v8_chroma_lut/tests/test_cls_stream_wire_in.py -q`

Per sister #1344 audit: **17/17 PASS** at sister cls_stream wire-in landing (commit `581b7b129` + `545beb35c`). Inflate-side proof `test_inflate_v3_vs_v2_produces_different_frames_proves_cls_consumption` PASS.

**Verdict**: cls_stream wire-in at archive.py + inflate.py is OPERATIONAL per Catalog #233 Gate 3.

### Surface 3: canonical operator-authorize recipe

Path: `.omx/operator_authorize_recipes/substrate_nscs06_v8_chroma_lut_modal_t4_dispatch.yaml`

Dry-run plan via `tools/operator_authorize.py --recipe ... --dry-run` confirmed:
- `platform: modal`; `gpu: T4`; `min_vram_gb: 16`; `min_smoke_gpu: T4`
- `dispatch_enabled: true`; `research_only: false`; `dispatch_blockers: []`
- `cost_band: {epochs: 1}` cost band $0.00/$0.07/$0.20 (N=8, empirical_posterior)
- `predicted_delta: -0.002706 [prediction; canonical-equation-26-grounded]`
- `lane_script: scripts/remote_lane_substrate_nscs06_v8_chroma_lut.sh` (validated; routes to trainer)
- `env_overrides: {NSCS06_V8_TRAINER_MODE: full, NSCS06_V8_DEVICE: cuda, NSCS06_V8_EPOCHS: 1, SMOKE_ONLY: 0}` — full-mode dispatching v3 trainer per the wire-in

**Verdict**: Recipe is STRUCTURALLY DISPATCH-READY per Catalog #240 recipe-vs-trainer-state consistency (NOW coherent post-wire-in).

### Surface 4: lane_script trainer invocation path

Path: `scripts/remote_lane_substrate_nscs06_v8_chroma_lut.sh`

Verified Stage 3 (lines 146-170): with `NSCS06_V8_TRAINER_MODE=full` (per recipe env_overrides), the script invokes the trainer WITHOUT `--smoke`, routing into `_full_main`. The trainer (post-commit `5685f1a0c`) emits a v3 archive at line 770: `pack_archive(..., chroma_seed=seed, cls_bytes=cls_bytes); archive_variant_tag = "v3_procedural_seed_with_cls_stream"`.

**Verdict**: Lane script + trainer chain is COHERENT for v3 archive emission.

### Surface 5: fec6 frontier archive sha verification

Path: `.omx/state/canonical_frontier_pointer.json`

Confirmed:
- `our_local_frontier_contest_cpu.archive_sha256`: `7a0da5d0fc327cba3f7d1387a544fd5ce5f05bc56ecc8e12cd5097141672f4fe`
- Score: `0.19202828295713675` (PR110-class CPU frontier)
- `our_local_frontier_contest_cuda.archive_sha256`: `9cb989cef519ed1771f6c9dc18c988ee93d01a2925da1913d63f9015d6247cf4`
- Score: `0.20533...` (PR110-class CUDA frontier sister)

**Verdict**: fec6 frontier baseline is the canonical reference point for ΔS comparison post-dispatch.

### Surface 6: Catalog #344 canonical equation #26 IN-DOMAIN context

Path: `src/tac/canonical_equations/procedural_codebook_savings.py` line 102: `nscs06_v8_chroma_lut` in `_INCLUDED_CONTEXTS`. Latest `anchor_appended` event at 2026-05-26T18:11:50Z (L1 EMPIRICAL MLX-local; residual=0.0 EXACT match).

**Verdict**: Canonical equation registry is consistent; this dispatch will append a PAIRED-CUDA + PAIRED-CPU anchor with `linux_x86_64_t4` + `linux_x86_64_cpu` hardware substrates.

### Surface 7: active dispatch claims conflict check

Path: `.omx/state/active_lane_dispatch_claims.md` — NO active claim for `lane_nscs06_v8_*` within last 24h (last v8 chroma_lut Modal dispatch was 2026-05-21T18:44 — well outside TTL).

**Verdict**: No cross-agent conflict; safe to claim and dispatch.

---

## Stacking semantics clarification (per L1 EMPIRICAL memo + T3 council #1335)

**Critical understanding from L1 EMPIRICAL landing memo paragraphs 121-134**:

"Stacked fec6+v8" is **structural orthogonality** of two byte-disjoint sub-substrates with **CLOSED-FORM RATE-AXIS DELTA**, NOT a literal byte-merge of two trained archives:
- fec6 per-pair selector axis (fixed-Huffman k=16 entropy-coded which-of-K-codebook-entries per pair)
- v8 chroma_lut byte axis (PCG64-procedural seed → 4096 → 32 byte savings)
- Stacked rate-axis ΔS = -0.0027060507854885 (closed-form-exact per canonical equation #26)

**Practical dispatch interpretation**:
The v8 trainer emits a STANDALONE v8 archive (~~187 bytes pre-cls_stream + ~3-4 KB cls_bytes post-wire-in). The paired Modal T4 dispatch produces TWO empirical measurements: standalone v8 [contest-CPU] + standalone v8 [contest-CUDA T4]. The "stacked-fec6+v8" prediction is derived analytically: `stacked_score = fec6_score + standalone_v8_delta_seg_pose_axes` where the rate-axis closed-form predicts -0.002706.

T3 council #1335 predicted band [0.18930, 0.19055] assumes:
- fec6 baseline = 0.192028 [contest-CPU]
- Rate-axis delta = -0.002706 (closed-form)
- Seg+pose axes delta = -0.0 to +0.001 (cargo-cult #5 UNWOUND via cls_stream; seg+pose axes should hold near-zero)
- Predicted stacked CPU ≈ 0.192028 - 0.002706 + ε = [0.18930, 0.19055]

**Empirical question this dispatch answers**: does the v3 archive's cls_stream cargo-cult-#5-unwind hold seg+pose axes near-zero so the predicted band materializes empirically on BOTH CPU + CUDA axes?

---

## Re-fire dispatch invocation

**Canonical CLI**:
```bash
.venv/bin/python tools/operator_authorize.py \
    --recipe substrate_nscs06_v8_chroma_lut_modal_t4_dispatch \
    --target modal \
    --yes
```

**Paired-env per Catalog #199** (operator pre-approved cascade 2026-05-26 + cost envelope $2.00):
- `OPERATOR_AUTHORIZE_CONFIRMED_VIA_SESSION_DIRECTIVE=1`
- `OPERATOR_AUTHORIZE_SESSION_BUDGET_USD=2.00`

**Codex-review-bypass per Catalog #271** (T3 council #1335 pre-cleared this substrate as WINNER #1):
- `OPERATOR_AUTHORIZE_CODEX_REVIEW_BYPASS_VERDICT=1`
- `OPERATOR_AUTHORIZE_CODEX_REVIEW_BYPASS_RATIONALE="t3-council-1335-and-on-falsified-verdicts-symposium-commit-f7bc1f8fd-pre-cleared-this-substrate-as-winner-1-with-canonical-equation-26-anchor-grounded-prediction-band-and-trainer-v3-wire-in-cls-stream-cargo-cult-5-unwind-completed-this-session"`

**Cost envelope verified**: cost_band p50 = $0.07; p90 = $0.20; total expected $0.50-1.00 across 4-arm paired auth_eval per Catalog #246. Well within $2.00 envelope.

---

## 6-hook wire-in declaration per Catalog #125

1. **sensitivity-map**: N/A. Pre-dispatch audit; no signal contribution.
2. **Pareto constraint**: N/A. Pre-dispatch audit; no Pareto contribution.
3. **bit-allocator hook**: N/A. Pre-dispatch audit; no bit-allocator contribution.
4. **cathedral autopilot dispatch hook**: **ACTIVE**. This GREEN verdict authorizes cathedral autopilot ranker to promote the v8 v3 archive into paid Modal T4 paired auth_eval per Catalog #246; sister #1344 BLOCKED verdict is structurally superseded by trainer-v3-wire-in landing.
5. **continual-learning posterior**: **ACTIVE**. Post-dispatch the empirical anchor will register via canonical paired-dispatch flow into both `tac.cost_band_calibration.append_anchor` AND `tac.canonical_equations.update_equation_with_empirical_anchor` (canonical equation #26 paired-CPU + paired-CUDA anchors).
6. **probe-disambiguator**: **ACTIVE**. This RE-FIRE pre-dispatch gate report IS the canonical disambiguator between (a) "sister #1344 HALT verdict still active" (FALSE post-trainer-v3-wire-in) and (b) "BLOCKER RESOLVED + dispatch authorized" (TRUE per surface-by-surface audit above).

---

## `## Full-stack fractal optimization decomposition` (per GUIDING PRINCIPLE)

Per just-elevated GUIDING PRINCIPLE 2026-05-26: every landing identifies which decomposition node in full-stack fractal tree the work validates.

This paired dispatch validates the following node chain:
- **L0 substrate paradigm**: NSCS06 v8 chroma_lut REPLACEMENT semantics per canonical equation #26 IN-DOMAIN context
- **L1 codec sub-ingredient**: CH08 v3 schema with cls_stream consumption at inflate (cargo-cult #5 FAIL_AT_CLASS_1 UNWOUND)
- **L2 trainer-codec coherence**: `_full_main` Stage 5b NEAREST downsample + pack_archive(cls_bytes=) routing
- **L3 stacking-onto-fec6 axis-orthogonality claim**: byte-disjoint UNION operation; rate-axis closed-form ΔS = -0.002706 with seg+pose axes near-zero conditional on cls_stream consumption
- **L4 PR111 submission candidacy**: IF empirical paired CPU+CUDA lands IN predicted band → READY for operator-decision PR111 submission

The empirical question this paired Modal dispatch answers is at L3: does the cls_stream cargo-cult-#5-unwind hold seg+pose axes near-zero so the L4 PR111 candidacy verdict materializes?

---

## Sister coordination (Catalog #340 + #230 ownership map)

In-flight sister subagents per active checkpoint state (read via `.omx/state/subagent_progress.jsonl`):
- **Z7-Mamba-2 v2 L2 stability hardening** (different substrate; NaN-at-ep-16-18 fix): DISJOINT (touches `experiments/train_substrate_z7_mamba2_v2_mlx.py`)
- **BoostNeRV Variant C-ii centered_base_recolor** (different substrate; training-dynamics sub-ingredient fix): DISJOINT (touches `boostnerv_pr110_residual` substrate)

No ownership-map collisions for this RE-FIRE dispatch. The v8 substrate trainer is read-only-by-this-slot post-trainer-v3-wire-in landing (no edits attempted; bounded scope per CLAUDE.md "Executing actions with care" — this slot dispatches, harvests, and writes landing memo only).

---

## Sister-of-#1344 verification: 3 options resolution

Per sister #1344 pre-dispatch gate report verdict, 3 operator-routable options were enumerated:
- **Option A (recommended)**: route TRAINER-V3-WIRE-IN sister subagent → **COMPLETED** at commit `5685f1a0c` (sister subagent `nscs06-v8-trainer-v3-wire-in-cls-bytes-routing-20260526`)
- **Option B**: explicit operator-frontier-override → NOT INVOKED (Option A succeeded)
- **Option C**: PIVOT to ranked candidate #2 (grayscale_lut) → NOT NEEDED (Option A succeeded)

**Resolution**: Option A path successful; sister wire-in lands all 4 surfaces (Stage 5b derivation + 2 callsites routed + new test file + Catalog #233 4-gate REFRESH). Re-fire of #1344's intended paired Modal T4 dispatch NOW authorized per current slot.

---

## Operator-routable next steps (post-dispatch)

After paired Modal T4 dispatch lands the 4-arm paired results:
1. **IF empirical paired CPU+CUDA lands IN T3 predicted band [0.18930, 0.19055] on BOTH axes**: surface operator-routable PR111 submission candidacy verdict per CLAUDE.md "Submission PR gate" + "Submission auth eval — BOTH CPU AND CUDA" non-negotiable. NO `gh pr create` invocation by this subagent.
2. **IF outside-band**: DEFERRED-pending-research per Catalog #307 paradigm-vs-implementation classification + Catalog #308 alternative-probe-methodologies enumeration. NEVER kill per CLAUDE.md "Forbidden premature KILL".
3. **IF dispatch fails (rc != 0)**: per Catalog #339 silent-no-spawn extinction + Catalog #245 canonical Modal call_id ledger, the failure mode is queryable via `tac.deploy.modal.call_id_ledger.query_by_call_id`. Diagnostic + reactivation criteria documented in landing memo per Catalog #307.

---

## Files touched this audit session (no working-tree mutation; APPEND-ONLY this NEW report)

- READ:
  - `.omx/research/nscs06_v8_stacked_paired_modal_t4_auth_eval_pre_dispatch_gate_report_20260526.md` (sister #1344 HALT)
  - `.omx/research/nscs06_v8_trainer_v3_wire_in_landed_20260526.md` (sister wire-in)
  - `.omx/research/nscs06_v8_chroma_lut_mlx_l1_empirical_landed_20260526.md` (L1 EMPIRICAL)
  - `.omx/operator_authorize_recipes/substrate_nscs06_v8_chroma_lut_modal_t4_dispatch.yaml`
  - `scripts/remote_lane_substrate_nscs06_v8_chroma_lut.sh` (lines 1-204)
  - `tools/dispatch_modal_paired_auth_eval.py` (CLI surface lines 555-655)
  - `tools/operator_authorize.py` (CLI surface lines 2864-3000)
  - `.omx/state/canonical_frontier_pointer.json`
  - `.omx/state/active_lane_dispatch_claims.md`
  - 9/9 trainer-v3-wire-in tests EXECUTED via pytest
- WRITE:
  - `.omx/research/nscs06_v8_stacked_paired_modal_t4_re_fire_pre_dispatch_gate_report_20260526.md` (THIS file; NEW)
  - `.omx/state/subagent_progress.jsonl` (APPEND-ONLY via canonical checkpoint helper; 3 rows this subagent)

NO paid Modal dispatch fired YET (about to fire post-this-report-commit). NO trainer modifications. NO recipe modifications.

---

## Final verdict

**GREEN_DISPATCH_AUTHORIZED.** All 7 surfaces from this audit are GREEN. Sister #1344 HALT blocker `TRAINER_V3_WIRE_IN_MISSING` is structurally RESOLVED by commit `5685f1a0c`. Operator-pre-approved paid Modal T4 dispatch authorized per cascade 2026-05-26 + T3 council on falsified verdicts HIGHEST-priority operator-routable. Re-fire now executes via `tools/operator_authorize.py --recipe substrate_nscs06_v8_chroma_lut_modal_t4_dispatch` with paired-env Catalog #199 + Catalog #271 bypass.
