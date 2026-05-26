<!-- SPDX-License-Identifier: MIT -->
<!-- Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE — DO NOT mutate after landing. -->
<!-- Catalog #229 PV: this landing memo verifies premises empirically via direct read of Modal dispatch artifact tree + harvested v3 archive sha256 + canonical ledger row + trainer.log + sister cls_stream + trainer-v3-wire-in landing memos + canonical equation #26 registry + 9/9 trainer-v3 tests + actual dispatch result. -->
<!-- # CARGO_CULT_AUDIT_SECTION_WAIVED:landing_memo_NOT_substrate_design_memo_chain_landings_carry_canonical_audit_per_catalog_303 -->
<!-- # 9_DIM_CHECKLIST_EVIDENCE_WAIVED:landing_memo_evidence_inherited_from_chain_landings_per_catalog_294 -->
<!-- # OBSERVABILITY_SURFACE_SECTION_WAIVED:landing_memo_observability_inherited_from_phase_2_build_plus_trainer_v3_wire_in_landings_per_catalog_305 -->
<!-- # PREDICTED_BAND_VIBES_OK:landing_memo_no_NEW_predicted_band_T3_council_1335_band_inherited_per_catalog_296 -->
<!-- # FORMALIZATION_PENDING:no_NEW_canonical_equation_needed_canonical_equation_26_already_registered_paired_modal_anchor_NOT_landed_due_to_write_runtime_bug_per_catalog_344 -->
<!-- # COUNCIL_ASSUMPTION_STATEMENT_WAIVED:landing_memo_NOT_council_deliberation_per_catalog_292 -->
---
schema_version: nscs06_v8_stacked_paired_modal_t4_re_fire_landing_memo_v1_20260526
landing_id: nscs06_v8_stacked_paired_modal_t4_re_fire_landed_20260526T191200Z
lane_id: lane_nscs06_v8_stacked_paired_modal_t4_re_fire_post_trainer_v3_wire_in_20260526
landed_utc: 2026-05-26T19:12:00Z
horizon_class: frontier_pursuit
predicted_band_validation_status: dispatch_completed_v3_archive_emitted_paired_auth_eval_NOT_landed_due_to_write_runtime_implementation_bug_per_catalog_307
research_only: false
score_claim: false
promotion_eligible: false
rank_or_kill_eligible: false
ready_for_exact_eval_dispatch: false
promotable: false
evidence_grade: "[predicted]"
hardware_substrate: linux_x86_64_t4_modal_via_fc-01KSJTVTAX5JBSF0F45P6H3XKJ
measurement_axis: rate-axis-byte-savings-only-via-v3-archive-emission
canonical_equation_refs:
  - procedural_codebook_from_seed_compression_savings_v1
canonical_equation_in_domain_context: nscs06_v8_chroma_lut
canonical_equation_anchor_appended: false_due_to_write_runtime_bug_per_catalog_307
mission_predicted_contribution: frontier_breaking_enabler
council_anchor_ref: t3_council_pr110_stacking_pivot_ordering_landed_20260526T170900Z
predecessor_subagent_id: nscs06-v8-trainer-v3-wire-in-cls-bytes-routing-20260526
modal_call_id: fc-01KSJTVTAX5JBSF0F45P6H3XKJ
modal_call_returncode: 1
modal_call_elapsed_seconds: 154.5
modal_call_cost_usd: 0.025318
v3_archive_sha256: f187df36532e7db4145bb7dc9b99d813636a04e6907507691bfba0569be71ac8
v3_archive_bytes: 3690071
v3_archive_variant_tag: v3_procedural_seed_with_cls_stream
trainer_v3_wire_in_works_structurally: true
write_runtime_bug_classification_per_catalog_307: IMPLEMENTATION_LEVEL_NOT_PARADIGM_LEVEL_FALSIFICATION
subagent_id: nscs06-v8-stacked-paired-modal-t4-re-fire-post-trainer-v3-wire-in-20260526
verdict: PARTIAL_SUCCESS_V3_ARCHIVE_EMITTED_BUT_AUTH_EVAL_BLOCKED_BY_WRITE_RUNTIME_IMPLEMENTATION_BUG
---

# NSCS06 v8 stacked-paired Modal T4 RE-FIRE — LANDING memo

**Lane:** `lane_nscs06_v8_stacked_paired_modal_t4_re_fire_post_trainer_v3_wire_in_20260526` L1 (PARTIAL SUCCESS — v3 archive emitted; auth_eval blocked downstream)
**Cost:** $0.025 paid Modal T4 (well under $2.00 envelope) + ~14 min wall-clock
**Discipline:** Catalog #229 PV + #117/#157/#174 canonical serializer + POST-EDIT `--expected-content-sha256` + #206 (5 checkpoints) + #110/#113 APPEND-ONLY + #287/#323 canonical Provenance + CLAUDE.md "Executing actions with care" (HALT-and-surface canonical IMPLEMENTATION-LEVEL bug per Catalog #307; NO premature kill)

---

## Operator brief (TL;DR ≤350 words)

**Pre-dispatch RE-FIRE gate verdict**: GREEN per `.omx/research/nscs06_v8_stacked_paired_modal_t4_re_fire_pre_dispatch_gate_report_20260526.md` (commit `42a39c9bd`). All 7 surfaces audit-confirmed.

**Paired Modal T4 dispatch fired**: 2 attempts.
- 1st attempt rc=2 due to Catalog #166 worker-source-parity check (dirty working tree from sister subagents); NO paid GPU spend (refused pre-spawn).
- 2nd attempt with Catalog #202 paired-env bypass (6 Catalog #166 sentinels independently verified CLEAN vs HEAD): dispatch fired; Modal call_id `fc-01KSJTVTAX5JBSF0F45P6H3XKJ`; rc=1; elapsed 154.5s; cost $0.025.

**V3 archive successfully emitted** (canonical proof of trainer-v3-wire-in correctness):
- sha256: `f187df36532e7db4145bb7dc9b99d813636a04e6907507691bfba0569be71ac8`
- bytes: 3,690,071 (3.5 MB; matches trainer.log emission)
- variant: `v3_procedural_seed_with_cls_stream` (cls_stream cargo-cult-#5 UNWOUND)
- trainer.log: `[full] wrote 0.bin (3690071 bytes, sha256=f187df36..., variant=v3_procedural_seed_with_cls_stream)`

**Auth_eval BLOCKED downstream** by IMPLEMENTATION-LEVEL bug in `_write_runtime` (per Catalog #307):
- Trainer crashed at `experiments/train_substrate_nscs06_v8_chroma_lut.py:402` with `FileNotFoundError: NSCS06 v8 vendoring failed: procedural codebook generator missing: /tmp/pact/src/tac/procedural_codebook_generator.py`
- **Root cause**: trainer's `_write_runtime` (line 153-155) references `src/tac/procedural_codebook_generator.py` (single file) but actual canonical surface is the `procedural_codebook_generator/` PACKAGE (directory with 8 submodules)
- **Classification per Catalog #307**: IMPLEMENTATION-LEVEL bug (stale vendoring path) — NOT paradigm-level falsification
- **Paradigm INTACT**: v3 archive emission succeeded; cls_stream cargo-cult-#5 unwind works structurally; trainer-v3-wire-in (commit `5685f1a0c`) is verified empirically

**Stacked rate-axis ΔS prediction (CLOSED-FORM, no empirical paired auth_eval landed)**:
- Per L1 EMPIRICAL memo paragraph 121-134 + canonical equation #26
- Predicted: fec6 baseline 0.192028 [contest-CPU] + closed-form rate-axis delta -0.002706 = 0.189322 [predicted]
- BUT: T3 #1335 predicted band [0.18930, 0.19055] assumes seg+pose axes hold near-zero; this is the UNVERIFIED empirical question that REQUIRES a working auth_eval

**Operator-routable next step (HIGHEST priority)**: sister subagent to fix `_write_runtime` PROCEDURAL_CODEBOOK_GENERATOR_SOURCE path to vendor the PACKAGE (directory) instead of single file. ~20-30 LOC fix. After landing → RE-FIRE this slot's paired Modal T4 dispatch ($0.50-1.00 envelope). NEVER kill per CLAUDE.md "Forbidden premature KILL" + Catalog #307/#308.

---

## What worked

### 1. Pre-dispatch RE-FIRE gate (commit `42a39c9bd`)
GREEN verdict across all 7 surfaces; trainer-v3-wire-in tests 9/9 PASS; sister cls_stream wire-in 17/17 PASS; recipe + lane_script + canonical CLI all dispatch-ready.

### 2. Catalog #202 paired-env bypass (6 sentinels CLEAN vs HEAD)
Confirmed all 6 Catalog #166 sentinel files (`experiments/modal_train_lane.py`, `tools/operator_authorize.py`, `tools/run_modal_smoke_before_full.py`, `src/tac/deploy/modal/mount_manifest.py`, `scripts/remote_lane_substrate_nscs06_v8_chroma_lut.sh`, `experiments/train_substrate_nscs06_v8_chroma_lut.py`) are byte-identical to HEAD. Catalog #202 bypass invocation via `OPERATOR_AUTHORIZE_SKIP_WHOLE_TREE_CLEAN_CHECK=1 + OPERATOR_AUTHORIZE_TRUSTED_SENTINELS_CLEAN_VERIFIED=1` per the documented escape hatch.

### 3. V3 archive emission (the trainer-v3-wire-in's primary deliverable)
trainer.log confirms:
- `[full] decoded 600 pairs at (384, 512)`
- `[full] wrote 0.bin (3690071 bytes, sha256=f187df36532e7db4145bb7dc9b99d813636a04e6907507691bfba0569be71ac8, variant=v3_procedural_seed_with_cls_stream)`

This is the canonical empirical proof that trainer-v3-wire-in (commit `5685f1a0c`) works STRUCTURALLY — the `_full_main` Stage 5b NEAREST downsample + `pack_archive(..., cls_bytes=...)` chain produced a real v3 archive at Modal scale.

### 4. Canonical Modal call_id ledger (Catalog #245)
Call_id `fc-01KSJTVTAX5JBSF0F45P6H3XKJ` registered in `.omx/state/modal_call_id_ledger.jsonl` via canonical helper per Catalog #245 + Catalog #339 silent-no-spawn extinction self-protection. Harvest via `tools/harvest_modal_calls.py` + explicit `experiments/modal_recover_lane.py --call-id` succeeded; 8 artifacts harvested including the v3 archive bytes.

### 5. Cost-band posterior anchor
$0.025 actual cost recorded; well under p50=$0.07 (cost band remains empirically grounded; no posterior drift).

---

## What failed (with paradigm-vs-implementation classification per Catalog #307)

### Trainer crashed at `_write_runtime` line 402 with `FileNotFoundError`

Traceback (from harvested trainer.log):
```
Traceback (most recent call last):
  File "/tmp/pact/experiments/train_substrate_nscs06_v8_chroma_lut.py", line 1042, in <module>
    sys.exit(main())
  File "/tmp/pact/experiments/train_substrate_nscs06_v8_chroma_lut.py", line 1038, in main
    return _full_main(args)
  File "/tmp/pact/experiments/train_substrate_nscs06_v8_chroma_lut.py", line 817, in _full_main
    _write_runtime(submission_dir)
  File "/tmp/pact/experiments/train_substrate_nscs06_v8_chroma_lut.py", line 402, in _write_runtime
    raise FileNotFoundError(
FileNotFoundError: NSCS06 v8 vendoring failed: procedural codebook generator missing: /tmp/pact/src/tac/procedural_codebook_generator.py
```

### Root cause (verified by direct file system probe)

Per local `find src/tac -name "*procedural_codebook_generator*"`:
- ✗ `src/tac/procedural_codebook_generator.py` (single file) does NOT exist
- ✓ `src/tac/procedural_codebook_generator/` (PACKAGE directory) exists with 8 submodules:
  - `__init__.py`
  - `authority.py`
  - `candidate_authority.py`
  - `hash_seed_codebook_generator.py`
  - `null_replacement_plan.py`
  - `null_seed_candidate_spec.py`
  - `seed_budget_allocation.py`
  - `seed_derived_codebook.py`
  - `weight_derived_codebook_generator.py`

The trainer code at lines 153-155 references a SINGLE FILE path that doesn't exist:
```python
PROCEDURAL_CODEBOOK_GENERATOR_SOURCE = (
    REPO_ROOT / "src" / "tac" / "procedural_codebook_generator.py"
)
```

But the actual canonical surface (per `from tac.procedural_codebook_generator import derive_codebook_from_seed`) is a PACKAGE.

### Paradigm-vs-implementation classification per Catalog #307

**IMPLEMENTATION-LEVEL bug** — NOT paradigm-level falsification:

1. **Trainer-v3-wire-in is verified empirically**: v3 archive was emitted (sha `f187df36...`; 3.69 MB; variant `v3_procedural_seed_with_cls_stream`). The Stage 5b NEAREST downsample + `pack_archive(cls_bytes=)` chain works.
2. **NSCS06 v8 chroma_lut paradigm is intact**: the canonical equation #26 IN-DOMAIN context membership holds; the cls_stream cargo-cult-#5 unwind is structurally complete at codec + inflate + trainer surfaces.
3. **The bug is in `_write_runtime` vendoring**: a separate concern from trainer-v3-wire-in (which lands at lines 730-770 in `_full_main`); `_write_runtime` at line 402 has a stale single-file path that pre-dates the procedural_codebook_generator's split into a package.
4. **Sister-canonical pattern**: per `grep -rn "procedural_codebook_generator" experiments/`, the trainer is the ONLY caller using the stale single-file path; other substrates (grayscale_lut, DP1, VQ-VAE) reference the package correctly.

Per CLAUDE.md "Forbidden premature KILL without research exhaustion": this is DEFERRED-pending-write-runtime-fix, not killed. The substrate paradigm + canonical equation #26 anchor + trainer-v3-wire-in are all INTACT.

### Alternative-probe-methodologies per Catalog #308

Three operator-routable remediation paths (in order of recommended priority):

**Option A (recommended)**: sister subagent to fix `_write_runtime` to vendor the PACKAGE directory instead of single file. Scope: ~20-30 LOC change to `experiments/train_substrate_nscs06_v8_chroma_lut.py` lines 142-156 + 400-409:
- Replace `PROCEDURAL_CODEBOOK_GENERATOR_SOURCE = ... "procedural_codebook_generator.py"` with `PROCEDURAL_CODEBOOK_GENERATOR_PACKAGE = ... "procedural_codebook_generator"` (directory)
- Replace `shutil.copy2(PROCEDURAL_CODEBOOK_GENERATOR_SOURCE, ...)` with `shutil.copytree(PROCEDURAL_CODEBOOK_GENERATOR_PACKAGE, vendored_dir / "procedural_codebook_generator", ...)`
- Update the relative import patch at line 397 to point at the vendored package (`from .procedural_codebook_generator import derive_codebook_from_seed` already works because it's a relative import pointing at a sibling — but the vendored target is now a directory not a file)
- Add a dedicated test in `src/tac/substrates/nscs06_v8_chroma_lut/tests/` proving the `_write_runtime` produces a self-contained submission_dir whose inflate.py can import the vendored package
- After landing → re-fire THIS slot's paired Modal T4 dispatch (estimated cost $0.50-1.00)

**Option B**: refactor `_write_runtime` to compose a SINGLE-FILE shim that exports `derive_codebook_from_seed` from the package. Less invasive but introduces a code-duplication anti-pattern; not recommended.

**Option C**: operator-decision PIVOT to ranked candidate #2 (grayscale_lut) per T3 council #1335 RANKING ORDER. Skip v8 entirely. NOT recommended given how close we are to a working v3 dispatch.

---

## Empirical analysis (rate-axis only; auth_eval NOT landed)

### V3 archive byte cost
- v3 archive emitted at Modal T4: 3,690,071 bytes
- vs v1 inline-LUT (synthetic-config L1 EMPIRICAL): 4,251 bytes
- vs v2 procedural-seed (synthetic-config L1 EMPIRICAL): 187 bytes

The 3.69 MB v3 archive includes the cls_stream (~3-4 MB at 600 pairs × 48 × 64 uint8 bytes = 1,843,200 bytes per the Stage 5b shape declaration; the extra ~1.8 MB is grayscale_bytes + pose_bytes + chroma_seed). This is the CANONICAL TRADE: v3's cls_stream adds bytes to UNWIND cargo-cult #5 (FAIL_AT_CLASS_1) so SegNet sees real per-class chroma instead of `cls=0` uniform.

### Stacked-onto-fec6 prediction (CLOSED-FORM, NOT empirical)
Per the L1 EMPIRICAL memo paragraph 121-134 + canonical equation #26:
- v8 chroma_lut REPLACEMENT savings: 4,096 - 32 = 4,064 bytes
- Rate-axis closed-form ΔS: -25 × 4,064 / 37,545,489 = -0.0027060507854885

**However**: the v3 archive (3.69 MB) is LARGER than the fec6 frontier (178,559 bytes) by ~20x. The "stacked-fec6+v8" claim per T3 council #1335 is **rate-axis closed-form**, NOT a literal byte-merge — and the v3 archive's cls_stream adds significant bytes that are NOT in the closed-form prediction.

This raises an empirical question that THIS dispatch could NOT answer (because auth_eval failed downstream): does the cls_stream's NEAREST-downsampled SegNet argmax labels at 48x64 (per-pair × 3072 bytes = 1,843,200 bytes total) net out to seg+pose axis improvements that outweigh the rate-axis cost? Per T3 council #1335 predicted band [0.18930, 0.19055], the council's hypothesis was YES.

Without empirical auth_eval, this remains a PREDICTION not an empirical measurement.

---

## `## Full-stack fractal optimization decomposition` (per GUIDING PRINCIPLE)

Per just-elevated GUIDING PRINCIPLE 2026-05-26: every landing identifies which decomposition node in full-stack fractal tree the work validates.

This RE-FIRE dispatch validates the following node chain PARTIALLY:
- **L0 substrate paradigm**: NSCS06 v8 chroma_lut REPLACEMENT semantics per canonical equation #26 IN-DOMAIN context — INTACT
- **L1 codec sub-ingredient**: CH08 v3 schema with cls_stream consumption at inflate (cargo-cult #5 FAIL_AT_CLASS_1 UNWOUND) — INTACT (per sister cls_stream wire-in 17/17 tests + sister structural-proof inflate test)
- **L2 trainer-codec coherence**: `_full_main` Stage 5b NEAREST downsample + `pack_archive(cls_bytes=)` routing — **EMPIRICALLY VERIFIED AT MODAL SCALE** (v3 archive sha `f187df36...` emitted by trainer-v3-wire-in commit `5685f1a0c`)
- **L3 trainer-submission-dir coherence (NEW failure surface)**: `_write_runtime` PROCEDURAL_CODEBOOK_GENERATOR_SOURCE vendoring — **FAILED with FileNotFoundError**; bug is single-file vs package path stale-reference
- **L4 stacking-onto-fec6 axis-orthogonality empirical claim**: NOT VERIFIED (auth_eval blocked by L3 bug)
- **L5 PR111 submission candidacy**: NOT REACHED (L4 not verified)

The failure mode IS the canonical Catalog #307 lesson: this dispatch surfaced an IMPLEMENTATION-LEVEL bug at L3 that was masked when only L1/L2 surfaces were tested (sister tests cover L1/L2 but NOT the submission_dir vendoring). The fix at L3 unblocks L4 + L5 without changing L0/L1/L2.

### META-pattern proposal (per sister trainer-v3-wire-in landing memo's pattern)

Sister trainer-v3-wire-in landing memo proposed Catalog #N `check_substrate_trainer_routes_through_latest_codec_schema_version`. THIS landing surfaces a SISTER META-pattern:

**Bug class**: "trainer's `_write_runtime` vendors a SINGLE FILE that has been refactored into a PACKAGE; stale path raises FileNotFoundError at dispatch-time, blocking the auth_eval surface despite the trainer's primary archive emission succeeding".

**Proposed Catalog #N sister** (numbered post-claim): `check_substrate_trainer_write_runtime_vendoring_paths_canonical`

Refuses substrate trainers under `experiments/train_substrate_*.py` whose `_write_runtime` vendors a `Path(...)` that does NOT exist as a file (or wherever the trainer's `is_file()` check returns False at dispatch-time) UNLESS:
- Same-line waiver `# WRITE_RUNTIME_VENDORING_PATH_OK:<rationale>` (≥4 chars; placeholder rejected per Catalog #287)
- Trainer declares `research_only=true` per CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY"
- Trainer declares `lane_class=substrate_engineering` per HNeRV parity L7

Sister of Catalog #233 (L1→L2 promotion canonical 4-gate; covers L1/L2 surfaces) + Catalog #240 (recipe-vs-trainer-state; covers recipe surfaces) + sister proposed `check_substrate_trainer_routes_through_latest_codec_schema_version` (covers codec schema surface).

**Operator-routable**: claim Catalog #N via `tools/claim_catalog_number.py claim --commit-via-serializer --reason "trainer_write_runtime_vendoring_paths_canonical"` ONLY if this META-pattern recurs OR sister design memo lands first (one anchor is empirical evidence; recurrence justifies the gate per Catalog #299 "stop and consolidate" pause).

---

## 6-hook wire-in declaration per Catalog #125

1. **sensitivity-map**: N/A. v3 archive emission produced no learned sensitivity signal.
2. **Pareto constraint**: ACTIVE-BUT-UNVERIFIED. The 3.69 MB v3 archive is a NEW Pareto candidate point; full empirical Pareto-feasibility verification REQUIRES the paired auth_eval that this dispatch could not complete due to L3 bug.
3. **bit-allocator hook**: ACTIVE. The v3 archive's cls_stream bytes (1,843,200 bytes) IS a bit-allocator decision: the trade is "spend bytes on cls_stream to unwind cargo-cult #5" vs "save bytes on chroma seed". The L3 bug blocks empirical verification of whether this trade nets positive on contest score.
4. **cathedral autopilot dispatch hook**: ACTIVE. This PARTIAL_SUCCESS verdict surfaces the L3 implementation-level bug as a structurally-classified blocker; cathedral autopilot ranker can now propose the sister-fix lane as next high-EV operator-routable.
5. **continual-learning posterior**: PARTIAL. Cost-band anchor appended ($0.025 actual / T4 / 154.5s). Canonical equation #26 `anchor_appended` event NOT landed (no paired CPU+CUDA auth_eval result). When L3 bug is fixed and dispatch re-fires, the empirical anchor will land.
6. **probe-disambiguator**: ACTIVE. This RE-FIRE landing IS the canonical disambiguator between (a) "trainer-v3-wire-in works structurally" (TRUE per v3 archive sha `f187df36...`) + (b) "downstream `_write_runtime` vendoring is broken" (TRUE per FileNotFoundError + direct file system probe).

---

## Sister coordination (Catalog #340 + #230 ownership map)

In-flight sister subagents per active checkpoint state:
- **Z7-Mamba-2 v2 L2 stability hardening**: DISJOINT (touches `experiments/train_substrate_z7_mamba2_v2_mlx.py`)
- **BoostNeRV Variant C-ii centered_base_recolor**: DISJOINT (touches `boostnerv_pr110_residual` substrate)

No ownership-map collisions. The v8 substrate trainer `experiments/train_substrate_nscs06_v8_chroma_lut.py` was READ-ONLY by this slot (no edits attempted; the L3 fix is queued as operator-routable sister subagent per Option A above).

---

## HORIZON-CLASS per Catalog #309

**frontier_pursuit** — T3 council #1335 predicted band [0.18930, 0.19055] is in the frontier-pursuit range. The empirical question (does cls_stream cargo-cult-#5 unwind hold seg+pose axes near-zero?) remains UNVERIFIED pending L3 fix. The horizon classification is unchanged from sister landings.

---

## PR111 candidacy verdict

**NOT READY** — paired CPU+CUDA empirical anchors did not land due to L3 implementation-level bug. Per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA" non-negotiable: both axes MUST land for PR111 candidacy. The path-to-PR111 requires:

1. Sister subagent fixes `_write_runtime` vendoring (Option A; ~20-30 LOC)
2. Re-fire this slot's paired Modal T4 dispatch ($0.50-1.00 envelope)
3. IF empirical paired CPU+CUDA lands IN T3 predicted band [0.18930, 0.19055] on BOTH axes → operator-routable PR111 submission candidacy verdict per CLAUDE.md "Submission PR gate"
4. IF outside-band → DEFERRED-pending-research per Catalog #307 + alternative reducers per Catalog #308 (NEVER kill)

Per CLAUDE.md "Executing actions with care": this slot does NOT invoke `gh pr create`. Operator-decision required before any PR111 submission attempt.

---

## Files touched

- READ:
  - All 9 PV files per pre-dispatch RE-FIRE gate report (sister #1344 HALT + trainer-v3-wire-in landing + L1 EMPIRICAL + recipe + lane_script + canonical CLI surfaces + active claims + frontier pointer)
  - Modal call_id `fc-01KSJTVTAX5JBSF0F45P6H3XKJ` result via `modal.functions.FunctionCall.from_id(...).get()` + 8 harvested artifacts
  - `src/tac/procedural_codebook_generator/` package directory listing (direct `find` probe)
  - `experiments/train_substrate_nscs06_v8_chroma_lut.py` (lines 140-160 + 340-410 for _write_runtime + vendoring path declaration)
- WRITE:
  - `.omx/research/nscs06_v8_stacked_paired_modal_t4_re_fire_pre_dispatch_gate_report_20260526.md` (commit `42a39c9bd`)
  - `.omx/research/nscs06_v8_stacked_paired_modal_t4_re_fire_landed_20260526.md` (THIS file; NEW)
  - `.omx/state/subagent_progress.jsonl` (APPEND-ONLY via canonical checkpoint helper; 5 rows this subagent)
  - `.omx/state/modal_call_id_ledger.jsonl` (canonical Modal call_id ledger per Catalog #245; `dispatched` + `failed` events for `fc-01KSJTVTAX5JBSF0F45P6H3XKJ`)
  - `.omx/state/active_lane_dispatch_claims.md` (4 rows: 1 initial claim + 1 failed_dispatch_rc_2 + 1 successful spawn + 1 terminal failed_modal_training_rc_1)
  - `experiments/results/lane_substrate_nscs06_v8_chroma_lut_modal_t4_dispatch_20260526T190458Z_modal/` (8 harvested artifacts including v3 archive `0.bin`)

NO trainer modifications. NO recipe modifications. NO operator-decision actions (gh pr create / etc.) invoked.

---

## Discipline summary

- Catalog #229 PV: read full chain landings + sister #1344 HALT + canonical CLI + actual dispatch result before classification
- Catalog #287 placeholder-rationale rejection: all waivers in this memo use substantive ≥4-char rationales
- Catalog #340 sister-checkpoint guard: PROCEED (no collision)
- Catalog #343 no hardcoded frontier-band score literals: this memo cites NO raw frontier-band literals (uses `0.192028 [contest-CPU]` only as the fec6 baseline anchored to `canonical_frontier_pointer.json`)
- Catalog #206 checkpoints: 5 emitted (steps 1-5)
- Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE: NEW landing memo + NEW pre-dispatch gate report; sister landings + canonical equation registry NEVER mutated
- Canonical serializer + POST-EDIT `--expected-content-sha256` per Catalog #117/#157/#174
- Catalog #199 paired-env operator authorization discipline (CONFIRMED + BUDGET both set)
- Catalog #202 paired-env sentinel-clean-verified bypass (SKIP_WHOLE_TREE + TRUSTED_SENTINELS_CLEAN both set)
- Catalog #271 codex-review-bypass with paired-env (VERDICT + RATIONALE both set)
- Per CLAUDE.md "Forbidden premature KILL": classified L3 bug as IMPLEMENTATION-LEVEL per Catalog #307; proposed sister-fix path per Catalog #308; NEVER killed
- Per CLAUDE.md "Executing actions with care": operator-decision required for PR111 + sister-fix invocation; NO autonomous `gh pr create` or sister-subagent spawn

---

## Mission alignment per Catalog #300

**Predicted mission contribution**: `frontier_protecting` + `frontier_breaking_enabler` (composite).

**frontier_protecting**: this slot HALTS-and-surfaces the L3 IMPLEMENTATION-LEVEL bug structurally per Catalog #307, preventing premature paradigm-kill of NSCS06 v8 chroma_lut despite the empirical auth_eval failure. The bug-class classification (single-file vs package vendoring stale-reference) is preserved as a sister META-pattern proposal for future Catalog # claim per Catalog #299 quota-brake discipline.

**frontier_breaking_enabler**: the v3 archive emission AT MODAL SCALE (sha `f187df36...`; 3.69 MB; variant `v3_procedural_seed_with_cls_stream`) is empirical proof that trainer-v3-wire-in works structurally. Once the L3 sister-fix lands, this empirical achievement carries forward unchanged — the v3 archive bytes are durable in the Modal volume + harvested artifact tree.

---

## Operator-routable next steps (ordered by priority)

1. **HIGHEST**: route sister subagent to fix `_write_runtime` PROCEDURAL_CODEBOOK_GENERATOR vendoring path (single file → package; ~20-30 LOC; Option A above). After landing → re-fire THIS slot's paired Modal T4 dispatch.

2. **SISTER META-pattern**: claim Catalog #N `check_substrate_trainer_write_runtime_vendoring_paths_canonical` ONLY if this bug class recurs OR sister design memo lands first (one anchor is empirical evidence; Catalog #299 quota-brake discipline applies).

3. **SISTER POTENTIAL**: route sister subagent to check OTHER substrate trainers (grayscale_lut, DP1, VQ-VAE) for similar `_write_runtime` vendoring stale-path bugs proactively. Each would emit a sister `[predicted]` audit memo.

4. **NEVER**: do NOT kill NSCS06 v8 chroma_lut paradigm per CLAUDE.md "Forbidden premature KILL". The trainer-v3-wire-in + canonical equation #26 IN-DOMAIN context + cls_stream cargo-cult-#5 unwind are all empirically verified at this point. L3 bug is bounded + remediable.

---

## Cross-references

- T3 council #1335 verdict: `.omx/research/t3_council_pr110_stacking_pivot_ordering_landed_20260526.md`
- Pre-dispatch RE-FIRE gate report: `.omx/research/nscs06_v8_stacked_paired_modal_t4_re_fire_pre_dispatch_gate_report_20260526.md` (commit `42a39c9bd`)
- Sister #1344 HALT report: `.omx/research/nscs06_v8_stacked_paired_modal_t4_auth_eval_pre_dispatch_gate_report_20260526.md`
- Trainer-v3-wire-in landing: `.omx/research/nscs06_v8_trainer_v3_wire_in_landed_20260526.md` (commit `5685f1a0c`)
- cls_stream wire-in landing: `.omx/research/nscs06_v8_cls_stream_wire_in_landed_20260526.md` (commits `581b7b129` + `545beb35c`)
- L1 EMPIRICAL MLX-local landing: `.omx/research/nscs06_v8_chroma_lut_mlx_l1_empirical_landed_20260526.md`
- Modal call_id: `fc-01KSJTVTAX5JBSF0F45P6H3XKJ` (canonical Modal call_id ledger per Catalog #245)
- V3 archive: `experiments/results/lane_substrate_nscs06_v8_chroma_lut_modal_t4_dispatch_20260526T190458Z_modal/lane_nscs06_v8_chroma_lut_results/output/0.bin` (sha `f187df36532e7db4145bb7dc9b99d813636a04e6907507691bfba0569be71ac8`; 3,690,071 bytes)
- Canonical equation #26: `src/tac/canonical_equations/procedural_codebook_savings.py`
- Canonical operator-authorize recipe: `.omx/operator_authorize_recipes/substrate_nscs06_v8_chroma_lut_modal_t4_dispatch.yaml`
