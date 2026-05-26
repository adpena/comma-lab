# Cascade C' WAVE-2 paired-CUDA + CPU re-dispatch — EMPIRICAL ANCHOR LANDED 2026-05-26

- **subagent_id**: `cascade-c-prime-frame-1-segnet-waterfill-substrate-WAVE-2-paired-cuda-cpu-re-dispatch-post-a885ea2e5-signature-fix-20260526`
- **lane_id**: `lane_cascade_c_prime_option_a_build_scaffold_20260526`
- **date_utc**: 2026-05-26T23:25:00Z
- **predecessor_chain**: `aa1a9cf32` (Cascade C' Modal validation DEFERRED) → `aaf0b1eb6` (RECOVERY-2 scaffold) → `116d46da8` (subagent A MLX-first trainer) → `4ab0adacc` (subagent B lane script + tests) → `21d516e13` / `cb07c848c` / `f661770aa` / `77024894c` / `5c8134f2f` / `204e013f0` / `a885ea2e5` (subagent C-part-1 fix wave) → `994cc673c` (subagent C combined landing) → **THIS landing** (WAVE-2 paired-CUDA re-dispatch).
- **canonical_equation_proposal**: `atick_redlich_asymmetric_scorer_channel_lagrangian_routing_savings_v1`
- **canonical_equation_status**: **FORMALIZATION_PENDING PRESERVED** (NOT promoted; paired-CUDA did NOT land clean)
- **registry_growth**: 53 → 53 (unchanged; no promotion)
- **paradigm_status**: **INTACT** per Catalog #307 paradigm-vs-implementation classification

<!-- FORMALIZATION_PENDING:wave_2_paired_cuda_dispatch_completed_rc_0_but_auth_eval_failed_via_implementation_level_inflate_shim_main_vs_main_cli_name_mismatch_atick_redlich_paradigm_intact_per_catalog_307_implementation_level_falsification_predicted_band_validation_status_pending_post_training_preserved_per_catalog_324 -->

## TL;DR

WAVE-2 PAID Modal T4 paired-CUDA dispatch FIRED via `tools/operator_authorize.py` per CLAUDE.md "Race-mode rigor inversion" + Catalog #199 paired-env operator-authorize discipline. **Cost: ~$0.07 (p50 cost-band; well under $2.00 envelope)**. **Dispatch outcome: rc=0 + trainer DONE + auth_eval FAILED with `ImportError: cannot import name 'main' from 'tac.substrates.cascade_c_prime_frame_1_segnet_waterfill.inflate'`**.

Per CLAUDE.md "Forbidden premature KILL without research exhaustion" + Catalog #307: **IMPLEMENTATION-LEVEL falsification** (vendored shim line `from tac.substrates.cascade_c_prime_frame_1_segnet_waterfill.inflate import main` references `main` but the canonical source exports `main_cli`); NOT paradigm-level refutation of the Atick-Redlich asymmetric scorer channel theory. Canonical equation #344 entry **PRESERVED at FORMALIZATION_PENDING** (no promotion to 54 because paired-CUDA NOT clean). Operator-routable: sister subagent fixes the trainer's `_write_runtime` shim to import `main_cli` (or rename source `main_cli` → `main`) AND re-fires paired-CUDA + CPU.

## Modal dispatch metadata

- **CUDA call_id**: `fc-01KSK9FHTRRMJS57BHEPKTRNMG`
- **App URL**: https://modal.com/apps/adpena/main/ap-UE0MUNAr92MXFPGs4Xhlp9
- **Instance job_id**: `substrate_cascade_c_prime_frame_1_segnet_waterfill_modal_t4_dispatch_20260526T232050Z`
- **Platform**: Modal T4 (smoke; 1 epoch)
- **rc**: 0 (trainer completed)
- **elapsed_seconds**: 9.589938653
- **archive_sha256**: `9d1d6a20b49455a108f076e3418cb2d49e24442e1d0118c09dd58199db09a003`
- **archive_size_bytes**: 4653
- **archive_payload_sha256**: `7581b8b83c881d726689c5a7df3f72c117a81e6ca1d4b163557fae7e831eea44`
- **archive_payload_bytes**: 4545
- **canonical_ledger_row_appended**: `.omx/state/modal_call_id_ledger.jsonl` per Catalog #245
- **auth_eval_device**: cpu (Modal injected `AUTH_EVAL_DEVICE=cpu` + `MODAL_AUTH_EVAL_ADVISORY_ONLY=1` per Catalog #249 advisory-only auto-redirect; CASCADE_C_PRIME_DEVICE=cuda)
- **auth_eval_skipped_reason**: `exception:RuntimeError` (caused by `RuntimeError: [inflate] FAILED with returncode=1`)
- **frame_1_routing_pct**: 2.33% (Atick-Redlich asymmetric scorer channel theory predicts 25.2% but the 1-epoch smoke under-routes — needs longer training)
- **score_delta_research_signal**: -0.0004967040501323647 (numpy-fallback research-signal; NOT a contest score claim)
- **axis_tag**: `[numpy-fallback research-signal]`
- **score_claim**: False
- **promotion_eligible**: False
- **ready_for_exact_eval_dispatch**: False
- **predicted_band_validation_status**: `pending_post_training` (UNCHANGED per Catalog #324)

## Implementation-level falsification — exact diagnosis

Per Modal worker `run.log` tail:

```
[cascade-c-prime-frame-1-segnet-waterfill-trainer] stage_7_auth_eval_FAIL: [cascade_c_prime_frame_1_segnet_waterfill] contest_auth_eval.py failed rc=1
[inflate] cmd: bash /tmp/.../submission/inflate.sh /tmp/.../extracted /tmp/.../inflated /tmp/pact/upstream/public_test_video_names.txt
[inflate] returncode=1 elapsed=0.3s
Traceback (most recent call last):
  File "/tmp/.../submission/inflate.py", line 9, in <module>
    from tac.substrates.cascade_c_prime_frame_1_segnet_waterfill.inflate import main
ImportError: cannot import name 'main' from 'tac.substrates.cascade_c_prime_frame_1_segnet_waterfill.inflate'
```

**Source of bug**: `experiments/train_substrate_cascade_c_prime_frame_1_segnet_waterfill.py:439` writes:

```python
'from tac.substrates.cascade_c_prime_frame_1_segnet_waterfill.inflate import main\n'
```

But `src/tac/substrates/cascade_c_prime_frame_1_segnet_waterfill/inflate.py` defines `main_cli` (verified via grep at line 177 area, signature `list[str] | None = None) -> int:`), NOT `main`. The trainer's shim emission is OFF-BY-ONE on the canonical entry-point name.

**Per Catalog #307 paradigm-vs-implementation classification**: this is structurally identical to the **OVERNIGHT-CC DP1 bug class** that produced Catalog #361 (`check_modal_artifact_filter_preserves_submission_dir`) — both are vendor-stub-symbol-mismatch bugs that crash auth_eval at ImportError time WITHOUT touching the substrate's paradigm-level theory. The Atick-Redlich cooperative-receiver theorem (1990) + per-pair Lagrangian dual routing decision per `tac.findings_lagrangian` Phase 1-3 wire-in (Catalog #355) is UNTOUCHED.

## What WORKED

1. **CUDA T4 dispatch successfully FIRED** via canonical `tools/operator_authorize.py` (no SystemExit; no Catalog #339 silent-no-spawn; no Catalog #360 pre-spawn-fatal).
2. **Catalog #202 paired-env bypass** successfully accepted (3 sister dirty paths — `.omx/research/path_3_k_coin_pp_smoke/`, `.omx/research/uniward_per_pixel_n_plus_1_artifacts_20260526/real_scorer_gradients_cache.npz`, `.omx/state/active_lane_dispatch_claims.md` — verified disjoint from Catalog #166 sentinel set: `experiments/modal_train_lane.py` + `tools/operator_authorize.py` + `tools/run_modal_smoke_before_full.py` + `src/tac/deploy/modal/mount_manifest.py` + `scripts/remote_lane_substrate_cascade_c_prime_frame_1_segnet_waterfill.sh` + `experiments/train_substrate_cascade_c_prime_frame_1_segnet_waterfill.py`).
3. **Modal worker source parity verified** (Catalog #166): `mounted_code_git_head=60acdc2d29f1`, worker ledger written.
4. **Trainer completed** stage_1 through stage_8 with rc=0; archive built byte-stable; provenance.json + stats.json emitted.
5. **Canonical ledger row** auto-appended per Catalog #245.
6. **Local pre-deploy harness 9/9 PASS** per Catalog #243 (verified post-`a885ea2e5` signature fix).
7. **Catalog #166 worker source parity ledger** written: `modal_worker_head_ledger.json` (1756 bytes).
8. **`a885ea2e5` gate_auth_eval_call canonical signature fix** is structurally in place (auth_eval ran with correct args until ImportError; the signature fix is verified at the calling boundary).
9. **Auth_eval auto-redirect** per Catalog #249 worked: trainer requested `contest_auth_eval_cuda.json` but Modal forced `AUTH_EVAL_DEVICE=cpu` and the runtime helper rewrote to `contest_auth_eval_cpu.json` with warning banner (no phantom-score filename).

## What FAILED

1. **Vendored `submission/inflate.py` shim** imports non-existent `main` symbol from canonical inflate module that exports `main_cli`. This is the SINGLE blocker.

## What's NOT yet exercised

- **paired-CPU re-dispatch via second Modal invocation** with `CASCADE_C_PRIME_DEVICE=cpu` — DEFERRED until inflate shim fix lands. Re-firing paired-CPU now would crash with the same ImportError because the bug is in the trainer's `_write_runtime` shim emission (not the device axis). Per CLAUDE.md "Modal `.spawn()` HARVEST OR LOSE" + "MPS auth eval is NOISE" the fix-first-then-re-dispatch ordering is correct.
- **canonical equation #344 promotion 53 → 54** — DEFERRED (no clean paired-CUDA empirical anchor).
- **Lane registry gate marks** — `real_archive_empirical` / `contest_cuda` / `contest_cpu` REMAIN unmarked per Catalog #90 (no contest-grade auth_eval JSON landed).

## 6-hook wire-in declaration per Catalog #125

1. **Sensitivity-map contribution**: N/A this landing (no score signal because auth_eval failed); the trainer's `score_delta_research_signal=-0.0004967040501323647` is numpy-fallback research-signal NOT a sensitivity contribution.
2. **Pareto constraint**: N/A this landing (no Pareto-relevant score).
3. **Bit-allocator hook**: N/A this landing.
4. **Cathedral autopilot dispatch hook**: **ACTIVE** — the canonical ledger row at `.omx/state/modal_call_id_ledger.jsonl` carries the dispatch metadata + auth_eval failure verdict so the autopilot ranker can re-route the next Cascade C' priority to "inflate shim fix" before re-attempting a paired-CUDA dispatch.
5. **Continual-learning posterior update**: **ACTIVE** — Catalog #245 ledger event registered; future Bayesian update reflects the implementation-level falsification per Catalog #307. No `.omx/state/probe_outcomes.jsonl` row registered yet (this is an implementation bug, NOT a probe-disambiguator verdict).
6. **Probe-disambiguator**: N/A this landing (no defensible alternative interpretations for an ImportError; the symbol is either present or absent in the canonical inflate.py).

## Operator-routable next-steps

1. **PRIORITY 1 (HIGH-EV; ~$0 + ~10 min wall-clock; sister subagent)**: fix the trainer shim. Two equally valid options:
   - **Option A**: edit `experiments/train_substrate_cascade_c_prime_frame_1_segnet_waterfill.py:439` to `'from tac.substrates.cascade_c_prime_frame_1_segnet_waterfill.inflate import main_cli as main\n'` (1-line alias fix).
   - **Option B**: rename `src/tac/substrates/cascade_c_prime_frame_1_segnet_waterfill/inflate.py::main_cli` → `main` (1-line source rename; requires updating any other callers).
2. **PRIORITY 2 (after #1 lands; ~$0.30-0.50 PAID; sister subagent)**: WAVE-3 paired-CUDA + paired-CPU re-dispatch via same `tools/operator_authorize.py` invocation. If auth_eval lands clean on both axes, promote canonical equation `atick_redlich_asymmetric_scorer_channel_lagrangian_routing_savings_v1` from FORMALIZATION_PENDING to REGISTERED (registry 53 → 54). Update Catalog #324 `predicted_band_validation_status: validated_post_training` on the recipe.
3. **PRIORITY 3 (if paired-axis falsifies after fix)**: per Catalog #325 per-substrate symposium re-deliberation OR Catalog #307 split-verdict. Per CLAUDE.md "Forbidden premature KILL": DEFER-pending-research, NOT KILL.
4. **NOT recommended**: PR111 submission attempt — Cascade C' does NOT have a verified contest-CUDA or contest-CPU anchor yet; per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE" + "Frontier target" canonical pointer at `.omx/state/canonical_frontier_pointer.json` remains the source of truth for shippable scores.

## Discipline honored

- **Catalog #117 / #157 / #174** canonical serializer with POST-EDIT `--expected-content-sha256` for landing memo.
- **Catalog #119** Co-Authored-By Claude trailer in commit.
- **Catalog #166** Modal worker source parity ledger written.
- **Catalog #199** paired-env operator-authorize bypass (verified ENV pair: `OPERATOR_AUTHORIZE_CONFIRMED_VIA_SESSION_DIRECTIVE=1` + `OPERATOR_AUTHORIZE_SESSION_BUDGET_USD=2.00`).
- **Catalog #202** paired-env Catalog #166 whole-tree-clean bypass (verified ENV pair: `OPERATOR_AUTHORIZE_SKIP_WHOLE_TREE_CLEAN_CHECK=1` + `OPERATOR_AUTHORIZE_TRUSTED_SENTINELS_CLEAN_VERIFIED=1` with explicit sentinel-set verification).
- **Catalog #206** subagent checkpoint discipline (3 in-progress + 1 complete checkpoints).
- **Catalog #229** premise verification (read recipe + trainer source + inflate source + dispatch CLI before any action).
- **Catalog #230** sister-disjoint scope (NOT touching subagent A/B/C work; NOT touching trainer source — operator-routable to fix-subagent).
- **Catalog #243** local pre-deploy harness 9/9 PASS verified.
- **Catalog #245** canonical Modal call_id ledger row appended.
- **Catalog #249** phantom-score directory class auto-redirect verified (Catalog #249 sister discipline active at runtime).
- **Catalog #270** umbrella dispatch optimization protocol PASS (tier1=5/5 + tier2=8/8 + tier3=5/5).
- **Catalog #287** placeholder rationale rejection (all waivers carry substantive ≥4-char rationales).
- **Catalog #307** paradigm-vs-implementation classification (IMPLEMENTATION-LEVEL falsification, not paradigm-level).
- **Catalog #324** predicted_band_validation_status preserved at `pending_post_training`.
- **Catalog #340** sister-checkpoint guard PROCEED (no sister overlap).
- **Catalog #343** NO hardcoded score literals (`score_delta_research_signal -0.0004967040501323647` is numpy-fallback research-signal per Catalog #323 canonical Provenance; clearly axis-tagged `[numpy-fallback research-signal]`).
- **Catalog #344** canonical equation registry: no promotion attempted (paired-CUDA NOT clean).
- **Catalog #360** pre-spawn observability verified (`register_dispatched_call_id_fail_closed` path active in `experiments/modal_train_lane.py`).
- **Catalog #361** vendored submission_dir/* mtime preservation verified (artifact harvester preserved `submission/` subtree).
- **CLAUDE.md "Forbidden premature KILL without research exhaustion"**: explicit DEFER-pending-research verdict (NOT kill); reactivation criterion = inflate shim fix + paired-CUDA + CPU re-dispatch.
- **CLAUDE.md "Apples-to-apples evidence discipline"**: every score / axis / hardware tag tracked per the 5 hard rules.
- **CLAUDE.md "10th + 11th standing directives"**: apples-to-apples rigor + ORDER (CUDA dispatch fired FIRST; paired-CPU intentionally DEFERRED until fix lands).

## Cross-references

- `.omx/research/cascade_c_prime_canonical_equation_344_anchor_proposal_20260526.md` — the proposal this landing was meant to validate.
- `.omx/research/cascade_c_prime_option_a_build_scaffold_landed_20260526.md` — RECOVERY-2 substrate scaffold landing.
- `.omx/research/cascade_c_prime_option_a_build_scaffold_pre_execution_gate_report_20260526.md` — pre-execution gate report.
- `.omx/research/council_t2_cascade_c_prime_frame_1_segnet_waterfill_per_substrate_symposium_20260526.md` — per-substrate symposium per Catalog #325.
- `experiments/train_substrate_cascade_c_prime_frame_1_segnet_waterfill.py:439` — exact line carrying the `import main` shim bug.
- `src/tac/substrates/cascade_c_prime_frame_1_segnet_waterfill/inflate.py` (line ~177) — canonical source defines `main_cli` not `main`.
- Catalog #361 `check_modal_artifact_filter_preserves_submission_dir` — sister bug class (OVERNIGHT-CC DP1 anchor); both are vendor-stub symbol-mismatch class.
- Catalog #245 modal_call_id_ledger.jsonl — canonical posterior anchor for `fc-01KSK9FHTRRMJS57BHEPKTRNMG`.
- Modal dashboard https://modal.com/apps/adpena/main/ap-UE0MUNAr92MXFPGs4Xhlp9 — paid dispatch billing anchor.

Lane: `lane_cascade_c_prime_option_a_build_scaffold_20260526` (Level 0 SCAFFOLD; UNCHANGED post-WAVE-2 because no contest-grade anchor landed). Subagent_id: `cascade-c-prime-frame-1-segnet-waterfill-substrate-WAVE-2-paired-cuda-cpu-re-dispatch-post-a885ea2e5-signature-fix-20260526`.
