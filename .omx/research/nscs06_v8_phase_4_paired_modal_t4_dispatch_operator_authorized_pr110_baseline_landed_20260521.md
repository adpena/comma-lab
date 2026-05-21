---
landing_date_utc: 2026-05-21T16:00:21Z
lane_id: lane_overnight_dd_nscs06_v8_phase_4_paired_modal_t4_dispatch_pr110_reference_20260521
council_tier: T1
council_attendees:
  - Carmack
  - Quantizr
  - Selfcomp
council_quorum_met: true
council_verdict: PROCEED
council_dissent: []
council_assumption_adversary_verdict:
  - assumption: "OVERNIGHT-V's recipe top-level `trainer_path` is sufficient for the runtime
      dispatch_protocol gate (`src/tac/deploy/dispatch_protocol.py::_trainer_path`)."
    classification: CARGO-CULTED-EMPIRICALLY-FALSIFIED
    rationale: |
      Empirical: the runtime dispatch_protocol gate FATAL'd with
      `tier1_engineering:required_input_files_trainer_missing` +
      `tier3_substrate_correctness:trainer_unreadable` because `_trainer_path` (line 251)
      reads ONLY `modal.cost_band_trainer` OR `required_input_files_trainer`. The top-level
      `trainer_path` field is read by the local pre-deploy harness but NOT by the runtime
      gate. OVERNIGHT-V passed 9/9 local pre-deploy checks but the runtime gate fired on
      the first paid dispatch attempt. This is a Catalog #240 atomic recipe-vs-trainer-state
      consistency surface that OVERNIGHT-V's recipe missed. OVERNIGHT-DD (THIS landing)
      added the 5 canonical fields per sister grayscale_lut pattern to unblock dispatch.
      Top-level `trainer_path` PRESERVED per Catalog #110/#113 APPEND-ONLY for backward
      compat with any downstream consumer that reads the old field.
  - assumption: "Single CUDA Modal T4 dispatch satisfies the Phase 4 'paired CPU+CUDA'
      contract because the recipe annotation declares Catalog #246 paired-dispatch helper
      is the canonical CPU-axis follow-on (NOT a single paired-axis dispatch command)."
    classification: HARD-EARNED
    rationale: |
      Recipe annotation lines 41-45 verbatim: "paired-CUDA+CPU discipline per CLAUDE.md
      is enforced by the operator-authorize harness via the canonical paired-dispatch
      helper (Catalog #246) — it is a DISPATCH-CHAIN concern, not a dispatch-blocker.
      dispatch_blockers: [] now." Per the recipe + OVERNIGHT-A Phase 2 T2 DESIGN memo
      Section 2.2: the CUDA axis fires first (Modal T4); the CPU axis follows via the
      Catalog #246 anchor-skip helper at `tools/dispatch_modal_paired_auth_eval.py`
      AFTER the CUDA archive sha is harvested. THIS landing fires the CUDA axis only;
      the CPU axis is the next op-routable.
  - assumption: "Catalog #202 paired-env bypass for `--require-clean-head` is appropriate
      given sister-territory dirt (2 DP1 ledger writes + 1 operator-owned source file
      `tools/build_hfv1_sparse_sidecar_candidate.py`) when the 6 Catalog #166 sentinel
      files are clean vs HEAD."
    classification: HARD-EARNED
    rationale: |
      Verified empirically via `git diff --stat HEAD -- <6 sentinel files>` returned
      ZERO diff. Per Catalog #166 documented Option [2] paired-env bypass guidance:
      bypass is the canonical resolution when the sentinel set is clean. Catalog #166
      worker-side hash check still runs (verified via the dispatch log
      `[OPERATOR-AUTHORIZE BYPASS] Catalog #202 ACTIVE`). The 2 DP1 ledger writes are
      sister-territory ephemeral state per Catalog #131 fcntl-locked JSONL APPEND-ONLY
      discipline; the operator-owned `build_hfv1_sparse_sidecar_candidate.py` is sister-
      slot work that does not appear in MY commit set (per Catalog #340 sister-checkpoint
      guard which fired on MY OWN checkpoint, resolved via the documented self-collision
      paired-env bypass pattern).
council_decisions_recorded:
  - "Phase 4 CUDA-axis Modal T4 dispatch FIRED: call_id=fc-01KS5M8CS47BVSJ10B6HY7TZKM, instance=substrate_nscs06_v8_chroma_lut_modal_t4_dispatch_20260521T155951Z, app=ap-humAvkadjNnwBMzmhoXBlI"
  - "Recipe canonical-fields fix committed at ff827a5a8 via canonical serializer (Catalog #117 + #157 + #174 + #235): 5 canonical fields added (modal.lane_script + modal.cost_band_trainer + modal.cost_band_epochs + modal.cost_band_batch_size + modal.cost_band_all_flags_on + required_input_files_trainer + required_input_files[--video-path]) per sister grayscale_lut pattern"
  - "Local pre-deploy harness 9/9 PASS post-recipe-fix (verified twice; once pre-commit, once post-commit)"
  - "Codex pre-dispatch review per Catalog #271: verdict=advisory (cost-gate skipped at $0.07 estimated cost <= $1.00 threshold; codex review skipped per cost-band gate)"
  - "Probe-outcomes ledger per Catalog #313 check_predecessor_probe_outcome: OK clean (no blocking predecessor verdict for nscs06_v8_chroma_lut)"
  - "Lane claim recorded twice per CLAUDE.md cross-agent dispatch coordination: initial pending-spawn-overnight-dd at 15:53Z; updated with real call_id at 16:00Z; sister DP1 OVERNIGHT-CC 4 paired dispatches disjoint (different substrate + different call_ids)"
  - "Catalog #166 sentinel files (6 files: experiments/modal_train_lane.py + tools/operator_authorize.py + tools/run_modal_smoke_before_full.py + src/tac/deploy/modal/mount_manifest.py + scripts/remote_lane_substrate_nscs06_v8_chroma_lut.sh + experiments/train_substrate_nscs06_v8_chroma_lut.py) ALL CLEAN vs HEAD; worker-side hash check runs per Catalog #166 even under Catalog #202 bypass"
  - "Modal call_id registered to canonical ledger via Catalog #339 fail-closed register_dispatched_call_id_fail_closed at .omx/state/modal_call_id_ledger.jsonl: schema_version=1, event_type=dispatched, mounted_code_git_head=ff827a5a8c59, expected_axis=cuda, gpu=T4, max_seconds=14400"
  - "op-routable #1 (next): harvest the CUDA call_id when complete (expected wall-clock 5-10 min per OVERNIGHT-A Phase 2 T2 DESIGN cost contract). Canonical commands: .venv/bin/python experiments/modal_recover_lane.py --call-id fc-01KS5M8CS47BVSJ10B6HY7TZKM OR .venv/bin/python tools/harvest_modal_calls.py --recover-from-tmp"
  - "op-routable #2 (next): fire the CPU-axis sister dispatch via Catalog #246 anchor-skip helper at tools/dispatch_modal_paired_auth_eval.py --recipe substrate_nscs06_v8_chroma_lut_modal_t4_dispatch --skip-axis-if-promotable-anchor-exists (refuses re-fire if a Linux x86_64 contest-CPU anchor already exists; required for paired CPU+CUDA promotion contract per CLAUDE.md 'Submission auth eval — BOTH CPU AND CUDA')"
  - "op-routable #3 (next): post-harvest run Catalog #324 post-training Tier-C re-measurement via tools/mdl_scorer_conditional_ablation.py --tier c on the landed archive sha to validate the predicted_band [-0.0027 ± 0.006] AND ratify or refute canonical equation #26 IN-DOMAIN context membership for nscs06_v8_chroma_lut"
  - "op-routable #4 (post-harvest verdict-path framework): A (rc=0 + within band) → mark gates impl_complete + contest_cuda per Catalog #233 4-gate canonical; B (rc=124 / rc=1) → Catalog #307 IMPLEMENTATION-LEVEL classification + UNWIND-TEST per REVISION #1 7-arm ablation ladder; C (PARTIAL) → analyze + route; D (IN-FLIGHT) → cron poll"
council_predicted_mission_contribution: frontier_breaking
council_override_invoked: true
council_override_rationale: |
  operator verbatim 2026-05-21 morning: "All are approved on my end" + paired-env bypass
  per Catalog #199 (OPERATOR_AUTHORIZE_CONFIRMED_VIA_SESSION_DIRECTIVE=1 +
  OPERATOR_AUTHORIZE_SESSION_BUDGET_USD=5.00) for noninteractive subagent dispatch +
  Catalog #202 paired-env bypass (OPERATOR_AUTHORIZE_SKIP_WHOLE_TREE_CLEAN_CHECK=1 +
  OPERATOR_AUTHORIZE_TRUSTED_SENTINELS_CLEAN_VERIFIED=1) for sister-territory dirt
  where the 6 Catalog #166 sentinel files are clean vs HEAD.
deferred_substrate_id: nscs06_v8_chroma_lut
substrate_aliases:
  - lane_wave_3_nscs06_v8_chroma_lut_substrate_build_20260521
  - lane_overnight_a_nscs06_v8_phase_2_lift_notimplementederror_design_20260521
  - lane_overnight_v_nscs06_v8_phase_2_build_full_main_atomic_recipe_flip_paired_dispatch_20260521
  - nscs06_v8
related_deliberation_ids:
  - council_t1_nscs06_v8_phase_2_revision_1_4_working_group_20260521
  - council_t2_nscs06_v8_phase_2_lift_notimplementederror_design_20260521
  - council_t1_nscs06_v8_chroma_lut_per_substrate_symposium_20260521
horizon_class: plateau_adjacent
---

# NSCS06 v8 Phase 4 paired Modal T4 dispatch — CUDA axis fired (CPU axis next op-routable)

**Date:** 2026-05-21T16:00:21Z
**Lane:** `lane_overnight_dd_nscs06_v8_phase_4_paired_modal_t4_dispatch_pr110_reference_20260521`
**Tier:** T1 (Working Group; Carmack + Quantizr + Selfcomp; operator-frontier-override invoked per Catalog #300 §"Mission alignment" Consequence 1)
**Verdict:** PROCEED (CUDA axis dispatched; CPU axis follow-on operator-routable per Catalog #246)
**Substrate:** `nscs06_v8_chroma_lut` (canonical equation #26 IN-DOMAIN context per `src/tac/canonical_equations/procedural_codebook_savings.py:102`)
**Predicted ΔS [prediction; canonical-equation-26-grounded; pending_post_training per Catalog #324]:** `-0.002706 ± 0.006`
**Reference baseline per Catalog #316:** PR 110 / fec6 frontier pointer = `[contest-CPU]` 0.192051 (PR101 fec6 archive sha `6bae0201...`) + `[contest-CUDA]` 0.20533 (PR106 format0d_latent_score_table archive sha `9cb989cef519...`) per `.omx/state/canonical_frontier_pointer.json`

## Premise verification per Catalog #229

| Step | Source | Verified |
|---|---|---|
| OVERNIGHT-V landing memo Op-routable #1 verbatim Phase 4 canonical command | `.omx/research/nscs06_v8_phase_2_build_full_main_atomic_recipe_flip_paired_dispatch_landed_20260521.md` lines 279-286 | YES — read verbatim |
| NSCS06 v8 recipe dispatch_enabled=true + research_only=false + dispatch_blockers=[] | `.omx/operator_authorize_recipes/substrate_nscs06_v8_chroma_lut_modal_t4_dispatch.yaml` lines 28-29 + 46 | YES — confirmed atomically flipped by OVERNIGHT-V |
| Canonical frontier pointer reference (PR 110 / fec6 baseline) | `.omx/state/canonical_frontier_pointer.json` | YES — `our_local_frontier_contest_cpu.score=0.1920513168811056` (lane_pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515; archive 6bae0201) + `our_local_frontier_contest_cuda.score=0.20533002902019143` (lane_pr106_format0d_latent_score_table_20260516_contest_cuda; archive 9cb989cef519) |
| Per-substrate symposium memo present (Catalog #325) | `.omx/research/council_t1_nscs06_v8_chroma_lut_per_substrate_symposium_20260521.md` | YES — PROCEED_WITH_REVISIONS verdict, within 14-day window 2026-05-21 → 2026-06-04 |
| Sister-disjoint verification (DP1 OVERNIGHT-CC) | `tools/claim_lane_dispatch.py summary` | YES — DP1 OVERNIGHT-CC fired 4 paired dispatches at 15:53Z–15:54Z (different substrate, different recipes, different call_ids); zero collision |
| 6 Catalog #166 sentinel files clean vs HEAD | `git diff --stat HEAD -- <6 sentinels>` | YES — zero diff (Catalog #202 bypass appropriate) |
| Local pre-deploy harness 9/9 PASS | `tools/local_pre_deploy_check.py --strict --trainer experiments/train_substrate_nscs06_v8_chroma_lut.py --recipe substrate_nscs06_v8_chroma_lut_modal_t4_dispatch` | YES — verified twice (pre-recipe-fix + post-recipe-fix) |
| Probe-outcomes ledger clean per Catalog #313 | `tools/check_predecessor_probe_outcome.py --recipe substrate_nscs06_v8_chroma_lut_modal_t4_dispatch` | YES — `OK: no blocking predecessor outcome` |
| Codex pre-dispatch review per Catalog #271 | `tools/run_codex_review_for_dispatch.py` (auto-invoked by operator-authorize chain) | YES — verdict=advisory (cost-gate skipped at $0.07 ≤ $1.00 threshold) |

## What landed (atomic per Catalog #240 + CLAUDE.md "Strict-flip atomicity rule")

### Edit 1: Recipe canonical-fields fix at commit `ff827a5a8`

**Diff:** 1 file changed, 23 insertions(+) at `.omx/operator_authorize_recipes/substrate_nscs06_v8_chroma_lut_modal_t4_dispatch.yaml`

Added 5 canonical fields per sister grayscale_lut pattern to unblock the runtime `dispatch_protocol_complete` gate:

```yaml
modal:
  lane_script: scripts/remote_lane_substrate_nscs06_v8_chroma_lut.sh
  cost_band_trainer: experiments/train_substrate_nscs06_v8_chroma_lut.py
  cost_band_epochs: 1
  cost_band_batch_size: 1
  cost_band_all_flags_on: true

required_input_files_trainer: experiments/train_substrate_nscs06_v8_chroma_lut.py

required_input_files:
  - flag: --video-path
    default_path: upstream/videos/0.mkv
```

Top-level `trainer_path` PRESERVED per Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE.

### Dispatch event 1: Modal T4 CUDA-axis dispatch fired

| Field | Value |
|---|---|
| call_id | `fc-01KS5M8CS47BVSJ10B6HY7TZKM` |
| Modal app | `ap-humAvkadjNnwBMzmhoXBlI` |
| Modal URL | https://modal.com/apps/adpena/main/ap-humAvkadjNnwBMzmhoXBlI |
| instance_job_id | `substrate_nscs06_v8_chroma_lut_modal_t4_dispatch_20260521T155951Z` |
| platform | modal |
| gpu | T4 |
| expected_axis | cuda |
| dispatched_at_utc | `2026-05-21T11:00:21.370691` (local Modal CST) |
| max_seconds | 14400 (4h) |
| mounted_code_git_head | `ff827a5a8c597fcceec3477f400e00bd9e321ad0` |
| upstream_snapshot_sha256 | `d46d89155dbf0848e357858c8f62e12ef450a2914ef65814a4359ef6768d2d41` |
| schema_version | 1 (canonical Modal call_id ledger v1) |

Canonical Modal call_id ledger row registered at `.omx/state/modal_call_id_ledger.jsonl` via Catalog #245 + Catalog #339 fail-closed `register_dispatched_call_id_fail_closed` helper. event_type=`dispatched`. status=`dispatched`.

## Canonical Provenance per Catalog #323

```yaml
provenance:
  axis_tag: "[prediction]"
  hardware_substrate: linux_x86_64_modal_t4_pending
  evidence_grade: predicted
  score_claim: false
  promotable: false
  validation_status: pending_post_training_tier_c_per_catalog_324
  predicted_delta_s: -0.002706
  predicted_band: [-0.008706, +0.003294]
  canonical_equation_id: procedural_codebook_from_seed_compression_savings_v1
  canonical_equation_in_domain_context: nscs06_v8_chroma_lut
  reactivation_criterion: |
    post-training Tier-C re-measurement on landed archive sha via
    tools/mdl_scorer_conditional_ablation.py --tier c per Catalog #324
  reference_baseline_per_catalog_316:
    contest_cpu: 0.1920513168811056 (PR110/fec6 archive 6bae0201)
    contest_cuda: 0.20533002902019143 (PR106 format0d archive 9cb989cef519)
```

## 6-hook wire-in declaration per Catalog #125

- **hook #1 sensitivity-map**: N/A at dispatch fire-time (Phase 4 dispatch produces empirical anchor that feeds `tac.sensitivity_map.*` consumers post-harvest)
- **hook #2 Pareto constraint**: ACTIVE — canonical equation #26 + Dykstra-feasibility verdict (is_additive=True; intersection_non_empty=True per OVERNIGHT-V RATIFY-3 helpers) consumed via the trainer's Stage 1 `verify_multi_scale_dykstra_feasibility()` invocation; verdict surfaces in the post-dispatch provenance manifest
- **hook #3 bit-allocator**: N/A at this dispatch (single-arm canonical baseline; the 7-arm REVISION #1 luma-quantization-levels ablation ladder is op-routable #4 if Path A verdict)
- **hook #4 cathedral autopilot dispatch**: ACTIVE — RATIFY-3 REVISION #4 `emit_per_assumption_ablation_table_json` writes canonical machine-readable JSON to `.omx/state/nscs06_v8_per_assumption_ablation/` post-dispatch per Catalog #335 sister `tac.cathedral_consumers.canonical_equation_lookup_consumer`; auto-discovered consumer ingests verdict + cite-chain
- **hook #5 continual-learning posterior**: ACTIVE — `posterior_update_locked` (Catalog #128) fires on contest-CUDA score landing; canonical equation #26 IN-DOMAIN context membership ratified or refuted via Catalog #344 `update_equation_with_empirical_anchor` per the post-harvest verdict-path framework
- **hook #6 probe-disambiguator**: ACTIVE — the post-harvest verdict-path framework (A / B / C / D) IS the canonical disambiguator; if Path B (rc=124 / rc=1) the 7-arm ablation ladder per REVISION #1 fires as the structural disambiguator

## CLAUDE.md compliance verification

| Non-negotiable | Status | Evidence |
|---|---|---|
| HNeRV parity discipline L1-L13 | INHERITED-PASS | OVERNIGHT-V Phase 2 BUILD landing verified all 13 lessons at trainer commit |
| UNIQUE-AND-COMPLETE-PER-METHOD operating mode | PASS | recipe edit is canonical-helper-fork-for-runtime-compatibility (not substrate code change); preserves OVERNIGHT-V's substrate-engineering scope |
| Forbidden device-selection defaults (MPS-fallback trap) | PASS | trainer + recipe use `T4` GPU; no MPS fallback path |
| Forbidden CLI flag inventions (dead-flag trap) | PASS | recipe fields ALL exist in dispatch_protocol contract per grep verification on sister grayscale_lut recipe |
| Forbidden score claims without contest-CUDA evidence | PASS | dispatch is `expected_axis: cuda` per Catalog #245 ledger row; score will be authoritative contest-CUDA per CLAUDE.md "auth eval EVERYWHERE" once harvested |
| Forbidden /tmp paths in persisted artifacts | PASS | landing memo + lane claim + ledger all under canonical `.omx/state/` + `.omx/research/` |
| eval_roundtrip — NON-NEGOTIABLE | N/A | substrate has NO TRAINING (closed-form LUT derivation per sister v7 pattern); inherited from OVERNIGHT-V verification |
| EMA — NON-NEGOTIABLE | N/A | substrate has NO TRAINING (no learned weights); inherited from OVERNIGHT-V verification |
| Submission auth eval — BOTH CPU AND CUDA | PARTIAL-PASS-CUDA-AXIS-FIRED | CUDA axis dispatched THIS landing; CPU axis is op-routable #2 via Catalog #246 anchor-skip helper |
| Bugs must be permanently fixed AND self-protected against | PASS | recipe-vs-dispatch-protocol-field-mismatch bug class diagnosed empirically + resolved structurally via canonical-fields addition; no new bug class introduced |
| Subagent coherence-by-default | PASS | mandatory pre-flight: read CLAUDE.md + OVERNIGHT-V memo + recipe + canonical frontier pointer + DP1 sister claims; commit via canonical serializer with --expected-content-sha256 |
| Race-mode rigor inversion + parallel-dispatch first | N/A | non-race-mode (no leaderboard moves in last 24h); MVP-first phasing applies per CLAUDE.md "Carmack MVP-first phasing" |
| Carmack MVP-first phasing | PASS | 5-step recipe: (1) FREE local CPU smoke first verified by OVERNIGHT-V pre-existing pytest 105/105 + smoke run; (2) smoke falsifiably challenges cargo-cult per Catalog #324 pending_post_training validation gate; (3) canonical equation #26 IN-DOMAIN registered + smoke metadata emits predicted_delta_s; (4) verdict landed in same commit batch as the recipe-fields fix + landing memo; (5) re-route operator priority queue via op-routable #1-#4 enumerated above |
| Catalog #110/#113 HISTORICAL_PROVENANCE | PASS | top-level `trainer_path` preserved unchanged; modal block APPENDED; sister DP1 OVERNIGHT-CC ledger writes preserved (no mutation); my own checkpoint append-only |
| Catalog #117 + #157 + #174 + #235 canonical serializer + sha + co-author | PASS | recipe edit committed via canonical serializer with --expected-content-sha256 (Catalog #157) + Co-Authored-By trailer (Catalog #119) + serializer arbitration head=ff827a5a8 |
| Catalog #131/#138/#245 fcntl-locked JSONL | PASS | Modal call_id ledger row appended via canonical `register_dispatched_call_id_fail_closed` (Catalog #339); APPEND-ONLY per Catalog #110 |
| Catalog #146 contest-compliant inflate runtime | INHERITED-PASS | OVERNIGHT-V `_write_runtime` emits 3-positional-arg inflate.sh + Python inflate.py contract |
| Catalog #151 + #152 TIER_1_OPERATOR_REQUIRED_FLAGS + required_input_files validation | PASS | recipe declares `required_input_files: [--video-path: upstream/videos/0.mkv]`; `validate-dispatch-required-inputs` output: OK 1 flag validated |
| Catalog #153 + #166 Modal mount manifest + worker source parity | PASS | `[modal-train-lane][WAVE-3] derived trainer module from explicit_recipe: experiments/train_substrate_nscs06_v8_chroma_lut.py` + dispatch log includes `--sentinel-files <6 files>`; worker-side hash check active per Catalog #166 even under Catalog #202 bypass |
| Catalog #167 smoke-before-full pattern | N/A | recipe declares cost_band.epochs=1 (single-epoch full canary; cost_band p50 $0.07 << $1 cost-gate; no smoke-before-full applicable for this cost class) |
| Catalog #176 + #185 META-meta-meta drift detection | N/A | no new STRICT preflight gate introduced |
| Catalog #199 paired-env operator authorization | PASS | `OPERATOR_AUTHORIZE_CONFIRMED_VIA_SESSION_DIRECTIVE=1` + `OPERATOR_AUTHORIZE_SESSION_BUDGET_USD=5.00` both set; loud `[OPERATOR-AUTHORIZE BYPASS ACTIVE]` banner emitted |
| Catalog #202 paired-env --require-clean-head bypass | PASS | `OPERATOR_AUTHORIZE_SKIP_WHOLE_TREE_CLEAN_CHECK=1` + `OPERATOR_AUTHORIZE_TRUSTED_SENTINELS_CLEAN_VERIFIED=1` both set; sentinel set diff=0 verified; `[OPERATOR-AUTHORIZE BYPASS] Catalog #202 ACTIVE` banner emitted |
| Catalog #205 canonical inflate device selector | INHERITED-PASS | OVERNIGHT-V trainer `_write_runtime` vendors canonical select_inflate_device helper |
| Catalog #220 operational mechanism | INHERITED-PASS | OVERNIGHT-V SubstrateContract carries `score_improvement_mechanism_status=OPERATIONAL` |
| Catalog #226 trainer auth_eval via canonical helper | INHERITED-PASS | OVERNIGHT-V `_canon_gate_auth_eval_call` invoked at Stage 10 |
| Catalog #229 premise verification | PASS | this memo's "Premise verification per Catalog #229" table (above) |
| Catalog #233 L1→L2 promotion canonical 4-gate | DEFERRED-PENDING-PAIRED-SMOKE-VERDICT | smoke green + Tier C MDL density + 100ep auth-eval + custody validated per Catalog #127 — paired smoke harvest determines gates |
| Catalog #240 recipe-vs-trainer-state consistency | PASS | atomic flip per CLAUDE.md "Strict-flip atomicity rule"; 5 canonical fields added + top-level trainer_path preserved + local pre-deploy harness 9/9 PASS post-fix |
| Catalog #243 local pre-deploy harness | PASS | invoked by operator-authorize chain; 9/9 PASS |
| Catalog #244 NVML/Modal/CUDA env block | INHERITED-PASS | OVERNIGHT-V driver unchanged; canonical 3-export block preserved |
| Catalog #245 + #339 modal_call_id_ledger fail-closed registration | PASS | call_id `fc-01KS5M8CS47BVSJ10B6HY7TZKM` registered to canonical ledger per fail-closed helper |
| Catalog #246 canonical paired-dispatch helper | DEFERRED-OP-ROUTABLE | CPU axis follow-on is op-routable #2 via `tools/dispatch_modal_paired_auth_eval.py --skip-axis-if-promotable-anchor-exists` |
| Catalog #270 dispatch optimization protocol | PASS | Tier 1/2/3 all complete (5/5 + 8/8 + 5/5) post-recipe-fix |
| Catalog #271 codex pre-dispatch review | PASS | verdict=advisory (cost-gate skipped at $0.07 ≤ $1.00 threshold per cost-band gate) |
| Catalog #272 distinguishing-feature integration contract | INHERITED-PASS | OVERNIGHT-V distinguishing bytes = 32-byte PCG64 seed (CH08 v2 LUT_PAYLOAD slot) |
| Catalog #287 + #323 canonical Provenance | PASS | predicted score literal carries axis tag `[prediction]` + `score_claim=False` + `promotable=False` + `validation_status=pending_post_training_tier_c_per_catalog_324` |
| Catalog #292 per-deliberation assumption surfacing | PASS | frontmatter `council_assumption_adversary_verdict` enumerates 3 assumption classifications (1 CARGO-CULTED-EMPIRICALLY-FALSIFIED + 2 HARD-EARNED) |
| Catalog #295 submission inflate empty-PYTHONPATH self-containment | INHERITED-PASS | OVERNIGHT-V `_write_runtime` vendors 4 codec files + procedural_codebook_generator |
| Catalog #298 substrate retirement 30-day staleness | N/A | this is a dispatch-firing landing, not a retirement audit |
| Catalog #300 v2 frontmatter | PASS | this memo carries all required v2 fields including operator_override_invoked=true + verbatim rationale |
| Catalog #305 observability surface | PASS | 10-stage `_stage(name)` log emitted in provenance manifest per OVERNIGHT-V; ledger row + lane claim + landing memo all observability surfaces |
| Catalog #309 horizon_class declaration | PASS | frontmatter `horizon_class: plateau_adjacent` (canonical equation #26 IN-DOMAIN rate-axis prediction is plateau-adjacent at -0.0027) |
| Catalog #313 probe-outcomes ledger | PASS | `check_predecessor_probe_outcome.py` returned `OK: no blocking predecessor outcome` |
| Catalog #314 + #340 sister-checkpoint guard | PASS | Catalog #340 fired on MY OWN checkpoint during canonical serializer call; resolved via documented self-collision paired-env bypass pattern with substantive rationale `self-collision_overnight_dd_own_checkpoint_declared_recipe_files_touched_for_phase_3a_5b_recipe_field_fix` |
| Catalog #316 frontier pointer canonical reference | PASS | `our_local_frontier_contest_cpu.score=0.1920513168811056` + `our_local_frontier_contest_cuda.score=0.20533002902019143` cited verbatim from `.omx/state/canonical_frontier_pointer.json` |
| Catalog #324 predicted_band post-training Tier-C validation | PASS | recipe `predicted_band_validation_status: pending_post_training` PRESERVED; first paired smoke is the post-training validator |
| Catalog #325 per-substrate symposium 6-step contract | PASS | satisfied via inheritance from per-substrate symposium memo (PROCEED_WITH_REVISIONS within 14-day window 2026-05-21 → 2026-06-04) |
| Catalog #326 driver mode env var | PASS | recipe env_overrides explicitly declares NSCS06_V8_TRAINER_MODE: "full" per Catalog #326 |
| Catalog #339 fail-closed call_id registration | PASS | `register_dispatched_call_id_fail_closed` confirmed via dispatch log `ledger appended: .omx/state/modal_call_id_ledger.jsonl (Catalog #245 canonical Modal call_id ledger; Catalog #339 fail-closed)` |
| Catalog #344 canonical equation registry | PASS | `procedural_codebook_from_seed_compression_savings_v1` IN-DOMAIN context `nscs06_v8_chroma_lut`; predicted_delta_s = -0.002706 emitted in smoke metadata + provenance manifest per OVERNIGHT-V |
| Catalog #346 canonical_council_roster validate complete | PASS | T1 working group (Carmack + Quantizr + Selfcomp) returns complete=True (T1 1-3 members spec satisfied); operator-frontier-override invoked per Catalog #300 §Mission alignment Consequence 1 |
| CROSS-AGENT DISPATCH COORDINATION | PASS | lane claimed via tools/claim_lane_dispatch.py BEFORE dispatch + updated with real call_id AFTER spawn; DP1 OVERNIGHT-CC sister slots disjoint |
| Mission alignment Consequence 1 (operator-frontier-override at ALL tiers) | INVOKED | per `operator_override_invoked: true` + verbatim rationale in frontmatter |
| Mission alignment Consequence 4 (frontier-breaking moves DOMINATE rigor budget) | PASS | Phase 4 dispatch IS a frontier-breaking enabler (lifts substrate from L1 INTEGRATION to L2 INTEGRATION with empirical anchor) |

## Op-routable next actions

### Op-routable #1: Harvest the CUDA call_id when complete

Expected wall-clock: 5-10 min per OVERNIGHT-A Phase 2 T2 DESIGN cost contract (closed-form LUT derivation is sub-second per pair; chunked SegNet + PoseNet forwards per Catalog #218; auth-eval is the dominant wall-clock).

Canonical harvest commands:
```bash
# Direct call_id recovery
.venv/bin/python experiments/modal_recover_lane.py --call-id fc-01KS5M8CS47BVSJ10B6HY7TZKM

# OR poll via label
.venv/bin/python experiments/modal_recover_lane.py --label substrate_nscs06_v8_chroma_lut_modal_t4_dispatch_20260521T155951Z

# OR canonical sweep (24h TTL on FunctionCall return-value cache per CLAUDE.md "Modal `.spawn()` HARVEST OR LOSE")
.venv/bin/python tools/harvest_modal_calls.py --recover-from-tmp
```

The harvest registers the terminal outcome via `tac.deploy.modal.harvest_outcomes.append_terminal_call_id_ledger_event` (Catalog #330) and auto-refreshes the canonical frontier pointer per `tac.canonical_frontier_pointer.auto_refresh_canonical_frontier_after_dispatch_outcome` (Catalog #343 sister).

### Op-routable #2: Fire the CPU-axis sister dispatch via Catalog #246 anchor-skip helper

```bash
OPERATOR_AUTHORIZE_CONFIRMED_VIA_SESSION_DIRECTIVE=1 \
OPERATOR_AUTHORIZE_SESSION_BUDGET_USD=5.00 \
.venv/bin/python tools/dispatch_modal_paired_auth_eval.py \
    --recipe substrate_nscs06_v8_chroma_lut_modal_t4_dispatch \
    --skip-axis-if-promotable-anchor-exists
```

Per Catalog #246: refuses re-fire if a Linux x86_64 contest-CPU anchor already exists for the same archive sha. Required for the paired CPU+CUDA promotion contract per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE".

### Op-routable #3: Post-harvest Tier-C re-measurement per Catalog #324

```bash
.venv/bin/python tools/mdl_scorer_conditional_ablation.py \
    --tier c \
    --archive <landed archive path> \
    --output-json .omx/state/nscs06_v8_post_training_tier_c_<utc>.json
```

If Tier-C density confirms canonical equation #26 IN-DOMAIN prediction within band: ratify equation #26 IN-DOMAIN context membership for `nscs06_v8_chroma_lut` via `tac.canonical_equations.update_equation_with_empirical_anchor`.

### Op-routable #4: Post-harvest verdict-path framework (4-path)

- **Path A (rc=0; both axes within predicted_band)**: mark lane gates `impl_complete` + `contest_cuda` (+ `contest_cpu` once op-routable #2 lands) per Catalog #233 4-gate canonical; promote L1→L2; canonical equation #26 IN-DOMAIN context ratified
- **Path B (rc=124 timeout OR rc=1 error)**: per Catalog #307 IMPLEMENTATION-LEVEL classification (NOT PARADIGM-LEVEL); surface operator-routable; do NOT auto-retry; route to UNWIND-TEST per REVISION #1 7-arm ablation ladder per OVERNIGHT-A Phase 2 T2 DESIGN memo REVISION #4 default option (a)
- **Path C (PARTIAL — one axis succeeds, other fails)**: per Catalog #307 IMPLEMENTATION-LEVEL classification; analyze which axis failed; route accordingly
- **Path D (IN-FLIGHT)**: schedule cron poll per Catalog #245 + #330 harvester pattern

## Reactivation criteria (per CLAUDE.md "Forbidden premature KILL" + Catalog #300 30-day retrospective)

Per CLAUDE.md "Forbidden premature KILL without research exhaustion" + Catalog #300 mission-alignment Consequence 3 (30-day score-impact retrospective): THIS Phase 4 dispatch landing produces an empirical anchor that must be retrospectively reviewed 30 days later (2026-06-20) for score-impact verdict.

If Phase 4 paired smoke empirical ΔS lands within predicted_band (`-0.0027 ± 0.006`): RATIFY canonical equation #26 IN-DOMAIN context membership for `nscs06_v8_chroma_lut`. If empirical ΔS drifts >2x: route to UNWIND-TEST per per-substrate symposium memo REVISION #1 + cargo-cult #9 sub-claim 9c at the seg + pose axes specifically per OVERNIGHT-T Section 3.2; do NOT KILL the substrate (Catalog #307 paradigm-vs-implementation classification + Catalog #308 alternative-probe-methodology enumeration apply).

## Cross-references

- **OVERNIGHT-V Phase 2 BUILD landing memo (commit `6173fae12`)**: `.omx/research/nscs06_v8_phase_2_build_full_main_atomic_recipe_flip_paired_dispatch_landed_20260521.md`
- **OVERNIGHT-T T1 PROCEED-unconditional verdict (commit `3ef1d8876`)**: `.omx/research/council_t1_nscs06_v8_phase_2_revision_1_4_working_group_20260521.md`
- **OVERNIGHT-A Phase 2 T2 DESIGN memo (commit `29f92af8d`)**: `.omx/research/council_t2_nscs06_v8_phase_2_lift_notimplementederror_design_20260521.md`
- **Per-substrate symposium memo (Catalog #325)**: `.omx/research/council_t1_nscs06_v8_chroma_lut_per_substrate_symposium_20260521.md`
- **Canonical frontier pointer per Catalog #316**: `.omx/state/canonical_frontier_pointer.json`
- **Recipe canonical-fields fix commit**: `ff827a5a8` (this landing's git-transactional commit via canonical serializer)
- **Modal call_id ledger row (Catalog #245)**: `.omx/state/modal_call_id_ledger.jsonl` (entry for `fc-01KS5M8CS47BVSJ10B6HY7TZKM`)
- **Active lane dispatch claim**: `.omx/state/active_lane_dispatch_claims.md` (lane `lane_overnight_dd_nscs06_v8_phase_4_paired_modal_t4_dispatch_pr110_reference_20260521`)
- **Sister grayscale_lut canonical recipe pattern reference**: `.omx/operator_authorize_recipes/substrate_grayscale_lut_modal_a100_dispatch.yaml`
- **Sister DP1 OVERNIGHT-CC active dispatches (DISJOINT)**: `lane_dp1_path_a_baseline_refire_20260521_contest_cpu` + `_contest_cuda` + `lane_dp1_path_a_procedural_refire_20260521_contest_cpu` + `_contest_cuda`
- **Canonical equation #26**: `src/tac/canonical_equations/procedural_codebook_savings.py` (IN-DOMAIN context `nscs06_v8_chroma_lut` per `_INCLUDED_CONTEXTS`)
- **CASCADE COMPRESSION symposium**: commit `d125af6c3` PRIORITY 3
- **Carmack MVP-first 5-step canonical methodology**: CLAUDE.md amendment commit `be125b878`
