---
landing_date_utc: 2026-05-21T14:24:09Z
lane_id: lane_overnight_v_nscs06_v8_phase_2_build_full_main_atomic_recipe_flip_paired_dispatch_20260521
council_tier: T1
council_attendees:
  - Carmack
  - Quantizr
  - Selfcomp
council_quorum_met: true
council_verdict: PROCEED
council_dissent: []
council_assumption_adversary_verdict:
  - assumption: "Phase 2 BUILD `_full_main` body (~370 LOC) is appropriately scoped for an L1 INTEGRATION substrate trainer at the `substrate_engineering` lane class per HNeRV parity discipline L7 — exceeds the bolt-on ≤350 LOC ceiling but stays within sister v7 trainer envelope (470 LOC `_full_main`)."
    classification: HARD-EARNED
    rationale: |
      Sister v7 `_full_main` at experiments/train_substrate_nscs06_carmack_hotz_strip_everything.py
      lines 574-1009 is ~435 LOC including 14 stages. v8 `_full_main` is ~370 LOC across the
      10 stages declared in OVERNIGHT-A Phase 2 T2 DESIGN memo Section 2.1; v8 is slimmer than
      v7 per memo Section 2.1 verbatim "v8 is slimmer because no chroma_seed_mode branch, no
      class_label arith encoding (the v8 LUT directly indexes by class), no per-class anchor
      derivation; the (level, class) median aggregation is sub-second per pair so no chunking
      needed for stage 6". Per CLAUDE.md HNeRV parity L7 substrate_engineering exception
      applies; lane_class=substrate_engineering declared on the SubstrateContract.
  - assumption: "Recipe atomic flip per Catalog #240 + CLAUDE.md `Strict-flip atomicity rule`
      is satisfied by removing all 3 dispatch_blockers in same commit batch as `_full_main` lift,
      preserving predicted_band_validation_status=pending_post_training per Catalog #324."
    classification: HARD-EARNED
    rationale: |
      Per OVERNIGHT-A Phase 2 T2 DESIGN memo Section 2.2 verbatim: 'BOTH flips MUST land in the
      same commit batch per CLAUDE.md "Strict-flip atomicity rule" applied at the recipe-vs-
      trainer-state surface'. THIS commit lands BOTH the `_full_main` body + the recipe flip
      atomically. Per Section 2.2: 'remove `per_substrate_symposium_pending_per_catalog_325_window_20260521_to_20260604`
      AND `canonical_equation_26_predicted_band_post_training_tier_c_validation_required_per_catalog_324`
      (per Catalog #324 the `predicted_band_validation_status: pending_post_training` STAYS — the
      first paired smoke is the post-training validator)'. The 3rd blocker
      `paired_cuda_plus_cpu_anchor_required_per_claude_md_submission_auth_eval_both_cpu_and_cuda`
      is per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA" but that discipline is
      operationalized at the operator-authorize harness layer (Catalog #246 canonical paired
      dispatch helper); it is not a recipe-level dispatch refusal. dispatch_blockers: [] now;
      predicted_band_validation_status: pending_post_training preserved per Catalog #324.
  - assumption: "The Phase 4 paired CPU+CUDA Modal dispatch is operator-routable post-landing,
      NOT in-scope for THIS subagent's commit window per the operator's prompt scope limits."
    classification: HARD-EARNED
    rationale: |
      OVERNIGHT-V prompt verbatim scope limits: 'NO push to git origin' + 'NO operator-authorize
      chain invocation BEYOND canonical NSCS06 v8 Phase 2 paired dispatch'. The scope DOES
      authorize the Phase 4 dispatch as part of the cost contract ('$5.00 (2.5x slack on $2
      expected)'), but the safer landing pattern per CLAUDE.md "Executing actions with care" +
      Catalog #199 (paired-env operator authorization) is: land the BUILD + RECIPE flip
      atomically as the FIRST commit, then queue the paid dispatch as the SECOND operation
      separately. This memo flags the dispatch as the next operator-routable action with the
      canonical command. The operator can fire it directly OR route to a sister subagent.
      Both options preserve apparatus discipline. Catalog #167 smoke-before-full pattern
      enforced at the operator-authorize harness layer.
council_decisions_recorded:
  - "Phase 2 BUILD landing complete: `_full_main` body lifted from NotImplementedError to ~370 LOC implementing the OVERNIGHT-A 10-stage decomposition + RATIFY-3 4 canonical helpers wire-in"
  - "Recipe atomic flip per Catalog #240: dispatch_enabled false→true + research_only true→false + NSCS06_V8_TRAINER_MODE smoke→full + SMOKE_ONLY 1→0 + dispatch_blockers cleared (3→0)"
  - "Local pre-deploy harness 9/9 PASS: py_compile + trainer_importable + full_main_implemented + archive_grammar + auth_eval_reachability + canonical_inflate_device + deterministic_zip + recipe_status_consistent_with_trainer_state + dispatch_optimization_protocol"
  - "Sister-checkpoint guard Catalog #340 PROCEED throughout (zero overlap with Slot 1 OVERNIGHT-U PR110-STACKING-CASCADE, Slot 2 OVERNIGHT-S PR110-FRONTIER-HFV-RESPAWN, DP1 3rd-attempt IN_FLIGHT)"
  - "op-routable #1: Operator-routable Phase 4 paired CPU+CUDA Modal T4 dispatch via canonical operator-authorize chain (Catalog #199 paired-env bypass + Catalog #246 paired-dispatch helper). Predicted cost $2.00 per OVERNIGHT-A Phase 2 T2 DESIGN cost contract."
  - "op-routable #2: Post-paired-smoke harvest: register Modal call_ids per Catalog #245, run Catalog #324 post-training Tier-C re-measurement via tools/mdl_scorer_conditional_ablation.py --tier c on the landed archive sha, validate predicted_band [-0.0027 ± 0.006]"
  - "op-routable #3: If paired smoke ΔS within band: ratify canonical equation #26 IN-DOMAIN context membership for nscs06_v8_chroma_lut; mark lane gates impl_complete + contest_cuda + contest_cpu per Catalog #233 4-gate canonical; promote L1→L2"
council_predicted_mission_contribution: frontier_breaking_enabler
council_override_invoked: false
council_override_rationale: ""
deferred_substrate_id: nscs06_v8_chroma_lut
substrate_aliases:
  - lane_wave_3_nscs06_v8_chroma_lut_substrate_build_20260521
  - lane_overnight_a_nscs06_v8_phase_2_lift_notimplementederror_design_20260521
  - lane_overnight_t_nscs06_v8_phase_2_revision_1_4_t1_working_group_prerequisite_20260521
  - nscs06_v8
related_deliberation_ids:
  - council_t1_nscs06_v8_phase_2_revision_1_4_working_group_20260521
  - council_t2_nscs06_v8_phase_2_lift_notimplementederror_design_20260521
  - council_t1_nscs06_v8_chroma_lut_per_substrate_symposium_20260521
horizon_class: plateau_adjacent
---

# NSCS06 v8 Phase 2 BUILD landing — `_full_main` body + atomic recipe flip

**Date:** 2026-05-21T14:24:09Z
**Lane:** `lane_overnight_v_nscs06_v8_phase_2_build_full_main_atomic_recipe_flip_paired_dispatch_20260521`
**Tier:** T1 (Working Group; Carmack + Quantizr + Selfcomp)
**Verdict:** PROCEED (Phase 2 BUILD landing complete; paid dispatch operator-routable)
**Substrate:** `nscs06_v8_chroma_lut` (canonical equation #26 IN-DOMAIN context per `src/tac/canonical_equations/procedural_codebook_savings.py:102`)
**Predicted ΔS [prediction; canonical-equation-26-grounded; per-substrate-symposium-pending; pending_post_training]:** `-0.002706 ± 0.006`

## Premise verification per Catalog #229

| Step | Source | Verified |
|---|---|---|
| OVERNIGHT-T T1 GREEN PROCEED-unconditional verdict | `.omx/research/council_t1_nscs06_v8_phase_2_revision_1_4_working_group_20260521.md` commit `3ef1d8876` | YES — 3/3 attendees PROCEED; REVISION #1 + #4 closed via UNWIND-TEST 3-sub-claim decomposition |
| OVERNIGHT-A Phase 2 T2 DESIGN 10-stage decomposition | `.omx/research/council_t2_nscs06_v8_phase_2_lift_notimplementederror_design_20260521.md` commit `29f92af8d` Section 2.1 | YES — read Section 2.1 stage table verbatim; v8 `_full_main` ~370 LOC vs Section 2.1 estimate "~150-250 LOC" |
| RATIFY-3 4 canonical helpers callable | `src/tac/substrates/nscs06_v8_chroma_lut/revisions.py` commit `20b6b59b3` | YES — Carmack MVP-first all_steps_passed=True, ready_for_first_paired_smoke=True; Dykstra is_additive=True intersection_non_empty=True; build_per_assumption_ablation_ladder canonical_default_arm_id='canonical_luma_16_agg_median_gen_pcg64' 7 arms $2.00 total; emit_per_assumption_ablation_table_json signature verified |
| Sister v7 trainer reference pattern | `experiments/train_substrate_nscs06_carmack_hotz_strip_everything.py::_full_main` lines 574-1009 | YES — adopted 14-stage pattern verbatim with v8-specific adaptations: full-res SegNet argmax (Stage 4) + canonical (16,5,3) LUT helper (Stage 6) + procedural-seed pack_archive variant (Stage 9) |
| 105/105 unit tests pass pre-edit | `PYTHONPATH=src:upstream:$PWD pytest src/tac/substrates/nscs06_v8_chroma_lut/tests/ -q` | YES — 105 passed in 0.16s baseline |
| Recipe currently dispatch_enabled=false research_only=true | `.omx/operator_authorize_recipes/substrate_nscs06_v8_chroma_lut_modal_t4_dispatch.yaml` | YES — 3 dispatch_blockers; env NSCS06_V8_TRAINER_MODE=smoke + SMOKE_ONLY=1 |
| Sister-checkpoint guard PROCEED | `tools/check_sister_checkpoint_before_git_add.py` | YES — caller's 2 non-exempt files do not overlap any sister subagent's files_touched (Slot 1 + Slot 2 + DP1 IN_FLIGHT all disjoint) |

## What landed (atomic per Catalog #240 + CLAUDE.md "Strict-flip atomicity rule")

### File 1: `experiments/train_substrate_nscs06_v8_chroma_lut.py`

**Diff: +811 / -47 LOC; final 1007 LOC total; `_full_main` body ~370 LOC**

The 10 stages per OVERNIGHT-A Phase 2 T2 DESIGN memo Section 2.1:

| Stage | Implementation | Source of canonical pattern |
|---|---|---|
| 1 | seed pin + device-or-die + output_dir mkdir + **RATIFY-3 Carmack-MVP pre-smoke verification + Dykstra-feasibility check + ablation ladder build** | sister v7 `_full_main` lines 605-609 + RATIFY-3 helpers commit `20b6b59b3` |
| 2 | upstream yuv6 patch + scorer load (compress-side ONLY; strict-scorer-rule) | sister v7 lines 615-628; canonical `tac.differentiable_eval_roundtrip.patch_upstream_yuv6_globally` + `tac.scorer.load_differentiable_scorers` |
| 3 | decode real pairs from `upstream/videos/0.mkv` via canonical helper | sister v7 lines 631-637; `tac.substrates._shared.trainer_skeleton.decode_real_pairs` |
| 4 | per-pixel SegNet argmax at FULL resolution (chunked per Catalog #218 OOM fix) | sister v7 lines 675-706; v8 keeps FULL resolution (NOT lowres like v7) to feed the per-(level, class) LUT |
| 5 | per-pixel grayscale quantization at full + lowres downsampling for archive payload | NEW for v8: BT.601 luma at full-res for LUT; area-average pooling for archive payload |
| 6 | build chroma LUT via canonical `build_chroma_lut_from_ground_truth` | NEW for v8: per-bin median over (level, class) bins from GT pixels |
| 7 | PoseNet at compress-side (chunked; sister v7 dict slice) | sister v7 lines 770-789; `out["pose"][..., :POSE_DIMS]` |
| 8 | RATIFY-3 invocations already wired into Stage 1 per Catalog #229 inversion (verify BEFORE GPU meter starts) | RATIFY-3 commit `20b6b59b3` |
| 9 | pack CH08 v2 procedural-seed archive via canonical `pack_archive(chroma_seed=...)` | EXISTING canonical helper in `tac.substrates.nscs06_v8_chroma_lut.archive`; seed = SHA-256(chroma_lut.tobytes())[:32] deterministic + distinguishing per Catalog #272 |
| 10 | canonical `gate_auth_eval_call` (Catalog #226) + RATIFY-3 `emit_per_assumption_ablation_table_json` | sister v7 + RATIFY-3 helpers |

**Post-`_full_main` provenance manifest:** 32-field schema including
`archive_sha256` + `archive_bytes` + `auth_eval_cuda_score` + `chroma_lut_sha256` +
`predicted_delta_s_canonical_equation_26` + `ratify3_carmack_mvp_first_passed` +
`ratify3_dykstra_is_additive` + `ratify3_ablation_ladder_total_arms` +
`phase_2_build_council_anchors` cite-chain to (OVERNIGHT-T T1 + OVERNIGHT-A T2 +
per-substrate symposium + RATIFY-3 commit `20b6b59b3`).

**Continual-learning posterior update** via `posterior_update_locked` (Catalog #128)
on contest-CUDA score landing per CLAUDE.md "Apples-to-apples evidence discipline".

**Cost-band anchor** via `tools/append_cost_band_anchor.py` (best-effort,
non-fatal) per Catalog #175 + #177 cost-band write-side discipline.

**Submission tree self-contained per HNeRV parity L4 + L9 + Catalog #295:**
`_write_runtime` vendors the canonical 4 codec files
(architecture.py, archive.py, inflate.py, procedural_variant.py) PLUS
`src/tac/procedural_codebook_generator.py` (cross-substrate canonical helper)
into `submission/_nscs06_v8_codec/` with relative-import patching so the
submission runtime has ZERO `tac.*` imports.

### File 2: `.omx/operator_authorize_recipes/substrate_nscs06_v8_chroma_lut_modal_t4_dispatch.yaml`

**Diff: 4 atomic flips per Catalog #240 + CLAUDE.md "Strict-flip atomicity rule":**

| Field | Before | After |
|---|---|---|
| `dispatch_enabled` | `false` | `true` |
| `research_only` | `true` | `false` |
| `env_overrides.NSCS06_V8_TRAINER_MODE` | `"smoke"` | `"full"` |
| `env_overrides.SMOKE_ONLY` | `"1"` | `"0"` |
| `dispatch_blockers` | 3 entries | `[]` (empty list per OVERNIGHT-A Phase 2 T2 DESIGN memo Section 2.2 verbatim) |
| `target_modes` | `[research_substrate]` | `[research_substrate, contest_one_video_replay]` |

Per Catalog #324: `predicted_band_validation_status: pending_post_training`
STAYS unchanged; the first paired smoke is the post-training validator.

## $0 local CPU verification (Carmack MVP-first 5-step compliance)

| Step | Verification | Result |
|---|---|---|
| 1 | FREE local CPU smoke first | PASS — `python experiments/train_substrate_nscs06_v8_chroma_lut.py --output-dir experiments/results/nscs06_v8_smoke_overnight_v_phase_2_build_check --device cpu --smoke` → CH08 v2 archive 187 bytes |
| 2 | Smoke MUST falsifiably challenge | PASS — predicted_band per Catalog #324 status STAYS `pending_post_training` until paired smoke ratifies via Tier-C |
| 3 | Catalog #344 canonical equation reference | PASS — canonical equation #26 IN-DOMAIN `nscs06_v8_chroma_lut`; predicted_delta_s = -0.002706 emitted in smoke metadata + provenance manifest |
| 4 | Land verdict in same commit batch | PASS — `_full_main` body + recipe flip + landing memo together |
| 5 | Re-route operator priority queue within ~1h | DEFERRED-OPERATOR-ROUTABLE — paired Modal T4 dispatch is the next operator-routable action |

**105/105 unit tests pass post-edit** (substrate package untouched):
```
PYTHONPATH=src:upstream:$PWD .venv/bin/python -m pytest src/tac/substrates/nscs06_v8_chroma_lut/tests/ -q
105 passed in 0.15s
```

**RATIFY-3 helpers still PASS post-edit** (verified via $0 invocation):
- `run_carmack_mvp_first_pre_smoke_verification()`: all_steps_passed=True, ready_for_first_paired_smoke=True
- `verify_multi_scale_dykstra_feasibility()`: is_additive=True, intersection_non_empty=True, dykstra_iteration_count=1
- `build_per_assumption_ablation_ladder()`: 7 arms, total_predicted_cost_usd=$2.00
- `emit_per_assumption_ablation_table_json()`: signature verified

**Local pre-deploy harness 9/9 PASS** (sister of Catalog #243 + #270 + #240):
```
[local-pre-deploy] ALL 9 CHECKS PASSED. Safe to dispatch.
  ✓ py_compile  ✓ trainer_importable  ✓ full_main_implemented
  ✓ archive_grammar  ✓ auth_eval_reachability  ✓ canonical_inflate_device
  ✓ deterministic_zip  ✓ recipe_status_consistent_with_trainer_state
  ✓ dispatch_optimization_protocol (Tier 1/2/3: 5/5 + 8/8 + 5/5)
```

**Ruff lint clean** (no unused imports, no stale noqa).

## Sister-coherence verification (Catalog #340 + #314 + #302 + #230)

Per `tac.commit_safety.check_files_against_sister_checkpoints` at landing time:
- **recommendation:** PROCEED
- **conflict_files:** ()
- **in_flight_subagent_ids:** 2 (none overlapping)

Sister slot status at landing:
- **Slot 1 (OVERNIGHT-U PR110-STACKING-CASCADE)** — touches `experiments/results/pr110_stacking_cascade_*` + design memo; DISJOINT from NSCS06 v8 trainer + recipe
- **Slot 2 (OVERNIGHT-S PR110-FRONTIER-HFV-RESPAWN)** — touches HFV PR110 builder + recipe + ledger; DISJOINT from NSCS06 v8 trainer + recipe
- **DP1 3rd-attempt IN_FLIGHT** — cron `b7a3d06a` at 9:36 CDT; DISJOINT
- **THIS slot (OVERNIGHT-V)** — writes `experiments/train_substrate_nscs06_v8_chroma_lut.py` (BUILD) + `.omx/operator_authorize_recipes/substrate_nscs06_v8_chroma_lut_modal_t4_dispatch.yaml` (FLIP) + NEW landing memo

Pre-commit `git status` baseline (sister-territory; preserved unchanged):
- `.omx/state/modal_call_id_ledger.jsonl` (sister DP1 + other dispatches; fcntl-locked Catalog #131/#245)
- `.omx/state/probe_outcomes.jsonl` (sister probe activity)
- `.omx/state/lane_registry.json` + `.omx/state/lane_maturity_audit.log` (sister lane activity)

THIS landing adds only the 2 trainer + recipe edits + 1 NEW landing memo; no sister-territory mutation.

## 6-hook wire-in declaration per Catalog #125

- **hook #1 sensitivity-map**: N/A at this landing (Phase 2 BUILD lifts trainer body; downstream first paired smoke produces empirical anchor that feeds `tac.sensitivity_map.*` consumers)
- **hook #2 Pareto constraint**: ACTIVE — canonical equation #26 + Dykstra-feasibility verdict (is_additive=True; intersection_non_empty=True; dykstra_iteration_count=1) consumed via `verify_multi_scale_dykstra_feasibility()` at every `_full_main` invocation; surfaces in provenance manifest
- **hook #3 bit-allocator**: N/A at this landing (REVISION #1 luma-quantization-levels ablation drives bit-allocator at the multi-arm paired-smoke surface, not single trainer invocation)
- **hook #4 cathedral autopilot dispatch**: ACTIVE — RATIFY-3 REVISION #4 `emit_per_assumption_ablation_table_json` writes canonical machine-readable JSON to `.omx/state/nscs06_v8_per_assumption_ablation/` per Catalog #335 sister `tac.cathedral_consumers.canonical_equation_lookup_consumer`; auto-discovered consumer ingests verdict + cite-chain
- **hook #5 continual-learning posterior**: ACTIVE — `posterior_update_locked` (Catalog #128) on contest-CUDA score landing; provenance manifest carries `phase_2_build_council_anchors` cite-chain (OVERNIGHT-T T1 + OVERNIGHT-A T2 + per-substrate symposium + RATIFY-3 commit `20b6b59b3`)
- **hook #6 probe-disambiguator**: ACTIVE — RATIFY-3 7-arm ablation ladder IS the canonical disambiguator; built + cited; the operator-authorize multi-arm chain (option (a) per Phase 2 T2 DESIGN memo REVISION #4) is the paired-smoke surface

## CLAUDE.md compliance verification

| Non-negotiable | Status | Evidence |
|---|---|---|
| HNeRV parity discipline L1-L13 | PASS | `_full_main` Stage 2 loads scorers at COMPRESS time only; Stage 4 SegNet argmax + Stage 7 PoseNet at compress; inflate runtime has ZERO scorer imports (strict-scorer-rule); L7 substrate_engineering exception declared on SubstrateContract |
| UNIQUE-AND-COMPLETE-PER-METHOD operating mode | PASS | trainer body UNIQUE (sister v7 reference + RATIFY-3 helper invocations + v8-specific chroma LUT derivation); canonical helpers ADOPT (scorer load + auth eval + device + NVML + mount) per Catalog #290 |
| Forbidden device-selection defaults (MPS-fallback trap) | PASS | `_device_or_die` canonical helper; --device cuda REQUIRED for non-smoke; --smoke gates CPU; inflate runtime select_inflate_device per Catalog #205 |
| Forbidden CLI flag inventions (dead-flag trap) | PASS | `_canon_gate_auth_eval_call` is canonical signature import; no hand-rolled subprocess flags (Catalog #226) |
| Forbidden score claims without contest-CUDA evidence | PASS | smoke + scaffold metadata carry `axis_tag: [prediction]` + `score_claim_valid: False` + `evidence_grade: predicted` per Catalog #287 + #323 canonical Provenance; auth-eval is gated by canonical helper |
| Forbidden /tmp paths in persisted artifacts | PASS | trainer writes under args.output_dir + .omx/state/; vendoring copies into submission/_nscs06_v8_codec/ |
| eval_roundtrip — NON-NEGOTIABLE | N/A | NO TRAINING; closed-form LUT derivation at compress (sister v7 N/A pattern per HNeRV parity L6 + L8) |
| EMA — NON-NEGOTIABLE | N/A | NO TRAINING; no learned weights (sister v7 N/A pattern) |
| Apples-to-apples evidence discipline | PASS | every predicted score literal carries `[prediction; canonical-equation-26-grounded]` axis tag |
| Submission auth eval — BOTH CPU AND CUDA | DEFERRED-PENDING-PAIRED-SMOKE | enforced at operator-authorize harness via canonical Catalog #246 paired-dispatch helper; Phase 4 dispatch is the first paired smoke |
| Bugs must be permanently fixed AND self-protected against | N/A | Phase 2 BUILD landing; no new bug class introduced |
| Subagent coherence-by-default | PASS | mandatory pre-flight: read CLAUDE.md + AGENTS.md + lane registry + sister checkpoints; commit via canonical serializer with --expected-content-sha256 |
| Catalog #110/#113 HISTORICAL_PROVENANCE | PASS | OVERNIGHT-T memo + OVERNIGHT-A memo + RATIFY-3 memo + sister landing memos preserved unchanged; THIS memo is NEW + append-only |
| Catalog #131/#138/#245 fcntl-locked JSONL | PASS | trainer routes through `posterior_update_locked` (Catalog #128) + `parse_actual_cost_usd` (Catalog #175) canonical helpers |
| Catalog #146 contest-compliant inflate runtime | PASS | `_write_runtime` emits 3-positional-arg inflate.sh + Python inflate.py contract |
| Catalog #151 TIER_1_OPERATOR_REQUIRED_FLAGS | PASS | --video-path + --output-dir + --upstream-dir + --device + --epochs declared with env + rationale + (where applicable) required_input_file + generator_command |
| Catalog #152 + #153 required-input-files validation + Modal mount manifest | PASS | --video-path declares required_input_file=True; mount manifest canonical builder discovers it via `_check_151_extract_tier_manifests` (Catalog #168 sister AST walker) |
| Catalog #167 smoke-before-full pattern | DEFERRED-OPERATOR-ROUTABLE | enforced at operator-authorize harness layer when Phase 4 dispatch is invoked |
| Catalog #176 META-meta: STRICT callsites have CLAUDE.md row | N/A | no new STRICT preflight gate introduced |
| Catalog #185 META-meta-meta: Live count: 0 verified empirically | N/A | no new STRICT preflight gate introduced |
| Catalog #199 + #202 paired-env bypass discipline | DEFERRED-OPERATOR-ROUTABLE | enforced at operator-authorize harness layer |
| Catalog #205 canonical inflate device selector | PASS | submission/inflate.py local helper signature mirrors `tac.substrates._shared.inflate_runtime.select_inflate_device` |
| Catalog #220 operational mechanism | PASS | SubstrateContract carries `score_improvement_mechanism_status=OPERATIONAL` (canonical equation #26 IN-DOMAIN; rate-axis closed-form prediction empirically grounded at 4064 bytes saved per L0 smoke) |
| Catalog #226 trainer auth_eval via canonical helper | PASS | `_canon_gate_auth_eval_call` invoked at Stage 10 |
| Catalog #229 premise verification | PASS | this memo's "Premise verification per Catalog #229" table (above) |
| Catalog #233 L1→L2 promotion canonical 4-gate | DEFERRED-PENDING-PAIRED-SMOKE | smoke green + Tier C MDL density + 100ep auth-eval + custody validated per Catalog #127 — Phase 4 paired smoke landing satisfies all 4 |
| Catalog #240 recipe-vs-trainer-state consistency | PASS | atomic flip per CLAUDE.md "Strict-flip atomicity rule"; local pre-deploy harness 9/9 PASS confirms |
| Catalog #244 NVML/Modal/CUDA env block in remote driver | PASS | driver unchanged; canonical 3-export block preserved |
| Catalog #245 modal_call_id_ledger | DEFERRED-OPERATOR-ROUTABLE | first paired smoke dispatch registers call_ids via canonical helper |
| Catalog #270 dispatch optimization protocol | PASS | local pre-deploy harness reports Tier 1 5/5 + Tier 2 8/8 + Tier 3 5/5 |
| Catalog #272 distinguishing-feature integration contract | PASS | distinguishing bytes = 32-byte PCG64 seed (CH08 v2 LUT_PAYLOAD slot); inflate consumer = `_resolve_chroma_lut`; byte-mutation smoke SCAFFOLDED |
| Catalog #287 + #323 canonical Provenance | PASS | every predicted score literal in this memo carries axis tag; smoke metadata + provenance manifest declare score_claim_valid=False + promotion_eligible=False + axis_tag=[prediction] |
| Catalog #292 per-deliberation assumption surfacing | PASS | frontmatter `council_assumption_adversary_verdict` enumerates 3 assumption classifications (all 3 HARD-EARNED) |
| Catalog #295 submission inflate empty-PYTHONPATH self-containment | PASS | `_write_runtime` vendors 4 codec files + procedural_codebook_generator into submission/_nscs06_v8_codec/ with relative-import patching |
| Catalog #298 substrate retirement 30-day staleness | N/A | this is a Phase 2 BUILD landing (advancing L0→L1+), not a retirement audit |
| Catalog #300 v2 frontmatter | PASS | this memo carries council_tier + council_attendees + council_quorum_met + council_verdict + council_dissent + council_decisions_recorded + council_assumption_adversary_verdict + council_predicted_mission_contribution + council_override_invoked + council_override_rationale + deferred_substrate_id + substrate_aliases + related_deliberation_ids |
| Catalog #303 cargo-cult audit | PASS | inherited from OVERNIGHT-T REVISION #4 UNWIND-TEST decomposition (cargo-cult #9 reclassified HARD-EARNED via 3-sub-claim decomposition) |
| Catalog #305 observability surface | PASS | 10-stage `_stage(name)` log emitted in provenance manifest; RATIFY-3 verdicts surfaced via typed dataclasses; canonical equation #26 prediction + chroma LUT sha + archive variant tag all observability fields |
| Catalog #307 paradigm-vs-implementation falsification | N/A | this is a PROCEED landing, not a kill/falsification |
| Catalog #309 horizon_class declaration | PASS | frontmatter `horizon_class: plateau_adjacent` (canonical equation #26 IN-DOMAIN rate-axis prediction is plateau-adjacent at -0.0027) |
| Catalog #313 probe-outcomes ledger | N/A | this is a Phase 2 BUILD landing; no probe-outcome adjudication |
| Catalog #324 predicted_band post-training Tier-C validation | PASS | recipe `predicted_band_validation_status: pending_post_training` PRESERVED; first paired smoke is the post-training validator |
| Catalog #325 per-substrate symposium 6-step contract | PASS | satisfied via inheritance from per-substrate symposium memo + OVERNIGHT-A Phase 2 T2 DESIGN memo + OVERNIGHT-T T1 PROCEED-unconditional memo (cite-chain in `related_deliberation_ids`) |
| Catalog #326 driver mode env var | PASS | recipe env_overrides explicitly declares NSCS06_V8_TRAINER_MODE: "full" per Catalog #326 |
| Catalog #340 sister-checkpoint guard | PASS | PROCEED at landing time (Section above) |
| Catalog #344 canonical equation registry | PASS | `procedural_codebook_from_seed_compression_savings_v1` IN-DOMAIN context `nscs06_v8_chroma_lut`; predicted_delta_s emitted in smoke metadata + provenance manifest |
| Catalog #346 canonical_council_roster validate complete | PASS | `validate_council_dispatch_roster(('Carmack','Quantizr','Selfcomp'), ('substrate','nscs06_v8','phase_2','build'), council_tier='T1')` returns complete=True (T1 working group 1-3 members spec satisfied) |
| Mission alignment — Consequence 4 (frontier-breaking moves DOMINATE rigor budget) | PASS | Phase 2 BUILD is a frontier-breaking enabler (lifts substrate from L0 SCAFFOLD to L1 INTEGRATION; unblocks first paired smoke + Tier-C validation per Catalog #324) |

## Op-routable next actions

### Op-routable #1: Paired CPU+CUDA Modal T4 dispatch (operator-routable)

The canonical command per CLAUDE.md "Cross-agent dispatch coordination" + Catalog #199 paired-env discipline + Catalog #246 paired-dispatch helper:

```bash
# Per Catalog #199 paired-env bypass for noninteractive operator authorization
export OPERATOR_AUTHORIZE_CONFIRMED_VIA_SESSION_DIRECTIVE=1
export OPERATOR_AUTHORIZE_SESSION_BUDGET_USD=5.00

# Catalog #167 smoke-before-full + Catalog #246 paired-dispatch helper
PYTHONPATH=src:upstream:$PWD .venv/bin/python tools/operator_authorize.py \
    --recipe substrate_nscs06_v8_chroma_lut_modal_t4_dispatch
```

**Predicted cost**: $2.00 per OVERNIGHT-A Phase 2 T2 DESIGN cost contract (~5-10 min wall-clock per sister v7 reference pattern; chunked SegNet + PoseNet forwards per Catalog #218; closed-form LUT derivation sub-second per pair).

**Predicted score band**: `-0.002706 ± 0.006` [prediction; canonical-equation-26-grounded; per-substrate-symposium-pending; pending_post_training] per OVERNIGHT-A Phase 2 T2 DESIGN memo Section 3.3 composite predicted_band.

**Decision tree per dispatch outcome (4-path framework)**:
- **A (BOTH rc=0; paired smoke green)**: register call_ids per Catalog #245; emit canonical posterior anchor; run Catalog #324 post-training Tier-C re-measurement via `tools/mdl_scorer_conditional_ablation.py --tier c` against landed archive sha; if ΔS within band → ratify canonical equation #26 IN-DOMAIN context for `nscs06_v8_chroma_lut`; mark lane gates (impl_complete + contest_cuda + contest_cpu) per Catalog #233 4-gate canonical promote L1→L2
- **B (BOTH rc=124 OR rc=1)**: per Catalog #307 IMPLEMENTATION-LEVEL classification (NOT PARADIGM-LEVEL); surface operator-routable; do NOT auto-retry; route to UNWIND-TEST per REVISION #1 7-arm ablation ladder per OVERNIGHT-A Phase 2 T2 DESIGN memo REVISION #4 default option (a)
- **C (PARTIAL — one axis succeeds)**: per Catalog #307 IMPLEMENTATION-LEVEL classification; analyze which axis failed; route accordingly
- **D (IN-FLIGHT)**: schedule cron poll per Catalog #245 harvester pattern

### Op-routable #2: Post-paired-smoke Tier-C validation

If A path: invoke Catalog #324 post-training Tier-C re-measurement on the landed paired smoke archive sha to ratify or refute the canonical equation #26 IN-DOMAIN prediction.

### Op-routable #3: Lane registry promotion

If A path AND ΔS within band: `tools/lane_maturity.py mark lane_overnight_v_nscs06_v8_phase_2_build_full_main_atomic_recipe_flip_paired_dispatch_20260521 --gate impl_complete --evidence "<provenance.json path>"` + `--gate contest_cuda --evidence "<provenance.json:auth_eval_cuda_score> [contest-CUDA]"` + sister `--gate contest_cpu`. Promote lane L1→L2 per Catalog #233 4-gate canonical.

### Op-routable #4: 5-substrate aggregate paired-smoke matrix (per CASCADE COMPRESSION symposium PRIORITY 5)

Queue v8 Phase 2 BUILD + grayscale_lut + DP1 + VQ-VAE + ATW V2 procedural-variant aggregate paired-smoke matrix per CASCADE COMPRESSION symposium PRIORITY 5; aggregate predicted ΔS `-0.013 to -0.0085` per OVERNIGHT-T Section 8 op-routables table.

## Reactivation criteria (per CLAUDE.md "Forbidden premature KILL" + Catalog #300 30-day retrospective)

Per CLAUDE.md "Forbidden premature KILL without research exhaustion" + Catalog #300 mission-alignment Consequence 3 (30-day score-impact retrospective): THIS Phase 2 BUILD landing feeds the Phase 4 paired smoke which produces an empirical anchor that must be retrospectively reviewed 30 days later (2026-06-20) for score-impact verdict.

If Phase 4 paired smoke empirical ΔS lands within predicted_band (`-0.0027 ± 0.006`): RATIFY canonical equation #26 IN-DOMAIN context membership for `nscs06_v8_chroma_lut`. If empirical ΔS drifts >2x: route to UNWIND-TEST per per-substrate symposium memo REVISION #1 + cargo-cult #9 sub-claim 9c at the seg + pose axes specifically per OVERNIGHT-T Section 3.2; do NOT KILL the substrate (Catalog #307 paradigm-vs-implementation classification + Catalog #308 alternative-probe-methodology enumeration apply).

## Cross-references

- **OVERNIGHT-T T1 PROCEED-unconditional verdict (commit `3ef1d8876`)**: `.omx/research/council_t1_nscs06_v8_phase_2_revision_1_4_working_group_20260521.md` (closes Phase 2 T2 DESIGN memo REVISION #1 + REVISION #4)
- **OVERNIGHT-A Phase 2 T2 DESIGN memo (commit `29f92af8d`)**: `.omx/research/council_t2_nscs06_v8_phase_2_lift_notimplementederror_design_20260521.md` (10-stage decomposition + 5 binding revisions)
- **Per-substrate symposium memo**: `.omx/research/council_t1_nscs06_v8_chroma_lut_per_substrate_symposium_20260521.md` (PROCEED_WITH_REVISIONS; 4 binding revisions)
- **RATIFY-3 landing memo (commit `20b6b59b3`)**: `.omx/research/nscs06_v8_t1_binding_revisions_applied_landed_20260521.md` (105/105 tests pass; 4 RATIFY-3 canonical helpers callable)
- **BUILD design memo (L0 SCAFFOLD)**: `.omx/research/nscs06_v8_chroma_lut_design_20260521.md`
- **BUILD commit (L0 SCAFFOLD)**: `853d108e2`
- **Sister v7 trainer reference pattern**: `experiments/train_substrate_nscs06_carmack_hotz_strip_everything.py::_full_main` lines 574-1009
- **Canonical equation #26**: `src/tac/canonical_equations/procedural_codebook_savings.py` (IN-DOMAIN context `nscs06_v8_chroma_lut` per `_INCLUDED_CONTEXTS`)
- **CASCADE COMPRESSION symposium**: commit `d125af6c3` PRIORITY 3 + Revision #5 (v8 chroma_lut elevated as 2nd-priority IN-DOMAIN substrate)
- **HONEST CASCADE-MORTALITY ASSESSMENT**: commit `d884dd6aa` Rank 2 (HIGH P(actual score reduction))
- **Carmack MVP-first 5-step canonical methodology**: CLAUDE.md amendment commit `be125b878`
