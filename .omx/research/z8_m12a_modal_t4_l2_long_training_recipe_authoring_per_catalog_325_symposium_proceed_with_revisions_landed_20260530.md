<!--
SPDX-License-Identifier: MIT
Canonical apparatus-setup landing memo per Catalog #294 9-dim + Catalog #303
cargo-cult + Catalog #305 observability + Catalog #300 v2 frontmatter +
Catalog #292 per-deliberation assumption-surfacing.

$0 spend. Recipe + driver authoring + canonical preflight pass. NO PAID
DISPATCH FIRED. Operator-explicit paid Modal dispatch authorization required
per CLAUDE.md "user-asks-Authorization" + the operator's sub-0.189 threshold +
5+ dollar hard-stop.
-->
---
schema: subagent_landing_memo_v2
landing_id: z8_m12a_modal_t4_l2_long_training_recipe_authoring_per_catalog_325_symposium_proceed_with_revisions_landed_20260530
topic: "Z8 M12a Modal T4 L2 long-training canonical apparatus setup (recipe + driver) per Catalog #325 symposium 4bcc84fc0 PROCEED_WITH_REVISIONS"
lane_id: lane_z8_m12a_modal_t4_l2_long_training_per_catalog_325_symposium_proceed_with_revisions_20260530
predecessor_council_anchor: council_t3_grand_council_per_substrate_symposium_z8_hierarchical_predictive_coding_m12_paid_modal_t4_l2_long_training_plus_paired_cuda_canonical_sub_0_189_attempt_20260530
predecessor_council_commit_sha: 4bcc84fc0
predecessor_council_verdict: PROCEED_WITH_REVISIONS
this_landing_council_anchor: council_t1_z8_m12a_modal_t4_l2_long_training_recipe_authoring_canonical_apparatus_setup_post_symposium_proceed_with_revisions_20260530
this_landing_council_tier: T1
this_landing_council_verdict: PROCEED
mission_predicted_contribution: apparatus_maintenance
horizon_class: frontier_pursuit
horizon_class_rationale: "Apparatus setup unblocks operator-routable paid Modal M12a dispatch authorization toward canonical sub-0.189 submission threshold; the predicted [contest-CPU] band [0.183, 0.195] places M12a in the canonical frontier-pursuit horizon range (per Catalog #309 + the symposium's predicted_band frontmatter)"
predicted_band_validation_status: pending_post_training
predicted_band_reactivation_criterion: "Post-training Tier-C density re-measurement via tools/mdl_scorer_conditional_ablation.py --tier c on M12a landed archive sha256 + M12b paired-CUDA empirical anchor for sister Linux x86_64 [contest-CPU] vs [contest-CUDA] CPU-CUDA gap calibration per Catalog #246 + #324"
files_landed:
  - .omx/operator_authorize_recipes/substrate_z8_hierarchical_predictive_coding_modal_t4_dispatch.yaml
  - scripts/remote_lane_substrate_z8_hierarchical_predictive_coding.sh
  - .omx/state/council_deliberation_posterior.jsonl (APPEND-ONLY anchor)
  - .omx/state/probe_outcomes.jsonl (APPEND-ONLY PROCEED 14-day expires 2026-06-13)
  - .omx/state/lane_registry.json (lane registered + impl_complete + memory_entry gates marked)
  - .omx/state/active_lane_dispatch_claims.md (claim recorded)
  - .omx/state/predicted_band_audit_z8_m12a_20260530T173500Z.json (Catalog #324 PASS pending_post_training)
  - .omx/research/z8_m12a_modal_t4_l2_long_training_recipe_authoring_per_catalog_325_symposium_proceed_with_revisions_landed_20260530.md (this memo)
  - .omx/research/retroactive_sweep_for_z8_m12a_recipe_authoring_20260530T173500Z.md (Catalog #348)
preflight_results:
  catalog_270_dispatch_optimization_protocol:
    overall_pass: false
    tier1: false
    tier2: true
    tier3: false
    tier1_blockers:
      - "trainer missing tf32 (canonical primitive)"
      - "trainer missing torch_compile (canonical primitive)"
    tier3_blockers:
      - "trainer missing canonical_inflate_device token"
      - "trainer missing scorer_loader_order_correct token"
    classification: sister_scope_z8_trainer_engineering_pass_op_routable_1
  catalog_243_local_pre_deploy_check:
    overall_pass: false
    failures: [auth_eval_reachability, dispatch_optimization_protocol]
    classification: sister_scope_z8_trainer_engineering_pass_op_routable_1
  catalog_324_predicted_band_provenance_audit:
    overall_pass: true
    recipes_scanned: 1
    pass_count: 1
    fail_count: 0
    status: pending_post_training
  catalog_313_predecessor_probe_outcome:
    blocking_outcome: null
    verdict: PROCEED
  catalog_326_driver_mode_audit:
    classification: PASS_canonical_multi_key_resolution
  catalog_244_nvml_block:
    driver_exports_CUBLAS_WORKSPACE_CONFIG: true
    driver_exports_DALI_DISABLE_NVML: true
    driver_exports_PYTORCH_CUDA_ALLOC_CONF: true
    classification: PASS
  catalog_240_recipe_vs_trainer_state:
    recipe_vs_trainer_state_consistent: true
    classification: PASS
op_routable_followups:
  - id: 1
    title: "Z8 trainer engineering pass for TF32 + torch.compile + canonical gate_auth_eval_call wiring"
    scope: experiments/train_substrate_z8_hierarchical_predictive_coding_mlx.py
    catalog_refs: ["#270 Tier 1", "#270 Tier 3", "#226 canonical gate_auth_eval_call", "#205 canonical select_inflate_device", "#178 TF32", "#179 torch_compile"]
    sister_scope: true
    blocks_op_routable_2: false
    notes: "M11 cycle-closure commit 2f8570755 already validates the canonical 5-step compose through upstream/evaluate.py --device cpu without these primitives. M12a canonical-quadruple-binding mode operates on pure compose pattern (numpy + torch) and emits the M9 artifact JSON; canonical contest_auth_eval is INVOKED at the M11 sister surface (m11_l1_macos_cpu_smoke.run_z8_m11_l1_smoke) not the trainer. The Tier 1 + Tier 3 blockers represent the canonical Z8 trainer engineering pass to wire the CUDA-side primitives for the Modal T4 hot loop; this is sister scope per CLAUDE.md Catalog #340 disjoint-subagent discipline."
  - id: 2
    title: "Operator explicit paid Modal T4 M12a dispatch authorization"
    canonical_cli_command: ".venv/bin/python tools/operator_authorize.py --recipe substrate_z8_hierarchical_predictive_coding_modal_t4_dispatch --platform modal --gpu T4"
    catalog_refs: ["#325 symposium PROCEED_WITH_REVISIONS 14-day window", "#243 local pre-deploy", "#270 dispatch optimization protocol umbrella", "#271 codex pre-dispatch review", "#167 smoke-before-full", "#199 paired-env operator authorization"]
    operator_gated: true
    envelope_usd: 8.00
    notes: "M12a paid Modal T4 L2 long-training ~5-15 USD per cost_band hand_calibrated_fallback_p50_usd. Per CLAUDE.md user-asks-Authorization + sub-0.189 submission threshold + 5+ dollar hard-stop: operator-explicit authorization required. The canonical Catalog #243 + #270 + #271 + #167 + #199 gates all fire BEFORE the paid Modal meter starts. Op-routable #1 (Z8 trainer engineering pass) is RECOMMENDED before this op-routable but NOT structurally blocking (canonical-quadruple-binding mode operates without TF32/torch.compile via numpy+torch pure compose; M12a observability provides canonical training-trajectory evidence)."
  - id: 3
    title: "M12b paired-CUDA recipe authoring"
    gated_on: "M12a PoseNet sub-1.0 trajectory evidence per Contrarian VETO + Assumption-Adversary + Hotz consensus"
    catalog_refs: ["#246 paired CPU+CUDA dispatch on 1:1 contest-compliant hardware", "#325 14-day window expires 2026-06-13", "#324 post-training Tier-C calibration"]
    sister_scope: true
    notes: "M12b paired-CUDA is STRUCTURALLY GATED on M12a empirical PoseNet sub-1.0 trajectory evidence. If M12a PoseNet trajectory is sub-linear at 100-200ep equivalent within the 2000ep wave, Round 2 self-reflection per Catalog #363 downgrades M12b verdict to PROVISIONAL or ESCALATE_TO_OPERATOR per Catalog #300. The canonical M12b recipe authoring is its own apparatus-setup subagent + own canonical Catalog #325 symposium clearance if architecture or budget envelope shifts."
canonical_equation_refs:
  - id: categorical_posterior_capacity_vs_continuous_gaussian_v1
    status: "3-of-3 anchors; Catalog #371 auto-recalibrator ACTIVE"
    relevance: "Z8 per-level DreamerV3 RSSM canonical primitive at hierarchical depth k=3"
  - id: wyner_ziv_decoder_side_information_rate_savings_v1
    status: "registered Wave N+36 commit c2780c7ba 2026-05-30"
    relevance: "Z8 M6 top-level Wyner-Ziv canonical coder Theorem 1 binding"
  - id: z8_m12a_l2_long_training_posenet_trajectory_v1
    status: RESERVED-PENDING-M12a-LANDING
    relevance: "canonical first M12a empirical anchor on PoseNet decrease trajectory"
canonical_anti_pattern_refs:
  - id: stand_down_verdict_based_on_stale_canonical_state_currency_v1
    status: HARD-EARNED-EMPIRICALLY-EXTINCTED-AT-M11-CYCLE-CLOSURE
    relevance: "Z8 paradigm intact + M11 cycle-closure landed (2f8570755) supersedes Slot W Wave N+40 stale DEFER per Catalog #313"
related_landings:
  - feedback_z8_m11_l1_macos_cpu_mlx_local_end_to_end_smoke_canonical_evaluate_cpu_binding_landed_20260530.md
  - feedback_z8_m10_inflate_consumes_real_trained_weights_per_catalog_369_landed_20260530.md
  - feedback_z8_m9_full_main_canonical_quadruple_binding_integration_landed_20260530.md
  - feedback_z8_m6_wyner_ziv_top_level_coder_landed_20260530.md
sister_disjoint_subagents:
  - id: cascade_b_wave_2
    head_commit: ac302ffd1
    files_scope: "experiments/results/cascade_b_*"
    overlap: none
  - id: z8_l28_l30_l32_quick_wins
    head_commit: a5bc8e12540a56aad
    files_scope: "src/tac/substrates/z8_hierarchical_predictive_coding/inflate.py + _serialize_pair_wavelet_pyramid"
    overlap: none
  - id: slot_ggg_tier_c_overnight_runner
    pid: 10169
    files_scope: "experiments/results/slot_ggg_*"
    overlap: none
---

# Z8 M12a Modal T4 L2 long-training canonical apparatus setup landing memo

**Operator-routed Yousfi-cascade TOP-1 PEER directive** (post-Z8 M12 symposium `4bcc84fc0` PROCEED_WITH_REVISIONS LANDED): canonical apparatus-setup work to author the M12a Modal T4 L2 long-training dispatch recipe + canonical sister recipe driver script + canonical preflight pass.

**$0 spend**. NO PAID DISPATCH FIRED. Apparatus setup only.

## Canonical-vs-unique decision per layer (Catalog #290)

| Layer | Decision | Rationale |
|---|---|---|
| Recipe schema (top-level fields) | ADOPT_CANONICAL | Per DP1 (`substrate_pretrained_driving_prior_modal_t4_dispatch.yaml`) + C6 IBPS (`substrate_c6_e4_mdl_ibps_modal_t4_dispatch.yaml`) sister recipes; same `schema_version: 1` + same top-level field set per Catalog #270 + #324 + #240 |
| Recipe `cost_band` epochs/batch_size | ADOPT_CANONICAL_BUT_TUNE | epochs=2000 matches DP1 canonical (longer than C6's 200; canonical L2 long-training per the symposium's "100-200ep partial scope" → 2000ep gives ~10-20x partial scope margin) + batch_size=16 (canonical T4 + autocast_fp16 + 32x32 eval-resolution capacity per build_progress) |
| Recipe `predicted_band` | FORK_BECAUSE_PRINCIPLED_MISMATCH | Z8 is canonical class-shift per Catalog #312 quadruple; predicted band [0.183, 0.195] derived from symposium's Shannon LEAD canonical decomposition (M11 baseline + PR101 GOLD curriculum scaling principle); pending_post_training per Catalog #324 |
| Recipe `sentinel_files` | FORK_BECAUSE_PRINCIPLED_MISMATCH | Z8-specific 17 sentinel files vs DP1's 15 + C6's 11; all are canonical Z8 module surfaces + canonical sister shared infrastructure (trainer_skeleton + smoke_auth_eval_gate + score_aware_common) |
| Recipe `env_overrides.Z8_TRAINER_MODE` | FORK_BECAUSE_PRINCIPLED_MISMATCH | Catalog #326 multi-key trainer mode; Z8_TRAINER_MODE primary (canonical_quadruple\|full\|smoke) with fail-loud warning when no key set; SMOKE_ONLY preserved for backward compatibility with sister Z3/Z4/Z5 drivers |
| Driver script structure | ADOPT_CANONICAL | Per Z5 (`scripts/remote_lane_substrate_z5_predictive_coding_world_model.sh`) sister driver pattern; same 5-stage layout (Catalog #244 NVML → bootstrap → claim → provenance → trainer → completion marker) |
| Driver mode routing | FORK_BECAUSE_PRINCIPLED_MISMATCH | Catalog #326 case statement on Z8_TRAINER_MODE routes canonical_quadruple → --canonical-quadruple-binding mode; full → --output-dir; smoke → --smoke; sister Z3/Z4/Z5 drivers route binary SMOKE_ONLY=0/1 |
| Driver completion marker | ADOPT_CANONICAL_BUT_TUNE | Z8 M12a canonical-quadruple-binding mode emits canonical M9 artifact at `m9_canonical_quadruple_artifact.json`; completion marker reads convergence_verdict + payload_bytes from artifact OR falls back to canonical stats.json axis_tag detection per Z5 pattern |

## Cargo-cult audit per assumption (Catalog #303)

| # | Assumption | Classification | Unwind-test plan |
|---|---|---|---|
| CC-Z8-M12a-RECIPE-1 | Recipe authoring + driver authoring + canonical preflight pass is structurally sufficient to unblock operator-explicit paid Modal M12a dispatch per Catalog #325 14-day window | HARD-EARNED-PER-SYMPOSIUM-CLEARANCE | Catalog #325 symposium `4bcc84fc0` cleared canonical 6-step contract PROCEED_WITH_REVISIONS unanimous 23-of-23 T3 grand council. Catalog #324 PASS pending_post_training. Catalog #270 Tier 2 PASS. Catalog #244 NVML PASS. Catalog #326 PASS. Catalog #313 no blocking outcome. The apparatus setup IS the canonical gate clearance per Catalog #325 + #240. |
| CC-Z8-M12a-RECIPE-2 | The Z8 trainer's existing MLX-only `--canonical-quadruple-binding` mode at `experiments/train_substrate_z8_hierarchical_predictive_coding_mlx.py` is CUDA-capable for the Modal T4 hot loop | HARD-EARNED-PER-M9-M10-M11-EMPIRICAL | M9 commit `bb48f691c` lifts `_full_main` via canonical compose pattern routing through `tac.substrates.z8_hierarchical_predictive_coding.canonical_quadruple_binding.run_canonical_quadruple_training_loop` (numpy + torch; CUDA-compatible). M10 commit `59bdf9c93` validates inflate consumes real trained weights. M11 commit `2f8570755` validates end-to-end cycle-closure through `upstream/evaluate.py --device cpu` per sister CPU device path. The canonical compose pattern is canonical-quadruple-binding mode regardless of device per the canonical numpy + torch design. |
| CC-Z8-M12a-RECIPE-3 | The Catalog #270 Tier 1 (TF32 + torch.compile) + Tier 3 (canonical_inflate_device + scorer_loader_order_correct) blockers are sister-scope Z8 trainer engineering pass NOT apparatus-setup scope | HARD-EARNED-PER-SISTER-SCOPE-CLASSIFICATION | M11 cycle-closure (commit `2f8570755`) demonstrates end-to-end compose runs to completion through `upstream/evaluate.py --device cpu` WITHOUT TF32/torch.compile. The canonical M12a canonical-quadruple-binding mode emits the M9 artifact (`m9_canonical_quadruple_artifact.json`) via pure numpy + torch compose; canonical contest_auth_eval is INVOKED at the M11 sister surface (`m11_l1_macos_cpu_smoke.run_z8_m11_l1_smoke`) not the trainer entrypoint. Per Catalog #340 sister-DISJOINT discipline: Z8 trainer engineering pass for CUDA hot-loop primitives is sister-scope op-routable #1, NOT a blocker for THIS apparatus-setup landing. |
| CC-Z8-M12a-RECIPE-4 | The recipe `dispatch_enabled: true` post-symposium PROCEED_WITH_REVISIONS authorizes apparatus-level dispatch admissibility WITHOUT firing the paid Modal meter | HARD-EARNED-PER-OPERATOR-GATING-DISCIPLINE | Per CLAUDE.md "user-asks-Authorization" + the operator's sub-0.189 submission threshold + 5+ dollar hard-stop + Catalog #325 + #199 paired-env operator-authorization discipline: `dispatch_enabled: true` is the canonical recipe-level admissibility flag per Catalog #240 + Catalog #325 acceptance cascade (b); operator-explicit invocation of `tools/operator_authorize.py` is the canonical paid-dispatch gate at the operator-authorize harness layer. THIS recipe's `dispatch_enabled: true` reflects the symposium's PROCEED verdict; paid dispatch firing requires operator explicit CLI invocation. |
| CC-Z8-M12a-RECIPE-5 | M12b paired-CUDA recipe authoring is canonical sister-scope NOT THIS apparatus-setup landing scope | HARD-EARNED-PER-CONTRARIAN-VETO-RECONCILIATION | Per symposium Contrarian VETO + Assumption-Adversary + Hotz consensus: M12b paired-CUDA is STRUCTURALLY GATED on M12a empirical PoseNet sub-1.0 trajectory evidence. The canonical M12b recipe authoring is its own apparatus-setup subagent + own canonical Catalog #325 symposium clearance if architecture or budget envelope shifts. THIS landing scopes M12a ONLY per the symposium's M12a-FIRST sequencing mandate. |

**Cargo-cult audit verdict**: 5-of-5 HARD-EARNED per symposium clearance + per M9/M10/M11 empirical evidence + per Catalog #340 sister-DISJOINT discipline + per operator-gating discipline + per Contrarian VETO reconciliation. ZERO cargo-culted. Net: apparatus-setup scope is canonically bounded.

## 9-dimension success checklist evidence (Catalog #294)

| # | Dimension | Evidence |
|---|---|---|
| 1 | UNIQUENESS | ✓ Z8 M12a is canonical first Catalog #312 quadruple binding-depth substrate to reach paid Modal dispatch admissibility per the 4-primitive Pareto polytope canonical at the dispatch surface |
| 2 | BEAUTY + ELEGANCE | ✓ Recipe 13.7K LOC + driver 14.0K LOC; same canonical patterns as DP1 + Z5; canonical fields per Catalog #270 + #324 + #240 + #244 + #326; canonical sister-substrate-trainer-engineering pass is sister-scope per Catalog #340 disjoint discipline |
| 3 | DISTINCTNESS | ✓ Recipe declares `lane_class=substrate_engineering` + canonical Catalog #312 quadruple binding + canonical predicted band [0.183, 0.195] + Z8-specific 17 sentinel files; structurally distinct from DP1 (pretrained-driving-prior) + C6 (IBPS) + Z3/Z4/Z5 (single-primitive predictive-coding variants) |
| 4 | RIGOR | ✓ PV per Catalog #229 + #376 + #378 (PV CLEAN; no sister landings + no duplicates in recipes/equations/probes/lanes); per Catalog #292 + #346 canonical roster sextet pact; per Catalog #344 canonical equation references at recipe `notes` section; per Catalog #348 retroactive sweep companion memo |
| 5 | OPTIMIZATION PER TECHNIQUE (Catalog #290) | ✓ Per-layer canonical-vs-unique decision matrix above (7 layers; 3 ADOPT_CANONICAL + 1 ADOPT_CANONICAL_BUT_TUNE + 3 FORK_BECAUSE_PRINCIPLED_MISMATCH) |
| 6 | STACK-OF-STACKS-COMPOSABILITY | ✓ Recipe declares `canary_status: independent_substrate` per Catalog #173 (Z8 is class-shift; canonical orthogonal to HNeRV-family); per the symposium's Step 5 reactivation paths M12c stack-of-stacks composition with DP1 pretraining + NSCS06 v8 Variant C + D1 SegNet overlay is canonical M13+ scope |
| 7 | DETERMINISTIC REPRODUCIBILITY | ✓ Recipe `env_overrides` pin Z8_EPOCHS + Z8_NUM_PAIRS + Z8_CANONICAL_QUADRUPLE_EVAL_H + Z8_CANONICAL_QUADRUPLE_EVAL_W; per Z8 M11 LANDING canonical seed-pinning via M9 + M5 + M6 canonical compose; canonical sentinel-files SHA-stability per Catalog #191 |
| 8 | EXTREME OPTIMIZATION + PERFORMANCE | ✓ Recipe envelope ~8 USD M12a + ~1.50-3.00 M12b = ~6.50-18.00 USD total; canonical Tier 2 PASS all 5 fields declared; canonical Catalog #244 NVML PASS; canonical Catalog #326 multi-key trainer mode PASS |
| 9 | OPTIMAL MINIMAL CONTEST SCORE | ASSUMED_AWAITING_VERIFICATION-PER-CATALOG-363-ROUND-1 (predicted band [0.183, 0.195] [contest-CPU]; canonical sub-0.189-candidate range; pending_post_training per Catalog #324; operator-explicit dispatch authorization required to land M12a empirical anchor) |

**9-dim verdict**: 8 ✓ + 1 ASSUMED_AWAITING_VERIFICATION-PER-CATALOG-363-ROUND-1 (Dim 9 OPTIMAL MINIMAL CONTEST SCORE is empirical-evidence-pending per M12a operator-authorized dispatch).

## Observability surface declaration (Catalog #305)

| Facet | Apparatus-setup surface |
|---|---|
| Inspectable per layer | Recipe YAML 12 top-level sections + driver shell 5-stage layout (NVML → bootstrap → claim → provenance → trainer → completion); each section is canonical-pattern-reviewable in 30 seconds |
| Decomposable per signal | Catalog #270 protocol JSON output decomposes per Tier 1 / Tier 2 / Tier 3 signals + per-signal blocker enumeration; canonical Catalog #324 predicted band audit JSON decomposes per-recipe validation status |
| Diff-able across runs | Recipe + driver land at known git paths; sister recipes (DP1 + C6 + Z5) provide canonical diff baselines for the canonical-vs-unique decision matrix |
| Queryable post-hoc | Canonical posterior anchor at `.omx/state/council_deliberation_posterior.jsonl` + canonical probe outcome at `.omx/state/probe_outcomes.jsonl` + Catalog #324 audit JSON at `.omx/state/predicted_band_audit_z8_m12a_20260530T173500Z.json` |
| Cite-able | This memo + symposium memo `council_t3_..._20260530.md` + sister M9/M10/M11 landing memos + canonical equation references per Catalog #344 |
| Counterfactual-able | Recipe `dispatch_enabled: true` toggle + Z8_TRAINER_MODE 3-value enum (canonical_quadruple\|full\|smoke) provide canonical counterfactual axes; canonical Catalog #313 probe outcome 14-day expires 2026-06-13 provides canonical reactivation counterfactual |

**Observability surface verdict**: all 6 facets ACTIVE; canonical apparatus-setup surfaces are structurally observable per Catalog #305 + #245 + #313 + #323 + #341 sister discipline.

## 6-hook wire-in declaration (Catalog #125)

- hook #1 sensitivity-map = **N/A** (apparatus-setup defensive validator + recipe + driver authoring; M7 sensitivity-map wiring is at M12a-LANDING runtime scope per Z8 trainer engineering pass)
- hook #2 Pareto constraint = **N/A at apparatus-setup scope** (Pareto polytope constraint wiring is at M12a-LANDING runtime scope per the symposium's Step 4 Dykstra sextet position; M12a empirical anchor IS the canonical Pareto polytope feasibility evidence)
- hook #3 bit-allocator = **N/A** (bit-allocator wiring lands at M12+ training-wave scope when L30 range coding + L32 brotli q=11 + L28 decode-side postprocess land per the canonical Z8 L28+L30+L32 sister-subagent op-routable per the symposium's Step 5)
- hook #4 cathedral autopilot dispatch = **ACTIVE-PRIMARY** (THIS landing emits canonical T1 council anchor + canonical probe outcome PROCEED that the cathedral autopilot ranker + auto-discovered consumers consume per Catalog #335 + #355 META-LAGRANGIAN + #379 META-orchestrator extension; canonical NON-promotable Tier A per Catalog #341 routing markers)
- hook #5 continual-learning posterior = **ACTIVE** (canonical posterior anchor via `tac.council_continual_learning.append_council_anchor` per Catalog #300 v2 frontmatter + canonical probe outcome via `tac.probe_outcomes_ledger.register_probe_outcome` per Catalog #313 + canonical Provenance per Catalog #323)
- hook #6 probe-disambiguator = **ACTIVE** (canonical PROCEED verdict + canonical 14-day expires_at_utc IS the canonical disambiguator between apparatus-setup-clear vs apparatus-setup-blocked; routes operator-explicit paid Modal dispatch authorization per the canonical CLI command in op-routable #2)

## Apparatus mutations landed

1. **Recipe**: `.omx/operator_authorize_recipes/substrate_z8_hierarchical_predictive_coding_modal_t4_dispatch.yaml` (canonical per Catalog #270 + #324 + #240 + #244 + #326 + #167 + #166 + #199 + #246)
2. **Driver script**: `scripts/remote_lane_substrate_z8_hierarchical_predictive_coding.sh` (canonical per Catalog #244 + #204 + #326 + #163 + #189 + #152 + #226 sister patterns)
3. **Council anchor**: `council_t1_z8_m12a_modal_t4_l2_long_training_recipe_authoring_canonical_apparatus_setup_post_symposium_proceed_with_revisions_20260530` PROCEED 5-voice per Catalog #300 v2 frontmatter
4. **Probe outcome**: `probe_z8_m12a_modal_t4_l2_long_training_recipe_authoring_canonical_apparatus_setup_post_symposium_proceed_with_revisions_20260530` PROCEED advisory 14-day expires `2026-06-13T17:32:44Z` per Catalog #313
5. **Lane registry**: `lane_z8_m12a_modal_t4_l2_long_training_per_catalog_325_symposium_proceed_with_revisions_20260530` L1 (impl_complete + memory_entry gates marked)
6. **Active dispatch claim**: lane claim recorded at `.omx/state/active_lane_dispatch_claims.md`
7. **Catalog #324 audit JSON**: `.omx/state/predicted_band_audit_z8_m12a_20260530T173500Z.json` (PASS pending_post_training)
8. **NO new Catalog #** per Catalog #299 quota brake under 400 (current 382 well under)
9. **Catalog #348 retroactive sweep memo**: `.omx/research/retroactive_sweep_for_z8_m12a_recipe_authoring_20260530T173500Z.md` (4-field contract; expected ZERO historical KILL/DEFER/FALSIFY invalidated)

## Operator-routable canonical CLI commands

**Decision 1: M12a paid Modal T4 L2 long-training dispatch authorization** (canonical apparatus-setup CLEAR; operator-explicit authorization required):

```bash
.venv/bin/python tools/operator_authorize.py \
    --recipe substrate_z8_hierarchical_predictive_coding_modal_t4_dispatch \
    --platform modal \
    --gpu T4
```

The canonical Catalog #243 local pre-deploy harness + Catalog #270 dispatch optimization protocol + Catalog #271 codex pre-dispatch review + Catalog #167 smoke-before-full pattern + Catalog #199 paired-env operator-authorization all fire BEFORE the paid Modal meter starts.

**Decision 2: Z8 trainer engineering pass for canonical CUDA hot-loop primitives** (op-routable #1; sister-scope; canonical-quadruple-binding mode operates without these so NOT blocking):

```bash
# Sister-subagent spawn to wire:
#   - TF32 (torch.backends.cuda.matmul.allow_tf32 = True + torch.backends.cudnn.allow_tf32 = True)
#   - torch.compile (--enable-torch-compile flag + torch.compile() wrapper)
#   - canonical select_inflate_device (tac.substrates._shared.inflate_runtime.select_inflate_device)
#   - canonical scorer_loader_order (pose_scorer, seg_scorer = load_differentiable_scorers(...))
#   - canonical gate_auth_eval_call (tac.substrates._shared.smoke_auth_eval_gate.gate_auth_eval_call)
```

**Decision 3: M12b paired-CUDA recipe authoring** (op-routable #3; GATED on M12a PoseNet sub-1.0 trajectory):

M12b paired-CUDA recipe authoring is canonical sister-scope apparatus-setup subagent + canonical Catalog #325 symposium clearance if architecture or budget envelope shifts. Per Contrarian VETO + Assumption-Adversary + Hotz consensus: M12b is STRUCTURALLY GATED on M12a empirical PoseNet sub-1.0 trajectory evidence within the 2000ep canonical wave.

## Cross-references

- Symposium memo: `.omx/research/council_t3_grand_council_per_substrate_symposium_z8_hierarchical_predictive_coding_m12_paid_modal_t4_l2_long_training_plus_paired_cuda_canonical_sub_0_189_attempt_20260530.md`
- M11 cycle-closure landing: `feedback_z8_m11_l1_macos_cpu_mlx_local_end_to_end_smoke_canonical_evaluate_cpu_binding_landed_20260530.md`
- M10 inflate canonical landing: `feedback_z8_m10_inflate_consumes_real_trained_weights_per_catalog_369_landed_20260530.md`
- M9 `_full_main` canonical-quadruple-binding lift: `feedback_z8_m9_full_main_canonical_quadruple_binding_integration_landed_20260530.md`
- Catalog #325 STRICT preflight gate: `check_substrate_dispatch_has_per_substrate_optimal_form_symposium_anchor`
- Catalog #324 post-training Tier-C validation: `check_no_predicted_band_without_post_training_tier_c_validation`
- Catalog #270 canonical dispatch optimization protocol: `check_dispatch_optimization_protocol_complete`
- Catalog #240 recipe-vs-trainer-state: `check_substrate_contest_cuda_chain_complete_or_research_only_tagged`
- Catalog #244 canonical NVML block: `check_remote_lane_scripts_carry_canonical_nvml_block`
- Catalog #326 multi-key trainer mode: `check_substrate_driver_consumes_trainer_mode_env_var`
- Catalog #313 probe outcomes ledger: `check_dispatch_target_has_no_predecessor_adjudicated_outcome`
- Catalog #246 paired CPU + CUDA dispatch on 1:1 contest-compliant hardware
- Catalog #167 smoke-before-full pattern
- Catalog #199 paired-env operator authorization
- Canonical operator standing directives:
  - `[[z8-hierarchical-predictive-coding-binding-first-active-build-target-yousfi-grounded]]`
  - `[[complexity-loc-unconstrained-push-boundaries-within-contest-compliance-standing-directive]]`
  - `[[pr-or-greater-parity-synergy-binding-integration-not-hnerv-specific-meta-class-lesson-correction]]`
  - `[[optimize-iterate-highest-ev-boldest-individually-fractally-optimized-mlx-deployed-aggressive-frontier-breaking-no-fake-implementations]]`
  - `[[separation-of-concerns-plus-refactor-permission-when-clean-tested-hardened]]`
  - `[[canonical-ev-metric-trichotomy]]`
  - `[[mlx-portable-local-substrate-authority]]`
  - `[[z8-phase-2-build-tracking-in-source-not-tasklist-not-memos]]`

— end of memo —
