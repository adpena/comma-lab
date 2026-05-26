---
name: z7-mamba2-full-main-design-20260518
metadata:
  node_type: memory
  council_tier: T1
  council_attendees:
    - Hafner
    - Quantizr
    - Contrarian
    - Assumption-Adversary
  council_quorum_met: false
  council_verdict: DESIGN_MEMO_ONLY_PENDING_WAVE_N_PLUS_1_COUNCIL_APPROVAL
  council_dissent:
    - member: Contrarian
      verbatim: "This is a DESIGN MEMO not a build authorization. The _full_main implementation path described here is pre-cached for the Wave-N+1 council that convenes AFTER (a) Z6 Candidate 4c TRAINED paired exact-eval lands AND (b) Z7-LSTM/GRU FALLBACK Wave 2 outcome lands AND (c) C6 IBPS Phase 2 empirical β-optimal lands. NO implementation work authorized at THIS memo."
  council_assumption_adversary_verdict:
    - assumption: "_full_main design memo IS sufficient deliverable for THIS lane (no IMPLEMENTATION required per parent prompt)"
      classification: HARD-EARNED
      rationale: "Per parent prompt 'NO _full_main IMPLEMENTATION (design memos only — leaves NotImplementedError gates in place)'. This memo satisfies the deliverable contract without violating the Catalog #240 + #325 scaffold discipline."
  council_decisions_recorded:
    - "VERDICT: DESIGN_MEMO_ONLY_PENDING_WAVE_N_PLUS_1_COUNCIL_APPROVAL — _full_main implementation path documented as pre-cached design memo for the Wave-N+1 council convened after dependency cascade lands. NO build authorization from this memo."
  council_predicted_mission_contribution: frontier_breaking
  council_override_invoked: false
  council_override_rationale: ""
  horizon_class: asymptotic_pursuit
  canonical_frontier_anchor:
    contest_cpu: "0.19205 [contest-CPU] (lane pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean; archive sha 6bae0201; per Catalog #316)"
    contest_cuda: "0.20533 [contest-CUDA] (lane pr106_format0d_latent_score_table; archive sha 9cb989cef519; per Catalog #316)"
  deferred_substrate_id: time_traveler_l5_z7_mamba2
  deferred_substrate_retrospective_due_utc: "2026-06-17T00:00:00Z"
  predicted_dispatch_risk: 0
  originSessionId: lane_z7_mamba2_lstm_full_landing_integration_audit_20260518
  related_deliberation_ids:
    - council_per_substrate_symposium_z7_mamba2_plus_lstm_unified_20260518
    - z7_mamba2_substrate_design_memo_20260518
    - council_per_substrate_symposium_z7_lstm_predictive_coding_20260517
---

# Z7-Mamba-2 PRIMARY `_full_main` DESIGN MEMO 2026-05-18

**Lane**: `lane_z7_mamba2_lstm_full_landing_integration_audit_20260518` (sister deliverable; design only)
**Parent design memo**: `.omx/research/z7_mamba2_substrate_design_memo_20260518.md` (§7 Architectural specification + §13 LOCAL M5 MAX PROXY)
**Parent symposium**: `.omx/research/council_per_substrate_symposium_z7_mamba2_plus_lstm_unified_20260518.md` (THIS lane)
**Catalog #240 compliance**: this memo is DESIGN ONLY; `_full_main` raises NotImplementedError per current scaffold; implementation deferred to Wave-N+1 PROCEED-unconditional council verdict.

## TL;DR (60 seconds)

The Z7-Mamba-2 PRIMARY `_full_main` implementation path follows the canonical Z6-v1 training curriculum with Mamba-2 selective state-space predictor substituted at the substrate-distinguishing layer. Wave 2 smoke = 100ep on a single Modal T4 ($5-10 envelope; 60-90 min wall-clock); Wave 3 full = 1000ep on Modal A100 ($20-30 envelope; 6-10 hour wall-clock). Byte budget per the empirical predicted band [0.167, 0.184] ⇒ ~110-140 KB archive (Z7MCM2 grammar). Post-training Tier-C validation per Catalog #324 = density measurement on the TRAINED archive after Wave 2 smoke completes.

**Build deferred** to Wave N+1 PROCEED-unconditional verdict per parent symposium Revision #1 binding (cascade step b: Z7-Mamba-2 PRIMARY fires SECOND after Z7-LSTM/GRU FALLBACK Wave 2 outcome lands).

## 1. Canonical training curriculum

### Wave 2 smoke (100ep, ~$5-10, 60-90 min Modal T4)
```python
def _full_main(args: argparse.Namespace) -> int:
    # 1. Resolve config
    config = Mamba2PredictorConfig(
        latent_dim=24,
        d_model=64,
        d_state=16,
        expand=2,
        ego_motion_dim=8,                        # PoseNet-projection baseline; OR scorer_logit_compressed per --ego-source flag
        ego_source=args.ego_source,              # "posenet_projection" OR "scorer_logit_compressed"
        backend="mamba_ssm" if torch.cuda.is_available() else "reference_torch",
    )

    # 2. Substrate skeleton (Z6-v1 sister pattern)
    device = device_or_die()                     # Catalog #190 detect_hardware_substrate
    pose_scorer, seg_scorer = load_differentiable_scorers(device)   # Catalog #222 canonical order
    patch_upstream_yuv6_globally()               # Catalog #164 + #226 canonical preprocess

    # 3. Real-pair training data (NO synthetic per Catalog #114)
    pairs = decode_real_pairs(args.video_path)   # 600 pairs from upstream/videos/0.mkv via pyav
    # pairs :: list of (B=1, T=2, C=3, H=384, W=512) tensors per pair

    # 4. Z7-Mamba-2 model
    encoder = _Z6Encoder()                       # ADOPT from Z6-v1 (Catalog #290 layer 2)
    decoder = _Z6Decoder()                       # ADOPT from Z6-v1
    predictor = Mamba2Predictor(config)          # FORK: substrate-distinguishing primitive
    model = Z7Mamba2Substrate(encoder, decoder, predictor)
    model = model.to(device)

    # 5. EMA(0.997) + AdamW + cosine schedule per Tier-1 engineering (Catalogs #172/#178/#179/#180)
    ema = EMAModel(model, decay=0.997)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 6. β-IB-Lagrangian initialization (per parent symposium Revision #4 binding)
    beta_ib = args.beta_ib if args.beta_ib else 0.5    # Literature-canonical default until C6 Phase 2 empirical β-optimal lands

    # 7. Training loop with eval_roundtrip=True (CLAUDE.md non-negotiable)
    for epoch in range(args.epochs):
        for pair_t, latent_pair_t_minus_1, ego_motion_t in iter_pairs(pairs, predictor):
            with autocast_fp16():                 # Catalog #172
                predicted_latent = predictor(latent_pair_t_minus_1, ego_motion_t)
                residual = latent_pair_t - predicted_latent
                # Quantize residual int8 + entropy code (estimate of archive rate)
                rate_est = estimate_rate_term(residual)
                # Score-aware loss via canonical helper (Catalog #164 + #226)
                loss = score_pair_components(
                    pair_t, model, pose_scorer, seg_scorer,
                    eval_roundtrip=True,         # NON-NEGOTIABLE
                    beta_ib=beta_ib,
                    rate_estimate=rate_est,
                )
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            ema.update(model)
        scheduler.step()

        # Eval every N epochs with EMA shadow (per Catalog #88)
        if epoch % args.eval_every == 0:
            with torch.no_grad():                # Catalog #180
                eval_score = eval_with_ema_shadow(model, ema, pairs, pose_scorer, seg_scorer)
                print(f"epoch={epoch} eval_score={eval_score:.5f}")

    # 8. Export: pack archive (Z7MCM2 grammar) using EMA shadow (NEVER live weights)
    apply_ema_shadow_to_model(model, ema)
    archive_bytes = pack_archive_z7mcm2(
        encoder=model.encoder, decoder=model.decoder, predictor=model.predictor,
        latent_init=compute_latent_init(pairs, model),
        residuals=compute_residuals(pairs, model),
        ego_motion=compute_ego_motion(pairs, args.ego_source),
        meta={"config": config.to_dict(), "epochs": args.epochs, "device": device.type},
    )
    archive_path = args.output_dir / "0.bin"
    archive_path.write_bytes(archive_bytes)

    # 9. Canonical auth-eval via gate (Catalog #226)
    auth_eval_result = gate_auth_eval_call(
        archive_path=archive_path,
        inflate_sh_path=args.output_dir / "inflate.sh",
        json_out_path=args.output_dir / f"contest_auth_eval_{auth_eval_device}.json",
        device=auth_eval_device,
    )

    # 10. Require contest-CUDA claim before promotion (Catalog #127 + #193)
    require_contest_cuda_auth_eval_claim(auth_eval_result)

    # 11. Stats output with fail-closed authority fields (Catalog #221)
    stats = {
        "epochs": args.epochs,
        "auth_eval_score": auth_eval_result.score,
        "auth_eval_score_axis": auth_eval_result.score_axis,
        "auth_eval_score_claim_valid": True,
        "promotion_eligible": False,             # Wave 2 smoke is research-only until Wave 3
        "rank_or_kill_eligible": False,
        "result_review_blockers": [],
        "archive_sha256": sha256(archive_bytes).hexdigest(),
        "archive_bytes": len(archive_bytes),
    }
    write_stats_json(args.output_dir, stats)

    return 0
```

### Wave 3 full dispatch (1000ep, ~$20-30, 6-10 hour Modal A100)
SAME structure as Wave 2 smoke; differences:
- `args.epochs = 1000` (vs 100)
- `args.eval_every = 50` (vs 10)
- Modal A100 GPU with mamba_ssm backend (vs reference_torch fallback on T4 or MPS)
- Larger torch.compile (Catalog #179) acceptance (justified by 10x epoch budget)
- Predicted ΔS realization: [0.167, 0.184] [contest-CPU] from research wave §0 TOP-5 #2

## 2. Byte budget per empirical predicted band

Per parent design memo §7 parameter count breakdown:
- Encoder (Z6-v1 reused): ~37K params
- Decoder (Z6-v1 reused): ~37K params
- Mamba-2 block (d_model=64, d_state=16, expand=2): ~25-40K params
- Input projection (24+8 → 64): ~3K params
- Output projection (64 → 24): ~2K params
- Ego MLP: ~3K params
- Latent init: ~50K params
- **Total: ~155-175K params**

Archive size estimate (Z7MCM2 grammar):
- HEADER ~1 KB
- encoder_state_dict_fp16_brotli ~30 KB
- decoder_state_dict_fp16_brotli ~30 KB
- predictor_state_dict_fp16_brotli ~30-50 KB
- latent_init_int8 ~5 KB
- residuals_int8 ~10 KB
- ego_motion_int8_sidecar ~3 KB
- meta_json ~0.5 KB
- **Total: ~110-140 KB**

Rate term contribution: `25 * 130_000 / 37_545_489 ≈ 0.0866`.

To realize predicted ΔS band [-0.025, -0.008] over PR101 frontier 0.19205:
- seg + pose at PR101 frontier ~= 0.19205 - 0.0866 (rate) = 0.10545
- Z7-Mamba-2 needs seg + pose ≤ 0.10545 - 0.008 = 0.09745 (upper-bound realization)
- Z7-Mamba-2 best-case seg + pose ≤ 0.10545 - 0.025 = 0.08045 (lower-bound realization)
- Translates to ~7-24% reduction in seg + pose loss vs PR101 frontier
- Per parent design memo §8 Dykstra-feasibility intersection: Mamba-2 selective state-space + Wyner-Ziv implicit side-info + ego-motion conditioning predicted to deliver 10-20% residual entropy reduction → maps cleanly to the predicted band

## 3. Wave N+1 dispatch precondition checklist

Before Wave 2 smoke fires, ALL of the following must be satisfied:
- [ ] **TRAINED Z6 Candidate 4c paired exact-eval lands** (NOT the 2026-05-18 zero-epoch packet) per Z7-Mamba-2 parent design memo §11 cross-pollination wiring #1 + parent symposium Revision #3 binding.
- [ ] **Z7-LSTM/GRU FALLBACK Wave 2 outcome lands** per parent symposium Revision #1 dispatch cascade Path 1 default (cheap-signal-first).
- [ ] **C6 IBPS Phase 2 empirical β-optimal anchor lands** per parent design memo §11 cross-pollination wiring #2 + parent symposium Revision #4 binding.
- [ ] **Wave N+1 council convened** (T2 sextet minimum; T3 grand-council preferred per Catalog #325 + #300) with PROCEED-unconditional verdict on Z7-Mamba-2 dispatch authorization.
- [ ] **mamba_ssm install path verified** on Modal A100 image per parent design memo §2 CC-4 HARD-EARNED-PARTIAL + Catalog #270 Tier 1/2/3 protocol.
- [ ] **MPS proxy training pattern landed** per parent design memo §13 LOCAL M5 MAX PROXY (operator-routable; produces curve-shape evidence + CC-1/CC-2 disambiguation BEFORE paid dispatch).

## 4. Post-training Tier-C validation per Catalog #324

After Wave 2 smoke completes, MUST measure Tier-C density on TRAINED Z7-Mamba-2 archive via:
```bash
.venv/bin/python tools/mdl_scorer_conditional_ablation.py \
    --tier c \
    --archive-path experiments/results/<z7_mamba2_wave_2_smoke_dir>/0.bin \
    --output-json .omx/state/mdl_ablation_z7_mamba2_<utc>.json
```

Required outcome semantics per Catalog #324:
- `predicted_band_validation_status: validated_post_training` (replaces `pending_post_training`) ONLY when Tier-C density < 0.70 (within-class threshold)
- IF Tier-C density ≥ 0.70 (within-class saturated): Catalog #219 fires; Z7-Mamba-2 lane REFUSED L2+ promotion; reactivation via class-shift architecture per Catalog #308

## 5. Wave 3 full dispatch operator-authorize recipe extension

After Wave 2 smoke PROCEED-unconditional, the operator-authorize recipe at `.omx/operator_authorize_recipes/substrate_time_traveler_l5_z7_mamba2_modal_a100_dispatch.yaml` flips:
- `research_only: false` (from `true`)
- `dispatch_enabled: true` (from `false`)
- `predicted_band_validation_status: validated_post_training` (from `pending_post_training`)
- `predicted_band` field reflects measured Wave 2 smoke result (NOT predicted [0.167, 0.184] band)
- `min_smoke_gpu: A100` (Catalog #215)
- `min_vram_gb: 40` (Catalog #170; Mamba-2 + 600-pair batch fits A100 40GB easily)
- `target_modes: [contest_exact_eval, asymptotic_pursuit_pivot_branch_b]` (Catalog #182)
- `canary_status: post_canary_dependent` + `canary_dependency: time_traveler_l5_z7_lstm_predictive_coding` (Catalog #173)
- `video_input_strategy: per_dispatch_local_copy` (Catalog #171)
- `pyav_decode_strategy: cpu_thread_async_upload` (Catalog #181)

## 6. Bidirectional cross-pollination with Z6 4c + C6 IBPS Phase 2 + ATW V2-1

### Z6 4c (PRIMARY ego-source dependency)
- IF TRAINED Z6 4c paired exact-eval lands full-FiLM-WIN at ΔS ≥ 0.005 contest-CUDA: Z7-Mamba-2 dispatches with `--ego-source scorer_logit_compressed` (Z6 4c winning channel).
- IF TRAINED Z6 4c paired exact-eval lands DEFER (identity-WIN or |ΔS| < 0.005): Z7-Mamba-2 dispatches with `--ego-source posenet_projection` (Z6-v1 baseline).
- Both paths supported via runtime-configurable flag (verified via `test_runtime_configurable_ego_source_posenet_projection_baseline` + `test_runtime_configurable_ego_source_scorer_logit_compressed` BOTH PASS).

### C6 IBPS Phase 2 (β-IB-Lagrangian dependency)
- IF C6 Phase 2 empirical β-optimal lands within [0.1, 1.0] range: Z7-Mamba-2 dispatches with `--beta-ib <C6_empirical_anchor>`.
- IF C6 Phase 2 Phase 3 cargo-cult-unwind redesign required (per Assumption-Adversary VETO): Z7-Mamba-2 dispatch DEFERRED until C6 Phase 3 lands.
- Default literature-canonical β=0.5 used if C6 anchor unavailable at dispatch time.

### ATW V2-1 (channel-pick cross-reference, NOT direct dependency)
- ATW V2-1 channel-pick (scorer-softmax-sketch / per-region SegNet softmax histograms / pose-bin discretization) provides EVIDENCE about which side-info channel structurally beats noise floor at contest scorer subspace.
- IF ATW V2-1 channel-pick lands a WINNING channel that overlaps Z7-Mamba-2 ego-source choices: Z7-Mamba-2 `--ego-source` flag value cross-references the V2-1 winning channel for consistency.
- NOT a hard dependency; Z7-Mamba-2 dispatch can fire WITHOUT ATW V2-1 outcome via `--ego-source posenet_projection` baseline.

## 7. Failure-mode contingency paths

### Z7-Mamba-2 Wave 2 smoke LOSS (|ΔS| < 0.005 vs Z7-LSTM/GRU FALLBACK at SAME archive bytes)
→ Mamba-2-vs-GRU expressive-power CARGO-CULTED-PENDING-EMPIRICAL hypothesis (CC-2) empirically FALSIFIED at single-level surface.
→ Advance to: (a) Z8 hierarchical $42 envelope OR (b) Z7-RWKV-7 $20-25 envelope OR (c) DEFER per Catalog #298.

### Z7-Mamba-2 Wave 2 smoke WIN BUT Wave 3 full LOSS
→ Mamba-2 advantage at 100ep does NOT survive 1000ep convergence.
→ Investigation paths: (a) hyperparameter sweep (d_state ∈ {8, 16, 32, 64} per CC-9); (b) learning rate sweep; (c) architectural variant (multi-block Mamba-2 stack).

### mamba_ssm install fails on Modal A100
→ Fall back to pure-PyTorch Mamba-2 reference implementation (reference_torch backend; ~10× slower; documented in parent design memo §13).
→ Wave 3 full dispatch wall-clock budget extends from 6-10 hours to 60-100 hours; cost envelope $200-300 (NOT acceptable).
→ Operator-routable: pre-verify mamba_ssm install BEFORE Wave 3 dispatch authorization.

## 8. Test coverage requirements per Wave N+1 build

When Z7-Mamba-2 trainer is built (`_full_main` replaces NotImplementedError), the following tests MUST pass:

### Existing tests (36 PASS at this memo; per Stage 1 verification)
- ALL 36 tests in `src/tac/tests/test_z7_mamba2_scaffold.py` continue to PASS.

### NEW tests required for `_full_main` implementation
1. `test_z7_mamba2_full_main_produces_byte_closed_archive` — Wave 2 smoke produces valid Z7MCM2 archive.
2. `test_z7_mamba2_full_main_archive_passes_strict_scorer_rule` — no scorers loaded at inflate time (Catalog #6).
3. `test_z7_mamba2_full_main_archive_consumes_predictor_bytes` — byte-mutation smoke per Catalog #272 distinguishing-feature integration contract.
4. `test_z7_mamba2_full_main_uses_canonical_auth_eval_helper` — Catalog #226 routing.
5. `test_z7_mamba2_full_main_writes_fail_closed_stats_authority_fields` — Catalog #221 + #193.
6. `test_z7_mamba2_full_main_uses_ema_shadow_at_eval_and_archive` — Catalog #88 EMA discipline.
7. `test_z7_mamba2_full_main_uses_real_pairs_no_synthetic` — Catalog #114 (no `make_synthetic_*` in non-smoke path).
8. `test_z7_mamba2_full_main_eval_roundtrip_true_in_loss` — CLAUDE.md non-negotiable.
9. `test_z7_mamba2_full_main_records_hardware_substrate_via_canonical_helper` — Catalog #190 detect_hardware_substrate(axis='cpu' or 'cuda').

## 9. Cross-references

- **Parent design memo**: `.omx/research/z7_mamba2_substrate_design_memo_20260518.md` §7 Architectural specification + §13 LOCAL M5 MAX PROXY
- **Parent symposium**: `.omx/research/council_per_substrate_symposium_z7_mamba2_plus_lstm_unified_20260518.md` (THIS lane; Revision #1-#7 binding)
- **Sister Z7-LSTM/GRU _full_main design**: `.omx/research/z7_lstm_full_main_design_20260518.md` (sister deliverable from this lane)
- **Sister integration audit**: `.omx/research/z7_integration_audit_20260518.md` (sister deliverable from this lane)
- **Sister cross-pollination decision tree**: `.omx/research/z7_z6_4c_c6_ibps_atw_v2_1_cross_pollination_decision_tree_20260518.md` (sister deliverable from this lane)
- **Catalog #240**: substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY
- **Catalog #324**: post-training Tier-C validation required before predicted_band promotion
- **Catalog #325**: per-substrate-symposium-evidence requirement (this lane satisfies for 14-day window)

## Observability surface

### Observability invariants

This memo IS the canonical pre-cached `_full_main` implementation design observability surface for Z7-Mamba-2 PRIMARY. The 6 facets per Catalog #305:

1. **Inspectable per layer** — 11-step Wave 2 smoke pseudocode (§1) decomposes _full_main into 11 explicit operations.
2. **Decomposable per signal** — byte budget §2 decomposes archive into 8 byte-counted sections.
3. **Diff-able across runs** — Wave 2 smoke vs Wave 3 full §1 enumerates per-section differences (epoch count + eval cadence + GPU + torch.compile + predicted band realization).
4. **Queryable post-hoc** — frontmatter `related_deliberation_ids` + cross-references §9 provide cite-chain.
5. **Cite-able** — `originSessionId` + `deferred_substrate_id` provide cite-chain.
6. **Counterfactual-able** — §7 failure-mode contingency paths enumerate 3 explicit counterfactual scenarios + reactivation paths.


# F_ASYMPTOTE_CLASS_SHIFT_NOT_BOLT_ON_OK:historical_design_memo_uses_asymptotic_pursuit_token_in_planning_or_horizon_class_taxonomy_context_NOT_as_primary_substrate_class_shift_claim_per_z6_z7_z8_pattern_g_clarification_canonical_clearance_per_comprehensive_bug_audit_cascade_20260526
