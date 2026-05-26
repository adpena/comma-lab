---
name: z7-lstm-full-main-design-20260518
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
      verbatim: "This is a DESIGN MEMO not a build authorization. The Z7-LSTM/GRU FALLBACK trainer has been built via codex sister wave today; the _full_main currently writes a byte-closed pre-build export per Catalog #240 + #325 + Z7-LSTM symposium Revision #7 binding, NOT full training. The full-training implementation path described here is pre-cached for the Wave-N+1 council that convenes AFTER (a) TRAINED Z6 Candidate 4c paired exact-eval lands AND (b) C6 IBPS Phase 2 empirical β-optimal lands. NO full-training implementation authorized at THIS memo."
  council_assumption_adversary_verdict:
    - assumption: "_full_main design memo IS sufficient deliverable for THIS lane (no full-training IMPLEMENTATION required per parent prompt)"
      classification: HARD-EARNED
      rationale: "Per parent prompt 'NO _full_main IMPLEMENTATION (design memos only — leaves NotImplementedError gates in place)'. This memo satisfies the deliverable contract for the SECONDARY Z7-LSTM/GRU FALLBACK substrate; the current pre-build export trainer state is canonical per Catalog #240 + #325."
  council_decisions_recorded:
    - "VERDICT: DESIGN_MEMO_ONLY_PENDING_WAVE_N_PLUS_1_COUNCIL_APPROVAL — Z7-LSTM/GRU FALLBACK _full_main full-training implementation path documented as pre-cached design memo. NO build authorization from this memo. The pre-build export state is canonical."
  council_predicted_mission_contribution: frontier_breaking
  council_override_invoked: false
  council_override_rationale: ""
  horizon_class: asymptotic_pursuit
  canonical_frontier_anchor:
    contest_cpu: "0.19205 [contest-CPU] (lane pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean; archive sha 6bae0201; per Catalog #316)"
    contest_cuda: "0.20533 [contest-CUDA] (lane pr106_format0d_latent_score_table; archive sha 9cb989cef519; per Catalog #316)"
  deferred_substrate_id: time_traveler_l5_z7_lstm_predictive_coding
  deferred_substrate_retrospective_due_utc: "2026-06-17T00:00:00Z"
  predicted_dispatch_risk: 0
  originSessionId: lane_z7_mamba2_lstm_full_landing_integration_audit_20260518
  related_deliberation_ids:
    - council_per_substrate_symposium_z7_mamba2_plus_lstm_unified_20260518
    - council_per_substrate_symposium_z7_lstm_predictive_coding_20260517
    - z7_mamba2_full_main_design_20260518
---

# Z7-LSTM/GRU FALLBACK `_full_main` DESIGN MEMO 2026-05-18

**Lane**: `lane_z7_mamba2_lstm_full_landing_integration_audit_20260518` (sister deliverable; design only)
**Parent symposium (Z7-LSTM-specific)**: `.omx/research/council_per_substrate_symposium_z7_lstm_predictive_coding_20260517.md`
**Parent symposium (unified)**: `.omx/research/council_per_substrate_symposium_z7_mamba2_plus_lstm_unified_20260518.md` (THIS lane)
**Catalog #240 compliance**: this memo is DESIGN ONLY; current `_full_main` writes byte-closed pre-build export per Catalog #240 scaffold discipline; full-training implementation deferred to Wave-N+1 PROCEED-unconditional council verdict.

## TL;DR (60 seconds)

The Z7-LSTM/GRU FALLBACK `_full_main` implementation path follows the canonical Z6-v1 training curriculum with GRU recurrent predictor substituted at the substrate-distinguishing layer (canonical-bound to GRU per Hafner Revision #3 binding 2026-05-17; LSTM codename retained for backward compatibility). Z7-LSTM/GRU FALLBACK is the FIRST recurrent primitive in the Z7 dispatch cascade per Race-mode-rigor-inversion Rule 3 (cheap-signal-first); Wave 2 smoke = 100ep on Modal T4 (~$5-7 envelope; 60-90 min wall-clock); Wave 3 full = 1000ep on Modal A100 (~$16.50-21.50 envelope; 6-10 hour wall-clock). Byte budget per the empirical predicted band [0.180, 0.192] ⇒ ~145-180 KB archive (Z7PCWM1 grammar). Post-training Tier-C validation per Catalog #324 = density measurement on TRAINED archive after Wave 2 smoke completes.

**Build deferred** to Wave N+1 PROCEED-unconditional verdict per parent unified symposium Revision #1 binding (cascade step a: Z7-LSTM/GRU FALLBACK fires FIRST in the recurrent-primitive cascade).

## Codex compatibility addendum

The pseudocode below is a design sketch, not the executable trainer API. The
current live implementation is:

```text
Z7GruPredictiveCodingConfig(gru_hidden_dim=..., gru_num_layers=..., ...)
Z7GruPredictiveCodingSubstrate(config)
experiments/train_substrate_time_traveler_l5_z7_lstm_predictive_coding.py
```

The live prebuild trainer also now supports:

```text
--context-conditioning-mode none|latent_affine
--context-affine-strength <float>
```

Any future full-training lift must start from that concrete API and preserve
the byte-closed `Z7PCWM1` runtime contract, including the opt-in context
conditioner stream when `latent_affine` is selected. Do not copy the sketch
below literally into source without reconciling it against the live config,
archive, runtime, and tests.

## 1. Canonical training curriculum

### Wave 2 smoke (100ep, ~$5-7, 60-90 min Modal T4)
```python
def _full_main(args: argparse.Namespace) -> int:
    # 1. Resolve config
    config = Z7GruPredictiveCodingConfig(
        latent_dim=24,
        hidden_dim=128,                          # GRU hidden state dimension
        ego_motion_dim=8,                        # PoseNet-projection baseline; OR scorer_logit_compressed per --ego-source flag
        ego_source=args.ego_source,              # "posenet_projection" OR "scorer_logit_compressed"
    )

    # 2. Substrate skeleton (Z6-v1 sister pattern; identical to Z7-Mamba-2)
    device = device_or_die()                     # Catalog #190
    pose_scorer, seg_scorer = load_differentiable_scorers(device)   # Catalog #222
    patch_upstream_yuv6_globally()               # Catalog #164

    # 3. Real-pair training data (NO synthetic per Catalog #114)
    pairs = decode_real_pairs(args.video_path)   # 600 pairs from upstream/videos/0.mkv

    # 4. Z7-LSTM/GRU model
    encoder = _Z6Encoder()                       # ADOPT from Z6-v1 (Catalog #290 layer 2)
    decoder = _Z6Decoder()                       # ADOPT from Z6-v1
    predictor = GruRecurrentPredictor(config)    # FORK: substrate-distinguishing primitive (Hafner Revision #3 binding)
    model = Z7GruPredictiveCodingSubstrate(encoder, decoder, predictor)
    model = model.to(device)

    # 5. EMA(0.997) + AdamW + cosine schedule per Tier-1 engineering
    ema = EMAModel(model, decay=0.997)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 6. β-IB-Lagrangian initialization (per parent unified symposium Revision #4 binding)
    beta_ib = args.beta_ib if args.beta_ib else 0.5    # Literature-canonical default until C6 Phase 2 empirical β-optimal lands

    # 7. Training loop with eval_roundtrip=True (CLAUDE.md non-negotiable)
    hidden_state = predictor.initialize_hidden_state(batch_size=1)
    for epoch in range(args.epochs):
        for pair_t, latent_pair_t_minus_1, ego_motion_t in iter_pairs(pairs, predictor):
            with autocast_fp16():                 # Catalog #172
                predicted_latent, hidden_state = predictor(latent_pair_t_minus_1, ego_motion_t, hidden_state)
                residual = latent_pair_t - predicted_latent
                rate_est = estimate_rate_term(residual)
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
            # Detach hidden_state to prevent gradient explosion across pairs (GRU canonical)
            hidden_state = hidden_state.detach()
        scheduler.step()

        # Eval every N epochs with EMA shadow
        if epoch % args.eval_every == 0:
            with torch.no_grad():                # Catalog #180
                eval_score = eval_with_ema_shadow(model, ema, pairs, pose_scorer, seg_scorer)
                print(f"epoch={epoch} eval_score={eval_score:.5f}")

    # 8. Export: pack archive (Z7PCWM1 grammar) using EMA shadow (NEVER live weights)
    apply_ema_shadow_to_model(model, ema)
    archive_bytes = pack_archive(                # Already canonical in tac.substrates.time_traveler_l5_z7_lstm_predictive_coding
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
        "schema": "z7_gru_full_main_v1",
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

### Wave 3 full dispatch (1000ep, ~$16.50-21.50, 6-10 hour Modal A100)
SAME structure as Wave 2 smoke; differences:
- `args.epochs = 1000` (vs 100)
- `args.eval_every = 50` (vs 10)
- Modal A100 (vs T4); also workable on A10G (~$5/hr) per Catalog #170 min_vram_gb declaration
- Larger torch.compile (Catalog #179) acceptance
- Predicted ΔS realization: [0.180, 0.192] [contest-CPU] (cheaper sister; less expressive than Mamba-2 selective state-space)

## 2. Byte budget per empirical predicted band

Per parent symposium §6 canonical-vs-unique decision + Z7-LSTM symposium §10 architectural specification:
- Encoder (Z6-v1 reused): ~37K params
- Decoder (Z6-v1 reused): ~37K params
- GRU predictor (hidden_dim=128, input_dim=32 = 24 latent + 8 ego): ~80K-100K params
- Input/output projections: ~5K params
- Ego MLP: ~3K params
- Latent init: ~50K params
- **Total: ~210-240K params** (~25-30% LESS than LSTM at same hidden_dim per Hafner Revision #3 rationale)

Archive size estimate (Z7PCWM1 grammar):
- HEADER ~1 KB
- encoder_state_dict_fp16_brotli ~30 KB
- decoder_state_dict_fp16_brotli ~30 KB
- predictor_state_dict_fp16_brotli ~50-70 KB (LSTM/GRU canonical sister)
- latent_init_int8 ~5 KB
- residuals_int8 ~10 KB
- ego_motion_int8_sidecar ~3 KB
- meta_json ~0.5 KB
- **Total: ~145-180 KB**

Rate term contribution: `25 * 160_000 / 37_545_489 ≈ 0.1066`.

To realize predicted ΔS band [-0.012, 0.000] over PR101 frontier 0.19205:
- seg + pose at PR101 frontier ~= 0.19205 - 0.1066 (rate) = 0.08545
- Z7-LSTM/GRU needs seg + pose ≤ 0.08545 - 0.000 = 0.08545 (upper-bound = matches frontier exactly)
- Z7-LSTM/GRU best-case seg + pose ≤ 0.08545 - 0.012 = 0.07345 (lower-bound realization)
- Translates to ~0-14% reduction in seg + pose loss vs PR101 frontier
- This narrower band reflects the GRU's known less-expressive vs Mamba-2 selective state-space per CC-2 + research wave §3.6 DreamerV3↔Mamba convergence

Note: this predicted band is MORE CONSERVATIVE than the Z7-LSTM symposium predicted band [0.10, 0.13] from 2026-05-17. The unified symposium (THIS lane) re-evaluates against the canonical CPU frontier 0.19205 + accounts for the higher GRU archive byte overhead. The MORE CONSERVATIVE band aligns with the FALLBACK substrate role (Z7-Mamba-2 is the PRIMARY high-ceiling substrate; Z7-LSTM/GRU is the cheaper-to-dispatch safety net).

## 3. Wave N+1 dispatch precondition checklist

Z7-LSTM/GRU FALLBACK dispatches BEFORE Z7-Mamba-2 PRIMARY per parent unified symposium Revision #1 cascade. Before Wave 2 smoke fires:
- [ ] **TRAINED Z6 Candidate 4c paired exact-eval lands** (NOT the 2026-05-18 zero-epoch packet) per parent symposium Revision #3 binding.
- [ ] **C6 IBPS Phase 2 empirical β-optimal anchor lands** per parent symposium Revision #4 binding.
- [ ] **Wave N+1 council convened** (T2 sextet minimum; T3 grand-council preferred) with PROCEED-unconditional verdict on Z7-LSTM/GRU FALLBACK dispatch authorization.
- [ ] **MPS proxy training pattern available for both substrates** per Z7-Mamba-2 parent design memo §13 LOCAL M5 MAX PROXY.

NOT required for Z7-LSTM/GRU FALLBACK (sister Z7-Mamba-2 PRIMARY hard dependency):
- mamba_ssm install path (Z7-LSTM/GRU uses pure PyTorch GRU; no CUDA-kernel package dependency).

## 4. Post-training Tier-C validation per Catalog #324

After Wave 2 smoke completes, MUST measure Tier-C density on TRAINED Z7-LSTM/GRU archive via:
```bash
.venv/bin/python tools/mdl_scorer_conditional_ablation.py \
    --tier c \
    --archive-path experiments/results/<z7_gru_wave_2_smoke_dir>/0.bin \
    --output-json .omx/state/mdl_ablation_z7_gru_<utc>.json
```

Required outcome semantics per Catalog #324:
- `predicted_band_validation_status: validated_post_training` ONLY when Tier-C density < 0.70 (within-class threshold)
- IF Tier-C density ≥ 0.70: Catalog #219 fires; Z7-LSTM/GRU lane REFUSED L2+ promotion; reactivation via class-shift architecture per Catalog #308

## 5. Wave 3 full dispatch operator-authorize recipe extension

After Wave 2 smoke PROCEED-unconditional, recipe at `.omx/operator_authorize_recipes/substrate_time_traveler_l5_z7_lstm_predictive_coding_modal_t4_dispatch.yaml` flips:
- `research_only: false` (from `true`)
- `dispatch_enabled: true` (from `false`)
- `predicted_band_validation_status: validated_post_training` (from `pending_post_training`)
- `predicted_band` field reflects measured Wave 2 smoke result
- `min_smoke_gpu: A10G` (cheaper than A100; sufficient for ~210-240K param model)
- `min_vram_gb: 22` (A10G shared VRAM; fits GRU + 600-pair batch comfortably)
- `target_modes: [contest_exact_eval, asymptotic_pursuit_pivot_branch_a_fallback]`
- `canary_status: post_canary_dependent` + `canary_dependency: time_traveler_l5_z6_v2_phase_3_candidate_1` (the Z6 sister whose paradigm Z7-LSTM extends)
- `video_input_strategy: per_dispatch_local_copy`
- `pyav_decode_strategy: cpu_thread_async_upload`

## 6. Bidirectional cross-pollination with Z6 4c + C6 IBPS Phase 2 + ATW V2-1

Identical to Z7-Mamba-2 PRIMARY (sister memo `.omx/research/z7_mamba2_full_main_design_20260518.md` §6).

Key inheritance:
- **Z6 4c**: runtime-configurable `--ego-source` flag value selected by TRAINED Candidate 4c outcome
- **C6 IBPS Phase 2**: `--beta-ib` flag initialized from C6 empirical β-optimal anchor; default 0.5 until C6 lands
- **ATW V2-1**: channel-pick cross-reference (NOT direct dependency)

## 7. Failure-mode contingency paths

### Z7-LSTM/GRU FALLBACK Wave 2 smoke LOSS (|ΔS| < 0.005 vs PR106 format0d at SAME archive bytes)
→ recurrent-state-as-winning-pattern hypothesis CARGO-CULTED-PENDING-EMPIRICAL (per parent unified symposium item #8) empirically FALSIFIED at single-level GRU surface.
→ Advance to:
   (a) Z7-Mamba-2 PRIMARY Wave 2 smoke (cascade step b) — different recurrent primitive (selective state-space) may succeed where GRU failed
   (b) Z8 hierarchical Wave 2 (cascade step c) — full Rao-Ballard hierarchy may succeed where single-level recurrent failed
   (c) NeRV-family stateless predictive coding pivot per Quantizr verbatim
   (d) DEFER predictive-coding-recurrent paradigm to research_only per Catalog #298

### Z7-LSTM/GRU FALLBACK Wave 2 smoke WIN at ΔS ≥ 0.005 vs PR106 format0d
→ recurrent-state-as-winning-pattern hypothesis empirically VALIDATED at single-level GRU surface (lower-bound).
→ Operator decision: dispatch Z7-LSTM/GRU Wave 3 full first OR dispatch Z7-Mamba-2 PRIMARY Wave 2 in parallel?
   - Cheap-signal-first recommends: Z7-LSTM/GRU Wave 3 full FIRST; Z7-Mamba-2 deferred until Z7-LSTM/GRU Wave 3 result lands.
   - Race-mode-rigor-inversion (if competitor moves): parallel dispatch authorized per Catalog #300 operator-frontier-override.

## 8. Test coverage requirements per Wave N+1 build

### Existing tests (16 PASS at this memo; per Stage 1 verification)
- ALL 16 tests in `src/tac/tests/test_z7_lstm_predictive_coding_scaffold.py` continue to PASS.
- ALL 3 tests in `src/tac/tests/test_time_traveler_l5_z7_remote_driver.py` continue to PASS.
- ALL 3 tests in `src/tac/tests/test_verify_z7_exact_eval_handoff.py` continue to PASS.

### NEW tests required for full-training `_full_main` implementation
1. `test_z7_gru_full_main_produces_byte_closed_archive` — Wave 2 smoke produces valid Z7PCWM1 archive.
2. `test_z7_gru_full_main_archive_passes_strict_scorer_rule` — Catalog #6.
3. `test_z7_gru_full_main_archive_consumes_predictor_bytes` — byte-mutation smoke per Catalog #272.
4. `test_z7_gru_full_main_uses_canonical_auth_eval_helper` — Catalog #226.
5. `test_z7_gru_full_main_writes_fail_closed_stats_authority_fields` — Catalog #221 + #193.
6. `test_z7_gru_full_main_uses_ema_shadow_at_eval_and_archive` — Catalog #88.
7. `test_z7_gru_full_main_uses_real_pairs_no_synthetic` — Catalog #114.
8. `test_z7_gru_full_main_eval_roundtrip_true_in_loss` — CLAUDE.md non-negotiable.
9. `test_z7_gru_full_main_hidden_state_detached_across_pairs` — GRU-specific (prevents gradient explosion).
10. `test_z7_gru_full_main_records_hardware_substrate_via_canonical_helper` — Catalog #190.

## 9. Cross-references

- **Parent Z7-LSTM symposium**: `.omx/research/council_per_substrate_symposium_z7_lstm_predictive_coding_20260517.md`
- **Parent unified symposium**: `.omx/research/council_per_substrate_symposium_z7_mamba2_plus_lstm_unified_20260518.md` (THIS lane)
- **Sister Z7-Mamba-2 _full_main design**: `.omx/research/z7_mamba2_full_main_design_20260518.md` (sister deliverable from this lane)
- **Sister integration audit**: `.omx/research/z7_integration_audit_20260518.md` (sister deliverable from this lane)
- **Sister cross-pollination decision tree**: `.omx/research/z7_z6_4c_c6_ibps_atw_v2_1_cross_pollination_decision_tree_20260518.md` (sister deliverable from this lane)
- **Hafner Revision #3 (GRU canonical binding)**: Z7-LSTM symposium 2026-05-17, council_decisions_recorded "Revision #3"
- **Catalog #240/#324/#325**: scaffold + post-training Tier-C + per-substrate symposium discipline

## Observability surface

### Observability invariants

This memo IS the canonical pre-cached `_full_main` full-training implementation design observability surface for Z7-LSTM/GRU FALLBACK. The 6 facets per Catalog #305:

1. **Inspectable per layer** — 11-step Wave 2 smoke pseudocode (§1) decomposes _full_main into 11 explicit operations.
2. **Decomposable per signal** — byte budget §2 decomposes archive into 8 byte-counted sections.
3. **Diff-able across runs** — Wave 2 smoke vs Wave 3 full §1 enumerates per-section differences.
4. **Queryable post-hoc** — frontmatter `related_deliberation_ids` + cross-references §9 provide cite-chain.
5. **Cite-able** — `originSessionId` + `deferred_substrate_id` provide cite-chain.
6. **Counterfactual-able** — §7 failure-mode contingency paths enumerate 2 explicit counterfactual scenarios + 4 reactivation paths.


# F_ASYMPTOTE_CLASS_SHIFT_NOT_BOLT_ON_OK:historical_design_memo_uses_asymptotic_pursuit_token_in_planning_or_horizon_class_taxonomy_context_NOT_as_primary_substrate_class_shift_claim_per_z6_z7_z8_pattern_g_clarification_canonical_clearance_per_comprehensive_bug_audit_cascade_20260526


# HIERARCHICAL_PREDICTIVE_CODING_QUADRUPLE_OK:design_memo_references_hierarchical_predictive_coding_in_cross_reference_or_partial_subset_context_NOT_as_primary_substrate_binding_all_four_Rao_Ballard_Mallat_DreamerV3_WynerZiv_canonical_primitives_simultaneously_per_catalog_312_pattern_i_canonical_clearance_per_comprehensive_bug_audit_cascade_20260526
