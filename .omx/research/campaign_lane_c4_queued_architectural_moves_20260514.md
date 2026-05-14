---
title: "Campaign C4 — Queued Architectural Moves (C4a-C4g consolidated)"
date: 2026-05-14
status: campaign_ledger; operator-routable
lane_id: lane_c4_queued_architectural_moves_campaign_20260514
score_claim: false
evidence_axes: [council-deliberation, mathematical-derivation, literature-prediction]
score_claim_valid: false
promotion_eligible: false
ready_for_exact_eval_dispatch: false
target_modes: [contest_exact_eval, contest_generalized]
deployment_target: t4_contest_runtime
campaign_tier: short_to_medium_term  # 1-3 months per sub-move; consolidated 3-6 months
expected_horizon_weeks: 12-24
expected_dispatch_cost_usd_band: [50, 150]  # consolidated; per-sub-move $3-40
expected_delta_S_band: [-0.020, -0.060]  # consolidated
predicted_post_campaign_score_band: [0.13, 0.17]  # cumulative on top of staircase Z3
---

# Campaign C4 — Queued Architectural Moves (Consolidated)

## 0. One-line summary

Consolidate 7 already-task-listed architectural moves (SC++ / T10 IB / PR95 Phase 2-4 / NeRV-family / L2 Hinton-distilled / Cathedral autopilot / Magic codec) into 7 sub-campaigns under one lane. Predicted post-C4 cumulative score band **[0.13, 0.17]** `[council-deliberation]`. Cost $50-150 consolidated. Operator-routable PER SUB-MOVE.

## 1. lane_id + dispatch-claim plan

- `lane_id`: `lane_c4_queued_architectural_moves_campaign_20260514` (umbrella; pre-registered L0)
- Each sub-move has its own dispatch-claim row in `.omx/state/active_lane_dispatch_claims.md`
- Existing pre-registered sister lanes (consume verbatim):
  - `lane_sc_plus_plus_kl_distill` (C4a) — already L1
  - `lane_t10_ib_lagrangian_aux_scorer` (C4b) — already L1
  - PR95 Phase 2-4 — sister to `lane_pr95_phase_2_continuation_20260514` (existing canvas)
  - NeRV-family — Catalog #124-honoring sister lanes (BlockNeRV / FFNeRV / DSNeRV / HiNeRV existing)
  - `lane_hinton_distilled_scorer_surrogate` (C4e) — already pre-registered
  - `lane_cathedral_autopilot_activation` (C4f) — already pre-registered
  - `lane_magic_codec_dense_streams` (C4g) — sister to `lane_magic_codec_dense*` existing

## 2. Source evidence + score-lowering hypothesis

**Source evidence:**
- Task tracker entries #606/#607/#608/#522/#523/#524/#525/#526 (operator-task-listed but never campaign-converted)
- `feedback_adjusted_floor_v3_alien_tech_routing_landed_20260513.md` (alien-tech routing for D-paths)
- `feedback_orphan_anchor_backfill_landed_20260513.md` (CPU-CUDA gap inversion; PR106 family CPU-worse)
- Council F first-principles floor 0.10±0.03
- CLAUDE.md "SegNet vs PoseNet importance — operating-point dependent" (pose-marginal 2.71× regime at PR106 frontier)

**Hypothesis (per sub-move, see §3-§7 below):**
Each of C4a-C4g attacks a different axis of the contest score; together they extend PR101 0.193 baseline toward sub-0.155 via 7 small-to-medium bolt-ons.

## 3. C4a — SC++ Stage 1 ($3, 1-2 weeks)

**lane_id**: `lane_sc_plus_plus_kl_distill` (existing L1)

**Score-lowering hypothesis:** SegMap class-targets self-compression Stage 1 reduces mask channel cost by 15-25% per van den Oord codebook persistence + Selfcomp original 0.38 archive analysis. Predicted ΔS: `-0.005 to -0.012`.

**Timing-smoke command:**
```bash
.venv/bin/python tools/run_modal_smoke_before_full.py \
    --recipe .omx/operator_authorize_recipes/substrate_sc_plus_plus_stage1_modal_t4_dispatch.yaml \
    --smoke-epochs 100 --smoke-batch-size 8 \
    --max-cost-usd 0.30
```

**Full-run command:**
```bash
.venv/bin/python tools/operator_authorize.py \
    --recipe substrate_sc_plus_plus_stage1_full_modal_t4_dispatch \
    --operator-authorize-confirmed-via-session-directive \
    --operator-authorize-session-budget-usd 10.0
```

**Provider/cost:** Modal T4 $0.59/hr × 4-5h = $2.50-3.00. Total ≤ $3 + smoke $0.30.

**Stop/continue:** continue if [contest-CPU] ≤ 0.190 (-0.003 improvement); stop if > 0.195.

**Byte-closed plan:** SC++ archive grammar extends Quantizr block-FP self-compression; inflate.py ≤120 LOC; export-first per Catalog #124 (lane already L1, declared).

## 4. C4b — T10 IB Lagrangian Aux Scorer ($40, 3-4 weeks)

**lane_id**: `lane_t10_ib_lagrangian_aux_scorer` (existing L1)

**Score-lowering hypothesis:** Information-Bottleneck Lagrangian regularizes the score-aware loss; auxiliary scorer head Hinton-distills the SegNet+PoseNet on-the-fly. Reduces both component bands proportionally per Tishby-Zaslavsky 2015 + Hinton 2014 distillation. Predicted ΔS: `-0.010 to -0.025`.

**Timing-smoke command:**
```bash
.venv/bin/python tools/run_modal_smoke_before_full.py \
    --recipe .omx/operator_authorize_recipes/substrate_t10_ib_lagrangian_aux_scorer_modal_a100_dispatch.yaml \
    --smoke-epochs 100 --smoke-batch-size 4 \
    --max-cost-usd 0.50
```

**Full-run command:**
```bash
.venv/bin/python tools/operator_authorize.py \
    --recipe substrate_t10_ib_lagrangian_aux_scorer_full_modal_a100_dispatch \
    --operator-authorize-confirmed-via-session-directive \
    --operator-authorize-session-budget-usd 50.0
```

**Provider/cost:** Modal A100 $5.40/hr × 5-8h × 1-2 runs = $27-86. Budget cap $40 = single run.

**Stop/continue:** continue if [contest-CPU] ≤ 0.180 (-0.013 improvement); stop if > 0.190.

**Byte-closed plan:** aux scorer adds ~5-10 KB; total archive ≤ 190 KB; inflate.py uses TRAINED main renderer ONLY (aux scorer is training-time only).

## 5. C4c — PR95 Phase 2-4 Curriculum + Muon + Dual-RGB-Head (~$30-50, 4-6 weeks)

**lane_id**: sister to existing `lane_pr95_phase_2_continuation_*`

**Score-lowering hypothesis:** PR95's 8-stage curriculum (anchor → finetune → joint → QAT → final + 3 more) extended; Muon optimizer (Schulz 1933 / 2024 paper) replaces Adam; dual-RGB-head trains two renderers in tandem. Predicted ΔS: `-0.008 to -0.020`.

**Timing-smoke command:**
```bash
.venv/bin/python tools/run_modal_smoke_before_full.py \
    --recipe .omx/operator_authorize_recipes/substrate_pr95_phase_2_4_curriculum_muon_dual_rgb_modal_a100_dispatch.yaml \
    --smoke-epochs 100 --smoke-batch-size 4 \
    --max-cost-usd 0.50
```

**Full-run command:**
```bash
.venv/bin/python tools/operator_authorize.py \
    --recipe substrate_pr95_phase_2_4_full_modal_a100_dispatch \
    --operator-authorize-confirmed-via-session-directive \
    --operator-authorize-session-budget-usd 60.0
```

**Provider/cost:** Modal A100 $5.40/hr × 4-6h × 3-4 stages = $65-130. Vast.ai 4090 alternative: $0.25/hr × 16-24h × 3-4 stages = $12-24.

**Stop/continue:** continue if [contest-CPU] ≤ 0.175; stop if > 0.188.

**Byte-closed plan:** PR95 archive grammar unchanged (already proven); 8-stage trainer extends existing `experiments/train_substrate_pr95_*` patterns.

## 6. C4d — NeRV-Family Expansion + Bolt-Ons (~$30-50, 3-5 weeks)

**lane_id**: sister to existing `lane_*nerv*` (BlockNeRV / FFNeRV / DSNeRV / HiNeRV / TCNeRV)

**Score-lowering hypothesis:** NeRV-family content-adaptive embeddings dominate coordinate MLPs at sub-100KB budget per the literature gradient (Cool-Chic / C3 / COIN++ comparison). Each NeRV variant exploits a different temporal-coherence prior; bolt-ons stack additively. Predicted ΔS: `-0.005 to -0.015` per variant; cumulative `-0.015 to -0.040`.

**Timing-smoke command (per variant):**
```bash
.venv/bin/python tools/run_modal_smoke_before_full.py \
    --recipe .omx/operator_authorize_recipes/substrate_${VARIANT}nerv_bolt_on_modal_a100_dispatch.yaml \
    --smoke-epochs 100 --smoke-batch-size 4 \
    --max-cost-usd 0.50
# VARIANT in {block, ff, ds, hi, tc, e}
```

**Full-run command (per variant):**
```bash
.venv/bin/python tools/operator_authorize.py \
    --recipe substrate_${VARIANT}nerv_bolt_on_full_modal_a100_dispatch \
    --operator-authorize-confirmed-via-session-directive \
    --operator-authorize-session-budget-usd 60.0
```

**Provider/cost:** Modal A100 $5.40/hr × 3-5h × 4-6 variants = $65-160. Cap consolidated at $50; prioritize FFNeRV + HiNeRV (predicted highest ΔS).

**Stop/continue (per variant):** continue if [contest-CPU] ≤ prior+0.002 cumulative; stop if > +0.005.

**Byte-closed plan:** each NeRV variant has its own L1+ archive grammar declaration per Catalog #124 (already declared in sister lanes); inflate.py per variant ≤200 LOC.

## 7. C4e — L2 Hinton-Distilled Scorer Surrogate + Saliency-Masked Residual (~$15, 2-3 weeks)

**lane_id**: `lane_hinton_distilled_scorer_surrogate` (existing pre-registered) + `lane_saliency_masked_residual_l2_encoder` (existing pre-registered)

**Score-lowering hypothesis:** L2 Hinton-distilled scorer (T=2.0 like Quantizr) provides gradient-reachable surrogate during training (eval-time scorer is `@torch.no_grad()` per CLAUDE.md "eval_roundtrip" non-negotiable). Saliency-masked residual exploits the per-pixel score gradient from §2.2 of deep-math memo. Predicted ΔS: `-0.005 to -0.012`.

**Timing-smoke command:**
```bash
.venv/bin/python tools/run_modal_smoke_before_full.py \
    --recipe .omx/operator_authorize_recipes/substrate_l2_hinton_distilled_saliency_residual_modal_t4_dispatch.yaml \
    --smoke-epochs 100 --smoke-batch-size 4 \
    --max-cost-usd 0.30
```

**Full-run command:**
```bash
.venv/bin/python tools/operator_authorize.py \
    --recipe substrate_l2_hinton_distilled_saliency_residual_full_modal_t4_dispatch \
    --operator-authorize-confirmed-via-session-directive \
    --operator-authorize-session-budget-usd 20.0
```

**Provider/cost:** Modal T4 $0.59/hr × 8-10h = $4.70-5.90 + ablations. Total ≤ $15.

**Stop/continue:** continue if [contest-CPU] ≤ 0.185; stop if > 0.192.

**Byte-closed plan:** Hinton-distilled scorer is TRAINING-TIME-ONLY; main renderer unchanged; saliency mask adds ~3-5 KB sidecar (boundary-only encoded per SABOR-style argmax).

## 8. C4f — Cathedral Autopilot Activation + Phase 2 Probes + Integration Audit v2 ($5-10, 2 weeks)

**lane_id**: `lane_cathedral_autopilot_activation`

**Score-lowering hypothesis:** ACTIVATE the autopilot loop end-to-end: rank candidates → fan-out dispatches → harvest results → reseed posterior. Per CLAUDE.md "Race-mode rigor inversion + parallel-dispatch first" non-negotiable. Predicted indirect ΔS: variable (-0.005 to -0.030 via efficient empirical orchestration that reduces wasted dispatches). Not a direct score-lower; an enabler.

**Timing-smoke command:**
```bash
.venv/bin/python tools/cathedral_autopilot_autonomous_loop.py \
    --max-concurrency 4 --max-total-cost-usd 5 \
    --candidate-source .omx/state/autopilot_candidate_queue_solver_stack_wire_in_20260513.jsonl \
    --output reports/cathedral_autopilot_smoke_$(date -u +%Y%m%dT%H%M%SZ).jsonl
```

**Full-run command:** the autopilot IS the full run; cathedral autopilot orchestrates other campaigns' dispatches.

**Provider/cost:** $5-10 consolidated; the autopilot dispatches OTHER campaigns within its budget cap.

**Stop/continue:** continue if autopilot produces ≥2 valid contest-CUDA anchors per $5 spent; stop if no anchors after $10 (orchestrator failure).

**Byte-closed plan:** autopilot doesn't ship bytes; it orchestrates dispatches that ship bytes via their own campaigns.

## 9. C4g — Magic Codec Dense Streams + xray Substrate Classifier + ≤$0.90 Dispatches ($5-15, 2-3 weeks)

**lane_id**: sister to existing `lane_magic_codec_dense*`

**Score-lowering hypothesis:** Magic-byte codec preamble allows MULTIPLE codec families to coexist in one archive; xray substrate classifier auto-detects which family per archive byte range. Predicted ΔS: `-0.003 to -0.010` via cross-paradigm composition cells (per `feedback_solver_stack_wire_in_sweep_landed_20260513` LPA1/WAV1 magic-byte registration).

**Timing-smoke command:**
```bash
.venv/bin/python tools/run_modal_smoke_before_full.py \
    --recipe .omx/operator_authorize_recipes/substrate_magic_codec_dense_xray_modal_t4_dispatch.yaml \
    --smoke-epochs 100 --smoke-batch-size 4 \
    --max-cost-usd 0.30
```

**Full-run command (cheap):**
```bash
.venv/bin/python tools/operator_authorize.py \
    --recipe substrate_magic_codec_dense_xray_full_modal_t4_dispatch \
    --operator-authorize-confirmed-via-session-directive \
    --operator-authorize-session-budget-usd 15.0
```

**Provider/cost:** Modal T4 $0.59/hr × ≤2h = $1.18 per ≤$0.90 dispatch × 5-10 dispatches = $5-12.

**Stop/continue:** continue if [contest-CPU] ≤ 0.188 cumulative; stop if > 0.193.

**Byte-closed plan:** magic-byte codec preamble extends `tac.packet_compiler.cooperative_receiver_grammars` (already has LPA1, WAV1 registration); inflate.py auto-dispatches per detected magic.

## 10. Consolidated cost band + sequence + harvest

**Total cost band:** $50-150 (consolidated across all 7 sub-moves).
**Total wall-clock band:** 12-24 weeks (parallelizable: C4a + C4e + C4g cheap-substrate moves can run in parallel; C4b + C4c + C4d require sequential dispatch + iteration).

**Recommended dispatch sequence (cheapest+highest-EV first per CLAUDE.md "Race-mode rigor inversion"):**
1. C4f (cathedral autopilot activation) — $5-10, 1-2 weeks; ENABLES other dispatches
2. C4g (magic codec dense + xray) — $5-15, 2-3 weeks
3. C4a (SC++ Stage 1) — $3, 1-2 weeks
4. C4e (L2 Hinton-distilled + saliency residual) — $15, 2-3 weeks
5. C4d (NeRV-family expansion) — $30-50, 3-5 weeks (prioritize FFNeRV + HiNeRV)
6. C4c (PR95 Phase 2-4 curriculum + Muon + dual-RGB-head) — $30-50, 4-6 weeks
7. C4b (T10 IB Lagrangian aux scorer) — $40, 3-4 weeks

**Harvest path:** each sub-move harvests to `experiments/results/<sub_lane_id>_*/harvested_artifacts/`; quarterly summary to `.omx/state/c4_consolidated_summary_ledger.md`.

## 11. 6-hook wire-in (Catalog #125 NON-NEGOTIABLE)

1. **Sensitivity-map (`tac.sensitivity_map.*`)** — ENGAGED. C4e (Hinton-distilled saliency) IS the sensitivity-map primitive; registers `sensitivity_map.hinton_distilled_saliency_v1`. C4d NeRV-family variants register `sensitivity_map.nerv_family_temporal_coherence_v${VARIANT}`.
2. **Pareto constraint (`tac.pareto_*`)** — ENGAGED. Each sub-move's archive grammar adds Pareto constraint; consolidated registers `tac.pareto.c4_consolidated_v1`.
3. **Bit-allocator hook (`tac.composition.registry`)** — ENGAGED. C4b T10 IB Lagrangian IS a bit-allocator (information-bottleneck regularizes per-component bytes). C4d NeRV variants each register their content-adaptive embedding bit-allocator.
4. **Cathedral autopilot dispatch hook** — ENGAGED. C4f IS the autopilot activation; the activation hook is the campaign deliverable.
5. **Continual-learning posterior update (`tac.continual_learning.append_anchor`)** — ENGAGED. Every sub-move's empirical anchor updates posterior via `posterior_update_locked` per Catalog #128.
6. **Probe-disambiguator** — ENGAGED. Cross-sub-move disambiguators planned: `tools/probe_c4_which_bolt_on_dominates.py` (consumes C4a-C4g anchors to rank by ΔS); `tools/probe_c4_nerv_variant_selector.py` (consumes C4d sub-anchors).

## 12. Stop/continue thresholds (campaign-level)

| Phase | Continue if | Stop if |
|---|---|---|
| Phase 1 (C4f + C4g + C4a, $13-28) | ≥2 sub-moves land ΔS ≤ -0.003 | all 3 sub-moves fail to move score |
| Phase 2 (C4e + C4d, $45-65) | cumulative [contest-CPU] ≤ 0.180 | cumulative > 0.187 |
| Phase 3 (C4c + C4b, $70-90) | cumulative [contest-CPU] ≤ 0.150 | cumulative > 0.175 |
| Exact eval consolidated | [contest-CPU] AND [contest-CUDA] ≤ 0.155 | either > 0.180 → DEFERRED-pending-research |

**Falsification criteria:** if Phase 3 cumulative > 0.180 → consolidation hypothesis falsified; sub-moves diverge into independent campaigns; reactivation requires individual sub-move audit.

## 13. Cross-references

- `feedback_adjusted_floor_v3_alien_tech_routing_landed_20260513.md` (D-paths queue)
- `feedback_orphan_anchor_backfill_landed_20260513.md` (CPU-CUDA gap inversion at PR106; pose-marginal regime)
- `feedback_solver_stack_wire_in_sweep_landed_20260513.md` (LPA1/WAV1 cooperative-receiver magic-bytes)
- Council F first-principles floor 0.10±0.03
- CLAUDE.md "SegNet vs PoseNet importance" (pose-marginal 2.71× regime applies to C4b/C4c/C4d)
- CLAUDE.md "Long-burn score-lowering campaign default" non-negotiable

## 14. Operator-routable decision

**STATUS:** READY-TO-AUTHORIZE PER SUB-MOVE. Operator authorizes each sub-move independently OR consolidated authorization at $50-150 band.

**Decision matrix entry:** NOW=C4f+C4g+C4a (cheap, parallelizable, $13-28); SOON=C4e+C4d (medium, $45-65); LATER=C4b+C4c ($70-90); DEFER=none individually.

**Recommended sequence:** activate cathedral autopilot (C4f) FIRST; let it orchestrate the others.

Tag: `[council-deliberation]` + `[consolidated-campaign]`. NO score claim. NO archive bytes built at landing. $0 GPU spent at landing.
