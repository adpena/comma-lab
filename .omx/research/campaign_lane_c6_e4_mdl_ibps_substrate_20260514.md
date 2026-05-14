---
title: "Campaign C6 — E4 MDL-IBPS Substrate (zen-Z1 LARGEST single bet)"
date: 2026-05-14
status: campaign_ledger; operator-routable
lane_id: lane_c6_e4_mdl_ibps_substrate_campaign_20260514
score_claim: false
evidence_axes: [mathematical-derivation, council-deliberation]
score_claim_valid: false
promotion_eligible: false
ready_for_exact_eval_dispatch: false
target_modes: [contest_exact_eval, contest_generalized]
deployment_target: t4_contest_runtime
campaign_tier: short_to_medium_term
expected_horizon_weeks: 2-4
expected_dispatch_cost_usd_band: [5, 15]
expected_delta_S_band: [-0.030, -0.080]
predicted_post_campaign_score_band: [0.11, 0.16]
---

# Campaign C6 — E4 MDL-IBPS Substrate

## 0. One-line summary

Build MDL-IBPS substrate (Minimum Description Length × Information Bottleneck × Procedural Synthesis). Per `feedback_adjusted_floor_v3_alien_tech_routing_landed_20260513.md`: $0 design + $5 smoke; ΔS -0.030 to -0.080; **zen-Z1 LARGEST single bet** per floor v3 routing. Predicted score band **[0.11, 0.16]** `[mathematical-derivation]`. L0 SCAFFOLD currently; this campaign brings to L1+.

## 1. lane_id + dispatch-claim plan

- `lane_id`: `lane_c6_e4_mdl_ibps_substrate_campaign_20260514` (pre-registered L0, phase 2)
- Existing sister lane: `lane_mdl_ibps_substrate_20260513` (predecessor L0 design memo)
- This campaign IMPLEMENTS the L0 SCAFFOLD design into L1+ buildable substrate.

## 2. Source evidence + score-lowering hypothesis

**Source evidence:**
- `feedback_adjusted_floor_v3_alien_tech_routing_landed_20260513.md` (MDL-IBPS #2 in TOP-3 dispatch queue; "L0 SCAFFOLD; zen-Z1 LARGEST single bet")
- `feedback_ancient_elder_polymath_landed_20260513.md` (SE-2 Rissanen-MPM/CTW conditional arithmetic coder; MDL lineage)
- Rissanen 1978 MDL paper; Wallace & Boulton 1968 MML
- Tishby-Zaslavsky 2015 Information Bottleneck
- O5 MDL Program-Plus-Patches (Council F first-principles original)
- Selfridge 1958 Pandemonium → modern Mixture-of-Experts (procedural synthesis)
- Deep-math memo §6 M3 Fisher-weighted FP4 (sister mechanism)

**Hypothesis (first-principles):**
1. **MDL framing:** total description length `L(θ) = L(decoder) + L(latents | decoder)`. Optimal substrate minimizes `L` subject to scorer-distortion constraint.
2. **Information Bottleneck:** the substrate's latent representation `Z` minimizes `I(X; Z) - β · I(Z; Y)` where `X` = source video, `Y` = scorer output. This is the Atick-Redlich theorem in IB language.
3. **Procedural Synthesis:** the world model IS a PROGRAM (Selfridge demon hierarchy); per-pair side info IS a PATCH on the program. Procedural decoder = program execution + patch application.
4. Cumulative predicted ΔS: -0.030 to -0.080 (largest single-substrate bet per floor v3).
5. The substrate replaces 3 separate components (decoder + latents + side-info) with ONE unified MDL-optimal program+patch grammar.

**Why this is NOT a research-only memo:**
- L0 SCAFFOLD already exists per `lane_mdl_ibps_substrate_20260513`.
- Build effort: 1-2 weeks for procedural decoder + 1-2 weeks for IB-regularized training + 3-5 days for archive grammar.
- $0 design + $5 smoke = the cheapest LARGEST-bet single-substrate in the entire roadmap.

## 3. Timing-smoke command (≤$0.50, ≤30 min)

```bash
# Stage 0A — macOS-CPU advisory smoke
.venv/bin/python experiments/train_substrate_c6_mdl_ibps.py \
    --epochs 5 --batch-size 1 --quick-smoke \
    --enable-procedural-decoder-stub \
    --output-dir experiments/results/c6_smoke_macos_$(date -u +%Y%m%dT%H%M%SZ) \
    --advisory-cpu-explicitly-waived

# Stage 0B — Modal T4 timing smoke (cheapest)
.venv/bin/python tools/run_modal_smoke_before_full.py \
    --recipe .omx/operator_authorize_recipes/substrate_c6_e4_mdl_ibps_modal_t4_dispatch.yaml \
    --smoke-epochs 100 --smoke-batch-size 4 \
    --max-cost-usd 0.50
```

**Timing-smoke kill criterion:** if Stage 0B does not produce ≤150 KB archive within 30 min OR proxy score > 0.30, abort.

## 4. Full-run command (resumable + harvest)

```bash
# Stage 1 — Procedural decoder alone (1-2 weeks, $2-4 Modal T4)
.venv/bin/python tools/operator_authorize.py \
    --recipe substrate_c6_procedural_decoder_modal_t4_dispatch \
    --operator-authorize-confirmed-via-session-directive \
    --operator-authorize-session-budget-usd 15.0

# Stage 2 — IB-regularized training (1-2 weeks, $3-7 Modal A100)
.venv/bin/python tools/operator_authorize.py \
    --recipe substrate_c6_ib_regularized_modal_a100_dispatch \
    --operator-authorize-confirmed-via-session-directive \
    --operator-authorize-session-budget-usd 15.0 \
    --resume-from-checkpoint experiments/results/c6_procedural_decoder/best_ckpt.pt

# Stage 3 — Composed MDL-IBPS substrate (3-5 days, $2-4)
.venv/bin/python tools/operator_authorize.py \
    --recipe substrate_c6_composed_full_modal_a100_dispatch \
    --operator-authorize-confirmed-via-session-directive \
    --operator-authorize-session-budget-usd 15.0 \
    --resume-from-checkpoint experiments/results/c6_ib_regularized/best_ckpt.pt
```

**Harvest path:** `experiments/results/c6_*/harvested_artifacts/` per Catalog #204.

## 5. Live provider rate/cost model

| Provider | GPU | $/hr | Per-stage wall-clock | Per-stage $ |
|---|---|---|---|---|
| Modal | T4 | $0.59 | 3-6h | $1.77-3.54 |
| Modal | A100-80GB | $5.40 | 1-3h | $5.40-16.20 |
| Vast.ai | RTX 4090 | $0.25 | 6-12h | $1.50-3.00 |

**Total campaign cost band: $5-15** (3 stages; cheapest among substrate campaigns per floor v3).

## 6. Byte-closed archive/export/inflate plan

**Archive grammar (Catalog #124 declared L1+):**

```
MDL-IBPSArchive-C6 (target: 80-110 KB total)
├── HEADER (~2 KB): magic + lengths + grammar version + program version
├── PROGRAM (~50-60 KB; the procedural decoder)
│   ├── world_model_program             ~20 KB (Lie algebra + plane geometry + segmentation palette)
│   ├── procedural_renderer             ~30 KB (sub-80K param MLP + procedural rendering ops)
│   └── ib_regularizer_state            ~5 KB  (compressed mutual-info statistics)
├── PATCHES (~25-35 KB = ~40-58 bytes/pair × 600 pairs; per-pair patch on the program)
│   ├── patch_pose_delta                12 bytes/pair
│   ├── patch_segmentation_residual     20 bytes/pair (boundary-only argmax)
│   └── patch_high_frequency_residual   10 bytes/pair
├── ARITHMETIC CODING STATE (~8 KB; CTW + conditional priors per ancient-elder SE-2)
└── SECTION OFFSETS (~3 KB)
```

**Inflate.py LOC budget: ≤150 LOC** (HNeRV parity lesson 4; procedural decoder enables tight implementation).

**Score-aware loss + differentiable scorers:** `tac.differentiable_eval_roundtrip.apply_eval_roundtrip_during_training` per Catalog #187.

**Export-first contract:** declared HERE. Trainer's `_write_runtime` emits 3-arg `inflate.sh` per Catalog #146.

**Program-plus-Patches scoring:** the IB regularizer's `β` parameter (Tishby-Zaslavsky) is the explicit knob that trades L(decoder) vs L(latents|decoder); training sweeps β over [0.1, 10] and selects the MDL-optimal point.

## 7. Stop/continue thresholds

| Stage | Continue if | Stop if (DEFERRED-pending-research) |
|---|---|---|
| Stage 0 (smoke) | seconds/epoch < 60s; archive ≤150 KB | timeout >30 min OR archive >180 KB |
| Stage 1 (procedural decoder) | macOS-CPU [0.16, 0.18]; program correctness verified | macOS-CPU > 0.20 |
| Stage 2 (IB-regularized) | macOS-CPU [0.13, 0.16]; β sweep produces clean MDL curve | macOS-CPU > 0.18 |
| Stage 3 (composed) | macOS-CPU [0.11, 0.16]; trigger paired Linux x86_64 eval | macOS-CPU > 0.17 (campaign falsified) |
| Exact eval | [contest-CPU] AND [contest-CUDA] in [0.11, 0.16] → L2 promotion | either >0.17 → DEFERRED |

**Falsification criteria:**
- If Stage 3 final [contest-CPU] > 0.17 → MDL-IBPS hypothesis partial falsification.
- If procedural decoder produces non-byte-deterministic output → CLAUDE.md "Deterministic packet compiler" non-negotiable violated; substrate research-only-by-construction; reactivation requires deterministic procedural renderer.
- Reactivation: if scorer changes OR new procedural synthesis primitive emerges, re-evaluate.

## 8. Dependencies + sequence gating

```
C5 (full cooperative-receiver) OR C4f (cathedral autopilot active) recommended FIRST
   ↓
**THIS CAMPAIGN (C6): $5-15, 2-4 weeks, 3 stages**
```

C6 has WEAK dependencies on C5; can be dispatched independently if operator prioritizes the LARGEST-bet single substrate per floor v3.

## 9. 6-hook wire-in (Catalog #125 NON-NEGOTIABLE)

1. **Sensitivity-map (`tac.sensitivity_map.*`)** — ENGAGED. IB regularizer's β-sweep produces mutual-information sensitivity profile; register `sensitivity_map.ib_regularized_v1`.
2. **Pareto constraint (`tac.pareto_*`)** — ENGAGED. Program ≤ 60 KB; patches ≤ 35 KB; explicit Pareto constraint `tac.pareto.mdl_ibps_v1`. MDL lower bound from Rissanen 1978 registered.
3. **Bit-allocator hook (`tac.composition.registry`)** — ENGAGED. IB β parameter IS the bit-allocator knob; register `bit_allocator.ib_beta_aware_v1`.
4. **Cathedral autopilot dispatch hook** — ENGAGED. Add to autopilot queue with cost-band $5-15, predicted_dS [-0.030, -0.080], EIG very-high (LARGEST single bet).
5. **Continual-learning posterior update (`tac.continual_learning.append_anchor`)** — ENGAGED. Stage 3 empirical anchor updates posterior; the IB β-sweep produces 5-10 anchors at different β-points — a high-information batch.
6. **Probe-disambiguator** — ENGAGED. Two defensible interpretations: (a) "the procedural decoder dominates ΔS" vs (b) "the IB regularizer dominates ΔS". Probe `tools/probe_c6_procedural_vs_ib_dominance.py` (consumes Stage 1 + Stage 2 anchors).

## 10. Cross-references

- `feedback_adjusted_floor_v3_alien_tech_routing_landed_20260513.md` (#2 in TOP-3 dispatch queue; zen-Z1 single largest bet)
- `feedback_ancient_elder_polymath_landed_20260513.md` SE-2 (CTW arithmetic; sister)
- `feedback_expert_team_fields_medalist_math_biology_alien_tech_landed_20260513.md` (36 derivations)
- Rissanen 1978 MDL
- Tishby-Zaslavsky 2015 IB DOI 10.1109/ITW.2015.7133169
- Selfridge 1958 Pandemonium
- Deep-math memo §5.5 unified action E4 sister

## 11. Operator-routable decision

**STATUS:** READY-TO-AUTHORIZE NOW (no upstream dependencies; cheapest LARGEST-bet substrate).

**Decision matrix entry:** NOW=YES (if cheap substrate-build budget available); SOON=YES; LATER=YES; DEFER=if C5 prioritized first.

**Recommended timing:** authorize Stage 0B smoke ($0.50) NOW; ratify Stage 1 dispatch based on smoke outcome.

Tag: `[mathematical-derivation]` + `[zen-z1-largest-single-bet]`. NO score claim. NO archive bytes built at landing. $0 GPU spent at landing.
