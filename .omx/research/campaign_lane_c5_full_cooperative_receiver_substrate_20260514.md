---
title: "Campaign C5 — Full Cooperative-Receiver Substrate (Atick-Redlich H(X|scorer))"
date: 2026-05-14
status: campaign_ledger; operator-routable
lane_id: lane_c5_full_cooperative_receiver_substrate_campaign_20260514
score_claim: false
evidence_axes: [mathematical-derivation, literature-prediction, council-deliberation]
score_claim_valid: false
promotion_eligible: false
ready_for_exact_eval_dispatch: false
target_modes: [contest_exact_eval, contest_generalized]
deployment_target: t4_contest_runtime
campaign_tier: medium_to_long_term
expected_horizon_weeks: 4-8
expected_dispatch_cost_usd_band: [30, 50]
expected_delta_S_band: [-0.025, -0.060]
predicted_post_campaign_score_band: [0.13, 0.17]
---

# Campaign C5 — Full Cooperative-Receiver Substrate

## 0. One-line summary

Build the FULL cooperative-receiver substrate (Atick-Redlich 1990 + Wyner-Ziv 1976 + Slepian-Wolf 1973) — frame-0 + frame-1 + pair-conditional combined. Predicted score band **[0.13, 0.17]** `[mathematical-derivation; first-principles-bound]`. Cost $30-50. Distinct from D4 (which is frame-0 only).

## 1. lane_id + dispatch-claim plan

- `lane_id`: `lane_c5_full_cooperative_receiver_substrate_campaign_20260514` (pre-registered L0, phase 3)
- Existing sister lanes:
  - `lane_d4_wyner_ziv_frame_0_substrate_20260514` (in flight; frame-0 only)
  - `lane_cooperative_receiver_primitive_20260513` (primitive, not substrate)
  - `lane_wyner_ziv_cooperative_receiver_substrate_20260513` (predecessor design memo)
- This campaign EXTENDS D4 to frame-1 + pair-conditional after D4 lands.
- Dispatch claim row per stage; multi-week TTL refresh.

## 2. Source evidence + score-lowering hypothesis

**Source evidence:**
- Atick & Redlich 1990 (*Neural Computation* 2:308-320) cooperative-receiver theorem: `H(source | scorer_state)`
- Wyner & Ziv 1976 (DOI 10.1109/TIT.1976.1055508) — source coding with side information at decoder
- Slepian & Wolf 1973 (DOI 10.1109/TIT.1973.1055037) — distributed source coding
- `feedback_expert_team_signal_processing_alien_tech_landed_20260513.md` (#1 candidate N3 Wyner-Ziv cooperative-receiver; predicted ΔS -0.05)
- `feedback_adjusted_floor_v3_alien_tech_routing_landed_20260513.md` (top-3 dispatch queue including full cooperative-receiver)
- `feedback_yucr_substrate_landed_20260514.md` (YUCR L1; cooperative-receiver primitive operationalized)
- D4 in-flight dispatch (frame-0 only) — empirical anchor expected within hours
- Deep-math memo §9 (`R_min = H(source | scorer_weights+architecture+preprocessing)`)

**Hypothesis (first-principles):**
1. The contest IS cooperative-receiver compression: the scorer is FIXED + KNOWN + PUBLIC. The optimal encoder compresses to `H(V_GT | S)` where `S = scorer(V_GT)`, NOT `H(V_GT)`.
2. D4 attacks frame-0 only (~1/600 of pair rate). The full substrate extends to frame-1 + pair-conditional:
   - Frame-0 (per D4): Wyner-Ziv reconstruction from decoder side information ~ ΔS `-0.025 to -0.045`
   - Frame-1 (extension): symmetric Wyner-Ziv with pair structure ~ ΔS `-0.005 to -0.010` additive
   - Pair-conditional (extension): Slepian-Wolf joint encoding of the pair given the scorer ~ ΔS `-0.005 to -0.015` multiplicative
3. Cumulative predicted ΔS: `-0.025 to -0.060` (sub-additive Amdahl composition).
4. Per the deep-math §9 lower bound, the substrate hits `H(X | scorer)` ≈ 5-15 KB per pair after conditional entropy estimation. Practical achievable: 30-50 bytes/pair × 600 pairs = 18-30 KB total per-pair channel.

**Why this is NOT a research-only memo:**
- D4 is in flight; the frame-0 substrate empirically validates the cooperative-receiver hypothesis WITHIN HOURS.
- Frame-1 + pair-conditional extensions reuse D4's substrate scaffolding with ~30% additional implementation effort.
- Existing primitives: `tac.cooperative_receiver_primitive`, `tac.differentiable_eval_roundtrip`, YUCR cost-map.

## 3. Timing-smoke command (≤$0.30, ≤30 min)

```bash
# Stage 0A — macOS-CPU advisory smoke (FREE; assumes D4 has landed)
.venv/bin/python experiments/train_substrate_c5_full_cooperative_receiver.py \
    --epochs 5 --batch-size 1 --quick-smoke \
    --enable-frame-0-only \
    --output-dir experiments/results/c5_smoke_macos_$(date -u +%Y%m%dT%H%M%SZ) \
    --advisory-cpu-explicitly-waived

# Stage 0B — Modal T4 timing smoke
.venv/bin/python tools/run_modal_smoke_before_full.py \
    --recipe .omx/operator_authorize_recipes/substrate_c5_full_cooperative_receiver_modal_t4_dispatch.yaml \
    --smoke-epochs 100 --smoke-batch-size 4 \
    --max-cost-usd 0.50
```

**Timing-smoke kill criterion:** if Stage 0B does not produce ≤120 KB archive within 30 min OR proxy score > 0.30, abort.

## 4. Full-run command (resumable + harvest)

```bash
# Stage 1 — Frame-1 extension on top of D4 (3-5 days, $10-15 Modal A100)
.venv/bin/python tools/operator_authorize.py \
    --recipe substrate_c5_frame_1_extension_modal_a100_dispatch \
    --operator-authorize-confirmed-via-session-directive \
    --operator-authorize-session-budget-usd 50.0 \
    --resume-from-checkpoint experiments/results/d4_frame_0/best_ckpt.pt

# Stage 2 — Pair-conditional joint encoder (5-7 days, $15-25 Modal A100)
.venv/bin/python tools/operator_authorize.py \
    --recipe substrate_c5_pair_conditional_full_modal_a100_dispatch \
    --operator-authorize-confirmed-via-session-directive \
    --operator-authorize-session-budget-usd 50.0 \
    --resume-from-checkpoint experiments/results/c5_frame_1/best_ckpt.pt

# Stage 3 — Composed full cooperative-receiver substrate (3-5 days, $5-10)
.venv/bin/python tools/operator_authorize.py \
    --recipe substrate_c5_composed_full_modal_a100_dispatch \
    --operator-authorize-confirmed-via-session-directive \
    --operator-authorize-session-budget-usd 50.0 \
    --resume-from-checkpoint experiments/results/c5_pair_conditional/best_ckpt.pt
```

**Harvest path:** `experiments/results/c5_*/harvested_artifacts/` per Catalog #204.

## 5. Live provider rate/cost model

| Provider | GPU | $/hr | Per-stage wall-clock | Per-stage $ |
|---|---|---|---|---|
| Modal | A100-80GB | $5.40 | 2-4h smoke; 3-7h full | $11-38 per stage |
| Modal | T4 | $0.59 | 4-8h | $2.36-4.72 per stage |
| Vast.ai | RTX 4090 | $0.25 | 8-16h | $2-4 per stage |

**Total campaign cost band: $30-50** (3 stages on Modal A100).

## 6. Byte-closed archive/export/inflate plan

**Archive grammar (Catalog #124 declared L1+):**

```
FullCooperativeReceiverArchive-C5 (target: 100-130 KB total)
├── HEADER (~2 KB): magic + lengths + grammar version
├── SCORER-CONDITIONAL DECODER (~50-60 KB, FP4 + Brotli)
│   ├── frame_0_wyner_ziv_decoder      ~20 KB
│   ├── frame_1_wyner_ziv_decoder      ~20 KB (sister of frame-0; shared weights via LoRA)
│   └── pair_conditional_joint_decoder ~15 KB
├── SCORER-AWARE SIDE INFO (~30-40 KB = ~50-67 bytes/pair × 600 pairs)
│   ├── frame_0_side_info              25 bytes/pair (Slepian-Wolf encoded)
│   ├── frame_1_side_info              25 bytes/pair
│   └── pair_residual                  17 bytes/pair (Wyner-Ziv conditional)
├── ARITHMETIC CODING STATE (~10 KB)
└── SECTION OFFSETS (~3 KB)
```

**Inflate.py LOC budget: ≤200 LOC** (HNeRV parity lesson 4); cooperative-receiver decoder reuses YUCR cost-map infrastructure.

**Score-aware loss + differentiable scorers:** `tac.differentiable_eval_roundtrip.apply_eval_roundtrip_during_training` per Catalog #187.

**Export-first contract:** declared HERE; trainer's `_write_runtime` emits 3-arg `inflate.sh` per Catalog #146.

## 7. Stop/continue thresholds

| Stage | Continue if | Stop if (DEFERRED-pending-research) |
|---|---|---|
| Stage 0 (smoke) | seconds/epoch < 60s; archive ≤130 KB | timeout >30 min OR archive >150 KB |
| Stage 1 (frame-1) | macOS-CPU [0.15, 0.17]; cumulative D4+C5 frame-1 | macOS-CPU > 0.18 |
| Stage 2 (pair-conditional) | macOS-CPU [0.13, 0.15] | macOS-CPU > 0.16 |
| Stage 3 (composed) | macOS-CPU [0.13, 0.17]; trigger paired Linux x86_64 eval | macOS-CPU > 0.17 (campaign falsified) |
| Exact eval | [contest-CPU] AND [contest-CUDA] in [0.13, 0.17] → L2 promotion; council review | either >0.18 → DEFERRED |

**Falsification criteria:**
- If Stage 3 final [contest-CPU] > 0.17 → cooperative-receiver hypothesis partial falsification at the full-pair scope.
- D4 frame-0 anchor MAY survive even if C5 fails (substrate-engineering scope smaller for frame-0).
- Reactivation: if scorer changes (Yousfi-side update), all bets reset.

## 8. Dependencies + sequence gating

```
D4 (Wyner-Ziv frame-0 substrate, in flight, $5-15) lands successfully
   ↓
**THIS CAMPAIGN (C5): $30-50, 4-8 weeks**
   ↓
C6 (E4 MDL-IBPS) or C7 (DARTS-SuperNet): TRIGGERS conditionally
```

C5 cannot proceed before D4 lands. If D4 fails (CUDA-CPU drift; archive >180 KB), escalate to grand council BEFORE C5.

## 9. 6-hook wire-in (Catalog #125 NON-NEGOTIABLE)

1. **Sensitivity-map (`tac.sensitivity_map.*`)** — ENGAGED. Frame-0 + frame-1 + pair-conditional sensitivity priors registered as `sensitivity_map.wyner_ziv_per_pair_v1`. The scorer-conditional residual is the cooperative-receiver-canonical sensitivity primitive (Atick-Redlich orthogonal-complement projector — sister of YUCR's `compute_cost_map`).
2. **Pareto constraint (`tac.pareto_*`)** — ENGAGED. Total archive ≤ 130 KB; per-pair side info ≤ 70 bytes; explicit Pareto constraint `tac.pareto.full_cooperative_receiver_v1`. Shannon-1959 vector R(D) lower bound from ancient-elder §16 + Slepian-Wolf 1973 lower bound from N3 alien-tech.
3. **Bit-allocator hook (`tac.composition.registry`)** — ENGAGED. Per-pair byte allocation between frame-0 / frame-1 / pair-residual Slepian-Wolf-optimal; register `bit_allocator.full_cooperative_receiver_v1`.
4. **Cathedral autopilot dispatch hook** — ENGAGED. Add to autopilot queue with cost-band $30-50, predicted_dS [-0.025, -0.060], EIG very-high (closes the cooperative-receiver substrate question that has been open since 2026-05-08).
5. **Continual-learning posterior update (`tac.continual_learning.append_anchor`)** — ENGAGED. Empirical anchors from Stages 1-3 + composition cells update posterior; expected to update zen-floor band [0.05, 0.08] reachability prior (per zen-floor Section 9 reactivation criteria).
6. **Probe-disambiguator** — ENGAGED. Two defensible interpretations: (a) "frame-1 + pair-conditional add value linearly on top of D4 frame-0" vs (b) "frame-1 + pair-conditional Amdahl-dominate frame-0; D4 was the leading edge". Probe `tools/probe_c5_amdahl_vs_linear_composition.py` (consumes D4 + C5 Stage 1 + C5 Stage 2 anchors).

## 10. Cross-references

- D4 in-flight: `lane_d4_wyner_ziv_frame_0_substrate_20260514` (frame-0 anchor)
- `feedback_yucr_substrate_landed_20260514.md` (YUCR L1 sister)
- `feedback_expert_team_signal_processing_alien_tech_landed_20260513.md` N3 (#1 candidate)
- `feedback_adjusted_floor_v3_alien_tech_routing_landed_20260513.md`
- Deep-math memo §3.5 (Wyner-Ziv frame-0); §9.4 (Atick-Redlich operationalization)
- Atick & Redlich 1990 DOI 10.1162/neco.1990.2.3.308
- Wyner & Ziv 1976 DOI 10.1109/TIT.1976.1055508
- Slepian & Wolf 1973 DOI 10.1109/TIT.1973.1055037

## 11. Operator-routable decision

**STATUS:** READY-TO-AUTHORIZE after D4 lands.

**Decision matrix entry:** NOW=NO (wait for D4); SOON=YES (post-D4-success); LATER=NO; DEFER=if D4 fails.

**Recommended sequence:** D4 landing → ratify C5 → smoke → 3 staged dispatches → consolidated exact eval.

Tag: `[mathematical-derivation]` + `[first-principles-bound]`. NO score claim at landing. NO archive bytes. $0 GPU spent at landing.
