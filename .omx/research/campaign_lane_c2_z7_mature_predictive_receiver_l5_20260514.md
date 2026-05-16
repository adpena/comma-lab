---
title: "Campaign C2 — Z7 Mature Predictive-Receiver L5 Substrate"
date: 2026-05-14
status: campaign_ledger; operator-decision-required
lane_id: lane_c2_z7_mature_predictive_receiver_l5_campaign_20260514
score_claim: false
evidence_axes: [time-traveler-prediction, mathematical-derivation, council-deliberation]
score_claim_valid: false
promotion_eligible: false
ready_for_exact_eval_dispatch: false
target_modes: [contest_exact_eval, contest_generalized, production_generalized]
deployment_target: t4_contest_runtime  # also production_generalized at maturation
campaign_tier: long_term
expected_horizon_weeks: 8-12
expected_dispatch_cost_usd_band: [50, 100]
expected_delta_S_band: [-0.010, -0.020]  # cumulative on top of C1
predicted_post_campaign_score_band: [0.035, 0.07]  # staircase asymptote
council_tally: 6/11  # Hotz/Contrarian/Selfcomp/Yousfi/Fridrich dissent on cost
operator_decision_required: true
---

# Campaign C2 — Z7 Mature Predictive-Receiver L5 Substrate

## 0. One-line summary

Mature the full Time-Traveler L5 architecture: cooperative-receiver + predictive coding + foveation + world model + Tikhonov regularization + sub-100K params. Predicted final post-staircase score band **[0.035, 0.07]** — the Time-Traveler post-L5-solved asymptote. Cost $50-100, 2-3 months. **OPERATOR DECISION REQUIRED** (cost gate; 6/11 council vote with cost dissent).

## 1. lane_id + dispatch-claim plan

- `lane_id`: `lane_c2_z7_mature_predictive_receiver_l5_campaign_20260514` (pre-registered L0, phase 3)
- Dispatch claim: append row to `.omx/state/active_lane_dispatch_claims.md` per stage; multiple substrate-iteration cycles get explicit `iteration_N` suffixes.
- 24-hr TTL applies; multi-week campaigns require periodic claim renewal per CLAUDE.md cross-agent coordination non-negotiable.

## 2. Source evidence + score-lowering hypothesis

**Source evidence:**
- `zen_floor_field_medal_grade_council_20260514.md` §8 Decision Z7 (6/11 vote; OPERATOR DECISION REQUIRED)
- `time_traveler_architecture_reverse_engineered_20260513.md` (full L5 architecture)
- Atick-Redlich 1990 + Wyner-Ziv 1976 + Slepian-Wolf 1973 (cooperative-receiver lineage)
- Rao-Ballard 1999 + Friston 2010 (predictive coding hierarchy)
- Shannon 1959 (vector-valued R(D) — recovered by ancient-elder memo §16 as a single highest-leverage theoretical work)

**Primary URLs / identifiers (retrieved 2026-05-16):**
Retrieved 2026-05-16.

- Official challenge/runtime contract:
  https://github.com/commaai/comma_video_compression_challenge
- Public frontier lineage:
  - PR95: https://github.com/commaai/comma_video_compression_challenge/pull/95
  - PR100: https://github.com/commaai/comma_video_compression_challenge/pull/100
  - PR101: https://github.com/commaai/comma_video_compression_challenge/pull/101
  - PR103: https://github.com/commaai/comma_video_compression_challenge/pull/103
  - PR106: https://github.com/commaai/comma_video_compression_challenge/pull/106
- HNeRV:
  https://openaccess.thecvf.com/content/CVPR2023/html/Chen_HNeRV_A_Hybrid_Neural_Representation_for_Videos_CVPR_2023_paper.html
  and https://arxiv.org/abs/2304.02633
- DCVC-RT:
  https://arxiv.org/abs/2502.20762 and https://github.com/microsoft/DCVC
- TeCoNeRV:
  https://arxiv.org/abs/2602.16711 and https://namithap10.github.io/teconerv/
- Atick-Redlich:
  https://doi.org/10.1162/neco.1990.2.3.308
- Rao-Ballard:
  https://doi.org/10.1038/4580
- Friston free-energy:
  https://doi.org/10.1038/nrn2787
- Slepian-Wolf:
  https://doi.org/10.1109/TIT.1973.1055037
- Wyner-Ziv:
  https://doi.org/10.1109/TIT.1976.1055508

**Claim blockers:**
- The `[0.035, 0.07]` asymptote is a planning prior, not an empirical score.
- Provider cost rows are planning estimates unless refreshed by a live provider
  rate snapshot at dispatch time.
- Mature-L5 claims require a byte-closed archive grammar, exact archive SHA,
  runtime tree/content SHA, paired CPU/CUDA exact eval, and component deltas
  before public/paper wording can call the result empirical.

**Hypothesis (Time-Traveler post-L5 asymptote):**
1. Full L5 substrate composes 5 first-principles design moves: (1) cooperative-receiver, (2) predictive coding, (3) foveation matched to ego-motion, (4) differentiable world model, (5) sub-100K params with Tikhonov regularization.
2. Each prior staircase step (Z3, Z4, Z5, Z6) leaves cumulative slack; Z7 closes the gap to Time-Traveler's deep-future prediction by iterating + composing all 5 mechanisms in ONE coherent archive grammar.
3. Cumulative predicted ΔS = -0.090 to -0.160 from PR101 0.193 → S ∈ [0.035, 0.07].
4. The unified-Lagrangian action principle (zen-memo E4 sister) operationalizes the substrate: optimize `δS_total/δθ = 0` where `S_total = α·B(θ)/N + β·d_seg(θ) + γ·√d_pose(θ)` AND each term is computed via the actual scorer with eval_roundtrip + differentiable YUV6.

**Why this is NOT a research-only memo:**
- The substrate is BUILDABLE from primitives that exist (PR95 score-aware, A1 cooperative-receiver, LAPose foveation, D4 Wyner-Ziv frame-0, PR101 LoRA, score_pair_components canonical training loss).
- Missing pieces: (a) iteration cycle loop (substrate-generation framework), (b) automated probe-disambiguator orchestration, (c) cross-substrate composition matrix maturation.
- Build effort: 8-12 weeks (3-6 generations of L5 evolution; each generation 1-3 weeks).

## 3. Timing-smoke command (≤$0.50 per iteration, ≤60 min)

```bash
# Stage 0A — macOS-CPU advisory smoke per iteration (FREE)
.venv/bin/python experiments/train_substrate_c2_predictive_receiver_l5.py \
    --epochs 5 --batch-size 1 --quick-smoke \
    --iteration-gen 1 \
    --output-dir experiments/results/c2_iter1_smoke_macos_$(date -u +%Y%m%dT%H%M%SZ) \
    --advisory-cpu-explicitly-waived

# Stage 0B — Modal A100 timing smoke per iteration
.venv/bin/python tools/run_modal_smoke_before_full.py \
    --recipe .omx/operator_authorize_recipes/substrate_c2_z7_predictive_receiver_l5_iter1_modal_a100_dispatch.yaml \
    --smoke-epochs 100 --smoke-batch-size 4 \
    --max-cost-usd 0.50 \
    --advisory-cpu-explicitly-waived
```

**Timing-smoke kill criterion:** if any iteration's Stage 0B does not show ≥-0.005 cumulative ΔS vs prior iteration's archive on macOS-CPU advisory eval, escalate to grand council for probe-disambiguator routing.

## 4. Full-run command (resumable + harvest)

```bash
# PER-ITERATION (3-6 iterations expected over 8-12 weeks)
# Iteration N: ($10-20 Modal A100, 1-2 weeks build + run)
.venv/bin/python tools/operator_authorize.py \
    --recipe substrate_c2_z7_predictive_receiver_l5_iterN_modal_a100_dispatch \
    --operator-authorize-confirmed-via-session-directive \
    --operator-authorize-session-budget-usd 100.0 \
    --resume-from-checkpoint experiments/results/c2_iterN_prior/best_ckpt.pt \
    --enable-substrate-iteration-cycle \
    --iteration-gen N

# CROSS-SUBSTRATE COMPOSITION MATRIX EXHAUSTIVE (1× at iteration 3-4)
.venv/bin/python tools/build_cross_substrate_composition_matrix.py \
    --substrates c1,a1,d4,yucr,pr95_lora,siren_residual,wyner_ziv_pair \
    --max-cells 16 --cost-band-usd 20 \
    --output-dir experiments/results/c2_composition_matrix_$(date -u +%Y%m%dT%H%M%SZ)
```

**Harvest path:** `experiments/results/c2_iter*/harvested_artifacts/` per Catalog #204 durable provider output. Iteration log to `experiments/results/c2_iteration_log_$(date).md`.

**Resume guarantee:** every iteration writes `best_ckpt.pt` + `iteration_metadata.json` to Modal Volume; next iteration initializes from prior best.

## 5. Live provider rate/cost model

| Provider | GPU | $/hr | Per-iter wall-clock | Per-iter $ |
|---|---|---|---|---|
| Modal | A100-80GB | $5.40 | 2-5h | $10-27 |
| Modal | A100-40GB | $4.10 | 3-7h | $12-29 |
| Vast.ai | RTX 4090 | $0.25 | 8-24h | $2-6 |
| Vast.ai | A100-40GB | $1.00-1.50 | 3-7h | $3-11 |

**Total campaign cost band: $50-100** (3-6 iterations × $10-20 each) + composition matrix $20-30.

**Cost gate consultation:** `tac.cost_band_calibration.estimate_cost_usd("modal", "a100", elapsed_sec)` at every dispatch. Refuse if estimated > 1.5× band.

**OPERATOR FUNDING DECISION:** per CLAUDE.md "Long-burn" non-negotiable, $24 historical cap is superseded; this campaign requires explicit operator authorization at the $100 band.

## 6. Byte-closed archive/export/inflate plan

**Final L5 archive grammar (maturation target):**

```
TimeTravelerArchive-C2-L5-mature (target: 60-95 KB total)
├── HEADER (~2 KB)
├── STAGE 1: WORLD MODEL + FOVEATION (~40-55 KB) — co-trained, iterated
│   ├── scene_geometry_prior         ~6 KB  (FP4 small MLP, Tikhonov-regularized)
│   ├── ego_motion_dynamics_prior    ~2 KB  (Markov + Lie algebra)
│   ├── segmentation_class_palette   ~2 KB
│   ├── foveation_grid               ~1.5 KB (log-polar, ego-motion-matched)
│   ├── predictive_decoder           ~25 KB (sub-60K param, score-aware-trained)
│   └── differentiable_physics_op    ~5 KB  (Lie algebra + bilinear plane)
├── STAGE 2: PER-PAIR PREDICTION ERROR (~12-20 KB = ~20-33 bytes/pair × 600 pairs)
│   ├── pose_delta_SE3_lie_algebra   8 bytes/pair
│   ├── segnet_argmax_boundary       8 bytes/pair (matured: tighter than C1's 18)
│   ├── prediction_error_arithmetic  4 bytes/pair (CTW + conditional priors)
│   └── per_pair_hyperprior_state    3 bytes/pair (Ballé hyperprior)
├── STAGE 3: ARITHMETIC CODING STATE (~8-12 KB)
└── STAGE 4: SECTION OFFSETS (~2-3 KB)
```

**Inflate.py LOC budget: ≤200 LOC** (HNeRV parity lesson 4; matured substrate may push the limit — explicit waiver allowed up to 250 with rationale per the maturation context).

**Cross-substrate composition support:** the C2 substrate MUST expose a Catalog #108 cross-paradigm composition row so it can be HStacked with D4 (Wyner-Ziv frame-0) or A1 (cooperative-receiver) for the composition matrix.

**Export-first contract:** declared HERE. Trainer's `_write_runtime` emits 3-arg `inflate.sh` per Catalog #146.

**Score-aware loss + differentiable scorers:** `tac.differentiable_eval_roundtrip.apply_eval_roundtrip_during_training` + `patch_upstream_yuv6_globally` + `load_differentiable_scorers` per Catalog #187 hnerv_training_parity_guard.

## 7. Stop/continue thresholds

| Iteration | Continue if | Stop if |
|---|---|---|
| Iter 1 | macOS-CPU [0.07, 0.10] (matches C1 final + improvement) | macOS-CPU > 0.13 (substrate doesn't compose with Z6) |
| Iter 2 | macOS-CPU [0.05, 0.08]; trigger Linux x86_64 paired eval | macOS-CPU > 0.10 |
| Iter 3 | macOS-CPU [0.04, 0.06]; cross-substrate composition matrix dispatched | macOS-CPU > 0.07 |
| Iter 4+ | each iteration improves macOS-CPU by ≥0.005 cumulative | 2 consecutive iterations show <0.002 improvement (saturation) |
| Final exact eval | [contest-CPU] AND [contest-CUDA] both in [0.035, 0.07] → council L3 promotion | either >0.10 → DEFERRED-pending-research |

**Falsification criteria:**
- If iteration 3 final [contest-CPU] > 0.10 → Time-Traveler prediction at the L5 asymptote partially falsified.
- Time-Traveler frame still survives at the staircase Z3-Z6 level; only the Z7 deep-future asymptote is revisited.
- Reactivation: requires alien-tech composition (E4 MDL-IBPS, Wyner-Ziv full substrate, DARTS-SuperNet output).

## 8. Dependencies + sequence gating

```
C1 (Z6 world model + foveation) lands in [0.06, 0.10] band
   ↓
**THIS CAMPAIGN (Z7 / C2): $50-100, 8-12 weeks, 3-6 iterations**
   ↓
C3 (multi-year zen-floor sub-0.05): TRIGGERS on successful C2 landing
```

C2 cannot proceed before C1 lands successfully. If C1 lands >0.13, escalate to grand council before authorizing C2.

## 9. 6-hook wire-in (Catalog #125 NON-NEGOTIABLE)

1. **Sensitivity-map (`tac.sensitivity_map.*`)** — ENGAGED. Maturation registers `sensitivity_map.time_traveler_l5_unified_v1` combining foveation + cooperative-receiver + predictive-coding sensitivity priors.
2. **Pareto constraint (`tac.pareto_*`)** — ENGAGED. Time-Traveler L5 final archive ≤ 95 KB; per-pair ≤ 33 bytes; explicit Pareto constraint `tac.pareto.time_traveler_l5_mature_v1`. Shannon-1959 vector R(D) lower bound from ancient-elder §16 registered as theoretical floor.
3. **Bit-allocator hook (`tac.composition.registry`)** — ENGAGED. Stage 1 (world model) vs Stage 2 (prediction error) split iterated via Fisher-water-filling. Register `bit_allocator.time_traveler_l5_unified_v1`.
4. **Cathedral autopilot dispatch hook** — ENGAGED. Add to `.omx/state/autopilot_candidate_queue_solver_stack_wire_in_20260513.jsonl` with cost-band $50-100, predicted_dS [-0.090, -0.160] cumulative.
5. **Continual-learning posterior update (`tac.continual_learning.append_anchor`)** — ENGAGED. Every iteration's empirical anchor updates posterior. The cross-substrate composition matrix (iteration 3-4) produces 16+ anchors in one batch — a key high-information empirical event.
6. **Probe-disambiguator** — ENGAGED. Multiple defensible interpretations: (a) "is the gain from prediction coding or foveation?" (b) "is differentiable physics necessary or is Tikhonov-regularized MLP enough?" (c) "does cross-substrate composition Amdahl-compose or does redundancy dominate?" Planned probes: `tools/probe_c2_predictive_vs_foveation.py`, `tools/probe_c2_diff_physics_necessity.py`, `tools/probe_c2_composition_amdahl_vs_redundancy.py`.

## 10. Cross-references

- `.omx/research/zen_floor_field_medal_grade_council_20260514.md` §8 Decision Z7
- `.omx/research/time_traveler_architecture_reverse_engineered_20260513.md`
- `.omx/research/grand_council_maximize_value_with_time_traveler_seat_20260514.md`
- `.omx/research/deep_math_geometry_manifolds_synthesis_20260514.md` (unified action E4 + §5.5 first-principles)
- `feedback_ancient_elder_polymath_landed_20260513.md` §16 Shannon 1959 vector R(D)
- CLAUDE.md "HNeRV parity discipline" (all 13 lessons honored; explicit LOC waiver allowed up to 250 for matured L5)
- CLAUDE.md "Long-burn score-lowering campaign default" (this ledger satisfies all 7 mandatory fields)

## 11. Operator-routable decision

**STATUS:** OPERATOR DECISION REQUIRED (cost gate $50-100, 8-12 weeks).

**Council vote:** 6/11 SUPPORT (Selfcomp/MacKay/Ballé/Time-Traveler/Quantizr/Dykstra). 5/11 DISSENT on cost (Hotz/Contrarian/Selfcomp-cost-axis/Yousfi/Fridrich).

**Recommended timing:** authorize at the moment C1 lands successfully in [0.06, 0.10] band. NOT before.

**Decision matrix entry:** NOW=NO; SOON=NO; LATER=PENDING-OPERATOR-AUTHORIZATION; DEFER=if C1 lands >0.13.

**OPERATOR routing question:** is the operator's MULTI-YEAR target sub-0.05 (Time-Traveler trajectory) or 1-YEAR target sub-0.10 (staircase Step 3-4)? If multi-year, C2 authorize; if 1-year, defer C2 and converge.

Tag: `[council-deliberation]` + `[operator-decision-required]`. NO score claim. NO archive bytes built. $0 GPU spent at landing.
