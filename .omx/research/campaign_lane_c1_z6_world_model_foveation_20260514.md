---
title: "Campaign C1 — Z6 World Model + Foveation Substrate (Time-Traveler L5 Moves 3-4)"
date: 2026-05-14
status: campaign_ledger; operator-routable
lane_id: lane_c1_z6_world_model_foveation_campaign_20260514
score_claim: false
evidence_axes: [time-traveler-prediction, mathematical-derivation, council-deliberation]
score_claim_valid: false
promotion_eligible: false
ready_for_exact_eval_dispatch: false
research_only: false  # this is a campaign ledger, not research
target_modes: [contest_exact_eval, contest_generalized]
deployment_target: t4_contest_runtime
campaign_tier: long_term
expected_horizon_weeks: 3-4
expected_dispatch_cost_usd_band: [30, 50]
expected_delta_S_band: [-0.020, -0.040]
predicted_post_campaign_score_band: [0.06, 0.10]  # cumulative after Z3+Z4+Z5+Z6 staircase
---

# Campaign C1 — Z6 World Model + Foveation Substrate

## 0. One-line summary

Build differentiable world model + foveation-matched-to-ego-motion substrate (Time-Traveler L5 moves 3-4). Predicted post-Z6 cumulative score band **[0.06, 0.10]** `[time-traveler-prediction]`. Cost $30-50, 3-4 weeks. Trigger: AVAILABLE after Z5 (predictive coding substrate) lands.

## 1. lane_id + dispatch-claim plan

- `lane_id`: `lane_c1_z6_world_model_foveation_campaign_20260514` (pre-registered L0, phase 3)
- Dispatch claim: append row to `.omx/state/active_lane_dispatch_claims.md` via `tools/claim_lane_dispatch.py claim` at first GPU spend. Per CLAUDE.md cross-agent coordination non-negotiable.
- 24-hr TTL on each smoke/full row; multiple stages (foveation alone → world model alone → composed) get separate `instance/job_id` suffixes.

## 2. Source evidence + score-lowering hypothesis

**Source evidence:**
- `zen_floor_field_medal_grade_council_20260514.md` §8 Decision Z6 (8/11 council vote; Hotz/Contrarian/Selfcomp dissent on cost)
- `time_traveler_architecture_reverse_engineered_20260513.md` (Stage 1 world model ~55-70 KB + foveation grid ~2 KB)
- Atick-Redlich 1990 cooperative-receiver theorem
- Gibson 1950 + Lee 1976 + LAPose existing canvas
- Rao-Ballard 1999 + Friston 2010 predictive coding

**Primary URLs / identifiers (retrieved 2026-05-16):**
Retrieved 2026-05-16.

- Atick-Redlich efficient coding:
  https://doi.org/10.1162/neco.1990.2.3.308
- Gibson visual-world bibliographic record:
  https://philpapers.org/rec/GIBTPO-2
- Lee time-to-collision / visual braking:
  https://doi.org/10.1068/p050437
- Rao-Ballard predictive coding:
  https://doi.org/10.1038/4580
- Friston free-energy principle:
  https://doi.org/10.1038/nrn2787
- Time-Traveler source architecture ledger:
  `.omx/research/time_traveler_architecture_reverse_engineered_20260513.md`

**Hypothesis (first-principles):**
1. Foveation-matched-to-ego-motion (log-polar grid around FOE) is a planning
   hypothesis for allocating more bytes to score-relevant pixels at fixed byte
   budget per Gibson 1950 / Lee 1976 and the SegNet stride-2-stem blind spot
   pattern that motivated the YUCR cost-map approach. It is not an empirical
   Pact score or byte-saving claim until measured with archive/runtime custody.
2. Differentiable physics renderer encodes scene state (~50-100 bytes/scene) rather than pixels (~590 KB raw). Tikhonov regularization principle: under-parameterized with good prior beats over-parameterized without prior.
3. Cumulative predicted ΔS = -0.020 to -0.040 on top of Z3+Z4+Z5 staircase steps; total cumulative -0.080 to -0.140 from PR101 0.193 → S ∈ [0.06, 0.10].

**Why this is NOT a research-only memo (per CLAUDE.md "Long-burn score-lowering campaign default"):**
- Mathematical derivations come from already-landed primitives (LAPose, A1 score-aware, PR101 score-aware loss).
- The missing piece is a SINGLE new substrate that composes 3 existing primitives + 1 new differentiable physics module.
- Build effort: 2-3 days for differentiable physics renderer (estimate from time-traveler memo); 5-7 days for foveation grid extension (sister to existing LAPose); 7-10 days integration.

## 3. Timing-smoke command (≤$0.30, ≤30 min)

```bash
# Stage 0A — macOS-CPU advisory smoke (FREE; pre-validates code paths)
.venv/bin/python experiments/train_substrate_c1_world_model_foveation.py \
    --epochs 5 --batch-size 1 --quick-smoke \
    --enable-foveation-only \
    --output-dir experiments/results/c1_smoke_macos_$(date -u +%Y%m%dT%H%M%SZ) \
    --advisory-cpu-explicitly-waived

# Stage 0B — Modal T4 timing smoke at $0.30 (validates GPU throughput before full $30-50 spend)
.venv/bin/python tools/run_modal_smoke_before_full.py \
    --recipe .omx/operator_authorize_recipes/substrate_c1_z6_world_model_foveation_modal_t4_dispatch.yaml \
    --smoke-epochs 100 --smoke-batch-size 4 \
    --max-cost-usd 0.50 \
    --advisory-cpu-explicitly-waived
```

**Timing-smoke kill criterion:** if Stage 0B does not produce ≥1 valid archive (≤120 KB after FP4 quantization) within 30 min on T4, abort and escalate to grand council. **Predicted Stage 0B cost: $0.20-0.40.**

## 4. Full-run command (resumable + harvest)

```bash
# Stage 1 — Foveation grid alone ($3-5 Modal A100, 1-2 days)
.venv/bin/python tools/operator_authorize.py \
    --recipe substrate_c1_z6_foveation_only_modal_a100_dispatch \
    --operator-authorize-confirmed-via-session-directive \
    --operator-authorize-session-budget-usd 50.0 \
    --resume-from-checkpoint experiments/results/c1_smoke_*/best_ckpt.pt

# Stage 2 — Differentiable physics renderer alone ($5-10 Modal A100, 3-4 days)
.venv/bin/python tools/operator_authorize.py \
    --recipe substrate_c1_z6_diff_physics_only_modal_a100_dispatch \
    --operator-authorize-confirmed-via-session-directive \
    --operator-authorize-session-budget-usd 50.0 \
    --resume-from-checkpoint experiments/results/c1_z6_foveation_*/best_ckpt.pt

# Stage 3 — Composed substrate ($15-25 Modal A100, 7-10 days; FULL CAMPAIGN COST)
.venv/bin/python tools/operator_authorize.py \
    --recipe substrate_c1_z6_world_model_foveation_composed_modal_a100_dispatch \
    --operator-authorize-confirmed-via-session-directive \
    --operator-authorize-session-budget-usd 50.0 \
    --resume-from-checkpoint experiments/results/c1_z6_diff_physics_*/best_ckpt.pt
```

**Harvest path:** `experiments/results/c1_z6_*/harvested_artifacts/` (per CLAUDE.md `tools/harvest_modal_calls.py` + Modal Volume durable output per Catalog #204).

**Resume guarantee:** every stage writes `best_ckpt.pt` + `provenance.json` to a Modal Volume; the next stage initializes from the previous via `--resume-from-checkpoint`. Smoke artifacts harvested to local `experiments/results/c1_z6_*/` for forensic audit.

## 5. Live provider rate/cost model

| Provider | GPU | $/hr | Per-stage wall-clock | Per-stage $ |
|---|---|---|---|---|
| Modal | A100-80GB | $5.40 | 0.5-2h (smoke) | $2.70-10.80 |
| Modal | A100-80GB | $5.40 | 3-7h (full) | $16-37 |
| Modal | T4 | $0.59 | 1-3h (smoke) | $0.59-1.77 |
| Vast.ai | RTX 4090 | $0.25 | 4-12h (full) | $1-3 |

**Live cost gate:** `tac.cost_band_calibration.estimate_cost_usd("modal", "a100", elapsed_sec)` consulted at every dispatch per cost-band posterior (Catalog #175/#177). Refuse dispatch if estimated > 1.5× expected band.

**Vast.ai 4090 alternative:** $1-3 per stage but slower wall-clock; use ONLY for non-time-critical full runs (per CLAUDE.md "Race-mode rigor inversion" — no active race window for long-term work).

**Total campaign cost band: $25-55** (smoke + 3 full stages); CLAUDE.md "Long-burn" non-negotiable removes the $24 historical cap.

## 6. Byte-closed archive/export/inflate plan

**Archive grammar declaration (Catalog #124 required for L1+ promotion):**

```
TimeTravelerArchive-C1 (target: 95-110 KB total)
├── HEADER (~2 KB): magic + lengths + grammar version
├── STAGE 1: WORLD MODEL (~55-70 KB, encoded ONCE)
│   ├── scene_geometry_prior         ~8 KB  (FP4-quantized small MLP)
│   ├── ego_motion_dynamics_prior    ~3 KB  (Markov state-transition table)
│   ├── segmentation_class_palette   ~2 KB  (5-class FP4 codebook)
│   ├── foveation_grid               ~2 KB  (log-polar grid params)
│   ├── predictive_decoder           ~35 KB (sub-100K param renderer)
│   └── differentiable_physics_op    ~10 KB (Lie algebra pose transform + plane geometry)
├── STAGE 2: PER-PAIR SIDE INFO (~25-35 KB = ~45 bytes/pair × 600 pairs)
│   ├── pose_delta_SE3_lie_algebra   12 bytes/pair
│   ├── segnet_argmax_boundary_only  18 bytes/pair
│   ├── hf_residual_dsss             6 bytes/pair
│   └── prediction_error_residual    9 bytes/pair
├── STAGE 3: ARITHMETIC CODING STATE (~10 KB)
└── STAGE 4: SECTION OFFSETS (~3 KB)
```

**Inflate.py LOC budget: ≤200 LOC** (HNeRV parity lesson 4; explicit waiver if >100 with rationale).

**Runtime closure (HNeRV parity lesson 9):** Stage 0B smoke MUST run `inflate.sh archive_dir output_dir file_list` in clean Modal environment before any full dispatch.

**Export-first contract:** archive grammar declared HERE, in this ledger, BEFORE first training script lines. The trainer's `_write_runtime` function emits the exact 3-arg `inflate.sh` signature per Catalog #146.

**Score-aware loss (HNeRV parity lesson 1):** trainer must use `tac.substrates._shared.score_aware_common.score_pair_components` + `tac.differentiable_eval_roundtrip.apply_eval_roundtrip_during_training` + `patch_upstream_yuv6_globally()` per `check_hnerv_training_parity_guard` (Catalog #187).

## 7. Stop/continue thresholds

| Stage | Continue if | Stop if |
|---|---|---|
| Smoke (Stage 0B) | seconds/epoch < 60s ON T4 AND archive ≤120 KB | timeout >30 min OR archive >150 KB OR proxy score >0.30 |
| Stage 1 (foveation only) | macOS-CPU advisory score [0.180, 0.195] | macOS-CPU > 0.20 (predictive-coding base broken) |
| Stage 2 (diff physics only) | macOS-CPU advisory score [0.140, 0.160] | macOS-CPU > 0.175 (composition not additive) |
| Stage 3 (composed) | macOS-CPU advisory [0.100, 0.130]; trigger paid Linux x86_64 [contest-CPU] | macOS-CPU > 0.150 → diagnostic escalation only, not campaign falsification |
| Exact eval ([contest-CPU] + [contest-CUDA]) | both axes in [0.06, 0.10] band with matching archive SHA, runtime tree/content SHA, sample count, logs, component recomputation, and paired CPU/CUDA exact custody → operator frontier review; council L2 review | either axis >0.13 → DEFER-pending-research per HNeRV parity lesson 8 (DO NOT KILL) |

**Falsification criteria (Z6 specific, council-deliberated):**
- If Stage 3 final [contest-CPU] > 0.13 → zen-floor band revises UP to [0.10, 0.15]; Time-Traveler trajectory partially falsified; reactivation = staircase Step 7+ requires alien-tech composition (E4 MDL-IBPS or Wyner-Ziv full substrate).
- If pose_avg moves AWAY from PR106 frontier (3.4e-5 → >1e-4) → diagnostic flag; per CLAUDE.md "SegNet vs PoseNet importance — operating-point dependent" the pose-marginal 2.71× regime is the most sensitive axis.

**Reactivation criteria if KILL is genuinely warranted (per CLAUDE.md "KILL is LAST RESORT"):**
- Every plausible architectural / training / codec / quantization angle attempted empirically (≥6 configurations)
- Grand council CONSENSUS (not majority); inner-quintet 5/5 + at least 8/11 grand council
- Reactivation criteria documented in `feedback_<lane_id>_DEFERRED_pending_<reason>_20260*.md`

## 8. Dependencies + sequence gating

```
Z3 (Ballé hyperprior bolt-on, $2 GPU, 4 days)  [SEPARATE CAMPAIGN]
   ↓
Z4 (conditional entropy substrate, $5-8 GPU, 5-7 days)  [SEPARATE CAMPAIGN]
   ↓
Z5 (predictive coding substrate, $10 GPU, 5-7 days)  [SEPARATE CAMPAIGN]
   ↓
**THIS CAMPAIGN (Z6 / C1): $30-50 GPU, 3-4 weeks**
   ↓
C2 (Z7 mature L5 substrate): TRIGGERS on successful C1 landing
```

C1 cannot proceed before Z3+Z4+Z5 land. If Z3 falsifies the staircase (Z3 lands at 0.193 unchanged), the operator should re-examine before authorizing C1 spend.

## 9. 6-hook wire-in (Catalog #125 NON-NEGOTIABLE)

1. **Sensitivity-map (`tac.sensitivity_map.*`)** — ENGAGED. Foveation grid IS a sensitivity-map primitive; the log-polar weighting comes from `tac.sensitivity_map.score_gradient_field`. New entry to register: `sensitivity_map.foveation_log_polar_v1`.
2. **Pareto constraint (`tac.pareto_*`)** — ENGAGED. World-model + per-pair side info ≤ 110 KB; foveation grid + pose-delta ≤ 4 KB/pair; explicit Pareto constraint registered in `tac.pareto.time_traveler_substrate_v1`.
3. **Bit-allocator hook (`tac.composition.registry`)** — ENGAGED. Per-pair byte budget (45 bytes/pair) Fisher-water-fillable; register `bit_allocator.foveation_aware_v1` consumable by `tac.composition.registry.allocate_bits`.
4. **Cathedral autopilot dispatch hook** — ENGAGED. Add row to `.omx/state/autopilot_candidate_queue_solver_stack_wire_in_20260513.jsonl` for `lane_c1_z6_world_model_foveation_campaign_20260514` with cost-band $30-50, predicted_dS [-0.020, -0.040], EIG high.
5. **Continual-learning posterior update (`tac.continual_learning.append_anchor`)** — ENGAGED. Empirical anchor from Stage 3 exact eval updates posterior via `posterior_update_locked` per Catalog #128; zen-floor band revision logic triggered.
6. **Probe-disambiguator** — ENGAGED. Two defensible interpretations: (a) "full differentiable-physics" vs (b) "cheap-foveation-only fallback". Planned probe `tools/probe_c1_world_model_vs_foveation_only.py` (consumes Stage 1 + Stage 2 outputs to disambiguate which mechanism contributes most ΔS).

## 10. Cross-references

- `.omx/research/zen_floor_field_medal_grade_council_20260514.md` §8 Decision Z6
- `.omx/research/time_traveler_architecture_reverse_engineered_20260513.md` (full architecture)
- `.omx/research/grand_council_maximize_value_with_time_traveler_seat_20260514.md` (STAIRCASE Round 3 verdict)
- `feedback_adjusted_floor_v3_alien_tech_routing_landed_20260513.md` (top-3 dispatch queue including Time-Traveler L5)
- CLAUDE.md "HNeRV parity discipline" (13 inviolable lessons; this campaign must honor all 13)
- CLAUDE.md "Long-burn score-lowering campaign default" (this ledger satisfies all 7 mandatory fields)
- CLAUDE.md "KILL is LAST RESORT" (falsification triggers DEFERRED, not KILLED)

## 11. Operator-routable decision

**STATUS:** READY-TO-AUTHORIZE after Z5 lands. Operator must explicitly authorize $30-50 GPU spend + 3-4 weeks build time.

**Sequence:** wait for Z3 → Z4 → Z5 landings → council ratify Z6 → authorize.

**Decision matrix entry:** NOW=NO; SOON=NO; LATER=YES (post-Z5); DEFER=if Z3 falsifies staircase.

Tag: `[council-deliberation]`. NO score claim. NO archive bytes built. $0 GPU spent at landing.
