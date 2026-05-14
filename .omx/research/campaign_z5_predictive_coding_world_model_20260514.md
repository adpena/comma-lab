# Campaign ledger: Z5 predictive-coding world-model (staircase Step 3)

`research_only=false` — L1 SCAFFOLD; Phase 2 council approval required to lift `_full_main` `NotImplementedError`. NO score claims. NO dispatch this turn.

Per CLAUDE.md "Long-burn score-lowering campaign default — NON-NEGOTIABLE":

## 1. lane_id + dispatch-claim plan

- **Lane**: `lane_z5_predictive_coding_world_model_step3_20260514`
- **Pre-registered at L0** via `tools/lane_maturity.py add-lane`. Promoted to L1 after this landing (impl_complete + memory_entry + strict_preflight via Catalog #124).
- **Dispatch-claim plan**: per CLAUDE.md "CROSS-AGENT DISPATCH COORDINATION" non-negotiable, smoke + full each claim via `tools/claim_lane_dispatch.py claim`. 24h TTL; terminal row appended on completion. NO concurrent same-`lane_id` dispatches. **canary_status=`post_canary_dependent` with canary_dependency=`lane_z4_cooperative_receiver_loss_step2_20260514`** — Z5 dispatch is gated on Z4 having ≥ 1 successful contest-CUDA anchor (Catalog #173 + #167).

## 2. Source evidence + score-lowering hypothesis

- **Source evidence**:
  - Rao & Ballard (1999) "Predictive coding in the visual cortex: a functional interpretation of some extra-classical receptive-field effects" Nature Neuroscience 2(1):79-87 — predictive-coding hierarchy foundational paper.
  - Friston (2010) "The free-energy principle: a unified brain theory?" Nature Reviews Neuroscience 11:127-138 — world-model free-energy framing.
  - Time-Traveler peer-seat council (`feedback_grand_council_maximize_value_landed_20260514.md`): Step 3 staircase end-state, sub-0.188 reachable; Time-Traveler asymptote ~0.03 only after this step or further.
  - Zen-floor field-medal council (`feedback_zen_floor_field_medal_grade_council_landed_20260514.md`): Time-Traveler's `[0.03, 0.07]` staircase asymptote requires class-shift moves; this is one of them.
  - CLAUDE.md "Long-burn score-lowering campaign default": "Visible high-EV directions such as ... RAFT/ego-motion, LA-pose/telescopic foveation ... arithmetic/range/ANS compiler passes, and scorer-inverse representations must become either a campaign ledger plus timing-smoke/launch decision in the same session or an explicit blocker."

- **Hypothesis**:
  Training a 2-3 layer hierarchical predictive-coding network with ego-motion conditioning reduces residual entropy by 20-40% vs Z4 → **score band [0.155, 0.180]** on contest-CUDA T4 vs Z4's predicted [0.180, 0.188]. Δ predicted: −0.025 to −0.038 vs Z4.

  Mechanism: for stationary-ergodic driving video, the asymptotic entropy is dominated by frame-to-frame surprise. A predictor that forecasts `z_t` from `z_{t-1}` + ego-motion encodes only the surprise residual, which has lower entropy than the marginal latent distribution.

  **Probe-disambiguator regime control**: `identity_predictor=True` (no learning; predicts z_t = z_{t-1}) is the canonical ablation. If the full hierarchical predictor does NOT beat the identity-predictor variant by Δ ≥ 0.005, the Rao-Ballard predictive-coding hypothesis is refuted and the gain comes purely from added decoder capacity.

## 3. Timing-smoke command (~$1 Modal T4 1-epoch smoke)

```bash
# Local smoke (no GPU; ~30s on M5 Max):
.venv/bin/python experiments/train_substrate_z5_predictive_coding_world_model.py \
    --output-dir experiments/results/z5_smoke_$(date -u +%Y%m%dT%H%M%SZ) \
    --epochs 3 --device cpu --smoke

# Modal T4 smoke ($1, ~20 min — predictor + residuals + autoregression):
.venv/bin/python tools/run_modal_smoke_before_full.py \
    --recipe .omx/operator_authorize_recipes/substrate_z5_predictive_coding_world_model_modal_t4_dispatch.yaml \
    --smoke-only \
    --operator-approved 'adpena:<UTC>'
```

Expected wall-clock: ~20 min on Modal T4 (longer than Z4 due to autoregressive predictor unroll across 600 pairs at training time). Smoke validates substrate forward, predictor unroll, archive packs all 7 sections, inflate roundtrip succeeds, no NaN watchdog.

## 4. Full-run command (~$10 Modal T4 300-epoch full) — REQUIRES PHASE 2 COUNCIL APPROVAL

`_full_main` body is currently `NotImplementedError("Phase 2 council approval required")`. Operator-routable decision in §7. After council approval:

```bash
# Modal T4 full run ($10, ~30 min once scheduled):
.venv/bin/python tools/operator_authorize.py \
    --recipe substrate_z5_predictive_coding_world_model_modal_t4_dispatch \
    --operator-approved 'adpena:<UTC>' \
    --auto-fan-out
```

The autoregressive predictor unroll across 600 pairs is the dominant per-step cost. Resume support via per-epoch EMA + predictor checkpoint.

## 5. Live provider rate + cost model

- **Modal T4**: $0.59/hr × (~20 min smoke + ~30 min full) ≈ **$1 smoke + $10 full = $11 total** at p50 fallback.
- **Modal A10G (faster for predictor)**: $1.40/hr × ~15 min full ≈ $5; but recipe defaults to T4 for portability.
- **Vast.ai 4090**: $0.25/hr; total ~$3-5.
- **Empirical anchor pending Z4 smoke** — once Z4 lands and we have measured (smoke_wall_clock × hourly_rate), this row updates.

## 6. Byte-closed archive/export/inflate plan

- **Archive grammar**: Z5PCWM1 monolithic single-file `0.bin`. 39-byte header + 3 brotli-compressed state_dicts (encoder + decoder + **predictor**) + 3 int8 blobs (latent_init + residuals + ego_motion) + JSON meta with `predictive_coding_world_model_meta` provenance tag. Implemented at `src/tac/substrates/z5_predictive_coding_world_model/archive.py`.
- **Inflate runtime**: `src/tac/substrates/z5_predictive_coding_world_model/inflate.py` ≤ 200 LOC (substrate-engineering waiver); no scorer imports; CPU/CUDA-agnostic via `select_inflate_device`; 3-positional-arg contract per Catalog #146. The autoregressive unroll is honored at inflate time: `z_t = predictor(z_{t-1}, ego_motion[t]) + residuals[t]`.
- **Export contract**: trainer `_full_main` (post-council) emits archive.zip via deterministic-ZIP helper; writes runtime tree per canonical pattern.
- **Tests confirm roundtrip**: `test_archive_roundtrip_preserves_residuals_and_ego_motion`, `test_inflate_one_video_writes_raw`, `test_archive_predictive_coding_meta_tag_present`, `test_substrate_autoregression_recurrence`. 31 dedicated tests passing.

## 7. Stop/continue thresholds

### Smoke gate (smoke-before-full per Catalog #167)
- **GREEN** (advance to full): smoke completes rc=0; archive bytes in [80_000, 250_000]; CPU smoke training loss converges (not NaN); autoregressive recurrence preserves shape; predictor produces non-trivial z_pred.
- **YELLOW** (operator review): smoke completes but residual norm not decreasing OR archive bytes outside band.
- **RED** (DEFERRED-pending-research): NaN, autoregressive recurrence fails (z_t shape mismatch), predictor gradient zero.

### Mid-stage gate (epoch 150 of full)
- **GREEN**: residual L2 norm decreasing > 10% vs initial; pose distortion approaching Z4 baseline; predictor weights not collapsing to zero.
- **YELLOW**: residual L2 norm flat → operator decides identity_predictor ablation OR continue.
- **RED**: predictor diverging — STOP, save checkpoint, DEFER per "KILL = LAST RESORT".

### Export gate (after EMA shadow saved)
- **GREEN**: archive packs deterministically; CPU local inflate parity passes; archive bytes within [80_000, 250_000]; predictor weights survive int8 quantization (negligible drift in autoregressive reconstruction).
- **YELLOW**: archive bytes outside band → operator decides repack with smaller predictor OR retire as `measured-config-retired`.
- **RED**: archive determinism fails OR inflate-runtime > 200 LOC budget.

### Exact eval gate (CUDA auth eval on EMA shadow)
- **GREEN** (mark `contest_cuda` gate): contest-CUDA T4 score in [0.150, 0.180] (target [0.155, 0.180] + 0.005 tolerance for scorer numerical drift on T4).
- **YELLOW** (operator review): score in [0.180, 0.200] — operator-route as `measured-config-retired-suboptimal`; document reactivation criteria. Run identity-predictor disambiguator probe to check if hypothesis is refuted.
- **RED** (DEFERRED-pending-research): score > 0.200 — DEFER. Reactivation criteria: identity-predictor ablation shows the full predictor adds zero ΔS (refutes Rao-Ballard hypothesis on this scorer); OR identify autoregression-related numerical bug; OR observe predictor instability across pairs.

## 6-hook wire-in (Catalog #125 NON-NEGOTIABLE)

1. **Sensitivity-map**: predictor gradient norm IS the per-tensor importance signal — `∂L_full/∂θ_predictor` reveals which predictor weights dominate forecast accuracy; register `sensitivity_map.predictive_coding_v1` post Phase 2.
2. **Pareto constraint**: `predictor_residual_entropy ≤ ε_residual` ∩ Z3+Z4 polytope; register `tac.pareto.predictive_coding_v1` post-smoke.
3. **Bit-allocator hook**: per-pair-residual bit allocation derives from predictor forecast uncertainty — high-uncertainty pairs get more bits. Register `bit_allocator.predictive_coding_residual_v1` post-smoke.
4. **Cathedral autopilot dispatch hook**: recipe registered at `.omx/operator_authorize_recipes/substrate_z5_predictive_coding_world_model_modal_t4_dispatch.yaml`; gated by Catalog #167 smoke-before-full and `canary_dependency=lane_z4_cooperative_receiver_loss_step2_20260514`. Ranker v2 (Catalog #219) reads `literature_anchor=Rao-Ballard1999` and applies the canonical -0.02 to -0.03 class-shift reward.
5. **Continual-learning posterior**: every Z5 empirical anchor seeds the posterior via `posterior_update_locked` (Catalog #128). The paired `(L_full, L_identity)` measurement is the canonical disambiguator data point.
6. **Probe-disambiguator**: `tools/probe_z5_predictive_coding_vs_no_prediction_disambiguator.py` (planned post Phase 2). Compares full-hierarchical vs identity-predictor at matched parameter budgets. Returns: "Rao-Ballard wins" (Δ ≥ 0.005 in favor of hierarchical), "capacity wins" (within ±0.002), or "predictive-coding refuted" (identity wins).

## Catalog #124 8 archive-grammar fields

All 8 declared inline in `src/tac/substrates/z5_predictive_coding_world_model/__init__.py`:

1. `archive_grammar`: monolithic single-file `0.bin` extends Z4CR1
2. `parser_section_manifest`: Z5PCWM1 header + 7 sections (3 state_dicts + 3 int8 blobs + meta)
3. `inflate_runtime_loc_budget`: ≤200 LOC substrate-engineering waiver
4. `runtime_dep_closure`: torch + brotli
5. `export_format`: Z5PCWM1 monolithic single-zip-member `0.bin`
6. `score_aware_loss`: `PredictiveCodingScoreAwareLoss` via `score_pair_components` (Catalog #164) + residual-entropy term (Rao-Ballard)
7. `bolt_on_loc_budget`: `lane_class=substrate_engineering` (HNeRV L7); predictor + autoregression is substrate engineering
8. `no_op_detector_planned`: predictor section MUST be consumed by inflate runtime; empirical detector mutates predictor bytes and verifies decoded frames change

## HNeRV parity discipline (CLAUDE.md non-negotiable 13 lessons)

1. ✅ Score-aware substrate via canonical `score_pair_components` (Catalog #164)
2. ✅ Export-first design (Z5PCWM1 grammar declared before training)
3. ✅ Monolithic single-file `0.bin`
4. ✅ Inflate.py ≤ 200 LOC (substrate-engineering waiver) — actual: ~165 LOC
5. ✅ Full renderer architecture (RGB out)
6. ✅ Score-domain Lagrangian per Catalog #164 + Rao-Ballard residual term
7. ✅ `lane_class=substrate_engineering` per HNeRV L7
8. ⏳ Eval-roundtrip + differentiable YUV6 (Catalog #187) — enforced in `_full_main` post-council
9. ⏳ Runtime closure — emit in `_full_main`'s `_write_runtime` helper
10. N/A — Mask/pose coupling gate (Z5 is full renderer)
11. ⏳ No-op detector — empirical; runs after first smoke completes (mutate predictor bytes → verify decoded frames differ)
12. ✅ Single-LOC-per-LOC review discipline — `score_aware_loss.py` ~225 LOC, every line reviewable in 30s
13. ✅ KILL = LAST RESORT (per CLAUDE.md non-negotiable)

## Cross-references

- **Step 1 sister**: `lane_z3_balle_hyperprior_bolton_campaign_20260514`
- **Step 2 sister**: `lane_z4_cooperative_receiver_loss_step2_20260514` (canary dependency)
- **C1 foveation sister**: world-model integration handoff (foveation hint feeds the predictor's ego-motion proxy)
- **Grand council (Time-Traveler peer seat)**: `feedback_grand_council_maximize_value_landed_20260514.md` (STAIRCASE 10/11)
- **Zen-floor council**: `feedback_zen_floor_field_medal_grade_council_landed_20260514.md` (Step 3 asymptote)
- **Long-burn campaign roadmap**: `feedback_long_term_multi_year_campaigns_landed_20260514.md`

Lane: `lane_z5_predictive_coding_world_model_step3_20260514`
Status: L1 SCAFFOLD; Phase 2 dispatch approval required to lift `_full_main` `NotImplementedError`.
