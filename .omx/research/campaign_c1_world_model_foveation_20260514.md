# Campaign C1: world-model + foveation substrate (long-term)

**Lane**: `lane_c1_world_model_foveation_campaign_l1_scaffold_20260514`
**Operator directive (2026-05-14)**: *"we are aggressively pursuing across class too"*
**Tag**: `research_only=true` for ALL planning text; smoke dispatches are `[smoke-no-scorer]`; full anchors will be `[contest-CUDA]` after Phase 3 council approval.

## 7-field campaign required envelope (CLAUDE.md "Long-burn score-lowering campaign default")

### 1. `lane_id` + dispatch-claim plan

- **lane_id**: `lane_c1_world_model_foveation_campaign_l1_scaffold_20260514`
- **Pre-registered**: L0 SKETCH 2026-05-14T15:28Z via `tools/lane_maturity.py add-lane`
- **Phase 2** (council-tracked)
- **Dispatch-claim plan**:
  - L1 SCAFFOLD landing (this commit batch): mark `impl_complete` + `three_clean_review` + `memory_entry` + `deploy_runbook` gates after this landing
  - L2 SCAFFOLD → SMOKE-VALIDATED: smoke dispatch (~$1 Modal T4 100ep) via `tools/operator_authorize.py --recipe substrate_c1_world_model_foveation_modal_t4_smoke_dispatch`; mark `real_archive_empirical` if smoke produces a valid C1WMFV1 archive
  - L3 PRODUCTION (NOT yet): requires Phase 3 council + multi-stage full dispatch ($30-50 over 3-4 weeks); mark `contest_cuda` only after a verified `[contest-CUDA]` auth-eval anchor

### 2. Source evidence + score-lowering hypothesis

**Source evidence**:
- Z1 MDL ablation (`feedback_z1_mdl_ablation_landed_20260514.md`): A1 archive 178KB has MDL density **99.29%** WITHIN-CLASS; PR106 r2 187KB has **97.21%** density. Both archives MDL-saturated within HNeRV-class. **Sub-0.10 needs CLASS SHIFT, not better encoding.**
- Carmack's shower thought (`feedback_zen_floor_field_medal_grade_council_landed_20260514.md` Round 1 eureka #4): camera 1164×874 → scorer 384×512 resize matrix is rank-deficient. Only NEAR-FOV pixels carry full-detail bits.
- Time-Traveler L5 staircase framing (`feedback_long_term_multi_year_campaigns_landed_20260514.md`): predictive-receiver paradigm Step 4 = world-model + foveation; predicted asymptote ~0.03 at Step 5.
- Atick-Redlich 1990 (Neural Computation 2): cortical magnification factor; 2-10x bit savings on periphery for stationary-ergodic vision.
- Ha & Schmidhuber 2018 (arXiv:1803.10122): world models compress temporal redundancy; per-frame residual surprise < 100 B/frame for predictable dynamics.
- Hafner DreamerV3 2023 (arXiv:2301.04104): mastered diverse domains via RSSM + GRU recurrence; the canonical modern world-model architecture.

**Score-lowering hypothesis**:

A1 baseline 0.1928 (verified [contest-CUDA] anchor `87ec7ca5...492b5`). C1 substitutes:

1. **World-model recurrent latent (z_t = GRU(z_{t-1}))** for HNeRV's per-frame independent decoder. Predicted byte savings: per-frame residual surprise ~50 B/frame * 1200 = 60 KB vs HNeRV's ~120 B/frame * 1200 = 144 KB. **Rate Δ ≈ -0.056 from residual alone** if asymptote holds.
2. **Foveation matched to ego-motion vanishing point** for uniform bit allocation. Predicted scorer-side Δ: PoseNet sees both frames YUV6 → forward FOV matters more; SegNet sees frame_1 only at 384x512 → 2D resize rank ~24K of 192K. **Distortion Δ ≈ -0.005 to -0.010** from foveation alone.

Total predicted ΔS = -0.04 to -0.06 vs A1 0.1928 → predicted band **[0.13, 0.16]** `[mathematical-derivation; first-principles-bound]`.

### 3. Timing-smoke command (seconds/epoch measurement)

```bash
# Free macOS CPU smoke (validates plumbing; ~3 sec wall-clock)
.venv/bin/python experiments/train_substrate_c1_world_model_foveation.py \
    --output-dir .omx/tmp/c1_smoke_cpu_$(date +%Y%m%dT%H%M%SZ) \
    --device cpu --smoke --epochs 3

# Modal T4 timing-smoke (~$0.30, 100 epochs)
OPERATOR_AUTHORIZE_CONFIRMED_VIA_SESSION_DIRECTIVE=1 \
OPERATOR_AUTHORIZE_SESSION_BUDGET_USD=1.00 \
C1_WORLD_MODEL_FOVEATION_SMOKE_ONLY=1 \
.venv/bin/bash scripts/operator_authorize_substrate_c1_world_model_foveation_modal_t4_smoke_dispatch.sh
```

**Expected smoke output**:
- macOS CPU: ~3-5 sec total wall-clock; archive ~12 KB; sha256 byte-identical across runs.
- Modal T4: ~5-15 min total wall-clock; archive ~120-160 KB; per-epoch ~3-9 sec.

### 4. Full-run command (resumable checkpoints + harvest)

```bash
# Phase 3 BLOCKED at L1: --full-main raises NotImplementedError.
# Unlock requires inner-quintet council sign-off (see §7 below).
# Once unlocked:

OPERATOR_AUTHORIZE_CONFIRMED_VIA_SESSION_DIRECTIVE=1 \
OPERATOR_AUTHORIZE_SESSION_BUDGET_USD=50.00 \
C1_WORLD_MODEL_FOVEATION_EPOCHS=2000 \
.venv/bin/bash scripts/operator_authorize_substrate_c1_world_model_foveation_modal_t4_smoke_dispatch.sh

# Harvest within 24h (per CLAUDE.md "Modal .spawn() HARVEST OR LOSE")
.venv/bin/python tools/harvest_modal_calls.py
```

### 5. Live provider rate / cost model

| Provider | GPU | $/hr | Stage cost | Total (6-8 stages) |
|----------|-----|------|------------|---------------------|
| Modal T4 | T4 | $0.59 | $1-3/stage | $6-24 |
| Modal A10G | A10G | $1.10 | $3-5/stage | $18-40 |
| Modal A100 | A100 | $4.00 | $5-15/stage | $30-120 (council-overrideable cap $50) |
| Vast.ai 4090 | 4090 | $0.25 | $1-3/stage | $6-24 (best $/wall-clock) |

**Recommended ladder**: Stage 1 (world-model alone) Modal T4 100ep ~$1 → Stage 2 (foveation alone) T4 200ep ~$2 → Stage 3 (combined fine-tune) Modal A100 500ep ~$10 → Stage 4 (residual codec) T4 200ep ~$2 → Stage 5 (archive byte sweep) T4 100ep ~$1 → Stage 6 (full Lagrangian convergence) A100 1000ep ~$20 → optional Stage 7-8 ablations $5-10.

**Total budget**: $30-50 (council-overrideable cap).

### 6. Byte-closed archive / export / inflate plan

**Archive grammar**: `C1WMFV1` monolithic single-file `0.bin` (HNeRV parity discipline L3 substrate-engineering scope).

**Parser-section manifest** (declared in `src/tac/substrates/c1_world_model_foveation/archive.py:80-89`):
- MAGIC(4) + VERSION(1) + NUM_PAIRS(2) + RECURRENCE_MODE(1) + FOVEATION_STRATEGY(1)
- LATENT_DIM(2) + OUTPUT_H(2) + OUTPUT_W(2)
- WM_LEN(4) + DECODER_LEN(4) + ZINIT_LEN(4) + FOV_META_LEN(4) + RESIDUAL_LEN(4) + META_LEN(4)
- Total header = 39 bytes
- 6 sections: WM_BLOB / DECODER_BLOB / ZINIT_BLOB / FOV_META_BLOB / RESIDUAL_BLOB / META_BLOB

**Section byte targets**:
| Section | Raw | After FP16 + brotli | After FP4 (future) |
|---------|-----|---------------------|---------------------|
| WM_BLOB (GRU) | ~25 KB | ~15 KB | ~6 KB |
| WM_BLOB (LSTM) | ~32 KB | ~18 KB | ~8 KB |
| DECODER_BLOB | ~50 KB | ~30 KB | ~12 KB |
| ZINIT_BLOB | ~256 B | ~256 B | ~256 B |
| FOV_META_BLOB | ~200 B (JSON) / ~1 KB (learned) | ~200 B / ~1 KB | unchanged |
| RESIDUAL_BLOB | ~60-120 KB | ~60-120 KB (already brotli'd) | unchanged |
| META_BLOB | ~300 B | ~200 B | unchanged |
| **TOTAL (GRU + EGO_MOTION_RADIAL)** | **~135 KB FP16** | **~105 KB** | **~80 KB FP4** |
| **TOTAL (LSTM + LEARNED_PER_PIXEL)** | **~165 KB FP16** | **~150 KB** | **~110 KB FP4** |

**Inflate.py**: 213 LOC in `src/tac/substrates/c1_world_model_foveation/inflate.py` -- substrate-engineering waiver to HNeRV parity L4 (≤ 200 LOC default with rationale ≤ 250). Loads C1WMFV1 archive, rehydrates world-model + decoder + foveation modules, unrolls for `2 * num_pairs` steps, decodes RGB per frame, adds residual surprise back, writes contest `.raw` output.

**Runtime dep closure**: torch + brotli only (HNeRV parity L4 ≤ 2 deps).

**Export contract**: `pack_archive(...)` in `archive.py` is deterministic byte-identical given the same input state_dicts (verified by test `test_pack_is_deterministic_byte_identical`).

### 7. Stop / continue thresholds (smoke + mid-stage + export + exact eval)

| Stage | Threshold | Action |
|-------|-----------|--------|
| Smoke (CPU, 3 epochs) | final_loss < 0.5 | continue to Modal smoke |
| Smoke (Modal T4, 100 epochs) | rc=0 + valid archive + plumbing end-to-end | continue to Phase 3 council |
| Phase 3 council vote | ≥ 4/5 inner-quintet PROCEED | continue to Stage 1 of multi-stage |
| Stage 1 (world-model alone) | residual_l2 asymptote < 0.01 | continue to Stage 2 |
| Stage 2 (foveation alone) | foveation_l1 < 0.5 (concentration > 50%) | continue to Stage 3 |
| Stage 3 (combined) | proxy_loss < 0.04 (training-loss-only proxy) | continue to Stage 4 |
| Stage 4 (residual codec) | residual_blob bytes ≤ 80 KB | continue to Stage 5 |
| Stage 5 (archive byte sweep) | total bytes in [100, 180] KB | continue to Stage 6 |
| Stage 6 (full Lagrangian) | auth_eval [contest-CUDA] score in predicted band [0.13, 0.16] | promote to L2/L3 |
| Stage 6 fallback | score [0.17, 0.20] | DEFERRED-pending-research (no kill) |
| Stage 6 worst-case | score > 0.22 | DEFERRED-pending-research-with-foveation-rescope (no kill) |

**KILL is LAST RESORT** per CLAUDE.md non-negotiable: every Stage failure produces a DEFERRED memo, not a KILL.

## Phase 3 unlock conditions (operator-routable decisions)

The C1 multi-stage training schedule ($30-50 over 3-4 weeks) is gated by the trainer's `_full_main` `NotImplementedError`. The 5 unlock conditions are:

1. **Inner-quintet council unanimous PROCEED** (Shannon LEAD + Dykstra CO-LEAD + Yousfi + Fridrich + Contrarian, plus grand-council advisory from Hassabis on world-model design + Carmack on foveation engineering shortcuts). The council file path is `.omx/research/council_c1_world_model_foveation_phase_3_unlock_<YYYYMMDD>.md` (to be created by the council-running subagent).
2. **Z1 MDL ablation confirms C1 is across-class** (density < 0.90 on a C1 archive after Stage 1 world-model training). Run `tools/mdl_scorer_conditional_ablation.py --archive <c1_archive>` after Stage 1 to verify the class shift hypothesis empirically.
3. **Smoke dispatch validates substrate plumbing end-to-end** (this is the L2 gate). Need rc=0 + valid C1WMFV1 archive + inflate roundtrip preserves the raw output byte-count.
4. **Sister long-term campaign C5 (cooperative-receiver loss) or C6 (MDL-IBPS) anchors land** for stacking analysis. C1+C5 or C1+C6 compound could push toward [0.10, 0.13].
5. **Operator-routed budget approval** via `tools/operator_authorize.py` with explicit `OPERATOR_AUTHORIZE_SESSION_BUDGET_USD=50.00` (or council-approved larger cap).

## 6-hook wire-in (Catalog #125)

1. **Sensitivity-map contribution**: foveation map `M_t` IS the per-pixel importance signal (encoded in the substrate's `FoveationMapModule.map(z_t)` output). World-model latent dimension `z_t` is the per-frame importance signal (smaller latent norm = lower-importance frame). After Stage 1 the substrate registers `c1_world_model_latent_sensitivity` + `c1_foveation_pixel_sensitivity` into the unified sensitivity map.
2. **Pareto constraint**: adds `foveation_concentration >= 0.30` AND `world_model_residual_entropy <= 0.01` constraints to the Pareto feasible set. These constraints intersect with A1's `archive_bytes <= 180000` constraint to define the C1 feasibility region.
3. **Bit-allocator hook**: `FoveationMapModule.map(z_t)` is the bit-allocator hook -- per-pixel bit cost is modulated by `(1 + foveation_attenuation * (1 - M_t(x, y)))`. The archive build path consumes this map to allocate residual bytes per pixel.
4. **Cathedral autopilot dispatch hook**: register C1 row in v2 ranker with `lane_class=substrate_engineering` + `literature_anchor=Ha-Schmidhuber-2018+Atick-Redlich-1990` (combined -0.03 + -0.02 = -0.05 across-class reward for stacked class-shift) + `mdl_density=<measured after Stage 1>`. The autopilot will rank C1 ahead of within-class sidecar bolt-ons even at modest predicted ΔS per the Z1 revision logic.
5. **Continual-learning posterior update**: triggered on every Stage's empirical anchor via `tac.continual_learning.posterior_update_locked` (Catalog #128 atomic fcntl). Each Stage's `[contest-CUDA]` anchor (or `[smoke-no-scorer]` proxy) feeds the cost-band posterior + the across-class predicted-band posterior.
6. **Probe-disambiguator**: TWO probes ship per the design-tension memo:
   - `tools/probe_c1_world_model_vs_independent_frames_disambiguator.py` -- arbitrates GRU vs LSTM vs Transformer vs independent-frame baseline (HNeRV-class control)
   - `tools/probe_c1_foveation_vs_uniform_quantization_disambiguator.py` -- arbitrates UNIFORM vs EGO_MOTION_RADIAL vs LEARNED_PER_PIXEL

Both probes run on CPU in seconds; verdicts are `[proxy]`-tagged per CLAUDE.md axis-discipline.

## Cross-references

- `feedback_long_term_multi_year_campaigns_landed_20260514.md` -- the 7-campaign roadmap that anchors C1
- `feedback_grand_council_maximize_value_landed_20260514.md` -- Carmack's shower thought on resize-rank + Time-Traveler's staircase framing
- `feedback_zen_floor_field_medal_grade_council_landed_20260514.md` -- across-class predictions; C1 ranked in [0.13, 0.16] band
- `feedback_z1_mdl_ablation_landed_20260514.md` -- the empirical anchor for the within-class trap (99.29% MDL density)
- `feedback_d4_wyner_ziv_frame_0_landed_20260514.md` -- sister across-class substrate (frame-0 derivation; orthogonal mechanism)
- `feedback_time_traveler_full_cpu_mode_landed_20260514.md` -- sister predictive-receiver L5 autonomy lane
- `src/tac/substrates/c1_world_model_foveation/__init__.py` -- substrate docstring with all 8 Catalog #124 fields declared
- `.omx/operator_authorize_recipes/substrate_c1_world_model_foveation_modal_t4_smoke_dispatch.yaml` -- canonical recipe
- `scripts/remote_lane_substrate_c1_world_model_foveation.sh` -- remote driver
- `scripts/operator_authorize_substrate_c1_world_model_foveation_modal_t4_smoke_dispatch.sh` -- smoke-before-full wrapper
- `tools/probe_c1_world_model_vs_independent_frames_disambiguator.py` -- world-model recurrence probe
- `tools/probe_c1_foveation_vs_uniform_quantization_disambiguator.py` -- foveation strategy probe

## Tagging

`research_only=true` for all planning text. Score claims are `[smoke-no-scorer]` (CPU smoke + Modal T4 smoke) or `[contest-CUDA]` (only after full Stage 6 with valid auth-eval anchor). No KILL verdicts in this ledger; all Stage failures produce `DEFERRED-pending-research` memos per CLAUDE.md non-negotiable.
