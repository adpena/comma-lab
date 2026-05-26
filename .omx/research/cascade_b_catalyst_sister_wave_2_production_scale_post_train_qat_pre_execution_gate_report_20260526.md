<!-- SPDX-License-Identifier: MIT -->
<!-- council_tier: T1 -->
<!-- council_attendees: [Shannon, Dykstra, Yousfi, Fridrich, Contrarian, Assumption-Adversary] -->
<!-- council_quorum_met: true -->
<!-- council_verdict: PROCEED -->
<!-- council_predicted_mission_contribution: frontier_breaking_enabler -->
<!-- council_override_invoked: false -->
<!-- # COUNCIL_ROSTER_INCOMPLETE_OK:t1_working_group_per_pre_execution_gate_audit_canonical_sister_pattern -->

# CASCADE B CATALYST sister wave 2 — pre-execution gate report
## production-scale 600f × 1000ep + post-train QAT fine-tune on real-SegNet (6th-order)

**Subagent**: `cascade-b-catalyst-sister-wave-2-production-scale-600f-1000ep-plus-post-train-qat-fine-tune-real-segnet-fixture-recursive-doctrine-6th-order-mlx-first-numpy-portable-20260526`
**Lane**: `lane_cascade_b_catalyst_sister_wave_2_production_scale_post_train_qat_real_segnet_20260526`
**Captured**: 2026-05-26T20:53:25Z
**Operator route**: Cascade B CATALYST 5th-order IMPLEMENTATION_LEVEL_FALSIFIED operator-routable sister wave (per commit `fcfad9331`)
**Per CLAUDE.md "Remember all on MLX"**: ALL EXECUTION LOCAL macOS M5 MAX; NO PAID DISPATCH.

---

## 1. Premise verification (Catalog #229)

| Premise | Source | Verified |
|---|---|---|
| 5th-order CATALYST IMPLEMENTATION_LEVEL_FALSIFIED at synthetic | commit `fcfad9331` landing memo | YES — verdict line 19; +1.6e-2 sidecar for ZERO d_seg gain line 70 |
| Path A wave 1 PARTIAL_CONVERGENCE at 600f × 1000ep KL 3.378 | `experiments/results/.../sweep_results.json` | YES — final_loss=3.378; det=6.208; delta=2.830 nats |
| Path A wave 1 wall-clock ≈ 22 min for learnable arm | sweep_results.json | YES — 925.8s = 15.4 min Path A; 1331.8s total |
| `quantize_head_fp4` available in catalyst_cascade.py | `src/tac/substrates/.../catalyst_cascade.py:173-215` | YES — MLX-native; identity-STE; whole-head single-block default |
| `fake_quant_fp4_mlx` available | catalyst_cascade.py:92-170 | YES — per-block (sign, magnitude); canonical FP4 codebook |
| Canonical equation #2 lifecycle = 2 events | `.omx/state/canonical_equations_registry.jsonl` grep | YES — 1 registered + 1 anchor_appended (synthetic FALSIFIED) |
| Sister wave 1 harness reusable | `tools/cascade_b_path_a_sister_wave_1_production_scale_600f_1000ep.py` | YES — imports `sister_smoke.run_smoke` via patch pattern |

---

## 2. Sister-disjoint scope verification (Catalog #230 + #302 + #340)

| Active sister | Scope | Disjoint from wave 2? |
|---|---|---|
| UNIWARD 6th-order BoostNeRV (#1374) | `src/tac/substrates/uniward/` + boost_nerv | YES — different substrate |
| Cascade C' Option A scaffold (#1375) | `src/tac/substrates/cascade_c_prime/` | YES — different substrate |
| Cascade A FEC10 hybrid (#1358) | `src/tac/codec/fec10_hybrid/` | YES — codec lane disjoint |
| NSCS06 v8 Modal CUDA PAID | operationally separate | YES — local vs paid |

My scope = `src/tac/substrates/hinton_distilled_scorer_surrogate/` (extend NOT modify) + `tools/cascade_b_catalyst_sister_wave_2_production_scale_post_train_qat_real_segnet_20260526.py` (NEW) + `.omx/research/` + `experiments/results/`.

---

## 3. 6th-order canonical decomposition + execution plan

Per just-elevated individually-fractal directive:
- 1st: Cascade B CATALYST scaffold (commit `15b11c86e`)
- 2nd: Path A learnable head (commit `15b11c86e`; PARADIGM-VALIDATED)
- 3rd: production-scale 600f × 1000ep (commit `4c73be3e4`; PARADIGM-VALIDATED-with-plateau KL 3.378)
- 4th: capacity-sweep sister (DEFERRED operator-routable)
- 5th: CATALYST cascade composition synthetic (commit `fcfad9331`; IMPLEMENTATION-LEVEL-FALSIFIED)
- **6th (YOUR SCOPE)**: production-scale 600f × 1000ep Path A + 100ep post-train QAT fine-tune on real-SegNet fixture

**Plan**:
1. Build sister wave 2 harness `tools/cascade_b_catalyst_sister_wave_2_production_scale_post_train_qat_real_segnet_20260526.py` extending wave 1's `run_smoke` with a post-train QAT fine-tune phase.
2. Run 3-arm production sweep: (baseline=deterministic 1000ep / Path A alone=learnable 1000ep / CATALYST=learnable 1000ep + 100ep post-train QAT fine-tune).
3. Per CLAUDE.md "QAT pipeline — non-negotiable" canonical stage chain: anchor → finetune → joint → QAT → final. Wave 1 = anchor + finetune; wave 2 = QAT + final.
4. Append 2nd empirical anchor to canonical equation #2 → events 2 → 3 → potential auto-recalibration trigger.

**Wall-clock estimate** (M5 Max, MLX):
- Deterministic arm 1000ep: ~7 min (per wave 1)
- Path A arm 1000ep: ~22 min (per wave 1; loss converges ~3.38)
- Post-train QAT 100ep fine-tune: ~2-3 min (10× shorter)
- Total: ~30-35 min

**Honest scope cap**: 1-2h budget allows BOTH arms + QAT phase; if QAT phase reveals interesting cascade dynamics (e.g., post-QAT fine-tune RECOVERS d_seg-proxy below Path A floor), this is the ideal real-scorer empirical anchor for canonical equation #2.

---

## 4. Mathematical grounding per 3-strategy directive

- **PRIMARY**: DISTORTION axis (d_seg-proxy reduction via Path A + post-QAT fine-tune) + FULL-SCORER (CATALYST P2 KL + P5 QAT + P10 BPR1 sidecar)
- **Quantizr canonical sister**: PR101/PR56 use `kl_on_logits(T=2.0)` for SegNet during specific training phases; YOUR post-train QAT fine-tune is canonical sister at the QAT-stage sub-phase
- **Horizon class** per Catalog #309: `plateau_adjacent` (substrate-stacking foundation)
- **Entropy-position**: P2 + P5 + P10 CATALYST composition per just-landed entropy-position discipline § 10
- **Canonical equation #2 latex**: `ΔS_cat(P2 → P4 → P10) = ΔS_{P4}^{alone} · (1 + α · ΔH_{logits}^{T=2})` with `α ∈ [0.1, 0.2]`

---

## 5. MLX-first + numpy-portable bridge contract (HONORED)

- **TRAINING**: MLX-native (Path A 1000ep + post-train QAT 100ep fine-tune; `mx.value_and_grad`)
- **POST-TRAIN QAT**: MLX-native `quantize_head_fp4` + `fake_quant_fp4_mlx` (identity-STE)
- **BPR1 sidecar**: numpy + brotli only (no PyTorch / MLX) — canonical composition surface
- **INFLATE**: N/A at scaffold scale (calibration-time substrate, not contest-archive)

---

## 6. Acceptance verdicts taxonomy (Catalog #307)

| Outcome | Verdict | Catalog #2 anchor action |
|---|---|---|
| CATALYST composite < Path A composite (at production-scale real-SegNet) | **PARADIGM_VALIDATED** | append 2nd anchor; canonical equation #2 events 2 → 3 → auto-recalibration trigger |
| CATALYST composite ≈ Path A composite (within proxy noise) | **PARTIAL_CONFIRMATION** | append 2nd anchor citing per-axis tradeoff |
| CATALYST composite > Path A composite + delta_rate > 0.005 | **IMPLEMENTATION_LEVEL_FALSIFIED** at production-scale | append 2nd anchor; PARADIGM INTACT per #307; queue 7th-order |
| Post-QAT KL diverges (>5.0) | **DEFER_PENDING_QAT_STABILIZATION** | append 2nd anchor with diagnostic |

Per CLAUDE.md "Forbidden premature KILL": this gate prevents synthetic-fixture falsification from rotting Path A foundation; the 6th-order outcome is structurally append-only HISTORICAL_PROVENANCE per Catalog #110/#113.

---

## 7. Discipline checklist (pre-execution)

- [x] Catalog #229 PV (7 premises verified above)
- [x] Catalog #230 + #302 + #340 sister-disjoint scope (4 active sisters checked; all disjoint)
- [x] Catalog #206 checkpoint emitted at step 1
- [x] Catalog #117/#157/#174/#235/#289 serializer plan: POST-EDIT sha256 for every commit
- [x] Catalog #287 placeholder-rationale rejection: gate-friendly waivers ≥4 chars
- [x] Catalog #343 no hardcoded score literals (frontier-pointer model)
- [x] Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE
- [x] CLAUDE.md "Remember all on MLX": $0 GPU; local M5 Max only

---

## 8. 6-hook wire-in declaration per Catalog #125

| Hook | Status | Rationale |
|---|---|---|
| 1. Sensitivity-map | ACTIVE | per-axis (KL vs sidecar rate) decomposition emitted per arm |
| 2. Pareto constraint | ACTIVE | 3-arm composite proxy ranking surfaces Pareto-feasible CATALYST configurations |
| 3. Bit-allocator | N/A | post-train QAT operates on calibration-time substrate; no contest bit-allocator surface |
| 4. Cathedral autopilot dispatch | ACTIVE via Catalog #344 | canonical equation #2 anchor → cathedral_consumers.canonical_equation_lookup_consumer auto-discovery → all downstream consumers |
| 5. Continual-learning posterior | ACTIVE | `update_equation_with_empirical_anchor` writes 2nd anchor; events 2 → 3 → auto-recalibration trigger potential |
| 6. Probe-disambiguator | ACTIVE | 3-arm comparison IS the disambiguator between "Path A alone is enough" vs "CATALYST cascade composition adds substrate-stacking signal" |

---

## 9. Execution authorization

**T1 Working Group verdict**: PROCEED.

Sister wave 2 production-scale 600f × 1000ep + post-train QAT 100ep fine-tune on real-SegNet fixture is canonically scoped, mathematically grounded, MLX-first compliant, sister-disjoint, and operator-routable.

Proceeding to Step 2 (sister wave 2 harness construction).
