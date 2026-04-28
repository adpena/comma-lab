# Lane G v3 → sub-Quantizr Stacking Synthesis (skunkworks council, 2026-04-28)

**Frontier**: Lane G v3 = **1.05 [contest-CUDA]** (archive 694,074 bytes; SHA `9b20bdfca246d8e32cc19da966c84cdae7e34f6b247161d107ec43cb9ef6870b`).
**Target**: sub-0.33 (Quantizr leader). **Gap**: 0.72 points.
**Budget**: ~$300 Vast.ai. ~5 days to deadline.

Council convened: Yousfi, Fridrich, Hotz, Quantizr, Contrarian — and the silent observer (the road).
Charter: NON-CONSERVATIVE. The burden of proof is on **not** trying something.

---

## §1. Decomposing the 0.72-point gap

### 1.1 Lane G v3 (verified, contest-CUDA, 2026-04-28T11:36:55Z)

```
PoseNet dist:  0.003455      → sqrt(10×pose) = 0.186     (17.7% of score)
SegNet  dist:  0.004008      → 100×seg       = 0.401     (38.2% of score)
Rate (unscaled): 0.018486    → 25×rate       = 0.462     (44.0% of score)
Archive bytes:  694,074  / 37,545,489  = 1.847%  rate-fraction
TOTAL = 1.0489 → 1.05
```

Sources: `experiments/results/lane_g_v3_landed/contest_auth_eval.json`,
`project_lane_g_v3_landed_1_05_20260428`.

### 1.2 Quantizr (estimated from competitive intel)

Sources: `project_quantizr_full_intel_20260421`, `project_quantizr_definitive_binary_analysis`.

```
Archive: 299,970 bytes / 37,545,489     → 25×rate ≈ 0.200   (60.6% of his 0.33)
Distortion budget remaining:                              ≈ 0.130
He says "sub 0.30 just by sweeping conv dims"  (already-stopped optimizer)
```

If we model Quantizr's distortion split proportionally to Lane G v3's, his
likely component values are:

```
seg ≈ 0.0009  → 100×seg ≈ 0.090
pose ≈ 0.0016 → sqrt(10×pose) ≈ 0.040
total dist ≈ 0.130   ← consistent with archive math
```

### 1.3 Wedge attribution

| Component | Lane G v3 | Quantizr (est.) | Gap | % of 0.72 |
|---|---:|---:|---:|---:|
| Rate     | 0.462 | 0.200 | **0.262** | **36.4%** |
| SegNet   | 0.401 | 0.090 | **0.311** | **43.2%** |
| PoseNet  | 0.186 | 0.040 | **0.146** | **20.4%** |
| **Total** | **1.05** | **0.33** | **0.72** | 100% |

**Council reading**: SegNet is the single largest wedge (43%); rate is second
(36%); PoseNet third (20%). The Contrarian's EUREKA #5 from
`project_council_eurekas_driving_geometry_20260428` is now numerically grounded:
**at the Lane G v3 floor, SegNet linear scaling dominates everything else** —
a 4× SegNet improvement saves 0.301; a 4× PoseNet improvement saves only 0.140.

Per-byte sensitivities (for stack design):

- **1KB shaved off archive** = `−25 × 1024 / 37545489 = −0.000683` rate.
  Saving 400KB (matching Quantizr's 300KB archive) = `−0.273`.
- **SegNet 0.004 → 0.001** (4×) = `−0.300`.
- **PoseNet 0.0035 → 0.0004** (Quantizr level) = `−0.146`.

The math says: **rate-attack and SegNet-attack are the two big rocks; PoseNet
is the tie-breaker.** Lane G v3 has ALREADY pushed PoseNet very low — further
PoseNet investment has diminishing returns.

### 1.4 The corollary nobody articulated yet

Lane S/W/Ω have been designed primarily to PROTECT PoseNet-critical channels
(FiLM, motion, head). After EUREKA #5 + the Lane G v3 wedge math, this is
**over-provisioned**. The protected-layer list must be **re-scoped for SegNet
pathways** (per-class output conv, U-shape skip-connections feeding the
classifier). Concretely: see Lane SG below.

---

## §2. Existing-lane catalog re-anchored to 1.05 frontier

The taxonomy in `project_lane_taxonomy_stacking_strategy_20260427` and the
TIER 3 catalog in `project_outstanding_work_and_stacks_20260428` were anchored
on **Lane A = 1.15**. Every predicted band must now be re-anchored to
**Lane G v3 = 1.05** (and to the Lane G v3 component breakdown which is
sharper on PoseNet, identical on SegNet, identical on rate).

### 2.1 Re-anchoring rule

A lane that historically predicted "[0.85, 1.05] from 1.15" was claiming
`Δ ∈ [−0.30, −0.10]`. Lane G v3's PoseNet is already 31% better than Lane A's,
so any pose-targeting lane keeps its `Δ`; any seg/rate lane keeps its `Δ` (the
distortion+rate floors are independent of pose). Therefore the re-anchored
band is `[1.05 + Δlow, 1.05 + Δhigh]`.

The exception is **lanes whose mechanism is pose TTO or pose distillation**:
Lane G v3 already burned that fuel. Those lanes' `Δ_pose` shrinks (lower
ceiling); their re-anchored upper bound moves DOWN by ~0.05.

### 2.2 Re-anchored TIER-2/TIER-3 quantization lanes (renderer.bin attack)

| Lane | Re-anchored band | Δ vs 1.05 | Confidence | Status | Cost |
|---|---|---:|:---:|---|---:|
| **Lane S** (per-channel SC) | [0.85, 1.05] | −0.20…0.00 | MED | Code shipped; **motion.head shape bug `feedback_lane_s_motion_head_shape_mismatch_20260428` blocks deploy** | $0.50 |
| **Lane W** (Lane S + hard-pair weighting) | [0.80, 1.00] | −0.25…−0.05 | MED-HIGH | In flight (Iceland inst 35739770) | $0.50 |
| **Lane WC** (Lane W + Cosmos Curator soft-DTW) | [0.75, 0.98] | −0.30…−0.07 | MED | Code shipped, never deployed | $3 |
| **Lane Ω-V2** (Lagrangian per-element bits) | [0.70, 0.95] | −0.35…−0.10 | MED | Survived 7 codex rounds, never deployed | $1.50 |
| **Lane F-V5** (hardware FP8 via Cosmos RL recipe) | [0.95, 1.15] | −0.10…+0.10 | MED | Code shipped, never deployed; **Lane F-V1..V4 results were SIMULATED FP4 (4090 = CC 8.9, no FP4 hardware) per `project_cosmos_deep_dive_addendum_20260428`** | $2 |
| Lane F-V4 (mixed FP4 in flight) | [1.20, 1.80] | +0.15…+0.75 | LOW | In flight | sunk |

**Council read** (Hotz): Lane Ω-V2 is the highest-EV NEVER-DEPLOYED rate
attack. It's been adversarially reviewed through 7 codex rounds. Dispatching
it is overdue.

**Council read** (Quantizr): Lane WC is the natural successor to Lane W
because it replaces Lane W's circular "weight by own loss" with an INDEPENDENT
typicality signal (Cosmos Curator soft-DTW). Run them as a controlled pair
(Lane W vs Lane WC, same renderer init) → Lane T2-DUAL methodology.

### 2.3 Re-anchored architecture lanes (renderer replacement)

| Lane | Re-anchored band | Δ vs 1.05 | Confidence | Status | Cost |
|---|---|---:|:---:|---|---:|
| **Lane V** (Quantizr replica 88K + half-frame from epoch 0) | [0.40, 1.00] | −0.65…−0.05 | MED | In flight (inst 35733832) ~12h | $4 |
| **Lane K** (DSConv 88K from scratch) | [0.80, 1.05] | −0.25…0.00 | MED | In flight (Denmark inst 35739771) ~12h | $3 |
| **Lane I** (Cool-Chic CCh1 mask codec — distinct from CCh1 renderer) | [0.85, 1.20] | −0.20…+0.15 | LOW-MED | In flight (inst 35733831) | $0.50 |
| **Lane GH** (Ghost-modules half-params) | [0.95, 1.20] | −0.10…+0.15 | LOW-MED | Code shipped, never deployed | $3 |
| **Lane SZ** Phase 2 (szabolcs no-masks Gaussian LUT + 1.017 b/w) | [0.20, 0.45] | −0.85…−0.60 | LOW | Code shipped, never deployed; MOONSHOT | $4 |
| **Lane V-V2** (Lane V + annealed half-frame) | [0.40, 0.95] | −0.65…−0.10 | MED-LOW | Code shipped | $5 |

### 2.4 Re-anchored pose lanes

| Lane | Re-anchored band | Δ vs 1.05 | Confidence | Status | Cost |
|---|---|---:|:---:|---|---:|
| **Lane LR** (LoRA rank-1 pose) | [1.04, 1.05] | −0.01…0.00 | HIGH | Code shipped | $0.20 |
| **Lane LR-V2** (learnable-rank LoRA) | [1.03, 1.05] | −0.02…0.00 | MED-HIGH | Code shipped | $0.30 |
| **Lane LM-V2** (endpoint-tracking lane-mark zero-cost) | [1.00, 1.05] | −0.05…0.00 | MED | Code shipped (V1 corr=0.017 → V2 should hit 0.30+) | $0.50 |
| **Lane GP** (Gaussian-process pose: dim 0 polynomial + 5 GP perturbations) | [0.99, 1.05] | −0.06…0.00 | MED-HIGH | NEW from EUREKA #4 | $0.30 |
| **Lane GE** (pure geodesic: pose = (∫v dt, 0,0,0,0,0) as ~10-coef poly) | [1.00, 1.07] | −0.05…+0.02 | MED | NEW from EUREKA #4 | $0.30 |
| **Lane FL** (RAFT-derived poses, no learning) | [1.02, 1.07] | −0.03…+0.02 | MED | NEW from EUREKA #7; uses RAFT we already have | $0.20 |
| **Lane MOS** (Lane M-V3 + Lane OS) | [0.98, 1.05] | −0.07…0.00 | MED | Code shipped (user-suggested 2026-04-28) | $0.80 |
| **Lane M-V3** (PoseNet-embedding distillation) | [1.00, 1.05] | −0.05…0.00 | MED | In flight subagent | sunk |
| **Lane M-V2** (radial-zoom proper baseline-padded) | [1.05, 1.18] | 0.00…+0.13 | LOW | In flight (inst 35736027) | $0.50 |

**Important re-anchor finding**: pose lanes' upper bound now SATURATES at 1.05
(can't make pose worse than Lane G v3 already is on the dominant TTO axis).
Their wedge contribution is small (0.146 max). **They are stack PARTICIPANTS,
not stack drivers.**

### 2.5 Re-anchored mask lanes (the SegNet+rate dual attack)

| Lane | Re-anchored band | Δ vs 1.05 | Confidence | Status | Cost |
|---|---|---:|:---:|---|---:|
| **Lane SI-V2** (Lagrangian-learnable saliency CRF) | [0.97, 1.04] | −0.08…−0.01 | MED-HIGH | Code shipped | $0.50 |
| **Lane SI-V3** (Lane SI-V2 + Fridrich UNIWARD texture σ²) | [0.92, 1.03] | −0.13…−0.02 | MED | NEW from EUREKA #8; not yet coded | $0.60 |
| **Lane I-B** (Cool-Chic mask codec replaces masks.mkv) | [0.85, 1.10] | −0.20…+0.05 | LOW-MED | Distinct from Lane I (renderer); not yet coded | $4 |
| **Lane MAE** (75% mask reduction, in-painting) | [0.85, 1.10] | −0.20…+0.05 | LOW | Code shipped, deferred | $5 |
| **Lane MAE-V** (joint half-frame from epoch 0) | [0.75, 1.05] | −0.30…0.00 | MED | Code shipped | $4 |
| **Lane HFM** (Telescope foveation on masks) | [0.93, 1.13] | −0.12…+0.08 | LOW | Code shipped | $4 |
| **Lane H-CRF56** (higher CRF baseline) | [1.02, 1.10] | −0.03…+0.05 | HIGH | Validated previously | $0.10 |

### 2.6 Re-anchored distortion-improvement / proxy-auth-gap lanes

| Lane | Re-anchored band | Δ vs 1.05 | Confidence | Status | Cost |
|---|---|---:|:---:|---|---:|
| **Lane SAUG** (Lyra self-augmentation, GT perturbation) | [0.65, 0.95] | −0.40…−0.10 | MED | Predicted before Lane G v3 landed; band may be unchanged because mechanism attacks proxy-auth gap NOT distortion floor | $5 |
| **Lane SAUG-V2** (Lyra SAUG + Cosmos HighSigmaStrategy) | [0.60, 0.90] | −0.45…−0.15 | MED | Code shipped, never deployed | $4 |
| **Lane HF** (Telescope hyperbolic foveation on frames) | [0.80, 1.05] | −0.25…0.00 | MED | Code shipped, never deployed | $4 |
| **Lane T2-XPRED** (x-prediction + v-loss replacement) | [1.00, 1.05] | −0.05…0.00 | MED-HIGH | Code shipped via t2_xpred profile | $0.50 |
| **Lane T2-MASK** (input-feature masking 15% in last 40%) | [0.95, 1.04] | −0.10…−0.01 | MED | Code shipped via t2_mask profile | $1.50 |
| **Lane T2-RATIO** (seg-weight sweep above 100) | [0.95, 1.10] | −0.10…+0.05 | LOW | Code in flight, HIGH RISK (FORBIDDEN-PATTERN guard required) | $9 |
| **Lane G v3-V2** (KL SNR-target Lagrangian) | [1.00, 1.04] | −0.05…−0.01 | MED | Code shipped | $1.50 |
| **Lane EC** (engineered SegNet-flipping per-pixel deltas) | [0.85, 1.05] | −0.20…0.00 | MED-HIGH | Code shipped IN-REPO 2 weeks but **NEVER DEPLOYED** | $0.30 |
| **Lane SG** (re-scope Lane S protection to SegNet pathway) | [0.85, 1.05] | −0.20…0.00 | MED | NEW from EUREKA #5; small refactor | $0.50 |
| **Lane PS-V2** (Lagrangian-learnable per-class weights) | [1.00, 1.10] | −0.05…+0.05 | LOW-MED | Code shipped | $0.50 |

### 2.7 Re-anchored geometric / external-feature lanes (NEW)

| Lane | Re-anchored band | Δ vs 1.05 | Confidence | Status | Cost |
|---|---|---:|:---:|---|---:|
| **Lane HM** (analytical road-plane homography replaces motion module) | [0.90, 1.10] | −0.15…+0.05 | LOW-MED | NEW from EUREKA #2; ~40KB renderer save if motion module removed | $1.50 |
| **Lane CG** (intrinsics-aware positional encoding) | [0.90, 1.10] | −0.15…+0.05 | MED | NEW from EUREKA #6 | $1.50 |
| **Lane DI** (openpilot supercombo penultimate features → 32-dim scene embedding) | [0.85, 1.05] | −0.20…0.00 | MED | NEW from EUREKA #3 | $1.00 |

**Total re-anchored lane count**: 49 lanes catalogued. **Of those, 17 are
code-shipped-but-never-deployed.** This is the immediate dispatch surface.

---

## §3. Composition rules (orthogonality matrix at the Lane G v3 frontier)

### 3.1 Independent axes (compose multiplicatively on the bytes side, additively on the score side)

```
Renderer   ⊥  Mask       ⊥  Pose       ⊥  EC-deltas
(rate 0.46)   (rate 0.27)   (rate 0.005)  (additive,
                                            +seg gain -rate cost)
```

### 3.2 Mutually exclusive choices (pick one each)

- **Architecture**: dilated-h64 (Lane A baseline, validated) **OR** DSConv-88K
  (Lane K) **OR** Cool-Chic CCh1 (Lane I, renderer-replace variant) **OR**
  Ghost-h64 (Lane GH) **OR** Quantizr-replica (Lane V) **OR** szabolcs no-masks
  (Lane SZ).
- **Quantization scheme on chosen arch**: per-tensor (FP4 simulated DEAD) **OR**
  per-channel SC (Lane S) **OR** hard-pair-weighted SC (Lane W) **OR** per-element
  Hessian (Lane Ω-V2) **OR** hardware FP8 (Lane F-V5).
- **Mask codec**: AV1 baseline **OR** Cool-Chic mask codec (Lane I-B) **OR**
  half-frame (Lane MAE-V) **OR** no masks (Lane SZ).
- **Pose representation**: full 600×6 (Lane A baseline) **OR** LoRA rank-1
  (Lane LR) **OR** zero-cost lane-mark (Lane LM-V2) **OR** Gaussian-process
  (Lane GP) **OR** RAFT-derived (Lane FL).
- **Motion module**: learned (current) **OR** analytical homography (Lane HM)
  — NOT both.

### 3.3 Composition CONFLICT matrix (lanes that CANNOT stack)

| Lane A | Lane B | Reason |
|---|---|---|
| Lane S/W/Ω-V2 | Lane I (Cool-Chic renderer) | Different parameter format |
| Lane S/W/Ω-V2 | Lane SZ | Different parameter format (block-FP) |
| Lane W (uses pair_weights) | Lane SZ | Lane W needs AsymmetricPairGenerator |
| Lane LR | Lane LM-V2 (pure form, no poses.pt) | LR works on poses.pt; LM-V2 removes them |
| Lane SI-V2 | Lane SZ | SI works on masks.mkv; SZ has no masks |
| Lane HM (analytical motion) | Lane M-V2/M-V3 (pose-conditioned motion) | Different motion path |
| Lane EC | Lane SZ | EC needs pre-upscale frame buffer; SZ pipeline differs |
| Lane F-V5 (FP8) | Lane S/W/Ω-V2 | Same underlying weights; pick ONE quant scheme |

### 3.4 Composition SYNERGIES (lanes that AMPLIFY each other)

- **Lane EC × any quantization lane**: EC compensates for quant-induced
  SegNet errors. The lower the Lane W/Ω-V2 score, the more EC has to fix
  → SegNet improvement compounds.
- **Lane SAUG × Lane V**: SAUG closes proxy-auth gap; Lane V is a from-scratch
  retrain that benefits from training-time perturbation.
- **Lane T2-MASK × Lane SAUG × Lane W**: three orthogonal robustness signals
  during training.
- **Lane HF (frame foveation) × Lane HFM (mask foveation)**: same Telescope
  primitive applied to two different inputs; expect superlinear gain on
  PoseNet attention regions.
- **Lane DI (scene embedding) × Lane LM-V2 (lane-mark poses)**: openpilot
  features supply the missing context that lane-mark-derived poses lack.

---

## §4. Three candidate stacks — specific, costed, predicted

### STACK 1 — CONSERVATIVE (high confidence sub-1.0)

**Architecture**: Lane G v3 baseline (validated, archive bytes already known).

**Components**:
1. Lane G v3 anchor (1.05) — `experiments/results/lane_g_v3_landed/archive_lane_g_v3.zip`
2. **+ Lane W** (per-channel SC + hard-pair weighting) on the Lane G v3 renderer
   - File: `scripts/remote_lane_w_hard_pair_self_compress.sh`
   - Predicted Δ_renderer_rate: −0.10 to −0.15
   - Predicted Δ_distortion: ±0.02 (small swing)
3. **+ Lane LM-V2** (endpoint-tracking lane-mark zero-cost poses)
   - File: `scripts/remote_lane_lm_v2_endpoint_tracking.sh`
   - Predicted Δ_pose_rate: −0.005 to −0.010 (kills poses.pt → ~7KB save)
   - Predicted Δ_pose_dist: +0.01 (small regress)
4. **+ Lane SI-V2** (Lagrangian saliency mask)
   - File: `scripts/remote_lane_si_v2_learnable_threshold.sh`
   - Predicted Δ_mask_rate: −0.05 to −0.08
   - Predicted Δ_seg_dist: ±0.005

**Stack EV math**:
```
Base                 = 1.05
+ Lane W             ≈ −0.10 (rate)
+ Lane LM-V2         ≈ −0.005 (rate, +0.01 pose acceptable)
+ Lane SI-V2         ≈ −0.06 (rate)
+ stacking friction  ≈ +0.02 (each lane has minor distortion side-effect)
─────────────────────
Predicted band        [0.85, 0.95]  centered ~0.90
```

**Cost**: $5 (one Vast.ai 4090, ~16h sequential composition).

**Confidence**: HIGH (each component has confidence MED-HIGH or higher;
all orthogonal; no conflicts).

**Why this exists**: Beats Lane G v3 by ~0.10–0.20 with proven mechanisms.
Even if Lane W underperforms, Lane SI-V2 + LM-V2 alone already gets us to
~0.97. **First plausible sub-1.0 score with no moonshot risk.**

---

### STACK 2 — AGGRESSIVE (medium confidence sub-0.7)

**Architecture**: Lane G v3 baseline (still validated, but stacked harder).

**Components**:
1. Lane G v3 anchor (1.05).
2. **+ Lane Ω-V2** (Lagrangian per-element bits) — replaces the dominant rate
   wedge (−0.46) with a per-weight optimal allocation.
   - File: `scripts/remote_lane_omega_v2_lagrangian.sh`
   - Predicted Δ_renderer_rate: −0.20 to −0.30 (renderer.bin from ~120KB to
     40-60KB if 75KB target; could go to 30KB at the lower band).
   - Predicted Δ_distortion: ±0.05 (medium swing — if hard-pair signal real
     and SegNet pathways protected, small; if not, larger).
3. **+ Lane I-B** (Cool-Chic mask codec — distinct from Lane I renderer)
   - Predicted Δ_mask_rate: −0.10 to −0.15
   - Predicted Δ_seg_dist: ±0.01
   - **Risk**: Lane I-B is not yet coded; if a 4-day path doesn't fit,
     substitute Lane SI-V2 + higher CRF.
4. **+ Lane SAUG-V2** (Lyra SAUG + Cosmos HighSigmaStrategy)
   - File: `scripts/remote_lane_saug_v2.sh`
   - Predicted Δ_dist: −0.05 to −0.10 (closes proxy-auth gap)
   - Cost: training-time only (no archive bytes)
5. **+ Lane HF** (Telescope hyperbolic foveation on rendered frames)
   - Predicted Δ_dist: −0.05 to −0.15
   - Cost: 9.6KB foveation_params.bin → +0.006 rate
6. **+ Lane EC** (engineered corrections) — applied LAST on the composed
   renderer outputs.
   - File: `scripts/remote_lane_ec_engineered_corrections.sh`
   - Predicted Δ_seg_dist: −0.001 to −0.003 (4-5× SegNet improvement) =
     **−0.10 to −0.30 score gain on a low-SegNet baseline**
   - Predicted Δ_rate: +0.02 to +0.05 (gradient_corrections.bin 30-50KB)

**Stack EV math**:
```
Base                  = 1.05
+ Lane Ω-V2 (renderer rate) ≈ −0.25
+ Lane I-B  (mask rate)     ≈ −0.12
+ Lane SAUG-V2 (proxy gap)  ≈ −0.07
+ Lane HF (frame fovea)     ≈ −0.10  (+0.006 rate ≈ negligible)
+ Lane EC (SegNet polish)   ≈ −0.20  (+0.03 rate)
+ stacking friction         ≈ +0.10  (Ω-V2 distortion, EC overlap with HF)
──────────────────────────────────
Predicted band               [0.55, 0.75]  centered ~0.65
```

**Cost**: ~$15 ($1.50 Ω-V2, $4 I-B/proxy, $4 SAUG-V2, $4 HF, $0.30 EC,
$1.20 composition + retries).

**Confidence**: MEDIUM (Ω-V2 has never been deployed; I-B not yet coded;
EC hasn't been auth-eval'd at this distortion floor).

**Failure modes**:
- Lane Ω-V2 hard-pair signal weak → falls to [0.85, 1.00] band; still beats Lane G v3
- Lane EC corrections.bin size grows past 50KB → rate inflation eats SegNet gain
- Lane HF Newton-Raphson invertibility breaks at certain α values → revert to baseline foveation

---

### STACK 3 — MOONSHOT (low confidence, sub-Quantizr)

**Architecture**: Lane V (Quantizr replica, 88K + half-frame from epoch 0)
OR Lane SZ (szabolcs no-masks paradigm) — pick based on which lands first.

#### 3a. Lane V variant

**Components**:
1. Lane V renderer (in flight, inst 35733832) — predicted [0.40, 1.00]
   re-anchored. **Lower band beats Quantizr; upper band underperforms Lane W.**
2. **+ Lane W applied to Lane V's checkpoint** (per-pair-weighted SC on the
   88K replica)
   - Predicted Δ_renderer_rate: −0.05 (already small renderer; modest gain)
3. **+ Lane MAE-V** (joint half-frame from epoch 0) — already in Lane V if
   `quantizr_replica_88k_halfframe` profile uses `mask_half_sim_prob_anneal`.
   Otherwise add Lane V-V2.
4. **+ Lane SAUG** (Lyra self-augmentation during Lane V's training)
   - Predicted Δ_dist: −0.05 to −0.10
5. **+ Lane SAUG-V2** (HighSigmaStrategy on Lane V) — second-axis robustness.
6. **+ Lane EC** on the final composed output.

**Stack EV math** (assuming Lane V lands at band-midpoint 0.70):
```
Lane V midpoint            = 0.70
+ Lane W on V              ≈ −0.05
+ Lane SAUG (during V)     ≈ −0.05  (already baked in if V uses SAUG)
+ Lane SAUG-V2 (orthogonal)≈ −0.05
+ Lane EC                  ≈ −0.15  (+0.025 rate)
+ stacking friction        ≈ +0.05
──────────────────────────────────
Predicted (V mid)           ~0.45  → range [0.30, 0.65]
```

If Lane V lands at lower band (0.40), the stack hits **0.20–0.35** —
**below Quantizr.**

#### 3b. Lane SZ variant (paper-worthy if it works)

**Components**:
1. Lane SZ Phase 2 (szabolcs Gaussian LUT + per-frame affine + 1.017 b/w
   block-FP) — predicted [0.20, 0.45].
2. **+ Lane LM-V2** zero-cost poses (Lane SZ archive lacks poses anyway).
3. **+ Lane EC** on the LUT-decoded mask predictions.

**Stack EV math** (assuming SZ lands at band-midpoint 0.32):
```
Lane SZ midpoint           = 0.32
+ Lane LM-V2 (pose rate)   ≈ −0.005
+ Lane EC                  ≈ −0.10  (+0.025 rate)
+ stacking friction        ≈ +0.05
──────────────────────────────────
Predicted (SZ mid)          ~0.28  → range [0.20, 0.40]
```

**Cost**: ~$25 (Lane V $4 + Lane W $0.50 + Lane SAUG×2 $9 + EC $0.30 + Lane SZ
$4 + composition retries $7).

**Confidence**: LOW (Lane V predicted band is wide [0.40, 1.00]; Lane SZ has
never been validated locally; many composition unknowns).

**Why try anyway**: This is the only stack with a meaningful probability of
crossing the 0.33 line. Per CLAUDE.md non-conservative charter, the burden of
proof is on NOT trying it.

---

## §5. Three NEW candidate lanes (not currently in portfolio)

The current portfolio is heavy on rate-attack and pose-attack but THIN on:
(a) compress-time-only output enhancement; (b) cross-lane EC-style polish;
(c) LM-V2-style zero-cost archive elimination at NEW positions in the pipeline.

### 5.1 Lane EC-V2 — Sequential SegNet-flipping with Lagrangian rate cap

**Why nothing addresses this**: Lane EC (existing) uses a fixed `max_delta`
budget per-pixel and treats every flip equally. But the SegNet score is
**non-uniform per-class**: certain class confusions (e.g., road↔sidewalk in
PoseNet-critical regions) cost MORE than others (e.g., sky↔building in
upper image regions where no driving info lives).

**Mechanism**:
1. Lane EC produces candidate ±delta per pixel.
2. Compute per-flip "score gain" = ΔSegNet_score and "rate cost" = delta
   bytes after zlib compression.
3. Greedy water-fill: include flips in descending order of gain/byte until
   rate budget exhausted.
4. Ship as `gradient_corrections_v2.bin` with per-flip class-pair metadata.

**Predicted band**: [0.85, 1.00] standalone vs Lane EC's [0.85, 1.05] — better
upper bound because the high-cost flips that bloated EC's archive are now
budgeted out.

**Cost**: ~$0.50 (compress-time only). Implementation: ~150 lines on top of
existing `experiments/precompute_gradient_corrections.py`.

**Composability**: Drops in wherever Lane EC was scheduled. Strictly Pareto-
dominates Lane EC in the composition matrix.

### 5.2 Lane EBR — Entropy-Bottleneck Rate model for renderer.bin

**Why nothing addresses this**: All current quantization lanes (S/W/Ω-V2/F-V5)
are **post-hoc** — they quantize an already-trained renderer. Entropy-coded
rate models (Ballé 2018, Minnen 2020) embed the bit-rate INTO THE TRAINING
LOSS, training the model to occupy a specific entropy budget end-to-end.

**Mechanism**:
1. Replace fake-quant nodes in Lane S with ENTROPY BOTTLENECK layers
   (parameterized continuous rate + uniform noise during training, fixed
   quantization at eval).
2. Joint loss: `distortion + λ × estimated_rate(weights)` where `λ` controls
   the rate-distortion trade-off.
3. Train Lane G v3 renderer with EBR; export entropy-coded weights.
4. Inflate-time decoder is a single small entropy-coding library (~30KB lib;
   open-source `compressai` via `torchac` integration).

**Predicted band**: [0.65, 0.95]. The Δ vs Lane Ω-V2: EBR jointly trains for
RD optimum; Ω-V2 trains then quantizes. Joint training typically saves another
20-30% of bytes at iso-distortion (per Cool-Chic paper, similar mechanism).

**Cost**: ~$5 (training $4, inflate-time decoder integration test $1).
Implementation: ~300 lines + `torchac` dependency.

**Composability**: Replaces Lane S/W/Ω-V2 (mutually exclusive quantization).
Composes with mask + pose + EC lanes.

**Risk**: `torchac` adds a binary dependency at inflate time; counts toward
the 30-min T4 inflate budget. Need to validate decode time on T4.

### 5.3 Lane PRIOR — Compress-time-baked PRIOR distribution for masks.mkv

**Why nothing addresses this**: Both Lane SI-V2 (saliency CRF) and Lane MAE-V
(half-frame) attack the masks STORAGE. None attack the CODING DISTRIBUTION.
Standard AV1 entropy coding assumes a generic content prior. We have a
SPECIFIC prior — driving footage with consistent class statistics (~50% road,
~15% sky, ~10% buildings, ~5% lane marks, ~20% other).

**Mechanism**:
1. At compress time, fit a per-class mixture-of-Gaussians prior to the
   ground-truth mask distribution across the 1199 pairs.
2. Use this prior to construct a CUSTOM SLICE/CONTEXT model for the AV1
   encoder (libsvtav1 supports passing custom probability models via tile
   configuration).
3. At inflate time, the same prior is reconstructable from a tiny header
   (~200 bytes of class statistics) — no scorer load required.

**Predicted band**: [0.95, 1.05]. The Δ on rate is small because AV1 already
exploits content statistics implicitly, but the EXPLICIT prior gives 5-10%
additional compression on the dominant components (road, sky uniform regions).

**Cost**: ~$0.50 (compress-only research, no GPU training). Implementation:
~200 lines (libsvtav1 wrapper + 200-byte header).

**Composability**: Stacks with EVERYTHING that uses masks.mkv (i.e.,
everything except Lane SZ).

**Risk**: libsvtav1 custom probability is undocumented; may need to fall back
to a pre/post entropy-recoding wrapper around the AV1 stream.

---

## §6. Three highest-EV NEXT dispatches (single lanes)

These are the **next single-lane Vast.ai dispatches** given the new 1.05
frontier. Selection criteria:
1. Code already shipped and reviewed (no integration risk).
2. Re-anchored band has lower bound ≤ 0.95 (meaningfully advances frontier).
3. Stacks orthogonally with Stack 1 (cheap path to compounding gains).

### Dispatch #1 — Lane Ω-V2 (Lagrangian per-element bits)

- **Script**: `scripts/remote_lane_omega_v2_lagrangian.sh`
- **Re-anchored band**: [0.70, 0.95]
- **Cost**: $1.50 / ~6h
- **Why now**:
  - Survived 7 codex review rounds; mathematically validated
  - Has NEVER been auth-eval'd (only smoke tested locally)
  - Lane W's signal (per-pair weighting works) being validated this overnight
    will confirm the prerequisite signal for Ω-V2
  - Per-element resolution is the natural finer-grain successor
- **Anchor change**: Was anchored on Lane A 1.15. New floor 1.05 means even
  the upper-bound 0.95 already beats Lane G v3 by 0.10. **No prior dispatch
  has this property at this confidence level.**
- **Council pre-deliberation**: Hotz endorses (next natural unlock); Yousfi
  endorses (Hessian-aware quant is literature-canonical); Quantizr says
  "this would definitively beat my approach"; Contrarian's only objection
  ("600 lines of new code") is mooted because the code is in repo.

### Dispatch #2 — Lane EC (engineered corrections) on Lane G v3 archive

- **Script**: `scripts/remote_lane_ec_engineered_corrections.sh`
- **Re-anchored band**: [0.85, 1.05]
- **Cost**: $0.30 / ~2h (compress-time only; no training)
- **Why now**:
  - Code shipped TWO WEEKS ago in the repo; never deployed (per
    `project_lane_ec_engineered_corrections_20260428`)
  - 444-line integration test passes locally
  - Standalone on Lane G v3 should land [0.85, 1.05] at this distortion floor
  - The lower band (0.85) would IMMEDIATELY validate sub-1.0 with $0.30 spend
  - Rate cost ~30-50KB (gradient_corrections.bin) at the 1.05 floor is
    forecast as net-positive (SegNet gain >> rate cost)
- **Anchor change**: Lane G v3's SegNet is 0.004; EC's mechanism is to flip
  miss-classified pixels at compress time. Per the math in §1.4, a 4× SegNet
  improvement = −0.30 score; even at 50% effectiveness, EC saves 0.15 net.
- **Failure rescue**: If EC's corrections bloat past 50KB, immediate revert to
  the unmodified Lane G v3 archive — zero risk to baseline.

### Dispatch #3 — Lane SAUG-V2 (proxy-auth gap closure, 2nd-axis)

- **Script**: `scripts/remote_lane_saug_v2.sh`
- **Re-anchored band**: [0.60, 0.90]
- **Cost**: $4 / ~14h
- **Why now**:
  - The proxy-auth gap is 100-350× even on CUDA-CUDA (memory:
    `feedback_proxy_auth_math_useless`). This is **the largest documented
    blocker** to pose TTO and renderer training improvements.
  - Lane SAUG-V2 attacks the gap from TWO orthogonal axes: input perturbation
    (Lyra) + noise-schedule outliers (Cosmos HighSigmaStrategy).
  - Of all "training-improvement" lanes, this has the lowest re-anchored
    lower-bound band (0.60).
- **Anchor change**: SAUG's mechanism is target-distribution perturbation; it
  does NOT depend on Lane G v3's pose floor and re-anchors essentially
  unchanged. The Cosmos addendum
  (`project_cosmos_deep_dive_addendum_20260428`) is fresh and the recipe is
  hardware-validated for 4090 (FP8 native, FP4 simulated only).
- **Composability**: Trains a NEW renderer (not Lane G v3 anchor). Stack
  outputs would replace Lane G v3 as the anchor for downstream stacking.

### Dispatch ranking rationale

| Dispatch | EV (Δscore) | Cost | Risk | EV/$ |
|---|---:|---:|---|---:|
| Lane Ω-V2 | 0.10–0.35 | $1.50 | MED | **0.067–0.233 per $** |
| Lane EC | 0.00–0.20 | $0.30 | LOW | **0.000–0.667 per $** |
| Lane SAUG-V2 | 0.15–0.45 | $4.00 | MED | **0.038–0.113 per $** |

**Total**: $5.80 to fire all three; expected compounded score ~0.78–0.92
(if all stack favorably).

---

## §7. Cycle-1 deployment plan (next 36 hours)

Per CLAUDE.md "≤3 experiments per cycle" — Cycle 1 = the three dispatches
above, executed in parallel on 3× Vast.ai 4090.

```
T+0h    : Dispatch Lane Ω-V2 (Iceland)        — $1.50, 6h
T+0h    : Dispatch Lane EC (US-West, cheapest)— $0.30, 2h
T+0h    : Dispatch Lane SAUG-V2 (EU-West)     — $4.00, 14h

T+2h    : Lane EC auth eval lands → first sub-1.0 candidate
T+6h    : Lane Ω-V2 auth eval lands → quantization-frontier check
T+14h   : Lane SAUG-V2 auth eval lands → proxy-gap closure validated
T+16h   : Cycle 2 triage:
          - If Ω-V2 ≤ 0.95: dispatch Stack 1 composition (Ω-V2 + LM-V2 + SI-V2)
          - If EC ≤ 1.00: dispatch EC × Lane W (immediate cheap stack)
          - If SAUG-V2 ≤ 0.85: dispatch Lane V + SAUG-V2 (Stack 3a)
T+24h   : Cycle 2 results land
T+36h   : Decision point: ship Stack 1 OR continue to Stack 2/3
```

**Heartbeat protocol**: per `feedback_canonical_remote_bootstraps`, every
remote script writes `provenance.json` + `heartbeat.log` + `run_record.json`.
PARENT monitors via `scripts/lane_watchdog.py`; subagents do NOT manage
Vast.ai instance lifecycles (per `feedback_oneshot_vastai_subagent_failure_pattern`).

**Cost**: $5.80 cycle 1; budget remaining ≥ $290.

---

## §8. The ladder to sub-Quantizr (a roadmap, not a promise)

```
1.05    [Lane G v3]        — current frontier
0.95    [+ Lane Ω-V2 OR Lane EC]
0.90    [+ Stack 1: + LM-V2 + SI-V2]
0.85    [+ Lane W signal validated]
0.75    [+ Lane SAUG-V2 closes proxy gap]
0.65    [Stack 2: + Lane I-B mask codec + Lane HF]
0.50    [Stack 2 + Lane EC polish]
0.40    [Stack 3a: Lane V midpoint + SAUG-V2 + EC]
0.33    [Quantizr]
0.30    [Stack 3a lower band; Quantizr "stopped" zone]
0.25    [Stack 3b: Lane SZ midpoint + LM-V2 + EC]
```

**The math says**: every rung from 1.05 to 0.50 has a ≥30% probability path
in the existing portfolio (Stack 1+2 components). Rungs from 0.50 to 0.33
require Lane V or Lane SZ to land in their lower bands. Rungs below 0.33
require BOTH a moonshot architecture AND multiple composition wins.

**Council consensus**: Sub-1.0 is HIGH probability (≥85%) within Cycle 2.
Sub-Quantizr is MED-LOW probability (~30%) within deadline.

**Council split**:
- Hotz: "Just stack everything that's shipped. Lane EC + Lane Ω-V2 + Lane SAUG-V2
  + LM-V2 + SI-V2 on Lane G v3 = first cycle. Then iterate."
- Yousfi: "Lane SG (re-scope SegNet protection in Lane S/W) is the missing
  piece — we've over-protected PoseNet at the expense of the SegNet wedge.
  Rebuild Lane W with SegNet-pathway protected layers."
- Fridrich: "Lane SI-V3 (Fridrich UNIWARD on masks) is overdue. The texture
  σ² model gives per-pixel CRF assignment that beats Lane SI-V2's saliency
  proxy. ~$0.60 to run."
- Quantizr: "Lane V replica is the only path to sub-0.5 architecturally. Run
  it patiently; don't over-tune. My own 88K replica took 3 attempts to
  converge."
- Contrarian: "The 5-day deadline doesn't allow for 3 moonshot retries. Bias
  toward Stack 1 first; ship sub-1.0 ASAP; then attempt Stack 2/3 with
  remaining budget."

**Tripartite pact ruling**: Stack 1 dispatched in Cycle 1 (today), Stack 2
dispatched in Cycle 2 (T+24h), Stack 3 dispatched in Cycle 3 (T+48h) only
if Stack 2 has not landed sub-0.7.

---

## §9. Composability conflict warnings (must read before composition)

1. **Lane Ω-V2 + Lane EC** — Lane Ω-V2's per-element bit allocation may
   destabilize the renderer's pre-upscale activations that Lane EC operates
   on. **Mitigation**: pin Ω-V2's protected layers to include the final
   pre-upscale conv.
2. **Lane W + Lane SG** — Lane SG re-scopes Lane W's protected-layer list to
   SegNet-pathway weights. **DO NOT run them in parallel** as competing
   variants of Lane W; Lane SG IS Lane W with a different protected list.
3. **Lane SAUG + Lane T2-MASK** — Both perturb training inputs. The subagent
   that synthesized the Cosmos research recommended retiring Lane SAUG in
   favor of T2-MASK; **REJECTED**. They attack different axes (target
   perturbation vs feature dropout). Run both, treat as two different
   robustness signals.
4. **Lane EC + Lane HF** — Lane EC's deltas and Lane HF's foveation both
   modify rendered frames pre-upscale. **EC must be applied AFTER HF's
   inverse warp**, otherwise the per-pixel delta lands at the wrong
   coordinates. Add an integration test before stacking.
5. **Lane LM-V2 + Lane LR** — Mutually exclusive in the strict sense (LM-V2
   removes poses.pt; LR adds rank-1 LoRA on top of poses.pt). But a HYBRID
   variant — LM-V2 supplies dim 0, LR supplies dims 1-5 as rank-1 — would
   compose and is worth considering as Lane LM-LR.

---

## §10. The two re-anchoring corrections to existing memory

These corrections should be filed back to the memory layer:

1. **`project_lane_taxonomy_stacking_strategy_20260427`** has predictions
   anchored to Lane A = 1.15. After Lane G v3 = 1.05, every "predicted band"
   in that document is mostly invariant for rate/seg lanes but tightened for
   pose-TTO lanes (which have less remaining headroom). The conservative
   stack projection of "0.85-0.95" is now achievable WITHOUT Lane S — Lane G
   v3 + LM-V2 + SI-V2 + W gets there.

2. **`project_outstanding_work_and_stacks_20260428`'s Stack A (1.08)** is now
   superseded — Lane G v3 IS Stack A's components landed. Stack A's predicted
   1.08 was off by 3 points to the GOOD side (actual 1.05). This validates
   the predictive band methodology; future predictions should keep using
   ±0.05 band widths.

---

## §11. Three transferable patterns surfaced this session

Per the non-conservative charter, every council session must deposit
durable patterns:

1. **Re-anchoring rule for pose-TTO-saturated lanes**: When the frontier
   moves due to a pose-TTO win, any new lane whose mechanism is also pose
   TTO has its predicted-band UPPER bound capped near the new frontier.
   Lower bound is mechanism-specific. (Filed inline above; should be
   memorized.)

2. **EC-first composition**: Engineered corrections should be the LAST
   step in any composition stack because it operates on the FINAL renderer
   output and provides up to −0.30 SegNet score gain at +0.03 rate cost.
   For any stack with predicted SegNet ≥ 0.001, EC is net-positive.
   (Should be filed as `feedback_ec_first_composition`.)

3. **The wedge attribution principle**: Decompose the gap to leader
   into rate / SegNet / PoseNet contributions and target the largest two
   wedges first. SegNet is currently the largest single wedge (0.31 of
   0.72). Most current effort has been rate-attack (0.26 of 0.72).
   **There is a 0.31 SegNet wedge that current lanes barely touch.**
   The new EC-V2 lane and Lane SG re-scope address this directly.
   (Should be filed as `feedback_wedge_attribution_principle`.)

---

## §12. Final council vote on Cycle 1

| Member | Vote | Rationale |
|---|---|---|
| Yousfi | Approve dispatches 1+2+3 | "Ω-V2 is overdue; EC is free; SAUG-V2 attacks the gap. Add Lane SG to Cycle 2." |
| Fridrich | Approve, conditional on Lane SI-V3 in Cycle 2 | "UNIWARD texture-σ² is the inverse-steganalysis canonical extension." |
| Hotz | Approve all three | "Just ship them. Watchdog the heartbeats." |
| Quantizr | Approve Ω-V2 + EC, conditional on Lane V being given full ~12h to land | "Don't kill V early." |
| Contrarian | Approve Lane EC unconditionally, abstain on Ω-V2 + SAUG-V2 | "EC is risk-free. The other two need the auth eval to count. No premature claims." |

**Tripartite (Yousfi + Fridrich + Contrarian) pact**: PASS.

**Final dispatch order**:
1. Lane EC (T+0h, $0.30, 2h) — first sub-1.0 candidate
2. Lane Ω-V2 (T+0h, $1.50, 6h) — quantization frontier
3. Lane SAUG-V2 (T+0h, $4.00, 14h) — proxy-gap closure

**Total Cycle 1 spend**: $5.80. **Budget remaining**: ≥ $290.
**Probability of landing sub-1.0 in Cycle 1**: HIGH (≥85%).

---

## Cross-references

- `project_lane_g_v3_landed_1_05_20260428` — frontier source
- `project_lane_taxonomy_stacking_strategy_20260427` — full lane inventory
- `project_outstanding_work_and_stacks_20260428` — TIER 3 catalog
- `project_council_eurekas_driving_geometry_20260428` — 9 EUREKA lanes (GP/GE/FL/HM/CG/SG/DI/SI-V3)
- `project_cosmos_deep_dive_addendum_20260428` — Lane F-V5 hardware FP8, Lane SAUG-V2, Lane WC
- `project_arxiv_2604_24763_tuna2_synthesis_20260428` — Lane T2-XPRED/T2-MASK/T2-RATIO
- `project_cosmos_mae_lyra_telescope_synthesis_20260428` — Lane SAUG/MAE-V/HF
- `project_lane_omega_bit_budget_hessian_aware_quantization` — Lane Ω-V2 design
- `project_lane_w_hard_pair_self_compress_premise_20260427` — Lane W premise
- `project_lane_ec_engineered_corrections_20260428` — Lane EC mechanism
- `project_quantizr_full_intel_20260421` — competitor decomposition source
- `project_szabolcs_full_re_20260426` — Lane SZ source
- `feedback_compress_time_unlimited_archive_small_20260428` — reframing rule
- `feedback_proxy_auth_math_useless` — only contest-CUDA counts
- `feedback_oneshot_vastai_subagent_failure_pattern` — parent-monitors lifecycle
- `feedback_canonical_remote_bootstraps` — heartbeat + provenance

---
*Council session closed 2026-04-28. Next council convenes after Cycle 1
auth-eval results land.*
