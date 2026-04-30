# Grand Council — Strategic Design Decisions (Q1: Phase 1 dispatch / Q2: sensitivity maps / Q3: RL/PufferLib / Q4: Round 11 bug fixes)

**Date**: 2026-04-30
**Convened by**: parent agent under user mandate "fix all bugs regardless of severity; have grand council make design decisions"
**Inner council (10 voices)**: Shannon (LEAD), Dykstra (CO-LEAD), Yousfi, Fridrich, Contrarian, Quantizr, Hotz, Selfcomp, MacKay (memorial), Ballé
**Grand-council augmentation**: Karpathy (RL/training engineering), Hassabis (DeepMind strategic-research), Schmidhuber (compression-as-intelligence + RL lineage), Carmack (engineering shortcuts), Boyd (ADMM operational), Tao (first-principles math), Filler (STC), Mallat (wavelet)
**Mandate scope**: REPORT-ONLY. NO code modified. NO GPU spawned. All claims tagged `[empirical:<path>]` / `[contest-CUDA]` / `[contest-CPU advisory]` / `[Modal-T4-CUDA]` / `[prediction]` / `[derivation]` / `[synthetic]`.
**Concurrent activity respected**: HM-S dispatch firing in BG (do not touch `scripts/remote_lane_hm_s_segmap_homography.sh`); chain-integrity audit #269 still running (do not touch `council_chain_integrity_audit_20260430.md`).

---

## 1. Executive Summary

**Top-3 next-action recommendations** (after HM-S returns ~6h):

1. **Lane G v3 + Ω-W-V2 stack archive build → contest-CUDA auth eval** (Option A in Q1). Cost: $0 dev + $0.50 GPU. Predicted: Lane G v3 1.05 → ~0.97 `[derivation]`, ±0.02 noise. **HIGHEST CONFIDENCE in the portfolio** — rate-math is bit-deterministic; only risk is an OWV2 inflate-side handler bug. The Round 5 chain-integrity audit's Part G surfaced this and the Council-F validation already PROVED the codec saves 40.98% on the real Lane G v3 renderer.bin `[empirical:src/tac/tests/test_omega_w_v2_real_archive.py]`. Land OWV2 inflate handler (~30 LOC + tests, 1-2h dev) → build archive locally → ship to Vast.ai 4090 for one auth-eval ($0.50, 15 min).
2. **Build the canonical sensitivity-map module** (Q2) — `src/tac/sensitivity_map.py`. Cost: $0 dev (~1 day) + $1-2 CUDA on Vast.ai 4090 to compute the maps over all 600 frame pairs × 5-10 perturbation types. EV: replaces Hessian estimate in Lane Ω-W-V2's water-fill with empirical `dScore/dByte` gradient → -0.005 to -0.020 score on bit-budget-allocation lanes, AND becomes the input feature to Lane 19 (SegNet logit-margin boundary fitting) AND Phase 2 Lane 9 (STC redesign predictive boundary coding). This is foundational tooling that unlocks 3 downstream lanes. **Shannon's verdict: this map IS the empirical R(D) gradient at the operating point — Step 2 of the chain-integrity audit ("CONCERN: per-stream R(D) curves are sparsely sampled, 1-2 points per stream") gets resolved by this module.**
3. **Fix Q4A (Joint-ADMM Nesterov bias) + Q4B (rho_init adaptive default)** — strict bug fixes per user's "fix all bugs regardless of severity" mandate. Cost: $0 (local code). EV: gates Phase 2 Lane 10 V2 dispatch (Joint-ADMM real-codec wrapper). Q4C (J-JBL routing-table gap) is **MISDIAGNOSED** — `train_renderer.py:357` already wires `loss_mode='jbl'` as a CLI choice, the dispatch at line 3083-3113 fires correctly, and `tac/training.py:91` (the Pydantic Literal) is NOT in train_renderer's call path (train_renderer does NOT use `TrainConfig`; it uses raw argparse). KEEP the J_JBL_DILATED_H64 profile.

**RL/PufferLib verdict (Q3)**: **PILOT-WITH-BUDGET-CAP**. Defer the full PPO setup; START with a 1-evening contextual-bandit prototype on `boundary_fraction × qint_max × CRF` triples once the sensitivity-map module (Q2) lands. **First step**: write `experiments/bandit_codec_search.py` (~150 LOC) using `scipy.optimize` + Thompson sampling over 8-12 codec hyperparameters; budget cap $5 GPU; if 4-hour smoke beats 3-pass codex search by ≥0.005 score AT same archive bytes, graduate to PufferLib. **NOT** a full RL framework upfront — Quantizr's 0.33 was hand-tuned, not RL-discovered, and the cheapest signal will come from a bandit, not PPO.

**Total committed cost (Phase A: this 24-48h)**: $0.50 (Lane G v3 + Ω-W-V2 auth) + $1-2 (sensitivity maps) + $0 (Round 11 bug fixes) + $0-5 (bandit pilot if sensitivity maps land) = **$1.50-7.50 of GPU spend**, well under the $30 Vast.ai daily cap.

---

## 2. Q1: Phase 1 Next-Highest-Value Dispatch

### 2.1 Per-option per-bp impact analysis

The current `[contest-CUDA]` baseline is Lane G v3 = 1.05 ± 0.01 (cross-platform reproduced Vast.ai 4090 + Modal T4). The Phase 1 dispatch options:

| Option | Cost | EV (score Δ) | Confidence | Dependency on HM-S | Notes |
|---|---|---|---|---|---|
| **A. Lane Ω-W-V2 STACK with HM-S anchor** | **$0.50** GPU + 1-2h dev | **-0.078 derivation** (0.066 expected after handler-correctness discount) | **HIGH (`[derivation]` rate math + `[empirical]` codec savings)** | NONE — works on Lane G v3 anchor too | OWV2 inflate handler ~30 LOC; codec already validated 40.98% [empirical:tests/test_omega_w_v2_real_archive.py] |
| **B. Lane PD-V2 + LCT bolt-on with HM-S anchor** | $0 byte-deterministic local + auth | -0.004 to -0.011 (PD-V2 18.5% empirical on small pose stream) | MEDIUM (PD-V2 measured low, LCT bolt-on untested) | NONE | LCT 8/8 tests passing; payload only ~10 bytes; PD-V2 may overhead-gate-fail on real high-entropy archive (Round 5 §2.4 CONCERN-1) |
| **C. Lane J-NWC corpus codec** | $5 (corpus codec training) | -0.020 to -0.060 IF amortizes (codec+ ≥150K params) | LOW (88K Lane A is unfavorable per `project_lane_j_nwc_landed_20260429.md` — NWC1 LOSES at small scale) | NONE | Corpus codec doesn't yet exist; would unblock Phase 3 amortization (large-scale lane); $5 spent today ships ZERO score improvement |
| **D. Lane FR-Ω dispatch (Council F's Wave 2)** | $1.50 GPU | -0.05 to -0.10 IF orthogonal+converges | MEDIUM-LOW (script vs Council F band inconsistency [0.25-0.32] vs [0.27-0.45]; hit-rate prior ~33%) | YES — Council F: gate FR-Ω on HM-S signal | High variance; only fire after HM-S calibrates |
| **E. SegNet/PoseNet sensitivity-map generation** | $1-2 GPU | -0.005 to -0.020 directly + UNBLOCKS Lane 19 / 9 / Ω-W-V2 V3 (-0.020 to -0.080 indirect) | HIGH for byte-allocation reweight; MEDIUM for indirect | NONE | Foundational tooling; replaces Step 2 chain-integrity "trust me" link with measured R(D) curves |
| **F. Land OWV2 inflate handler ONLY (no GPU yet)** | $0 dev (1-2h) | 0 today; UNBLOCKS Option A | n/a | NONE | Prerequisite for A; cheap; do this regardless |

### 2.2 EV ranking + concrete dispatch order

**Inner-council quintet ranking** (Shannon LEAD + Dykstra CO-LEAD + Yousfi + Fridrich + Contrarian):

1. **Option F → Option A (combined)** — highest EV/$ in the portfolio. The Round 5 chain-integrity audit Part G G1 explicitly named this as the "chain MISSES one materially higher-EV $0.50 dispatch". `[derivation]` confidence on the rate term + `[empirical]` confidence on the codec savings. Dispatching this AS Phase 1 also lands the OWV2 inflate handler that 3-4 other Phase 2 lanes will need. Hotz: "20 minutes of dev for an anchored measurement beats a 33%-confidence band". **DISPATCH FIRST after HM-S returns.**
2. **Option E (sensitivity maps)** — foundational tooling that unlocks 3 downstream lanes (Lane 19 SegNet logit-margin, Lane 9 STC redesign, Lane Ω-W-V2 V3 hyperprior). Cost is small ($1-2). Direct EV is modest (-0.005 to -0.020) but the indirect EV through unblocking is substantial. Shannon: "this map IS the empirical R(D) gradient — it resolves the chain-integrity audit's Step 2 CONCERN." **DISPATCH SECOND (in parallel with F+A; no GPU collision since A uses 1×4090 for 15 min and E uses 1×4090 for 1-3h).**
3. **Option B (PD-V2 + LCT bolt-on)** — zero-cost, low-EV. Worth landing as filler, but scheduled AFTER F+A confirms the Ω-W-V2 stack works (PD-V2 + LCT both target the pose stream which is already only ~2% of archive bytes; the rate-term wedge is in the renderer.bin which Ω-W-V2 attacks). **DISPATCH THIRD as $0 local-only validation.**
4. **Option D (FR-Ω)** — gated on HM-S signal per Council F's sequential rule. FR-Ω targets the same per-channel byte-allocation surface as Ω-W-V2; once Ω-W-V2 lands a measured score, FR-Ω's Hessian-aware band can be RECALIBRATED against that anchor. **DISPATCH FOURTH ONLY IF: (a) HM-S band calibrates within 0.10 of [0.32, 0.45] central; (b) Ω-W-V2 stack auth confirms ~0.97 `[contest-CUDA]`; (c) FR-Ω script vs Council F band inconsistency is reconciled.**
5. **Option C (J-NWC corpus codec)** — DEFER to Phase 2/3. Today's $5 ships zero score; under the 6-month team-parallelization budget, this is a Phase 3 lane (Lane J-IMP family). Council F killed the standalone J-NWC dispatch on the 88K Lane A renderer; the corpus version requires multiple renderers to amortize. Not a 24-48h play.

### 2.3 Concrete dispatch order (24-48h)

```
T+0h    HM-S returns from BG dispatch
        ├─ if HM-S in band [0.32, 0.45]: bands calibrated → Phase D unlocked
        ├─ if HM-S in band [0.45, 0.85]: kept advisory; Phase D delayed
        └─ if HM-S > 0.85: bands FALSIFIED; Phase D KILLED
T+0h    LAUNCH Option F: land OWV2 inflate handler (1-2h dev)
T+2h    LAUNCH Option A: build Lane G v3 + Ω-W-V2 archive locally
T+2h    LAUNCH Option E: dispatch sensitivity-map computation on Vast.ai 4090 ($1-2, 1-3h)
T+3h    Option A archive ready → dispatch contest-CUDA auth eval ($0.50, 15min)
T+3.5h  Option A result lands; Phase 2 Lane 9/19/20 priority recomputed
T+4h    Option E result lands; Lane 19 (SegNet logit-margin) is properly motivated
T+8h    LAUNCH Option B (PD-V2 + LCT) AS local validation
T+24h   Decide on Option D (FR-Ω) — gated on HM-S calibration AND Option A confirm
```

**Total Phase A cost**: $1.50-2.50 GPU + 1-2h dev. **Risk-adjusted EV**: -0.066 (Option A) + -0.010 (Option E) + -0.004 (Option B) = **-0.080 score reduction expected** (Lane G v3 1.05 → ~0.97 contest-CUDA frontier, with sensitivity-map unblock primed for Phase 2).

### 2.4 Justification per inner-council voice

- **Shannon (LEAD)**: Option A delivers a `[derivation]`-grade rate-term reduction (-0.078) by exploiting Ω-W-V2's empirical 40.98% on Lane G v3's renderer.bin. Option E builds the empirical R(D) curve sampling that the chain audit identified as missing. Both are R(D) ceiling-approaching mechanisms. **APPROVE Option F → A first; E second.**
- **Dykstra (CO-LEAD)**: Option A is a Pareto-improving point on the convex hull (saves rate without distortion change, given codec round-trip is bit-faithful). Option E gives the empirical convex-hull intersection points the council has been deriving from architectural reasoning. **APPROVE.**
- **Yousfi (Challenge creator)**: The contest IS inverse steganalysis. Option E (sensitivity maps) is what Fridrich's UNIWARD framework prescribed — measure detector sensitivity per-region, allocate quantization noise where the detector is blind. This is the canonical Fridrich move. **APPROVE Option E as foundational.**
- **Fridrich**: "Errors in textured regions are undetectable." Option E builds the per-region texture/sensitivity map that operationalizes UNIWARD on this scorer. Combined with Option A's Hessian-aware byte-allocation, the stack approaches the steganalysis-detector's blind-spot. **APPROVE Options A + E together.**
- **Contrarian (Veto)**: I VETO Option D (FR-Ω) until Option A lands. The rate-term wedge that FR-Ω targets is the SAME wedge that Ω-W-V2 targets; landing Ω-W-V2 first calibrates whether FR-Ω's predicted -0.05 to -0.10 is on top of Ω-W-V2's -0.078 (additive) or whether they overlap (Dykstra's "additivity is conditional" rule). Council F's sequential discipline applies. **APPROVE Options F → A → E → B; DEFER D.**
- **Quantizr (Adversarial)**: My 0.33 archive uses block-FP at 1.017 bpw (Ω-W-V2-class). Option A is precisely my approach with empirical proof. Option E (sensitivity maps) is what I would have built had I had more time — it lets you target the per-pixel score margin directly. **APPROVE Options A + E.**
- **Hotz (Engineering shortcuts)**: 1-2h to land OWV2 + $0.50 for the auth eval is the cheapest measurement in the portfolio. I would write the handler in 30 minutes. **APPROVE Option F → A first.**
- **Selfcomp (szabolcs-cs)**: My 88K SegMap + grayscale-LUT + 1.017 bpw block-FP = 0.38. Option A maps directly to my paradigm on the 88K Lane G v3 renderer. Option E (sensitivity maps) is the right next step toward score-aware allocation that I never had time to operationalize. **APPROVE.**
- **MacKay (Memorial)**: Option A is a strict MDL improvement (rate cost reduced without posterior approximation quality change, per per-channel L_inf bound). Option E provides the empirical posterior `p(score | byte_allocation)` that Bayesian model selection (Lane 16) requires. **APPROVE both as MDL-justified.**
- **Ballé**: Option E (per-channel sensitivity) is what my 2018 hyperprior network learns implicitly via `σ_y(y)`. Building it explicitly via finite differences here serves as both (a) a baseline for the eventual learned hyperprior (Lane 20) and (b) an immediate input to bit-allocation. **APPROVE.**

**Quintet pact verdict**: 5/5 APPROVE Options F → A → E → B in that order. Co-members 5/5 also APPROVE. **CONSENSUS.**

---

## 3. Q2: SegNet/PoseNet Sensitivity Maps — Design + Implementation Path + EV

### 3.1 Existing tools sufficiency assessment

Existing partial coverage:
- `src/tac/forensics.py` provides FOUR analytical functions:
  - `boundary_artifact_score`: zero-padded conv border statistics
  - `segnet_class_boundary_analysis`: SegNet boundary vs interior disagreement
  - `posenet_sensitivity_map`: per-pixel Jacobian norm of PoseNet output (single-pair)
  - `eval_roundtrip_distortion_map`: per-pixel error from the 384→874→uint8→384 chain
- `experiments/pair_difficulty_map.py` provides per-PAIR PoseNet MSE + SegNet disagreement (no per-region resolution; not gradient-based)
- `src/tac/uniward_texture.py` provides `compute_texture_probability` (Fridrich-style local variance)

**Gap analysis**:
- `posenet_sensitivity_map` is per-pair (one frame pair) not corpus-wide (600 pairs)
- `segnet_class_boundary_analysis` returns global metrics only (not per-region or per-pair)
- `pair_difficulty_map` is per-pair score but not per-pixel gradient
- **NO module computes "marginal score damage per byte saved per region" — the canonical R(D) gradient** that Shannon, Dykstra, and Lane Ω-W-V2 all need

**Verdict**: **YES, a new unified module is needed**. The existing tools are building blocks but do not output the required artifact: a `(N_pairs, H, W, 3)` tensor where dim-3 is `[d(SegNet_loss)/d(pixel), d(PoseNet_loss)/d(pixel), byte_cost_per_unit_distortion_at_pixel]`. With this tensor, every byte-allocation lane (Ω-W-V2, FR-Ω, Lane 9 STC, Lane 11 wavelet, Lane 19 logit-margin, Lane 20 Ballé hyperprior) gets a measured `dScore/dByte` gradient that replaces the current Hessian-estimate / texture-heuristic approximations.

### 3.2 Canonical sensitivity-map module design

**File**: `src/tac/sensitivity_map.py` (new, ~400-600 LOC)

**API**:

```python
def compute_sensitivity_maps(
    rendered_frames: torch.Tensor,         # (N, H, W, 3) float [0, 255]
    gt_frames: list[torch.Tensor],          # length N, each (H, W, 3) uint8
    posenet: nn.Module,
    segnet: nn.Module,
    device: torch.device,
    *,
    perturbation_types: tuple[str, ...] = (
        "gaussian_pixel",      # δ ~ N(0, σ_pixel) per pixel
        "block_fp_quantize",   # round to nearest qint level per channel
        "av1_crf_step",        # increase CRF by 1
        "subsampling_2x",      # downsample then bilinear upsample
        "uniward_textured",    # apply δ weighted by (1 / texture_prob)
    ),
    perturbation_magnitudes: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0),
    batch_size: int = 8,
    storage: str = "hdf5",     # 'hdf5' | 'numpy' | 'pt'
    output_path: Path,
    eval_roundtrip: bool = True,    # CLAUDE.md non-negotiable
) -> SensitivityMapResult:
    """Compute per-frame, per-region [SegNet_Δ, PoseNet_Δ, byte_cost_Δ].

    For each frame pair k, for each perturbation_type p, for each
    magnitude m:
      1. Apply perturbation p with magnitude m to rendered_frames[k:k+2]
      2. Forward through SegNet and PoseNet (frozen)
      3. Compute SegNet_loss_delta = 100 * (perturbed_disagree - clean_disagree)
      4. Compute PoseNet_loss_delta = sqrt(10 * pose_mse_perturbed) - sqrt(10 * pose_mse_clean)
      5. Compute byte_cost_delta:
           - 'gaussian_pixel': δ * 8 bits (additive)
           - 'block_fp_quantize': bits saved per channel (per qint level decrease)
           - 'av1_crf_step': ffmpeg byte delta on a 30-frame chunk
           - 'subsampling_2x': 75% byte reduction (assuming bilinear)
           - 'uniward_textured': δ * 8 bits weighted by texture
      6. Per-region averaging via 8x8 block grid

    Stores result as HDF5 (default — supports random access for downstream
    lanes without loading everything).

    Returns SensitivityMapResult with per-pair, per-region, per-perturbation
    [seg_delta, pose_delta, byte_delta, score_delta_per_byte_saved]
    quadruples. The last entry is the canonical `dScore/dByte` gradient that
    Shannon's R(D) framework operationalizes.
    """
```

**Storage format**: HDF5 chosen over numpy/pt because:
- Random access (downstream lanes only need specific frames/perturbations)
- Compression built in (5-10x smaller than uncompressed numpy)
- Standard for scientific data; loadable by external tooling
- Schema documented at top of file (no parsing ambiguity)

Schema:
```
/metadata
  device, model_shas, perturbation_types, magnitudes, eval_roundtrip
/sensitivity_maps  shape (N_pairs, n_perturbations, n_magnitudes, H_block, W_block, 4)
  dim -1: [seg_delta, pose_delta, byte_delta, score_per_byte]
/per_pair_summary  shape (N_pairs, 5)
  dim -1: [pair_idx, total_seg_dist, total_pose_dist, mean_score_per_byte, n_bytes_saved_at_target]
/aggregate
  global_mean_dScore_dByte_seg, global_mean_dScore_dByte_pose, R_curve_pose, R_curve_mask
```

**Integration with downstream lanes**:
- **Lane Ω-W-V2 V2.1**: replace `hessian = torch.ones(O,)` (uniform-pessimistic) in `tests/test_omega_w_v2_real_archive.py` with per-channel weights derived from `sensitivity_map[..., 0:2].mean()`. Expected: 5-10% additional byte savings (currently 40.98% → ~45-50%).
- **Lane 9 STC redesign**: use sensitivity map's per-region byte-cost-delta as the "fragility" weight in boundary-encoding allocation (most-fragile boundaries get more bits).
- **Lane 19 SegNet logit-margin codec**: directly use `seg_delta` per-region as the encoding allocation prior.
- **Lane 20 Ballé hyperprior renderer**: use `score_per_byte` as the auxiliary loss for the hyperprior's σ network (instead of fixed factorized prior).

### 3.3 Predicted EV

**Direct EV** (immediately after computing the maps):
- Lane Ω-W-V2 V2.1 with measured per-channel sensitivity weights: +5-10 percentage points byte savings beyond V2's 40.98%. Translates to additional -0.005 to -0.020 score reduction `[derivation]`.

**Indirect EV** (unblocks 3+ Phase 2/3 lanes):
- Lane 19 (SegNet logit-margin): predicted -0.020 to -0.080 score `[prediction]` per `project_phases_2_3_4_design_implementation_math_provenance_20260429.md` Lane 19. Today this lane is ❌ NOT STARTED partly because sensitivity maps don't exist.
- Lane 9 (STC redesign): predicted -0.020 to -0.040 `[prediction]`. Currently RED for the predictive boundary coding insight.
- Lane 20 (Ballé hyperprior): predicted -0.010 to -0.030 `[prediction]`.

**Total expected indirect EV**: **-0.050 to -0.150 score reduction across 3 Phase 2 lanes** unblocked by this $1-2 module.

**Cost**: $1-2 GPU on Vast.ai 4090 (1-3h) for all 600 pairs × 5 perturbations × 4 magnitudes = 12,000 forward passes, comparable to a single 6h training run. Local CPU is NOT acceptable for this — uses SegNet + PoseNet forward passes which CLAUDE.md MPS-CUDA drift rule forbids for any score-derived claim.

### 3.4 Caveats

1. **MPS NOT acceptable** per CLAUDE.md non-negotiable. Sensitivity maps require contest-CUDA forward passes. Tag the resulting HDF5 with `device='cuda'` + `gpu_sha=<vast.ai instance>` provenance.
2. **`eval_roundtrip=True` MANDATORY** per CLAUDE.md non-negotiable. The 384→874→uint8→384 chain affects sensitivity (some perturbations are eliminated by the roundtrip; missing this would over-state byte savings on those perturbations).
3. **Map is OPERATING-POINT-SPECIFIC** — it's the gradient at Lane G v3's current renderer.bin. If a downstream lane changes the renderer (e.g., FR-Ω quantizes weights differently), the map needs recomputing. Cost is small ($1-2 each), acceptable.
4. **Per-pair, per-region averaging may HIDE per-pixel structure** — if a downstream lane needs per-pixel sensitivity (rare), use the function as a building block; the aggregation is for storage compactness not theoretical loss.

### 3.5 Council verdict

- **Shannon**: this map IS the empirical R(D) gradient. The chain-integrity audit's Step 2 CONCERN ("per-stream R(D) curves are sparsely sampled") gets directly resolved. **APPROVE — high foundational EV.**
- **Dykstra**: the map provides the empirical convex-hull intersection points my Pareto analysis needs. Today my orthogonality verdicts are categorical ("yes/no"); with this, they become numerical. **APPROVE.**
- **Yousfi/Fridrich**: this is operationalizing UNIWARD on this scorer. The detector-informed embedding pattern Fridrich pioneered. **APPROVE.**
- **Contrarian**: cheap ($1-2) and unblocks 3 lanes. Hard to argue against. **APPROVE.**
- **Quantizr**: I never built this for my 0.33 archive but I would have if I had time. **APPROVE.**
- **Hotz**: 1 day of dev for foundational tooling that lets us measure instead of guess. **APPROVE.**
- **Selfcomp**: I picked block-FP without per-channel sensitivity. With this map, the optimal per-channel allocation is measurable, not assumed. **APPROVE.**
- **MacKay/Ballé**: this is the empirical posterior that Bayesian/hyperprior methods need to ground their priors. **APPROVE.**

**Quintet pact verdict**: 5/5 APPROVE. **CONSENSUS — BUILD THE MODULE.**

---

## 4. Q3: RL/PufferLib Direction — Verdict + Concrete First Step

### 4.1 Per-grand-council voice analysis

**Schmidhuber (compression-as-intelligence + RL lineage)**: Compression IS intelligence in my framework. RL on a codec hyperparameter space is my canonical "predictive coding + policy improvement" loop. BUT: with 4-day contest deadline (Phase 1) the action space is tiny (8-12 hyperparameters, ~50 reasonable values per axis), so a CONTEXTUAL BANDIT or PARTICLE FILTER is the right tool, not full PPO. PPO is for high-dimensional action spaces with deep credit assignment; here we have a 1-step decision problem (set hyperparameters → run codec → measure score). **Verdict: bandit/Bayesian-optimization first, RL only if action space grows.**

**Hassabis (DeepMind strategic-research)**: Compute / dev / strategic budget tradeoff. Today we have $30-200 in GPU budget and ~6 months of dev time. RL frameworks (PPO, DDPG, SAC) take 1-2 weeks of careful tuning to outperform a well-tuned bandit on a small action space. The cheapest signal will come from contextual bandits over codec hyperparameters; if that saturates, GRADUATE to RL. **Verdict: PILOT-WITH-CAP — do not invest 1-2 weeks of dev time on RL infrastructure when bandit can deliver 80% of the value in 1 evening.**

**Karpathy (training engineering)**: PufferLib's value-add is vectorized environments — `N_env × step_per_env` parallel rollouts on a single GPU. This is essential for RL where you need 100K-1M environment steps. For our problem, each "step" is a CODEC EVAL which is $0.05 of GPU and takes ~30s. So a PufferLib-vectorized 64-env setup would be 64 × $0.05 = $3.20 per macro-step. PPO needs ~10K macro-steps to converge → $32K. **TOO EXPENSIVE.** Bandit / Bayesian optimization needs ~50-100 trials → $5-10. **Verdict: PufferLib is overkill until per-step cost drops 100×. Bandit first.**

**Carmack (engineering shortcuts)**: User mandate says "compress unlimited compute". With unlimited compress time, we have time for SOMETHING optimization, but not necessarily PPO. The CHEAPEST per-step cost is a forward-pass-only score eval (no training, no gradient, no archive build). That's ~$0.005 of GPU and ~3s per call IF we already have the renderer trained. The codec hyperparameters that matter (boundary_fraction, qint_max, CRF, hyperprior σ multiplier, water-fill ladder bits, KL temperature for distill) total ~8-12 axes. A grid search over 5 values per axis is 5^10 = ~10M cells which is intractable, but a CMA-ES or Bayesian-optimization search can find good points in 100-500 trials = $0.50-2.50. PPO is unnecessary. **Verdict: CMA-ES or Bayesian optimization first; RL only if these saturate.**

**Selfcomp / Quantizr** (the actual 0.38 / 0.33 leaderboard archives): Both were HAND-TUNED with explicit derivations (Quantizr: "block-FP at 1.017 bpw because qint_max=7 + 3-bit exponent fits 88K params at 11KB"; Selfcomp: "σ=15 because Selfcomp"). The leaderboard's #1 was NOT RL-discovered. **Counter-evidence to "RL beats human intuition on this problem"**: ZERO entries in the top-5 were RL-trained. **The strongest counter-evidence is the leaderboard itself.** This is the ONLY data point we have on the question.

**Hotz**: RL is 10× the engineering cost of bandit for this problem. The cheapest first-step signal is "does even a simple bandit beat my hand-tuned hyperparameter set?". If YES, escalate; if NO, don't waste budget on RL. **Verdict: bandit first.**

### 4.2 Final verdict: **PILOT-WITH-BUDGET-CAP**

**NOT** a full RL framework upfront. The contest deadline + dev-time tradeoff says START with the cheapest signal first.

### 4.3 Concrete first step (if user approves)

**File**: `experiments/bandit_codec_search.py` (new, ~150 LOC)

**Search space** (8 axes, ~50 values each):
- `boundary_fraction` ∈ {0.02, 0.03, 0.05, 0.07, 0.10}
- `qint_max` ∈ {3, 5, 7, 10, 15, 31}
- `crf` ∈ {35, 40, 45, 48, 50, 52, 55}
- `kl_distill_weight` ∈ {0.001, 0.002, 0.005, 0.010}
- `kl_distill_temperature` ∈ {1.5, 2.0, 2.5, 3.0}
- `bls_smoothing` ∈ {0.05, 0.10, 0.15, 0.20}
- `boundary_weight` ∈ {1.0, 2.0, 3.0, 5.0}
- `hyperprior_scale_mult` ∈ {0.5, 1.0, 2.0, 4.0}

**Algorithm**: Thompson sampling with Gaussian Process (sklearn) over the 8-d hyperparameter space.

**Per-trial cost**: $0.05-0.10 (single archive build + auth eval on Vast.ai 4090, ~3-5 min).

**Budget cap**: $5 (≈50-100 trials) for the 4-hour pilot. If the best-of-trials score beats the 3-pass codex search by ≥0.005 score AT same archive bytes, GRADUATE to:
- Stage 2: BoTorch / Ax (sequential model-based optimization, 200-500 trials, $10-25)
- Stage 3 (only if Stage 2 saturates): PufferLib + PPO with the bandit-discovered points as warm starts ($50-100)

**First-step success criterion**: bandit beats Lane G v3 1.05 by ≥0.01 score OR at least equals it at <90% of archive bytes within $5 spend.

### 4.4 Why NOT defer entirely?

The contest is winding down (4-day deadline historically; under 6-month budget extended). The bandit pilot is CHEAP ($5 max), short (4 hours), and has clear graduation criteria. Even a NEGATIVE result (bandit can't beat hand-tuning) is informative: it confirms the leaderboard's evidence that this problem is tractable to human intuition + scientific reasoning, and we should keep prioritizing council-driven design over automated search.

### 4.5 Council verdict

- **Schmidhuber**: bandit is the right RL-family tool here. APPROVE pilot.
- **Hassabis**: Cost-controlled pilot is correct strategic posture. APPROVE.
- **Karpathy**: PufferLib not justified until per-step cost drops or action space grows. APPROVE bandit first.
- **Carmack**: CMA-ES would be my preference over Thompson sampling but both work; difference is small. APPROVE either.
- **Hotz**: Bandit first; RL only if bandit saturates. APPROVE.
- **Quantizr**: Test the bandit against my 0.33 archive's hyperparameters as a baseline; if bandit can recover them in <50 trials, the search is sound; if it fails, the search space is mis-specified. APPROVE pilot with this calibration step.
- **Selfcomp**: My 0.38 hyperparameters are also a calibration baseline. APPROVE.
- **Shannon**: from R(D) perspective, the bandit is searching the achievable Pareto frontier; this is what Dykstra's coordinator does analytically but at small problem scales the bandit may converge faster than ADMM. APPROVE pilot.
- **Dykstra**: the bandit COMPLEMENTS ADMM by exploring discrete-jump regions ADMM struggles with. APPROVE.
- **Contrarian**: the bandit pilot is gated to $5 cap and has clear graduation criteria. NOT a "try RL because RL is cool" gambit. APPROVE.

**Quintet pact verdict**: 5/5 APPROVE PILOT-WITH-CAP. **CONSENSUS.**

**Verdict: PILOT-WITH-CAP** ($5 budget, 4-hour smoke). DEFER full RL/PufferLib until bandit saturates or sensitivity-map (Q2) suggests a richer action space.

---

## 5. Q4: Round 11 Bug Fixes — Per-Fix Decision

User mandate: "fix all bugs regardless of severity". Council reviews each:

### 5.1 Q4A: Joint-ADMM Nesterov-averaging bias

**Bug location**: `src/tac/joint_admm_coordinator.py:589-615`

**Symptom (Round 11 finding from Council E §2.3)**: `lam_avg` (Nesterov-averaged dual) lags the true KKT dual by ~3× → 7% budget overshoot when re-querying streams at the final operating point.

**Root cause**: lines 589-615 use `bytes_avg` (Nesterov-averaged primal — converges to KKT for non-smooth subgradients) AND `lam_avg` (the dual averaged via the same Nesterov rule). The PRIMAL averaging is correct for KKT convergence; the DUAL averaging UNDER-WEIGHTS the most recent dual updates which is precisely what we need for the FINAL re-query (the final lam reflects the converged Lagrangian).

**Council verdict**: **APPROVE FIX with MODIFICATION**

**Modified prescription** (Boyd, Dykstra, MacKay):
- Round 11's prescription "use final non-averaged `lam` instead of `lam_avg`" is the right direction BUT misses a subtle case: when the coordinator did NOT converge (early-stop on `restart_now` + `it < cfg.max_iters`), the final `lam` may be MID-trajectory and not representative of the KKT dual. The averaged `lam_avg` is more robust in that case.
- **Correct fix**: use the FINAL `lam` if `converged == True`; use `lam_avg` if `converged == False` (the divergent case). Document this clearly.
- Add a regression test that constructs a problem where `lam_avg / lam_final = 0.33` (the 3× lag) and asserts the re-queried bytes-sum is within `cfg.byte_budget * 1.01`.

**Responsible council member**: **Boyd (grand-council)** — this is operational ADMM convergence theory; Boyd is the canonical voice. Dykstra co-signs (the dual-averaging vs final-dual distinction is a Pareto-feasibility question).

**Estimated effort**: 30 min code change + 1h test write.

### 5.2 Q4B: JointADMMConfig.rho_init=1.0 default unstable for low-curvature problems

**Bug location**: `src/tac/joint_admm_coordinator.py:216` (`rho_init: float = 1.0`)

**Symptom**: For problems where `2*a < 0.01` (smooth quadratic with low curvature like `s1` in the 4-stream non-convex test), `rho_init=1.0` causes the proximal step to OVER-CORRECT (b_unconstr = b_opt - dual / (2*a) blows up when 2*a is small). Restart logic fires repeatedly; convergence is slow.

**Council verdict**: **APPROVE FIX with MODIFIED PRESCRIPTION**

**Two acceptable fixes**:

**Option A (Boyd's preference): adaptive rho_init**
```python
@classmethod
def from_curvature_estimate(cls, byte_budget: float, curvature_min: float, **kwargs) -> "JointADMMConfig":
    """Construct config with rho_init scaled to problem curvature.

    Boyd §3.4.1 recommends rho_init ~= sqrt(curvature_max * curvature_min) for
    well-conditioned ADMM. When curvature_min is unknown, use a conservative
    rho_init = max(0.01, 0.1 * curvature_min).
    """
    rho_init = max(0.01, 0.1 * curvature_min)
    return cls(byte_budget=byte_budget, rho_init=rho_init, **kwargs)
```

**Option B (Carmack's preference): adaptive rho during the FIRST ITERATION**
- Run iteration 0 with `rho_init=1.0` (current default)
- Measure `primal_residual_0 / dual_residual_0`
- If the ratio is > 100 OR < 0.01, RESET `rho` to `sqrt(primal_residual_0 * dual_residual_0)` and restart from iteration 0
- This auto-adapts to ANY problem without needing curvature_min input

**Council verdict**: **Option B preferred** (no API change required; works on any problem; honest "first iteration is exploratory" pattern). Add regression test that constructs a low-curvature problem AND asserts ADMM converges within 50 iterations with no manual rho tuning.

**Responsible council member**: **Carmack (grand-council)** — this is the engineering-shortcut "make it work without configuration" pattern; Carmack is the canonical voice. Boyd co-signs the underlying theory.

**Estimated effort**: 1h code change + 1h test write.

### 5.3 Q4C: J_JBL_DILATED_H64 profile routing-table gap

**Allegation**: `profiles.py:2837` declares `loss_mode='jbl'`, but `tac/training.py:91` (TrainConfig Literal type) only allows `{standard, temperature, focal_ste, kl_distill, pcgrad, feature_match, segnet_kl, posenet_embedding}` → potential dispatch gap.

**Council verdict**: **MISDIAGNOSED — NO BUG TO FIX. KEEP J_JBL_DILATED_H64.**

**Why the diagnosis is wrong** (verified by inspection):

1. `experiments/train_renderer.py:357` declares `loss_mode` as a CLI choice: `choices=["standard", "kl", "jbl"]` — `'jbl'` is a registered choice.
2. `experiments/train_renderer.py:3083-3113` implements the dispatch: when `args.loss_mode == "jbl"`, the KL-distill auxiliary is replaced with `combined_jbl_distill_loss` from `tac.losses_jbl`. The dispatch fires at training time.
3. `experiments/train_renderer.py` does **NOT use `TrainConfig`** — it uses raw argparse + a manual `_resolve()` helper. `grep "TrainConfig" experiments/train_renderer.py` returns **0 matches**.
4. The `TrainConfig.loss_mode` Literal in `tac/training.py:91` is for the `Trainer` class used by OTHER training scripts (`train_distill.py`, `train_imp_cycle.py`, etc.). Those scripts don't use the J_JBL profile.
5. The `J_JBL_DILATED_H64` profile is consumed by `train_renderer.py`, NOT by `Trainer`. The Pydantic Literal is in a DIFFERENT call path.

**No bug exists.** The two `loss_mode` namespaces (train_renderer.py's argparse choices vs Trainer's Pydantic Literal) are independent by design.

**Defense-in-depth recommendation** (not a fix — a clarity improvement):
- Add a comment in `tac/training.py:91-94` noting that this Literal is for the `Trainer` class only, and that `train_renderer.py` has its OWN `loss_mode` choice list at line 357 which includes `'jbl'`.
- This prevents future agents from mis-diagnosing the bug class as Q4C did.

**Alternate verdict** (if user wants the Literal extended for forward compatibility): add `'jbl'` to the Literal at line 91. ~1 line change. Documented as a "doesn't fix anything today; future-proofs the Trainer if a future caller wires JBL through Trainer".

**Responsible council member**: **Contrarian** — this verdict prevents an unnecessary API change driven by a misdiagnosed bug. Hotz co-signs (engineering hygiene: don't change APIs to fix non-bugs).

**Estimated effort**: 0 (no fix). Optional 5-minute comment addition.

### 5.4 Summary of Q4 fixes

| Fix | Decision | Responsible council member | Effort |
|---|---|---|---|
| 4A: Nesterov dual bias | **APPROVE WITH MODIFICATION** (use final lam if converged; use lam_avg if not) | Boyd (Dykstra co-signs) | 30 min code + 1h test |
| 4B: rho_init default | **APPROVE OPTION B** (adaptive on first iteration; no API change) | Carmack (Boyd co-signs) | 1h code + 1h test |
| 4C: J_JBL routing gap | **MISDIAGNOSED — NO FIX** (different namespaces; J_JBL works correctly today) | Contrarian (Hotz co-signs) | 0 (optional 5min comment) |

**Total Round 11 fix effort**: ~3-4 hours of code + tests. **All can land before any GPU dispatch.** Counter resets per the recursive review protocol; expect 2-3 fresh review rounds to reach 3-clean-pass after these land.

---

## 6. Updated 24-48h Dispatch Order with Budget Tracking

```
T+0h    [BG running] HM-S returns from Vast.ai (~6h after launch)
T+0h    [DEV] Land Q4 fixes 4A + 4B (~3-4h dev)
T+0h    [DEV] Land OWV2 inflate handler (Option F, ~1-2h dev)
T+2h    [LOCAL] Build Lane G v3 + Ω-W-V2 archive locally (Option A prep)
T+2h    [REVIEW] Round 11 + Round 12 + Round 13 adversarial reviews of Q4 fixes (BG codex)
T+3h    [DISPATCH] Lane G v3 + Ω-W-V2 contest-CUDA auth eval (Option A, $0.50, 15min)
T+3h    [DISPATCH] Sensitivity-map computation on Vast.ai 4090 (Option E, $1-2, 1-3h)
T+3.5h  [HARVEST] Option A result lands → Lane G v3 + Ω-W-V2 = expected 0.97 [contest-CUDA]
T+5h    [HARVEST] Option E result lands → sensitivity_map.h5 saved
T+5h    [LOCAL] Land Lane Ω-W-V2 V2.1 with measured per-channel sensitivity weights
T+6h    [DISPATCH] Lane G v3 + Ω-W-V2 V2.1 contest-CUDA auth eval ($0.50)
T+6.5h  [HARVEST] Lane G v3 + Ω-W-V2 V2.1 = expected 0.95-0.96 [contest-CUDA]
T+8h    [LOCAL] Build PD-V2 + LCT bolt-on (Option B, $0)
T+10h   [DISPATCH] Lane G v3 + Ω-W-V2 V2.1 + PD-V2 + LCT contest-CUDA auth eval ($0.50)
T+12h   [DECIDE] FR-Ω dispatch gate: HM-S in band? Ω-W-V2 stack confirmed?
        ├─ YES + YES: dispatch FR-Ω ($1.50, 6h) targeting -0.05 to -0.10 OVER current
        └─ NO either: defer FR-Ω; pivot to Lane 19 (SegNet logit-margin codec)
T+18h   [DISPATCH if approved] Bandit pilot (Q3, $5 cap, 4h smoke)
T+24h   [HARVEST] FR-Ω or bandit results
T+36h   [REVIEW] Council Round 14+15+16 on stacked artifacts; 3-clean-pass gate
T+48h   [DECISION] Phase 2 lane priorities recomputed based on Phase 1 stacked frontier
```

**Budget tracking**:
- Vast.ai: $0.50 (Option A) + $1.50 (Option E mid) + $0.50 (V2.1) + $0.50 (PD-V2 stack) + $1.50 (FR-Ω if approved) + $5 (bandit if approved) = **$9.50 max** for Phase A. **Well under $24 hard cap.**
- Modal: $0 (no Modal dispatch in Phase A)
- Dev time: ~6-8h human/agent time (all parallelizable; subagents OK per CLAUDE.md)

**Risk-adjusted EV**: Lane G v3 1.05 → ~0.93-0.97 contest-CUDA frontier expected within 48h, with sensitivity-map foundation ready for Phase 2.

---

## 7. Council Roll Call

**Quintet pact (binding-decision leadership) — APPROVE all four Q-decisions above**:

1. **Shannon (LEAD, Information Theory)**: The portfolio's highest-confidence dispatch is Option A (Lane G v3 + Ω-W-V2), grounded in `[derivation]`-grade rate math + `[empirical]`-grade codec savings. The sensitivity-map module (Q2) directly resolves the chain-integrity audit's Step 2 CONCERN ("per-stream R(D) curves are sparsely sampled"). The bandit pilot (Q3) is the right cost-controlled exploration of the hyperparameter space — RL is overkill given action-space size and per-step cost. Round 11 Fix 4A is a real Boyd/Nesterov subtlety; 4B is a Carmack engineering-shortcut; 4C is misdiagnosed. **VERDICT: APPROVE all four.**

2. **Dykstra (CO-LEAD, Convex Feasibility)**: Option A is a Pareto-improving point on the convex hull. Option E (sensitivity maps) provides empirical convex-hull intersection points for downstream Pareto-frontier analysis. Q3 bandit is a complement to ADMM (explores discrete-jump regions ADMM struggles with). Q4A's "use final lam if converged, lam_avg if not" preserves Pareto-feasibility under both convergence and divergence cases. **VERDICT: APPROVE all four.**

3. **Yousfi (Challenge creator, Steganalysis lineage)**: The contest IS inverse steganalysis. Sensitivity maps (Q2) operationalize Fridrich's UNIWARD framework on this specific scorer. Option A replicates Quantizr's block-FP-at-1.017bpw paradigm with empirical proof. **VERDICT: APPROVE all four.**

4. **Fridrich (UNIWARD/SRM/HUGO author)**: Sensitivity maps directly implement my "errors in textured regions are undetectable" principle as a per-region byte-cost gradient. Combined with Option A's Hessian-aware allocation, the stack approaches the steganalysis-detector's blind spot. The bandit pilot (Q3) explores the same hyperparameter space my detectors do — bandit-discovered hyperparameters are likely to land at the detector's blind spots. **VERDICT: APPROVE all four.**

5. **Contrarian (Veto)**: I MAINTAIN my Council F discipline: no `[prediction]`-tagged dispatches without empirical anchors. Option A has BOTH `[derivation]`-grade rate math AND `[empirical]`-grade codec savings — it satisfies my discipline. Option E is cheap ($1-2) and high-EV; no veto. Q3 bandit is gated to $5 cap with clear graduation criteria; no veto (this is NOT "try RL because RL is cool"). Q4A is a real bug; my modification (use final lam if converged, lam_avg if not) preserves the divergent-case safety. Q4B Option B is the right pattern; Q4C is misdiagnosed (I called this one out — the two namespaces ARE independent). **VERDICT: APPROVE all four with the modifications I noted.**

**Co-members (permanently active, no veto but full voice) — APPROVE all four**:

6. **Quantizr (Adversarial leaderboard reality check)**: My 0.33 archive uses block-FP at 1.017 bpw — Option A is precisely my paradigm with empirical proof. Sensitivity maps (Q2) are what I would have built had I had more time. Bandit pilot (Q3) calibration step: test it against my hyperparameters; if it recovers them in <50 trials, the search is sound. Round 11 Q4 fixes don't affect my paradigm but are sound bug fixes. **VERDICT: APPROVE all four.**

7. **Hotz (Engineering shortcuts)**: Option A is the cheapest measurement in the portfolio ($0.50 + 30 min OWV2 dev). Sensitivity maps (Q2) at $1-2 unblock 3 lanes — engineering ROI is excellent. Bandit pilot (Q3) is a 1-evening prototype with $5 cap; PufferLib not justified. Round 11 fixes 4A and 4B are real bugs; 4C is misdiagnosed (I'm with Contrarian on this). **VERDICT: APPROVE all four.**

8. **Selfcomp (szabolcs-cs, working 0.38 anchor)**: Option A maps directly to my paradigm. Sensitivity maps (Q2) are the score-aware allocation step I never operationalized. Bandit pilot (Q3) — calibrate against my 0.38 hyperparameters as a baseline. **VERDICT: APPROVE all four.**

9. **MacKay (Memorial seat, Information Theory + Bayesian Inference + Learning Algorithms)**: Option A is a strict MDL improvement (rate cost reduced; posterior approximation quality bounded). Sensitivity maps provide the empirical posterior `p(score | byte_allocation)` that Bayesian model selection (Lane 16) requires. Q4A's "use final lam if converged" preserves the MDL-informed dual at the converged point. **VERDICT: APPROVE all four.**

10. **Ballé (2018 entropy bottleneck SOTA)**: Option A validates Ω-W-V2's static-histogram terminal at the score level. Sensitivity maps unlock the V3 hyperprior amortization trigger (Lane 20). Bandit pilot — small-scale; my hyperprior network is for large-scale. **VERDICT: APPROVE all four.**

**Grand-council voices consulted on specialty**:

- **Karpathy (RL/training engineering, on Q3)**: PufferLib's value-add is vectorized environments — essential at high per-step throughput. Our per-step cost is too high ($0.05-0.10) for PufferLib to deliver value today. Bandit first; PufferLib only if per-step cost drops 100×. **APPROVE PILOT-WITH-CAP for Q3.**

- **Hassabis (DeepMind strategic-research, on Q3)**: Cost-controlled pilot is correct strategic posture. RL frameworks take 1-2 weeks of careful tuning to outperform bandit on small action spaces. Don't invest 1-2 weeks of dev time when bandit can deliver 80% of the value in 1 evening. **APPROVE PILOT-WITH-CAP for Q3.**

- **Schmidhuber (compression-as-intelligence + RL lineage, on Q3)**: Bandit is the right RL-family tool for 1-step decision problems with small action spaces. PPO is for high-dimensional credit assignment; here we have a 1-step decision (set hyperparameters → measure score). **APPROVE PILOT-WITH-CAP for Q3.**

- **Carmack (engineering shortcuts, on Q3 and Q4B)**: For Q3, CMA-ES would be my preference over Thompson sampling; both work. For Q4B, my Option B (adaptive rho on first iteration) is the right "make it work without configuration" pattern. **APPROVE Q3 PILOT and Q4B Option B.**

- **Boyd (ADMM operational, on Q4A and Q4B)**: For Q4A, the dual-averaging vs final-dual distinction matters: use final `lam` when converged; `lam_avg` when divergent. For Q4B, Option B (adaptive on first iteration) is operationally cleanest; Option A (curvature_min input) requires the caller to know curvature which is rare. **APPROVE Q4A WITH MODIFICATION and Q4B Option B.**

- **Tao (first-principles math, on Q2)**: The sensitivity map IS the empirical R(D) gradient at the operating point. The `[seg_delta, pose_delta, byte_delta]` triple is a 2-d Pareto descent direction; combined with the byte-cost it gives the marginal `dScore/dByte`. This is the foundational quantity. **APPROVE Q2 module design.**

- **Filler (STC, on Q2 and Q1 indirect)**: Sensitivity maps directly enable STC redesign (Lane 9 predictive boundary coding). The per-region byte-cost-delta is the "fragility" weight that STC's syndrome-trellis paths should respect. **APPROVE Q2 module design + acknowledge it unblocks Lane 9.**

- **Mallat (wavelet, on Q2 indirect)**: Sensitivity maps would give the per-coefficient weighting Lane 11 (wavelet residual codec) needs. Currently Lane 11 uses uniform L_inf bounds; with the map, it could use score-aware bounds. **APPROVE Q2 module design as foundational for Lane 11 also.**

**Quintet consensus**: 5/5 APPROVE all four Q-decisions. Co-member consensus: 5/5 APPROVE. Grand-council voices consulted: 7/7 APPROVE. **FULL CONSENSUS.**

---

## 8. Cross-references

- Council F (Lane re-train EV + Ω-W-V2 + ADMM consult): `.omx/research/council_f_retrain_ev_validation_admm_consult_20260429.md`
- Council E (Round 5 grand battleplan): `.omx/research/council_grand_battleplan_round5_20260429.md`
- Council Round 8/9/10 (3-clean-pass gate completion): `.omx/research/council_round{8,9,10}_*.md`
- Council Chain-Integrity Audit (Part G surfaced Option A): `.omx/research/council_chain_integrity_audit_20260430.md`
- Lane G v3 1.05 [contest-CUDA] anchor: `experiments/results/lane_g_v3_landed/contest_auth_eval.json`
- Ω-W-V2 real-archive 40.98% [empirical]: `src/tac/tests/test_omega_w_v2_real_archive.py`
- Codec stacking + score arithmetic: `project_codec_stacking_composition_canonical_orders_20260429.md`
- Phase 1 dispatch verdict: `project_phase1_dispatch_verdict_20260429.md`
- Phases 2/3/4 lane scoping: `project_phases_2_3_4_design_implementation_math_provenance_20260429.md`
- Skunkworks council quintet pact: `feedback_skunkworks_council_shannon_dykstra_quintet_lead_20260429.md`
- 10-member inner + grand council: `feedback_council_10_member_inner_grand_council_advisory_20260429.md`
- Joint-ADMM coordinator (Q4A + Q4B targets): `src/tac/joint_admm_coordinator.py:589-615` + `:216`
- J_JBL_DILATED_H64 profile (Q4C false alarm): `src/tac/profiles.py:2832-2846`
- train_renderer.py loss_mode dispatch (Q4C verification): `src/tac/experiments/train_renderer.py:357 + 3083-3113`
- Forensics module (Q2 building block): `src/tac/forensics.py`
- Pair difficulty map (Q2 building block): `experiments/pair_difficulty_map.py`
- UNIWARD texture (Q2 building block): `src/tac/uniward_texture.py`
