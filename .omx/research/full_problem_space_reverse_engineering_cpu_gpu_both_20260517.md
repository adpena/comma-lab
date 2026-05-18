---
title: "Full Problem Space Reverse-Engineering: CPU/GPU Hardware Axes, Scorer Mechanics, Contest Runtime Architecture"
date: 2026-05-17
author: explore-subagent-full-problem-space-revrng
lane: lane_full_problem_space_reverse_engineering_20260517
horizon_class: frontier_pursuit
---

# Executive Summary

The contest is fundamentally a SCORER MINIMIZATION problem over a dual-axis (CPU/CUDA) hardware constraint, not an architecture search. The scoring function `S = 100·d_seg + √(10·d_pose) + 25·R` decomposes into three inelastic Pareto corners: SegNet (EfficientNet-B2 on frame-1, stride-2 stem blindspot below 192×256) @ marginal 100, PoseNet (FastViT-T12 + RepMixer on 12-ch YUV6 pair) @ marginal ~271 at frontier, Rate @ marginal ~6.66e-7/byte. The CPU/CUDA axis exhibits **opposite device-scaling across substrate families** (A1: 5.18× pose-worse on CUDA; PR106 r2: 5.1× pose-better on CUDA), indicating substrate-specific numerical trajectories through scorer-kernel precision boundaries. The video dataset (1200 frames @ 384×512, 600 frame-pairs) exhibits **odd-frame compression opportunity** (Quantizr innovation: encode 600 masks, warp 600), delegating 50% of temporal bandwidth to differentiable ego-motion estimation. The theoretical floor per Blahut-Arimoto/Boyd convex duality is `S* ∈ [0.10, 0.15]` given the contest's rate-distortion constraints and frozen scorer numerics. Unoperationalized research-grade insights from `.omx/research/*.md` include: Tishby information-bottleneck pure (U-DIE-KL framing), Wyner-Ziv side-information (L5 Time-Traveler), Atick-Redlich cooperative-receiver (ATW codec), Rao-Ballard predictive-coding (Cathedral autopilot), Carmack-Hotz strip-everything (NSCS02/06 minimalism). The hardware-axis twin (~49 substrate trainers × 2 devices) remains **fundamentally separated at the contest runner level** (GHA Linux/CPU vs Modal/Lightning T4 CUDA), preventing within-trainer gradient-flow CPU/CUDA comparison and forcing an empirical-anchor dual-eval discipline that consumes ~$0.10-0.15/candidate for paired validation.

---

## 1. The Scoring Function Decomposition: Inelastic Corners & Pareto Frontier

### 1.1 Canonical formula (upstream/evaluate.py:92)
```
S = 100 · d_seg + √(10 · d_pose) + 25 · (archive_bytes / 37_545_489)
```

### 1.2 Elasticity at frontier operating point (PR106 r2, d_pose ≈ 3.2e-5)

Per **expert_team_aerospace_stealth_analytic_alien_tech_20260513.md** §3:

| Component | Marginal value at frontier | Elasticity |
|-----------|---------------------------|-----------|
| d_seg | 100 | Constant (linear, inelastic) |
| d_pose | √(10·d_pose)' = 5/√(10·d_pose) ≈ **271** | **Hyperbolic:** diminishing at d_pose → 0 |
| Rate (R) | 25 × (1 / 37_545_489) ≈ **6.66e-7** | Constant per byte; swamps to zero at budget >500KB |

**Interpretation**: The frontier lies in pose-dominated territory where pose marginal ≈ 2.71× SegNet's per-unit distortion. Bytes beyond SegNet ε-improvement are **overwhelmingly directed at pose-axis reduction** (per L5 Time-Traveler per-pair pose allocation). The rate term is a **rounding error beyond 200KB archive size**.

### 1.3 Where diminishing returns peak: three Pareto corners

- **Corner 1 (pure rate)**: `S ≈ 0.21` at perfect SegNet (d_seg=0) + perfect PoseNet (d_pose=0) + maximal rate (R=1.0). Not achievable.
- **Corner 2 (pose-dominated)**: Current frontier `S ≈ 0.193-0.205` at d_seg ≈ 0.002, d_pose ≈ 3e-5, R ≈ 0.005. **THIS CORNER is where the contest lives.**
- **Corner 3 (information-theoretic floor)**: `S* ≈ 0.10` per Blahut-Arimoto solver in `src/tac/symposium_impls/blahut_arimoto_theoretical_floor.py` (Cover & Thomas, Boyd convex duality). This floor is **unreachable without architecture class-shift** (current HNeRV family estimated at 0.171 floor per Z1 Tier-C ablation; alien-tech staircase candidates estimated 0.10-0.15).

**Citation**: `upstream/evaluate.py:92`, `tac.symposium_impls.blahut_arimoto_theoretical_floor`, `expert_team_aerospace_stealth_analytic_alien_tech_20260513.md §3`.

---

## 2. Scorer Architecture Quirks: SegNet Stride-2 Blindspot + PoseNet RepMixer Precision

### 2.1 SegNet (upstream/modules.py, smp.Unet on EfficientNet-B2)

**Pipeline:**
1. Input: frame-1 only (index -1 of T=2 sequence, per modules.py:117); frame-0 is **completely ignored**
2. Interpolate 874×1164 → 384×512 (bilinear, Catalog #180 DIE impact)
3. EfficientNet-B2 stem: stride-2 convolution (Catalog #163 cited in aerospace_stealth memo as "S2SBS blindspot")
4. Output argmax decision: `argmax(out[:, 5, :, :])` (5 classes), distortion = fraction of argmax-disagreement pixels

**Critical insight** (from aerospace_stealth_analytic_alien_tech_20260513.md §5.2, citing Yousfi DIE):
- Stride-2 stem in EfficientNet-B2 creates **texture-blindness below 192×256 effective resolution**
- Per-pixel logit margins `m(p) = logit_1 - logit_2` are STABLE (±32 RGB noise) for ~97% of pixels
- Only 3-5% of pixels are **argmax-boundary pixels** (logit margin < 2 standard deviations)
- SABOR encoding (Technique A1) exploits this: encode equivalence-class INDEX (cheap) for interior pixels; lossless RGB (expensive) for boundary pixels only

**Archive encoding implication**: Quantizr (0.33 baseline) + PR101 (0.193) + alien-tech memos all point to **per-pixel boundary classification** as the next frontier move; wasting bytes on texture-interior RGB is the dominant compression inefficiency.

**Citation**: `upstream/modules.py:117-122`, `expert_team_aerospace_stealth_analytic_alien_tech_20260513.md §5.1-5.2`, `s2sbs_blindspot_audit_20260513.md`.

### 2.2 PoseNet (upstream/modules.py, FastViT-T12 + Hydra head)

**Pipeline:**
1. Input: **BOTH frames** (T=2 at shape B×2×3×874×1164), rearranged to 12-channel YUV6 @ 192×256 (modules.py:73-74)
2. YUV6 format: 6 channels per frame (4 luma + 2 chroma subsampled × 2 frames = 12 total; Bayer-pattern semantics per rgb_to_yuv6 patch)
3. FastViT-T12 backbone: RepMixer (convolutional, NOT attention-based; Catalog #166 hardware-friendly)
4. Hydra head: ResBlock + multi-head output (only first 6 of 12 pose dims are scored per evaluate.py:79)
5. Distortion: MSE on first 6 dims only; dims 7-12 are **wasted/free** per contest rules

**Critical insight** (from cpu_cuda_xray_synthesis_20260511.md):
- Device-axis flip between A1 (5.18× worse on CUDA) and PR106 r2 (5.1× better on CUDA) suggests **per-layer FastViT numerical precision** is substrate-dependent
- Compound factor `(1+ε)^L` across L≈256 FastViT blocks can reach 5× at ε ≈ 0.0065 per-block (Lipschitz triangle inequality upper bound)
- CPU sequential accumulation vs CUDA fused kernels create **additive-vs-reordered floating-point** divergence (TF32 matmul, cuDNN heuristics)

**Archive encoding implication**: Pose dimension 7-12 are FREE; per-pair pose deltas (L5 innovations) exploit this by encoding selective pose-residual corrections only where (d_pose/db) > λ.

**Citation**: `upstream/modules.py:70-80`, `cpu_cuda_xray_synthesis_20260511.md §8`, `feedback_cuda_cpu_pose_drift_mechanism_deep_dive_20260508.md`.

---

## 3. CPU vs GPU Hardware-Axis Decomposition: Beyond Two-Pass Additive

### 3.1 The opposite-scaling empirical anchor (device_axis_paired_anchor_matrix_20260511.md)

| Substrate | Pose CUDA/CPU ratio | Seg CUDA/CPU ratio | Winning axis |
|-----------|------|------|------|
| A1 (HNeRV, score-gradient training) | 5.18× worse CUDA | 1.18× worse CUDA | **CPU** |
| PR106 r2 (latent sidecar additive) | **5.1× BETTER CUDA** | 1.017× (≈parity) | **CUDA** |

**The core mystery**: Format0d two-pass additive correction (inflate.py:549-575) exhibits **opposite device scaling** from PR101 GOLD, suggesting the algorithm itself (not just numerical precision) is device-sensitive.

### 3.2 Enumerated drift mechanisms (comprehensive catalog, cross-ref feedback_cpu_cuda_xray_p5_landed_20260511.md + cpu_cuda_xray_synthesis_20260511.md)

1. **AVVideoDataset NVDEC (CUDA) vs PyAV (CPU)** — loader-drift tool deployed; captures raw RGB SHA-256 divergence pre-scorer. Expected 1-3% frame decode error from hardware accelerator vs libav differences.

2. **rgb_to_yuv6 in-place mutation vs detached clone** — PR95/PR101 patch at modules.py monkey-patches upstream helper because the original `@torch.no_grad()` in-place YUV conversion severs PoseNet gradients. CPU path uses PyAV-decoded RGB (no in-place mutation risk); CUDA path uses DALI-decoded RGB + patched YUV6. Different code paths, different precision.

3. **F.interpolate bicubic kernel divergence** — PyTorch bicubic upsampling uses different kernel implementations on CPU (XNNPACK) vs CUDA (cuDNN). Per Catalog #180 (DIE impact), ~0.5-1% logit-magnitude drift per interpolation stage expected.

4. **Clamp / round / uint8 cast non-bit-identity** — The uint8 bottleneck (384→874 inference→clamp→uint8→384 preprocess) is simulated in training via `eval_roundtrip=True` per CLAUDE.md. CPU rounding modes (round-to-nearest vs banker's rounding) differ from CUDA atomic-add-then-cast semantics; ~1-2 LSB per sample.

5. **TF32 vs FP32 matmul** — NVIDIA A100/H100 default TF32 (Tensor Float 32: 10-bit mantissa, 8-bit exponent, 32-bit format) in cuBLAS GEMM. CPU uses full FP32. Per NVIDIA docs, TF32 trades ~1e-2 relative error for ~2-4× throughput. FastViT-T12 has ~1000 matmuls per forward; compound error ~1e-1 possible.

6. **cuDNN heuristic selection** — cuDNN runtime selects "best" convolution algorithm per input shape, which varies per GPU model (T4 vs A100 heuristics differ). CPU uses XNNPACK single-deterministic path.

7. **Ground-truth video decode (DALI/NVDEC vs PyAV)** — CUDA contest runner uses DALI/NVDEC (hardware decoder); CPU local uses PyAV (software). Codec is identical (H.264 baseline), but frame-reordering, color-space calibration (ITU-R BT.709 vs BT.601), and motion-picture-expert-group MPEG-2 vs AVC can diverge by ~1-3% peak-signal-ratio.

8. **Inflate-device-fork bug class (Catalog #205)** — Format0d inflate.py (lines 549-575) applies two sidecar correction passes. If the latent tensor indexing `latents[p, d] += delta` uses different atomic or scattering ops on CPU (sequential) vs CUDA (potential warp-level parallelism), the accumulated delta order changes. Expected ~0.1-1% distortion drift if corrections overlap on same (p, d).

9. **Per-archive substrate-specific routing** — Each substrate trainer (49 total) has its own SegNet/PoseNet scorer-loading path. Some may use `torch.jit.script`, others eager mode. Some may use `@torch.no_grad()` globally; others use `requires_grad=False` per-parameter. Inconsistency compounds at >100 substrate variance.

### 3.3 Verdict structure (cpu_cuda_xray_synthesis_20260511.md §3)

The three xray tools (loader_drift, segnet_layer_drift, posenet_layer_drift) produce a **4-cell verdict**:

- **Verdict A (Loader-dominated)**: large LSB delta + no per-block compounding → decoder + interpolation drift ~0.5-2%
- **Verdict B (Scorer-forward-dominated)**: byte-identical loader + per-block ε compounding ≈ observed final ratio → kernel numerics (TF32 / cuDNN / atomic-add) accumulate to 5×
- **Verdict C (Threshold-geometry)**: byte-identical + small ε but final-output large → scorer crosses decision boundary (argmax / MSE thresholds)
- **Verdict D (Mixed/coupled)**: both mechanisms contribute

**Current status**: Verdict A/B/C/D not yet **empirically decided** for A1 vs PR106 r2 on shared-input tensor (CUDA captures pending $0.05 Modal gate).

**Citation**: `cpu_cuda_xray_synthesis_20260511.md`, `cpu_cuda_drift_mechanism_validity_guard_20260511_codex.md`, `Catalog #205`.

---

## 4. The Video Data Itself: 1200 Frames, 600 Pairs, Ego-Motion Dominance, Odd-Frames Innovation

### 4.1 Canonical dataset structure (upstream/videos/0.mkv + contest spec)

- **Duration**: 1200 consecutive frames @ ~30 fps = 40 seconds of continuous driving footage
- **Resolution**: 384×512 (contest-scaled; original 874×1164 per camera_size in frame_utils.py)
- **Temporal structure**: 600 frame-pairs (frame[2i], frame[2i+1]) for i ∈ [0, 599]. Evaluation loop iterates over 600 pairs, NOT 1200 single frames.
- **Camera geometry**: consecutive driving footage → ego-motion dominates the pose dimension (vehicle translation/yaw; road-relative yaw/pitch secondary)

### 4.2 The Quantizr odd-frames innovation (cited in multiple memos, not yet operationalized in our codebase)

**Insight**: Encode ONLY the 600 frame-0 masks (lossless or quantized); for frame-1, **warp the decoded frame-0 mask using ego-motion estimates**, avoiding 50% of mask-codec bandwidth.

**Implementation path** (per alien_tech_reverse_engineering_pr106_format0_family_20260517.md + aerospace_stealth_analytic_alien_tech_20260513.md):
1. Estimate ego-motion (yaw/pitch/roll/x/y/z) from frame-pair PoseNet or RAFT (see raft_radial_openpilot_pose lane, tier 90)
2. Encode frame-0 mask (600 masks)
3. Encode ego-motion deltas (12-dim per pair × 600 = ~7.2KB @ 1.2 bytes/delta with quantization)
4. At inflate: warp decoded frame-0 mask by ego-motion to approximate frame-1 mask
5. Rate savings: 50% of mask bytes (~5-10KB per archive) at cost of warp-error distortion
6. **Predicted ΔS**: −0.003 to −0.008 per aerospace_stealth memo §5.3 (Tacit Blue analogy: "decoy targets")

**Status**: Not yet shipped. Requires (a) differentiable ego-motion estimator, (b) warp kernel, (c) loss calibration for warp-error vs mask-codec rate trade-off.

**Citation**: `alien_tech_reverse_engineering_pr106_format0_family_20260517.md`, `aerospace_stealth_analytic_alien_tech_20260513.md §5.3`, `raft_radial_openpilot_pose` lane (tier 90, not yet materialized).

### 4.3 The half-frame disaster lesson (CLAUDE.md CATASTROPHIC FAILURES 2026-04-21)

**Prior failure**: Early PR attempted frame-0-only encoding (ignoring frame-1 completely). This violated the 600-pair evaluation protocol, producing garbage scores.

**Lesson**: The contest REQUIRES both frames per pair for PoseNet computation. Masks can be frame-1-only (SegNet uses frame-1 only per modules.py:117), but poses MUST use both. Odd-frames warping respects this by always decoding frame-1 (via warp) for PoseNet input.

**Citation**: `CLAUDE.md` "CATASTROPHIC FAILURES 2026-04-21" section.

---

## 5. Contest Runner Architecture: GHA CPU vs Modal/Lightning CUDA, Dual-Axis Non-Negotiable

### 5.1 Canonical contest runner endpoints

**CPU evaluation (public-leaderboard authority)**:
- Platform: GitHub Actions `ubuntu-latest` (Linux x86_64, AMD/Intel CPU)
- Dataset loader: `AVVideoDataset` (PyAV software decoding)
- Scorer preprocess: canonical `modules.py` eager-mode CPU path
- Evaluation script: `upstream/evaluate.py` with `--device cpu`
- Runtime: ~10-15 minutes per archive
- Cost: free (GHA included)

**CUDA evaluation (private-leaderboard authority)**:
- Platform: Modal / Lightning (T4 / A100 / H100, varies by provider)
- Dataset loader: `DaliVideoDataset` (NVIDIA NVDEC hardware decoding)
- Scorer preprocess: canonical `modules.py` CUDA path with DALI distributed backend
- Evaluation script: `upstream/evaluate.py` with `--device cuda`
- Runtime: ~2-5 minutes per archive (10× faster due to NVDEC + batching)
- Cost: ~$0.05-0.15 per run (T4: $0.35/hr, 5 min = $0.03; A100: $1.20/hr, 5 min = $0.10)

### 5.2 The non-negotiable dual-axis submission discipline (CLAUDE.md, AGENTS.md)

**Standing directive** (per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE"):

> Any promotion claim REQUIRES paired [contest-CPU] + [contest-CUDA] evals on the EXACT same archive.zip.

This is NOT a suggestion; it is a hard gate per Catalog #220 + Catalog #233 L1→L2 promotion.

**Why**: The device-axis flip (A1 vs PR106 r2) proves that within-trainer CPU/CUDA comparison is insufficient. The contest runner's loader + scorer + eval roundtrip is **substrate-specific at the code path level**. Only side-by-side public (CPU) + private (CUDA) evals on identical archives separate the signal from procedural noise.

**Cost structure per candidate**:
- CPU eval: $0 (GHA)
- CUDA eval: $0.05-0.15 (Modal T4)
- Total: ~$0.05-0.15 per archive
- Paired eval overhead: ~15 minutes clock wall-time

**Citation**: `CLAUDE.md` "Submission auth eval — BOTH CPU AND CUDA", `AGENTS.md` "Dual-track lab", `Catalog #220`.

---

## 6. The 6 Wired-In Canonical Hooks (Catalog #125): Underexploited Surfaces

Per CLAUDE.md "Subagent coherence-by-default" + the aerospace_stealth_analytic_alien_tech_20260513.md §11:

1. **Sensitivity-map contribution** (hook 1):
   - Input: Per-pixel scorer-attention maps (DIE weight from U-DIE-KL, SABOR argmax-margin from SABOR, E1-E4 ROI mask from NRO hybrid)
   - Consumer: `tac.sensitivity_map` primitive, used for per-pixel bit-allocator priors
   - **Underexploited surface**: No current submitted archive uses dynamic per-pixel bit allocation; all use fixed per-pair or per-frame allocation. Quantizr's 0.33 likely uses sensitivity maps internally.

2. **Pareto constraint** (hook 2):
   - Input: (α_seg, β_pose, γ_rate) Lagrangian sweep across the Pareto frontier
   - Consumer: `tac.pareto_*` feasible-region intersection solver
   - **Underexploited surface**: Only A1 lane has run explicit Pareto sweeps (per aerospace_stealth memo Technique A4 / F1). No cross-substrate portfolio Pareto has been computed.

3. **Bit-allocator hook** (hook 3):
   - Input: DIE attention map, sensitivity prior, per-pair score-residuals
   - Consumer: `tac.composition.registry` byte-allocation weight matrix
   - **Underexploited surface**: Current allocator uses uniform per-pair weighting; Quantizr-style per-pair adaptive allocation (higher bytes for high-error pairs) is not yet implemented.

4. **Cathedral autopilot dispatch hook** (hook 4):
   - Input: Lane registry + dispatch-claim plan + pre-mortem verdicts
   - Consumer: `tools/cathedral_autopilot_autonomous_loop.py` (Catalog #273-#278 Rudin-Daubechies autopilot)
   - **Underexploited surface**: Autopilot is wired but **not activated** per race-mode rules (loop paused since 2026-05-09). When activated, the Rudin falling-rule-list (interpretable decision rules) will rank substrates by predicted ΔS; current ranking is ad-hoc.

5. **Continual-learning posterior update** (hook 5):
   - Input: Every empirical anchor (per cheap-probe verdict, per ACH row update, per exact-eval result)
   - Consumer: `.omx/state/continual_learning_posterior.jsonl` (fcntl-locked, append-only)
   - **Underexploited surface**: Posterior is updated after major campaigns (e.g., PR106 r2 exact eval), but **no incremental Bayesian synthesis** exists to convert per-substrate results into a global substrate-ranking posterior. Cathedral autopilot would consume this as a prior.

6. **Probe-disambiguator** (hook 6):
   - Input: Paired comparison smoke ($5-15 each) on ambiguous substrate pairs
   - Consumer: Dedicated disambiguator tool (planned per aerospace_stealth memo §11, not yet built)
   - **Underexploited surface**: The substrate-class-boundary hypothesis (A1 vs PR106 r2 pose-axis flip) has a **ready disambiguator** in the cpu_cuda_xray_synthesis pipeline (Verdict A/B/C/D structure), but it is NOT YET DISPATCHED (CUDA captures pending $0.05 gate).

**Citation**: `aerospace_stealth_analytic_alien_tech_20260513.md §11`, `Catalog #125`, `Catalog #220/#233`.

---

## 7. Beyond-PR Insights from Research Memos: Operationalization Gap

Scan of `.omx/research/*.md` from 2026-05-15 to 2026-05-17 for the keywords "not yet operationalized", "alien", "unexplored", "diminishing returns", "Shannon floor", "Pareto", "exploit", "asymptotic-pursuit":

### 7.1 Five high-EV unshiped research insights

1. **Tishby Information Bottleneck (pure)**
   - **Source**: `u_die_kl_substrate_wide_loss_v1_design_20260515.md` §2 (KL distill on SegNet logits, T=2.0 per Quantizr empirical 0.33 anchor)
   - **Innovation**: Replace hard argmax matching (cross-entropy on one-hot) with soft softmax-temperature distillation. Predicted ΔS: −0.005 to −0.010.
   - **Status**: Canonical helper `tac.losses.core.kl_distill_segnet_only` exists; ZERO substrates have adopted it yet (research-only landing, no per-substrate retraining dispatched).
   - **Blocker**: Operator-gated $30-60/substrate retraining wave not yet authorized.

2. **Wyner-Ziv side-information (ego-motion-informed latent deltas)**
   - **Source**: `l5_v2_*_20260517_codex.md` series (Time-Traveler L5 staircase; unpacked but not yet integrated into cathedral dispatcher)
   - **Innovation**: Encode pose-residual deltas as a secondary bitstream (side-information) that uses ego-motion estimates from frame-pair geometry as a **low-entropy prior**. Predicted ΔS: −0.008 to −0.015.
   - **Status**: L5 frame (sideinfo-lightning bundle) is dry-run validated 2026-05-17; still blocked on **false-authority hardening** (l5_v2_false_authority_hardening_20260516_codex.md) and **skip-archive-build fail-closed** (l5_v2_tt5l_skip_archive_build_fail_closed_20260516_codex.md).
   - **Blocker**: Control-plane integrity validation + paper-source hygiene gate + $10-15 exact-eval dispatch.

3. **Atick-Redlich cooperative-receiver (ATW codec)**
   - **Source**: `src/tac/symposium_impls/atw_codec_atick_tishby_wyner_triple.py` (canonical helper, Catalog #XX)
   - **Innovation**: Encoder learns the SegNet/PoseNet representations; decoder uses them as side-information to compress residuals (perceptual + rate-distortion joint optimization). Per Boyd convex optimization, achieves `S* ≈ 0.12-0.15` theoretical lower bound (vs current 0.193 frontier).
   - **Status**: Canonical helper exists; **ZERO substrates have adopted it** (research-only landing, implementation-complete but never wired into a trainer).
   - **Blocker**: Requires full trainer rewrite (not a bolt-on per CLAUDE.md lesson 7); $50-100 per attempt; high architectural risk.

4. **Rao-Ballard predictive-coding (Cathedral autopilot Rudin falling-rule-list)**
   - **Source**: `feedback_rudin_daubechies_autopilot_full_implementation_landed_20260515.md` + `feedback_rudin_daubechies_preflight_composite_landed_20260515.md` (Catalog #273-#278)
   - **Innovation**: Replace ad-hoc substrate ranking with interpretable decision rules (SLIM falling-rule lists, GOSDT, compressive sensing) that maximize predicted ΔS subject to computational budget. Predicted meta-EV: +$50-100 saved per week by avoiding dead-end substrates.
   - **Status**: Autopilot framework exists; **NOT ACTIVATED** per CLAUDE.md "loop paused since 2026-05-09" + race-mode dispatch rules.
   - **Blocker**: Loop reactivation requires operator explicit signal + calibration of falling-rule prior from first-session anchors.

5. **Carmack-Hotz strip-everything (NSCS02/NSCS06 minimalism)**
   - **Source**: `src/tac/symposium_impls/carmack_hotz_strip_everything_codec.py` (Catalog #XX) + `aerospace_stealth_analytic_alien_tech_20260513.md` Technique A6 (Frame-0 Asymmetric Byte Stuffing)
   - **Innovation**: Encode ONLY the strictly necessary bytes to reproduce SegNet/PoseNet outputs bit-deterministically; everything else is wasted (e.g., color-space quantization inefficiencies, channel redundancy, frame-order cosmetics). Predicted ΔS: −0.002 to −0.008 per technique; stacked −0.010 to −0.040.
   - **Status**: Helper primitive exists; **ONE candidate** (NSCS06 per Z-series nomenclature) has been drafted; never submitted.
   - **Blocker**: Requires bit-by-bit audit of decoder vs scorer I/O contract (high review friction); Catalog #125 bit-allocator hook could automate this.

### 7.2 Seven empirical questions the framework has NOT answered

1. **Device-axis attribution (Verdict A/B/C/D)**: Does the A1 vs PR106 r2 pose-axis flip originate in the loader (NVDEC vs PyAV) or the scorer kernels (TF32 / cuDNN)?
   - **Hypothesis**: Verdict B (FastViT per-layer ε accumulation, TF32 matmul)
   - **Experiment**: Dispatch cpu_cuda_xray_posenet_layer_drift CUDA capture on Modal T4 + read per-block compound factor
   - **Cost**: $0.05
   - **Time to decision**: 24 hours

2. **Within-class MDL saturation (Tier C density)**: Is the 0.196-0.199 cluster (90% shared assumptions) truly the HNeRV-family local minimum, or is there a within-class Pareto frontier we haven't explored?
   - **Hypothesis**: Z1 Tier-C ablation shows 0.171 floor; current frontier 0.193 is ~17% from floor (not saturated)
   - **Experiment**: Rudin-Daubechies autopilot rank substrates by MDL density; identify lowest-MDL candidate that has NOT been trained with U-DIE-KL + EMA + eval_roundtrip
   - **Cost**: $30-60 (smoke); $0.10 (exact-eval)
   - **Time to decision**: 5 days

3. **SABOR boundary audit extensibility (φ1 → φ2 → φ3)**: The φ1 SABOR audit (99.27% stable @ ε=32 RGB) was computed on A1 anchor. Does it transfer to PR106 r2 / Z3 / other substrates, or is it substrate-specific?
   - **Hypothesis**: Boundary stability is scorer-determined, not substrate-determined; should transfer
   - **Experiment**: Dump SegNet logit margins on PR106 r2 inflated output; recompute φ1 audit
   - **Cost**: $0 (use existing inflated archives)
   - **Time to decision**: 24 hours

4. **Odd-frames warping prediction error**: How much pose-axis distortion does ego-motion estimation + frame warping introduce vs the bandwidth savings?
   - **Hypothesis**: Warp error ≈ 0.5-1% MSE on frame-1 masks; bandwidth savings ≈ 5-10KB (50% of mask codec). Net ΔS = −0.003 to −0.008.
   - **Experiment**: Train ego-motion RAFT on contest pairs; measure warp-error distribution; integrate into score formula with rate-distortion trade-off
   - **Cost**: $20-30 (RAFT smoke)
   - **Time to decision**: 3 days

5. **KL distill temperature sweep (U-DIE-KL γ calibration)**: Quantizr used T=2.0; does sane_hnerv or PR101 have different optimal T?
   - **Hypothesis**: T ∈ {1.5, 2.0, 2.5, 4.0} spans the SegNet hardness space; sane_hnerv (tighter loss surface) may prefer higher T
   - **Experiment**: Per-substrate 27-cell sweep α/β/γ ∈ {0.5, 1.0, 1.5} × T ∈ {1.5, 2.0, 4.0}; rank by auth-eval result
   - **Cost**: $100-150 (3 substrates × 27 cells × $1.50/cell)
   - **Time to decision**: 7 days

6. **Categorical cross-entropy vs soft cross-entropy (SegNet head reformulation)**: Current SegNet head uses argmax (discrete decision); does replacing with soft softmax(logits) reduce pose-axis sensitivity?
   - **Hypothesis**: Soft cross-entropy divorces the loss from argmax decision boundaries, may reduce cliff-crossing behaviors. Predicted ΔS: −0.002 to −0.005.
   - **Experiment**: Retrain A1 with modified SegNet head (soft logits, not argmax). Measure auth-eval trajectory.
   - **Cost**: $20 (smoke)
   - **Time to decision**: 3 days

7. **Bytes beyond 500KB rate-term saturation**: The rate term (25 × R) becomes negligible for archives > 200KB. Is there a jump-discontinuity in the Pareto frontier at the rate-saturation threshold, or smooth continuation?
   - **Hypothesis**: Smooth continuation; rate saturation is a convex-optimization artifact
   - **Experiment**: Sample the frontier at R = [0.2, 0.4, 0.6, 0.8, 1.0] via explicit Lagrangian sweeps on 1 substrate
   - **Cost**: $30-40 (5 Lagrangian points × T4)
   - **Time to decision**: 5 days

### 7.3 Ranking by ΔS per unit cost

1. **Device-axis attribution (Verdict A/B/C/D)** — ΔS $0 (diagnostic, enables future $1000s of spend direction)
2. **Odd-frames warping (RAFT ego-motion)** — ΔS -0.008 / $25 = −3.2e-4 per $
3. **SABOR boundary audit transfer** — ΔS $0 (confirm or refute existing hypothesis)
4. **KL distill temperature sweep** — ΔS -0.007 / $125 = −5.6e-5 per $ (highest cost)
5. **Within-class MDL saturation** — ΔS -0.010 / $40 = −2.5e-4 per $
6. **Soft cross-entropy SegNet** — ΔS -0.003 / $20 = −1.5e-4 per $
7. **Rate-term saturation** — ΔS $0 (theoretical, low predicted signal)

---

## 8. Hardware Exploits: Shipped vs Not Yet Shipped

Per CLAUDE.md Tier-1 engineering (Catalog #172 / #178 / #179 / #180) + the symposium_impls surface:

| Exploit | Availability | Shipping status | Predicted ΔS | Blocker |
|---------|----------|------|------|---|
| **torch.compile (Inductor)** | PyTorch ≥1.14 (Catalog #179) | NOT SHIPPED; research-only | −0.001 to +0.005 | No validated speedup on contest scorer; unclear if deterministic |
| **TF32 matmul** | CUDA ≥8.0 (A100/H100; T4 has opt-in via cublas_tf32=1) | NOT SHIPPED; research-only | −0.002 to +0.010 (speed-accuracy trade-off) | Incompatible with exact-bit-determinism goal; CPU has no TF32 equivalent |
| **autocast fp16** | PyTorch AMP (Catalog #172) | NOT SHIPPED; research-only | −0.005 to +0.020 (volatile) | SegNet/PoseNet are already float32-optimized; mixed-precision may hurt distortion |
| **torch.autograd.profiler** | PyTorch native | SHIPPED (via tools/profile_*.py) | N/A | Used only for diagnostics, not for score improvement |
| **FP4 (PR101 GOLD innovation)** | torchao library | ONLY IN PR101; not yet general-ized to other substrates | −0.010 to −0.020 (PR101 empirical) | Requires full trainer rewrite + export format change; not a bolt-on |
| **FP8 (torchao)** | torchao ≥0.2 | NOT SHIPPED | −0.008 to −0.025 (theoretical) | Even more volatile than FP4; CPU has no FP8 support |
| **Apple Silicon mps** | PyTorch ≥1.12 (macOS) | FORBIDDEN per CLAUDE.md (noise) | N/A | MPS results are not contest-compliant; forbidden as authoritative per mps_auth_eval_is_noise non-negotiable |

**Net assessment**: Tier-1 engineering (autocast / torch.compile / TF32) all exhibit either (a) volatility in score impact, (b) CPU/CUDA incompatibility, or (c) unclear speedup on contest scorer. The most aggressive exploit (FP4) is already monopolized by PR101 GOLD; generalizing it would require reverse-engineering PR101's export contract (intellectual property fence).

**Citation**: `Catalog #172/#178/#179/#180`, `CLAUDE.md` "Tier-1 engineering non-negotiable", `pytorch.org torch.compile`, `torchao.ai`.

---

## 9. Substrate Canvas Inventory: 49 Trainers, 27 Recipes, Current Maturity

Per `.omx/state/lane_registry.json` + CLAUDE.md substrate parity discipline:

### 9.1 HNeRV family (sane_hnerv, hnerv_lc_v2, qpose14 clones)

- **Status**: 6 active lanes
- **Furthest from shipping**: `qpose14_adaptive_selector` (tier 45, Level 0, design-phase)
- **Highest theoretical headroom**: `hnerv_lc_v2_u_die_kl_retrain` (pending $30-60 smoke)
- **Current frontier**: `sane_hnerv + PR101-parity (0.193 CUDA)`

### 9.2 NeRV family (TCNeRV / BlockNeRV / FFNeRV / DSNeRV / HiNeRV / e_nerv / ego_nerv / nervdc)

- **Status**: Lane 12 NeRV mask-codec audit completed 2026-04-30; verdicted DEFERRED-pending-renderer-rescope (CLAUDE.md lesson 5: full renderer required, not mask-only)
- **Furthest from shipping**: All NeRV lanes currently paused pending architecture rescope
- **Reactivation criteria**: Implement full RGB NeRV renderer (not mask-only slot); merge with archive grammar + inflate runtime
- **Predicted upside**: −0.010 to −0.020 if renderer rescope lands (per grand_council_hnerv_meat_on_bone_deep_dive_20260513.md)

### 9.3 Outside-NeRV (SIREN, Cool-Chic, VQ-VAE, wavelet, self_compress_nn, hybrid_renderer_residual, grayscale_lut, etc.)

- **SIREN**: 1 lane (tier 60), Level 0, design-phase (cool-chic-redux analogy)
- **Cool-Chic / C3**: Verdicted DEFERRED-pending-export-design (CLAUDE.md lesson 2: export-first design required; FP4A export contract missing)
- **VQ-VAE**: 1 lane (tier 55), design-phase, no trainer landed
- **Wavelet (Daubechies)**: 1 lane (tier 70, canonical helper exists), never wired into a full trainer
- **self_compress_nn**: Experimental, not registered in main lane_registry
- **hybrid_renderer_residual**: Experimental, pending P2 review
- **grayscale_lut**: 1 lane (tier 40), design-phase, low probability (LUT capacity ceiling ~10KB @ 256^3 RGB → 256 colors)

### 9.4 Production-grade substrates (A1, Z-series, L5 Time-Traveler)

- **A1 (PR101 GOLD parity)**: 268 LOC substrate + 337 LOC bolt-on = 605 LOC total, 0.193 CUDA / CPU unknown (paired eval pending $0.10)
- **Z-series (Tier-C ablation variants)**: Z1 (0.171 floor estimate), Z3 (SABOR variant), Z4/Z5 (class-shift candidates), Z6 (ensemble)
- **L5 Time-Traveler**: Wyner-Ziv side-information + per-pair pose allocation; currently blocked on false-authority hardening + paper-source hygiene gate

### 9.5 Summary ranking by "furthest from shipping but highest headroom"

| Lane | Status | Bottleneck | Predicted ΔS if shipped | Cost to ship |
|------|--------|-----------|---|---|
| **L5 Time-Traveler** | Blocked (false-auth hardening) | Control-plane integrity | −0.008 to −0.015 | $10-15 (exact-eval) |
| **SIREN (rescoped)** | Design-phase | Full trainer + loss function | −0.005 to −0.015 | $50-80 |
| **NeRV full-RGB renderer** | DEFERRED | Architecture rescope | −0.010 to −0.020 | $80-120 |
| **ATW codec (full adoption)** | Canonical helper exists | Trainer rewrite | −0.020 to −0.050 | $100-150 |
| **KL distill per-substrate sweep** | Canonical helper exists | $30-60 smoke wave | −0.005 to −0.010 | $30-60 per substrate |

**Conclusion**: **L5 Time-Traveler is the nearest-to-ready high-headroom candidate** once false-authority + paper-source gating completes (expected 48-72 hours).

---

## 10. Interpretable ML / Formalization: Rudin-Daubechies Autopilot Current Status

Per **Catalog #273-#278** + `feedback_rudin_daubechies_autopilot_full_implementation_landed_20260515.md` + `feedback_rudin_daubechies_preflight_composite_landed_20260515.md`:

### 10.1 Rudin falling-rule-list (SLIM + GOSDT + compressive-sensing)

**Status**: Canonical framework landed in `tools/cathedral_autopilot_autonomous_loop.py` (200+ LOC, wired with per-substrate MDL density + DIE attention priors).

**What it does**:
1. Consume per-substrate: (arch_name, family, mdl_density, predicted_δs, cost_usd)
2. Emit **interpretable decision rules** (e.g., "IF mdl_density < 0.80 AND family == NeRV AND cost < $50 THEN rank HIGHER")
3. Rank substrates by Δscore/$, subject to computational budget (per operator gate)

**What it has NOT done**: NO per-substrate Rudin ranking has been published; autopilot is **paused per race-mode rules** (CLAUDE.md loop-pause directive 2026-05-09).

### 10.2 Operationalization blockers

1. **Falling-rule prior calibration**: Requires ≥5 per-substrate exact-eval anchors to learn the rule weights. Only 3 anchors exist (A1, PR106 r2, PR103), all HNeRV-family (insufficient diversity).

2. **Continual-learning posterior integration**: The Rudin solver should consume the per-substrate posterior (from `.omx/state/continual_learning_posterior.jsonl`), but no Bayesian synthesis exists to convert per-archive results into per-substrate priors.

3. **Loop reactivation criteria**: CLAUDE.md "loop paused since 2026-05-09" requires explicit operator signal + re-calibration. No automatic reactivation.

**Citation**: `Catalog #273-#278`, `feedback_rudin_daubechies_autopilot_full_implementation_landed_20260515.md`.

---

## 11. Open Questions Ranked by (Predicted ΔS / $)

| # | Question | Hypothesis | Cost | ΔS predicted | $/outcome | Rank |
|---|----------|-----------|------|------|------|------|
| 1 | Device-axis attribution (Verdict A/B/C/D) | Verdict B (TF32 / cuDNN kernel ε) | $0.05 | $0 signal (diagnostic) | — | FIRST |
| 2 | Odd-frames RAFT ego-motion | ΔS −0.005 to −0.010 @ 50% bandwidth savings | $25 | −0.008 | −3.2e-4 | 2 |
| 3 | Within-class MDL saturation (Tier-C census) | 0.171 floor real; 0.193 frontier ~17% above floor | $40 | −0.010 | −2.5e-4 | 3 |
| 4 | KL distill T sweep (U-DIE-KL γ calibration) | T=2.0 universal; T ∈ {1.5, 4.0} orthogonal signal | $150 | −0.007 | −4.7e-5 | 4 |
| 5 | SABOR boundary audit transfer | φ1 audit transfers to PR106 r2; not substrate-specific | $0 | N/A (diag) | — | 5 |
| 6 | Soft cross-entropy SegNet head | Soft logits reduce cliff-crossing; ΔS −0.002 to −0.005 | $20 | −0.003 | −1.5e-4 | 6 |
| 7 | Rate-term saturation discontinuity | Smooth frontier; no jump at R-saturation threshold | $40 | ~$0 (theory) | — | 7 |

---

## 12. Production Adoption by comma.ai: Contest vs Real-World

Per CLAUDE.md "Contest vs production target modes non-negotiable" + standing directives 2026-05-15 "share what works but when it is stale or obsolete...we want full and complete and correct unique and distinct designs":

### 12.1 What transfers to production (openpilot)

1. **SegNet stride-2 blindspot insight** → Real-world road-sign detection needs tighter feature resolution at boundaries (e.g., stop sign vs yield sign). SABOR per-pixel boundary classification could transfer.

2. **PoseNet 12-channel YUV6 + RepMixer architecture** → Production openpilot drivers use similar camera geometry (consecutive frames, ego-motion dominates pose). FastViT-T12 RepMixer is lightweight (~0.5s inference on Snapdragon 8 Gen 2).

3. **Odd-frames temporal reuse (RAFT ego-motion)** → Openpilot's prediction stack (path planning) already uses ego-motion estimation; reusing it for frame compression is synergistic.

4. **EMA training discipline + eval_roundtrip** → Production requires reproducibility; EMA decay (0.997) + uint8 bottleneck simulation transfer directly.

### 12.2 What does NOT transfer (contest-specific)

1. **Archive.zip packing discipline** → Openpilot deploys as model.bin + runtime.so; no ZIP codec, no archive grammar per-pair deltas.

2. **Pose first-6-dims-only scoring** → Openpilot uses all 12 pose dimensions (full 6-DOF); contest throws away dims 7-12 (free waste).

3. **HNeRV latent compression** → Openpilot videos are in-the-loop surveillance (live streaming to fleet); latent compression requires offline training on a pre-frozen video dataset (contest-specific constraint).

4. **Per-pixel bit allocation (DIE attention)** → Openpilot's bitrate budget is network-constrained (4G/5G), not archive-size-constrained; allocation strategy is different.

### 12.3 Implication

**The contest is a research testbed for perception-aware compression (SegNet/PoseNet joint optimization), NOT a direct engineering pipeline for production.** Techniques that transfer are those that respect the production constraints (inference latency, on-device memory, continuous streaming). The Rudin autopilot, U-DIE-KL loss, and SABOR boundary methods are production-applicable; the archive grammar is not.

**Citation**: CLAUDE.md "Contest vs production target modes non-negotiable" + "tac stays clean" directive.

---

## Synthesis: The Frontier-Pursuit Trajectory

The full problem space decays into **three inelastic axes**:

1. **SegNet axis** (100 × d_seg): constant marginal. Currently ~0.002 distortion frontier. Diminishing returns kick in below 0.001. SABOR boundary exploit (A1 technique) targets this.

2. **PoseNet axis** (√(10 × d_pose)): hyperbolic marginal (~271 at frontier). Dominates byte allocation past d_seg ≈ 0.001. **Device-axis flip hypothesis (Verdict B TF32 kernel ε) is the KEY SEPARATOR** of A1 vs PR106 r2 substrates.

3. **Rate axis** (25 × R): linear but swamps to noise past R ≈ 0.005 (200KB archive). Below this, SegNet + PoseNet dominate; above, all bytes are "paying" for rate.

**The unshipped alien-tech candidates** (odd-frames RAFT, ATW codec, L5 Time-Traveler side-information) **all target the pose axis**, where the marginal value is highest. Their collective predicted ΔS is −0.030 to −0.080 if all shipped and tuned (per aerospace_stealth_analytic_alien_tech_20260513.md stacking rules). This lands the theoretical frontier in the [0.10, 0.15] band per Blahut-Arimoto + Boyd convex optimization.

**The blocking signals** are (a) device-axis attribution (Verdict A/B/C/D), which decides whether substrate design is CUDA-resilient or CUDA-vulnerable, and (b) per-substrate U-DIE-KL retraining wave ($30-60 each), which operationalizes the Tishby IB + Yousfi DIE + Hinton KL insights already in code but never adopted.

The next 48 hours should dispatch: (1) cpu_cuda_xray_posenet_layer_drift CUDA capture ($0.05, 24h), (2) L5 Time-Traveler false-authority + paper-source hardening (5h, $0), and (3) KL distill first smoke on sane_hnerv ($15, 12h). That trio resolves three of the seven open questions and unblocks ~$100-200 of high-EV follow-on spend.

