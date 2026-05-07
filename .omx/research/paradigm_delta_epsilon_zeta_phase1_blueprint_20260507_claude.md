---
title: PARADIGM-δεζ Phase 1 Blueprint — Joint Training + MDL/Bayesian + Self-Compression NN
date: 2026-05-07
author: Track 3 architecture subagent (claude-sonnet-4-6) + Track-1 coordinator persistence pass
status: PHASE 1 DESIGN — not yet dispatched; Gate 0 approval pending
target_band: 0.155-0.175 [predicted-band, NOT contest-CUDA]
shannon_floor_realistic: 0.155
shannon_floor_t_minus_65h: 0.224
---

# PARADIGM-δεζ Phase 1 Blueprint

## 0. Executive Position

- **Contest frontier**: 0.20935 [contest-CUDA] (PR106x-lowlevel-brotli, 186080 bytes)
- **Public top 3**: 0.193 / 0.195 / 0.195 (PR #101 / #103 / #102)
- **Wave-Ω stack ceiling**: 0.180 [predicted-band, NOT contest-CUDA]
- **Shannon realistic floor**: 0.155 [council-derived]
- **PARADIGM-δεζ target**: 0.155–0.175 [predicted-band] — requires δ + ε + ζ composing correctly with Wave-Ω

PARADIGM-δεζ was deferred from the 2026-05-07 council dispatch candidate set as
"too much pre-dispatch work for the 60-minute race rule." Operator extended
scope to multi-day work under the post-deadline continuous-engineering mandate.
This blueprint is the design foundation for that multi-day program.

**Goal is lowest score, not paradigm-matching**. δεζ are selected because they
are the most credible paths to the realistic 0.155 floor that the Wave-Ω stack
alone cannot reach — not because they match any competitor's approach.

## 1. Phase Decomposition

| Phase | Wall-clock | GPU spend | Deliverable | Operator approval |
|---|---|---|---|---|
| 1: Design + scaffolding | 1 day | $0 | This blueprint + lane registry + stubs | Gate 0 |
| 2: δ first measurable run | 2-3 days | $8-15 | Joint-trained checkpoint + [contest-CUDA] eval | Gate 1 |
| 3: ε rate term + ζ self-compress | 2-3 days (parallel) | $8-15 | ε codec roundtrip + ζ full-renderer fine-tune | Gate 3 |
| 4: Composition dispatch | 1-2 days | $2-5 | Full δ+ε+ζ + Wave-Ω archive + [contest-CUDA] | Gate 4 |
| 5: Sanity ladder + promotion | 1 day | $0 | Lane maturity gates 1-7; deploy runbook | Gate 5 |

**Mandatory precondition (Gate 2)**: apogee_int6 contest-CUDA eval lands
($0.30-0.60 Lightning T4) BEFORE Phase 2 δεζ GPU spend. If int6 scores ≤ 0.180
[contest-CUDA], reassess whether Phase 2 is the highest-ROI move.

## 2. Architectural Designs

### 2.1 δ — Joint scorer-aware training

**Core problem**: current pipeline trains the renderer against pixel-MSE proxy.
Scorer (SegNet + PoseNet) is consulted only at auth-eval time. Proxy-auth gap
historically 2-350x.

**Joint loss**:
```
L_joint(θ) = λ_rate · R(θ) + λ_seg · D_seg(θ) + λ_pose · D_pose(θ)
```
where:
- `R(θ) = -log2 p_y(y|z)` — differentiable rate (Ballé entropy bottleneck) on quantized weights
- `D_seg(θ) = E_t [KL(SegNet(x_hat_t) || SegNet(x_gt_t))]` at T=2.0 (Hinton distillation)
- `D_pose(θ) = E [||PoseNet(x_hat) - PoseNet(x_gt)||²]` on **first-6 dims only** (Yousfi revision)
- Lagrange multipliers derived from contest formula: `λ_rate = 25/37545489`,
  `λ_seg = 100`, `λ_pose = 5/sqrt(10·pose_avg_at_init)` ≈ 271 at PR106 frontier

**Adaptive λ_pose** (Yousfi revision): recompute from CURRENT pose_avg at each
auth-eval checkpoint. As pose improves, λ_pose increases (marginal value rises
since `d(score)/d(pose) = 5/sqrt(10·pose)` → ∞ as pose → 0).

**λ_rate annealing** (Shannon revision): exponential ramp from epoch 0 (start
at 0.01× final, anneal to final value), NOT a phased "Phase A then activate."
Phased approach creates distribution shift; constant signal converges cleanly.

**Strict-scorer-rule compliance**: scorers loaded at compress time only.
Renderer is feedforward-only at inflate time. EMA decay=0.997, eval_roundtrip=True
both mandatory.

**New module**: `src/tac/joint_scorer_aware_training.py` (~350 LOC)
- `JointScorerAwareLoss(nn.Module)`
- `adaptive_lambda_scheduler(baseline_score, current_score) -> LambdaWeights`
- `JointTrainingConfig(dataclass)` (no silent defaults)
- `ScoreAwareEvalCallback` (eval-time wrapper)

### 2.2 ε — MDL/Bayesian learned entropy prior

**Current state**: `tac.mdl_bayesian_codec` is a meta-comparison framework — it
ranks codecs by MDL but does not produce archive bytes. ε extends this to a
codec that **ships a learned prior in the archive** and arithmetic-codes
quantized weights under it.

**Rate-prior-distortion (MacKay Ch. 28 + Ballé 2018)**:
```
L(D, M) = L(M) + L(D|M)         # bits to ship prior + bits to ship data given prior
L(D|M)  = -Σ_i log2 p(y_i | μ_i, σ_i)
```

**Hyper-encoder/hyper-decoder architecture**:
- HyperEncoder: maps weight latents `y` → hyper-latents `z`. **1D channel-wise
  conv** along channel axis of [C_out, C_in, kH, kW] tensors (Ballé revision —
  channel correlations dominate spatial in renderer weights).
- HyperDecoder: maps `z` → per-symbol `(μ, σ)`. **Mixture-Gaussian** (Fridrich
  revision) — distribution of quantized weights post-ζ is multi-modal (pruned
  channels at 0; retained channels at variable bit-depth).
- z is quantized and shipped as `renderer_prior.bin` in the archive.

**Spike-and-slab prior** (MacKay stretch goal, Phase 3): `(spike_weight,
slab_mean, slab_sigma)` per channel — 3 params/channel, ≤ 3 KB total. Captures
pruned-vs-retained structure perfectly. Phase 3 stretch.

**Archive structure**:
```
archive.zip/
  renderer.bin         ← arithmetic-coded quantized weights (under learned prior)
  renderer_prior.bin   ← hyper-decoder MLP, magic b"LEPR", ≤ 5 KB int8+brotli
  masks.{mkv|nerv}     ← AV1 or NeRV (α-paradigm)
  poses.{pt|bin}       ← PD codec
```

**New module**: `src/tac/learnable_entropy_model.py` (~250 LOC)
- `HyperEncoder` (1D channel-wise conv)
- `HyperDecoder` (Gaussian + mixture modes; spike-and-slab as Phase 3)
- `LearnableEntropyModel.rate(y) / encode(y) / decode(bits)`
- `LearnableEntropyModelCodec` (archive builder, magic `b"LEPR"`)

**Predicted savings**: 0.3–0.5 bpp vs factorized prior on natural images (Ballé
2018); applied to renderer weights with their stronger channel correlations,
20-30% rate reduction → ~9 KB saved → score delta `25 × 9000 / 37545489 ≈
0.006` [predicted].

### 2.3 ζ — Self-compression NN: full renderer

**Current state**: `tac.self_compress.SelfCompressingPostFilter` applies
self-compression to a small postfilter (~46 KB). ζ extends this to the **full
renderer** (88K-param JointFrameGenerator).

**Self-compression loop**:
1. Train R_θ normally (δ joint-training)
2. `swap_renderer_convs_with_self_compress()` replaces Conv2d → SelfCompressingConv2d
3. Fine-tune with `L_rate = λ_sc · Σ_l (b_l · params_l)` where `b_l` is
   learnable per-channel bit-depth (STE allows gradients through discrete
   quantization)
4. **Minimum 2000 QAT steps** (Selfcomp revision; 500 leaves bit allocation at
   ~3.5 bpw, far from optimal 1.5 bpw)
5. At export: channels with `b_l < 0.5` pruned; remaining packed at learned
   bit-depth. Magic `b"ZETA"` + 4-byte config header + per-layer channel counts +
   per-layer bit-depths + packed weight data.

**FiLM protection**: protect FiLM γ/β at TRAINING time (Hotz/Dykstra
clarification — at archive time, FiLM affines are baked into renderer weights
and not separately stored). `protect_film_layers=True` default; pattern match on
`["film", "cond", "gamma", "beta", "scale", "shift"]`.

**From-scratch alternative** (Hotz revision, Phase 3 stretch): build
`SelfCompressingJointFrameGenerator` from scratch with SelfCompressingConv2d
baked in. Predicted bpw: 1.5-2.0 (vs 2.5-3.0 for post-hoc swap). Saves an
additional 15-20 KB if the from-scratch architecture trains stably.

**New module**: `src/tac/self_compress_full_renderer.py` (~300 LOC)
- `FullRendererSelfCompress(nn.Module)`
- `FullRendererSelfCompressConfig` (target_bits_total, qat_steps=2000, protect_patterns)
- `train_full_renderer_self_compress(renderer, frames, scorers, config)`
- `export_full_renderer_self_compress(renderer) -> bytes` (magic `b"ZETA"`)
- `load_full_renderer_self_compress(blob, arch_config) -> renderer` (no scorer)

**Predicted savings**: 88K params × 1.5 bpw / 8 = ~16.5 KB vs current ~56 KB
FP4+brotli renderer. Delta: ~40 KB → score delta `25 × 40000 / 37545489 ≈
0.027` [predicted]. **Largest single-component saving.**

## 3. Composition Contract

### 3.1 Stacking with apogee_int6 + Wave-Ω

**Compress-time order matters**:
1. δ joint-training → renderer R_δ
2. ζ self-compress fine-tune on R_δ → R_δζ (with learned bit allocations)
3. ε learned-prior training on R_δζ weight distribution → P_ε
4. ε arithmetic-code R_δζ under P_ε → renderer.bin
5. apogee_int6 (alternative to ζ for fixed 6-bit) — see §3.2
6. Wave-Ω: SJ-KL residual + NeRV mask + block-FP (absorbed into ζ)
7. JCSP container build
8. Pose optimization (PD + Riemannian TTO)
9. Auth eval

**Inflate-time** (independent, order-free): each section dispatches by magic
byte; no scorer calls anywhere.

**Contrarian's revision** (CONDITIONAL ENDORSE, 10/10 council): the sequential
training protocol risks distribution mismatch — Stage 2 (ζ) changes the weight
distribution Stage 3 (ε) must encode. Required: implement an optional joint
mode (`use_joint_loss=True` flag) where all three are optimized in a single
loop:
```
L_full = λ_rate · R_ε(Q_ζ(θ)) + λ_seg · D_seg(decode(Q_ζ(θ))) + λ_pose · D_pose(...)
```
Sequential mode acceptable for Phase 2 proof-of-concept; joint mode REQUIRED
for Phase 4 final dispatch.

### 3.2 ζ vs apogee_int6

Both target the renderer.bin slot (weight compression). Composition:
- **Option A** (preferred): δ → ζ → ε. ζ is more flexible (per-channel bit
  allocation, can prune to 0). Subsumes apogee_int6.
- **Option B** (fallback): if ζ collapses FiLM numerics, fall back to
  apogee_int6 (meta-Lagrangian L=0.1999 [predicted]) with just δ+ε on top.

Lane registry tracks both; council selects between them at Phase 4 based on
Phase 2-3 empirical results.

## 4. Predicted-Score Arithmetic

| Component | Current [contest-CUDA] | Post-δεζ [predicted] | Delta | Method |
|---|---|---|---|---|
| seg_dist | 0.067 | 0.050 | -0.017 | δ joint-training (25% reduction) |
| sqrt(10·pose) | 0.0184 | 0.012 | -0.006 | δ + adaptive λ_pose; Riemannian TTO compounds |
| rate (bytes) | 0.124 | 0.080 | -0.044 | ζ ~40 KB + ε ~9 KB + NeRV ~150 KB |
| **TOTAL** | **0.20935** | **0.142** | **-0.067** | Stack of orthogonal savings |

**0.142 is OPTIMISTIC central-case** — assumes all components compose at
central estimates. Realistic range [0.155, 0.175] [predicted-band]. Conservative
target for Phase 4 dispatch planning: **0.165 [predicted-band]**.

Shannon floor crosscheck: 0.155 council-derived realistic. Our 0.142 sits 0.013
below — possible only if ζ hits ~16.5 KB AND δ eliminates proxy-auth gap.

## 5. Sanity Ladder

### Predispatch sanity for δ
1. Scorer-basin parity: `D_seg ∈ [0, 5]`, `D_pose ∈ [0, 100]` on smoke batch
2. Lagrange sanity: `λ_pose ∈ [100, 1000]` from current baseline pose_avg
3. Rate gradient sanity: `||∇_θ R||` finite and non-zero
4. EMA shadow sanity: shadow ≠ live by < 1% MAD after 10 steps
5. eval_roundtrip sanity: roundtripped frame ≠ raw by < 0.1 MSE (proves
   roundtrip is not a no-op)

### Predispatch sanity for ε
1. Encode-decode roundtrip: `y == decode(encode(y))` (lossless)
2. Prior size: `len(renderer_prior.bin) ≤ 5000` after int8+brotli
3. Rate gradient: `||∇_θ L_entropy||` finite and non-zero
4. Cross-symbol independence: rate(learned_prior) < rate(Laplace(0,1))

### Predispatch sanity for ζ
1. Roundtrip weight fidelity: per-layer MSE < 0.1
2. Bit allocation: ≥ 50% of channels with `b_l > 1.0` (anti-collapse)
3. FiLM protection asserted in `self_compress_layers` exclusion list
4. Archive size: `export_full_renderer_self_compress(renderer) ≤ 30000` bytes

### Module additions
`tac.predispatch_sanity` extended with:
- `check_joint_training_loss_gradient_flow`
- `check_entropy_model_roundtrip`
- `check_self_compress_fidelity` (MSSIM ≥ 0.90 — the gate apogee_int4 lacked)

## 6. Risk Register

| # | Risk | Probability | Mitigation |
|---|---|---|---|
| 1 | δ joint-training mode collapse (renderer hallucinates scorer-fooling pixels) | Medium (30-40%) | Curriculum: λ_seg=λ_pose=0 for Phase A; anneal over 200 steps; pixel-MSE floor ≥ 0.001; eval_roundtrip mandatory |
| 2 | ε prior overfits to single video | Low | Enforce ≤ 5 KB; validate `bits_under_learned < bits_under_Laplace` |
| 3 | ζ self-compress collapses FiLM conditioning | Medium (40%) | `protect_film_layers=True` hard default; smoke test ≥ 5% cross-frame variation |
| 4 | int4-catastrophe class inheritance | Medium for apogee_int6; LOW for ζ | Full 5-gate sanity ladder + `check_self_compress_fidelity` MSSIM ≥ 0.90 |
| 5 | SJ-KL Fisher OOM (Wave-Ω composition) | HIGH if naive | Hutchinson trace estimator OR Lanczos k=10 (council §5.3) |
| 6 | Archive compliance failure at inflate | Medium | New `b"LEPR"` member is OPTIONAL; fallback to static Laplace if absent; full inflate.sh→evaluate.py e2e test before Phase 4 |
| 7 | Phase 2 GPU duration overrun | Certain | H100 SXM (fast-chip directive); fallback to RTX 4090 for Phase C QAT |

## 7. Council Verdict (10/10 ENDORSE with 7 revisions)

| Member | Verdict | Revision incorporated |
|---|---|---|
| Shannon LEAD | ENDORSE | λ_rate exponential annealing from epoch 0 (removes Phase A) |
| Dykstra CO-LEAD | ENDORSE | FiLM protection scope: training only, not archive |
| Yousfi | ENDORSE | Pose loss restricted to first-6 dims; adaptive λ_pose required |
| Fridrich | ENDORSE | Mixture-Gaussian prior in ε hyper-decoder |
| Contrarian | CONDITIONAL ENDORSE | Joint-training mode flag REQUIRED for Phase 4 |
| Quantizr | ENDORSE | apogee_int6 dispatch FIRST before δεζ GPU spend (Gate 2) |
| Hotz | ENDORSE | From-scratch SelfCompressingJointFrameGenerator preferred |
| Selfcomp | ENDORSE | ζ QAT ≥ 2000 steps minimum |
| MacKay (memorial) | ENDORSE | Spike-and-slab prior as Phase 3 stretch |
| Ballé | ENDORSE | HyperEncoder: 1D channel-wise conv, NOT 2D depthwise-separable |

VERDICT: 10/10 ENDORSE. Architecture sound. apogee_int6 contest-CUDA dispatch
MANDATORY before Phase 2 GPU spend.

## 8. Operator Decision Points

- **Gate 0 (NOW)**: approve blueprint. Cost $0. Risk: none.
- **Gate 1 (after Phase 1)**: approve lane registration + Phase 2 dispatch
  plan. Cost $0; Phase 2 dispatch $8-15 H100 SXM.
- **Gate 2 (BEFORE Phase 2 GPU)**: apogee_int6 [contest-CUDA] eval lands
  ($0.30-0.60). MANDATORY precondition.
- **Gate 3 (after Phase 2)**: approve ε/ζ Phase 3 dispatch. Precondition:
  Phase 2 [contest-CUDA] shows seg_dist OR pose_dist improvement. If both
  worse, do NOT proceed.
- **Gate 4 (after Phase 3)**: approve composition + Phase 4 dispatch ($2-5).
  Precondition: ε/ζ roundtrip tests pass; archive size predicted < 100 KB.
- **Gate 5 (after Phase 4)**: approve public submission. IRREVERSIBLE
  disclosure. Precondition: [contest-CUDA] < 0.193; 5-turn clean-pass council
  review (CLAUDE.md submission gate).

## 9. Implementation Map

### Files to create

- `src/tac/joint_scorer_aware_training.py` (~350 LOC)
- `src/tac/learnable_entropy_model.py` (~250 LOC)
- `src/tac/self_compress_full_renderer.py` (~300 LOC)
- `experiments/train_joint_scorer_aware.py` (~200 LOC) — Phase 2 CLI
- `experiments/build_delta_epsilon_zeta_archive.py` (~150 LOC) — Phase 4 archive assembly
- `scripts/remote_lane_delta_epsilon_zeta.sh` (~100 LOC) — Phase 4 remote bootstrap
- `src/tac/tests/test_joint_scorer_aware_training.py`
- `src/tac/tests/test_learnable_entropy_model.py`
- `src/tac/tests/test_self_compress_full_renderer.py`

### Files to modify

- `experiments/pipeline.py`: add 4 PipelineConfig fields (`use_joint_scorer_aware`,
  `joint_training_config_path`, `use_learnable_entropy`, `use_full_renderer_self_compress`);
  WARN-on-unwired guards matching β/γ/α pattern (commit 999211e5)
- `src/tac/preflight.py`: new STRICT check `check_joint_training_scorers_not_at_inflate_time`
  (static scan for scorer load calls inside `inflate*.py`)

### Lane registry (Phase 1 step)

```bash
python tools/lane_maturity.py add-lane lane_delta_joint_training \
    --name "δ Joint Scorer-Aware Codec Retrain" --phase 1
python tools/lane_maturity.py add-lane lane_epsilon_learnable_entropy \
    --name "ε MDL/Bayesian Learned Prior Codec" --phase 1
python tools/lane_maturity.py add-lane lane_zeta_self_compress_renderer \
    --name "ζ Full-Renderer Self-Compression NN" --phase 1
```

## 10. Critical Implementation Details (CLAUDE.md non-negotiables)

- **EMA**: decay=0.997, snapshot+restore at eval, archive bytes from EMA shadow
- **eval_roundtrip**: True everywhere; raise if disabled
- **CUDA-only**: no `device = "cuda" if cuda.is_available() else "mps"` ternaries
- **Subagent commits**: via `tools/subagent_commit_serializer.py`
- **Auth eval everywhere**: every training phase ends with
  `scripts/remote_archive_only_eval.sh` on EXACT archive bytes
- **No /tmp paths**: artifacts to `experiments/results/lane_<id>_<timestamp>/`,
  `.omx/state/`, `.omx/research/`
- **Strict-scorer-rule**: scorers loaded at compress time only; never in archive;
  never at inflate

## 11. Cross-References

- `project_grand_council_FIELDS_MEDAL_shannon_floor_obsession_20260501` — S_min=0.155 derivation
- `feedback_grand_council_next_shannon_floor_pivot_20260507` — δεζ deferred to multi-day scope
- `feedback_goal_is_lowest_score_not_quantizr_paradigm_match_20260506`
- `project_post_deadline_continuous_engineering_mandate_20260502` — extreme rigor, no time/money limit
- `src/tac/joint_admm_proximal_pose_delta.py` — δ foundation
- `src/tac/joint_admm_proximal_water_filling_v2.py` — δ foundation
- `src/tac/mdl_bayesian_codec.py` — ε scaffold (extends to byte-producing codec)
- `src/tac/self_compress.py` — ζ baseline (extends from postfilter to full renderer)
- `src/tac/balle_sensitivity_weighted.py` — δ rate ingredient
- `src/tac/jcsp_stream_builder.py` + `src/tac/jcsp_score_marginals.py` — γ-JCSP composition point
- `experiments/pipeline.py` — cross-paradigm flag wiring (β/γ/α scaffolded; δεζ to add)

---

*Phase 1 scaffolding can begin pending Gate 0 approval. Estimated Phase 1
wall-clock: 1 day. First [contest-CUDA] evidence: Phase 2, ~3 days after
Gate 1 approval.*
