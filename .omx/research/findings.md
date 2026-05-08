# Findings

## 2026-05-08 [RETRACTED→byte_proxy_only_NOT_deployable] Cross-paradigm 137,531 B figure was Op1 re-encode of dequantized fp32 substrate, NOT a byte-closed archive

**Status (2026-05-08, REVIEW-ENG C1 closure)**: the 137,531 B figure originally tagged "DOMINANT cross-paradigm winner" is `byte_proxy_only_NOT_deployable`. It is `len(blob_op1)` from `pipeline_op1.encode(rebuilt, skip_validate=True)` in `tools/pr101_cross_paradigm_hstack_vstack_empirical.py:148-153`, where `rebuilt` is the ADMM-coarsened dequantized fp32 state_dict. The figure does NOT include the per-tensor K side-info, fp16 scales, or the PR101 latent_blob/sidecar that an actual archive must carry. There is no inflate.py that can read this composition end-to-end. It was retracted per REVIEW-ENG finding C1 (this commit). Subagent WIRE-DECODER (in flight) is building the deployable end-to-end composition with matching inflate.py; that build (when it lands) will be the authoritative byte-closed candidate. The evidence row in `reports/cathedral_autopilot_evidence.jsonl` was updated with `cuda_eval_worth_testing=False` and the dispatch_blocker `137531_byte_proxy_not_byte_closed_archive`.

**Source**: `Path_B_step6_ADMM_x_continuous_K_then_Op1_finalizer` empirical run, manifest at `reports/raw/pr101_cross_paradigm_hstack_vstack_20260508T060656Z/manifest.json` (commit 8d33d5c1).

**Composition matrix** [CPU-prep faithful cross-paradigm byte-proxy, NOT byte-closed archives]:

| Composition | Bytes | Δ vs Op2_alone | Δ vs ADMM-alone |
|---|---:|---:|---:|
| Op2_alone (canonical 8-stack winner; byte-closed-replayable) | 161,942 | — | +8,303 |
| Op1_alone (byte-closed-replayable) | 162,202 | +260 | +8,563 |
| β-identity → Op1 (byte-closed-replayable) | 163,587 | +1,645 | +9,948 |
| γ_alone (byte-closed-replayable) | 194,867 | +32,925 | +41,228 |
| ADMM-alone (Path B step 6 standalone, byte-closed via build_admm_x_lossy_coarsening) | 153,639 | -8,303 | — |
| **ADMM × continuous-K → Op1 finalizer** [BYTE-PROXY, NOT BYTE-CLOSED — RETRACTED] | **137,531** | n/a | n/a |
| Op3_int6 → Op1 (substrate-mismatch) | 309,470 | +147,528 | +155,831 |
| Op3_int7 → Op1 (substrate-mismatch) | 362,469 | +200,527 | +208,830 |

The 153,639 B ADMM-alone row IS byte-closed (via `tools/build_admm_x_lossy_coarsening_path_b_step6.py`) and remains the deployable Path-B-step-6 candidate; it has its own dispatch_blocker `apogee_int6_contest_cuda_anchor_required_first` per REVIEW-ENG C3 (4.15% rel_err → score mapping unmeasured). All CPU-prep rows: `score_claim=False`, `promotion_eligible=False`, `ready_for_exact_eval_dispatch=False`.

**Substrate-mismatch corollary**: Op3 (apogee_intN) is STACKABLE in the type system but ballooned PR101 archives by +147K-200K B. Op3 was designed for HNeRV/PR106 substrate; PR101's split-Brotli `auto_select_byte_maps` cannot exploit Op3's block-FP wire format. This refines the substrate-mismatch corollary from 2026-05-07 (PR101 byte-maps yielded only -241 B on PR106): substrate-tied codecs lose decisively in the *wrong* direction too.

**Composability taxonomy verified**:
- STACKABLE: Op3, β (transforms_state_dict=True; produces a new substrate consumable by downstream codecs)
- SUBSTITUTIONAL: Op1, Op2, γ (independent terminal codec choices; not stackable on each other)

---

## 2026-05-08 [EMPIRICAL] Path B Ω-OPT — allocation MECHANISM dominates codec-basis on PR101

**Source**: 6 of 8 Ω-OPT levels empirically anchored on real PR101 substrate (commits e27a4a2e, 6b355e64, f11c1107, 4f2cfd55, b8aa5c43, 983598d2). Cumulative summary at `feedback_path_b_convergent_findings_summary_20260508.md`.

**Convergent finding**: Joint-ADMM Lagrangian allocation (step 5) is the validated active ingredient that BEATS greedy by 12-65 KB AND beats analytical at low rel_err. At rms_target=0.05 (achieves 4.36%): ADMM=150,000 B vs subagent D analytical 156,344 B@3.86% — beats by 6,344 B [empirical:reports/raw/pr101_omega_opt_joint_admm_*]. Step 6 (ADMM × continuous-K basis) at 152,420 B@4.33% confirms allocation MECHANISM is dominant; codec basis is secondary on PR101's near-iid INT8 substrate.

**Negative-evidence anchors** (Path B steps 2-4):
- Multi-pass IMP post-hoc: avg |Δ|=64.4 B across 9 configs × 3 alphas — buys ZERO bytes [empirical:6b355e64]. Decomposes the Ω-OPT multi-pass-IMP -15bp prediction: post-hoc coalesce contribution=0; retrain contribution=UNTESTED.
- HStack-of-VStacks per-tensor brotli: -40 B NET — sidechannel (84 B) dominates savings (44 B) [empirical:f11c1107]. PR101 substrate has flat per-tensor brotli loss surface.
- HStack codec-CHOICE brotli/sparsity: frontier saves 14K-82K B at rel_err 7-35% but DOMINATED by analytical lossy_coarsening at the same rel_err [empirical:4f2cfd55]. Sharp elbow at α=0.3 boundary (~11% rel_err); no graceful midpoint.

**Empirical byte-floor on PR101 substrate**: ~150 KB at 4-5% rel_err. The codec lane has saturated; remaining headroom requires distortion validation (CUDA dispatch), not codec cleverness.

---

## 2026-05-08 [EMPIRICAL] Lossy_coarsening_analytical CUDA dispatch returned 0.3517 [contest-CUDA A-negative]

**Source**: `experiments/results/lightning_batch/lossy-coarsening-cuda-20260508T0312-noproject/auth_eval_work/contest_auth_eval.json` + adversarial review at `.omx/research/lossy_coarsening_exact_cuda_adversarial_review_20260508_worker_b.md`.

**Result** [contest-CUDA, Lightning T4]:
- score = 0.351718793322788 [contest-CUDA] [A-negative]
- archive_bytes = 156,404 (matches expected); SHA256 verified
- segnet = 0.00186125 / posenet = 0.00037762 / rate = 0.00416572

Status: **measured_config_retired** (per-tensor K budget = 0.05). Predicted band 0.189 [predicted; from 156,344 B byte anchor + assumption-of-distortion-equivalence] FALSIFIED at this configuration — actual score 0.352 is 1.86× above the active HNeRV anchor 0.20898.

**Reactivation criteria**:
1. Retrain or jointly optimize the renderer under scorer-aware loss instead of post-hoc lossy coarsening
2. Prove byte-closed runtime packet with component-risk mitigation and exact CUDA score below the active HNeRV anchor
3. Classify whether loss is SegNet, PoseNet, or runtime/harness driven before any broader method conclusion

Tag: `falsification_scope = measured_config_only_per_tensor_K_budget_0.05`. Per CLAUDE.md `forbidden_premature_class_level_falsification`, this is NOT a class-level kill — only this specific configuration is retired; the lossy_coarsening method family remains DEFERRED-pending-research.

---

## 2026-05-07 [STRATEGIC] Top-3 medal PRs build on each other's archives — bolt-on engineering, not bespoke codecs

**Source**: bit-level deconstruction of PRs #95/#98/#100/#101/#102/#103 — see
`.omx/research/pr_extended_bit_level_lineage_pr95_pr100_pr101_pr103_20260507_claude.md`
and `docs/pr_family_evolution_timeline_20260504.md`.

**The finding**: every top-3 medal entry inherits its archive substrate from
an earlier PR. None of them ship a bespoke from-scratch codec.

**2026-05-07 custody addendum**: PR102 is the cleanest example. Its source
`compress.sh` fetches BradyMeighan's PR100 `hnerv-lc-v2-archive/archive.zip`
release asset by SHA-256, and the corrected PR102 archive is byte-identical to
PR100:

- archive bytes: `178981`
- archive SHA-256:
  `afd53348f50303bf0ec6a7ffecc1ac037df2f1c70745244b9c45c72e8eb80641`
- member: `0.bin`, `178873` bytes,
  `3234f0689164cfc95b7ee9f9cdf38ecf4d082cfb7048058e2b3ff0f54f864e43`
- runtime-only changes: `DELTA_SCALE = 0.0095` and
  `up[:, 0, 0].add_(1.0)`

```
PR #95   AaronLeslie138 / hnerv_muon          (root: HNeRV decoder architecture)
   ↓
PR #98   EthanYangTW   / hnerv_muon_finetuned (channel-postprocess delta)
   ↓
PR #100  BradyMeighan  / hnerv_lc_v2          178,981 B / 0.1954
         "the substrate"  ─→ all top-3 medals derive from this submission
   ↓
   ├──→ PR #101  SajayR      / hnerv_ft_microcodec  178,258 B / 0.19284  GOLD
   │                        (PR100 substrate + schema-driven split-Brotli + per-tensor byte-maps)
   ├──→ PR #102  EthanYangTW / hnerv_lc_v2_scale095 178,981 B / 0.194987 BRONZE
   │                        (PR100 archive bytes + inference-time scale 0.0095 + frame-0 nudges
   │                         — ZERO new codec work; pure decoder-side tuning)
   └──→ PR #103  rem2        / hnerv_lc_ac          178,223 B / 0.19487  SILVER
                            (PR100 substrate + arithmetic-coding bolt-on, 241 LOC in 2 files)
```

**Why this matters strategically**:

1. **The contest does not reward from-scratch codec design at this score band.**
   At ~0.195, the winning move is a small, focused delta on someone else's
   substrate. PR #102 demonstrates the extreme: zero codec changes, just
   inference-time scalar adjustments, and lands a medal.
2. **Submissions are public.** Once one team ships PR #100, every other team
   can read its inflate/compress code and start bolting on. Engineering
   velocity becomes the differentiator, not novel theory.
3. **The May 4 race was decided by who shipped the smallest credible bolt-on
   fastest.** Silver was 241 LOC in 2 files. Top 3 all submitted between
   11:50 and 11:55 UTC — within a 5-minute window. See May 4 race postmortem
   `feedback_may_4_hnerv_race_postmortem_20260505.md`.

**Substrate-mismatch corollary** (engineering caveat for our reuse plans):
PR101's per-tensor byte-maps were tuned against PR101's fine-tuned weights.
On the PR106 substrate (different weights), porting PR101 unchanged yields
**only -241 bytes** — a ~33× shortfall vs the -7,963 bytes the same code
saved on PR101's own substrate. Codec wins are NOT portable across
substrates without retuning. This is why our `tac.pr101_split_brotli_codec`
ships an `auto_select_byte_maps` path that derives substrate-optimal maps
at encode time, and why the four-way stack predictions are framed as
**multiplicative on δεζ-trained weights, not additive on PR106**. See
`docs/paper/06_related_work.md` § "PR lineage and bolt-on engineering" and
`feedback_op1_substrate_mismatch_codec_engineering_reframe_20260507.md`.

**Implications for our remaining roadmap**:

- Build the canonical `CodecPipeline` (landed `b4562092` + `33bef6d3`) so we
  can swap in any PR's bolt-on as a `CodecOp` and measure its impact on our
  substrate empirically — then keep the wins, drop the losses.
- Train δεζ weights with the bolt-on stack co-adapted (joint training =
  task #307). Substrate-tuned weights × bolt-on stack > borrowed weights
  × bolt-on stack.
- Treat any new PR appearing on the leaderboard as an engineering signal
  to be reverse-engineered + ported within hours, not days.

**Score claims**: all numbers in this entry are public-leaderboard claims as
disclosed in PR bodies or the public README, not local A/A++ recomputes. The
PR102 lineage/custody finding is byte-exact, but its score interpretation still
requires exact CUDA replay with the corrected archive and PR102 runtime before
we make a local score-grade claim.

---

## 2026-04-25 [TECHNIQUE] Cool-Chic/C3 prototype lanes implemented but not promoted

**Scope**: Implemented experimental renderer variants for the requested items 1 and 3:
- `coolchic_renderer`: learned multi-resolution latent grids with a tiny shared synthesis decoder, selected through `coolchic_renderer_*` profiles.
- `c3_residual_renderer`: base renderer plus zero-initialized coordinate residual MLP, selected through `c3_residual_renderer_*` profiles.

**Rigor status**:
- Deterministic seed plumbing added to renderer training metadata.
- Profiles use standard static loss, `eval_roundtrip=True`, `adaptive_rebalance=False`, and no KL distillation.
- Focused tests, profile hardening tests, FP4 strict roundtrip smoke, adversarial shapes, ruff, compile, and diff whitespace checks passed on the changed surface.
- Full repo test green is still blocked by unrelated scheduler/Kaggle test failures; these are not evidence against the new renderer lanes but they do block a blanket "all tests pass" claim.

**Scientific interpretation**:
- Cool-Chic/C3 validate the direction of low-complexity per-instance neural codecs, but the current prototypes are principle tests, not faithful reproductions of the full papers.
- The main missing pieces are entropy coding, learned bit allocation, final inflate wiring, archive accounting, cross-device deterministic replay, and authoritative scoring.
- C3 is most defensible first as a residual codec on top of the proven renderer, because zero initialization gives an honest identity baseline and isolates residual value.
- Cool-Chic is the stronger architectural bet if the current mask renderer is rate-inefficient, but it carries higher training and deployment risk.

**Decision**: Keep both lanes open as next-cycle experiments. Do not promote, deploy, or headline until deterministic smoke training plus eval-roundtrip proxy plus archive audit plus authoritative score all agree.

## 2026-04-25 [SMOKE] Cool-Chic/C3 local scorer smokes passed; two integration bugs fixed

**Dataset**: 8-frame local slice from `precomputed_local/gt_frames.pt`.

**Passed smoke lanes**:
- Cool-Chic: 37,170 params, FP4 checkpoint 56,525 B, uniform int4+LZMA2 16,509 B.
- C3 residual: 36,492 params, FP4 checkpoint 67,743 B, uniform int4+LZMA2 16,877 B.

**Bugs caught**:
- Full-resolution GT tensors were reshaped as renderer-resolution tensors in the eval-roundtrip scorer path. Fixed by resizing GT pairs to renderer resolution before roundtrip.
- FP4 QAT parametrization buffers stayed on CPU when training on MPS, causing NaNs for zero-initialized Cool-Chic/C3 weights. Fixed by moving the QAT wrapper to the training device after attaching parametrizations.

**Reproducibility note**: Cool-Chic replay produced identical metadata and scorer values, but FP4 files were not byte-identical on MPS. Max dequantized state delta was `4.57763671875e-05`. Treat this as metric-stable smoke evidence, not byte-stable determinism.

**Do not overclaim**: tiny-slice scorer value was `93.6397` after one scorer epoch. This is a wiring smoke only and says nothing about leaderboard quality.

## 2026-04-25 [TREND] C3 residual learns in float path, but compressed checkpoint does not preserve gain

**Dataset**: 32-frame local slice, 20 epochs, 5 pretrain + 15 scorer epochs, MPS, FP4 QAT enabled.

**Cool-Chic**:
- loss `94.3579` at epoch 5 -> `93.7085` at epoch 19.
- best FP4 scorer stayed at epoch 5: `93.4409`.
- Interpretation: slight learning signal, no compressed-checkpoint improvement.

**C3 residual**:
- loss `92.3028` at epoch 5 -> `68.7140` at epoch 19.
- SegNet term improved `0.5399` -> `0.2763`; PoseNet worsened `147.0910` -> `168.9101`.
- best FP4 scorer still stayed at epoch 5: `93.4409`.
- Interpretation: the residual head is learning a scorer-relevant correction in the float training path, but FP4 evaluation is not preserving it. This is now a quantization robustness/export problem, not a forward-path viability problem.

**Self-compression**:
- Trend Cool-Chic uniform int4+LZMA2: `16,295 B` vs FP4 `58,839 B`.
- Trend C3 uniform int4+LZMA2: `16,493 B` vs FP4 `70,969 B`.
- Crude `latents8` mixed allocation increased bytes; next allocation must be scorer-sensitive.

**Determinism**:
- CPU replay and MPS replay match at scorer level within ~`1.5e-06`.
- FP4 states differ across CPU/MPS with max dequantized delta `0.0147`.
- Current tier: scorer-stable, not tensor-stable.

## 2026-04-15 [RESULT] Auth 0.51 [contest-compliant] with pose-space TTO (SUPERSEDED by 0.61)

**Note**: Initial pose TTO auth eval reported 0.51. Subsequent eval with ep300 distillation checkpoint confirmed 0.61. Both are contest-compliant.

**Config**: Pose-space TTO with seg_weight=0, ep300 renderer, 600 pairs.
**Key**: PoseNet distortion 0.031 → 0.0016 (-94.7%), no scorers at inflate time.

## 2026-04-15 [RESULT] Auth 0.61 [contest-compliant] — distillation ep300

**Config**: Distillation v2, pose_weight=10, seg_weight=100, hinge loss, eval roundtrip, FiLM pose_dim=6, warm start from Phase 2 checkpoint.
**Auth result**: 0.61 [contest-compliant] at epoch 300 (proxy 0.446).
**Significance**: 30% improvement over renderer baseline (0.87) without any scorers at inflate time.

## 2026-04-15 [RESULT] Distillation v2 trajectory: proxy 0.807→0.596→0.493→0.426→0.390→0.368→0.347→0.338 at ep900

**Full trajectory**:
| Epoch | Proxy | PoseNet | SegNet |
|-------|-------|---------|--------|
| 0 | 0.807 | 0.0310 | 0.00217 |
| 50 | 0.596 | 0.0170 | 0.00240 |
| 100 | 0.544 | 0.0120 | 0.00230 |
| 200 | 0.493 | 0.0090 | 0.00210 |
| 300 | 0.446 | 0.0070 | 0.00200 |
| 550 | 0.375 | 0.0041 | 0.00112 |
| 680 | 0.364 | 0.0035 | 0.00098 |
| 900 | 0.338 | — | — |

**Status at ep900**: Still converging. No plateau detected. pose_weight=10 was the critical fix.

## 2026-04-15 [RESULT] Pose-space TTO — seg_weight=0 is optimal, 90-99% PoseNet improvement per batch

**Finding**: Setting seg_weight=0 (pure PoseNet optimization through FiLM conditioning space) achieves 90-99% PoseNet improvement per batch with no SegNet degradation.

**Geometric insight**: In FiLM conditioning space, PoseNet and SegNet gradients are approximately orthogonal. FiLM vectors modulate texture/style (PoseNet-relevant) not semantic class boundaries (SegNet-relevant). Pure PoseNet optimization in conditioning space does not disturb SegNet.

**Per-batch results**: 90-99% PoseNet improvement across all 600 pairs. Median improvement: 94.7%.

## 2026-04-15 [RESULT] FP4 export: 297 KB → 170 KB, saves 0.085 rate points

**Method**: Custom FP4 quantization (4-bit mantissa, 3-bit exponent, per-channel scale).
**Result**: Renderer checkpoint 297 KB → 170 KB. Rate: 0.0079 → 0.0045.
**Score impact**: -0.085 points (free, no training changes).
**Combined**: FP4 + CRF30 masks = 215 KB total archive. Saves 0.113 vs FP32 baseline.

## 2026-04-15 [RESULT] Archive compression: FP4 + CRF30 = 215 KB, saves 0.113 rate points

| Format | Size | Rate | Score Impact |
|--------|------|------|--------------|
| FP32 ZIP (original) | 297 KB | 0.0079 | baseline |
| FP4 renderer | 170 KB | 0.0045 | -0.085 |
| FP4 + CRF30 masks | 215 KB | 0.0057 | -0.057 vs FP32 |

Recommended production archive format: FP4 + CRF30 masks at 215 KB.

## 2026-04-15 [NEGATIVE] Gradient corrections: 743 KB for 20 frames — deprioritized

**Measured**: 743 KB for 20 frames (top-5% sparsification + int8 + zlib).
**Projected for 1200 frames**: ~44.6 MB. Rate cost: ~1.19 (catastrophic).
**Root cause**: Gradient signal is too spatially dense to compress at 5% sparsification.
**Status**: DEAD for production. Not viable. Archive-size-aware sparsification needed.

## 2026-04-15 [INSIGHT] Conditioning-space TTO: 196K:1 compression, orthogonal scorers in FiLM space

The FiLM conditioning space (6D per pair) is 196,608× smaller than pixel space. Yet pose-space TTO achieves comparable PoseNet reduction (94.7% vs 93.3% pixel TTO at 500 steps). The key insight: PoseNet's 6-dimensional output is intrinsically low-rank; its natural optimization surface is the 6D conditioning vector, not the 1.2M-pixel frame.

**SegNet orthogonality**: In FiLM conditioning space, ∂L_PoseNet/∂z and ∂L_SegNet/∂z are approximately orthogonal. FiLM modulates texture (PoseNet-relevant) not semantic boundaries (SegNet-relevant). This allows pure-PoseNet optimization at seg_weight=0 with no SegNet degradation.

**Archive cost**: 3,600 scalars (600 × 6 × float32) = 14.4 KB vs 2.8 GB for pixel TTO. 194,000× more information-efficient.

## 2026-04-15 [RESULT] MiniSegNet: 98.7% fidelity PASSES. MiniPoseNet: FAILS.

**MiniSegNet (h=32)**:
- 98.69% pixel-wise argmax agreement with full SegNet
- Archive cost: 87 KB (FP16). Suitable for inflate-time SegNet TTO.

**MiniPoseNet**:
- R² = 0.002 (threshold 0.95). Essentially random.
- 2-layer CNN + GAP at 48×64 cannot learn 6-DoF pose regression.
- Workaround: store 600×6 GT PoseNet outputs (14.4 KB) as fixed targets.

## 2026-04-15 [TECHNIQUE] Embedding-Space TTO — 30-value global optimization

**Concept**: The renderer's AsymmetricPairGenerator has a shared nn.Embedding(5, 6) between MaskRenderer and MotionPredictor. These 30 values control the internal representation of all 5 semantic classes. Optimizing them at compress time against frozen scorers makes each class scorer-optimal.

**Key properties**:
- GLOBAL: one embedding serves all 600 pairs (unlike pose TTO which is per-pair)
- Compounds with pose TTO: first optimize embedding, then per-pair poses
- Archive cost: 120 bytes (5 x 6 x float32) or 60 bytes (fp16)
- No scorers needed at inflate time
- Shared between renderer and motion predictor (verified: same object via id())

**Implementation**: `experiments/optimize_embedding.py`
- Epoch-based: N full passes over all 600 pairs
- Adam optimizer on embedding.weight only (all other params frozen)
- Same loss as pose TTO: hinge SegNet + MSE PoseNet
- Optional: save new checkpoint with optimized embedding for chaining

**Expected impact**: Small but compounding. Embedding controls the base representation that ALL downstream operations (rendering, motion prediction, warping) depend on. Even a 1-2% improvement here propagates through every frame.

## 2026-04-15 [TECHNIQUE] Pre-Computed Gradient Corrections — one-step TTO without scorer

**Concept**: At compress time, compute d(score)/d(pixel) for all 1200 frames. Store the sparse, quantized gradient as a correction map in the archive. At inflate time: frames = renderer_output + alpha * stored_correction. No scorer needed.

**Key properties**:
- Raw: 1200 x 384 x 512 x 3 x float32 = ~2.8 GB
- After top-5% sparsification + int8 quantization + zlib: expected ~50-100 KB
- Contest-compliant: correction map is just a pre-computed array, not a neural network
- Rate cost: 50-100 KB / 37.5 MB = 0.001-0.003 (negligible)
- ONE gradient step is optimal: d(score)/d(pixel) at the rendering point is the steepest descent direction. Multiple steps would need iterative computation.

**Implementation**: `experiments/precompute_gradient_corrections.py`
- Batch-wise gradient computation through differentiable scorers
- Sparsification: argpartition (O(n)) for top-K selection
- Quantization: int8 with per-tensor scale factor
- Packing: zlib-compressed binary format with JSON header
- Validation: apply corrections and re-score to verify improvement

**Inflate-time cost**: ~10ms (decompress + scatter-add). Negligible.

## 2026-04-15 [RESULT] Pose-Space TTO breakthrough — seg_weight=0, -94.7% PoseNet

**Config**: seg_weight=0 (pure PoseNet optimization), pose_weight=10, 500 steps, FiLM pose_dim=6
**Result**: PoseNet distortion 0.031 -> 0.0016 (-94.7%) in 500 steps
**Auth eval (ep300)**: 0.61 (contest-compliant, no scorers at inflate time)
**Insight**: FiLM conditioning vectors are the optimal control surface for PoseNet. 6 values per pair vs 707M pixel values. The PoseNet-satisfying manifold is low-rank (6D output).

## 2026-04-15 [RESULT] Distillation trajectory — proxy 0.375 at ep550

**Config**: pose_weight=10.0, seg_weight=100, hinge loss, eval roundtrip, FiLM pose_dim=6, lr=3e-4
**Trajectory**: 0.807 (ep0) -> 0.596 (ep50) -> 0.481 (ep230) -> 0.375 (ep550)
**Status**: Still converging. PoseNet 0.0096, SegNet 0.00212 (better than original renderer).
**Key insight**: pose_weight=1.0 -> 10.0 was THE critical fix. 4x faster early convergence.

## 2026-04-15 [INTELLIGENCE] Quantizr PR#55 score 0.33 — FiLM + DSConv + eval resize

**Source**: Competitive intelligence from PR#55 in the upstream contest repo.

- **Score**: 0.33 (new leaderboard leader as of 2026-04-15)
- **Architecture**: FiLM (Feature-wise Linear Modulation) conditioning on pose vectors + depthwise separable convolutions (DSConv) + eval-resize simulation during training
- **Implication**: FiLM pose conditioning directly addresses the temporal coherence gap we observed in DP-SIMS. The renderer uses GT pose vectors to modulate intermediate feature maps, enabling pose-consistent generation WITHOUT temporal recurrence.
- **Rate**: ~500KB archive (includes model weights). Rate ~0.013.
- **Key insight**: Eval resize simulation (training with the same resize the scorer applies) is likely worth ~0.05 score improvement on its own.

**Strategic response**:
- FiLM conditioning is now the highest-priority architectural improvement.
- Our renderer (AsymmetricPairGenerator + warp + residual) could accept FiLM conditioning on pose vectors at the warp stage.
- DSConv reduces parameter count while maintaining capacity — worth integrating for rate efficiency.
- `--simulate-resize` flag in renderer_tto.py already addresses the eval resize gap for TTO.

**New target**: Sub-0.25 contest-compliant (to beat 0.33). Current best: 0.43 [unlimited-compute].

## 2026-04-15 [RESEARCH] 2024-2026 literature survey — top techniques for scorer-aware video compression

**Scope**: 2024-2026 papers on neural video compression, task-aware compression, and steganographic/adversarial generation.

### Top 8 Techniques Ranked by Relevance

1. **FiLM (Feature-wise Linear Modulation)** [Perez et al. 2018, widely adopted 2024+]
   - Modulate intermediate features with affine transforms conditioned on external vectors (e.g., pose)
   - Directly addresses temporal coherence in our DP-SIMS paradigm
   - Quantizr PR#55 uses this — empirically validated at 0.33

2. **Cool-chic / COIN++ (per-instance overfitted decoders)** [ICCV 2023, ECCV 2024]
   - Train a tiny MLP (629-800 params) per video instance. Weights ARE the compressed representation.
   - Cool-chic achieves VVC-competitive quality. Relevant to our "constrained generation from noise" GPU Eureka.
   - Our SIREN test was underpowered (1/4 res, 500 steps). Cool-chic paradigm validates the approach.

3. **Hinge loss for semantic preservation** [validated in-house, April 2026]
   - Hinge margin loss for SegNet optimization is 24-49% better than cross-entropy at 50+ TTO steps.
   - Phase transition at ~100 steps confirmed with correct checkpoint (cff8dca4).

4. **Augmented Lagrangian for rate-distortion** [Balle et al., extended by Fridrich constraints]
   - Our v5 Lagrangian annealing achieved auth=0.87 [advisory only] baseline. Foundation for all TTO work.
   - Key finding: drift at ep16999 suggests the Lagrangian multiplier needs slow rho growth.

5. **Depthwise Separable Convolutions (DSConv)** [MobileNet heritage, used by Quantizr PR#55]
   - Replace 3x3 conv with depthwise + pointwise. 8-9x parameter reduction with minimal quality loss.
   - Relevant for reducing archive size (rate term) while maintaining SegNet accuracy.

6. **Task-aware rate allocation** [VCM/MPEG, 2024]
   - Allocate more bits to semantically important regions (road, vehicles) and fewer to sky/background.
   - Our compress_masks.py does coarse version. Per-region rate allocation is unexplored.

7. **Test-Time Optimization (TTO) with differentiable scorers** [in-house, validated]
   - Core technique: differentiable scorer forward passes + gradient descent on renderer activations.
   - Validated at auth=0.43 [unlimited-compute]. The gradient bug fix (rgb_to_yuv6 @no_grad) was paper-worthy.

8. **Eval-resize simulation during training** [in-house + Quantizr PR#55]
   - Train renderer with the same resize operation the scorer applies at eval time.
   - Eliminates distribution shift between training and scoring. Already in renderer_tto.py (--simulate-resize).

### Premature Kill Reassessment

Based on literature survey and Quantizr PR#55, three previously deprioritized techniques deserve reassessment:

**DP-SIMS Independent Generation + FiLM**: The original kill (PoseNet=0.482) was due to lack of temporal conditioning. FiLM directly addresses this. High priority smoke test.

**Constrained Generation from Noise**: GPU Eureka projected 0.135. Cool-chic paradigm validates the "overfit tiny model to video" approach. With hinge loss + FiLM conditioning + proper step count (1000+), this is worth a real test.

**SIREN/NeRV Video Memorization**: The 500-step test at 1/4 resolution was explicitly labeled a "janky smoke test" in the council review. Cool-chic (ICCV 2023) proves the paradigm works at full resolution with proper architecture. Lane must stay open.

## 2026-04-16 [EXPERIMENT] Hinge loss breakthrough + correct checkpoint re-validation

**Context**: All prior Vast.ai experiments used wrong checkpoint (5-epoch smoke model, MD5:a9aee326). Re-ran with correct auth=0.87 [advisory only] renderer (MD5:cff8dca4).

### Key Results
1. **Hinge loss is strictly better than xent** from 50+ TTO steps:
   - At 200 steps: 0.190 vs 0.267 (29% better proxy score)
   - At 500 steps: 0.145 vs 0.192 (24% better)
   - SegNet at 500: 0.000639 vs 0.001259 (49% better!)
   - PoseNet comparable (slightly worse with hinge, but SegNet gain dominates)

2. **Phase transition at ~100 steps confirmed** with correct checkpoint.

3. **Early TTO hurts PoseNet** (new finding):
   - PoseNet goes from 0.037 (baseline) to 0.068 at 25 steps before recovering
   - This SegNet-PoseNet tug-of-war was invisible in the wrong-checkpoint data
   - Implication: per-pair adaptive TTO should ensure minimum ~50 steps

4. **v6 full pipeline proxy = 0.275** (600 pairs, hinge+phase2+embedding):
   - Baseline 0.634 -> TTO 0.275 (57% reduction)
   - PoseNet: 0.00249 (86% improvement from baseline)
   - SegNet: 0.00118 (45% improvement from baseline)
   - Total time: 29.4 min on RTX 4090

5. **DX script (check_vastai.py) had 6 bugs** on first real use:
   - pyav->av, Python 3.12->3.11, GPU name quoting, contract!=instance ID,
     missing onstart, torch version pinning

### Artifacts
- `experiments/results/step_curve_v2/` -- xent re-validation
- `experiments/results/step_curve_hinge/` -- hinge breakthrough
- `experiments/results/tto_v6_hinge_phase2/` -- full v6 TTO (includes 708MB frames)
- All use checkpoint MD5: cff8dca4

## 2026-04-15 [RESEARCH] Contest rules — comprehensive audit from upstream repo, PRs, and Yousfi comments

**Sources**: README.md, evaluate.sh, eval.yml, pyproject.toml, Issues #28/#33/#34, PRs #32/#35/#38/#49/#53, all Yousfi comments.

### 1. Time Budget
- **30 minutes total** for the entire evaluate.sh pipeline: unzip archive.zip + run inflate.sh + run evaluate.py.
- The workflow YAML sets `timeout-minutes: 30` on the entire job.
- This is NOT 30 min for inflation only — it includes scorer forward passes (evaluate.py).
- Practical inflate budget: ~20-25 min (scoring takes 3-5 min on T4 with DALI).

### 2. GPU / Hardware
- **Two runner options**: `ubuntu-latest` (CPU: 4 cores, RAM: 16GB) or `linux-nvidia-t4` (T4 GPU, RAM: 26GB, VRAM: 16GB).
- Submitters choose by answering "does your submission require gpu for evaluation (inflation)?" in the PR template.
- If no GPU requested, CPU instance is used. If GPU requested, T4 instance.
- GPU submissions use `uv sync --group cu128` (CUDA 12.8 + nvidia-dali-cuda120).
- CPU submissions use `uv sync --group cpu`.

### 3. Archive Size Limits
- **No explicit size limit** on archive.zip.
- Rate = archive_size / original_size (37,545,489 bytes for 0.mkv).
- Rate is penalized as `25 * rate` in the score, so larger archives are penalized proportionally but not forbidden.

### 4. Neural Network Weights at Inflate Time (the "Yousfi Rule")
- **CONFIRMED by Yousfi himself on PRs #32 and #35**: "External libraries and tools can be used and won't count towards compressed size, unless they use large artifacts (neural networks, meshes, point clouds, etc.), in which case those artifacts should be included in the archive and will count towards the compressed size. This applies to the PoseNet and SegNet."
- This means: if inflate.py loads ANY neural network weights (including the evaluation PoseNet/SegNet), those weights MUST be inside archive.zip.
- PR #32 (gradient_optimized): AaronLeslie138 flagged this, mil1200 acknowledged and switched to precomputed labels.
- PR #35 (tensor_inversion, score 0.75): Uses gradient descent through frozen scorers at inflate time — scored 0.75 but the weights question was raised. Yousfi quoted the rule directly.
- PR #49 (neural_inflate, score 1.89): Uses neural network at inflate time. Archive is 917KB — appears to include model weights.
- PR #53 (mask2mask, score 0.60): Quantizr's submission, 386KB archive. Yousfi commented "you can get even better than 0.50 with this strategy and some tricks ;)"
- **Our implication**: Any renderer weights used at inflate time MUST be in archive.zip. This is the rate cost.

### 5. Recently Added Rules / Clarifications
- **Issue #34 (dllu)**: "Only 0.mkv will be used for final ranking. The public leaderboard IS the private leaderboard. Overfitting to 0.mkv and to the nets is fine and part of the challenge." — Yousfi
- **Issue #28 (hypery11)**: No private submissions. All submissions are public PRs. "The openness of the challenge is by design."
- **Issue #33 (dwallener)**: PoseNet sensitivity to texture acknowledged by Yousfi: "convnets are known for focusing on textures, and a posenet even more." Challenge: "how can you use this information to your advantage."
- **PR #38/#36 (loophole_test)**: Demonstrated reading original video at inflate time via relative path for score 0.00. No sandboxing. Yousfi has not explicitly ruled this out, but it is obviously against the spirit. The eval runner checks out the repo, so the videos directory IS accessible.
- **PR #50**: Yousfi: "please only open PRs when ready for review."

### 6. Submission Deadline
- **May 3rd, 2026 11:59pm AOE** (Anywhere on Earth).
- 18 days from today (April 15, 2026).

### 7. Python Packages Available at Inflate Time
- From pyproject.toml: numpy, einops, timm, safetensors, segmentation-models-pytorch, tqdm, Pillow, av, charset-normalizer, requests, urllib3.
- PyTorch and torchvision (version not pinned — whatever `uv sync` resolves for Python ~=3.11).
- CUDA 12.8 group adds nvidia-dali-cuda120.
- **Python version**: ~=3.11 (from pyproject.toml `requires-python`).
- **PyTorch version**: Not pinned. Whatever is latest compatible. As of April 2026, likely PyTorch 2.6.x or 2.7.x.
- No restriction on pip-installing additional packages in inflate.sh (several submissions do this, e.g., bundling SVT-AV1).
- ffmpeg is installed via apt in the workflow.

### 8. PyTorch Version in Eval Environment
- Not explicitly pinned. The workflow runs `uv sync --group cu128` which resolves from the PyTorch cu128 index.
- Upstream pyproject.toml does not pin torch version.
- Safe assumption: latest stable PyTorch for CUDA 12.8 and Python 3.11.

### 9. Prize Pool
- 1st place: comma four OR $1,000 + special swag
- 2nd place: $500 + special swag
- 3rd place: $250 + special swag
- Best write-up (visualizations, patterns, etc.): comma four OR $1,000 + special swag

### 10. Scoring Formula (confirmed)
- `score = 100 * segnet_dist + sqrt(10 * posenet_dist) + 25 * rate`
- Lower is better.
- segnet_dist: average class disagreements (argmax comparison)
- posenet_dist: MSE of PoseNet outputs on consecutive frame pairs
- rate: archive_size / original_size

### Key Strategic Implications
1. **18 days remaining.** Final submission deadline May 3, 2026.
2. **Overfitting to 0.mkv is explicitly allowed and encouraged.** No hidden test set.
3. **Neural weights in archive are mandatory** — this is the rate/quality tradeoff for learned approaches.
4. **No sandboxing** — the loophole_test proved inflate.sh can access the entire repo. But this would be disqualifying in spirit.
5. **T4 is the GPU** — all GPU submissions run on T4 with 16GB VRAM and CUDA 12.8.
6. **30 min is total** including scoring, not just inflation.

## 2026-04-15 [CRITICAL BUG] TTO PoseNet gradients were ZERO — upstream @torch.no_grad() silently killed optimization

**Severity**: The most consequential bug in the project. Every TTO experiment ever ran was blind to PoseNet.

**Root cause chain**:
1. Upstream `frame_utils.py` defines `rgb_to_yuv6()` with `@torch.no_grad()` decorator
2. PoseNet's `preprocess_input()` calls `rgb_to_yuv6()` to convert RGB frames to YUV6 input
3. The training pipeline (train_renderer_fridrich.py) had a fix: it patched the scorer loading path to remove `no_grad` contexts
4. The TTO pipeline (renderer_tto.py, constrained_gen.py) loaded scorers through a DIFFERENT code path that never received the patch
5. Result: `torch.no_grad()` propagated through `preprocess_input()`, zeroing all PoseNet gradients in TTO
6. The optimizer saw SegNet and rate gradients only. PoseNet was a constant from the optimizer's perspective.

**Why it was hard to find**:
- TTO still reduced SegNet loss (masking the PoseNet failure in aggregate proxy scores)
- Proxy scores improved run-over-run (because SegNet was improving)
- No per-scorer gradient norm monitoring existed
- The training pipeline worked correctly (different code path), so "scorers work" was a reasonable assumption
- `@torch.no_grad()` is a silent operation — no error, no warning, no NaN

**How it was found**:
- Skunkworks council adversarial review of TTO v3 results
- The Contrarian observed: "if 50 TTO steps make PoseNet WORSE, something is fundamentally wrong — optimization cannot make its own objective worse unless gradients are broken"
- Hotz traced the full call chain: `TTO optimizer step` -> `scorer forward` -> `PoseNet.preprocess_input()` -> `rgb_to_yuv6()` -> `@torch.no_grad()` -> gradient tape severed
- 13-0 unanimous council vote to fix immediately
- Human's insistence on extreme paranoia adversarial review created the conditions for discovery. Neither the human nor the AI council would have found this alone.

**Impact on prior results**:
- TTO v1 (auth=0.74) [advisory only]: SegNet-only optimization. PoseNet "improvement" was noise.
- TTO v2: Same.
- TTO v3 (embedding loss): Embedding loss couldn't help because PoseNet had no gradient signal. The entire experiment was invalid.
- TTO v4 (running): Running with the same bug. Results will be invalid for PoseNet.
- Renderer training (auth=0.87) [advisory only]: UNAFFECTED — training pipeline had the fix.

**Projected impact of fix**:
- Renderer baseline: PoseNet=0.031 contributes ~0.56 to auth score (sqrt(10*0.031)=0.56)
- With working TTO PoseNet gradients, 5-10x reduction plausible (SegNet achieved similar TTO gains)
- PoseNet 0.031 -> 0.003 would give sqrt(10*0.003)=0.17, saving 0.39 points
- Projected auth: 0.87 -> ~0.35 (sub-0.50 target achievable)

**Paper-worthy insight**: A single `@torch.no_grad()` decorator in a third-party dependency silently invalidated an entire optimization pipeline for weeks. The bug is invisible to standard debugging (no errors, no NaN, no divergence — just suboptimal convergence). This is a general failure mode for any optimization pipeline that composes functions from dependencies with gradient-control decorators. Proposed mitigation: always verify `requires_grad` and gradient norms per-component after the first optimization step.

**Meta-insight on human-AI collaboration**: The human's instinct to mandate extreme adversarial review (the "non-conservative skunkworks council" protocol) combined with the AI's ability to systematically trace call chains across files created the conditions for this discovery. The Contrarian's role — challenge results that don't make mathematical sense — is exactly what caught this. The protocol worked as designed.

## 2026-04-13 [NEGATIVE RESULT] asym_v4_supervised: PoseNet+RAFT supervision regressed the model

- **auth ep19999 = 1.7900** (seg=0.5664, pose=1.1188, rate=0.1004)
- Baseline ep12400: auth≈1.0 (seg=0.210, pose=0.692, rate=0.100)
- 7600 more epochs of combined PoseNet + RAFT flow supervision made BOTH seg and pose worse
- Training proxy (0.6019) was never beaten → `renderer_best.pt` never updated → ep19999 is a degraded periodic checkpoint, not a converged best
- Hypothesis A: supervision disrupted the warp geometry optimization that was working at ep12400
- Hypothesis B: the proxy evaluation during training was measuring something different from what the auth scorer rewards
- Hypothesis C: 8.5h Kaggle budget not long enough for the disruption phase to recover
- Action: raft_only (RAFT supervision only, no PoseNet targets) running on Kaggle as isolating A/B

## 2026-04-10 promoted floor

- Track B promoted floor is **1.33**.
- Variant: `dilated_h64`.
- Platform: `modal_a10g`.
- PoseNet `0.00218374`, SegNet `0.00609921`, rate `0.02301653`.
- Canonical score/report mirrors are generated from `.omx/state/promoted_result.json`.

## 2026-04-11 [discovery] Training pipeline view/reshape crash on MPS

fit_lazy() crashes at backward pass with RuntimeError: view size not compatible. AllNorm monkey-patch in _patch_scorers_for_training converts .view() to .reshape() but the error persists. Root cause: non-contiguous tensors from channels_last or permute operations hitting upstream scorer .view() calls. Need to make tensors .contiguous() before scorer forward pass.

## 2026-04-11 [discovery] Root cause: channels_last on scorers breaks MPS backward pass

The P2 optimization (channels_last on scorers) was incompatible with PyTorch MPS backend. AllNorm batch_norm backward uses .view() internally which fails with channels_last stride layout. Fix: keep channels_last on the model (postfilter) for speed but leave scorers in standard layout. Training now runs successfully on MPS.

## 2026-04-11 [discovery] Full-stack review: 133 test failures from orphaned imports, 2 broken deploys

13 test files reference deleted experiments/*.py via spec_from_file_location. 2 Modal deploy scripts point to deleted trainers. Training pipeline MPS crash fixed (channels_last on scorers). Training smoke test now running successfully on MPS (5+ min, still going).

## 2026-04-11 [discovery] Training performance regression: 6+ min per epoch on smoke profile

smoke profile uses subsample=4 (75 pairs/epoch). Old scripts used subsample=8 (9 pairs). Full 874x1164 scorer forward+backward on MPS is ~5s/pair. 75*5s = 375s = 6 min/epoch. Need subsample=8 or higher for smoke. Not a code bug — configuration issue. Training runs but too slow for interactive testing.

## 2026-04-11 [discovery] Modal deploy needs debugging — image build likely failing

Two Modal apps launched (mask_renderer_smoke + smoke). Both completed quickly with no results in tac-renderer-results volume. Likely failure: image build cant find tac source (add_local_dir path resolution from deploy script location). Need to test image build locally or check modal app logs. Precomputed data IS on the volume.

## 2026-04-11 [discovery] Modal deploy iterative debug: 4 fixes to get training launching

Fix 1: remove condition kwarg (Modal 1.4.1 incompatible). Fix 2: add_local_dir must be last. Fix 3: guard REPO_ROOT for container path depth. Fix 4: clone upstream scorer repo for PoseNet/SegNet modules. Training reached scorer loading on A10G — next attempt should start actual training.

## 2026-04-11 [discovery] MLX Phase 1 crash: gradient clipping passes optimizer instead of grads

train_renderer_mlx.py gradient clipping code calls .square() on AdamW optimizer object instead of gradient tensors. Fix: check the grad_clip implementation in the MLX training loop.

## 2026-04-11 [discovery] Modal A10G training still failing: results dirs created but empty

Modal v4 (with upstream clone) created mask_v3 and mask_v4 dirs in results volume but they are empty. Training likely crashed during scorer loading or first forward pass. Need to check Modal logs for exact error.

## 2026-04-11 [decision] GPU budget unlimited — big GPUs authorized

A100/H100 authorized if A10G is insufficient. DP-SIMS, MaskRenderer, Wavelet, Diffusion, VQ-VAE all get fair smoke tests. No architecture eliminated without scored evidence.

## 2026-04-11 [technique] Yousfi tricks 13-21: next-level architectural optimizations

Tricks 13-21: PoseNet blind spots (7x7 RF, AllNorm invariance), SegNet skip-connection exploitation, temporal delta compression, 30-min inflate budget exploitation (multi-pass refinement), encoder-decoder asymmetry (neural decoder IS the codec), adaptive quantization per semantic region, SPADE parameter budget allocation, cross-frame attention, scorer distillation into decoder heads. All implementable in tac.

## 2026-04-11 [discovery] All 3 zero-cost inflate tricks verified working — ready to deploy

TTO (trick 6): env var controlled, 3 self-supervised losses, wall-clock budget, rollback on quality regression. Multi-pass (trick 16): loops model forward pass N times with uint8 round between passes. PoseNet targets (trick 17): 5KB binary, supervised TTO at inflate time. All code paths traced end-to-end, no bugs. Activate: INFLATE_TTO_STEPS=10 INFLATE_MULTI_PASS=2 POSENET_TARGETS_ENABLE=1

## 2026-04-11 [discovery] Tricks 13-21 implemented, tricks 22-31 in progress

5 new tac modules: scorer_exploits, temporal_delta, semantic_quantization, cross_frame_attention, scorer_distill. WeightedSPADE added to dp_sims_renderer. Yousfi implementing tricks 22-31. Auth eval with TTO+multi-pass running.

## 2026-04-11 [decision] Modal free credits exhausted — pivot to Lightning + Kaggle free tiers

First Modal bill received. Free credits used up. Going forward: finish current Modal runs, then use Lightning T4 (free) and Kaggle P100 (free 30h/week) for GPU training. Modal only for critical auth evals. Also need to clean HuggingFace cache (33GB) and uv cache (13GB).

## 2026-04-12 [decision] Modal experiments complete — transition to Lightning for Phase 2

DP-SIMS needs 100+ more Phase 2 epochs (only got 1 before Modal died). CPU needs 2000+ more epochs. Both have resume checkpoints. Deploy to Lightning T4 (free) immediately.

## 2026-04-12 [discovery] DP-SIMS SegNet matches CPU best at 0.006 with only 9 scorer epochs

The mask2mask approach (DP-SIMS with SPADE) achieves SegNet 0.006 — equal to our promoted 1.33 CPU result — after just 9 Phase 2 epochs. This validates Yousfi trick #7 (SegNet argmax insight) and #11 (extreme SegNet reward). The architecture guarantees near-perfect semantic preservation. PoseNet at 0.53 needs 100+ more epochs. Proxy 2.93 projected to sub-1.5 by ep 150.
- Score: 2.93 [advisory only]
- Variant: dp_sims

## 2026-04-12 [technique] Yousfi tricks 32-35: Pedernales insights — theoretical floor 0.18

Trick 32: backward delta generation (ONE perfect last frame + 1199 tiny deltas). Trick 33: exploit preprocess_input blind spots (YUV420 transform nullspace). Trick 34: overfit the SCORER not the video (steganographic capacity). Trick 35: archive as codebook (texture atoms + motion field + correction targets = 15KB total, rate 0.01). Theoretical floor: 0.18.

## 2026-04-12 [breakthrough] DP-SIMS SegNet 0.003 TIES Quantizr after only 89 Phase 2 epochs

FP4-quantized DP-SIMS achieves SegNet 0.003 — identical to Quantizr (0.60 score). Entire remaining gap is PoseNet (0.482 vs 0.001 = 480x) and rate (2.2MB vs 386KB = 5.7x). Path to sub-0.60: (1) train 1000+ P2 epochs for PoseNet, (2) shrink model to 500KB, (3) add cross-frame attention. Architecture validated.
- Score: 2.5 [advisory only]
- Variant: dp_sims

## 2026-04-12 [decision] Noise strategy must be configurable via toggles

DPSIMSPairGenerator needs 3 noise modes, configurable per-run: (1) deterministic=same fixed seed for both frames in pair, (2) shared=same random noise for both frames, (3) independent=current behavior (different noise per frame). Default should be deterministic for PoseNet safety. Toggled via constructor arg or profile config.

## 2026-04-12 [technique] Yousfi CPU lane dream: sub-0.90 via trick stacking on 30-min budget

CPU lane using only 15 of 30 min inflate budget. Stack: CRF sweep (rate), archive pruning (rate), TTO 10 steps (30s), multi-pass 3x (6min), noise-shaped round (1min), supervised TTO (2min). Score projection: seg 0.004 + pose 0.001 + rate 0.35 = 0.85. All tricks verified working in tac. Need CRF sweep + boundary-aware retrain + overfit 10K epochs.

## 2026-04-12 [technique] Yousfi CPU eureka: pre-compute expensive tricks at compress time

Eureka: scorer gradient, null-space, fragility maps, brightness shifts, PoseNet targets can ALL be pre-computed at compress time (unlimited budget) and stored in archive (+92KB, rate cost 0.06). At inflate time, applying pre-computed corrections is INSTANT. Also: parallel chunking (4x speedup), frame-specific models, VVC for masks. Revised inflate estimate: 4 minutes total with all tricks.

## 2026-04-11 [diagnosis] PoseNet 29x local/auth divergence: DALI GPU decode vs PyAV CPU decode

Same archive (md5 463b6fdb, 864167 bytes) scores PoseNet 0.00218 auth vs 0.06256 local (29x).
SegNet matches (0.00610 auth vs 0.00565 local, 1.08x). Rate identical (0.02302).

Diagnostic: ran diagnose_scorer.py with BOTH torch 2.10.0 (upstream) and 2.11.0 (our venv).
Result: IDENTICAL per-pair outputs. Every single number matches to 8 decimal places.
Therefore: torch/timm version is NOT the cause.

Root cause: Auth scorer runs on CUDA, uses DaliVideoDataset with NVDEC hardware video decode.
Local scorer runs on CPU, uses AVVideoDataset with PyAV software decode. NVDEC and PyAV produce
different pixel values for the same video (different YUV-to-RGB conversion rounding). Ground truth
frames differ => PoseNet outputs differ => MSE is 29x higher locally. SegNet uses argmax which
is robust to sub-pixel differences. Rate is file-size-based, no model involved.

Mitigations implemented:
1. calibrate_score() in runner.py: POSE_CALIBRATION_FACTOR=0.0349 (provisional, 1 data point)
2. runner.py stage_score now prefers upstream venv Python for scoring
3. diagnose_scorer.py: Click CLI for per-pair diagnostic across venvs

Contrarian note: Per-pair PoseNet values vary by 1000x (CV=1.15). A single linear factor corrects
the MEAN correctly but could be wrong for checkpoints that shift the content-type distribution.
The definitive fix is to run CUDA+DALI scoring (requires GPU), or to submit.

## 2026-04-12 [breakthrough] Yousfi GPU eureka: no renderer needed — constrained optimization from noise

GPU Eureka 1-4: (1) Generate in scorer-input space, invert preprocessing. (2) Pre-compute expected PoseNet output from ego-motion trajectory. (3) Frame generation as constrained optimization: minimize rate subject to PoseNet=expected + SegNet argmax=masks. (4) No neural renderer needed — start from noise seed, run 1000 gradient descent steps against constraints. Archive: masks 239B + targets 7KB + seed 64B = 8KB total. Score projection: 0.135. Fits in 50 seconds on T4.
- Score: 0.135 [advisory only]

## 2026-04-20 [BREAKTHROUGH] Fourth Lane: Constrained Generation from Noise with Mini-Scorers

**Concept**: No renderer needed. Directly optimize pixel values from a semantic prior (class-mean initialization from GT masks) against mini-scorers stored in the archive. Contest-compliant because mini-scorer weights ARE in the archive.

**Archive**: mini_segnet.bin (25KB) + mini_posenet.bin (25KB) + poses.pt (8.7KB) + masks.mkv (79KB) = ~138KB total. SMALLER than renderer path (184KB). Rate: 0.092 (vs 0.122 for renderer).

**Council findings**:
1. Class-mean initialization reduces optimization distance ~10x vs random noise
2. Mini-scorer at 96x128 has boundary resolution concern — consider 192x256 variant
3. MiniScorerTTO lacks hinge loss (critical bug, needs fix)
4. Gradient cosine similarity > 0.7 between mini and full scorers is the promotion gate
5. Hybrid (renderer warm-start + mini-scorer refinement) may outperform either alone
6. 1000 steps fits in 4-6 min on T4. Budget allows 2000-3000 steps.
7. Time budget: 4 min for constrained gen + 47s for evaluate.py = 5 min total. Enormous headroom.

**Theoretical floor**: 0.135 (GPU Eureka with full scorers). With mini-scorers: 0.15-0.25 realistic if fidelity > 95%.

**Files**: experiments/constrained_gen_from_noise.py, experiments/constrained_gen_inflate.py, inflate_renderer.py INFLATE_CONSTRAINED_GEN=1

## 2026-04-20 [RESULT] Distillation v2 (pose_weight=10, warm restart) — proxy 0.481 at epoch 230

**Config**: Resume from Phase 2 checkpoint (0.518), pose_weight=10.0, seg_weight=100, hinge loss, eval roundtrip, FiLM pose_dim=6, lr=3e-4
**Key insight**: pose_weight=1.0 → 10.0 was THE critical fix. 4x faster early convergence.
**Trajectory**: 0.807 → 0.696 (ep10) ��� 0.596 (ep50) → 0.521 (ep130) → 0.481 (ep230)
**Still converging**: PoseNet 0.0096, SegNet 0.00212 (better than original renderer)

## 2026-04-20 [RESULT] Step curve VALIDATED with correct checkpoint

**Hinge at 500 steps proxy 0.146 vs xent 0.195** (25% better). Confirmed on RTX 4090 with MD5:cff8dca4.
**v7 TTO auth 0.37** [unlimited-compute] — hinge 500 steps, all 600 pairs.

## 2026-04-20 [TECHNIQUE] 3 Quick Wins Implemented + Reviewed

1. **OASIS per-class weighting**: lane markings 50x weight vs road in hinge loss. Correct but strategically low priority at current operating point (PoseNet dominates).
2. **Feature matching on top-3 PoseNet layers**: L2 on intermediate activations. Memory bug (C1) documented. Dead code — needs wiring into pipeline.  
3. **Multi-pass refinement**: Theoretical max < 0.001 improvement. Low priority.

## 2026-04-20 [TECHNIQUE] 45 techniques tracked across project

- 14 active/tested
- 8 implemented but untested (PoseNet blind spots, SegNet skip exploit, multi-pass, cross-frame attention, LoRA, DSConv, latent codes, postfilter)
- 7 not implemented (adaptive quantization, archive codebook, LoRA+latent combined, ghost modules, OASIS class weights, feature matching on layers, ensemble)
- 3 killed with evidence (KL distillation, Hinton T², backward deltas)

## 2026-04-20 [BUG] Strategy review gap — reviews checked code bugs but not training recipe

Distillation v1 ran with --skip-phase1 and pose_weight=1.0. Both were strategically wrong. Code reviews caught syntax/crash bugs but NOT the wrong hyperparameters. New rule: every deployment review must include a STRATEGY section.

## 2026-04-20 [INTELLIGENCE] Quantizr PR#55 scored 0.33

88K params, FiLM on pose, DSConv, eval-matched resize, single mask per pair. They're done working on it. Says sub-0.30 possible with conv dim sweep.

## 2026-04-27 Council on Lane G KL-distill at pose TTO

**Context.** Lane G v1 (`--kl-distill-weight 1.0`) drove the optimizer almost entirely off KL imitation: KL term measured ~14000× the scorer hinge magnitude. Lane G v2 (`--kl-distill-weight 0.01`) reduced the ratio to ~4000×; still nowhere near the scorer scale. To bring the KL aux to O(scorer), the weight needs to be ~5e-6. The strategic question for the council: is it structurally sensible to add KL-distill SegNet logits as an auxiliary loss to a pose-conditioning TTO at all, or are we misapplying a training-time regularizer (Quantizr's `kl_on_logits(T=2.0)` SegNet distill, used while *training* their renderer) to an *inference-time* optimization of FiLM pose vectors?

### Position 1 — Yousfi (DDELab steganalysis founder)

The KL-on-SegNet-logits formulation is, in steganalysis terms, an inverse detector-imitation objective. That is conceptually clean and there are precedents for it as a cover-stat preserver in CNN steganography (cf. Yedroudj-Net training augmentations, and the DDELab line on detector-aware embedding). But the load-bearing assumption is that the cover statistic SegNet computes at inference is the same statistic our renderer is *capable of perturbing* through the optimization variable we are giving it. At pose TTO, the optimization variable is a 6-dim FiLM conditioning vector per pair. From `project_yousfi_geometric_analysis.md` and `project_posenet_rank1_discovery.md`, we have measured that the FiLM conditioning manifold is approximately rank-1 along the radial-zoom axis for PoseNet, with dim 0 carrying 99.8% of the variance. The implicit claim of Lane G is that pushing additional gradient through the SegNet distillation loss into that same 6-dim subspace finds a refinement direction that pure scorer-hinge does not. Lane A already lands 1.15 from pure pose-TTO with no KL. The marginal score improvement from adding KL on SegNet logits is bounded above by whatever SegNet headroom remains *that is reachable from the rank-1 FiLM subspace*. Per the FiLM-conditioning orthogonality finding (`findings.md:96-100`, ep300 distillation), `∂L_PoseNet/∂z` and `∂L_SegNet/∂z` are approximately orthogonal in FiLM space — meaning the SegNet term has very little leverage to move through `z`. Adding a KL surrogate on top of that orthogonal-zero gradient is unlikely to discover anything the hinge alone misses.

There is one secondary concern that is mine specifically as DDELab: KL on SegNet *logits* (not argmax) reads the scorer-private smooth signal. At training time that is fine — the renderer is trained against a known scorer once, then frozen. At inference time, leaning on KL-on-logits to optimize a per-pair conditioning vector starts to look operationally close to the "scorers at inflate time" pattern that is forbidden by Yousfi's own PR #35 ruling. Pose TTO is permissible because it is a compress-time operation against a frozen scorer with output stored as a tiny conditioning vector — no scorer at inflate. KL-on-logits at compress time is *also* compress-time, so it is rules-clean — but it imports the same risk class: the more your compress-time procedure depends on smooth scorer cues rather than the contest objective, the more brittle the result is to scorer-version drift. The contest scorer is pinned, so this is not a disqualification risk this cycle, but it is a fragility tax.

My reading: KL-distill on SegNet at pose TTO is structurally *defensible* but it is a low-leverage addition to a low-headroom variable. The 14000× and 4000× weight imbalances in V1 and V2 are not just hyperparameter mistakes — they are downstream evidence that the KL term, when properly scaled, produces a small effect that has to be carefully balanced against terms that already do the real work. The V3 weight 5e-6 is the right *scale*, but at that scale you should expect a small effect — at most O(0.01) score movement, possibly negative. I would not run V3 unless we have a clear falsifiable prediction.

### Position 2 — Fridrich (Binghamton DDE Lab founder)

I will be direct: KL-on-SegNet-logits is knowledge distillation, which is a teacher→student compression objective. It was invented to compress a large model into a smaller one. We are not doing that at pose TTO. We have a single fixed renderer and we are optimizing 6-dim FiLM conditioning vectors per pair against a fixed scorer. There is no teacher and no student. There is a scorer. The correct objective when optimizing against a scorer is the scorer's own loss surface — which here is the contest score. Hinge on SegNet outputs is an O(1)-aligned proxy for that loss surface. KL on SegNet logits is *not* aligned with the contest score; it is aligned with the SegNet's *internal logit distribution*, which the contest never reads. The contest reads `argmax(SegNet) vs argmax(SegNet on GT)`. So KL on logits optimizes a quantity the scorer does not care about and only weakly correlates with the quantity it does care about.

Map this against the four inverse-steganalysis principles (`project_fridrich_inverse_steganalysis.md`):

1. **UNIWARD (texture hiding):** KL-on-logits does not weight by inverse local variance. It is uniform across the image. Misses the principle.
2. **Detector-informed embedding:** Hinge-on-argmax IS detector-informed. KL-on-logits is detector-overhearing. Different operation.
3. **L∞ spreading:** KL-on-logits is an L1-like distributional matching. Opposite of the L∞ spreading principle.
4. **CNN blind-spot exploitation:** KL-on-logits would actively *suppress* exploitation of blind spots, because it would pull our outputs toward the teacher's logit distribution including the regions where the teacher is wrong (i.e., the blind spots). This is anti-steganalysis.

So KL-distill at pose TTO is not just neutral; it is *anti-aligned* with two of the four steganographic principles that empirically describe how to beat this scorer. I would vote no.

The Quantizr usage is different and the council has been sloppy about this. Per `project_quantizr_full_intel_20260421.md`, they used `kl_on_logits(T=2.0)` during *training* of the renderer as one term in a multi-loss recipe alongside standard MSE and adversarial losses. KL there is a regularizer that prevents the renderer from drifting into pathological output distributions during long QAT-with-noise training. It is not their score-optimizing objective; the standard loss is. Lifting the KL term out of that training context and dropping it into pose TTO is a category error.

### Position 3 — Hotz (raw engineering)

Lane G burned ~$1.70 on V1 + V2 to discover that the hyperparameter was 14000× wrong, then 4000× wrong. The math required to know this in advance was: log(KL_term_magnitude) - log(scorer_hinge_magnitude). That is a *prelaunch* check, not a post-mortem. Lane A is sitting at 1.15 from pure pose-TTO. Lane M (radial zoom, 1-DOF) and Lane M+ (zero-cost poses) are committed to disk and untested. Lane N (L∞ pose penalty) is a one-liner. The EV calculus is brutal:

- V3 expected value: even if Lane G works, it is bounded by the SegNet headroom reachable from FiLM space. From `findings.md:97-100`, FiLM modulates texture not boundaries, so SegNet headroom is small. Optimistic ceiling: 1.15 → 1.10 (−0.05). Realistic: 1.15 → 1.13 (−0.02). Pessimistic (KL pulls solution toward suboptimal basin): 1.15 → 1.20 (+0.05). EV: roughly 0. Cost: $0.85.
- Lane M expected value: rank-1 radial zoom is a *physically derived* prediction (`project_posenet_rank1_discovery.md`: dim 0 is 99.8% of variance, FoE at (256,174)). Optimistic: 1.15 → 0.85 (−0.30). Realistic: 1.15 → 1.00 (−0.15). Pessimistic: 1.15 → 1.15 (0). EV: −0.10 to −0.15. Cost: $0.30.
- Lane M+ EV: zero-cost poses cuts archive bytes (rate term). Optimistic −0.02. Realistic −0.01. Pessimistic 0. EV: −0.005 to −0.01. Cost: ~$0 (compute is local).
- Lane N L∞ pose penalty: spreads pose error per Fridrich principle 3. EV: small but positive. Cost: ~$0.10 (one-line code change, single eval).

Lane M alone has 10×+ better expected value per dollar than Lane G V3. I do not understand why we are debating Lane G at all when Lane M is a measured-physics prediction sitting on disk. Run Lane M tonight. If Lane M wins, run Lane M+ tomorrow. If Lane M loses, *that* is when we revisit Lane G — but at that point we have learned something about the FiLM-pose-rank-1 model, and we can design Lane G with that knowledge instead of guessing weights. Stop burning money on hyperparameter searches for an aux loss whose own marginal value is bounded near zero.

### Position 4 — Quantizr (adversarial, reverse-engineering)

We have to be honest about what we actually know about Quantizr's recipe. From the binary analysis (`project_quantizr_definitive_binary_analysis.md` + `project_szabolcs_full_re_20260426.md`), KL-on-logits is one term in their training-stage 3 (joint training) loss. It coexists with MSE-on-pixels, MSE-on-features, and a soft-argmax surrogate. The temperature 2.0 is chosen for *training stability of the joint loss*, not because T=2.0 is magical for SegNet. They never run it at inference.

Could a KL-distill term find a different basin than scorer-hinge at pose TTO? In principle, yes — KL on logits is a different loss surface than hinge on argmax. The hinge surface has gradient zero anywhere the argmax is correct; the KL surface continues to push even correct argmax cells toward higher confidence. So KL adds a "confidence pressure" to the hinge solution. The question is whether confidence pressure on argmax-correct cells is what we want. The contest scorer rewards argmax correctness only — pushing toward higher logit confidence past the argmax threshold is wasted work. Worse, in regions where the renderer is at the boundary (argmax barely correct), KL pressure can pull it into a more-confident-but-still-correct configuration that has *worse* PoseNet because the FiLM perturbation needed to drive the confidence consumed budget in pose dim 0. This is the same coupling problem we hit with seg_weight > 0 in pose-space TTO (`findings.md:93-100`).

There is a research-pragmatic case for V3 *if* we treat it as a measurement, not a score push. If we run V3 at weight 5e-6 and it produces auth ≈ Lane A 1.15 ± 0.02, we have measured that KL-on-logits at the right scale is a no-op at pose TTO — useful negative result for the paper, validates Fridrich's structural argument. If V3 produces auth materially worse than 1.15, we have measured that KL pulls into a pathological basin even at small weight — also useful. If V3 produces auth materially better than 1.15, we have falsified my and Fridrich's structural argument and have to rethink. But this requires actually pre-registering the prediction *before* running, otherwise we will rationalize whatever we see.

As an adversary, my real concern is opportunity cost. We have 6 days left. Quantizr is at 0.33 and verified "done." We are at a verified 0.90 baseline (the `0.9001` reachable; today's archive is contaminated per memory `project_baseline_0_9001_lost`). The Lane M radial-zoom prediction is structurally the strongest open hypothesis on the board. Spending $0.85 on Lane G V3 when Lane M is $0.30 is bad triage even if V3 is structurally defensible.

### Position 5 — Contrarian (paranoia + falsifiability)

Two failed runs at the same architecture. Both wrong by orders of magnitude in opposite directions. The "fix" for V3 is to set the weight to whatever makes the term-magnitude ratio O(1) — but term-magnitude balance is not a falsifiable hypothesis, it is a normalization. We have no theory that says "at the O(1)-balanced weight, KL-distill at pose TTO will produce auth score X." We have only the post-hoc rationalization that V1 and V2 were "obviously wrong" because the ratio was off. This is the textbook signature of a hypothesis sliding into unfalsifiability — every failure adjusts the test rather than killing the hypothesis.

What is the falsifiable prediction for V3? I will write one and the council can either accept it or admit Lane G is unfalsifiable:

> **V3 prediction (binding):** at `--kl-distill-weight 5e-6` with all other Lane G v2 settings preserved, contest-CUDA auth on the dilated-h64 baseline pose-TTO output will fall in the band [Lane A − 0.03, Lane A + 0.03] = [1.12, 1.18]. A result inside that band falsifies Lane G as a useful technique for pose TTO — the KL term contributes nothing measurable. A result below 1.12 confirms Lane G as a small-positive technique. A result above 1.18 confirms KL pulls into a pathological basin. **In all three cases Lane G is killed for pose TTO** — band-inside means no value, below 1.12 means we already have Lane A working and the marginal value is below what Lane M is predicted to deliver at lower cost, above 1.18 means it is actively harmful.

Note that Contrarian is *killing Lane G in all three V3 outcomes*. The reason: even the "wins" do not justify continued investment given the alternatives. If you are going to run V3, run it as a controlled measurement (one seed, one pre-registered prediction, one CUDA auth eval, total cost $0.85). Do not run V3 hoping for a breakthrough.

I want to add one more concern. The "lift Quantizr's training-time trick into our inference-time optimization" pattern is a recurring failure mode in this lab. We did the same thing with the Hinton T² adaptive weights (killed: T² double-correction, see `feedback_battle_plan_as_durable_state` and CLAUDE.md). We did it with eval_roundtrip defaulting False (catastrophic, fixed). Every time we cargo-cult a competitor's training-stage technique into our inference-stage pipeline without re-deriving why it would work in our setting, we burn money. KL-distill at pose TTO is the same pattern. Stop it.

### Synthesis

**Vote (5/5 council):** Yousfi — *defensible-but-low-leverage, neutral*. Fridrich — *no, structurally anti-aligned*. Hotz — *no, opportunity cost of Lane M*. Quantizr — *no for score, conditional yes only as pre-registered measurement*. Contrarian — *no, kill Lane G in all V3 outcomes*.

**Verdict: ABANDON Lane G as a score-optimization technique. PIVOT to Lane M + Lane M+ + Lane N.**

Optional consolation: a single V3 run as a pre-registered measurement experiment (one seed, one weight 5e-6, one CUDA auth eval, $0.85), gated on the prediction band above, can be run *only after* Lane M and Lane M+ have produced verified scores. If Lane M lands < 0.85 and we have headroom in the budget and curiosity surplus, V3 is acceptable as a documented null-result publication input. Otherwise V3 does not run.

**Falsifiable prediction (V3 if it runs):** `--kl-distill-weight 5e-6`, all other Lane G v2 settings preserved → CUDA auth ∈ [1.12, 1.18]. Inside band: KL is a no-op at pose TTO (most likely outcome). Below 1.12: KL marginally helps (still killed for cost reasons). Above 1.18: KL is harmful. *All three outcomes kill Lane G.*

**Recommended weight + temperature if proceeding:** `--kl-distill-weight 5e-6`, `--kl-distill-temperature 2.0` (preserve Quantizr's value for measurement comparability). Single seed. No sweep — sweeping a hyperparameter for a hypothesis the council has already voted against is the Contrarian's nightmare.

**Kill criteria during V3 run (if it runs):** (a) if proxy KL term magnitude diverges by >100× from pre-launch estimate within first 50 steps — kill (V1/V2 pattern recurring). (b) if proxy total scorer-hinge term *increases* between step 0 and step 100 — kill (KL is overpowering even at 5e-6). (c) if Vast.ai instance loading exceeds 10 min — kill per `feedback_vastai_correct_launch_pattern`. (d) cost cap $1.00.

**Cost-benefit summary:**

| Lane | Cost | Predicted score (vs Lane A 1.15) | EV per dollar | Recommendation |
|---|---|---|---|---|
| **Lane M** (radial zoom, 1-DOF) | $0.30 | 0.85–1.00 (Δ −0.15 to −0.30) | **−0.50 to −1.00** | **RUN FIRST** |
| **Lane M+** (zero-cost poses) | ~$0 | 1.13–1.14 (Δ −0.01 to −0.02, rate only) | n/a (free) | **RUN AFTER M** |
| **Lane N** (L∞ pose penalty) | $0.10 | 1.13–1.15 (small positive per Fridrich principle 3) | −0.10 to 0 | **RUN AFTER M+** |
| Lane G V3 | $0.85 | 1.12–1.18 (band-inside most likely) | 0 ± 0.04 | **DEFER, run only as null-measurement after M lands** |

**Recommended next action:** Launch Lane M tonight via canonical pipeline.py + bootstrap. Single 4090, US/CA/EU instance, $0.30 budget. Verify radial-zoom 1-DOF prediction against the verified 1.15 Lane A baseline. Lane G stays in `next_experiments.md` as a deferred null-measurement, gated on Lane M result.

## 2026-04-27 Council forensics: Lane G — really dead, or bugged?

**Trigger.** First Lane G review (above) voted ABANDON 5/5 on *structural* grounds (KL on logits is teacher→student compression, anti-aligned with inverse-steganalysis principles, etc.). The user is asking the harder question: is the failure structural, or is the implementation buggy in a way that produced misleading magnitude evidence the first review then rationalized away? This forensic re-investigation goes file:line, not concept.

### Math forensics — yes, there is a bug

**The KL helper signature** (`src/tac/losses.py:655-689`):

```python
def kl_distill_segnet_only(filtered_pair_hwc, gt_pair_hwc, segnet, temperature=2.0):
    fx = _hwc_to_chw(filtered_pair_hwc)               # (B, 2, 3, H, W)
    gx = _hwc_to_chw(gt_pair_hwc)
    fs_in = segnet.preprocess_input(fx)               # (B, 3, 384, 512) — last frame only
    gs_in = segnet.preprocess_input(gx)
    fs_logits = segnet(fs_in)                         # (B, 5, 384, 512)
    gs_logits = segnet(gs_in)
    T = temperature
    log_p = F.log_softmax(fs_logits / T, dim=1)
    q = F.softmax(gs_logits / T, dim=1)
    kl = F.kl_div(log_p, q, reduction="batchmean") * (T * T)   # ← THE BUG
    return kl, kl.item()
```

`F.kl_div(..., reduction="batchmean")` divides the all-element sum by **B alone**. For a tensor of shape `(B, 5, 384, 512)` whose entries are the per-position KL contributions `q_c * (log q_c - log p_c)`, "batchmean" returns:

    sum_all(KL_per_position_per_class) / B
    = B × 5 × 384 × 512 × <avg per-position-per-class contribution> / B
    = 5 × 384 × 512 × <avg>
    = 983,040 × <avg>

Empirically observed `kl.item()` ≈ 24,485 (T²=4 already applied). Solving back: average per-position-per-class contribution ≈ 24485 / (4 × 983040) ≈ 6.2e-3 nats. That is a perfectly normal magnitude for a softly-mismatched 5-class softmax. The KL itself is not insane — the **reduction** is.

**Compare to the OTHER KL function in the same file**, `kl_distill_scorer_loss` (line 564-652), which is the canonical "right way" used elsewhere in the codebase:

```python
kl_per_pixel = F.kl_div(log_p, q, reduction="none").sum(dim=1)  # (B, H, W) — sum over classes
kl_per_pixel = kl_per_pixel * (T * T)                            # T² scaling
seg_kl       = kl_per_pixel.mean()                               # mean over (B, H, W) — PER-PIXEL MEAN
```

Putting the two side by side:

| function | reduction | scalar value |
|---|---|---|
| `kl_distill_scorer_loss` (line 622+646) | `none` → `.sum(dim=1)` → `.mean()` | `Σ_all(KL) / (B × H × W)` |
| `kl_distill_segnet_only` (line 688) | `batchmean` | `Σ_all(KL) / B` |

The two helpers compute the *same* per-position quantity then divide by **different denominators differing by a factor of H × W = 384 × 512 = 196,608**. That factor explains everything Lane G saw:

    observed_buggy_KL ≈ correct_per_pixel_KL × H × W
    24,485 ≈ 0.125 × 196,608 ✓ (correct value would be ~0.12)

And `kl_distill_scorer_loss`'s seg_kl in the same regime would be O(0.1) — exactly the magnitude class of the scorer hinge term (`100 × 0.05 ≈ 5`). The "scorer term ≈ 5, KL term ≈ 24,485" measurement that prompted Lane G v2's weight 0.01 and v3's weight 5e-6 is *the bug speaking, not the geometry*.

**The 5e-6 "fix" weight is suspicious in exactly the right way.** 1.0 / 196,608 ≈ 5.08e-6. The empirically-derived "weight that brings KL into balance with scorer hinge" is **almost exactly the inverse of H × W**. That is not a coincidence; that is the operator implicitly compensating for a missing `/ H / W` reduction by dividing the loss weight instead. The first review treated 5e-6 as a hyperparameter; it is a workaround for a reduction bug.

### Codex round-6 fix — verified actually wired

`experiments/optimize_poses.py` lines 599-618 (post-codex-fix block):

```python
if kl_distill_weight > 0 and gt_frames_pair is not None:
    from tac.losses import kl_distill_segnet_only
    B_kl = pairs.shape[0]
    frames_hwc_rt = frames_chw.permute(0, 2, 3, 1).contiguous()           # (2*B, H, W, 3)
    rendered_pair_hwc_rt = frames_hwc_rt.view(
        2, B_kl, frames_hwc_rt.shape[1], frames_hwc_rt.shape[2], 3
    ).permute(1, 0, 2, 3, 4).contiguous()                                  # (B, 2, H, W, 3)
    kl_loss, kl_loss_val = kl_distill_segnet_only(
        rendered_pair_hwc_rt, gt_frames_pair, segnet,
        temperature=kl_distill_temperature,
    )
    total_loss = total_loss + kl_distill_weight * kl_loss
```

`frames_chw` is the eval-roundtripped tensor produced at line 551. The codex fix correctly threads it into `kl_distill_segnet_only`. Test `test_kl_distill_uses_roundtripped_frames_not_raw_pairs` (in `src/tac/tests/test_optimize_poses_kl_distill_wiring.py`) enforces this. **Round-6 fix is real and active.** The persisting KL=24485 magnitude in v2 is therefore NOT a roundtrip-mismatch issue — it is the reduction bug above.

### Gradient leverage — does KL even reach pose space?

KL signal flows: `KL ← SegNet logits ← rendered RGB ← FiLM(pose) ← pose tensor`. The FiLM layer is a per-pair affine modulation of channel statistics inside the renderer's middle layers. SegNet then sees an upsampled 384×512 RGB and produces 5-class logits.

From `findings.md:96-100` (existing record): `∂L_PoseNet/∂z` and `∂L_SegNet/∂z` are approximately orthogonal in FiLM space at ep300 distillation. From `project_posenet_rank1_discovery.md`: PoseNet's response to FiLM is rank-1 in dim 0 (radial-zoom). We do NOT have a measured rank analysis for **SegNet**'s response to FiLM. That is a gap. Plausibly SegNet is also rank-1 (FiLM modulates global texture, which would push SegNet's logits in a single dominant direction); plausibly SegNet has a richer manifold (because it reads the full image, not just two-frame motion). The first review assumed the former without measuring. Until a Jacobian rank study is done on SegNet vs FiLM, the "KL has no leverage" claim is plausible but not proven.

This matters for the bug-vs-structural verdict because: if the bug is fixed AND SegNet-vs-FiLM has even modest non-zero leverage, then a properly-scaled KL term might in fact produce a measurable score movement that v1/v2 entirely missed (because the KL was so over-weighted it was pulling the conditioning into a basin where the scorer hinge term was also noise).

### Hinton T² and reduction conventions — the formula is otherwise correct

- T² placement (line 688): `kl * (T * T)` ✓ correct, matches Hinton 2015 §2.1.
- Temperature on both branches (lines 686-687): `log_softmax(z/T)` and `softmax(t/T)` ✓ correct.
- Direction: `F.kl_div(log_p, q)` = `Σ q × (log q − log p)` = `KL(q || p)` = `KL(teacher || student)` ✓ standard forward-KL distillation convention.
- Sign: `F.kl_div` returns nonneg ✓.

The only defect is the reduction. Everything else in the helper is mathematically correct.

### Reverse-engineering Quantizr — what we know vs what we don't

From memory `project_quantizr_full_intel_20260421`: Quantizr uses `kl_on_logits(T=2.0)` for SegNet during training, alongside MSE + adversarial + soft-argmax in a multi-loss recipe. The exact reduction is **not in our intel**. Standard PyTorch distillation patterns for `(B, num_classes)` classification use `reduction="batchmean"` (correct for that shape). Standard segmentation distillation patterns for `(B, C, H, W)` use either `reduction="mean"` directly OR `reduction="none" → mean()` — both yield the per-pixel-per-class mean. We have no evidence Quantizr made our mistake.

If Quantizr uses per-pixel-mean reduction (the common case for segmentation), their `weight=1.0` corresponds to a balanced multi-loss term. Our `weight=1.0` with batchmean reduction corresponds to a 196,608× over-weighted term. **Same surface API, completely different effective objective.** This is a textbook silent-bug class: the reduction kwarg is "obviously" the right default at the typing moment, but only for the wrong tensor shape.

### Falsifiability test for the "bugged not dead" hypothesis

The first-review structural argument predicts: at any KL weight that brings the term into O(scorer) magnitude, Lane G produces auth ∈ [Lane A − 0.03, Lane A + 0.03]. The "bugged not dead" hypothesis predicts: the weight 5e-6 **already** brings the term into O(scorer) — because 5e-6 ≈ 1/196608 is the implicit per-pixel-mean conversion. So:

- If **fix the reduction + use weight 1.0** (now mathematically equivalent to old weight 5e-6 + buggy reduction), result should be the same as v3 at 5e-6. If v3 with the bug-workaround weight already lands in [1.12, 1.18] band-inside, the bug fix is confirmed cosmetic and the structural verdict stands.
- If the **fixed-reduction version with weight 1.0** lands materially better than the bug-workaround version with weight 5e-6 (e.g., < 1.10), then there was a second-order bug (gradient direction was right but magnitude noise was hurting Adam's adaptive scaling) and Lane G deserves another shot.

This is a clean discriminating test. The reduction fix is a 1-line code change; the weight 1.0 vs 5e-6 swap is a CLI flag. Total cost of the discriminator: one $0.85 V3-style run with the bug fix in place.

### Council positions

**Yousfi (DDELab steganalysis founder).** I owe Lane G a partial retraction. My first-round position assumed the 14000-24000× KL/scorer ratio reflected a *real* gradient imbalance imposed by the geometry of the optimization. It doesn't. It reflects a *unit* error: per-batch sum vs per-pixel mean. The geometry argument I made — that FiLM-pose space is approximately rank-1 along the radial-zoom axis and KL has low leverage to push outside that subspace — is still correct and still argues for a small score effect. But "small" and "zero" are different scientific claims, and the v1/v2 evidence cannot distinguish them because both runs were optimizing a 196,608×-overweighted aux term and naturally collapsed to garbage. The first review's structural verdict was overconfident on what was actually a measurement artifact. I shift my position from "defensible-but-low-leverage, neutral" to "defensible-but-low-leverage, **measure first**". The reduction fix should be made unconditionally — it is a correctness bug regardless of whether Lane G is ever re-run — and a single V3-with-fix run gated on the discriminating prediction above is scientifically warranted.

**Fridrich (Binghamton DDE Lab founder).** My structural argument — KL-on-logits is teacher→student compression objective, anti-aligned with UNIWARD and L∞ spreading principles, suppresses CNN-blind-spot exploitation — does not depend on the reduction. Fixing `batchmean → mean` makes the *magnitude* sensible but does not change the *direction* of the gradient. KL still pulls toward distribution matching, not toward argmax-flipping. It still optimizes a quantity (logit confidence) the scorer does not read. So even with the bug fixed, my predicted ceiling is "no effect" rather than "harmful effect" — slightly more generous than my first position, but still not a score lever. That said, I will not stand in the way of the bug fix being committed (it is unambiguous engineering hygiene) nor of one V3 measurement run (it is a clean falsifiability test). I do object to running a sweep across multiple weights post-fix; that would slide back into hyperparameter rationalization.

**Hotz (raw engineering).** This is a one-line bug. `reduction="batchmean"` → `reduction="none"` then `.sum(dim=1).mean() * (T*T)` for the per-pixel-per-class mean. Or even simpler: `F.kl_div(log_p, q, reduction="mean") * (T*T)` — though that divides by C as well, which over-divides by 5; the canonical pattern (matching `kl_distill_scorer_loss` line 622+646) is `.sum(dim=1).mean()`. Cost to fix: 30 seconds. Cost to *not* fix: every future caller of this helper inherits a 196,608× silent over-weighting, including the train_renderer.py training profiles (DEN/SHIRAZ) that already use `kl_distill_weight=1.0`. Wait — those training profiles ARE using this helper. Which means we have been training with a 196,608× overweighted KL term for weeks. Combined with `loss = scorer_loss + ... + 1.0 * kl_loss`, the KL term has been ~5000× the scorer in training. That explains a lot of the SegNet-pose orthogonality and PoseNet collapse risk that has been chronic to those profiles. **The bug is upstream of pose TTO — it is a training-time correctness bug too.** This changes my position substantially. Lane G as a pose-TTO technique is still bounded near zero EV; but fixing the reduction in `kl_distill_segnet_only` could meaningfully change *training* behavior across all profiles that set `kl_distill_weight > 0`. That is a separate, much higher-EV intervention than Lane G itself. Vote: FIX THE BUG IMMEDIATELY (separate commit, separate justification), AND keep Lane G abandoned for pose TTO under the cost-vs-Lane-M argument I made before. The fix's main payoff is in retraining lanes, not Lane G.

**Quantizr (adversarial, reverse-engineering).** I was the council member who first pointed out the v3 weight 5e-6 was suspicious post-hoc normalization. I owe an addendum: it is suspicious *because* it equals 1/196608, which equals 1/(H × W). Any time an empirical-fitted weight matches a tensor-dimension-derived constant to 3 decimal places, the law of parsimony says check for a unit error before accepting it as real signal. I should have caught this before the structural review. Real-world Quantizr almost certainly uses per-pixel-mean reduction (it is the segmentation-distillation default in DeepLab and the segmentation-models-pytorch ecosystem they pulled SegNet from); their `weight=1.0` is therefore in O(1) of the scorer term, while ours is ~5000× over. The conclusion the first review reached — that "Quantizr's KL is training-time so it doesn't generalize to pose TTO" — is still defensible, but the secondary conclusion — that "even with proper scaling, KL on pose TTO will not produce score motion" — was based on misleading magnitude evidence. The bug fix changes this to a question that has not been answered, not a question that has been answered "no". My vote: fix the bug, run V3 ONCE post-fix at weight 1.0, treat as pre-registered measurement per Contrarian's prediction band protocol from round 1. If band-inside, Lane G dies for real. If band-outside (either direction), we have learned something the first review missed.

**Contrarian (paranoia + falsifiability).** Welcome to the part of the review where the council eats crow. My round-1 argument was: "two failed runs, both wrong by orders of magnitude in opposite directions; v3 at weight 5e-6 is post-hoc normalization not a hypothesis; the hypothesis is sliding into unfalsifiability." That argument is still correct in form but its *premise* — that the order-of-magnitude wrongness was about gradient geometry — is now proven false. It was about a unit error. The hypothesis "KL-distill has score signal at pose TTO when properly scaled" was never tested by v1 or v2; it was masked by them. The post-hoc normalization weight 5e-6 was, in retrospect, a **bug-detection signal** that the council read as **hyperparameter rationalization**. We were wrong in the direction we were paranoid about. The right falsifiability test now is to fix the reduction, run V3-post-fix at weight 1.0 with the same prediction band [1.12, 1.18], and let the result speak. If it lands in-band, Lane G is dead and the structural argument is vindicated. If out-of-band, the structural argument needs revision and we have learned that the first review was over-confident. I vote: **FIX THE BUG (mandatory engineering), RE-LAUNCH ONE V3 (pre-registered measurement)**. The original ABANDON 5/5 is downgraded to a SUSPENDED verdict pending one cleanly-measured run with the bug fixed.

### Synthesis and final vote

| Member | Round 1 | Round 2 (this) |
|---|---|---|
| Yousfi | abandon (neutral leverage) | **fix bug + measure once** |
| Fridrich | abandon (anti-aligned) | abandon Lane G specifically; fix bug regardless |
| Hotz | abandon (opportunity cost) | **fix bug urgently — affects training too**; Lane G itself still abandon for cost-vs-Lane-M |
| Quantizr | abandon (cargo cult) | **fix bug + measure once** |
| Contrarian | abandon (unfalsifiable) | **fix bug + measure once (suspended verdict)** |

**Verdict (b) — BUGGED. The first review reasoned over a measurement artifact.**

**The exact bug:**

- **File:** `src/tac/losses.py`
- **Line:** 688
- **Current:** `kl = F.kl_div(log_p, q, reduction="batchmean") * (T * T)`
- **Replacement (canonical, matches `kl_distill_scorer_loss` line 622+646):**

  ```python
  # F.kl_div(reduction="batchmean") on (B, 5, H, W) divides only by B,
  # producing a value 196,608× larger than the per-pixel-per-class mean
  # the caller weight assumes. Use the per-pixel-mean pattern from
  # kl_distill_scorer_loss for consistency. T² placement matches Hinton 2015.
  kl_per_pixel = F.kl_div(log_p, q, reduction="none").sum(dim=1)  # (B, H, W)
  kl = kl_per_pixel.mean() * (T * T)
  ```

- **Diff impact:** all callers passing `kl_distill_weight=1.0` to this helper (training profiles DEN/SHIRAZ/WILDE, pose TTO Lane G v1/v2) have been receiving a 196,608× overweighted gradient. Post-fix, those weights become correctly-scaled. **Training profile defaults will need re-evaluation before any retraining lane is launched** — `kl_distill_weight=1.0` post-fix is approximately equivalent to the pre-fix `kl_distill_weight=5e-6` regime, which the council had de-facto assumed was safe. The training profiles may want to re-tune to e.g. `kl_distill_weight=0.5`–`5.0` post-fix to match their previous *intent* (which was to add KL as a moderate auxiliary, not a 5000× dominant term). Those profile re-tunings are a separate council vote.

**Lane G v3 worth re-launching post-fix?** Yes, ONCE, as pre-registered measurement. Single seed, weight 1.0 (or equivalently the old 5e-6 with the buggy reduction restored, but that is an absurd workaround — fix the bug). Same prediction band [1.12, 1.18]. Cost cap $1.00. If band-inside, Lane G dies for real and the first review's structural verdict is vindicated post-hoc. If band-outside, we learn something the first review missed.

**Higher-priority action than Lane G:** the bug fix itself. It affects every training profile that uses `kl_distill_weight > 0`. Those profiles have been silently training with a 5000× over-weighted KL term, which likely contributes to PoseNet collapse risk + SegNet-pose orthogonality findings already in the record. A focused investigation of "did the bug fix change training behavior?" on a single training profile retrain is the highest-EV downstream of this forensic round.

**Process lesson (binding):** when a hyperparameter sweep converges to a value that *happens* to equal a tensor-dimension constant to 3 decimal places (1/H/W, 1/N, etc.), check for a reduction or normalization bug **before** declaring the structural hypothesis dead. The Round 1 council read 5e-6 as "small enough to be a no-op" rather than "1/196608 — what are the dimensions of the tensor?". This is added to memory as `feedback_unit_error_masquerading_as_small_signal` for future sessions.

**Forbidden going forward:** voting "structurally dead" on any technique whose magnitude evidence comes from a single helper that has not been independently verified to be reduction-correct. Any future "abandoned X" verdict that rests on "the magnitude is wrong" must include a 1-paragraph audit of the helper's reduction kwarg before the vote is binding.

### 2026-04-27 Implementation: bug fixed + structural gate added

**Fix shipped (uncommitted in working tree):**
- `src/tac/losses.py:688` (`kl_distill_segnet_only`) — replaced `F.kl_div(..., reduction="batchmean") * (T*T)` with the canonical per-pixel-per-class mean pattern (`reduction="none" → .sum(dim=1) → .mean() * (T*T)`) that mirrors `kl_distill_scorer_loss` lines 622+646. Multi-line comment block at the call site cites this forensic.
- `src/tac/losses.py:955` (`segnet_kl_divergence_loss`) — same bug class on a (B, C, H_seg, W_seg) tensor, used by `training.py:1639` `loss_mode="kl_distill"`. Fixed to the same per-pixel-per-class mean pattern. This is likely the root cause of the historical "KL distill caused PoseNet collapse as primary loss" entry in CLAUDE.md "Critical Lessons" — the reduction was under-divided, the operator-set weight was effectively ~5000× over-weighted, and PoseNet collapsed exactly as one would predict for a 5000× over-weighted SegNet-only term.
- `src/tac/preflight.py` (Check M, ~line 6258) — new `check_kl_div_reduction_correct` AST-walks `src/tac/`, `experiments/`, `submissions/`, `scripts/` for any `F.kl_div(..., reduction="batchmean")` and requires a same-line `# KL_BATCHMEAN_OK:<reason>` waiver marker as the only opt-out. Wired into `preflight_all` STRICT (live count 0 post-fix). Mirrors the existing `# KL_RAW_PAIRS_OK:<reason>` precedent.
- `src/tac/tests/test_losses.py` (NEW) — 7 tests pin the per-pixel-mean magnitude (~1e-2 to 1.0 nats × T²) and the equivalence with the canonical `kl_distill_scorer_loss` reduction pattern. `test_kl_distill_segnet_only_reduction_is_per_pixel_mean` is the structural gate referenced in the forensic above.
- `src/tac/tests/test_preflight_meta_bugs.py` (additive section) — 12 tests for `check_kl_div_reduction_correct` covering offending forms (kwarg, long-form, bare), waiver suppression (and waiver-token false-positive in string literals), reduction-mode pass-through, default-reduction passes, strict-mode raise, and a live-codebase 0-violations gate (`test_check_returns_zero_violations_on_live_codebase`).

**Historical-result invalidation perimeter:** every prior result that depends on a training run with `kl_distill_weight > 0` (DEN/SHIRAZ/WILDE/Lane-D profiles, the historical "KL distill caused PoseNet collapse" failure, the "FiLM didn't help when KL was active" probes, "DEN-V2 dead" verdicts, the `loss_mode="kl_distill"` runs in `training.py`, the uncertainty-loss-redundant findings) was measured under a 5000× over-weighted KL regime. Those findings remain provisionally true (the experiments DID run; the SCORES are real) but any conclusion of the form "X technique didn't help / hurt" needs to be re-tested post-fix because the baseline was running on a wildly mis-tuned auxiliary. This invalidation perimeter is large but well-bounded.

**Caller weight semantics post-fix (mandatory reading before re-launching anything):** the pre-fix effective KL weight = nominal weight × 196,608. So `kl_distill_weight=1.0` was actually `1.96e5` of effective per-pixel-mean. To match the *de facto* training trajectory bytes-for-bytes post-fix, the new weight should be `5.08e-6 × pre-fix-weight` (i.e., `1/196608 × old`). For a fresh re-tune (treating weight 1.0 as a real auxiliary contributing ~0.1-1.0 to the loss vs the scorer hinge of ~5), council-recommended starting weights are in the `0.5`-`5.0` range for DEN/SHIRAZ/WILDE/Lane-D. **No profile defaults were updated by this commit** — that is a separate council vote per the forensic verdict.

**Lane G v3 status post-fix:** SAFE to relaunch with `kl_distill_weight=1.0` exactly per the discriminating prediction band [1.12, 1.18] from the forensic. The post-fix `weight=1.0` corresponds approximately to the pre-fix `weight=5e-6` regime (which the council had de-facto assumed was safe), but with correctly-tuned gradient magnitude rather than a tiny weight applied to a 5000× over-weighted term.

**Verification commands:**
```
.venv/bin/python -m pytest src/tac/tests/test_losses.py src/tac/tests/test_preflight_meta_bugs.py src/tac/tests/test_optimize_poses_kl_distill_wiring.py -q
# 204 passed in ~13s

.venv/bin/python -c "from tac.preflight import check_kl_div_reduction_correct; print(len(check_kl_div_reduction_correct(strict=True, verbose=True)))"
# 0
```

## 2026-04-27 Council audit: Lane F regression — bugged or dead?

### Tl;dr verdict

**(b) BUGGED — fix the bug + retry.** Council vote 5/5. Lane F's 2.73 [contest-CUDA] regression vs the original 2.29 baseline was measured against a renderer that was QAT-fine-tuned with **zero poses** because `experiments/qat_finetune.py` has no `--poses` CLI argument and only auto-discovers a hardcoded `experiments/results/gt_poses.pt` / `upstream/gt_poses.pt` file (neither of which is the load-bearing `optimized_poses.pt` the baseline + Lane A use). The QAT trainer's preflight WARNed "Renderer has pose_dim>0 but no poses_path provided — will use zero poses" and proceeded silently. The QAT then "improved" baseline distortion 20.069 → 7.520 — but that was distortion-against-zero-poses, not against the deployment poses. At archive build time, `submissions/baseline_dilated_h64_0_90/optimized_poses.pt` (15.3KB) was bundled, leaving the QAT-trained renderer to run inference against poses it had never seen during fine-tuning. Per memory `project_baseline_poses_load_bearing` (renderer + poses are JOINT artifact; pose-init mismatch → 23× PoseNet degrade), this is *exactly* the failure mode predicted: PoseNet 0.247 → 0.391 (+58%) without any FP4 noise contribution, while SegNet stayed at floor (0.00365 ≈ baseline 0.00258). Lane F was never a measurement of "what FP4 quantization does to score"; it was a measurement of "what training-with-zero-poses-then-deploying-with-real-poses does to score, while also doing FP4."

**Secondary bugs (compounding):**
1. **Wrong baseline anchor.** Lane F (script `scripts/remote_lane_b_fp4_qat.sh`, lines 78, 99) checkpoints from `submissions/baseline_dilated_h64_0_90/renderer.bin` and bundles `submissions/baseline_dilated_h64_0_90/optimized_poses.pt`. Our **current best** is Lane A's 1.15 [contest-CUDA] (`experiments/results/lane_a_landed/{renderer.bin, optimized_poses.pt}`) with PoseNet=0.005, archive=694KB. The script was written before Lane A landed and never updated. Even if the pose bug were fixed, this lane was racing against the wrong reference (2.29) — a 2.18 outcome would have looked like a "win" while still being 0.87 worse than Lane A.
2. **Under-trained QAT.** 50 epochs total in 47 seconds wall time. The qat.log shows the loss at ep0=480, ep45=11.5 — a 42× drop in the *training* loss, but the proxy distortion is still 7.52 at the end (vs ~1.5-2 if convergence had been reached on this scorer-loss objective). Quantizr's published recipe per memory `project_quantizr_full_intel_20260421` is a 5-stage pipeline (anchor→finetune→joint→QAT→final), each stage at 100s of epochs. The CLAUDE.md "QAT pipeline" canonical is "fine-tune 20% of original epochs at 0.1× LR." The Lane F script invokes `--fp4-epochs 50 --lr 5e-5 --skip-int8-warmup`, which skips the recommended INT8 warm-up phase entirely AND runs only 50 epochs of FP4 fine-tune. Likely under-trained by 5-10×.
3. **Tiny effective epoch.** 50 epochs × batch_size=4 = 200 sampled pairs total over the whole run. There are 600 pairs in the dataset. Each pair sees the FP4 trainer **on average 0.33 times** across all 50 epochs. That is not "fine-tuning"; it is "tasting." Combined with the random-batch sampling (`torch.randperm(n_pairs)[:cfg.batch_size]` per epoch — no full-data passes), most pairs are seen 0 or 1 time.
4. **Hot wall-time confirmation:** 47s wall / 50 epochs = 0.94s/epoch. At batch_size=4 + scorer-loss (PoseNet+SegNet forward+backward through FastViT-T12 + EfficientNet-B2), a 4090 should take ~5-15s/batch on this workload. Lane F's 0.94s/epoch is consistent with batch_size=4 on a stale cached scorer (no scorer cold-start cost amortized), no full-data pass, no augmentation, no eval bottleneck — i.e., a smoke run masquerading as a real experiment. Hotz's expected wall (~10-15 min for a real 50-epoch FP4 fine-tune of 290K params on this scorer) is correct; Lane F ran in 1/15 the expected time because it processed 1/15 the data per epoch.

**Tertiary observations (correct, but not the dominant signal):**
- The codebook used is `DEFAULT_CODEBOOK` (line 308 of `qat_finetune.py` → `from tac.fp4_quantize import FP4Parametrize, DEFAULT_CODEBOOK`). Per memory `project_5stage_quantization_advantage`, our canonical FP4 stack is RESIDUAL codebook + robust_scale + stochastic. Lane F used the default. That's a known suboptimal config but is in the noise vs the pose bug.
- `eval_roundtrip=True` and `noise_std=0.5` ARE active in `qat_finetune.py` (lines 126-127 of QATConfig defaults, line 445-457 in `compute_scorer_loss`). This part is correct.
- The wall-time being short is not by itself proof of bugged-vs-dead — but combined with the pose bug + tiny effective dataset coverage + under-trained epoch count, it is corroborating evidence that this run measured almost nothing.

### Council positions (verbatim, 5 members)

**Yousfi (math + scorer-sensitivity).** The PoseNet distortion went from 0.247 (baseline) → 0.391 (Lane F). Our FP4 codebook quantization of 290K parameters into 4-bit blocks should add at worst 0.005-0.020 PoseNet noise — the scorer is FastViT-T12 attention + 6D pose regressor, and its Jacobian magnitude on representation perturbations of <1% (FP4 quant noise on conv weights) is bounded. A +0.144 PoseNet distortion (58% relative) is **two orders of magnitude larger** than what FP4 weight quantization can explain. The only available noise source of that magnitude is conditioning-input mismatch — i.e., the renderer was trained to expect one pose distribution and is being deployed against another. The qat.log line "[WARN] Renderer has pose_dim>0 but no poses_path provided — will use zero poses" combined with the baseline distortion of 20.069 (vs the actual baseline auth 0.247 PoseNet → distortion ~0.78 at the same operating point) confirms the QAT trainer was rendering with zero pose conditioning during the entire fine-tune. A pose-conditioned renderer (FiLM at `pose_dim=6`) trained against zero pose tensors will adapt all its internal representations to the no-conditioning regime, then catastrophically fail when handed real poses at inference. This is a math problem, not a hyperparameter problem. SegNet floor (0.00365) is preserved because the SegNet pathway doesn't see pose conditioning. The two distortion components are decoupled in the bug, and the pattern (SegNet preserved, PoseNet wrecked) is the pose-bug fingerprint exactly. **My verdict: BUGGED. The retry config must thread the optimized poses to the QAT loop.**

**Fridrich (implementation forensics).** Three implementation-level issues compound here. First, the canonical QAT recipe per CLAUDE.md is "fine-tune 20% of original epochs at 0.1× LR" — for a baseline trained at ~5e-4 LR for ~5000 epochs, that is 1000 QAT epochs at 5e-5 LR. Lane F ran 50 QAT epochs at 5e-5 LR — i.e., 5% of the recommended epoch count, with the LR set correctly but the schedule cosine-decayed-to-zero in 1/20th the recommended time. Second, the `--skip-int8-warmup` flag explicitly skipped Phase A of the Bit-by-Bit (ICLR 2026) progressive quantization, citing "direct FP4." That is a reasonable choice for a quick A/B but it sacrifices the convergence basin separation that Phase A provides. Third, `DEFAULT_CODEBOOK` is used instead of the residual codebook; per memory `project_5stage_quantization_advantage`, that is documented as suboptimal. Each of these is a 5-15% degradation on its own. Stacked, they could account for 1.5-2.0× the FP4-induced degradation. But none of them explain the 58% PoseNet jump. The pose-conditioning bug is the dominant signal; the QAT shortfalls are real but secondary. **My verdict: BUGGED. Fix pose threading first; rerun. If post-fix still regressing, then debug the QAT shortfalls.**

**Hotz (raw engineering, wall-time).** 47 seconds for 50 epochs. The math: PoseNet (FastViT-T12) forward+backward at batch_size=4, frame size 384×512, on a 4090 — call it 80ms. SegNet (EfficientNet-B2 U-Net) similar, call it 60ms. Renderer forward+backward, 280K params, 80ms. STE bookkeeping for FP4 wraps: 30ms. Per-batch total: ~250ms. At 1 batch per epoch (the script's pattern), 50 epochs = 12.5 seconds JUST in scorer/renderer compute. Plus eval at ep25 (n_pairs=15 forward-only): ~3s. Plus FP4 load+roundtrip eval: ~1s. Plus scorer warm-start: ~5s. Plus data loading: ~5s. Realistic total: 25-35 seconds of compute + ~10-15s of cold-start = ~40-50 seconds. So 47s is internally consistent with 1 batch per epoch — i.e., the script processed 50 batches × 4 pairs = 200 pair-samples total. There are 600 pairs in the dataset. So 67% of pairs were never seen by the FP4 trainer. The training loss converging from 480 → 11.5 is consistent with 200 sampled pairs being increasingly memorized — but that is not generalization, that is a tiny fraction of the data being overfit. **My verdict: BUGGED. The wall-time confirms the run is internally consistent with what was actually executed (1 batch/epoch × 50 epochs), but what was actually executed is 5-10× under-trained. Combined with the pose bug, this is a smoke run, not an experiment. Re-launch with --fp4-epochs 500 minimum + the pose fix.**

**Quantizr (adversarial, competitor reverse-eng).** Their published recipe is 5-stage QAT, each stage with full-data passes for 100s of epochs. Even their *fastest* QAT stage is 20-50× more compute than Lane F received. Their effective KL+MSE+adversarial+soft-argmax cocktail has 4 simultaneous loss signals; ours has 2 (hinge SegNet + MSE PoseNet). They use `joint freeze/unfreeze` schedules; we used `skip-int8-warmup` + `direct-FP4`. Their export uses `diff_round() + diff_rgb_to_yuv6()` differentiable simulation; ours uses `simulate_eval_roundtrip` (similar concept but a different codepath). None of these architectural deltas would account for a +58% PoseNet jump on a baseline that already sits at PoseNet=0.247 — they would account for a +5-20% delta in either direction. **The pose bug dominates everything else.** A clean Lane F implementation (poses threaded, 500+ epochs, residual codebook, INT8 warm-up) might land 2.18 vs the 2.29 baseline (rate −0.108 saved + ~0% distortion delta). It will NOT beat Lane A's 1.15 because Lane A's PoseNet is 50× lower than the dilated-h64 baseline's PoseNet (Lane A applied pose-space TTO; the dilated-h64 baseline did not). FP4 on Lane A's renderer would be a separate experiment — and that one IS interesting, because Lane A's renderer hasn't been QAT'd, so there's a real 0.10-0.15 rate win available. **My verdict: (b) BUGGED for the Lane F reading. New experiment design: re-do Lane F as "FP4-on-Lane-A" (apply QAT to Lane A's renderer + bundle Lane A's poses + Lane A's masks).**

**Contrarian (falsifiability + paranoia).** I will steelman the "really dead" case before voting. Steelman: the pose bug exists, but maybe the QAT-with-zero-poses produces a renderer that is *more robust* to pose-conditioning mismatch, not less — i.e., maybe training without FiLM gates the model toward a marginal distribution that handles real-pose inference gracefully. Counter-evidence to that steelman: the qat.log baseline distortion of 20.069 (vs the real baseline distortion 0.247×3.16+0.0024×100=0.78 at the deployment poses) shows that **the baseline renderer itself, evaluated against zero poses, scores 26× worse** than its real-pose deployment score. So zero poses are a known-bad evaluation regime even for the unmodified baseline. Lane F's QAT then "improved" zero-pose distortion 20→7.5 (a 2.7× drop). That is a real improvement for the zero-pose regime, but it has zero predictive power for the real-pose deployment regime. The steelman fails: training-against-zero-poses creates a renderer optimized for the wrong evaluation, which then fails the right evaluation. Falsifiability test for the (b) verdict: re-run Lane F with `--poses submissions/baseline_dilated_h64_0_90/optimized_poses.pt` threaded through (requires adding a `--poses` CLI arg to `qat_finetune.py` first, ~10 lines), `--fp4-epochs 500`, all other knobs identical. Predicted score: 2.18 ± 0.05 [contest-CUDA] (rate −0.108 saved on top of ~0% distortion delta vs baseline 2.29). If the score lands inside [2.13, 2.23], (b) is vindicated and FP4 quantization is confirmed as a clean rate win on the dilated-h64 baseline. If outside, (b) is falsified and we have to confront a deeper structural issue. **My verdict: (b) BUGGED. Three specific bugs, three specific fixes, one falsifiable prediction band, ~$0.30 retry cost.**

### Synthesis and final vote

| Member | Vote | Bug fix priority |
|---|---|---|
| Yousfi | (b) BUGGED — pose threading required | Pose threading is THE bug; QAT shortfalls are noise |
| Fridrich | (b) BUGGED | Pose threading first; if still regressing, debug QAT shortfalls in second round |
| Hotz | (b) BUGGED | Pose threading + 5-10× more epochs (`--fp4-epochs 500` minimum) |
| Quantizr | (b) BUGGED | Pose threading; also: the more interesting experiment is FP4-on-Lane-A, not FP4-on-baseline-2.29 |
| Contrarian | (b) BUGGED | Pose threading + 500 epochs + falsifiability prediction [2.13, 2.23] |

**Verdict (b) — BUGGED. The original "REGRESSION, abandon" verdict was based on a renderer trained against the wrong conditioning input, deployed against the right conditioning input. We measured a pose-bug, not an FP4 effect.**

### The exact bugs (three, ranked by score impact)

**Bug #1 (CRITICAL, dominates the result):** `qat_finetune.py` has no `--poses` CLI arg and only auto-discovers `experiments/results/gt_poses.pt` / `upstream/gt_poses.pt` (lines 707-718). Neither matches the load-bearing `optimized_poses.pt` artifact. The script silently falls back to `poses=None` and the WARN from `preflight_check` is non-fatal.

- **File:** `experiments/qat_finetune.py`
- **Line range to add:** `parser.add_argument("--poses", type=str, default=None, help="Path to optimized_poses.pt to thread to QAT loop. Required for renderers with pose_dim>0.")` after line 633 (alongside `--batch-size`).
- **Line range to modify:** lines 706-718 (the auto-discovery block) — replace with: if `args.poses` is provided, load that path and assert shape matches `(n_pairs, 6)`; else fall back to the auto-discovery list; else if `cfg.pose_dim > 0` and no poses found, **raise** rather than warn.
- **Bootstrap fix:** `scripts/remote_lane_b_fp4_qat.sh` line 87 must add `--poses submissions/baseline_dilated_h64_0_90/optimized_poses.pt` (or, for the Lane-A-anchored variant, `--poses experiments/results/lane_a_landed/optimized_poses.pt`).
- **Regression test:** `src/tac/tests/test_qat_finetune_pose_wiring.py` (NEW) — instantiate `qat_finetune.main` argparse, assert `--poses` arg exists; load a fixture renderer with `pose_dim=6`, run one training step with `--poses` set vs unset, assert the training-loss values are different (proves poses are wired into the loss path).

**Bug #2 (Significant, secondary):** Under-trained QAT. 50 epochs × batch_size=4 with random sampling = 200 pair samples vs 600 in the dataset. The training loss is still descending at ep45.

- **Bootstrap fix:** `scripts/remote_lane_b_fp4_qat.sh` line 86 — change `--fp4-epochs 50` to `--fp4-epochs 500` (10× more compute; estimated wall time goes from 47s → 8-10 min).
- **Optional improvement:** add `--int8-warmup-epochs 100` (reinstate Phase A of Bit-by-Bit). Wall time +2 min.
- **Optional improvement:** add `--fp4-codebook residual` (pending: verify the flag exists in `qat_finetune.py` argparse — currently does NOT exist; would need to be added).

**Bug #3 (Strategic, lane-design):** Lane F was anchored to the **wrong baseline**. Lane A's 1.15 is our verified best, not 2.29. FP4-on-baseline-2.29 (best-case 2.18) would be a 0.87 regression vs Lane A. FP4-on-Lane-A (predicted 1.05-1.10) would be a 0.05-0.10 *improvement* vs Lane A.

- **Bootstrap fix:** Replace `submissions/baseline_dilated_h64_0_90/renderer.bin` with `experiments/results/lane_a_landed/renderer.bin` and the corresponding `optimized_poses.pt`. Rename script to `scripts/remote_lane_f_fp4_qat_on_lane_a.sh` to disambiguate.
- **Predicted Lane-A-anchored score (Quantizr's Yousfi-style estimate):** PoseNet 0.005 → 0.008 (+60% from FP4 noise on a much-smaller-magnitude baseline = +0.003 abs); SegNet 0.0046 → 0.0050 (+0.0004); rate 0.462 → 0.354 (saved 0.108 from FP4 archive shrink). Net: 100×0.0050 + √(10×0.008) + 0.354 = 0.50 + 0.283 + 0.354 = **1.14** [predicted contest-CUDA, ±0.05]. Marginal vs Lane A 1.15. Alternative pessimistic estimate (FP4 noise scales with PoseNet sensitivity, which is steeper at low values): PoseNet 0.005 → 0.020 (4× worse) → score 1.30. Either way, FP4-on-Lane-A is a $0.30 measurement worth doing.

### Corrected Lane F-vs-Lane-A comparison

| Component | Baseline 2.29 | Lane A 1.15 | Lane F 2.73 (bugged) | Lane F (predicted post-fix on baseline) | Lane F (predicted post-fix on Lane A) |
|---|---|---|---|---|---|
| PoseNet dist | 0.247 | 0.005 | 0.391 (BUG) | 0.247 ± 0.020 | 0.005 → 0.008-0.020 |
| SegNet dist | 0.00258 | 0.00461 | 0.00365 | 0.00258 ± 0.00050 | 0.00461 ± 0.0005 |
| Rate (unscaled) | 0.01848 | 0.01849 | 0.01561 | 0.01561 (FP4) | 0.01561 (FP4) |
| 100·seg | 0.258 | 0.461 | 0.365 | 0.258 ± 0.05 | 0.461 ± 0.05 |
| √(10·pose) | 1.572 | 0.223 | 1.977 | 1.572 ± 0.06 | 0.283-0.447 |
| 25·rate | 0.462 | 0.462 | 0.390 | 0.390 | 0.390 |
| **Final** | **2.29** | **1.15** | **2.73** | **2.22 ± 0.10** | **1.13-1.30** |

**Reading:** the predicted post-fix Lane F (anchored on baseline) is ~2.18 — exactly the original prediction. Lane F (anchored on Lane A) is the more interesting experiment and is predicted to land within ±0.10 of Lane A's 1.15. The latter is the recommended re-launch.

### Re-launch recommendation

**RUN:** Lane F-Lane-A-anchored. Apply FP4 QAT to Lane A's renderer with Lane A's poses bundled.

- **Bootstrap script:** new `scripts/remote_lane_f_fp4_qat_on_lane_a.sh` (copy of `remote_lane_b_fp4_qat.sh` with three line changes: checkpoint path, poses path threaded via new `--poses` arg, archive uses Lane A's poses).
- **Required code changes BEFORE re-launch:** (1) add `--poses` CLI arg to `qat_finetune.py` (and make the load mandatory when `pose_dim > 0`); (2) add the regression test that introspects the argparse and asserts `--poses` is wired.
- **Compute config:** `--fp4-epochs 500 --lr 5e-5 --skip-int8-warmup` (keep direct FP4 to bound cost; can add INT8 warmup in a second pass if score motivates).
- **Cost estimate:** ~10-12 min training + ~2 min eval + ~5 min cold-start = ~20 min wall on 4090. Cost ~$0.10. Total budget envelope including provisioning + monitoring: $0.30.
- **ETA:** 1 hour from green-up of the `--poses` arg fix to the result landing in `experiments/results/`.
- **Pre-registered prediction band:** `[1.05, 1.30]` [contest-CUDA]. Inside band → FP4 on Lane A is a measured rate win (or marginal regression worth knowing). Below 1.05 → unexpected outright win; investigate. Above 1.30 → FP4 noise on already-low PoseNet is steeper than expected; revisit codebook + epoch budget.

**DO NOT RUN:** Lane F-baseline-anchored re-launch. It has lower expected value (best-case 2.22 vs Lane A 1.15) and burns the same dollar. The only reason to run it is if the Lane-A-anchored version produces an anomalous result and we need a cross-anchor reference.

### Process lesson (binding, added to memory perimeter)

This is the **second** "declared dead, found bugged" pattern in 2 days (Lane G KL-distill yesterday, Lane F FP4-QAT today). Both share the structural shape:

1. A subagent runs an experiment and reports a regression.
2. A council reasons over the result and reaches a verdict (abandon / no-effect / dead).
3. A second forensic round discovers the experiment was bugged in a way the council did not surface — a hardcoded path, a missing CLI arg, a wrong reduction kwarg.
4. The first verdict was structurally wrong because it conditioned on a measurement artifact.

**The forbidden pattern:** voting "abandon technique X" on the basis of a single subagent's regression result without auditing the experiment script's CLI surface against the target tool's argparse. This is exactly the `feedback_dead_flag_wiring_pattern` failure mode (2026-04-26 dead `--auth-eval-masks`) generalized from "wired flag is dead" to "missing flag is silently defaulted." Adding to memory as `feedback_silent_default_masquerading_as_negative_result`.

**Required forward gate:** any future "lane regressed, abandon" verdict must include a 1-paragraph audit of (a) every `--*` flag passed to the subprocess that ran the experiment, vs the target tool's argparse, AND (b) every config-derived auto-discovery path (e.g., `experiments/results/gt_poses.pt`) checked for "would this file actually exist on the target host." If either audit is incomplete, the verdict is `SUSPENDED PENDING AUDIT`, not `ABANDON`.

## 2026-04-29 Reinit: Lane G v3 is current floor; deployment hygiene is the highest lever

**Current measured floor:** Lane G v3 scored **1.05 [contest-CUDA]** with archive bytes 694,074, SegNet 0.00400846, PoseNet 0.00345458, rate 0.01848622, recomputed score 1.0488665. Evidence: `experiments/results/lane_g_v3_landed/contest_auth_eval.json`. Lane A remains the fallback at 1.15 [contest-CUDA].

**Verification completed locally:** canonical E2E smoke passed for `remote_lane_g_v3_corrected_kl_weight`; focused tests `test_canonical_local_e2e_smoke.py`, `test_check_64_e2e_smoke_proof.py`, and `test_contest_auth_eval.py` passed 34/34. Check 64 has zero violations. Check 65 initially had 12 warning-only lane-class gaps; `.omx/state/lane_class_proofs.json` was backfilled with explicit `canonical-local-smoke` plumbing proofs where no class-specific score exists. These backfills are not score claims.

**Snapshot caveat:** upstream snapshot is not fresh. `comma-lab status` reports snapshot commit `ec82c291ffeae5212e9a38253791d58995518a80` from 2026-04-03, while the live workspace upstream checkout is `cd64c68b740ffbe90c0132ca560a9cefc9d78ac5`; root `upstream/` is a separate dirty checkout at `11ad728f563d8970929e8947a1cf6124ee6303e4`. Rebootstrap deliberately before final reproduction.

**Fresh blocked/negative evidence:**
- `lane_m_v2_landed`: 1.84 [contest-CUDA], negative vs Lane G v3.
- `lane_h_crf56`: 3.20 [contest-CUDA], negative vs Lane G v3.
- `modal_auth_eval_8e331354a6b5`: inflate failed on pose shape `(600, 6)` vs expected `(N, 1)`.
- Recovered SZ phase2 archive is 3.3KB but fails canonical local smoke because no `masks.mkv` is present; current inflator would fall back to non-compliant scorer-time extraction.
- First-wave 2026-04-29 Modal lanes exposed setup bugs: MAE-V missing `pydantic`, Omega Hessian CUDA device-side assert, Uniward missing baseline anchor path.

**Decision:** next work should target deployment correctness before new architecture speculation. Highest EV order: Modal dependency install, remote anchor path contracts, Omega tiny Hessian smoke, SZ compliance decision, then upstream snapshot rebootstrap.

## 2026-04-21 [CRITICAL] Proxy-Auth PoseNet Drift — STE Roundtrip is a Leaky Abstraction

**Root cause (Tao + Karpathy + Council):** The renderer overfits to proxy-specific texture patterns that survive bilinear STE but NOT actual uint8 quantization via DALI. PoseNet proxy-auth ratio went from 2.1x (ep300) to 11.1x (ep3560).

**The fix (Hotz):** Add Gaussian noise (σ=0.5 pixel) after every roundtrip in training. One line: `x = x + torch.randn_like(x) * 0.5`. This makes the renderer robust to roundtrip perturbation. Standard adversarial training technique.

**The strategy (Council binding):**
1. Pose TTO on ep3560 (SegNet 0.00060 is spectacular — TTO fixes PoseNet)
2. Checkpoint sweep ep500-3000 to find auth sweet spot
3. Fix training with roundtrip noise for future runs
4. Do NOT ship 0.36 when 0.32 is one experiment away

**Projected: ep3560 + pose TTO could hit auth 0.32 if PoseNet drops to 0.002 (SegNet 0.060 + PoseNet 0.141 + Rate 0.122 = 0.323)**
