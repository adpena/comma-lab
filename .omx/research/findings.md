# Findings

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
   - Our v5 Lagrangian annealing achieved auth=0.87 baseline. Foundation for all TTO work.
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

**Context**: All prior Vast.ai experiments used wrong checkpoint (5-epoch smoke model, MD5:a9aee326). Re-ran with correct auth=0.87 renderer (MD5:cff8dca4).

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
- TTO v1 (auth=0.74): SegNet-only optimization. PoseNet "improvement" was noise.
- TTO v2: Same.
- TTO v3 (embedding loss): Embedding loss couldn't help because PoseNet had no gradient signal. The entire experiment was invalid.
- TTO v4 (running): Running with the same bug. Results will be invalid for PoseNet.
- Renderer training (auth=0.87): UNAFFECTED — training pipeline had the fix.

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
- Score: 2.93
- Variant: dp_sims

## 2026-04-12 [technique] Yousfi tricks 32-35: Pedernales insights — theoretical floor 0.18

Trick 32: backward delta generation (ONE perfect last frame + 1199 tiny deltas). Trick 33: exploit preprocess_input blind spots (YUV420 transform nullspace). Trick 34: overfit the SCORER not the video (steganographic capacity). Trick 35: archive as codebook (texture atoms + motion field + correction targets = 15KB total, rate 0.01). Theoretical floor: 0.18.

## 2026-04-12 [breakthrough] DP-SIMS SegNet 0.003 TIES Quantizr after only 89 Phase 2 epochs

FP4-quantized DP-SIMS achieves SegNet 0.003 — identical to Quantizr (0.60 score). Entire remaining gap is PoseNet (0.482 vs 0.001 = 480x) and rate (2.2MB vs 386KB = 5.7x). Path to sub-0.60: (1) train 1000+ P2 epochs for PoseNet, (2) shrink model to 500KB, (3) add cross-frame attention. Architecture validated.
- Score: 2.5
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
- Score: 0.135

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
