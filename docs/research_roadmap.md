# Research Roadmap — Task-Aware Neural Video Compression

> **2026-05-06 status.** This file is a historical roadmap and idea ledger.
> It is kept for community context and post-contest planning, not as a live
> priority list or score authority. Any item below is `prediction` or
> `empirical` until a charged archive passes exact CUDA auth eval with full
> custody. Current ranked rows belong in `docs/paper/04_results.md`.

## Council Eureka Moments (2026-04-12)

1. **Scoring formula sqrt asymmetry** — PoseNet has diminishing returns below pose=0.0005. Knee of the curve. Stop PoseNet optimization there, focus on SegNet + rate.
2. **Odd frames are SegNet-free** — 600 of 1200 frames invisible to SegNet. Optimize purely for PoseNet + compressibility. Make smooth/simple for better rate.
3. **Scorer resolution = 384x512** — BOTH scorers downscale to 384x512. Store at scorer res = 5.2x rate reduction. Verify round-trip fidelity.
4. **YUV420 chroma null space** — 294,912 free dimensions per frame (2x2 block perturbations summing to zero). Invisible to scorer. Use for H.265 compressibility.
5. **SegNet argmax stability** — Only boundary pixels (~2-5%) need detail. Interiors can be flat color. Hard-quantize interiors, spend bits only on boundaries.
6. **Unlimited compress time = minimal recipe** — Pre-compute everything. Store masks + PoseNet targets + seed + corrections. Total 10-60KB.
7. **DALI bypass** — Generated frames use TensorVideoDataset (raw load), bypassing DALI entirely. Pre-compute targets with DALI at compress time = eliminates 29x calibration.

## Sprint Plan

### PHASE 1: NOW (while training runs, parallel)
- CRF sweep 32-38 on existing 1.33 checkpoint (2-3h, local)
- Precomputed corrections pipeline (3-4h, local)
- Scorer resolution round-trip verification (3-4h, local)
- Writeup + visualizations (ongoing)
- Auth scorer on bat00 with DALI (1-2h)

### PHASE 2: After first training results (hours 5-24)
- Auth-eval new checkpoint
- CRF sweep on new checkpoint
- Stack tricks: CRF + corrections + TTO
- Begin constrained gen smoke on Kaggle P100

### PHASE 3: Week 2 (April 14-20, DECISIVE)
- GPU lane: constrained generation (PRIMARY on Kaggle P100)
- CPU lane: trick stacking (SECONDARY)
- Auth eval pipeline (MUST HAVE)

### PHASE 4: Week 3 (April 21-May 3, LOCK & POLISH)
- April 21: LOCK final approach
- Ablation studies for paper
- Writeup cleanup/finalization
- May 1-3: Final submission

### PHASE 5: Post-competition
- optional long-form writeup or paper, only after evidence-grade cleanup
- 3DGS / 4D-GS experiments
- Open-source tac library

---

## Historical Priority: Beat Quantizr (deadline May 3, 2026)

This section records the April 2026 contest plan. It is not a current claim
that these lanes remain optimal, dispatch-ready, or score-bearing.

### NOW — Asymmetric Warp Training (deployed on Modal T4)
- AsymmetricPairGenerator with Fridrich 3-phase curriculum
- Warp-based pair generation (frame_t = warp(frame_t1) + gate * residual)
- Dashboard: check Modal for live training logs

### NEXT — Highest-Leverage Stacking (post-validation)

#### 1. Ego-Motion Pre-Computation (2 days, PoseNet -0.3 to -0.5)
- **Tools:** Scorer's own PoseNet (no external VO needed — we extract the scorer's own targets)
- **Concept:** Compute exact 6-DOF camera trajectory at compress time via PoseNet forward pass on GT
- **Store:** 600 pairs x 6 floats x 2 bytes (float16) = 7,200 bytes raw, ~5KB compressed
- **Use:** Supervised TTO at inflate time minimizes MSE against known PoseNet targets
- **Effect:** PoseNet distortion driven toward 0 — we optimize the exact scorer metric
- **Papers:** TartanVO (Wang et al., CoRL 2021), DPVO (Teed et al., NeurIPS 2023)
- **Status:** IMPLEMENTED — infrastructure complete, awaiting authoritative eval
  - Extract: `src/tac/scorer_targets.py` (extract_posenet_targets, save/load)
  - Bundle: `src/tac/precompute_corrections.py` (included in corrections bundle)
  - Compress: `POSENET_TARGETS_ENABLE=1` in `submissions/robust_current/compress.sh`
  - Inflate: `--supervised-tto-steps N` in `submissions/robust_current/inflate_postfilter.py`
  - TTO: `src/tac/tto.py` (supervised_tto, posenet_target_loss)
  - Training: `src/tac/training.py` (gt_scorer_cache pre-computes at training time)
  - Tests: `src/tac/tests/test_scorer_targets.py` (6 tests, all passing)
  - Experiment: `experiments/precompute_ego_motion.py` (extraction + analysis)
  - Council verdict: Loss Target (Option A) over Conditioning (Option B)
- **Implementation steps:**
  1. Run `POSENET_TARGETS_ENABLE=1` compress on current best checkpoint
  2. Inflate with `--supervised-tto-steps 10` on bat00 with DALI scorer
  3. Auth-eval to measure PoseNet improvement vs baseline (no TTO)
- **Dependencies:** Auth scorer environment (bat00 or Lightning T4 with DALI)
- **Estimated effort:** 2h remaining (code done, just eval needed)
- **Risk:** Supervised TTO may overfit to DALI-decoded targets while inflate sees PyAV-decoded frames. The 29x DALI/PyAV divergence (finding 2026-04-11) could mean TTO targets are calibrated for the wrong decode path.
- **Success criteria:** PoseNet auth score drops by at least 30% (from 0.148 to <0.104 component)
- **Council approval status:** Approved (Yousfi + Fridrich consensus)

#### 2. Learned Entropy Coding on Weights (1 day, rate -0.1)
- **Tools:** CompressAI entropy modules, arithmetic coding
- **Concept:** Replace raw FP4 byte packing with learned probability model
- **Effect:** 20-30% archive size reduction at zero quality cost
- **Papers:** COIN/COIN++ (Dupont et al., NeurIPS 2021 / TMLR 2022)
- **Framework:** `pip install compressai` — has ready-made entropy models
- **Status:** NOT STARTED
- **Implementation steps:**
  1. Profile current archive composition: `python -c "import zipfile; z=zipfile.ZipFile('submissions/robust_current/archive.zip'); [print(f'{i.filename}: {i.compress_size}B') for i in z.infolist()]"`
  2. Implement weight histogram analysis in `src/tac/archive/entropy_archive.py` — compute per-layer weight distributions after FP4 quantization
  3. Add `StaticEntropyModel` class: fit a Laplace/Gaussian mixture to each layer's weight histogram, generate CDF tables, use `compressai.entropy_models.EntropyBottleneck` for coding
  4. Modify `src/tac/quantization_audit.py` to output entropy-coded blob instead of raw FP4 bytes
  5. Update `submissions/robust_current/inflate_postfilter.py` to decode entropy-coded weights (add static CDF tables as header in blob, ~200 bytes)
  6. Verify round-trip: original weights == decode(encode(weights)) to full precision
- **Dependencies:** None (standalone rate improvement)
- **Estimated effort:** 6-8 hours
- **Risk:** CompressAI adds ~50MB dependency. If archive is already near-optimal under zlib, the marginal gain from learned entropy coding may be <5%. Also: CDF tables themselves consume bytes in the archive.
- **Success criteria:** Archive size reduction of at least 15% (from ~46KB postfilter to <39KB) with zero quality degradation. Rate component drops from 0.575 to <0.55.
- **Council approval status:** Approved (Shannon assessment: "There is at least 20% redundancy in raw FP4 byte packing vs optimal coding")

#### 3. Depth Conditioning (1 day, PoseNet -0.05 to -0.2)
- **Tools:** ZoeDepth, MiDaS (both open-source, run in seconds on GPU)
- **Concept:** Pre-compute monocular depth at compress time, store as side info
- **Store:** uint8 depth per frame, delta-coded + entropy = ~100KB for 1200 frames
- **Use:** Add as extra input channel to renderer (geometric awareness)
- **Effect:** Renderer knows which pixels are close/far = better PoseNet preservation
- **Papers:** ZoeDepth (Bhat et al., 2023), PixelNeRF (Yu et al., CVPR 2021)
- **Status:** NOT STARTED
- **Implementation steps:**
  1. Add depth extraction script: `experiments/extract_depth_maps.py` using `torch.hub.load('isl-org/ZoeDepth', 'ZoeD_NK')`. Run on all 1200 GT frames at compress time.
  2. Compress depth maps: delta-code temporally (frame[t] - frame[t-1]), quantize to uint8, zlib compress. Target: <100KB for 1200 frames.
  3. Modify `src/tac/architectures.py`: add `depth_conditioning: bool` to renderer config. When enabled, first conv takes 4 channels (RGB + depth) instead of 3.
  4. Update `src/tac/data.py` to load depth maps alongside frames during training.
  5. Retrain renderer with depth channel enabled (profile: `proven_baseline` + `--depth-conditioning`).
  6. Update `submissions/robust_current/inflate_postfilter.py` to load depth from archive and pass to renderer.
- **Dependencies:** Item 1 (ego-motion) should be eval'd first to establish baseline. Requires retraining the renderer (not a drop-in trick).
- **Estimated effort:** 1-2 days (extraction: 2h, architecture change: 4h, retraining: 8-12h GPU)
- **Risk:** 100KB depth maps add 0.065 to rate component (25 * 100KB/38.4MB). The PoseNet improvement must exceed 0.065 to be net-positive. Depth from monocular estimation is noisy and may confuse the renderer rather than help it. Also requires full retraining cycle.
- **Success criteria:** Net score improvement > 0.05 after accounting for rate penalty. PoseNet component drops by at least 0.08.
- **Council approval status:** Deferred — Contrarian assessment: "100KB rate cost is steep. Only pursue if GPU lane (constrained gen) is dead and we are stuck on the CPU lane renderer approach."

#### 4. Occlusion Mask from Depth + Ego-Motion (0.5 days, both -0.05)
- **Concept:** Compute WHERE warping fails (new content visible due to ego-motion)
- **Store:** Binary mask, ~5KB compressed per video
- **Use:** Extra input channel to renderer = focus correction on disoccluded regions
- **Papers:** SynSin (Wiles et al., CVPR 2020), Worldsheet (Hu et al., ICCV 2021)
- **Status:** NOT STARTED — depends on #1 and #3
- **Implementation steps:**
  1. Compute forward flow from ego-motion (item 1) and depth (item 3): `flow = K @ (R @ K_inv @ depth_pixel + t)` where K is camera intrinsics (fx=910, pp=(582,437)), R,t from PoseNet targets.
  2. Identify disoccluded pixels: forward-warp frame[t] to frame[t+1] position, pixels with no source = disoccluded. Implement in `src/tac/precompute_corrections.py` as `compute_occlusion_masks()`.
  3. Compress masks: run-length encode + zlib. 1200 binary masks at 384x512 = ~5KB total (masks are sparse, <5% of pixels).
  4. Add as channel 5 to renderer input (after RGB + depth). Modify `src/tac/architectures.py` accordingly.
  5. Train renderer to allocate correction capacity to disoccluded regions.
- **Dependencies:** Items 1 (ego-motion) AND 3 (depth). Cannot start until both are complete.
- **Estimated effort:** 4 hours (computation is straightforward geometry, main work is integration)
- **Risk:** Error compounds: noisy depth + noisy ego-motion = noisy occlusion masks. The renderer may learn to ignore the channel. Also: requires retraining again.
- **Success criteria:** Visible improvement in disoccluded regions (qualitative) + measurable PoseNet improvement (>0.02 component reduction)
- **Council approval status:** Deferred — blocked by items 1 and 3. Low priority given GPU lane may eliminate renderer entirely.

#### 5. AWQ-Style Mixed-Precision Quantization (1 day, rate + quality)
- **Tools:** AWQ (Lin et al., MLSys 2024)
- **Concept:** Quantize each weight proportional to its influence on scorer loss
- **Effect:** Higher quality per byte — salient weights at FP8, others at FP4
- **Status:** NOT STARTED
- **Implementation steps:**
  1. Implement saliency scoring in `src/tac/quantization_audit.py`: for each weight tensor, compute `saliency = abs(weight * grad)` where grad is the scorer loss gradient w.r.t. weights. Average over a calibration set of 10 representative frames.
  2. Rank weights by saliency. Top 10% get FP8 (1 byte each), remaining 90% get FP4 (0.5 bytes). This is a 10% size increase over uniform FP4 for 2x precision on the most important weights.
  3. Implement mixed-precision packing in `src/tac/quantization_audit.py`: `pack_mixed_precision(weights, saliency_mask)` and `unpack_mixed_precision(blob)`.
  4. Update `submissions/robust_current/inflate_postfilter.py` to use mixed-precision unpacking.
  5. Sweep the FP8 fraction: {5%, 10%, 20%, 50%} to find Pareto-optimal rate-quality tradeoff.
- **Dependencies:** Scorer models must be loadable for gradient computation. Current FP4 quantization code in `src/tac/quantization_audit.py` is the starting point.
- **Estimated effort:** 6 hours
- **Risk:** Mixed-precision adds complexity to the archive format. If the saliency distribution is flat (all weights equally important), this degrades to uniform FP8 at 2x the size. Quantizr likely already uses something similar. The rate cost of extra precision may exceed the quality gain.
- **Success criteria:** Same or better quality at same archive size, OR same quality at 10-15% smaller archive. Net score improvement > 0.02.
- **Council approval status:** Approved (Fridrich: "This is activation-aware steganalysis in reverse. Sound principle.")

### MEDIUM-TERM — Post-Validation Architecture Improvements

#### 6. RAFT-Lite Correlation Volume for MotionPredictor (3 days)
- Compute all-pairs pixel similarity between frames at low resolution
- Iteratively refine flow with GRU-style update
- Much better motion estimation than our current conv stack
- **Paper:** RAFT (Teed & Deng, ECCV 2020)
- **Implementation sketch:**
  1. Add `src/tac/contrib/raft_lite.py`: implement a lightweight RAFT variant with 4x downsampled correlation volume (96x128 instead of 384x512) and 4 GRU iterations (vs RAFT's 12). Target: 50K params for the flow module.
  2. Modify `src/tac/architectures.py` `MotionPredictor` class: replace the current 3-layer conv stack with RAFT-lite correlation lookup + GRU refinement. Keep the same input/output interface.
  3. The correlation volume is computed once per frame pair: `corr = einsum('bchw,bcHW->bhwHW', fmap1, fmap2)` at 1/4 resolution. Lookup is O(1) per pixel.
  4. Train with `proven_baseline` profile, replacing only the motion module. Compare proxy scores at epoch 500 vs current conv-stack baseline.
  5. If flow quality improves, the renderer needs fewer parameters for residual correction = smaller archive.
- **Dependencies:** Current renderer architecture must be stable (no ongoing architecture changes). Item 1 (ego-motion) results inform whether better flow estimation actually matters for PoseNet.
- **Estimated effort:** 2-3 days (1 day implementation, 1-2 days training + eval)
- **Risk:** RAFT-lite adds parameters to the archive. If the flow module is 50K params at FP4, that is 25KB = rate cost of 0.016. The quality improvement must exceed this. Also: correlation volumes are memory-intensive; may not fit in T4 during training.
- **Success criteria:** Proxy score improvement > 0.05 over conv-stack baseline at same or lower archive size.
- **Council approval status:** Needs council decision — Karpathy supports ("flow is the bottleneck"), Contrarian skeptical ("more params, unproven gain")

#### 7. NeRV-Style Positional Encoding (0.5 days)
- Add sinusoidal encoding of frame index as input to renderer
- Per-frame awareness without per-frame parameters
- **Paper:** NeRV (Chen et al., NeurIPS 2021), E-NeRV (Li et al., ECCV 2022)
- **Implementation sketch:**
  1. Add positional encoding function to `src/tac/architectures.py`: `pos_enc(t, L=10)` returns `[sin(2^0 * pi * t/T), cos(2^0 * pi * t/T), ..., sin(2^L * pi * t/T), cos(2^L * pi * t/T)]` where T=1200.
  2. Concatenate 2L=20 positional features to the renderer's input (alongside frame pixels). Modify `DilatedRenderer.forward()` to accept frame index and concatenate encoding.
  3. This adds 20 input channels to the first conv layer = ~720 extra parameters (20 * 36) at FP4 = 360 bytes. Negligible rate cost.
  4. Train with positional encoding enabled. The renderer can now learn frame-specific patterns (e.g., "frame 500 has a highway interchange") without per-frame weights.
- **Dependencies:** None. Can be tested independently.
- **Estimated effort:** 4 hours (trivial implementation, main cost is retraining)
- **Risk:** Positional encoding may cause the renderer to memorize training-set frame indices rather than learn generalizable patterns. With only 1200 frames, overfitting is likely. May need to combine with data augmentation (random frame index jitter).
- **Success criteria:** Proxy score improvement > 0.02 at same archive size. No PoseNet regression.
- **Council approval status:** Approved (low cost, easy to test and revert)

#### 8. RIFE Frame Interpolation for Backward-Delta (1 day)
- Store only keyframes, interpolate intermediates
- Reduces number of frames that need full rendering
- **Paper:** RIFE (Huang et al., ECCV 2022)
- **Implementation sketch:**
  1. At compress time: select keyframes (every Nth frame, N=4-8). Run full renderer on keyframes only. Interpolate intermediate frames using RIFE.
  2. Store keyframe outputs + RIFE model weights in archive. RIFE-lite (~200KB at FP4) may be too expensive for our rate budget.
  3. Alternative: use RIFE only at inflate time with no stored weights — download RIFE from a URL or include in the inflate environment. Check contest rules on allowed network access at inflate time.
  4. Implement in `src/tac/contrib/rife_interpolation.py`. Integrate with `src/tac/trick_stack.py` as step 4.5 (between multi-pass and rounding).
- **Dependencies:** Contest rules clarification on allowed inflate-time resources. Current inflate environment may not have RIFE weights.
- **Estimated effort:** 1 day
- **Risk:** RIFE weights are ~10MB at FP16, far too large for our archive. The interpolated frames may have temporal artifacts that hurt PoseNet (PoseNet evaluates consecutive pairs — interpolation errors compound). RIFE is trained on natural video, not our generated frames.
- **Success criteria:** Keyframe-only storage at N=4 reduces archive by 75% while maintaining score within 5% of full-frame storage.
- **Council approval status:** Deferred — rate budget arithmetic does not favor this for the CPU lane (we already ship a video file, not per-frame data). Only relevant if GPU lane (constrained gen) becomes the primary approach.

#### 9. Scale-Space Warping (2 days)
- Instead of single-scale optical flow, use multi-scale warp
- Decoder selects optimal blur/sharpness per pixel
- **Paper:** Scale Space Flow / SSF (Agustsson et al., CVPR 2020)
- **Implementation sketch:**
  1. Build a 3-level Gaussian pyramid of each frame in `src/tac/contrib/scale_space_warp.py`: levels at 1x, 0.5x, 0.25x resolution.
  2. Renderer predicts per-pixel scale selection (softmax over 3 levels) + flow at each scale. Total output: 3 + 2*3 = 9 channels instead of current 3 (RGB) + 2 (flow).
  3. Warp each pyramid level independently, blend using predicted soft weights. This lets the renderer choose "blurry background, sharp foreground" per-pixel.
  4. Modify `src/tac/architectures.py` to output 9 channels. Decoder head becomes 9-channel conv instead of 5-channel.
  5. Train and compare: scale-space warping may help PoseNet (it sees warped pairs) by providing more accurate flow in low-texture regions.
- **Dependencies:** Item 6 (RAFT-lite) would pair well with this — RAFT provides multi-scale features already.
- **Estimated effort:** 2 days (1 day implementation, 1 day training)
- **Risk:** 9-channel output means ~3x more decoder parameters. Rate cost may dominate. Also: scale-space is most useful for large motions, but our 20fps dashcam has small inter-frame motion.
- **Success criteria:** PoseNet improvement > 0.03 component reduction. Archive size increase < 20%.
- **Council approval status:** Needs council decision — interesting theoretically but high implementation cost relative to expected gain.

#### 10. Knowledge Distillation (3 days)
- Train large renderer (2M params, depth=2) for max quality
- Distill into small renderer (287K params) for deployment rate
- Teacher-student with MSE on teacher outputs + scorer loss
- Standard technique in model compression
- **Implementation sketch:**
  1. Train teacher model through the canonical pipeline with a teacher-capacity profile, e.g. `python experiments/pipeline.py compress --profile proven_baseline_teacher --device cuda --output-dir results/proven_teacher`. Train for 5000 epochs on a proxy GPU only when the run is tagged `score_claim=false`.
  2. Add distillation loss to `src/tac/training.py`: `loss_distill = MSE(student_output, teacher_output.detach())`. Weight: `distill_weight=1.0` alongside existing scorer losses.
  3. Train student through the canonical pipeline with an explicit distillation profile and teacher checkpoint manifest. Student is standard 287K params.
  4. The student learns to approximate the teacher's output distribution, which is higher quality than what the student discovers on its own.
  5. Eval both teacher (for quality ceiling) and student (for deployable quality).
- **Dependencies:** GPU hours: teacher training ~12h on P100, student distillation ~8h. Total ~20h GPU time = ~1 Kaggle weekly allocation.
- **Estimated effort:** 3 days (1 day each for teacher training, distillation, eval/iteration)
- **Risk:** Teacher may not be significantly better than student (our 287K model may already be near the capacity-quality frontier for this task). Distillation assumes teacher quality > student quality, which is unverified. Also: 20h GPU time is expensive.
- **Success criteria:** Distilled student scores at least 0.05 better than student trained from scratch, at same archive size.
- **Council approval status:** Needs council decision — Tao supports ("distillation is universal"), Contrarian: "Prove teacher is better first. Train teacher for 500 epochs, eval, then decide."

#### 11. Batch Size Scaling (1 day, requires council approval for larger GPU)
- Current: batch_size=4 on T4 (800MB VRAM usage, 20x headroom)
- Scaling to batch_size=16 or 32 reduces gradient noise, may speed convergence
- T4 can handle batch_size=16 easily (~3.2GB VRAM)
- Larger batches may need A10G (24GB) — requires human approval per GPU budget rule
- Also: gradient accumulation as a free alternative to larger batches
- **Implementation sketch:**
  1. Modify `src/tac/profiles.py`: add `batch_size` to profile configs. Current `proven_baseline` uses batch_size=4.
  2. Test batch_size=16 with a provider-neutral profile override through `experiments/pipeline.py` or `tac.deploy.deploy_config`; monitor VRAM with `nvidia-smi`. MPS is proxy-only and never auth eval.
  3. If T4 OOMs at 16, try gradient accumulation: `--accumulate-grad-batches 4` (effective batch=16 at batch=4 memory cost). Add to `src/tac/training.py` training loop.
  4. Compare convergence curves: proxy score at epoch 500 for batch=4 vs batch=16.
- **Dependencies:** Lightning T4 access (free tier). No code changes needed beyond config.
- **Estimated effort:** 4 hours (config + monitoring + eval)
- **Risk:** Larger batch size may hurt generalization (sharp minima). Linear scaling rule for LR (batch 4x = LR 4x) may not hold for our scorer-loss landscape.
- **Success criteria:** Faster convergence (same proxy score reached in fewer epochs) or better final proxy score at same epoch count.
- **Council approval status:** Approved for T4 batch=16. A10G requires human approval.
- **Status:** NOT STARTED — try batch_size=16 on T4 first

#### 12. Multi-Model Ensemble (2-3 days)
- Train multiple renderers with different seeds/architectures
- Average outputs at inflate time (reduces variance, smooths artifacts)
- Archive contains multiple models — rate cost vs quality gain tradeoff
- Ensemble of 3 models at 140KB each = 420KB archive, rate = 0.011, term = 0.28
- Only viable if quality gain from ensemble > 0.28 rate cost
- **Implementation sketch:**
  1. Train 3 renderers with different random seeds using `proven_baseline` profile: `--seed 42`, `--seed 123`, `--seed 7`.
  2. At inflate time: load all 3, generate frames from each, pixel-wise average. Implement in `src/tac/experiments/ensemble.py` (already has `EnsembleRenderer` class).
  3. Measure: (a) individual model proxy scores, (b) ensemble proxy score, (c) ensemble archive size.
  4. If ensemble improvement > 0.28 (rate penalty), promote. Otherwise, identify the single best model.
- **Dependencies:** 3x GPU training time. Items 1-5 should be settled first (no point ensembling a suboptimal architecture).
- **Estimated effort:** 2-3 days (mostly training time, minimal code)
- **Risk:** Rate penalty of 0.28 is very steep. Ensemble averaging may blur high-frequency details that SegNet needs. Also: 3 forward passes at inflate time = 3x inflate latency.
- **Success criteria:** Ensemble score < best single model score - 0.28 (must overcome rate penalty).
- **Council approval status:** Deferred — "Prove single model is optimized first. Ensemble is a last resort, not a first strategy." (Contrarian)
- **Status:** NOT STARTED — validate single model first

#### 13. Architecture Scaling — Channel Width Sweep (2 days)
- Current: base_ch=36, mid_ch=60 (Quantizr's values, 287K params)
- Sweep: (24,40), (36,60), (48,80), (64,128) at fixed depth=1
- Each config: 500 epoch training = eval = Pareto frontier
- Identifies optimal capacity allocation per rate budget
- Quantizr's "slightly different architecture gets 10% better" may be this
- **Implementation sketch:**
  1. Define sweep configs in `configs/channel_sweep.yaml`:
     ```
     sweep:
       - {base_ch: 24, mid_ch: 40, expected_params: ~90K, expected_archive: ~45KB}
       - {base_ch: 36, mid_ch: 60, expected_params: ~287K, expected_archive: ~140KB}
       - {base_ch: 48, mid_ch: 80, expected_params: ~500K, expected_archive: ~250KB}
       - {base_ch: 64, mid_ch: 128, expected_params: ~1.1M, expected_archive: ~550KB}
     ```
  2. Launch sweep on Kaggle P100: 4 configs x 500 epochs x ~2min/epoch = ~67 hours. Run 2 at a time across 2 Kaggle sessions.
  3. Plot Pareto frontier: x-axis = rate (archive size), y-axis = quality (seg + pose components). The knee of this curve is the optimal architecture.
  4. Add results to `reports/channel_sweep_pareto.json` for the writeup.
- **Dependencies:** Kaggle P100 hours (30h/week free). Requires 2 weeks of Kaggle allocation.
- **Estimated effort:** 2 days (setup + monitoring, training runs itself)
- **Risk:** Diminishing returns: the 287K model may already be near-optimal for our task. The sweep consumes significant GPU time that could go to higher-priority experiments.
- **Success criteria:** Find a configuration that scores at least 5% better than (36,60) on the rate-quality Pareto frontier.
- **Council approval status:** Deferred until after GPU lane decision (April 17). If GPU lane (constrained gen) wins, architecture scaling is irrelevant.
- **Status:** NOT STARTED

#### 14. CRF Sweep for Mask Encoding (1 day, CPU lane)
- If we ever encode masks as video (Quantizr's approach: 209KB AV1):
  - Sweep CRF 15-30 for mask video quality
  - Verify mask round-trip (encode=decode=argmax matches original)
  - Optimize mask video rate vs mask fidelity tradeoff
- Currently N/A (we extract masks at inflate time, no mask video in archive)
- **Implementation sketch:**
  1. Implement mask-to-video encoding in `src/tac/mask_codec.py` (extends existing `MaskCodec`): render each mask as a flat-colored image (class index = pixel value), encode as AV1 video.
  2. Sweep CRF 15-30: `for crf in range(15,31): encode_masks(masks, crf=crf); decoded = decode_masks(video); accuracy = (decoded.argmax == original.argmax).mean()`
  3. Find minimum CRF where argmax accuracy is 100% (lossless at the class level). This is the sweet spot.
  4. Compare archive size: current approach (inflate-time mask extraction, 0 bytes) vs mask video (~100-200KB).
- **Dependencies:** AV1 encoder (libaom or SVT-AV1 via ffmpeg). Only relevant if we switch to storing masks.
- **Estimated effort:** 4 hours
- **Risk:** Any CRF that achieves 100% argmax accuracy will have a specific size. If that size > 100KB, the rate penalty (25 * 100KB/38.4MB = 0.065) likely exceeds any quality gain from having explicit masks.
- **Success criteria:** Mask video at <50KB with 100% argmax round-trip accuracy.
- **Council approval status:** Deferred — only relevant if GPU lane uses mask-conditioned generation. NOT STARTED.

### LONG-TERM — Post-Competition Research Directions

#### 15. 3D Gaussian Splatting as Video Codec
- **Concept:** Fit 3D Gaussians to driving scene, store Gaussian params, render dashcam view
- **Challenge:** Monocular reconstruction quality, archive size (2-5MB), training time
- **Papers:** Street Gaussians (Yan et al., ECCV 2024), 4D-GS (Wu et al., CVPR 2024)
- **Timeline:** 1-2 months research project
- **Potential:** If monocular reconstruction improves, this could be the ultimate video codec
- **What this involves:** Use structure-from-motion (COLMAP or PoseNet ego-motion) to estimate camera trajectory. Initialize 3D Gaussians on a point cloud from depth estimation. Optimize Gaussian parameters (position, covariance, color, opacity) to minimize rendering loss against GT frames. Store optimized Gaussians as the archive. At inflate time, render from stored Gaussians using the known camera trajectory.
- **Key challenge for competition:** Archive size. A typical 3DGS scene requires 50K-200K Gaussians. At ~60 bytes/Gaussian (xyz, scale, rotation, SH coefficients), that is 3-12MB before compression. Current rate budget is ~40KB. Would need extreme pruning or a fundamentally different parameterization.
- **Research value:** High. Connects video compression to 3D scene understanding. Publishable regardless of competition outcome.

#### 16. 4D Gaussian Splatting (3D + Time)
- Deformable Gaussians that move/rotate over time
- Separate static background from dynamic objects
- **Papers:** Deformable 3D Gaussians (Yang et al., CVPR 2024), SC-GS (Huang et al., CVPR 2024)
- **Application:** Driving scene with moving cars/pedestrians
- **What this involves:** Extend 3DGS with per-Gaussian deformation fields: `G(t) = G_0 + delta(t)` where delta is a small MLP or polynomial per Gaussian. Static Gaussians (road, buildings) have delta=0. Dynamic Gaussians (cars, pedestrians) have learned motion. This decomposition is natural for driving scenes.
- **Key question:** Can a monocular dashcam provide enough parallax for 3D reconstruction? Multi-view 3DGS works well, but monocular is fundamentally ambiguous. Depth priors (ZoeDepth) and ego-motion (PoseNet) may provide sufficient constraints.

#### 17. Neural Radiance Fields for Video (NeRF-as-Codec)
- D-NeRF, HyperNeRF for dynamic scenes
- Model weights ARE the compressed video
- **Challenge:** Training time, monocular input, quality on driving scenes
- **What this involves:** Represent the entire 60-second video as a single NeRF conditioned on time. The MLP weights encode all appearance and geometry. At inflate time, volume-render from the stored camera trajectory. This is conceptually identical to our current renderer approach but with implicit 3D structure.
- **Why not now:** NeRF rendering requires ~100 forward passes per pixel (ray marching). At 384x512 resolution, that is ~20M forward passes per frame x 1200 frames = 24B total. Even on A100, this takes hours. The 10-minute inflate budget is prohibitive.

#### 18. Video Coding for Machines (VCM)
- MPEG standard in progress for machine-consumption video
- Formalized rate-accuracy tradeoff framework
- Our Fridrich approach is a concrete instantiation of VCM principles
- **Papers:** VCM ad-hoc group publications (Duan, Sun, et al.)
- **What this involves:** VCM defines a formal framework where video is compressed for machine consumption (object detection, segmentation, etc.) rather than human viewing. The rate-distortion tradeoff is measured against task accuracy rather than PSNR. Our competition IS a VCM problem: minimize scorer distortion (task accuracy) subject to rate constraint. A VCM paper framing our approach would be highly publishable at MPEG/ACM MM.
- **Paper opportunity:** Frame our entire approach as "task-aware VCM for autonomous driving perception" — positions the work within an active MPEG standardization effort.

#### 19. Ego-Motion Decomposition
- Separate global camera motion from local object motion
- Global: single 6-DOF transform (ego-motion)
- Local: per-object bounding box + motion vector
- **Paper:** MCNet (Villegas et al., ICML 2017)
- Improves both rate (less to encode) and quality (each motion type optimized separately)
- **What this involves:** At compress time, estimate ego-motion (6-DOF camera transform between frames) and subtract it. The residual motion is purely local (moving cars, pedestrians). Encode global motion as 6 floats/frame (tiny), encode local motion as per-object bounding boxes + vectors. At inflate time, reconstruct: warp by ego-motion, add local motion, render residuals.
- **Connection to item 1:** Ego-motion pre-computation (item 1) already extracts the camera trajectory. This item extends it to explicitly decompose and separately encode the two motion types.

#### 20. CompressAI Integration (post-competition)
- **Framework:** CompressAI (InterDigital), `pip install compressai`
- Production-grade hyperprior, autoregressive, channel-conditional entropy models
- Replace our hand-rolled arithmetic coder for weight blob compression
- GPU-accelerated, batched entropy coding, pre-trained models available
- **Timeline:** Post-competition (2-3 days)
- **What this involves:** Replace `src/tac/lossless/arithmetic.py` with CompressAI's `EntropyBottleneck` or `GaussianConditional` modules. These use learned prior distributions that adapt to the weight statistics, achieving near-optimal coding. Also enables end-to-end training of the entropy model jointly with the renderer (the holy grail of learned compression). Would require modifying `src/tac/training.py` to include rate loss from the entropy model during training.
- **Why post-competition:** CompressAI adds significant complexity and a large dependency. The entropy coding gain (~10-20% rate reduction) is small relative to other levers. Better to focus on scorer optimization during the competition.

#### 21. Neural Per-Symbol Entropy Model (post-competition research)
- Autoregressive neural model predicting each weight's probability given context
- Current `NeuralEntropyModel` in `entropy_archive.py` exists but O(N) forward passes = too slow
- Needs: batch context, cached CDF tables, or offline-trained static model
- Additional 5-10% beyond static coding (~7KB on 140KB). Low competition value, high research value.
- **Papers:** COIN++ (Dupont et al., TMLR 2022), Balle et al. "Variational Image Compression"
- **What this involves:** Train a small autoregressive network that predicts P(w_i | w_{i-1}, ..., w_{i-k}) for weight w_i given k previous weights. Use predicted probabilities as CDF for arithmetic coding. The key challenge is speed: autoregressive decoding is sequential. Solutions: (a) cache CDF tables for common contexts, (b) use a parallel masked model (like PixelCNN), (c) offline-train and store static CDFs. Research value: connects neural compression (COIN++) with weight quantization.

---

## MISSING: Items from Eureka Moments, Council Sprint, and Cross-Domain Research

### 22. Constrained Frame Generation from Noise (GPU Eureka #1-4)
- **Source:** Eureka moments 2026-04-12 (findings.md)
- **Concept:** No renderer needed. Start from seeded noise, run gradient descent against scorer constraints. Archive = masks (239B) + PoseNet targets (7KB) + seed (64B) = 8KB total.
- **Projected score:** 0.135 (theoretical floor)
- **Implementation:**
  - Core module: `src/tac/constrained_gen.py` (ConstrainedFrameGenerator, ALREADY IMPLEMENTED)
  - Experiment scripts: `experiments/archive/exp1_fridrich_proper.py`, `experiments/archive/exp2_tiny_dp_sims_proper.py`
  - Inflate: constrained_generate() runs ~1000 gradient steps, ~50ms/step on T4 = 50 seconds
  - Archive builder: `src/tac/constrained_gen.py` has `build_minimal_archive()` and `load_minimal_archive()`
- **Current status:** Smoke test showed seg=0.025, pose=0.078 on 100 frames. DP-SIMS variant reached SegNet 0.003 (ties Quantizr) after 89 Phase 2 epochs. PoseNet gap remains 480x. Needs 1000+ Phase 2 epochs.
- **Dependencies:** Kaggle P100 or Lightning T4 for GPU training/optimization
- **Estimated effort:** 1-2 weeks of GPU time (1000+ epochs Phase 2)
- **Risk:** PoseNet convergence may plateau. The 480x gap may be structural (constrained gen produces frames that look right to SegNet but wrong to PoseNet). Also: 50-second inflate time is tight; any regression in per-step speed could bust the 10-minute budget.
- **Success criteria:** Full 1200-frame run with score < 0.60 (ties Quantizr). SegNet < 0.003, PoseNet < 0.005, rate < 0.002.
- **Council approval status:** Approved as PRIMARY GPU lane experiment. Decision date: April 17.

### 23. Scorer Resolution Round-Trip (Eureka #3)
- **Source:** Eureka moment #3: both scorers downscale to 384x512
- **Concept:** Store frames at 384x512 instead of 874x1164. 5.2x pixel reduction = massive rate savings.
- **Implementation steps:**
  1. Create `experiments/scorer_resolution_roundtrip.py`:
     - Load GT frames at 874x1164
     - Downscale to 384x512 (bilinear, matching scorer's `F.interpolate`)
     - Run both scorers on downscaled frames
     - Compare scores: GT-at-full-res vs GT-at-scorer-res
  2. If scores match (within 1%): modify `submissions/robust_current/compress.sh` to downscale before H.265 encoding. At inflate time, upscale back to 874x1164 (or leave at 384x512 if scorer accepts it directly).
  3. Verify the scorer's input pipeline: does it downscale from whatever resolution it receives, or does it expect 874x1164? Check `upstream/score.py` preprocessing.
- **Dependencies:** None. Pure CPU experiment.
- **Estimated effort:** 3-4 hours
- **Risk:** The scorer may normalize based on input resolution, producing different results at 384x512 vs 874x1164 input. Bilinear downscale + upscale round-trip introduces aliasing that may hurt PoseNet. Also: the upstream video loader may reject non-standard resolutions.
- **Success criteria:** Full scorer pipeline runs on 384x512 input with <1% score difference from 874x1164 input.
- **Council approval status:** Approved for investigation (Shannon: "If the scorer downscales, we should never store the information it throws away")

### 24. YUV420 Chroma Null-Space Exploitation (Eureka #4)
- **Source:** Eureka moment #4: 294,912 free dimensions per frame
- **Concept:** In YUV420, each 2x2 luma block shares one chroma sample. Perturbations to chroma that average to zero within a 2x2 block are invisible after YUV420 subsampling.
- **Implementation steps:**
  1. The core logic exists in `src/tac/scorer_exploits.py` (`analyze_preprocess_nullspace`, `exploit_preprocessing_nullspace`).
  2. Create `experiments/yuv420_nullspace_exploit.py`: load inflated frames, compute null-space perturbations that reduce H.265 compressibility (smoother chroma = fewer bits), apply perturbations, re-encode, eval.
  3. Integrate into `src/tac/trick_stack.py` as step 6 (chroma channel exploitation, currently marked UNTESTED).
  4. Key detail: the perturbation must be computed BEFORE H.265 encoding (at compress time), because H.265 operates on YUV420.
- **Dependencies:** None. Pure CPU experiment.
- **Estimated effort:** 4-6 hours
- **Risk:** The null-space analysis assumes a specific YUV420 conversion matrix. Different H.265 encoders may use slightly different conversion coefficients, breaking the "invisible" guarantee. Also: the perturbation must survive H.265 lossy compression itself.
- **Success criteria:** Rate reduction > 0.02 with zero SegNet/PoseNet degradation. Verified by auth eval.
- **Council approval status:** Approved (Fridrich: "This is steganographic embedding 101 — exploit the cover medium's redundancy")

### 25. Odd-Frame Simplification (Eureka #2)
- **Source:** Eureka moment #2: odd frames are invisible to SegNet
- **Concept:** SegNet evaluates only even-indexed frames. Odd frames can be optimized purely for PoseNet + rate.
- **Implementation steps:**
  1. Verify claim: check `upstream/score.py` to confirm SegNet processes only even frames. Search for frame subsampling logic.
  2. Implement in `src/tac/trick_stack.py` or `src/tac/constrained_gen.py`: for odd frames, skip SegNet constraint entirely. Use only PoseNet loss + TV regularization (compressibility).
  3. At compress time: generate odd frames as smooth gradients matching PoseNet targets. These compress much better under H.265 (uniform regions = few bits).
  4. Measure: rate reduction from simpler odd frames, PoseNet impact (should be neutral), SegNet impact (should be zero).
- **Dependencies:** Verification of the even-frame-only SegNet claim.
- **Estimated effort:** 4 hours
- **Risk:** If the claim is wrong (SegNet evaluates all frames), simplifying odd frames could catastrophically hurt SegNet score. MUST verify first.
- **Success criteria:** >10% rate reduction from simpler odd frames with zero SegNet degradation.
- **Council approval status:** Approved pending verification of the even-frame-only claim.

### 26. Backward Delta Generation (Yousfi Trick #32)
- **Source:** Findings 2026-04-12 — Yousfi tricks 32-35
- **Concept:** Generate ONE perfect last frame, then propagate backward with tiny deltas. 1199 deltas are much smaller than 1200 independent frames.
- **Implementation steps:**
  1. Add `src/tac/contrib/backward_delta.py`: implement backward frame propagation using optical flow.
  2. At compress time: render frame 1199 at full quality. For frame 1198: compute flow from 1199 to 1198, store only the residual after warping. Repeat backward to frame 0.
  3. Store: one full frame (at scorer res 384x512 = ~590KB uncompressed, ~50KB JPEG) + 1199 residual maps (delta-coded, ~0.5KB each after quantization = ~600KB). Total ~650KB.
  4. At inflate time: decode last frame, apply deltas backward.
  5. The flow computation uses ego-motion (item 1) for efficiency.
- **Dependencies:** Items 1 (ego-motion) and optionally 22 (constrained gen for the last frame).
- **Estimated effort:** 1-2 days
- **Risk:** Error accumulates backward through 1199 frames. Even tiny per-frame errors compound to visible drift by frame 0. May need periodic keyframes to reset error.
- **Success criteria:** Total archive < 700KB with score comparable to current 893KB archive.
- **Council approval status:** Needs council decision — Tao: "Error accumulation is the fundamental challenge. Need rigorous analysis of per-frame error bounds."

### 27. Scorer Distillation into Decoder Heads (Yousfi Trick from findings)
- **Source:** `src/tac/archive/scorer_distill.py` (already implemented)
- **Concept:** Train small auxiliary heads on the renderer that approximate PoseNet/SegNet outputs. At inflate time, use these heads for self-evaluation without loading the full scorers.
- **Implementation steps:**
  1. Module exists: `src/tac/archive/scorer_distill.py`. Verify it integrates with current training pipeline.
  2. Add distill heads to renderer architecture: 2 small MLPs branching from the renderer's bottleneck. PoseNet head: bottleneck -> FC(128) -> FC(6). SegNet head: bottleneck -> conv(5) -> upsample.
  3. Train jointly: standard loss + `distill_weight * (MSE(pose_head, posenet_target) + CE(seg_head, segnet_target))`.
  4. At inflate time: use distill heads for TTO (self-supervised refinement) instead of running full PoseNet/SegNet = much faster.
- **Dependencies:** Scorer targets must be pre-computed (item 1). Architecture change requires retraining.
- **Estimated effort:** 1 day
- **Risk:** Distill heads may not accurately approximate scorer behavior, leading TTO astray. The heads add ~10K params = ~5KB to archive.
- **Success criteria:** TTO with distill heads achieves >80% of the improvement of TTO with full scorers, at 10x speed.
- **Council approval status:** Approved as a refinement after items 1-5 are settled.

### 28. Temporal Delta Compression (Yousfi Trick #15)
- **Source:** `src/tac/temporal_delta.py` (already implemented)
- **Concept:** Store temporal deltas between consecutive frames instead of full frames. Deltas are sparse and compressible.
- **Implementation steps:**
  1. Module exists: `src/tac/temporal_delta.py`. Verify integration with archive builder.
  2. At compress time: compute `delta[t] = frame[t] - frame[t-1]`. Quantize deltas to int8 (-128 to 127). Apply threshold: set abs(delta) < threshold to 0 (sparsity). Entropy-code sparse delta maps.
  3. At inflate time: reconstruct frame[t] = frame[t-1] + delta[t]. Starting from frame[0] (stored in full).
  4. Sweep threshold: {1, 2, 4, 8, 16} to find optimal sparsity-quality tradeoff.
- **Dependencies:** None. Works with any frame source.
- **Estimated effort:** 4 hours (integration + sweep)
- **Risk:** Quantization of deltas introduces cumulative error. Threshold too high = quality loss, too low = no rate savings. Also: int8 range may clip large motions.
- **Success criteria:** >20% archive size reduction at <1% score degradation.
- **Council approval status:** Approved for investigation.

### 29. Cross-Frame Attention (Yousfi Trick #19)
- **Source:** `src/tac/cross_frame_attention.py` (already implemented)
- **Concept:** Allow the renderer to attend to features from adjacent frames, enabling temporal coherence without explicit flow.
- **Implementation steps:**
  1. Module exists: `src/tac/cross_frame_attention.py`. Contains `CrossFrameAttention` module.
  2. Insert attention layers between renderer blocks in `src/tac/architectures.py`. Each layer attends to features from frames t-1 and t+1.
  3. Parameter cost: attention with d=36 channels, 2 heads = ~5K params per layer. 2 layers = ~10K params = ~5KB FP4.
  4. Train with cross-frame attention enabled. The renderer processes frame triplets instead of individual frames.
- **Dependencies:** Architecture change requires retraining. Not compatible with current frame-independent inference.
- **Estimated effort:** 1 day (integration) + training time
- **Risk:** Cross-frame attention makes inference sequential (frame t depends on frame t-1 features). This may 3x inflate time. Also: 10K extra params may not justify their rate cost.
- **Success criteria:** PoseNet improvement > 0.03 component reduction (temporal coherence helps pair-wise evaluation).
- **Council approval status:** Deferred — only pursue if single-frame renderer hits a quality wall.

### 30. Semantic Quantization (Yousfi Trick — SegNet interiors)
- **Source:** Eureka #5 + `src/tac/semantic_quantization.py` (already implemented)
- **Concept:** Hard-quantize SegNet interior pixels to flat class-mean colors. Spend bits only on boundary pixels (~2-5% of image).
- **Implementation steps:**
  1. Module exists: `src/tac/semantic_quantization.py`. Contains boundary detection and interior flattening.
  2. Integrate into `src/tac/trick_stack.py` as a pre-processing step before H.265 encoding.
  3. At compress time: compute SegNet masks, identify boundary pixels (morphological gradient), replace interior pixels with class-mean color, preserve boundary pixels at full fidelity.
  4. This produces much simpler frames that compress dramatically better under H.265 (large uniform regions).
  5. Verify SegNet argmax is preserved after flattening (it should be by construction — we USE the argmax to choose the flat color).
- **Dependencies:** SegNet mask extraction at compress time. Already available via `src/tac/mask_generation.py`.
- **Estimated effort:** 4 hours (integration + verification)
- **Risk:** PoseNet may be affected by the flattened interiors (loss of texture = loss of feature points for ego-motion estimation). Need to measure PoseNet impact carefully.
- **Success criteria:** >30% rate reduction from simpler frames with zero SegNet degradation and <5% PoseNet degradation.
- **Council approval status:** Approved (Yousfi + Fridrich consensus: "This is the single highest-leverage CPU trick.")

### 31. Multi-Pass Inference Refinement (Yousfi Trick #16)
- **Source:** Already in `src/tac/trick_stack.py` step 4, verified working
- **Concept:** Run the renderer forward pass N times, applying uint8 rounding between passes. Each pass corrects quantization errors from the previous pass.
- **Current status:** IMPLEMENTED and verified. Activate with `INFLATE_MULTI_PASS=N` environment variable.
- **Implementation:** `src/tac/trick_stack.py` handles multi-pass loop. `submissions/robust_current/inflate_postfilter.py` reads the env var.
- **Remaining work:** Auth-eval to measure actual improvement. Sweep N in {2, 3, 5} to find optimal pass count.
- **Estimated effort:** 1 hour (just eval)
- **Risk:** Each pass adds ~30 seconds inflate time. N=5 = 2.5 minutes, eating into the 10-minute budget. Diminishing returns after N=2-3.
- **Success criteria:** Measurable proxy improvement with N=2 or N=3.
- **Council approval status:** Approved. DEPLOY NOW.

---

## Key Principles

1. **Unlimited compute at compress time, constrained at inflate time.** Pre-compute everything.
2. **The architecture IS the score.** Pair-wise warp generation > loss engineering.
3. **Task-aware > perception-aware.** Optimize for scorer, not human eyes.
4. **Decode on target.** No precomputed frames cross environment boundaries.
5. **Council decides design.** Skunkworks team reviews everything before GPU time.
6. **Rate is the biggest lever.** Gap analysis: rate 0.374 > seg 0.310 > pose 0.048. Attack rate first.
7. **Shannon bound governs ambition.** Theoretical minimum rate ~0.125 (scorer sufficient statistics). Quantizr at 0.201 is near-optimal. We must fundamentally change what we store (not how we compress).

## Reference Papers (by relevance)

### Must-Read (directly applicable)
- TartanVO (Wang et al., CoRL 2021) — visual odometry for ego-motion
- RAFT (Teed & Deng, ECCV 2020) — optical flow
- CompressAI framework — learned entropy coding
- AWQ (Lin et al., MLSys 2024) — activation-aware weight quantization
- COIN/COIN++ (Dupont et al.) — neural codec as weight compression
- NeRV/E-NeRV (Chen et al.) — neural video representation
- S-UNIWARD (Holub et al., EURASIP 2014) — distortion cost for steganographic embedding

### Should-Read (architectural insights)
- SynSin (Wiles et al., CVPR 2020) — novel view synthesis with occlusion
- ZoeDepth (Bhat et al., 2023) — monocular depth estimation
- RIFE (Huang et al., ECCV 2022) — frame interpolation
- PWC-Net (Sun et al., CVPR 2018) — coarse-to-fine optical flow
- MCNet (Villegas et al., ICML 2017) — motion decomposition
- CAGrad (Liu et al., NeurIPS 2021) — conflict-averse multi-task gradient descent
- Nash-MTL (Navon et al., ICML 2022) — Nash bargaining for multi-task learning

### Explore Later (long-term research)
- Street Gaussians (Yan et al., ECCV 2024) — driving-specific 3DGS
- 4D Gaussian Splatting (Wu et al., CVPR 2024) — dynamic 3DGS
- DCVC-DC (Li et al.) — state-of-art learned video codec
- VCM MPEG standard publications
- Neural Wrapping (Kim et al., CVPR 2023) — neural video wrappers for task-aware coding
