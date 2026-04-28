# Cosmos / MAE / Lyra 2.0 (arXiv 2604.13036) / Telescope (arXiv 2604.06332) — Deep Synthesis for the comma video compression challenge

**Date:** 2026-04-28
**Author:** Recovery synthesis (prior subagent killed by molt OOM)
**Reading depth:** Full PDFs of MAE, Telescope, Lyra 2.0 extracted via pdftotext; cosmos-cookbook GitHub repo enumerated and key recipe markdown read in full
**Scope:** Inverse-steganalysis-driven 300KB renderer + masks + poses archive, May 3 2026 deadline, current Lane A frontier 1.15 contest-CUDA, Quantizr leads at 0.33

---

## TL;DR — what these 4 resources actually give us

1. **NVIDIA Cosmos cookbook** — almost entirely irrelevant. 14B-parameter video diffusion at 720p × 93 frames. Even the `cosmos_cabr` Beamr CABR recipe is hardware NVENC RC-MAXQ tuning, not a learned codec. **One transferable idea**: their LoRA recipe (rank=32, alpha=32, target = q/k/v/o/mlp1/mlp2, ~1-2% of params) is the cleanest published low-rank-delta recipe; could inform a "Lane LR-V3" — LoRA on the dilated-h64 renderer to produce per-clip deltas without re-shipping weights. Otherwise: skim-killed because their compute, model size, and target task are 1000× off our budget.

2. **MAE (He et al. 2111.06377)** — directly applicable as a **conditional generator architecture pattern**. The key insight isn't the masking; it's the **asymmetric encoder/decoder where the decoder operates on a different (larger) token set than the encoder, with a mask token + positional embeddings filling the absent slots**. This maps almost 1:1 onto our compress→inflate split: encoder does heavy work at compress time, decoder is lightweight and runs at inflate time. **Two named lane proposals below (Lane MAE, Lane MAE-Q).**

3. **Lyra 2.0 (arXiv 2604.13036)** — the Cosmos team's NVIDIA paper on long-horizon 3D-consistent video synthesis from a single image. Built on Wan 2.1-14B DiT (way too big for us). But three sub-mechanisms transfer cleanly: **(a) FramePack variable-kernel temporal compression** (recent frames small kernel, distant frames large kernel — same token budget), **(b) self-augmentation training** (corrupt history latents during training so the model learns to denoise from imperfect context — solves the proxy/auth gap problem we keep hitting), **(c) canonical-coordinate warping conditioning** (warp position channels not RGB — removes warp artifacts as a "crutch"). Three lane proposals below (Lane FP, Lane SAUG, Lane CCW).

4. **Telescope (arXiv 2604.06332)** — Princeton + Torc Robotics paper on long-range object detection via **learnable hyperbolic foveation**. The math is `Φ(x) = (1-w(r))·x + w(r)·h(x)` where `h(x) = o + tanh(α·r)/r · (x-o)` is a Poincaré-disk-like radial contraction, blended by `w(r) = (1 - min(r/R, 1))^p`. Critically: the transform is **per-image learnable** (only 4 params: `α, R, p, o`), **invertible via Newton-Raphson**, **differentiable**, and **applied to high-res input then inverted on detections**. This is a non-uniform spatial sampler that magnifies the foe / center / lane-marking region. Two lane proposals below (Lane HF, Lane HFM).

---

## 1. NVIDIA Cosmos Cookbook (`github.com/nvidia-cosmos/cosmos-cookbook`)

### 1.1 What it actually contains

Repository structure (verified via `gh api`):
- `docs/recipes/{data_curation, post_training, inference, end2end}/` — markdown recipe pages
- `scripts/examples/{predict2.5, transfer1, reason1, reason2}/` — runnable scripts (only `predict2.5/gr00t-dreams` and `transfer1/inference-x-mobility` exist as runnable code; most are markdown only)
- `docs/core_concepts/distillation/` — Cosmos Predict 2.5 distillation guide (DMD2)

Cosmos model lineup (verified from `glossary.md` + `index.md`):
- **Cosmos Predict 2 / 2.5** — diffusion-transformer video generator, future-state prediction. Predict 2.5 base model is **2B params** (LoRA recipe page), inference at **704×1280 × 93 frames**.
- **Cosmos Reason 1 / 2** — 7B vision-language model.
- **Cosmos Transfer 1 / 2.5** — multi-control video generator (depth, segmentation, LiDAR, HDMap conditioning).
- **Cosmos Curator** — Ray-based data curation pipeline; this is where Beamr CABR lives.
- **Cosmos RL** — distributed SFT/RL training, supports FP8/FP4 precision.

### 1.2 Core technique: the only one with concrete numbers

**LoRA on Predict 2.5** (`docs/recipes/post_training/predict2_5/sports/post_training.md`):
- Base model: 2B params, video diffusion transformer
- LoRA rank = 32, alpha = 32
- Target modules: `q_proj, k_proj, v_proj, output_proj, mlp.layer1, mlp.layer2`
- LR = 1e-4, weight decay = 1e-3, batch = 1, max iter = 10000
- Trains "only ~1-2% of total model parameters"
- Resolution 704×1280, 93 frames per clip, ~4350 training clips
- Distributed: `torchrun --nproc_per_node=8`

**DMD2 distillation on Predict 2.5** (`docs/core_concepts/distillation/distilling_predict2.5.md`):
- 36-step teacher → 4-step student via DMD2 (Variational Score Distillation + adversarial)
- Critic-to-student ratio K=4 (every 5th step updates student)
- "Satisfactory video quality after 1500 iterations = 300 student + 1200 critic steps"
- Eliminates CFG → additional 2× speedup at inference

**Beamr CABR recipe** (`docs/recipes/data_curation/cosmos_cabr/cabr_recipe.md`):
- It is **not a learned codec**. It is `ffmpeg -hwaccel cuda -vcodec h264_nvenc -rc maxq -preset p7 -tune hq` — NVENC's "find the maximum compression that preserves quality" mode.
- 41–57% bitrate savings vs default Curator AV pipeline, mostly from per-frame NVENC quality estimation.
- Quality validated downstream by checking T5-XXL caption embedding drift and detection mAP (0.96).

### 1.3 What COULD apply to us

- **Lane LR-V3 — LoRA-as-renderer-delta.** Premise: train base renderer once; for each new operating point (resolution, mask budget), ship a tiny LoRA delta (rank=4-8, only on conv weights) instead of a fresh checkpoint. **Predicted band [1.05, 1.35]** — barely helps single-clip task. **Skip unless we go multi-clip.** Comma challenge has one fixed 60s clip, so per-clip LoRA buys nothing. **Cost N/A — do not pursue.**
- **Distillation pattern audit.** DMD2's "K critic steps per student step" pattern suggests our 5-stage QAT could benefit from a discriminator co-trained with the student in stage 4. We do **not** currently use adversarial distillation. Worth a Lane V proposal: "Lane V-DMD" — adversarial QAT where a frozen base scorer-projection serves as critic. **Predicted band [0.95, 1.20]** vs current 1.15 (modest win possible, complexity high). **Cost: $2-4 + 8h on 4090.** **Recommendation: skip — Lane S (self-compress) and Lane W (hard-pair) are higher EV.**

### 1.4 What COULD NOT apply

- **Cosmos Predict / Transfer base models** are 2-14B params and generate 720p × 93-frame video clips. Our budget is ~80-100K params and we generate 1200 frames at 384×512. Wrong scale by 4-5 orders of magnitude. The 2B Predict model alone, even at INT4, is ~250MB — bigger than our entire archive budget × 1000.
- **Beamr CABR** is hardware NVENC tuning; the contest forbids any encoder that loads scorers at inflate, but more fundamentally it's a CRF-by-block knob, not a learned codec. We already use AV1 CRF tuning for masks.mkv; CABR offers nothing additional because (a) we use libsvtav1/AV1 not NVENC h264, and (b) our content is single-stream automotive driving, not the diverse Cosmos training distribution where CABR's per-frame quality model was tuned.
- **Cosmos Reason VLM 7B** for compression is absurd at our budget. Skip.

### 1.5 Stacking opportunity (one-line per existing lane)

| Existing lane | Cosmos add | Verdict |
|---|---|---|
| Lane A (1.15 baseline) | DMD2-style critic in QAT | Marginal; Lane W is better signal |
| Lane V (Quantizr replica) | LoRA delta per content | Useless single-clip |
| Lane I (Cool-Chic) | NVENC CABR for masks.mkv | We already use libsvtav1; switching to h264_nvenc costs rate, not gains |

**Council verdict on Cosmos: park.** No predicted EV > 0.05 score reduction at any reasonable cost. The cookbook is a 269MB repo of scripts for a different problem.

---

## 2. Masked Autoencoders (MAE) — He et al. arXiv 2111.06377

### 2.1 Core technique (verified from full PDF, 797 lines extracted)

**Asymmetric encoder–decoder for self-supervised pretraining:**
1. Image → 16×16 patches (ViT). 75% masked uniformly at random.
2. **Encoder** (heavy ViT-L: 24 blocks × 1024-d, or ViT-H: 32 blocks × 1280-d) processes **only the visible 25% of patches**. No mask tokens in encoder.
3. **Decoder** (lightweight: 8 blocks × 512-d, ~9% FLOPs of encoder) processes the **full 196-token set** = encoded visible tokens + shared learnable mask tokens, with positional embeddings added to all.
4. **Loss = MSE on masked patches only**, computed in **per-patch normalized pixel space** (subtract patch mean, divide by patch std). Normalized variant gives +0.5% on FT.

Key ablations (Table 1 verbatim):
- Mask ratio 75% optimal. 60% works. 90% degrades.
- Decoder depth: 1 block → 84.8% FT, 8 blocks → 84.9% FT (FT insensitive); 1 block → 65.5% lin, 8 blocks → 73.5% lin (linear probing wants depth).
- Decoder width: 512-d sweet spot.
- **Removing mask tokens from encoder = 3.3× FLOPs reduction, 2.8× wallclock speedup**, AND +14% linear probe accuracy (because train/deploy distribution match).
- Random sampling > block sampling > grid sampling (75% block degrades FT to 82.8%; random stays at 84.9%).

### 2.2 What this means for our problem

Map MAE onto compress/inflate:

| MAE term | Our analog |
|---|---|
| Visible patches (25%) | Per-frame masks.mkv (encoded with AV1) — what we explicitly send |
| Encoder | Compress-time TTO + renderer training (heavy, unlimited compute) |
| Mask tokens + decoder | Inflate-time renderer (lightweight, 30-min budget) |
| Reconstruction MSE | Our distortion = SegNet KL + PoseNet MSE |
| Per-patch pixel normalization | Pre-norm of mask logits before quantization |

The asymmetric architecture is the **mathematical justification** for what we already do empirically: do all the work at compress, ship sparse "visible-patch" data, and let the inflate-time renderer fill in the rest.

### 2.3 Lane proposals from MAE

**Lane MAE — MAE-style sparse mask supervision during renderer training.** 
Premise: instead of training the renderer to reconstruct full frames from full mask sequences, train it to reconstruct full SegNet/PoseNet outputs from a **75%-masked** mask stream. The renderer must learn to in-fill missing-mask frames using only neighboring temporal context. At inflate time we ship **only 300 of 1200 mask frames** (75% omitted) — rate cost drops by ~4× on the masks.mkv component (which is currently ~150-180KB of our 300KB archive).
- **Predicted band [0.80, 1.10]** — depends on how lossy the mask in-painting is. SegNet KL is the dominant term, so even 0.05 KL inflation × 100 = 5 points lost; we need the renderer to do real temporal interpolation.
- **Critical risk:** the Quantizr "encode only odd frames, warp even frames" trick already uses 50% reduction and we showed it BREAKS our PoseNet (memory: `feedback_half_frame_breaks_posenet`, 17.55 score). MAE-style 75% on dilated-h64 is even more aggressive. **Mitigation:** train with mask-half-sim-prob > 0 from the start (joint training), don't bolt it onto a baseline checkpoint.
- **Cost: $4 + 16h on 4090** (full retrain of dilated-h64 with mask masking augmentation).
- **Council pre-check:** The MAE paper Figure 4 shows reconstructions from 95% masking are still semantically plausible — but they are perceptually blurrier. SegNet argmax disagreement is harsh on blur because boundaries shift by 1 pixel. So this lane needs a gate: smoke test at 50% mask ratio first; if PoseNet doesn't blow up, scale to 75%.

**Lane MAE-Q — MAE-trained renderer + heavyweight (compress-time-only) MAE-encoder TTO.**
Premise: at compress time, fit an MAE-style heavy encoder (16-block ViT) over the FULL 1200-frame mask sequence. Quantize its 25% visible-patch output to FP4. Ship: (a) the FP4-quantized visible patches, (b) only the lightweight MAE decoder weights (not the encoder). At inflate, decoder + ship-side quantized latents reconstruct the 5-class mask grid the contest scorer needs.
- This is **structurally what Cool-Chic does** (Lane I) — encode is heavy, decode is small. Lane I uses INR + autoregressive code; Lane MAE-Q would use ViT-decoder + position-embedded mask tokens.
- **Predicted band [0.75, 1.05].** This is a different shape of bet than Lane I — wins if our scorer gradients prefer ViT decoder smoothness over INR sharpness. Loses if the ViT decoder needs >40K params to be competitive (puts us over budget).
- **Cost: $6 + 24h** (architecture exploration is not free).
- **Critical risk:** The contest 30-min T4 inflate budget. ViT decoder forward over 1200 mask frames at 384×512 with 8 blocks × 512-d = ~150 GFLOPs/frame × 1200 = 180 TFLOPs. T4 is ~8 TFLOPs FP16; that's 22 seconds — fine. But if we go to 16-block decoder, doubles to 44 seconds. Stay in the 8-block × 512-d zone.

### 2.4 What COULD NOT apply

- **MAE pretraining on ImageNet-1K** for the encoder is irrelevant; we don't need a generic feature extractor, we need a renderer that targets specific scorer outputs. The scorer IS the supervisor.
- **MAE's transfer-to-COCO-detection results** are not our problem. We don't transfer; we deploy one fixed checkpoint.
- **Per-patch pixel normalization as the reconstruction target** — we cannot use raw pixel space because the contest scorer takes raw RGB → SegNet/PoseNet, and their argmax/MSE losses work in scorer output space. So we keep our existing scorer-roundtrip loss; we just change the input sparsity pattern.

### 2.5 Stacking with existing lanes

| Existing lane | MAE add | Stacking notes |
|---|---|---|
| Lane A baseline | Replace per-frame inputs with 75%-masked sparse stream + temporal interp head | Lane MAE; replaces the rate term on masks.mkv (-30 to -50%) at SegNet cost |
| Lane I (Cool-Chic) | None — both target same compression slot | Pick one, not both |
| Lane V (Quantizr replica) | The half-frame trick IS a 50% MAE | Already discovered the principle; we just failed to train for it. Lane MAE-V = "Quantizr's half-frame BUT trained with mask-augmentation from epoch 0" |
| Lane W (hard-pair) | Hard-pair weighting on the in-paint loss | Strong stack; both attack PoseNet-on-hard-frames |
| Lane EC (engineered corrections) | Use EC-derived correction as the "mask token" content | Asymmetric; EC at compress, inject at inflate |

**Council verdict on MAE:** **Lane MAE-V** (Quantizr-style half-frame BUT trained jointly with mask-aug) is the highest-EV proposal in this synthesis. It directly attacks our biggest known regression (half-frame breaks PoseNet) with the right training recipe.

---

## 3. Lyra 2.0 — NVIDIA arXiv 2604.13036 (1414 lines extracted)

### 3.1 Core technical contributions (verbatim from §4)

Lyra 2.0 is a long-horizon camera-controlled video diffusion + 3DGS reconstruction system. Three mechanisms are extractable for our problem:

#### 3.1.1 FramePack variable-kernel temporal compression

History tokens are organized as `f1k1 f4k2 f1k1 f16k4 f2k2 f1k1 g20` where `fnkm` = `n` frames at spatial subsample factor `m`. Recent frames get **kernel 1 (full resolution)**, distant frames get **kernel 4 (16x downsample)**. This is **non-uniform temporal allocation under a fixed token budget**.

- VAE: Wan 2.1, downsamples 8× spatial, 4× temporal, latent C=16 channels.
- Anchor frame (initial image) is always at full resolution as "early-established endpoint" — prevents drift.

#### 3.1.2 Self-augmentation training (anti-drifting)

The killer insight (lines 433-483):

> Root cause of drift: **observation bias** — train on GT history, deploy on own outputs.

Math:
- With probability `p_aug`, sample `t ~ U(0, 0.5)`.
- Corrupt history latent: `z_hist_t = (1-t) z_hist_0 + t·ε`, ε ~ N(0,I).
- Run **one** denoising step: `ẑ_hist_0 = z_hist_t - t·v_θ(z_hist_t, t, c)`.
- **Substitute ẑ_hist_0 into the training context** in place of the clean GT.
- Compute loss against clean target — model learns to denoise toward clean target FROM corrupted context.

This is a single extra DiT forward pass, ~minimal overhead, and it directly closes the train/deploy distribution gap.

#### 3.1.3 Canonical coordinate warping (anti-forgetting)

Instead of warping RGB history frames to the target view (which inherits disocclusion holes, depth-bleeding, stretching), they warp a 4-channel **canonical coordinate map** `(u, v, slot_idx, depth)`. The DiT gets the geometric correspondence without RGB hallucination crutches. Quote: "warped image acts as a crutch that bypasses the generative prior rather than informing it."

### 3.2 Wan 2.1-14B is unusable for us

Lyra builds on a 14B-param DiT. We have ~100K. So forget the architecture. The **mechanisms** are what transfers.

### 3.3 Lane proposals from Lyra 2.0

**Lane FP — FramePack-style variable temporal kernel for masks.mkv.**
Premise: instead of encoding 1200 mask frames at uniform AV1 CRF, allocate bits non-uniformly by **per-frame importance to the contest scorer**. Frames where the SegNet/PoseNet pair is hard get full-res masks; frames where the pair is easy get 4× downsampled masks (with temporal interpolation at inflate). Total mask budget ~150KB; we can fit ~600 high-res + 600 low-res frames vs current 1200 medium-res.
- **Predicted band [0.85, 1.10].** Wins if Lane W's hard-pair signal generalizes to "hard mask quality"; loses if SegNet is uniformly demanding (currently it is — at our floor, SegNet rate is dominated by mask boundary precision, not by which frame).
- **Stacking:** stacks cleanly with **Lane W** (hard-pair detector) — Lane W tells us which frames are hard, Lane FP allocates bits there. **This is a natural composition.**
- **Cost: $3 + 12h** on 4090.
- **Critical risk:** the half-frame regression (`feedback_half_frame_breaks_posenet`). FramePack-on-masks is a generalization of that trick. **Mitigation: same as Lane MAE — train renderer jointly with variable-kernel mask aug.**

**Lane SAUG — Self-augmentation for the renderer (close the proxy/auth gap).**
Premise: the proxy/auth gap (memory: `feedback_proxy_auth_math_useless`, ratios 100-350×) IS observation bias. At training time we feed the renderer **clean masks + clean poses**; at inflate time it gets **AV1-quantized masks + TTO-noisy poses**. Solution = Lyra's recipe: with `p_aug = 0.5`, replace the training inputs with **one-step denoised reconstructions of the AV1-encoded masks** (decode → encode → decode round-trip). Force the renderer to learn from "what inflate actually sees."
- This is **eval_roundtrip on steroids**. Current eval_roundtrip simulates the 384→874→uint8→384 numerical roundtrip but uses clean masks. Lane SAUG would also simulate the AV1 CRF50 roundtrip, the FP4 quant roundtrip on the renderer weights themselves, and the pose-TTO drift.
- **Predicted band [0.75, 1.05].** This is potentially very high EV because every component of our 4× proxy/auth gap could be partially closed, and we have empirical evidence that gap is the single biggest blocker.
- **Stacking:** stacks with EVERYTHING. eval_roundtrip is already non-negotiable per CLAUDE.md; Lane SAUG is its missing complement.
- **Cost: $5 + 18h** for full retrain with self-aug; or $1 + 4h for a Lane SAUG smoke (50 epochs, see if proxy/auth ratio drops from 350× to <50×).
- **Critical risk:** the AV1 encode is non-differentiable, but Lyra's trick is just a single forward pass through the encoder (treated as a black-box noise source); we don't need gradients through AV1, only the corrupted output as input. This is feasible.

**Lane CCW — Canonical-coordinate-warping for pose representation.**
Premise: when we run pose TTO, we currently optimize 6-DOF SE(3) deltas. The renderer takes (mask_t, pose_t→t+1, mask_t+1) and tries to predict scorer outputs. Replace pose conditioning with a **canonical coordinate channel**: at compress time, compute the per-pixel optical flow induced by (pose_t→t+1) on the road plane (we have homography from CLAUDE.md — fx=910, pp=(582,437), road plane known); ship that flow as a 2-channel low-res map (~1-2KB at 24×32) per pair. At inflate, inject this map as MAE-style "position channel" alongside the masks.
- **Predicted band [0.90, 1.15].** This decouples the rank-1 PoseNet rank discovery (memory: `project_posenet_rank1_discovery` — dim 0 = 99.8% var) from a noisy 6-DOF parameterization. We've already seen rank-1 zoom helps; canonical coord warping is a 2D generalization that captures the same signal more directly.
- **Stacking:** Replaces or augments Lane M (pose-from-embedding) and Lane EC (engineered corrections). **Strong with Lane M+N (radial-zoom) which is currently NEGATIVE (2.35 vs baseline 2.29) — the failure mode there was 1-DOF poses fed to 6-DOF-trained renderer. Lane CCW is the right direction: feed the renderer a richer geometric channel, not a stripped-down one.**
- **Cost: $3 + 12h** for retrain with CCW conditioning replacing pose vector.

### 3.4 What COULD NOT apply

- The full Lyra system (camera-controlled video gen + 3DGS) doesn't apply — we don't generate video, we reconstruct a fixed video from a 300KB description. But:
- The 3DGS lifting branch (Depth Anything v3, 2-Gaussian-per-pixel decoder) is **interesting** because we could imagine encoding a sparse 3DGS for the scene and re-rendering, but at our budget (~100KB after masks), each Gaussian costs ~32 bytes, so we'd get ~3000 Gaussians for a 384×512 video — that's 1 Gaussian per ~65 pixels, far too sparse for the SegNet boundary detail we need. **Skip-killed: rate-budget infeasible.**
- DMD distillation in §4.5 (35 → 4 steps, 13× speedup) is interesting if we ever build a diffusion-based renderer, but our renderer is a single-pass U-Net-ish thing; nothing to distill multi-step from. **Skip.**

### 3.5 Stacking summary

| Lyra mechanism | Best stack with |
|---|---|
| FramePack (variable kernel) | Lane W (hard-pair difficulty) → Lane FP |
| Self-augmentation | Everything; especially Lane S, Lane V, Lane Ω |
| Canonical-coord warping | Lane M+N (revives a dead radial-zoom idea); Lane EC; Lane Ω |

**Council verdict on Lyra:** **Lane SAUG** is the second-highest-EV proposal in this synthesis. The proxy/auth gap is our single largest blocker (memory: `project_lane_b_pose_tto_proxy_auth_gap`, `feedback_mps_cuda_drift_critical`, `project_baseline_0_9001_lost_archive_test`). Lyra's self-augmentation IS the textbook fix for "observation bias" and the math is clean enough to implement in <100 LOC.

---

## 4. Telescope — Princeton/Torc arXiv 2604.06332 (872 lines extracted)

### 4.1 Core technique (verbatim from §3.1, 3.2)

**Hyperbolic foveated transform** in 4 parameters `(α, R, p, o)`:

```
h(x; o) = o + (tanh(α·r) / r) · (x - o)         where r = ||x - o||         (Eq. 1)
Φ(x)    = (1 - w(r)) · x + w(r) · h(x)          (Eq. 2)
w(r)    = (1 - min(r/R, 1))^p
```

For `r ≪ R`: hyperbolic contraction around `o` (magnifies the center).
For `r ≥ R`: identity (preserves periphery).

**Inverse `Φ⁻¹`** is computed differentiably via Newton-Raphson:
```
x^(k+1) = x^(k) + η · (y - Φ(x^(k))),   x^(0) = y, η ∈ (0, 1]      (Eq. 3)
```

**Bounding box reparameterization** in the warped Riemannian space:
```
b' = [Φ_x(c), Φ_y(c), ||t_x||, ||t_y||]
t_x = J_Φ(c) · [w, 0]^T,   t_y = J_Φ(c) · [0, h]^T          (Eq. 4, 5)
```
where `J_Φ(c)` is the Jacobian of Φ at center `c`.

**Foveation prediction net:** small FFN takes 256×256 or 512×512 downsampled image → predicts `(o, R)`. `α=2.0, p=2.0` set by grid search on TruckDrive.

### 4.2 Architecture (Tables 1, 2)

- Backbone: SAM3 image encoder (frozen) + Deformable DETR head + de-noising training [56].
- Backbone alternatives tested: DINOv2, DINOv3, Perception Encoder — all weaker than SAM3.
- Resolution: 1024×1024 input.
- Foveation FFN trained on downsampled 256×256 input → minimal compute overhead.

Result: mAP 0.185 → 0.326 at >250m range (76% relative improvement). Overall mAP 0.325 → 0.497.

### 4.3 Why this matters for us — the geometric connection

We have these memories:
- `project_posenet_rank1_discovery` — PoseNet Jacobian rank ≈ 1, dim 0 = 99.8% variance, optimal = scalar radial zoom from FoE (256, 174).
- `project_hardware_geometry_chroma_full` — EON camera fx=910, pp=(582, 437), FoE at (256, 174).
- `project_lane_marking_speed_estimation` — lane marks (MUTCD 3m × 15cm) in all 1200 frames; displacement → speed → zoom.
- `project_lane_mn_radial_zoom_negative` — radial-zoom 1-DOF + Fridrich L∞ on baseline = 2.35 (NEGATIVE, +0.06 vs baseline).

**The key insight from Telescope:** the foveation transform is **per-image learnable AND invertible**. Our radial-zoom failure was that we used a fixed parameterization (1-DOF zoom) on a renderer trained for 6-DOF poses; the geometry was off-manifold. Telescope says: make the transform itself learnable per-frame (per-image) AND ensure invertibility so loss can flow back through it.

### 4.4 Lane proposals from Telescope

**Lane HF — Hyperbolic foveation as the renderer's spatial conditioning.**
Premise: replace our current per-frame conditioning (mask + pose vector + frame index) with a **per-frame hyperbolic foveation map**. Compress-time: fit (α_t, R_t, p_t, o_t) per pair to maximize SegNet/PoseNet alignment between rendered output and GT; ship 4 floats × 1199 pairs = ~19KB to optimize_poses.pt (replaces or supplements the current SE(3) poses). Inflate: the renderer applies Φ to the input mask, runs its forward pass at the warped resolution (so distant lane markings get more pixels), then applies Φ⁻¹ to the output to align with the original frame.
- **Predicted band [0.85, 1.10].** This is a geometrically motivated pose alternative. It directly addresses the rank-1 PoseNet discovery: the foveation IS a learned radial zoom, but with center (`o`) and strength (`α`) per-frame.
- **Stacking:** REPLACES Lane M+N (radial-zoom) with the right framework. Stacks with **Lane W** (hard-pair) because hard pairs are exactly the ones where lane markings matter most and foveation helps most.
- **Cost: $4 + 16h** (renderer retrain + foveation FFN training).
- **Critical risk:** Φ⁻¹ via Newton-Raphson at inflate time per-frame × 1200 frames. If each NR converges in 5 iterations of 1 forward + 1 backward (autograd-free, just iterative), and renderer is ~1ms/frame, foveation NR is <0.1ms. T4 budget OK.
- **Pre-flight:** verify the inverse converges for our typical parameter range (small `α`, large `R`). Telescope's appendix proves convergence; we should sanity check empirically before building.

**Lane HFM — Hyperbolic foveation on the *masks* not the renderer input.**
Premise: encode masks.mkv at non-uniform resolution: high-res in the center (lane markings, foe region), low-res at periphery. Use the foveation transform to define the resolution map. Decode: apply Φ⁻¹ at inflate to recover full-res masks.
- This is **per-frame adaptive resolution coding** — a learned spatial-frequency allocator.
- **Predicted band [0.95, 1.15].** Modest. The current AV1 CRF tuning already does some of this implicitly (more bits in high-detail regions). Foveation makes it explicit and frame-controllable. But the masks are 5-class index maps, not RGB; AV1's content-adaptive bit allocation matters less.
- **Stacking:** stacks with Lane FP (FramePack temporal) for **2D adaptive (spatial × temporal) bit allocation**. Could be the canonical compose with Lane W as the difficulty signal.
- **Cost: $3 + 10h.**

### 4.5 What COULD NOT apply

- **SAM3 image encoder** is ~2-5 GFLOPs/image and we ship the encoder weights. Not viable as our renderer backbone.
- **Deformable DETR head** is overkill — we don't do detection, we do dense reconstruction.
- **TruckDrive dataset** focus on >250m objects is the wrong domain — our problem is short-range driving (lane markings 3-30m, vehicles 10-100m). The foveation **center** for us is the foe (256, 174), not "near horizon" — we'd need to retune `(α, R)` for our scene statistics. Empirically α=2.0, p=2.0 may not be optimal for the comma EON camera; grid search is required.
- **De-noising training scheme [56]** is detection-specific (random box noising added to query inputs). Not directly applicable but conceptually overlaps with Lyra's self-augmentation — both corrupt training inputs to match deploy distribution.

### 4.6 Stacking summary

| Existing lane | Telescope add | Verdict |
|---|---|---|
| Lane M+N (radial zoom NEGATIVE) | Lane HF — replaces 1-DOF with 4-DOF learnable foveation | Revives a dead lane with the right math |
| Lane W (hard-pair) | Lane HF center (o) co-trained with hard-pair signal | Strong stack |
| Lane FP (FramePack, proposed above) | Lane HFM (spatial foveation on masks) | 2D adaptive coding — composes |
| Lane EC (engineered corrections) | Foveation-conditioned corrections | Augments naturally |
| Lane Ω (per-weight bit alloc) | None — orthogonal | Stack independently |

**Council verdict on Telescope:** **Lane HF is the third highest-EV proposal**. It rescues our dead Lane M+N with the right framework, has clean math, and stacks naturally with Lane W (the current top in-flight bet).

---

## 5. Council Recommendation — Ranked Proposals by EV

EV = (predicted_band_midpoint_reduction × confidence) ÷ cost.
Current frontier: Lane A baseline 1.15 [contest-CUDA]; Quantizr 0.33; sub-1.0 = first-ever for us.
"Reduction" measured against 1.15 baseline.

| Rank | Lane | Predicted band | Mid Δ | Conf | Cost ($/h) | EV (Δ·conf/$) |
|---|---|---|---|---|---|---|
| 1 | **Lane SAUG** (Lyra self-augmentation, closes proxy/auth gap) | [0.75, 1.05] | -0.25 | 0.55 | 5 / 18 | **0.0275** ★★★ |
| 2 | **Lane MAE-V** (Quantizr half-frame TRAINED with mask-aug from epoch 0) | [0.80, 1.10] | -0.20 | 0.50 | 4 / 16 | **0.0250** ★★★ |
| 3 | **Lane HF** (Telescope hyperbolic foveation as per-frame conditioning) | [0.85, 1.10] | -0.18 | 0.40 | 4 / 16 | **0.0180** ★★ |
| 4 | **Lane FP** (FramePack variable kernel on masks.mkv, stacked w/ Lane W) | [0.85, 1.10] | -0.18 | 0.35 | 3 / 12 | **0.0210** ★★ |
| 5 | **Lane CCW** (canonical-coord warp pose channel) | [0.90, 1.15] | -0.13 | 0.40 | 3 / 12 | **0.0173** ★★ |
| 6 | **Lane MAE** (75% mask-frame omission + temporal interp) | [0.80, 1.10] | -0.20 | 0.30 | 4 / 16 | **0.0150** ★ |
| 7 | **Lane MAE-Q** (ViT-decoder + quantized visible-patch latent) | [0.75, 1.05] | -0.25 | 0.20 | 6 / 24 | **0.0083** ½ |
| 8 | **Lane HFM** (spatial foveation on masks) | [0.95, 1.15] | -0.10 | 0.30 | 3 / 10 | **0.0100** ½ |
| 9 | Lane V-DMD (DMD2-style adversarial QAT) | [0.95, 1.20] | -0.075 | 0.25 | 3 / 8 | 0.0063 |
| — | Lane LR-V3 (LoRA delta) | [1.05, 1.35] | +0.05 | 0.50 | 3 / 8 | NEG — skip |

### 5.1 Top 3 lanes — concrete next steps

#### #1 — Lane SAUG (highest EV)
**Why first:** every other lane assumes its training-time signal predicts deploy-time score. Empirically it does not (proxy/auth ratio 100-350×). Lane SAUG is the structural fix. If it works, every other lane gets retroactively more reliable.
- **Subagent prompt seed:** "Implement Lyra-2.0-style self-augmentation in `src/tac/training.py`. With `p_aug=0.5` per batch, replace the training-time mask input with `decode(av1_encode(mask_t, crf=50))` and replace pose with `pose_t + N(0, σ_pose)` where σ_pose matches our TTO step-size distribution. Compute loss against clean targets. Add CLI flag `--self-aug-prob 0.5`. Smoke test 50 epochs, measure proxy-auth ratio reduction; if >2× improvement, full retrain. NO ad-hoc scripts; modify `experiments/pipeline.py` profile system."
- **Pre-flight gate:** if proxy/auth gap doesn't drop below 50× in smoke, kill before full retrain.
- **Anti-pattern check:** ensure `eval_roundtrip=True` is preserved (CLAUDE.md non-negotiable). Self-aug REPLACES the clean-input shortcut, eval_roundtrip simulates the inflate-time numerical roundtrip. They are complementary not redundant.

#### #2 — Lane MAE-V (Quantizr half-frame BUT trained jointly)
**Why second:** half-frame is the single biggest known rate win (-0.108 vs Lane A baseline). Our previous half-frame attempt failed (17.55 score) because we bolted it onto a baseline checkpoint. MAE Table 1f result: block-50% works at 83.9% FT; block-75% degrades to 82.8%; **but the model has to be trained on the masking pattern**. The Quantizr trick at 50% mask should work IF trained jointly.
- **Subagent prompt seed:** "Replicate Quantizr's odd-frame-only mask encoding, BUT train the dilated-h64 renderer with `mask_half_sim_prob=0.5` from epoch 0 (not as a finetune). Renderer must learn to interpolate even-frame masks from odd-frame masks via temporal head. Use existing `src/tac/architectures.py:DilatedRendererH64`; add a temporal interpolation block in front. NO ad-hoc; profile in `src/tac/profiles.py:dilated_h64_mae_v`."
- **Pre-flight:** verify our motion module's `(e_t1 - e_t).abs()` diff feature still gets non-zero input under half-frame supervision. Per `feedback_half_frame_breaks_posenet`, this was the breakage point.

#### #3 — Lane HF (hyperbolic foveation conditioning)
**Why third:** revives Lane M+N with the right framework. Lower confidence than #1 and #2 because we have less direct evidence; higher upside than #4-#8 because it's the only lane that addresses geometric scale-disparity directly.
- **Subagent prompt seed:** "Implement Telescope's hyperbolic foveation Φ (Eq. 1, 2) and inverse via Newton-Raphson (Eq. 3) in `src/tac/foveation.py`. Add a 4-DOF parameter prediction head to the renderer that outputs (α_t, R_t, p_t, o_t) per frame from a downsampled mask + pose embedding. Train with `Φ` applied to mask input, renderer forward at warped resolution, `Φ⁻¹` applied to output before scorer loss. Verify Newton-Raphson converges in ≤10 iterations for our parameter ranges before integration."

### 5.2 Stacking strategy — what to compose IF top 3 individually win

Per CLAUDE.md operating rule: "Prefer at most 3 experiments per cycle." So:

**Cycle 1 (parallel, 3 lanes):**
- Lane SAUG (smoke first, then full)
- Lane MAE-V (full retrain — no smoke needed, half-frame is well-understood)
- Lane HF (smoke first — verify NR converges, foveation FFN trains)

**Total est cost: $13 + parallel 18h on 3× 4090 = ~$13 + 12h wallclock.**

**Cycle 2 (composition, only if Cycle 1 produces ≥1 winner):**
- Best of Cycle 1 + Lane W (hard-pair, in-flight) + Lane FP (FramePack on masks)
- Predicted composed band: [0.55, 0.85] — this is the path to first-ever sub-1.0

**Cycle 3 (architectural rate attack, only if budget remains):**
- Lane MAE-Q (ViT decoder for masks) — moonshot
- Lane Ω (per-weight bit alloc) — already in TIER 3 from MEMORY.md

### 5.3 What to NOT do

Per the FORBIDDEN PATTERNS section of CLAUDE.md and the meta-bug catalog:
- **No CPU/MPS for any of these.** All retrains on CUDA Vast.ai 4090; all auth eval on CUDA via `inflate.sh` + `upstream/evaluate.py`.
- **No new CLI flags without `grep "add_argument"` first.** If Lane SAUG needs `--self-aug-prob`, verify the target script has the parser entry before wiring it into `pipeline.py`.
- **No "default convenience" device fallbacks.** Every script must raise on no-CUDA.
- **Lane SAUG must NOT skip eval_roundtrip.** Self-aug + roundtrip stack; they don't substitute.
- **Tag every score with lane.** `[contest-CUDA]` only.
- **Use named profiles in `src/tac/profiles.py`** not ad-hoc CLI flags. `dilated_h64_mae_v`, `dilated_h64_saug`, `dilated_h64_hf`.

### 5.4 Bottom line

The 4 resources collectively suggest one bet is structurally distinct from anything we've tried: **Lane SAUG**, the Lyra self-augmentation pattern, attacks our single largest documented blocker (proxy/auth gap). The MAE asymmetric-decoder math justifies what we're already doing and validates a half-frame retraining (Lane MAE-V). Telescope's foveation gives us the right math to revive Lane M+N (Lane HF). NVIDIA Cosmos is mostly noise at our scale; only the LoRA recipe is concrete and it doesn't fit our single-clip task.

If we run only ONE additional experiment between now and the May 3 deadline, run **Lane SAUG**. If we run three, **SAUG + MAE-V + HF in parallel.**

— end synthesis —
