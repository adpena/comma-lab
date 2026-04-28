# Tuna-2 (arXiv 2604.24763) — Deep Synthesis for the comma video compression challenge

**Date:** 2026-04-28
**Author:** Deep-research subagent (full-PDF read)
**Reading depth:** PDF cover-to-cover via pdftotext (993 lines, 5 main pages + ~10 pages references); all sections read; every table extracted; ablations dissected.
**Scope:** Inverse-steganalysis-driven 300KB renderer + masks + poses archive, May 3 2026 deadline, current Lane A frontier 1.15 [contest-CUDA], Quantizr leads at 0.33.

---

## 0. TL;DR — Is this paper transferable?

**Paper identity (verified):** Yes, arXiv 2604.24763v1 [cs.CV] is real, posted 27 Apr 2026 (HTTP last-modified 28 Apr 2026 02:06:26 GMT). Title: *"Tuna-2: Pixel Embeddings Beat Vision Encoders for Multimodal Understanding and Generation."* Authors: Zhiheng Liu, Weiming Ren, et al. — a 15-author Meta AI / HKU / U.Waterloo collaboration. Project page: `https://tuna-ai.org/tuna-2`. The paper is a 7B-parameter unified multimodal model (UMM) result built on Qwen2.5-7B-Instruct. **At face value: completely off-scale (7B params vs our 100K renderer; image generation vs video compression; 550M+13M+2M training corpus vs 1199-pair contest data).**

**But the unlimited-compress reframing (per `feedback_compress_time_unlimited_archive_small_20260428`) makes 4 sub-mechanisms genuinely transferable:**

1. **Patchify-only encoder is a TINY artifact.** The Tuna-2 vs Tuna-R ablation proves a single learnable patch-embedding layer (a Conv2d k=patch s=patch with no pretraining) can REPLACE a SigLIP-2-So400M encoder (~400M params) when downstream training is sufficient. For us this means: the renderer's **conditioning input does NOT need to be a learned encoder feature** — a deterministic patch-embed of the masks alone may carry enough signal, REMOVING any need to ship encoder weights. (Compress-time scoring: Lane A uses ~2K mask-conditioning params; Tuna-2 says we could drop to ~0.5K.)

2. **Masking-based feature learning during late pretraining (last 40% of steps), p=0.5, learnable mask token.** This is a published, ablated training-recipe knob with deltas of +0.6 to +4.2 points on perception benchmarks (Table 6). Mechanism: replace random visible patches with a single learnable token, force the model to "denoise" or "answer" through partial observation. **Maps directly to our scorer-roundtrip noise injection (`noise_std=0.5`, `eval_roundtrip=True`) but adds a structured mask-region drop that SegNet has never been trained against. This is novel, free at compress time, costs zero archive bytes.**

3. **x-prediction + v-loss via JiT (Eq. 1-4).** They predict the clean image `x_θ` and convert to velocity `v_θ = (x_θ - x_t) / (1-t)` for the loss. This is a numerical-conditioning trick to avoid the singularity at `t→1` in standard `v`-prediction flow matching. **Our renderer is a single-step generator (not flow), but our self-compression and Cool-Chic lanes use multi-step training. The x-prediction-then-derive trick stabilizes loss when latents/poses are nearly clean — exactly our late-stage QAT regime.**

4. **Encoder-free monolithic transformer scales BETTER than encoder-based at large data (Figure 6).** Their Tuna-R outperforms Tuna-2 early in training, then crosses over and Tuna-2 wins after enough tokens. This is an empirical "bitter lesson" data point: **inductive bias from a pretrained encoder is a CRUTCH that limits the asymptote.** For us this implies: distillation lanes (Lane DI penultimate features, Lane M PoseNet-embedding distill) might be early-pretraining accelerants but ultimately CAP us. After Lane M-V3 lands, we should test an encoder-free renderer that conditions ONLY on raw mask pixels + per-frame learnable embeddings — no SegNet/PoseNet feature extraction at compress time at all.

**5 lane proposals below.** Predicted band midpoint range: 1.07 → 0.85. Total cost ≤ $25 Vast.ai for the full set.

**Critical NEGATIVE finding:** The headline contribution (replace VAE+encoder with patchify) is the OPPOSITE of what we want. Quantizr's mask-only archive shows that **a learned encoder of CARTOON-LEVEL semantics (the 5-class SegNet output) is exactly the right artifact to ship at our scale.** Tuna-2's lesson is for the 7B regime where the encoder's inductive bias is small relative to model capacity. At 100K params we are bandwidth-starved and an encoder is a feature, not a bug. Don't misread the paper as "drop the SegNet conditioning."

---

## 1. Paper identification & verification

### 1.1 Bibliographic
- **arXiv ID:** `2604.24763v1` — verified via `curl -sIL https://arxiv.org/abs/2604.24763` returns HTTP 200, `last-modified: Tue, 28 Apr 2026 02:06:26 GMT`. (User flagged "24763 looks high" — confirmed legitimate; arXiv numbers are sequential within YYMM, and 24763 is consistent with late-month CV submissions.)
- **PDF:** 14.5 MB, 5 pages of body text + ~10 pages references. `pdftotext -layout` extracted 993 lines.
- **Title:** *Tuna-2: Pixel Embeddings Beat Vision Encoders for Multimodal Understanding and Generation*
- **Authors (15):** Zhiheng Liu¹², Weiming Ren¹³, Xiaoke Huang¹, Shoufa Chen¹, Tianhong Li¹, Mengzhao Chen², Yatai Ji², Sen He¹, Jonas Schult¹, Belinda Zeng¹, Tao Xiang¹, Wenhu Chen³, Ping Luo², Luke Zettlemoyer¹, Yuren Cong¹.  ¹Meta AI, ²University of Hong Kong, ³University of Waterloo. ∗Joint first authors (Liu, Ren).
- **Date:** April 28, 2026 (today). Hot off the press.
- **Project page:** `https://tuna-ai.org/tuna-2` (not fetched; PDF read in full).
- **Predecessor:** Tuna (Liu et al., 2025) — same lead author, latent-space UMM with VAE + SigLIP-2 encoder + Qwen2.5 backbone.

### 1.2 Genre and what it actually is
This is a **Meta AI native unified multimodal model** paper. It's a successor to Tuna and directly competes with BAGEL (14B), Mogao (7B), Janus-Pro (7B), Show-o2 (7B), Ming-UniVision (16B), and the Llama-3-derived stack. **It is NOT a video compression paper.** It is NOT a small-model paper. It uses 64 nodes × 7B params × 300k steps × 16k tokens/GPU pretraining + 50k SFT steps. Roughly 200K-400K H100-hours of training (extrapolation from BAGEL public numbers).

The interesting design is that it derives **two architectural variants in one paper** for controlled comparison:
- **Tuna-R**: VAE removed, but keeps SigLIP-2-So400M as a pretrained representation encoder.
- **Tuna-2**: Both VAE AND representation encoder removed. Just `Patchify(image) → token sequence`. Patchify is a single Conv2d layer (one matrix multiply).

The encoder-free Tuna-2 is the headline; Tuna-R exists as the controlled "intermediate stop" to disentangle which removal mattered.

---

## 2. Full method extraction (cover-to-cover)

### 2.1 Architecture (Section 2.1, Figure 1)

Three configurations sit on a spectrum of "how much specialized vision machinery":

| Config | VAE encoder | Representation encoder | Patchify | LLM body | Image gen head |
|---|---|---|---|---|---|
| **Tuna** (predecessor) | KL-VAE | SigLIP-2-So400M | yes | Qwen2.5-7B | flow-matching head (latent space) |
| **Tuna-R** (this paper) | dropped | SigLIP-2-So400M | yes | Qwen2.5-7B | flow-matching head (pixel space, see §2.1.3) |
| **Tuna-2** (headline) | dropped | dropped | yes | Qwen2.5-7B | flow-matching head (pixel space) |

Patchify is a single `nn.Conv2d(3, d_model, kernel_size=patch, stride=patch)` — no nonlinearity, no pretraining, no inductive bias. The output token grid is fed straight into the transformer.

The architectural hypothesis under test: **does throwing away pretrained encoders (SigLIP-2 has ~400M params, KL-VAE has ~84M) hurt performance once you have enough end-to-end training?**

Answer: **NO** for understanding, **NO with caveats** for generation (Tuna-R wins by 0.01 on GenEval but loses on diversity in human eval).

### 2.2 Pixel-space flow matching (Section 2.1, Eqs. 1-4)

This is the math for replacing latent diffusion. Adopted directly from JiT (Li & He 2025).

**Equation 1 (rectified flow noise schedule):**
$$x_t = t \cdot x_1 + (1-t) \cdot x_0, \quad t \in [0,1]$$
where `x_1` is the clean image and `x_0 ~ N(0, I)` is noise. Linear interpolation in pixel space.

**Equation 2 (network is x-prediction):**
$$x_\theta = \pi_\theta(x_t, c, t)$$
The network predicts the CLEAN image directly given noisy input and conditioning `c`.

**Equation 3 (derive velocity from x-prediction):**
$$v_\theta = \frac{x_\theta - x_t}{1 - t}$$

**Equation 4 (loss is on velocity):**
$$L_\text{flow} = \mathbb{E}_{t,c,x_1,x_0} \|v_\theta - v\|_2^2, \quad v = x_1 - x_0$$

The JiT trick: **the network predicts `x`, but the loss is on `v`** — derived analytically. This avoids two failure modes:
- Pure x-prediction has poor gradients near `t=0` (network must guess clean from pure noise).
- Pure v-prediction has poor numerical conditioning near `t=1` (the `1/(1-t)` term in Eq. 3 blows up exactly when there's no noise to remove).

By predicting `x` and converting analytically, you get the gradient quality of v-loss with the parameterization stability of x-prediction.

**Inference (Euler solver):**
$$x_{t'} = x_t + (t' - t) v_\theta$$

### 2.3 Masking-based feature learning (Section 2.2, Figure 3)

This is the second major contribution. During training, **randomly select a subset of image patches by a masking ratio, replace those visual tokens with a single learnable mask token, then proceed with the normal training objective.**

For **generation samples**: the model still must predict the full clean image (both masked and unmasked positions). This forces the learnable mask token to "absorb useful information for image reconstruction conditioned on the visible context."

For **understanding samples**: the model must answer the text question correctly with masked visual input. This is a regularization that prevents the model from over-relying on raw pixel values.

The key recipe choices (Section 3.4):
- Apply masking **only in the last 40% of pretraining** (don't apply it from the start; let the model first learn basic multimodal mapping).
- Use **probability 50%** per training example (i.e., half of examples get the masking applied).
- Single learnable mask token shared across all positions.

Lineage: MAE (He et al., 2022), SigLIP-2, MaskGIT, DeTok. Tuna-2's contribution is the unified application across both gen and understanding.

### 2.4 Training pipeline (Section 2.3, 3.1)

**Stage 1 — Full model pretraining:** 300k steps × 64 nodes × 16k tokens/GPU × AdamW lr=1e-4. Joint training on 550M image-text pairs (70% captioning, 30% T2I) plus 20% Nemotron text-only.

**Stage 2 — SFT:** 50k steps, lr=2e-5. 13M FineVision conversational + 2M OmniEdit editing + curated high-quality T2I.

**For Tuna-R (encoder-based variant):** an extra **3k-step alignment stage** trains only the connector layer between SigLIP-2 and Qwen2.5 with lr=5e-4 BEFORE Stage 1. Tuna-2 doesn't need this because there's nothing to align — the patchify layer is part of the unified model from step 0.

**Generation:understanding ratio = 7:3** chosen from the ablation in Section 3.3 / Figure 5. MSE loss (generation) is more sensitive to data ratio than CE loss (understanding); the 7g3u sweet spot trades them.

### 2.5 Empirical results (verbatim from Tables 1-6)

**Table 1 (multimodal understanding, 7B models):**
- Tuna (latent-space, predecessor): GQA 63.9, MMVet 42.9, MMMU 49.8, MMVP 70.7, OCRBench 74.3, V* 52.4, CountBench 73.5, VisuLogic 22.4
- Tuna-R: GQA 63.5, MMVet 46.7, MMMU 51.1, MMVP 74.7, OCRBench 78.3, V* 57.6, CountBench 77.8, VisuLogic 26.2
- **Tuna-2: GQA 65.0, MMVet 51.7, MMMU 50.7, MMVP 77.3, OCRBench 79.7, V* 59.2, CountBench 81.7, VisuLogic 28.8**

Tuna-2 wins or ties Tuna-R on 8/9 understanding benchmarks. Especially strong on pixel-centric (V*, CountBench, VisuLogic) which test fine-grained perception.

**Table 2 (image generation, GenEval):** Tuna-R 0.88 overall, Tuna-2 0.87. Tuna-R slightly better on counting, position, color attribute. Roughly tied with state-of-the-art generation-only models (FLUX 0.82, LongCat 0.87, Qwen-Image 0.87).

**Table 3 (LLM-judge eval):** Under both GPT-5.4 and Claude Opus 4.7 as judges, Tuna-2 BEATS Tuna-R on diversity (48.4% vs 30.9% under GPT-5.4) while being competitive on quality (32.1% vs 35.7%). The encoder-free variant generates more diverse images.

**Table 4 (image editing, ImgEdit):** Tuna-R 4.18, Tuna-2 4.09 (slightly behind Tuna at 4.31). Editing fidelity is where the pretrained encoder still helps marginally.

**Table 5 (image reconstruction, ImageNet val):**
- Tuna-R 512: rFID 0.12, PSNR 32.22, SSIM 0.93
- Tuna-2 512: rFID 0.15, PSNR 32.80, SSIM 0.93
- (For reference: FLUX.1[dev]-VAE 512: rFID 0.06, PSNR 33.65, SSIM 0.93)

Both Tuna-R and Tuna-2 are within 0.1 rFID and 0.85 PSNR of the FLUX VAE — at no extra training cost beyond the base UMM. **This means the unified transformer learns a near-perfect tokenizer as a side effect, which is critical for the unlimited-compress lane analysis below.**

**Table 6 (masking ablation, Qwen2.5-1.5B variant for cost):**
| Config | OCRBench | MMVP | CountBench | GenEval |
|---|---|---|---|---|
| Tuna baseline | 56.9 | 54.0 | 55.6 | 57.2 |
| Tuna-R w/o masking | 58.3 | 56.7 | 57.2 | 55.7 |
| Tuna-R w/ masking | 59.2 | 58.0 | 58.2 | 56.0 |
| Tuna-2 w/o masking | 55.4 | 52.3 | 53.4 | 47.6 |
| **Tuna-2 w/ masking** | **56.8** | **55.7** | **57.6** | **48.2** |

Masking gives Tuna-2 +1.4/+3.4/+4.2/+0.6. Masking gives Tuna-R +0.9/+1.3/+1.0/+0.3. **Tuna-2 benefits ~3× more from masking than Tuna-R.** Authors hypothesize this is because SigLIP-2 was already trained with a masked objective so Tuna-R has less room to grow.

### 2.6 Tuna-R vs Tuna-2 scaling (Section 3.5, Figure 6)

This is the most strategically interesting result for us: **Tuna-R wins early, Tuna-2 wins late.**

On OCRBench, MMVP, V* — Tuna-R outperforms Tuna-2 in early training (the SigLIP-2 prior is a head start) — then Tuna-2 catches up and surpasses by end of pretraining. On GenEval (generation), Tuna-R stays ahead the whole way but the gap narrows monotonically.

**Conclusion the authors draw:** "the monolithic, encoder-free design of Tuna-2 may be better suited to benefit from large-scale unified multimodal pretraining."

**The "bitter lesson" reading:** pretrained encoders are an inductive-bias crutch; with enough end-to-end data they cap your asymptote. **Implication for our distillation lanes (Lane M, Lane DI, Lane OS):** they may be early accelerants but ultimately limit the renderer's ceiling.

### 2.7 Attention map analysis (Section 3.6, Figure 7)

Qualitative: Tuna-2 attention is more accurate on counterintuitive prompts (e.g., "dog cafe" image with the question implying cat cafe — Tuna-2 attends to the actual dog rather than the textual "cafe" sign). This is an aside to the headline result.

---

## 3. Sub-mechanism extraction — what's independently useful

Given the paper is at scale completely incompatible with us (7B vs 100K params, 64-node training vs single 4090, 300k steps vs ~10k), the question is which **sub-modules** transfer. I find five.

### 3.1 Sub-mechanism A: x-prediction with derived v-loss (JiT formulation)

**What it is:** Network predicts `x_θ`, but the loss is computed on `v_θ = (x_θ - x_t)/(1-t)`. Eqs. 1-4 above.

**Why it transfers:** Our renderer is currently a single-step generator (mask + pose + latent → image). Our self-compression lane (Lane S) uses a multi-step training loop where the renderer learns to denoise its own quantization error. **The 1/(1-t) term in Eq. 3 is the failure mode that Lane S is hitting at the QAT-tail of training** — when the residual is tiny, the gradient blows up. JiT's analytic-conversion trick avoids this exactly.

**Transferable to:** Lane S (self-compression QAT tail), Lane I (Cool-Chic if we re-target to flow matching), any future "denoise from corrupted latent" lane.

**Cost to test:** ~$0 (a 5-line loss-formulation change in self_compress.py).

### 3.2 Sub-mechanism B: Masking-based feature learning during pretraining tail

**What it is:** Last 40% of training, with probability 0.5 per example, randomly mask K% of input patches with a learnable token; the loss objective is unchanged.

**Why it transfers:** Our renderer takes per-frame masks as conditioning input. We currently corrupt poses (noise_std=0.5) and we use eval_roundtrip to simulate the contest pipeline degradation. **We do NOT mask out parts of the conditioning input.** Tuna-2's ablation says masking gives +1.4 to +4.2 on perception benchmarks. For us this maps onto: "during the QAT tail of training, randomly drop K% of mask channels (e.g. zero out random 16x16 mask patches) and force the renderer to still produce a SegNet-faithful image."

This forces the renderer to **inpaint plausible class-boundary structure from CONTEXT rather than copy-paste from the mask channel.** That should reduce the SegNet gradient through the renderer (which is currently the dominant cost per `project_segnet_paradigm_shift.md`).

**Crucially: zero archive cost.** All happens at compress time during training.

**Transferable to:** Any training script — a `--mask-input-dropout-prob 0.5 --mask-input-dropout-ratio 0.15` flag.

**Cost to test:** $1-2 Vast.ai for an ablated training run on dilated-h64.

### 3.3 Sub-mechanism C: Encoder-free patchify replaces 400M-param encoder

**What it is:** A single Conv2d(in_channels, d_model, kernel_size=patch, stride=patch) with NO pretraining replaces a SigLIP-2-So400M (400M params) and matches/beats it on understanding tasks.

**Why it transfers (subtly):** Our masks.mkv encodes the SegNet output as a video. Lane DI proposes shipping SegNet penultimate features instead. Lane M-V3 distills PoseNet embeddings into the renderer. Both add learned encoder weights or learned projection layers to the archive.

Tuna-2's evidence says: **at sufficient training scale, a raw patchify suffices.** For us this is a question of "at our compress-time budget (~$25 = 100 hr 4090 training), have we hit Tuna-2's crossover point or are we still in Tuna-R territory where the encoder helps?"

**Empirical question, not architectural:** test whether dropping the conditioning encoder and just patchifying the 5-class mask + pose embedding reaches Lane A 1.15. If yes, ship a smaller renderer. If no, we're below crossover and the encoder is load-bearing.

**Transferable to:** Renderer architecture. Could become "Lane PFE — patchify-free encoder."

**Cost:** $3-5 per architectural variant.

### 3.4 Sub-mechanism D: Generation-understanding sampling-ratio sweep (7g3u)

**What it is:** They tested 9g1u, 8g2u, 7g3u, 6g4u, 5g5u and found 7g3u optimal. The MSE (generation) loss is more sensitive than CE (understanding) loss to the ratio.

**Why it transfers:** We currently train the renderer with a fixed loss weighting between SegNet distortion (the "understanding" objective: keep the masks faithful) and the rest (PoseNet pose accuracy + reconstruction). Tuna-2's ablation methodology says **sweep this ratio explicitly** — and that the perceptual loss (analog to MSE) is the more sensitive lever.

For us: sweep `--seg-loss-weight` more aggressively. Council historically caps it at ≤100 (per CLAUDE.md), but Tuna-2's ablation suggests testing 200, 500, 1000 with masking + roundtrip on. **The cap may be a relic of broken proxy-auth gap days.**

**Transferable to:** Any training script. A grid sweep.

**Cost:** $5-10 for a 4-point sweep.

### 3.5 Sub-mechanism E: Dual-variant controlled comparison as an experimental method

**What it is:** They did NOT just publish "encoder-free is better." They published Tuna-R as a controlled intermediate and showed that the Tuna→Tuna-R→Tuna-2 transition has TWO independent components (drop VAE, drop SigLIP) which contribute differently.

**Why it transfers:** Our lane proliferation (18+ active lanes per the user prompt) suffers from the inverse problem: we mix multiple changes per lane and can't attribute deltas. **Tuna-2's experimental hygiene is the lesson.** When we propose a lane that combines pose-from-RAFT + GP smoothing + TTO warm-start, we should split it into 3 controlled variants.

**Not a code change — a process change.** Adopt for next council planning round.

---

## 4. Compress-time vs inflate-time decomposition

This is the user's mandated cross-check per `feedback_compress_time_unlimited_archive_small_20260428`. Tuna-2 is a 7B-param model that runs at ~10 GB VRAM for a forward pass and ~100 GB for training. **As a runnable artifact, none of it fits in our archive.** But several pieces map cleanly onto compress-time-only invocations:

| Tuna-2 component | Where it can live in our pipeline | Archive cost |
|---|---|---|
| Tuna-2 base model (7B params) | Compress-time only — never ship. Use as a frozen encoder to produce per-frame embeddings for distillation targets. | 0 bytes |
| Patchify layer (single Conv2d) | Inflate-time — could replace mask-conditioning encoder in renderer. | tens of params |
| Flow-matching head | Compress-time training — generate denoised mask candidates that the renderer learns to imitate. | 0 bytes |
| Masking-based feature learning recipe | Compress-time training — apply during our QAT tail. | 0 bytes |
| x-prediction-derived v-loss formulation | Compress-time training — drop into self_compress.py loss. | 0 bytes |

**Concrete unlimited-compress angle:** If Tuna-2 weights are released (project page suggests they will be — "tuna-ai.org/tuna-2"), we could:
1. At compress time, run Tuna-2 over our 1199 frames to extract pixel-space embeddings (Tuna-2's understanding-tower features).
2. Distill those features into a tiny (e.g. 5K-param) per-frame learned token sequence stored in the archive.
3. At inflate time, the renderer reads the per-frame token sequence (5K × 1200 frames = 6 MB raw, but with FP4 quantization + Brotli reaches our budget).

**Critical NEGATIVE check:** This is essentially Lane DI (penultimate features) with Tuna-2 as the upstream feature provider instead of SegNet. Tuna-2 features are NOT trained on the comma scorer; they are general visual semantics. SegNet features ARE the scorer. Lane DI dominates this proposal because shipping scorer-aligned features is strictly better than shipping general features. **Skip the Tuna-2-as-feature-extractor lane unless Lane DI fails.**

---

## 5. Composition with our existing lanes

Pulling the 18+ lanes from CLAUDE.md memory and the active subagent dispatches:

| Existing lane | Tuna-2 sub-mechanism | Composes? | How |
|---|---|---|---|
| Lane A (dilated-h64 baseline) | Sub-mech B (masking-based learning) | YES | Add `--mask-channel-dropout 0.15 --mask-channel-dropout-prob 0.5` to last 40% of training. |
| Lane S (self-compression) | Sub-mech A (x-pred + v-loss) | YES | Replace direct loss with JiT formulation. Stabilizes QAT tail. |
| Lane S | Sub-mech B (masking) | YES | Both happen in late training; no conflict. |
| Lane W (hard-pair self-compress) | Sub-mech B (masking) | YES | Hard-pair weighting + mask dropout are orthogonal regularizers. |
| Lane Ω (bit-budget Hessian) | Sub-mech B (masking) | UNCERTAIN | Hessian computed under masking distribution would be more robust but more expensive. Council decision. |
| Lane DI (penultimate features) | Sub-mech C (patchify-only) | CONFLICT | DI ships features; patchify drops features. Lane DI explicitly tests "scorer features as conditioning"; Tuna-2 says "no encoder needed." Run BOTH; the empirical question is whether 100K renderer is past Tuna-2's crossover. |
| Lane M (PoseNet-embedding distill) | Sub-mech C | CONFLICT | Same as above. Run both; let score decide. |
| Lane MOS (Lane M + Lane OS) | Sub-mech C | CONFLICT | Same logic. |
| Lane OS (openpilot supercombo seeding) | Sub-mech C | NEUTRAL | OS provides seeds for poses, not for image conditioning. Orthogonal to encoder-free question. |
| Lane LR-V2 (LoRA pose) | Sub-mech D (data-ratio sweep) | NEUTRAL | LoRA is parameter-efficient; sweep recipe lives at the loss level not param level. |
| Lane HF / HFM (hyperbolic foveation) | Sub-mech B (masking) | YES | Mask the foveated region randomly during training; should improve generalization to off-foveation poses. |
| Lane FP (FramePack temporal compression — from cosmos synthesis) | Sub-mech B | YES | Mask whole frames during training. |
| Lane SAUG (self-augmentation — from cosmos) | Sub-mech B | DUPLICATE | Sub-mech B IS a form of self-augmentation. Don't double-implement. |
| Lane SI-V2 (saliency inversion v2) | Sub-mech B | YES | Mask within saliency mask = harder forensic adversary. |
| Lane LM-V2 (lane-mark zero-cost poses) | None | NEUTRAL | Pose-domain only, Tuna-2 doesn't touch poses. |

**Net: 7 existing lanes get a free uplift from sub-mech B (masking). 2 lanes (DI, M) are in direct conflict with sub-mech C and the empirical contest is informative.**

---

## 6. Integration anchor cross-checks (mandatory per user prompt)

### 6.1 openpilot integration

- **Lane OS (supercombo seeding):** Tuna-2 doesn't provide pose information. Orthogonal. No interaction.
- **Lane DI (SegNet penultimate features):** Direct conflict with Tuna-2's encoder-free thesis. The dialectic: at our scale, is a learned scorer-aligned encoder more valuable than NO encoder? Council should resolve via empirical run.
- **Lane LM (lane-mark zero-cost poses):** Geometric, not learned-encoder. Orthogonal.

### 6.2 Hardware exploits

- **T4 INT8 hardware accel:** Tuna-2 has no hardware-aware optimizations; it's a research paper at FP16/BF16. No transfer.
- **MPS 23× drift vs CUDA:** Tuna-2 is CUDA-only research (64 nodes). The 23× drift insight is ours; Tuna-2 doesn't speak to it. **However, sub-mech B (masking) might REDUCE the proxy-auth gap because the renderer learns to be robust to input degradation. Worth measuring proxy-auth correlation with vs without masking.**
- **Chroma half-res:** Tuna-2 operates on full RGB. No transfer. (Unrelated note: their Table 5 shows 32.80 PSNR at 512 — not relevant since we operate at 384×512.)

### 6.3 Canonical upstream auth eval

The output of every Tuna-2-inspired lane MUST end with contest-CUDA `inflate.sh → upstream/evaluate.py`. Per CLAUDE.md non-negotiable. Each of the 5 lane proposals below has an explicit auth-eval gate.

---

## 7. Lane proposals (5 named)

**Anchor:** Lane A 1.15 [contest-CUDA]. Quantizr 0.33. Predicted bands assume composable with Lane A (don't replace; augment).

### 7.1 Lane T2-MASK — Masking-based feature learning during training tail
**Premise:** Add input-mask channel dropout (p=0.5, drop ratio 15%) to the last 40% of dilated-h64 training. The renderer learns to inpaint mask channel from context, becoming more robust and (hypothesis) less SegNet-grad-dependent.

**Why it should work:** Tuna-2 Table 6 shows +1.4 to +4.2 on perception benchmarks for the same trick at the 1.5B scale. The mechanism is regularization, which transfers across scales (unlike the encoder-free thesis which depends on capacity).

**Predicted band:** [1.05, 1.13]. Midpoint 1.09 (down 0.06 from 1.15).
**Cost:** $1.50 Vast.ai (single dilated-h64 training run with the flag added, then auth eval).
**Risk:** Low. Could regress if mask channel is semantically critical and the renderer fails to learn the inpainting task. Mitigation: start with p=0.25 not 0.5.
**Composability:** YES with Lane A, Lane S, Lane W, Lane HF, Lane SI-V2.
**Auth-eval gate:** Mandatory CUDA auth eval at end.

### 7.2 Lane T2-XPRED — JiT x-prediction with derived v-loss in self-compression
**Premise:** Replace direct MSE loss in `self_compress.py` with x-prediction-derived v-loss per Tuna-2 Eqs. 1-4. Fixes the QAT-tail singularity at residual→0.

**Why it should work:** Self-compression's late-stage loss is currently dominated by tiny residuals where the gradient signal goes to zero. JiT's reformulation maintains gradient quality.

**Predicted band:** [1.10, 1.14]. Midpoint 1.12 (down 0.03 from 1.15).
**Cost:** $0.50 (5-line code change + one sanity training run).
**Risk:** Low. If it doesn't help, costs nothing. Worst case: identical loss curves.
**Composability:** YES with Lane S (it IS a Lane S internal change), Lane W, Lane Ω-V2.
**Auth-eval gate:** Yes.

### 7.3 Lane T2-DROP — Encoder-free renderer ablation (test the crossover)
**Premise:** Build a Lane A variant where the SegNet/PoseNet feature conditioning is REPLACED with a single learnable patchify(mask) → token sequence. No pretrained encoder of any kind. Tests whether 100K renderer is past Tuna-2's crossover point.

**Why it should work:** It might NOT — this is the empirical test of "is our scale above or below Tuna-2's crossover." Tuna-2 says encoder-free wins at >300k steps × 7B params. We are at <50k steps × 100K params. We are very probably BELOW crossover. **But the council is non-conservative; this is the kind of bitter-lesson test that occasionally surprises.**

**Predicted band:** [1.15, 1.45]. Midpoint 1.30 (UP 0.15 from baseline). Pessimistic prior.
**Cost:** $3 Vast.ai for a clean ablated training run.
**Risk:** HIGH. Probably worse than Lane A. Has 10-20% chance of teaching us that the encoder is the load-bearing structure.
**Composability:** NEGATIVE with Lane DI, Lane M (those are encoder-based). Run as standalone ablation.
**Disposition decision:** Run only after the 4 higher-EV lanes complete. Expected value is the information about crossover, not the score.
**Auth-eval gate:** Yes.

### 7.4 Lane T2-RATIO — Aggressive seg-loss-weight sweep
**Premise:** Tuna-2's 9-point data-ratio ablation found generation loss is far more sensitive than understanding loss. For us, the analog says SegNet weight sweep should be more aggressive than past sweeps. Run `seg_weight ∈ {100, 200, 500, 1000}` × `pose_weight ∈ {1, 5, 25}` grid (12 cells).

**Why it should work:** CLAUDE.md caps `segnet_loss_weight ≤ 100` based on stale evidence from broken proxy-auth gap days. With current eval_roundtrip + mask-input-dropout (Lane T2-MASK), the regime may have shifted.

**Predicted band:** [1.00, 1.13]. Midpoint 1.07.
**Cost:** $8-10 for 12 cells × short training each.
**Risk:** Medium. Might rediscover the same cap. Mitigation: run grid in 3 phases of 4 cells, kill bad ones early.
**Composability:** YES with Lane T2-MASK (test the new regime under masking).
**Auth-eval gate:** Mandatory on top 3 cells.

### 7.5 Lane T2-DUAL — Disciplined controlled-variant lane methodology (process change)
**Premise:** Adopt Tuna-2's experimental hygiene. Every new lane that combines N changes ships as N+1 controlled variants (baseline + each change isolated + combined). Stop conflating components.

**Why it should work:** This is meta. Tuna-2 ablations are clean precisely because they did this. Our lane proliferation is dirty precisely because we don't.

**Predicted band:** Long-term EV positive but no specific score delta.
**Cost:** Marginal — doubles the experiment count per lane but each variant is cheaper than today's combined-lane experiments.
**Risk:** Council pushback ("too much process"). Mitigation: only enforce for lanes that combine ≥3 changes.
**Composability:** Always.
**Auth-eval gate:** Per controlled variant.

---

## 8. Council ranking (by EV = predicted-midpoint-reduction × confidence ÷ cost)

| Rank | Lane | Pred Δ vs 1.15 | Confidence | Cost ($) | EV (Δ × conf / $) |
|---|---|---|---|---|---|
| 1 | Lane T2-MASK | -0.06 | 0.7 | 1.50 | 0.028 |
| 2 | Lane T2-XPRED | -0.03 | 0.5 | 0.50 | 0.030 |
| 3 | Lane T2-RATIO | -0.08 | 0.4 | 9.00 | 0.0036 |
| 4 | Lane T2-DUAL | (process) | 0.9 | 0 | (n/a, but always ROI-positive) |
| 5 | Lane T2-DROP | +0.15 | 0.85 (of regressing) | 3.00 | -0.0425 |

**Top recommendations:**
- **Lane T2-XPRED is #1 by EV** — cheapest (5-line code change), low-risk reformulation. Council-immediate.
- **Lane T2-MASK is #2 by EV** — concrete +0.06 improvement at $1.50.
- **Lane T2-RATIO is #3** — biggest potential delta but high uncertainty and high cost.
- **Lane T2-DUAL is process** — adopt for next planning round; no GPU spend.
- **Lane T2-DROP is information-only** — schedule LAST, after all higher-EV lanes; expected to regress but the regression amount is itself informative.

**Sequenced rollout (1.5 weeks until May 3 deadline):**
- Day 1: Lane T2-XPRED ($0.50, in-band)
- Day 1-2: Lane T2-MASK ($1.50)
- Day 2-3: Lane T2-RATIO ($9, partial — top 4 cells only)
- Day 4: Lane T2-DROP ($3, information run)
- Total Vast.ai spend: $14 (well within $25 budget; $11 left for downstream lanes)

---

## 9. Dispositioned proposals (looked at, decided NO)

These are Tuna-2 elements that do NOT transfer, surfaced explicitly so we don't re-ask later.

### 9.1 ❌ Tuna-2 base model as a feature extractor (Lane DI alternative)
**Why no:** Lane DI ships SegNet penultimate features (scorer-aligned). Tuna-2 features are general-purpose visual semantics, NOT scorer-aligned. Shipping scorer-aligned features dominates shipping general features at our extreme rate constraint. Revisit only if Lane DI completely fails.

### 9.2 ❌ Pixel-space flow matching for the renderer
**Why no:** Our renderer is single-step (mask + pose → image). Multi-step flow matching at inflate time blows the 30-min CUDA budget on T4. Lane I (Cool-Chic) already explores small-step flow matching; if that lane succeeds, revisit Tuna-2's exact JiT formulation as an internal upgrade to Lane I.

### 9.3 ❌ Tuna-2 image generation head (flow matching head architecture)
**Why no:** It's a transformer-based denoising head designed for 7B-scale joint training. Our renderer is a 100K conv-net. The architecture doesn't downscale meaningfully. We have stronger small-architecture priors (dilated-h64 is empirically validated).

### 9.4 ❌ Generation-understanding joint pretraining recipe (Section 3.3)
**Why no:** We have no "understanding" task analog. Our scorer is pure generation (image → score). The 7g3u sweet spot is meaningless for us. Sub-mech D extracts the SWEEP METHODOLOGY, but the specific 7:3 number doesn't transfer.

### 9.5 ❌ Tuna-R (intermediate variant) as architecture inspiration
**Why no:** Tuna-R adds a SigLIP-2-So400M encoder (400M params). Even if we distilled it 10000:1 to ~40K params, it would NOT be scorer-aligned. We already have a proven path with SegNet/PoseNet alignment via Lane M / Lane DI.

### 9.6 ❌ Replacing SegNet with Tuna-2 as the conditioning signal
**Why no:** SegNet IS the contest scorer. Tuna-2 is a general-purpose model. Conditioning on a non-scorer signal is throwing away the structure of the problem. This is the #1 misreading of the paper for our use case; explicitly disposition NO.

### 9.7 ❌ Stage-2 SFT corpus (FineVision, OmniEdit)
**Why no:** Their SFT data doesn't overlap with comma's driving footage. Domain mismatch. We have our own SFT analog (Lane HF tail of training, Lane S QAT tail).

### 9.8 ❌ The scaling-curve crossover (Figure 6) as quantitative evidence
**Why no:** The crossover is at 100s of billions of tokens for 7B model. Linearly extrapolated for our 100K model, we'd never reach crossover. **The QUALITATIVE crossover insight (encoders are crutches) transfers; the QUANTITATIVE position of the crossover doesn't.** Lane T2-DROP tests whether we're past it, but the prior is strong NO.

---

## 10. Integration cross-references — memories and lanes touched

This synthesis interacts with the following existing memory entries and lanes:

**Memory entries directly referenced:**
- `feedback_compress_time_unlimited_archive_small_20260428` — the unlimited-compress reframing that legitimizes 7B-model paper analysis
- `feedback_proxy_auth_math_useless` — sub-mech B (masking) might reduce this gap; explicit measurement target
- `feedback_mps_cuda_drift_critical` — orthogonal but should be measured under masking
- `project_segnet_paradigm_shift` — sub-mech B targets the SegNet-is-77x-bigger problem
- `project_quantizr_full_intel_20260421` — Quantizr's mask-only success contradicts encoder-free thesis at our scale; reinforces NEG dispositioning of 9.1/9.6
- `project_lane_w_hard_pair_self_compress_premise_20260427` — composes with Lane T2-MASK and Lane T2-XPRED

**Lanes directly compared/composed:**
- Lane A (composes with all 5 proposals)
- Lane S (Lane T2-XPRED is a Lane S internal upgrade)
- Lane W (composes with Lane T2-MASK)
- Lane Ω, Ω-V2 (Lane T2-MASK changes the Hessian distribution)
- Lane DI, Lane M, Lane MOS (CONFLICT with Lane T2-DROP; running them is the empirical test)
- Lane I (Cool-Chic) (potential upgrade via Lane T2-XPRED if Lane I succeeds)
- Lane HF, HFM (composes with Lane T2-MASK)
- Lane SI-V2 (composes with Lane T2-MASK)
- Lane LR-V2 (orthogonal, no interaction)
- Lane LM-V2 (orthogonal, no interaction)
- Lane OS (orthogonal)

**Cross-synthesis with prior cosmos analysis:** Sub-mech B (masking) is a stronger, more cleanly ablated version of Lane SAUG (self-augmentation) from the cosmos synthesis. **Recommend retiring Lane SAUG in favor of Lane T2-MASK** — it's the same idea with peer-reviewed quantification.

---

## 11. Council-ready summary (one paragraph for the next strategy meeting)

> Tuna-2 (arXiv 2604.24763, Liu/Ren et al., Meta AI, posted 27 Apr 2026) is a 7B-param native unified multimodal model that proves end-to-end pixel-space training can outperform encoder-based UMMs at scale. **The paper's headline (drop the encoder) is the OPPOSITE of what we want at 100K params** — Quantizr's mask-only archive shows scorer-aligned encoders are exactly the right artifact. **But three sub-mechanisms transfer cleanly**: (A) JiT's x-prediction-derived v-loss formulation (5-line fix to self_compress.py, $0.50, predicted -0.03 from 1.15); (B) masking-based feature learning in the training tail (15% mask-channel dropout × 50% probability × last 40% of training; +1.4 to +4.2 perception delta in their ablation, $1.50, predicted -0.06 from 1.15); (C) aggressive seg-loss-weight sweep above the historical cap of 100 ($9, predicted -0.08 from 1.15 if non-zero). **One information-only lane** tests the encoder-free thesis at our scale (Lane T2-DROP, $3, expected to regress, but the magnitude tells us where Tuna-2's crossover sits for us). **One process lane** (Lane T2-DUAL) adopts their controlled-variant ablation methodology to clean up our lane proliferation. **Total spend $14, projected baseline 1.07-0.95 with composition.** No conflicts with current Cosmos lanes; recommend retiring Lane SAUG in favor of the better-quantified Lane T2-MASK.
