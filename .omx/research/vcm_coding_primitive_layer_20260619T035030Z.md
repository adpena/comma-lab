---
title: "VCM / neural-compression coding-primitive layer — full-paper math + OSS code + our-archive mapping (the rate axis, the binding 62%)"
authority: "[research/advisory] — pointer UNMOVED 0.19110; $0; no GPU; no PR; NON-PROMOTABLE"
score_claim: false
promotable: false
date: 2026-06-19
provenance:
  - "Operator grant: FULL online research authority — 'read the FULL papers and find code samples; OSS and papers we can learn from and DRAW FROM and CONTRIBUTE TO.'"
  - "4 deep-research subagents (full-paper method sections + cloned-OSS code quotes), each cross-checked against in-repo modules."
  - "Bridge target grounded in: experiments/results/pr110_payload_entropy_recode_20260610/submission_dir/inflate.py (frontier inflate, CONFIRMED), .omx/research/frontier_rate_cut_vs_small_basis_anchoring_probe_20260617T155535Z.md (the MEASURED lossless-floor decomposition), .omx/research/SESSION_SYNTHESIS_SoT_20260617_20260618.md, .omx/research/eval_roundtrip_deep_math_pr95_handling_and_exploits_20260619.md."
cross_refs:
  - src/tac/balle_hyperprior_codec.py
  - src/tac/learnable_entropy_model.py
  - src/tac/entropy_bottleneck.py
  - src/tac/neural_weight_codec.py
  - src/tac/codec/charm_range_coder.py
  - src/tac/codec/factorized_hnerv_codec.py
  - src/tac/codec/cooperative_receiver/
  - src/tac/differentiable_eval_roundtrip.py
  - src/tac/quantization.py
---

# The VCM / neural-compression coding-primitive layer — what to DRAW FROM, what to CONTRIBUTE, and the $0 probes on OUR archive

`[research/advisory]` · pointer UNMOVED **0.19110** · $0 · no GPU · no PR · NON-PROMOTABLE. This is a means
(a primitive survey), not an end. Every primitive below is bridged to the one end that matters: a **lower
exact rate term** on our `archive.zip`. All BD-rate / bits-per-param numbers are **image/video-domain priors,
NOT contest measurements** — flagged inline.

## 0. THE GOVERNING FACT (corrects the prompt's premise; read first)

The prompt said "our frontier already uses **brotli/range-coded** sections … we need lower-entropy coding."
The first half is half-right and the conclusion needs sharpening, because of a **measured** fact:

- The binding section is the **decoder weights = 161,104 B = 91.0% of the 177,169 B payload** (62% of the
  score via `rate = 25·B/37,545,489 = 0.118`).
- It is **NOT brotli**. The frontier (`lane_pr110_payload_entropy_recode`) already replaced PR#101's
  split-brotli + raw-LZMA1 with **PR#112's `codec_ctx`: a per-tensor adaptive 256-ary `constriction`
  RANGE CODER**. (`inflate.py:8,19,40` — "the `codec_ctx` entropy coder is from public contest PR #112";
  `constriction` ships `stream` + `symbol`, the same lib the inflate imports.)
- That section is **MEASURED at its lossless entropy floor**: 7.999 bits/byte; order-0 floor 161,082 B vs
  actual 161,104 B (22 B slack); **every general-purpose recompress GROWS it**; the range coder already
  beats naive order-0. Re-derived per-param: 161,104·8 / 228,958 params ≈ **5.63 bits/param** — already
  well below the 8-bit nominal of the int8 symbols. (`frontier_rate_cut_*` probe, $0 forensic.)

> **The single insight that governs the entire survey:** a learned entropy model / coder swap that only
> **RE-PACKS the same int8 weight symbols is DEAD** — we are at the lossless floor; CompressAI / Cool-Chic /
> ELIC / MLIC++ priors fit to the *frozen* histogram cannot beat a range coder already at the symbol entropy.
> Our OWN in-repo `balle_hyperprior_codec.py` already documents a `STATIC_WINS_FALLBACK` regression where the
> learned prior LOSES to the static coder on real weight streams — the wall, empirically, in our own code.
>
> **The ONLY live lever changes the SYMBOLS.** Lower the entropy of the weights/latents *themselves* at
> iso-task-distortion (lower-bit / mixed-precision quantization that survives the frozen SegNet+PoseNet;
> RD-selected quantization step against the task; entropy-regularized training; learned rounding). This is
> exactly the additive-uniform-noise relaxation + RD-step search that the whole Ballé→ELIC→Cool-Chic/C3
> lineage was built for — used as a **training pressure on the weights**, not as a re-pack of the floor.

Each primitive below is tagged **LIVE** (symbol-changing) or **DEAD** (re-pack), and each carries the $0
probe that tests it on OUR exact archive.

---

## PART A — Learned entropy-model lineage (Ballé → ELIC → MLIC++) mapped to weight coding

The lineage's reusable asset for us is **not the prior** (re-pack, DEAD) but the **noise surrogate** that makes
the rate term `E[−log₂ p(ŷ)]` differentiable so you can train the quantizer/weights to be low-entropy.

### A1 — Ballé 2017 factorized prior + the additive-uniform-noise surrogate (arXiv 1611.01704)
- **Math (read from PDF, §):** `y=g_a(x)`, `q=round(y)`, `x̂=g_s(q)`; rate = discrete entropy `H[P_q]`. Hard
  round has zero gradient a.e. → train with **additive uniform noise** `ỹ=y+U(−½,½)`; `p_ỹ` = true density ∗
  unit box, evaluated at integers = the symbol's probability mass. Loss `R+λD`, `R=E[−log₂ p_ỹ(ỹ)]`.
- **Conditioning:** none (per-channel non-parametric CDF). **Side-info: ~0.**
- **OSS (CompressAI, BSD-3-Clause-Clear), `compressai/entropy_models/entropy_models.py`** — the noise/STE split:
  ```python
  def quantize(self, inputs, mode, means=None):
      if mode == "noise":
          noise = torch.empty_like(inputs).uniform_(-0.5, 0.5)
          return inputs + noise            # RATE term path (differentiable)
      outputs = inputs.clone()
      if means is not None: outputs -= means
      outputs = torch.round(outputs)       # DISTORTION/coding path (STE elsewhere)
      ...
  # factorized likelihood = sigmoid(cdf(x+0.5)) - sigmoid(cdf(x-0.5))
  ```
- **Our mapping:** re-pack a factorized prior onto frozen int8 weights = **DEAD** (it IS what our adaptive
  coder does). The noise surrogate, added to decoder training as `λ·E[−log₂ p_ỹ(w+noise)]`, pushes weights
  toward a peaky low-entropy marginal = **LIVE**.
- **Rate delta (image-domain prior):** re-pack ≈ 0%; joint-train ≈ −3% to −10% of 161KB **if** d_seg/d_pose hold.
- **$0 probe:** fit `tac.entropy_bottleneck` per weight tensor, arithmetic-code, compare to 161,104 — this
  tests RE-PACK → **will wall** (predict ±0.5%); its value is to *confirm the wall* and stop re-pack effort.
- **Do we have it?** **YES** — `src/tac/entropy_bottleneck.py` (per-channel logistic CDF) + CompressAI
  `EntropyBottleneck`. Not the gap.

### A2 — Ballé 2018 scale hyperprior (arXiv 1802.01436)
- **Math:** second latent `z=h_a(y)`; `σ̂=h_s(ẑ)`; `p(ŷ|ẑ)=∏ [Φ((ŷ+½)/σ̂)−Φ((ŷ−½)/σ̂)]` (Gaussian∗uniform);
  `z` uses the factorized prior. Total rate = `E[−log₂ p(ŷ|ẑ)] + E[−log₂ p(ẑ)]`.
- **Conditioning:** the hyper-latent `z`. **Side-info = the entropy-coded `ẑ` bytes** (the byte we'd ADD to
  the archive — this is the cost half of the trade for us).
- **OSS (CompressAI `GaussianConditional`):** discretized-Gaussian likelihood via `erfc`:
  ```python
  def _standardized_cumulative(self, inputs):        # Φ via erfc (max precision)
      return 0.5 * torch.erfc(-(2 ** -0.5) * inputs)
  def _likelihood(self, inputs, scales, means=None):
      v = torch.abs((inputs - means) if means is not None else inputs)
      scales = self.lower_bound_scale(scales)
      return (self._standardized_cumulative(( 0.5 - v)/scales)
            - self._standardized_cumulative((-0.5 - v)/scales))
  ```
  and the bpp loss (`compressai/losses/rate_distortion.py`): `bpp = Σ log(likelihoods)/(−log2·N)`.
- **Our mapping:** per-tensor *scale* hyperprior over frozen weights ≈ our existing per-tensor fp16 scales +
  adaptive histogram → re-pack **DEAD**; region-adaptive quant step under a learned conditional, JOINTLY
  trained → **LIVE** (this is the "per-band quantization" lever the in-repo Daubechies analysis flagged).
- **Rate delta:** −2% to −8% of 161KB **net of side-info**, joint-train only; break-even is sharp (if `ẑ`+`h_s`
  cost > the conditional saving it loses — exactly our `STATIC_WINS_FALLBACK`).
- **$0 probe (the canonical one in the prompt):** fit per-tensor `GaussianConditional` / `BalleHyperpriorCodec`
  mode=1 to frontier weight histograms; arithmetic-code; compare to 161,104. **RE-PACK → walls** (predict net
  within ±1%, likely a LOSS after side-info). LIVE needs a retrain with `rate()` in the loss.
- **Do we have it?** **YES, twice** — `src/tac/balle_hyperprior_codec.py` (full Ballé mode=1 + Hotz-LITE
  chunked-static mode=0) + CompressAI `ScaleHyperprior`/`MeanScaleHyperprior`. **The static/Hotz-LITE Ballé is
  what we have; the jointly-trained-with-the-decoder version is what we DON'T.**

### A3 — Minnen 2018 joint autoregressive + hierarchical (arXiv 1809.02736)
- **Math:** add a masked-conv AR context `φ_i=g_cm(ŷ_<i)`; entropy-params net fuses hyper `ψ` + `φ_i` →
  per-latent `(μ_i,σ_i)`; same discretized-Gaussian likelihood, now mean+scale, context-conditioned. **8.4%
  smaller than BPG / 15.8% smaller than hyperprior-only** (image-domain).
- **OSS (CompressAI `models/google.py` + `MaskedConv2d`):**
  ```python
  self.context_prediction = MaskedConv2d(M, 2*M, kernel_size=5, padding=2)
  ctx = self.context_prediction(y_hat)
  scales, means = self.entropy_parameters(torch.cat((params, ctx), 1)).chunk(2, 1)
  ```
- **Our mapping:** a learned AR context over already-decoded weight elements → still bounded by the histogram
  floor (**DEAD as re-pack**); JOINT-train AR-rate is the strongest prior but the **serial masked-conv decoder
  is a 30-min-numpy-inflate RUNTIME RISK** over 228K weights. Subsumed by ELIC (parallel) below.
- **$0 probe:** byte-ESTIMATE only (`−Σ log₂ p` with masked-conv context over frozen weights), no real serial
  coder. **RE-PACK → walls;** also flags the decode-latency blocker.
- **Do we have it?** Partial — CompressAI `JointAutoregressiveHierarchicalPriors` + `MaskedConv2d` installed;
  `charm_range_coder.py` is the channel-AR (ChARM) coder. We do NOT have (and should NOT build) the serial
  spatial-AR over weights.

### A4 — Cheng 2020 discretized Gaussian MIXTURE + attention (arXiv 2001.01568)
- **Math:** `p(ŷ_i)=Σ_k w_i^k[Φ((ŷ_i+½−μ_i^k)/σ_i^k)−Φ((ŷ_i−½−μ_i^k)/σ_i^k)]`, `Σ_k w=1` (K=3). VVC/VTM-class on
  PSNR.
- **OSS (CompressAI `GaussianMixtureConditional._likelihood`):** sums K shifted Gaussian-conditional terms.
- **Our mapping — the best LIKELIHOOD family for weights:** int8 weight histograms are often **multimodal**
  (clusters around a few quant centroids, especially post-QAT) — a GMM models that where a single Gaussian
  smears it. Re-pack still bounded by the histogram (**DEAD**); a GMM *rate term in training* = differentiable
  soft-clustering pressure → collapse 256 levels toward 16–32 centroids at iso-quality = **LIVE & synergistic
  with QAT**.
- **Rate delta:** joint-train −5% to −20% of 161KB **if** task quality survives (in-repo PACT-NeRV memo: fp4
  per-channel hit −71.7% bytes but at rel_l2 0.097 — too lossy *without* QAT; a GMM-rate-trained intermediate
  is the sweet spot).
- **$0 probe:** **per-tensor effective-centroid count** on frontier int8 weights (how clusterable each
  histogram is) → predicts the GMM/QAT joint-train ceiling without a retrain; cheap acquisition signal for λ, K.
- **Do we have it?** **YES** — `learnable_entropy_model.py` `HyperDecoderConfig(mode="mixture")` + CompressAI
  `GaussianMixtureConditional`. The likelihood family is covered; the **joint-train-with-quantizer loop is the gap.**

### A5 — He 2021 checkerboard + He 2022 ELIC space-channel context (arXiv 2103.15306, 2203.10886)
- **Math:** checkerboard = 2-pass (anchors from hyperprior, non-anchors from decoded anchors via
  `CheckerboardMaskedConv2d`) → **>40× decode speedup at ~same RD**. ELIC = **Space-Channel Context (SCCTX)**:
  uneven channel groups `[16,16,32,64,…]` coded group-by-group, each conditioned on (a) hyperprior, (b)
  **decoded channel groups** (channel context `g_ch`), (c) **checkerboard spatial** within group (`g_sp`).
  Param-agg net → `(μ,σ)`. **SOTA RD with PARALLEL decode.**
- **OSS (CompressAI `models/sensetime.py` `Elic2022Chandelier`):** `ChannelGroupsLatentCodec` +
  `CheckerboardLatentCodec` + `HyperpriorLatentCodec` (BSD-3-Clause-Clear, lift-ready).
- **Our mapping — THE GENUINE GAP:** our decoder is conv tensors with **natural channel structure** →
  ELIC's *channel-group* context maps directly onto coding weight-tensor channels conditioned on
  already-decoded channels (CONV4 storage-perms already create the spatial locality the checkerboard wants).
  A jointly-trained ELIC-style space-channel prior over the weights is the strongest **practical
  (parallel-decodable)** model in the lineage. Re-pack **DEAD**; joint-train **LIVE & best-in-class**, and the
  only one whose decode stays parallel (fits the 30-min budget).
- **Rate delta:** joint-train −5% to −15% of 161KB net, **parallel decode** (unique advantage).
- **$0 step (design smoke, no GPU):** restructure `Elic2022Chandelier`'s `ChannelGroupsLatentCodec` to accept
  our flattened weight tensors as the latent `y`; correctness-smoke the entropy-param plumbing on
  weight-shaped tensors. No score claim — preps the retrain.
- **Do we have it?** **NO** — this is the one component to LIFT. (CompressAI ships `Elic2022Chandelier`,
  `Cheng2020AnchorCheckerboard` under BSD-3-Clause-Clear.)

### A6 — MLIC / MLIC++ (arXiv 2307.15421)
- **Math:** MEM++ = channel + local-checkerboard-attention (linear) + **global linear-attention over decoded
  slices**; **−13.39% BD-rate vs VTM-17.0 Intra on Kodak** (current practical SOTA). **License Apache-2.0.**
- **Our mapping:** global attention → cross-tensor weight correlations (e.g. depthwise↔pointwise redundancy)
  but the **heaviest decode** → likely incompatible with numpy-portable 30-min inflate. Re-pack DEAD; treat
  global-context as **research-only reference**; ELIC is the better effort/reward point for us.
- **Do we have it?** NO (not in CompressAI; own Apache-2.0 repo). Draw-from for the channel-slice idea (in
  ELIC already); defer global-attention.

---

## PART B — Entropy CODERS + network-WEIGHT / INR entropy coding (the binding axis)

### B0 — The coders are SOLVED; do not reimplement
- **constriction** (MIT OR Apache-2.0 OR Boost-1.0; `bamler-lab/constriction`) — **the coder our inflate
  already uses** (PR#112 `codec_ctx`). API: `stream.stack.AnsCoder` (rANS, LIFO), `stream.queue.RangeEncoder/
  RangeDecoder` (FIFO), models `QuantizedGaussian(min,max,mean,std)`, `Categorical(probs)`, and
  `CustomModel`/`ScipyModel` for an arbitrary CDF via *exactly-invertible fixed-point* arithmetic
  (deterministic across machines = byte-exact inflate). Symbols `np.int32`, **per-symbol vectorized means/stds**
  (exactly what a per-weight Gaussian/Laplace prior needs). **Anything in this report encodes with this lib —
  no new dependency.**
  ```python
  import constriction, numpy as np
  msg = np.array([6,10,-4,2,5], dtype=np.int32)
  em  = constriction.stream.model.QuantizedGaussian(-50, 50, 3.2, 9.6)
  enc = constriction.stream.queue.RangeEncoder(); enc.encode(msg, em)
  dec = constriction.stream.queue.RangeDecoder(enc.get_compressed()); out = dec.decode(em, 5)
  ```
- **rANS/tANS** (Duda 2013, ryg_rans) — near-Shannon, fast; rANS=stack, range=queue. constriction gives both.
  Background only.
- **CompressAI rANS backend** (vendored ryg_rans, BSD-3) — reference for mapping a learned Gaussian-conditional
  → CDF table; we don't need its C++ (constriction is cleaner + numpy-portable).
- **In-repo, do NOT rebuild:** `src/tac/lossless/range_coder.py` (integer range coder),
  `src/tac/codec/charm_range_coder.py` (per-symbol Gaussian-PMF range coder — **swapping the Gaussian PMF for
  a Laplace PMF is a one-function change**, numpy-portable), `src/tac/pr103_arithmetic_codec.py`.

### B1 — HiNeRV (Kwan, NeurIPS 2023; arXiv 2306.09818) — MOST directly relevant; MIT
The closest published thing to our archive (per-video INR, weights are the bitstream). VERBATIM from the repo:
- **Quant (`compression/quant_utils.py`):** symmetric **per-channel** STE, **fixed** scale (NOT LSQ):
  ```python
  def _quantize_ste(x, n, axis=None):
      x_max = abs(x).max(dim=axis, keepdim=True)[0] if axis is not None else abs(x).max()
      x_scale = 2*x_max/(2.**n - 1.) + 1e-6
      x_q = _ste(x / x_scale).clamp(-2**(n-1), 2**(n-1)-1)
      return x_q, x_scale          # dequant: x_q * x_scale
  ```
  Applied as **QAT** (STE in the training loop); README headline bit-width = **6-bit**.
- **Entropy (`compression/codec_utils.py`):** prior = **per-tensor EMPIRICAL HISTOGRAM** (unique→counts→cumsum
  CDF) coded with **torchac** arithmetic coding; model size = `len(byte_stream)`.
- **Result:** 72.3% bitrate saving over HNeRV, 43.4% over DCVC (UVG PSNR).
- **Our mapping — the single most important comparison:** ours = per-tensor **int8 (8-bit)** + fp16 scale +
  constriction; HiNeRV = **per-channel 6-bit QAT** + histogram + torchac. The coders are equivalent at the
  floor; the **LIVE deltas are bit-width 8→6, per-channel scales, and QAT-through-the-task-loss.** Our int8 was
  a hardware/range-coding convenience; HiNeRV proves 6-bit per-channel QAT is the SOTA INR weight code.
- **$0/low-cost probe:** 6-bit per-channel QAT fine-tune through the FROZEN SegNet+PoseNet (our
  `FakeQuantSTE`/`LSQScale` + MPS-train/CPU-authority split), re-export, byte-close, re-measure. Needs a short
  QAT fit (local, no paid GPU).
- **Do we have it?** Quant primitives YES (`quantization.py`); the **6-bit per-channel QAT through the frozen
  scorer + re-export is the missing piece.**

### B2 — NeuroQuant (arXiv 2502.11729, 2025) — beats HiNeRV-QAT at low bit-width; PTQ (no retrain)
- **Method:** reframes variable-rate INR-VC as **mixed-precision PTQ**: network-wise calibration +
  channel-wise quantization, models inter-layer dependencies. **Down to INT2 with minimal loss; >3 dB over
  HiNeRV's QAT at low bit-widths; up to 8× faster encoding. POST-TRAINING (no per-rate retrain).**
- **Our mapping — strongest evidence sub-6-bit mixed-precision is feasible WITHOUT a retrain.** Maps onto our
  28-tensor schema: allocate fewer bits to insensitive tensors (we have the master-gradient ledger +
  sensitivity map). **LIVE, symbol-changing, PTQ-only.**
- **$0 probe:** sensitivity-aware mixed-precision PTQ on the EXISTING frontier weights (4–6 bits per tensor by
  SegNet/PoseNet sensitivity), re-encode with constriction, byte-close, re-measure. **No GPU retrain.**
- **Do we have it?** NO mixed-precision-per-tensor calibrator — strong build target.

### B3 — AdaRound / Strümpler INR-compression (arXiv 2112.04267) — the safest LIVE lever
- **Method:** uniform PTQ at a set bit-width + **AdaRound** (learned per-weight rounding direction up/down to
  minimize task loss, not nearest) + optional QAT; meta-learned init; arithmetic/brotli coding.
- **Our mapping — AdaRound is genuinely novel for us, LIVE, low-risk, decode-UNCHANGED:** learned rounding on
  our int8 weights recovers distortion at the SAME bytes, OR allows 1 bit lower at iso-distortion. We do plain
  `round` today. **Zero inflate cost, zero new dependency** (still int8 + scale at decode).
- **$0/low-cost probe (W2):** apply AdaRound (learned rounding under a boundary-weighted d_seg surrogate +
  pose-MSE, frozen scorers) to the frontier int8 weights; measure d_seg/d_pose at iso-bytes; then drop 1 bit
  and test recovery. Local CPU/MPS short fit.
- **Do we have it?** NO AdaRound. Reference: AIMET (Apache-2.0) via the Strümpler repo (repo license unverified
  — flag before lifting). MAML-init = long-horizon class-shift, defer.

### B4 — Cool-Chic (ICCV 2023; arXiv 2212.05458; BSD-3) + C3 (CVPR 2024; arXiv 2312.02753; Apache-2.0)
The per-instance overfitted codecs — structurally our closest relatives (tiny net + latents, no encoder,
whole thing entropy-coded). The transferable lever is the **weight code**, and it reframes our "lossless floor":
- **5.63 bits/param is the floor of the int8 SYMBOLS we chose — NOT the floor of the weights.** Cool-Chic/C3
  do NOT fix int8; they **RD-SEARCH a per-module quantization STEP Δ (power of 2)** against the *actual decoded
  distortion + total rate*, then code the integers under an explicit parametric prior:
  - **Cool-Chic (`nnquant/quantizemodel.py`, `quantstep.py`, `expgolomb.py`):** per-module
    `(Δ_w,Δ_b)=argmin ‖x−x̂‖² + λ(R(x̂)+R_NN)`; `Δ_w ∈ 2^linspace(-12,0,13)`; integers Exp-Golomb-coded
    (exponent ALSO RD-searched). The **ARM** over latents = a small MLP → Laplace `(μ,b)`, rate `−log₂ ∫_{ŷ±0.5}
    L(t;μ,b)dt`. (`arm.py`; `b=exp(clamp(log_scale−4,−5,5))`.)
  - **C3 (`model/model_coding.py`, `model/laplace.py`):** `quantize_at_step(x,q)=round(x/q)`; per-tensor weight
    rate via a **zero-mean Laplace(0, std/√2) integrated-CDF code** (clip 32 b/sym); latents via a **conv masked
    Laplace ARM** (more parallel than Cool-Chic's per-pixel MLP) + optional previous-grid conditioning +
    Kumaraswamy-noise / soft-round-temperature-anneal training.
- **Our mapping:**
  - Weight axis: the **Δ-search is LIVE** (can pick <256 levels where d_seg/d_pose tolerate it). Exp-Golomb /
    Laplace-CDF as a *coder swap* is **DEAD** (constriction range coder ≥ Exp-Golomb ≥ a parametric prior the
    histogram doesn't match). The win is the *symbol choice*, not the coder.
  - Latent axis: a conv Laplace ARM over our temporal-delta latents (15KB) = **LIVE** but small ceiling.
  - **Strategic signal:** C3 says "cost is dominated by the LATENTS" and keeps the net tiny — the **opposite**
    of our 91%-weights archive. Whether a latent-heavy Cool-Chic representation of our 1200 frames beats our
    weight-heavy HNeRV is a substrate-class-shift question (the A1-memo's deferred V4), NOT a re-pack.
- **Decode feasibility (30-min numpy):** Exp-Golomb ✅; Laplace-CDF ✅ (swap the Gaussian PMF in
  `charm_range_coder.py` for Laplace); **the sequential ARM as a full decoder-replacement is the budget RISK**
  (per-pixel MLP over 1200 frames) — C3's conv ARM is more parallel. Latent-only ARM is fine; full
  decoder-replacement must validate decode wall-clock first.
- **$0 probe (W1):** on our exact decoder weights, per tensor: (a) compute C3 `laplace_rate` + Cool-Chic
  Exp-Golomb bits at the *current* int8 grid → compare to 5.63 bits/param AND to the per-tensor
  empirical-histogram range bound (what we already pay); (b) sweep Δ∈{2^k}, at each compute integer rate +
  the induced weight error + (at a few Δ) a real HNeRV decode-replay for **task** d_seg/d_pose. **Falsification:
  if no Δ beats 5.63 at iso-d_seg, the lossless-floor claim is vindicated and this route is DEAD.** Pure numpy.
- **Do we have it?** Cool-Chic referenced in-repo as a reopen candidate ("28.8× faster now"); the **Δ-search +
  task-RD weight code is the gap.** REC/bits-back (RECOMBINER, arXiv 2309.17182) is **NOT 30-min-numpy-feasible**
  (decode cost exponential in per-block KL) — DEFER/skip.

### B5 — Deep Compression (Han 2016) + DeepCABAC (arXiv 1907.11900) + NWC (arXiv 2510.11234)
- **Deep Compression:** prune → k-means weight clustering (4–5-bit indices) → Huffman. Clustering through the
  **task loss** = LIVE soft-VQ; the Huffman/clustering as re-pack = DEAD. Mostly historical.
- **DeepCABAC:** RD-quantize each weight (minimize rate AND task distortion jointly) + context-adaptive binary
  AC. The **RD-quant objective is the LIVE lesson** (= our task-RD weight quant); the CABAC coder is re-pack
  (DEAD, we range-code). Best for sparse weights (we're not sparse).
- **NWC (arXiv 2510.11234):** autoencoder weight codec, column-wise chunk+norm, importance-aware loss,
  output-guided error compensation; 4–6 b/w ≈ FP16 on LLM matrices. **CAUTION (NO-FAKE):** those wins are on
  huge LLM matrices with massive redundancy; **our largest tensor is 1728×28, total 228K params** — a codec
  that must SHIP its own decoder inside the archive can cost more than it saves at this scale. Our
  `src/tac/neural_weight_codec.py` VQ scaffold's "−126KB" docstring is **[predicted]**, unproven at our scale.
- **Do we have it?** NWC scaffold YES; RD-quant objective / context-adaptive coder / column-wise norm /
  error-comp NO.

---

## PART C — Task-loss-through-FROZEN-model (VCM/ICM): our contest IS this field

The contest is *literally* Video/Image Coding for Machines: the "machine" is the frozen SegNet (argmax-flip
`d_seg`) + PoseNet (`d_pose`), human PSNR is non-authority, `S=100·d_seg+√(10·d_pose)+25·B/N` is a task-RD
Lagrangian. The field validates our core mechanism; the genuine gaps are auxiliary *training* techniques.

### C1 — Singh 2020 "End-to-end Learning of Compressible Features" (arXiv 2007.11797) — our objective, verbatim
- **Eq. 6:** `θ*,φ* = argmin Σ [L(ŷ,y)]_distortion + λ[−log₂ p(ẑ;φ)]_rate`, `ẑ=⌊f_z(x)⌉`. Task loss *replaces*
  distortion. Two-quantizer split: noise for rate, STE-round for distortion.
- **Our mapping — HAVE-IT** (this is our `S`). **NEW sub-point:** Singh uses a *differentiable learned* rate
  `−log₂ p(ẑ;φ)` co-optimized; our `B` is the realized post-hoc byte count (non-differentiable). A
  differentiable rate surrogate would let us co-train bytes vs d_seg/d_pose.

### C2 — Iino 2024 ICM auxiliary loss (arXiv 2402.08267) — the deep-frozen-model gradient pathology
- **Taxonomy:** feature distillation (Eq. 3 `L=R+λΣ D(f_i,f̂_i)`), direct task loss (Eq. 4), **auxiliary
  lightweight surrogate head (Eq. 5 `L=R+λ{E(y,ŷ)+αE(y,ŷ_aux)}`)**, where `ŷ_aux` is a small head on an EARLY
  feature giving the encoder a SHORT gradient path. **Finding:** deep frozen model → hard shallow-layer
  gradient; the auxiliary head fixes it (BD-rate −27.7% det / −20.3% seg). Yamazaki trick: use the frozen
  model's OWN output as the target (= our d_seg-vs-GT-SegNet-argmax).
- **Our mapping — NEW, high value.** SegNet (deep EfficientNet-B2 Unet) + PoseNet (FastViT-T12) → backprop from
  the final argmax/pose head to our tiny INR decoder is exactly the "deep model → hard shallow gradient"
  pathology. A lightweight learned surrogate head on an EARLY scorer feature = a short, well-conditioned
  gradient. **Directly attacks the d_seg plateau** (sister to the in-repo EMA-shadow-lag finding that
  intermediate supervision breaks plateaus).
- **$0 probe:** add `λ_feat·Σ_l MSE(SegNet_l(GT), SegNet_l(recon))` for early/mid `l` (forward-only on the
  frozen SegNet; gradients → our decoder); A/B vs the CE/margin surrogate alone.

### C3 — Choi & Bajić "Scalable Coding for Humans and Machines" (TIP 2022; arXiv 2107.08373) — feature distillation as distortion
- **Eq. 6 (most-cited mechanism):** `D = MSE(X,X̂) + Σ_j γ_j·MSE(F_j, F̂_j)` over *intermediate* task-DNN
  feature tensors — "steers task-relevant info into the latent." DPI (Eq. 3) `I(ŷ;X̂)≥I(ŷ;F)≥I(ŷ;T)` is the
  information-theoretic license for coding less than pixels (= our eval-roundtrip null-space).
- **Our mapping — feature-domain distillation is the highest-value transferable technique we don't do.** We
  supervise the FINAL d_seg argmax + final 6-dim pose; we do NOT add `γ·MSE` on *intermediate* SegNet/PoseNet
  features. Pick layers by the master-gradient ledger's per-layer d_seg/d_pose sensitivity.
- **$0 probe:** multi-scale `Σ_l γ_l·MSE(SegNet_l(GT),SegNet_l(recon)) + Σ_l γ_l'·MSE(PoseNet_l(GT),PoseNet_l(recon))`.

### C4 — Sandwiched compression (arXiv 2402.05887) — differentiable codec proxy
- **Mechanism:** trainable neural pre/post around a FROZEN codec, trained via STE quant + a **closed-form
  differentiable rate proxy** `R = a·Σ log(1+|e|/Δ)` (calibrate `a` to real bytes).
- **Our mapping:** STE + frozen-block sandwich = HAVE-IT (frozen scorer + diff eval-roundtrip + diff YUV6). The
  **closed-form differentiable rate proxy is NEW & cheap** — fit `B̂=a·Σ log(1+|Δlatent|/Δ)` to realized
  archive bytes across the master-gradient ledger; if R²>0.9, replace the rate-sweep with a co-optimizable
  differentiable byte term.

### C5 — ProxIQA / learned surrogate of a non-differentiable metric (arXiv 2305.02024 survey)
- **Mechanism:** train a small differentiable net to mimic a non-differentiable objective (VMAF / here:
  argmax-flip `d_seg`), backprop through the surrogate (ProxIQA: −20% bitrate vs MSE).
- **Our mapping — partial-NEW, high value:** our `d_seg` is argmax-flip (non-diff), surrogated today by a
  HAND-DESIGNED CE/margin. A LEARNED `d̂_seg(recon,GT)≈flip-rate` would learn the boundary-margin geometry the
  contest actually rewards (our memory: d_seg concentrates at SegNet decision boundaries).
- **$0 probe:** collect (recon,GT,true-d_seg) triples from existing archives → train tiny `d̂_seg` → A/B vs CE/margin.

### C6 — Saliency/Jacobian bit allocation (Grad-CAM ROI VCM arXiv 2203.05944; IDSE-RDO arXiv 2504.02216)
- **Mechanism:** backprop the frozen task model → importance map / **Jacobian-quadratic distortion
  `(recon−GT)ᵀ JᵀJ (recon−GT)`** (IDSE) → bits where the model is most sensitive.
- **Our mapping — partial-HAVE-IT (per-byte master-gradient ledger), refinement-NEW:** the IDSE quadratic is
  the principled per-pixel refinement of our per-byte linear sensitivity, and it IS the "d_seg concentrates at
  the boundary" geometry. **$0 probe:** SegNet/PoseNet input-Jacobian on a few frames → IDSE recon weight in
  training; A/B vs CE/margin.
- **In-repo sister:** `src/tac/codec/cooperative_receiver/` (Atick-Redlich) + `jscc/` are the "scorer-as-receiver"
  framing — the same family under a different name; our novelty is the *multi-task frozen scorer* + per-video INR.

---

## DRAW-FROM list (lift / adapt — licenses verified)

| Source | License | Lift/adapt | numpy-decode in 30 min? |
|---|---|---|---|
| **CompressAI** `InterDigitalInc/CompressAI` | **BSD-3-Clause-Clear** ✅ | `EntropyModel.quantize` (noise/STE split — the key primitive); `GaussianConditional`; `GaussianMixtureConditional` (multimodal weight histograms); **`Elic2022Chandelier` / `ChannelGroupsLatentCodec` / `CheckerboardLatentCodec` (THE genuine gap)**; `RateDistortionLoss` (`λ·255²·mse + bpp` template); `MaskedConv2d` | likelihoods ✅; ELIC parallel ✅; serial AR ⚠️ |
| **constriction** `bamler-lab/constriction` | **MIT/Apache/Boost** ✅ | already imported — the encoder for EVERY symbol-changing lever (`QuantizedGaussian`, `Categorical`, `CustomModel`); swap Gaussian→Laplace for C3-style weight codes | ✅ (it's the inflate coder) |
| **HiNeRV** `hmkx/HiNeRV` | **MIT** ✅ | `_quantize_ste` (per-channel 6-bit symmetric STE) + `_arithmetic_encoding` (empirical-histogram CDF) — cleanest low-bit INR weight pipeline | ✅ |
| **Cool-Chic** `Orange-OpenSource/Cool-Chic` | **BSD-3** ✅ | per-module RD Δ-search (`quantizemodel.py`/`quantstep.py`); Exp-Golomb bit-counter (`expgolomb.py`); Laplace ARM math (`arm.py`) | Δ/Golomb ✅; full ARM decoder ⚠️ |
| **C3** `google-deepmind/c3_neural_compression` | **Apache-2.0** ✅ | `laplace_rate` per-tensor weight code; conv masked Laplace ARM; previous-grid conditioning; Kumaraswamy/soft-round training | math ✅ numpy-portable; code is JAX (reimplement) |
| **CompressAI-Vision** `InterDigitalInc/CompressAI-Vision` | BSD-3 | VCM eval harness, split/remote-inference scenarios (note `sic_sfu2022.py` multi-task = `NotImplementedError`) | n/a (harness) |
| **MLIC/MLIC++** `JiangWeibeta/MLIC` | Apache-2.0 | channel-slice / global-context DESIGN reference only (don't lift the heavy global-attention decoder) | ❌ global-attention |
| **AIMET** (via Strümpler) | Apache-2.0 | AdaRound learned-rounding reference (Strümpler repo license UNVERIFIED — flag) | ✅ encode-side only |
| **RECOMBINER** `cambridge-mlg/RECOMBINER` | check | concept only (learned weight prior) — **do NOT draw the REC coder** | ❌ not numpy-feasible |
| In-repo (do NOT rebuild) | MIT | `balle_hyperprior_codec.py`, `learnable_entropy_model.py`, `entropy_bottleneck.py`, `neural_weight_codec.py`, `charm_range_coder.py` (Laplace-PMF swap), `factorized_hnerv_codec.py` (SVD), `quantization.py` (`FakeQuantSTE`/`LSQScale`/FP4-FP8), `cooperative_receiver/`, `jscc/`, `differentiable_eval_roundtrip.py` | ✅ ours |

## CONTRIBUTE-BACK list (where OUR work is genuinely novel vs the field)
1. **Task-RD weight coding through a FROZEN MULTI-task scorer.** Every codec above codes weights/latents against
   MSE/PSNR/MS-SSIM (or single-task detection). We code against `100·d_seg + √(10·d_pose)` — argmax-flip-rate of
   a frozen SegNet + MSE of a frozen PoseNet, jointly, with the nonlinear √-pose term. Novel RD operating point.
2. **The latent IS the network weights (per-video INR), not an image transform.** ELIC/MLIC/Cool-Chic/C3 code a
   learned encoder latent or an overfit image latent; we'd code the **decoder weights themselves** as the latent
   `ŷ`, with the entropy model's channel-groups = conv-weight channels. Inverts the codec topology; fuses
   COIN/instance-compression with VCM — not in the VCM literature.
3. **The eval-roundtrip null-space as the primary degree of freedom.** We train through the contest's
   `bicubic-up → uint8-STE-round → bilinear-down` + monkey-patched differentiable YUV6 to live in the scorer's
   invariance null-space (DPI operationalized). The field acknowledges DPI (Choi-Bajić Eq. 3) but doesn't make a
   codec-roundtrip null-space search against a frozen scorer the lever.
4. **Argmax-flip-rate as the EXACT authority metric** (not a BD-rate/mAP/mIoU proxy) → makes the learned-surrogate
   (C5) and boundary-Jacobian (C6) problems sharp and the boundary-margin geometry the literal objective.
5. **30-min numpy-portable inflate constraint** forces parallel/closed-form decode — findings on *which* ELIC/
   Cool-Chic simplification survives a numpy decoder are a real-world contribution the SOTA (AR / global-attention)
   literature largely ignores.

---

## RANKED RATE-AXIS PROBES (the actionable output; weight axis = the binding 91%, do first)

**Tier 1 — $0 / no-retrain, byte-closeable on the EXISTING frontier archive (highest EV):**
1. **W1 — Δ-search + parametric-rate on our exact decoder weights (pure numpy, true $0; THE decisive test).**
   Per tensor: compute C3 `laplace_rate` + Cool-Chic Exp-Golomb at the current int8 grid vs 5.63 b/param AND vs
   the empirical-histogram range bound (what we pay); then sweep Δ∈{2^k}, computing integer rate + induced weight
   error + (at a few Δ) a real HNeRV decode-replay for **task** d_seg/d_pose. **Falsification: if no Δ beats 5.63
   at iso-d_seg, the lossless-floor verdict is vindicated and the lossy-quant route is DEAD.** Reuses
   `charm_range_coder.py` (Laplace PMF swap).
2. **Sensitivity-aware mixed-precision PTQ + AdaRound (NeuroQuant B2 + AdaRound B3; no GPU retrain).** Re-quantize
   the existing weights per-channel at mixed 4–6 bits chosen by SegNet/PoseNet sensitivity (master-gradient
   ledger), AdaRound the rounding (decode UNCHANGED, zero inflate risk), re-encode constriction, byte-close,
   re-measure. **Most likely no-retrain pointer-mover.**
3. **Re-pack confirmation probe (run once to STOP re-pack effort, then stop).** Fit per-tensor `GaussianConditional`/
   GMM/`BalleHyperpriorCodec` mode=1 to frozen weight histograms; compare to 161,104. Predict net within ±1%
   (likely a LOSS after side-info, matching the in-repo `STATIC_WINS_FALLBACK`). Quantitatively kills re-pack.
4. **Per-tensor effective-centroid count (acquisition signal).** How multimodal/clusterable each weight histogram
   is → predicts the GMM/QAT joint-train ceiling (#5) without a retrain; sets λ, K.

**Tier 2 — short LOCAL fit (MPS-train / CPU-authority split, no paid GPU):**
5. **6-bit per-channel QAT through the FROZEN SegNet+PoseNet (HiNeRV recipe).** `FakeQuantSTE`/`LSQScale` +
   per-channel scales; re-export; byte-close; re-measure. The strongest symbol-changing lever; needs a QAT fit.
6. **Joint GMM-rate / ELIC-channel-group rate term in decoder training (the LIVE entropy-model lever).** Wire
   `learnable_entropy_model.rate()` into the TRAINING loss (not just the compress-time estimate) + the missing
   uniform-noise STE on the weight quantizer; weights become lower-entropy at iso-d_seg/d_pose. Predicted −5% to
   −20% of 161KB, conditional/unverified. Build the one missing component first (#7).
7. **[$0 design smoke] Restructure CompressAI `Elic2022Chandelier.ChannelGroupsLatentCodec` to code weight-tensor
   channels.** Correctness-smoke on weight-shaped tensors; preps #6. The genuine component gap.

**Tier 3 — training-technique probes that attack the d_seg plateau (orthogonal to bytes but improve the RD trade):**
8. **Feature-domain distillation auxiliary loss (Choi-Bajić Eq.6 + Iino Eq.3).** `Σ_l γ_l·MSE(SegNet_l(GT),
   SegNet_l(recon)) + pose analog`; the field's most-validated mechanism, we don't do it; short, well-conditioned
   gradient into the deep frozen scorer.
9. **Auxiliary lightweight surrogate head (Iino Eq.5) + learned d_seg surrogate (ProxIQA C5) + IDSE Jacobian
   distortion (C6).** All attack the deep-model→tiny-INR gradient pathology / the boundary-margin geometry.
10. **L1 — conv Laplace ARM over the temporal-delta latents (C3/Cool-Chic).** vs our mixed-radix packer; ceiling
   ≤ 15KB (≤ ~0.01 rate), cheap, numpy-decodable (conv ARM is parallel).

**DEFER / DO-NOT-build:** MLIC++ global-attention + serial spatial-AR (numpy-30-min-inflate incompatible);
RECOMBINER REC/bits-back (decode cost exponential in per-block KL); Exp-Golomb as a coder swap (≤ our range
coder); full Cool-Chic synthesis+ARM decoder replacement (substrate-class-shift, sequential-ARM budget risk —
validate decode wall-clock first); NWC VQ weight codec at our 228K-param scale until decoder-ship-cost <
savings is PROVEN.

## One-paragraph synthesis for the operator
The contest is literally image/video Coding for Machines, and the coders are already solved (constriction range
coding, MEASURED at the 161KB section's lossless floor of 5.63 bits/param). The entire learned-entropy lineage
(Ballé→hyperprior→AR→GMM→ELIC→MLIC++) and the overfitted-INR family (Cool-Chic/C3) offer us **nothing as a
re-pack** — confirmed in our own `STATIC_WINS_FALLBACK`. The rate term (62% of S) moves ONLY by changing the
SYMBOLS: lossy, task-RD-aware quantization that survives the frozen SegNet+PoseNet. The two fastest, no-retrain
probes are **W1** (RD-search the per-tensor quantization step against the task, Cool-Chic/C3 mechanism, pure numpy
on the existing archive — also the decisive falsification of the "5.63 is the floor" claim) and **sensitivity-aware
mixed-precision PTQ + AdaRound** (NeuroQuant + AdaRound, decode-unchanged, byte-closeable today). The one component
worth LIFTING (BSD-3-Clause-Clear) is CompressAI's `Elic2022Chandelier` space-channel context codec, restructured
to code weight-tensor channels in a joint retrain. Our genuine contribution back is task-RD weight coding of a
per-video INR through a frozen MULTI-task scorer in the eval-roundtrip null-space — a regime none of these codecs
address.
