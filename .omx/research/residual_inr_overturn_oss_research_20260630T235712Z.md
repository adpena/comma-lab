# Residual-INR "NO-GO" overturn — OSS/papers research pass (advisory)

- **UTC**: 20260630T235712Z
- **Axis tag**: `[advisory only]` — online + CPU research; **NO score claim**; pointer UNMOVED 0.19110. This is a MEANS, not goal progress.
- **NO-FAKE**: every arXiv id + repo below was returned by a live web search this session; the two most load-bearing papers (2307.12864, 2305.02562) were additionally fetched and their abstracts read. Uncertainty is flagged inline per item. I did **not** independently re-open every PDF, so treat each "transferable mechanism" line as *from-abstract* unless it says *fetched*.
- **Operator ask (2026-06-30)**: "more research and deep math and research" to try to OVERTURN the residual-INR NO-GO before accepting the negative (per `[[feedback-not-pessimistic-first-results-adversarial-deepmath-oss-against-negatives]]`).

## The NO-GO being challenged (restated)

The lab's θ*-residual-INR plan = deterministic/learned **bulk** (store-canonical + per-class warp) + a **trained residual INR** that only fixes the binding lane-survival edge. It was declared NO-GO because a residual against a *naive* bulk had **"unreachable interior errors"** — the residual was NOT localized to a boundary mask, so a cheap boundary-only residual could not capture it.

The contest is **indirect (remote) rate-distortion / coding-for-machines / inverse-steganalysis**: the target is the frozen SegNet argmax partition (a **piecewise-constant** field) + frozen PoseNet 6-vector, scored only through the operator R on the exact archive bytes. RATE is the binding sub-0.15 lever (seg capped ~0.012; lossless rate on 0.19110 exhausted). So the relevant literature is (a) residual/conditional neural codecs, (b) Wyner-Ziv side-info residuals, (c) coding-for-machines residuals, (d) residual-locality theory, (e) INR weight-rate compression.

**Central deep-math reframe the literature supports:** the NO-GO measured a *bad-bulk* residual (interior-filling), not a *good-bulk* residual. For a **piecewise-constant** target, a residual against a *good* predictor is provably supported on (a neighborhood of) the **codim-1 boundary** — which is exactly the annulus we already localize. The field's name for "interior leaks because the predictor path is lossy" is the **information bottleneck / data-processing-inequality** problem in conditional coding, and it has a published remedy (Q1). The fix is therefore *improve the bulk + condition the residual*, NOT *abandon the residual*.

---

## Q1 — Residual / hybrid neural codecs: additive residual vs masked override, and the bottleneck

**The single most relevant result.**
- **Conditional Residual Coding: A Remedy for Bottleneck Problems in Conditional Inter Frame Coding** — Brand, Seiler, Kaup (FAU Erlangen), **IEEE TCSVT 2024**. arXiv **2307.12864**. *(fetched)*
  - Mechanism: pure **conditional** coders suffer an *information bottleneck in the prediction path* — by the **data-processing inequality**, not all prediction-signal info reaches the reconstruction, so quality is impaired. Pure **residual** coders avoid the bottleneck but are theoretically inferior. **Conditional residual coding** (feed the prediction as a condition *and* code a residual w.r.t. it) "significantly reduces the influence of bottlenecks while maintaining the theoretical performance of the conditional coder" — "best of both worlds."
  - **Overturn relevance (HIGH):** the lab's "unreachable interior errors" is textbook *bottleneck*: a naive deterministic bulk is a lossy prediction path; a *masked* residual is the pure-residual extreme. The published cure is the hybrid — make the residual **conditional on the bulk/lane prior AND full-frame additive**, not mask-gated. This directly contradicts the framing that produced the NO-GO.
- **On Benefits and Challenges of Conditional Interframe Video Coding in Light of Information Theory** — arXiv **2210.07737**. The information-theoretic companion: when conditional beats residual and why the bottleneck appears. Use to *derive* (not vibe) whether our bulk path is bottlenecked.
- **MaskCRT: Masked Conditional Residual Transformer for Learned Video Compression** — arXiv **2312.15829**. Mechanism: a learned **soft mask** spatially blends conditional coding vs conditional-residual coding per-pixel. *Directly* answers "additive residual vs masked override": you don't choose — you learn a per-pixel mixing weight. Relevant because our annulus is exactly a region where the mix should favor residual.
- **Conditional Residual Coding with Explicit-Implicit Temporal Buffering** — arXiv **2508.01818** (recent; abstract-only). Newer instance of the same family.
- **Cool-Chic / C3 (overfitted single-instance codecs)** — the closest architectural cousin to our witness (overfit one clip, tiny decoder, latents are the payload):
  - C3: **High-performance and low-complexity neural compression from a single image or video** — arXiv **2312.02753** (GitHub: google-deepmind/c3_neural_compression). 
  - Cool-Chic perceptually tuned — arXiv **2401.02156**; reduced-complexity — arXiv **2403.11651**; Cool-chic video 800 params — HAL **hal-04596496**. Their decoder ends in **residual conv layers** added to an upsampled-latent base — i.e. a *full-frame additive residual* on top of a cheap base, exactly the topology we want. *(Cool-chic 5.0 listed as arXiv 2605.02726 in search results — ID implies May-2026; NOT independently verified — treat as uncertain.)*
- **Hyperprior lineage (the additive-residual-as-entropy-model ancestor)**: Ballé scale-hyperprior + Minnen autoregressive context, operationalized in **ELIC** (uneven channel groups + checkerboard) and **MLIC / MLIC++** — arXiv **2211.07273** (+ MLIC++ on ResearchGate; arXiv id not re-confirmed this pass). These are the *entropy-model* half: how to code a dense field cheaply given context. Relevant to coding the residual field once it IS boundary-localized.

**Answer to "which use FULL-FRAME additive residual (no mask)?"** Cool-Chic/C3 (residual conv on upsampled base), classical hybrid codecs (HEVC/VVC residual path), and the *residual* arm of every conditional-residual paper above. The mask is **optional and learnable** (MaskCRT), not required.

---

## Q2 — Conditional / side-information residual (Wyner-Ziv): how cheap can a conditioned residual be?

The design-refine's `CONDITIONAL_ON_LANE_PRIOR` revival path **is** Wyner-Ziv: encode the residual given side information (lane geometry) the decoder also has. Theory says this is *strictly cheaper* than unconditioned.

- **Neural Distributed Source Coding (NDSC/NDIC)** — arXiv **2106.02797**. A small shared net preprocesses the side information; when absent, output is a zero tensor. Clean template for "condition the residual on the lane prior."
- **Deep Image Compression using Decoder Side Information** — arXiv **2001.04753**. Encoder exploits a correlated image available only at the decoder → fewer bits.
- **Learned Wyner-Ziv Compressors Recover Binning** — Özyılkan et al., arXiv **2305.04380**. Shows learned WZ codecs *rediscover binning* — evidence the savings are real and learnable, and quantifies how close to the WZ bound a small net gets. Use to bound "how cheap."
- **Importance Matching Lemma for Lossy Compression with Side Information** — arXiv **2401.02609**. A one-shot/short-blocklength tool for side-info compression — relevant because our payload is tiny (per-clip), i.e. the short-blocklength regime where asymptotic WZ doesn't directly apply.
- **Learned Layered Coding for Successive Refinement in the Wyner-Ziv Problem** — arXiv **2311.03061**. Successive-refinement WZ → a *scalable* residual (coarse boundary first, refine). Maps onto curriculum CE→tau→l7 as rate layers.
- **Distributed Deep JSCC with Decoder-Only Side Information** — arXiv **2310.04311**; classical **Graph-based Code Design for Quadratic-Gaussian WZ** — arXiv **1205.4332** (a *residual* WZ structure for arbitrary side info — the literal "residual + side info" combo).

**How cheap:** WZ theory says conditioning on a correlated prior reduces rate to the **conditional** rate-distortion (you pay only H(residual | lane-prior), not H(residual)). For a residual that is boundary-localized given the lane geometry, the conditional entropy is small — the lane prior already predicts *where* and *which way* the flip goes. This is the quantitative case that a conditioned residual is far cheaper than the unconditioned residual the NO-GO measured.

---

## Q3 — Task-space / coding-for-machines residuals: "code only where the base errs", NOT mask-limited

**The most contest-shaped result** (it literally codes the human-reconstruction enhancement *given a semantic-segmentation base task representation* on Cityscapes):
- **Conditional and Residual Methods in Scalable Coding for Humans and Machines** — de Andrade, Harell, Foroutan, Bajić. **IEEE ICME Workshop on Coding for Machines, 2023**. arXiv **2305.02562**. *(fetched)*
  - Mechanism: the enhancement representation is coded **conditioned on / as a residual w.r.t. the base task representation** via an **entropy model with increased modelling capacity** — **NOT a spatial mask**. The "code only where the base errs" effect is achieved by the *entropy model* (conditional probability collapses where base already determines the answer → ~0 bits there), not by masking pixels.
  - Finding: conditional ≈ residual here (curves "contained within our baselines"), i.e. both work and the choice is secondary to the entropy model. **Overturn relevance (HIGH):** this is direct evidence that a non-mask, entropy-model-driven residual on top of a *segmentation* base is a published, working mechanism — the exact thing the lab's NO-GO said was impossible against a naive bulk. The fix is the *entropy model*, not a mask.
- **Scalable Image Coding for Humans and Machines** — Choi & Bajić, **IEEE TIP 2022**. arXiv **2107.08373**. Latent-space scalability: the base (machine task) is a *subset* of the latent; the enhancement uses additional latent subsets. "Code only where base errs" = allocate enhancement latent only to the complement the base didn't capture.
- **Learned Scalable Video Coding for Humans and Machines** — arXiv **2307.08978**; **Scalable Video Coding for Humans and Machines** — arXiv **2208.02512**; **VVC+M: Plug-and-Play Scalable Image Coding** — arXiv **2305.10453**. Base layer = task bitstream; enhancement = conditional-coded reconstruction. 13–19% bit savings vs SOTA by optimizing the base for the task first.

**Answer to "any 'code only where base errs' that isn't mask-limited?"** Yes — the **conditional entropy model** is the canonical non-mask mechanism: it spends bits ∝ surprise of the residual given the base, so it spends ~0 in the interior automatically *if the base is good there*. No explicit mask needed.

---

## Q4 — Residual locality vs base quality (the deep-math crux of the overturn)

The NO-GO's core empirical claim ("interior errors unreachable by a boundary mask") is **conditional on a bad base**. The literature supports the opposite for a *good* base on a *piecewise-constant* target:

- **ResKD: Residual-Guided Knowledge Distillation** — arXiv **2006.04719**. The residual = teacher−student *knowledge gap*; a lightweight "res-student" rectifies exactly the former student's errors. Empirically the residual concentrates on **hard examples near decision boundaries** (the gap is small where the base is confident/correct). Direct analogy: bulk = base student, residual-INR = res-student, target = SegNet decision boundaries.
- **Intra-prediction residual energy concentrates at edges** (HEVC/VVC residual-coding literature; surfaced via patent corpus, e.g. residual-block rotation + residual-sign-prediction excluding discontinuous border pixels). The established codec fact: against a *good* predictor, residual magnitude is largest at object/block boundaries and ~0 in flat interiors. For a piecewise-constant argmax field this is sharpest of all: a perfect interior predictor leaves residual support **only** on the codim-1 boundary.
- **Predictive coding (Rao–Ballard lineage)**: error units fire in high-variance regions (boundaries/center structure), prediction units cover low-variance flats — the same locality claim from neuroscience. (General lineage; cited as principle, not a single codec arXiv id.)

**Deep-math statement (the overturn):** Let the target be a piecewise-constant labelling L on a partition with boundary set ∂ (codim-1, measure→0). A predictor p that is exact on the interior has residual r = L − p supported on a width-w tube around ∂ (w shrinks as p improves). Therefore the residual IS reachable by a boundary mask/annulus **iff the base is good on the interior**. The NO-GO measured w large (interior leakage) because the *naive bulk* was wrong in the interior, not because boundary residuals are intrinsically unreachable. **Conclusion: the NO-GO is a statement about base quality, not about residual coding.** Fix the bulk (store-canonical + per-class warp already does most of this) → residual collapses onto the annulus we already target → the boundary-conditioned residual becomes both *reachable* and *cheap* (Q2). This is the strongest single argument that the NO-GO should be RE-OPENED at implementation level (paradigm intact), per Forbidden-premature-KILL / janky-prototype-RE-OPEN.

---

## Q5 — Quantization / smaller-representation of a trained witness (the binding RATE lever)

SOTA INR weight-rate compression, best-first:

1. **NVRC: Neural Video Representation Compression** — Kwan, Bull et al. (Bristol), arXiv **2409.07414** (Sep 2024, rev 2025). **First INR codec to beat VVC VTM-RA** (~24% BD-rate gain on UVG, PSNR). Mechanism: fully **end-to-end entropy-coded** weights — *quantization parameters and the entropy model itself are learned and signalled*, with hierarchical/quantization-aware coding of the network parameters. This is the current ceiling for "shrink a trained neural field" and the canonical target for our RATE half. **Best overall Q5 pick.**
2. **HiNeRV** — Kwan et al., **NeurIPS 2023**, arXiv **2306.09818**. Refined train→**prune**→**quantize**→entropy-code pipeline; the canonical "lossy model compression that preserves an INR's quality." Our parity bank already; its compression *pipeline* (not its decoder) is the reusable asset.
3. **Implicit Neural Representations for Image Compression** — Strümpler et al., **ECCV 2022**, arXiv **2112.04267** (GitHub: YannickStruempler/inr_based_compression). The canonical INR compression *recipe*: **quantization + quantization-aware retraining + entropy coding**, plus **meta-learned (MAML) init** to cut encode steps and improve R-D. Directly transferable QAT-for-INR procedure.
4. **RECOMBINER** — cambridge-mlg, **ICLR 2024**, arXiv **2309.17182** (GitHub: cambridge-mlg/RECOMBINER; predecessor **COMBINER** arXiv **2305.19185**, GitHub cambridge-mlg/combiner). **Avoids quantization entirely** — directly optimizes rate-distortion via a variational posterior over INR weights + relative-entropy coding, with **linear reparameterization of weights** (≈ low-rank) + learned positional encodings for local detail. SOTA at *low bitrate* (CIFAR-10) — our exact regime (tiny per-clip payload). **Best pick if we want to skip QAT and code weights variationally.**
5. **On Quantizing Implicit Neural Representations** — arXiv **2209.01019**; **SINR: Sparsity-Driven Compressed INR** — arXiv **2503.19576** (sparse-coding the weights); **Enhancing INR via Symmetric Power Transformation** — arXiv **2412.09213** (reshape weight distribution for cheaper coding); **UAR-NVC** — arXiv **2503.02733** (memory-efficient autoregressive NVC). Supporting toolbox: low-rank/factorized fields, sparsity, weight-distribution reshaping before entropy coding.

**Best rate-axis technique (single recommendation):** adopt the **NVRC end-to-end learned-entropy-coded-weights** scheme as the RATE objective (learn the quantization + entropy model jointly with the witness), and if QAT instability bites, fall back to **RECOMBINER's variational/relative-entropy weight coding** (no quantization, low-rank reparam, SOTA at our low-bitrate regime). Strümpler's QAT+entropy+MAML pipeline (2112.04267) is the lowest-friction starting recipe and has public code.

---

## TOP-3 most-likely-to-overturn the NO-GO (ranked)

1. **Conditional Residual Coding (the bottleneck remedy)** — arXiv **2307.12864** (+ theory **2210.07737**, + learnable mix **MaskCRT 2312.15829**). *Why #1:* it names the exact failure ("interior leak" = information bottleneck via DPI) and gives the published cure (condition the residual on the bulk AND code a full-frame residual, optionally with a learned soft mask). Turns the NO-GO from a paradigm kill into a fixable architecture bug.
2. **Coding-for-machines conditional/residual on a segmentation base** — arXiv **2305.02562** (+ latent-scalable **2107.08373**). *Why #2:* near-identical problem (code the residual on top of a *segmentation* task base) is published and *works* with a **conditional entropy model, no mask** — and the "spend ~0 bits in the interior" effect falls out of the entropy model automatically when the base is good. Direct existence-proof against "unreachable interior."
3. **Residual-locality-vs-base-quality (the deep-math reframe)** — **ResKD 2006.04719** + intra-residual-at-edges (codec literature) + the piecewise-constant boundary-tube argument in Q4. *Why #3:* it reframes the NO-GO as a *base-quality* statement, not a residual-coding statement — predicting that once the bulk is good on the interior (store-canonical + per-class warp), the residual collapses onto the annulus we already localize, so the boundary-conditioned residual becomes reachable AND cheap (Wyner-Ziv, Q2). This is the cheapest test: improve the bulk, re-measure residual support.

**Cheapest $0 falsification of the NO-GO (recommended next unit):** take the *good* deterministic bulk (store-canonical + per-class warp, the v2 vehicle), compute the residual field against the frozen SegNet argmax, and **measure its support width w** (fraction of residual mass outside a k-px annulus) vs the *naive* bulk. If w shrinks toward the annulus as predicted by Q4, the NO-GO is implementation-level falsified and the conditional-residual path (Q1/Q3) re-opens. This is advisory until a byte-closed exact row through R moves the pointer.

---

## Source ledger (verification status)

All returned by live web search this session. ✅ = arXiv abstract page/listing seen in results; 📄 = additionally fetched + abstract read; ⚠️ = id appeared in results but NOT independently re-confirmed (flagged).

| arXiv / id | title (short) | status |
|---|---|---|
| 2307.12864 | Conditional Residual Coding (bottleneck remedy), TCSVT'24 | 📄 |
| 2305.02562 | Conditional & Residual Methods in Scalable Coding for H&M, ICME-W'23 | 📄 |
| 2210.07737 | Benefits/Challenges of Conditional Interframe Coding (IT) | ✅ |
| 2312.15829 | MaskCRT (masked conditional residual transformer) | ✅ |
| 2508.01818 | Conditional Residual Coding w/ Explicit-Implicit Temporal Buffer | ✅ |
| 2312.02753 | C3 single-image/video neural compression (DeepMind) | ✅ |
| 2401.02156 | Cool-Chic perceptually tuned | ✅ |
| 2403.11651 | Overfitted image coding at reduced complexity (Cool-Chic) | ✅ |
| hal-04596496 | Cool-chic video, 800 params (HAL) | ✅ |
| 2605.02726 | Cool-chic 5.0 (id implies May-2026) | ⚠️ NOT confirmed |
| 2211.07273 | MLIC multi-reference entropy model | ✅ |
| 2106.02797 | Neural Distributed Source Coding (NDIC) | ✅ |
| 2001.04753 | Deep Image Compression w/ decoder side info | ✅ |
| 2305.04380 | Learned Wyner-Ziv Compressors Recover Binning | ✅ |
| 2401.02609 | Importance Matching Lemma (side-info, one-shot) | ✅ |
| 2311.03061 | Learned Layered Coding, Successive Refinement WZ | ✅ |
| 2310.04311 | Distributed Deep JSCC, decoder-only side info | ✅ |
| 1205.4332 | Graph-based Code Design, Quadratic-Gaussian WZ (residual WZ) | ✅ |
| 2107.08373 | Scalable Image Coding for H&M (Choi-Bajić, latent scalability) | ✅ |
| 2307.08978 | Learned Scalable Video Coding for H&M | ✅ |
| 2208.02512 | Scalable Video Coding for H&M | ✅ |
| 2305.10453 | VVC+M plug-and-play scalable coding | ✅ |
| 2006.04719 | ResKD residual-guided distillation | ✅ |
| 2409.07414 | NVRC neural video representation compression | ✅ |
| 2306.09818 | HiNeRV (NeurIPS'23) | ✅ |
| 2112.04267 | Strümpler INR for image compression (ECCV'22) | ✅ |
| 2209.01019 | On Quantizing INRs | ✅ |
| 2309.17182 | RECOMBINER (ICLR'24) | ✅ |
| 2305.19185 | COMBINER | ✅ |
| 2503.19576 | SINR sparsity-driven compressed INR | ✅ |
| 2412.09213 | Symmetric Power Transformation for INR | ✅ |
| 2503.02733 | UAR-NVC autoregressive memory-efficient NVC | ✅ |
| 2201.12904 | COIN++ (cited from snippet, not re-listed this pass) | ⚠️ from snippet |

**Repos** (returned in search results): google-deepmind c3_neural_compression (from C3 listing); cambridge-mlg/RECOMBINER; cambridge-mlg/combiner; YannickStruempler/inr_based_compression. *(GitHub URLs appeared in results; not cloned/verified for liveness this pass.)*

## NO-FAKE / scope discipline

- This is **advisory research** — a MEANS. It does **not** move the exact frontier (still 0.19110) and is not goal progress until a byte-closed exact row through R beats it.
- The overturn is at **implementation level** (paradigm intact), consistent with Forbidden-premature-KILL: the NO-GO falsified a *naive-bulk masked residual*, not the residual-INR paradigm.
- ⚠️ items above must be re-verified before being cited as load-bearing. The two TOP-3 anchors (2307.12864, 2305.02562) were fetched and are safe to lean on.
