# Online research ledger — Domain G: Compression theory frontier

Per-paper notes; 10 entries.

---

## G.1 — Slepian-Wolf coding (1973; foundational)
- **Empirical claim**: Lossless compression of correlated sources can achieve the joint entropy rate even when sources are encoded SEPARATELY, as long as decoder has access to all.
- **Relevance**: Foundation for distributed source coding (DSC). Our archive HAS multiple correlated streams (renderer + masks + poses).

## G.2 — Wyner-Ziv coding (1976; foundational)
- **Empirical claim**: Lossy DSC: rate-distortion bound when decoder has side info but encoder does not.
- **Relevance**: TOP-5 OBSCURE-5 in master synthesis. **Decoder-side knowledge is FREE**. The inflate side has full access to upstream `evaluate.py` constants, video metadata, fixed architectures.

## G.3 — Neural Distributed Source Coding (Whang-Acharya-Kim-Pananjady-Cohen 2024)
- **arXiv**: https://arxiv.org/abs/2106.02797
- **IEEE J 2024**: https://ieeexplore.ieee.org/document/10557705/
- **Simons talk 2024**: https://simons.berkeley.edu/talks/hyeji-kim-university-texas-austin-2024-03-07
- **Empirical claim**: Conditional VQ-VAE for distributed encoder+decoder; arbitrary correlation structures.
- **Relevance**: Modern neural realization of Slepian-Wolf. Directly portable to our renderer+mask+pose architecture.
- **Integration cost**: ~3+ days dev.

## G.4 — Deep JSCC with Decoder-Only Side Info (Yilmaz et al., ICMLCN 2024)
- **Repo**: https://github.com/ipc-lab/deepjscc-wz
- **Empirical claim**: Joint source-channel coding with decoder-only side info; improved RD over equal-info baseline.
- **Relevance**: Top-20 EUREKA #18 + OBSCURE-5. Concrete realization of Wyner-Ziv for neural codecs.

## G.5 — Learned Layered Wyner-Ziv (Joukovsky et al., 2023)
- **arXiv**: https://arxiv.org/abs/2311.03061
- **Empirical claim**: RNN successive-refinement with side info; trained by minimizing variational rate-distortion bound.

## G.6 — RWZC Model-Driven Robust Wyner-Ziv (2025)
- **arXiv**: https://arxiv.org/html/2501.09520
- **Relevance**: Model-driven (not data-only) Wyner-Ziv; structurally robust.

## G.7 — COMBINER (NeurIPS 2023 Spotlight)
- See Domain A.8 for full notes.
- **Key**: Relative entropy coding — bypasses quantization; directly optimizes RD via β-ELBO.

## G.8 — RECOMBINER (ICLR 2024)
- See Domain A.9.

## G.9 — CTW (Context-Tree Weighting; Willems-Shtarkov-Tjalkens, 1995)
- **Empirical claim**: Universal source coding for tree sources; near-Shannon-entropy on binary data.
- **Relevance**: codex-mentioned in session memos. Possible entropy-coder upgrade for our archive payloads.

## G.10 — EZW / SPIHT / EBCOT (embedded wavelet bitplane)
- **Reference**: Shapiro 1993 / Said-Pearlman 1996 / Taubman 2000.
- **Relevance**: Embedded bitplane = progressive decode = stack-of-stacks composition primitive. Reference for any wavelet-domain sidecar.

## G.11 — Trellis Coded Quantization (Marcellin-Fischer 1990)
- **Reference**: Optimal trellis-structured VQ; achieves R-D bound asymptotically.
- **Relevance**: Reference for any structured-quantization codec atom.

## G.12 — Vector Quantization Linde-Buzo-Gray (LBG 1980)
- **Reference**: Classic VQ via k-means iteration.
- **Relevance**: Foundational; predecessor of VQ-VAE.

## G.13 — Fractal compression / PIFS (Barnsley-Jacquin 1992)
- **Reference**: Iterated Function Systems for image compression.
- **Relevance**: Speculative; fractal-based renderer not currently competitive but a META cross-domain idea worth archiving.

## G.14 — MDL / MML (Solomonoff-Kolmogorov-Rissanen-Wallace 1960s-1980s)
- **Reference**: Minimum Description Length / Minimum Message Length.
- **Relevance**: The theoretical floor of our 0.10 ± 0.03 estimate ties to MDL bounds. Per CLAUDE.md Shannon council member.

## G.15 — Rate-Distortion-Perception Tradeoff (Blau-Michaeli 2018-2023)
- **arXiv link family**: 1901.07821 and follow-ups.
- **Empirical claim**: Distortion (MSE-like) and perception (GAN-feature-like) are NOT the same axis; there's a 3-way RDP frontier.
- **Relevance**: Our contest scorer IS a perception axis (PoseNet + SegNet are perceptual proxies). The RDP framing suggests we have unexplored ground in the perception-distortion plane.

---

## Wyner-Ziv exploitation for our contest

Per Obscure-5 (master synthesis):

The inflate runtime has access to:
- Upstream `evaluate.py` source code (PUBLIC)
- PoseNet architecture (FastViT-T12; PUBLIC via Apple ml-fastvit repo)
- SegNet architecture (smp.Unet EfficientNet-B2; PUBLIC via segmentation_models.pytorch)
- The contest video filenames + structure (PUBLIC via the contest repo)

Anything DERIVABLE from these constants is FREE bytes on the decoder side. The strict-scorer-rule prohibits LOADING the scorer weights at inflate (~73MB rate hit), but it does NOT prohibit:
- Reconstructing fixed FastViT-T12 channel-width tables from the architecture spec (~hundreds of bytes saved)
- Hard-coding the contest's preprocess constants (mean=127.5, std=63.75) that inflate would otherwise carry
- Recomputing the YUV6 transform matrix at inflate from BT.601 spec instead of carrying numeric constants

These are typically ≤1KB savings each but **cumulative and FREE** in the Wyner-Ziv sense.

## Follow-up reads:
- Blau-Michaeli 2018 ("The Perception-Distortion Tradeoff", CVPR 2018)
- Theis-Wagner 2021 (rate-perception variational bound)
- Tishby 1999 IB foundational paper (G+D cross-link)
