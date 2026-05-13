# Online research ledger — Domain D: Score-aware training / inverse-steganalysis

Per-paper notes; 10 entries.

---

## D.1 — Syndrome Trellis Codes (Filler-Judas-Fridrich, IEEE TIFS 2011)
- **Authors**: Tomáš Filler, Jan Judas, Jessica Fridrich (Binghamton DDE Lab)
- **Venue**: IEEE Transactions on Information Forensics and Security 6(3), 2011
- **DOI**: 10.1109/TIFS.2011.2134094
- **IEEE link**: https://ieeexplore.ieee.org/document/5740590/
- **Empirical claim**: Minimizing additive distortion in steganography via syndrome-trellis codes; near-bound performance for arbitrary additive cost.
- **Relevance**: The CANONICAL framework for embedding a payload at MINIMUM perceptual cost. Direct dual of our problem: we EMBED bits in archive bytes; the contest scorer DETECTS distortion. Per CLAUDE.md: "The challenge IS inverse steganalysis." For our pose-marginal optimization, STC gives a CONSTRUCTIVE encoder: given a per-byte cost map, embed N bits at minimum total cost.
- **Integration cost**: ~2 days dev to wire STC encoder + cost-map producer for our archive structure.

## D.2 — UNIWARD (Holub-Fridrich-Denemark, EURASIP 2014)
- **Authors**: Vojtěch Holub, Jessica Fridrich, Tomáš Denemark
- **Empirical claim**: Wavelet-domain undetectability cost function. Errors in textured regions are LESS detectable — weight loss by inverse local variance.
- **Relevance**: Foundation for our score-aware loss design. The "weight by inverse local variance" trick suggests our pose/seg loss should be reweighted by local-texture energy.
- **Integration cost**: ~1 day dev (compute local-variance per pixel; multiply into score-aware loss).

## D.3 — EfficientNet steganalysis (Yousfi et al., 2020-2021)
- **Authors**: Yassine Yousfi, Jan Butora, Jessica Fridrich (Binghamton DDE Lab)
- **Empirical claim**: EfficientNet-B2 outperforms previous architectures on JPEG steganalysis (ALASKA challenge). Fridrich's student → directly informs SegNet scorer design.
- **Relevance**: This IS the SegNet architecture our scorer uses. Yousfi's repos:
  - https://github.com/DDELab/deepsteganalysis
  - https://github.com/YassineYousfi/alaska
  - https://github.com/YassineYousfi/OneHotConv
  - https://github.com/YassineYousfi/comma10k-baseline (the comma seg baseline)
  - https://github.com/YassineYousfi/autostego
- **Integration cost**: Reference — informs every interaction with SegNet.

## D.4 — Universal Adversarial Perturbations for Steganography (2024)
- **arXiv via Springer**: https://link.springer.com/article/10.1007/s11042-024-19122-x
- **Empirical claim**: UAP-based payload embedding has higher security vs steganalysis nets.
- **Relevance**: Direct application to our pose/seg attack: find a UAP delta that flips SegNet argmax with minimum L2 perturbation.
- **Integration cost**: ~2 days dev for UAP solver against fixed SegNet/PoseNet.

## D.5 — Adversarial Steganography Survey (Sci. Direct 2023-2024)
- **Link**: https://www.sciencedirect.com/science/article/abs/pii/S1051200423002166
- **Empirical claim**: Multi-adversarial-network framework + channel attention modules → improved steganography.
- **Relevance**: Survey-grade reading list for our score-aware attack design.

## D.6 — Adversarial Steganography via Adversarial Examples (MDPI Math. 2020)
- **Link**: https://www.mdpi.com/2227-7390/8/9/1446
- **Relevance**: Predecessor of #4.

## D.7 — Improved Syndrome-Trellis Codes for Audio (2021)
- **Link**: https://www.researchgate.net/publication/348361829
- **Empirical claim**: Adaptive STC for audio domain.
- **Relevance**: Cross-modal STC; suggests our archive could embed multiple typed payloads (mask sidecar + pose sidecar) each with independent STC.

## D.8 — Knowledge Distillation for SegNet surrogate (Hinton et al., 2015 + 2024 variants)
- See Domain J for Wasserstein-distill (NeurIPS 2024) and Rethinking-KL-for-LLMs (2024).
- **Relevance**: Our internal KL-T=2.0 distill (Quantizr stack). Wasserstein variant in TOP-10 actionable #9.

## D.9 — Information Bottleneck Lagrangian (Tishby 1999; 2024 Generalized IB)
- **arXiv**: https://arxiv.org/abs/1503.02406 (Tishby-Zaslavsky 2015)
- **2024 Generalized IB**: https://arxiv.org/abs/2509.26327
- **Empirical claim**: Lagrangian min I(X;T) - β·I(T;Y) yields compressed representation T preserving Y-relevance. Generalized IB (2024) introduces synergy term.
- **Relevance**: Direct foundation for codex's T10 IB Lagrangian lane. The 2024 generalized form's synergy term is novel — captures information available only through joint feature processing.
- **Integration cost**: ~3-5 days dev for T10 IB Lagrangian (already on backlog per session memos).

## D.10 — Score-Domain Validation pattern (operator + codex protocol, session-derived)
- **Reference**: Session memos `feedback_*_apples_to_apples_*.md` series + CLAUDE.md Apples-to-apples discipline.
- **Empirical**: Our `[contest-CUDA]` axis IS the score-domain validation; `[macOS-CPU advisory]` is the cheap proxy (Catalog #192 — within 2e-5 of `[contest-CPU]`, 60× below medal-band).
- **Relevance**: Operationally CRITICAL — every literature prediction in master synthesis must be measured against `[contest-CUDA]` before promotion.

---

## Follow-up reads:
- "J-UNIWARD Steganoanalysis" (Cybernetics 2021): https://link.springer.com/article/10.1007/s10559-021-00374-6
- "UT-GAN adaptive JPEG steganography" (MDPI 2024): https://www.mdpi.com/2079-9292/14/20/4046
- Andrew Ker "Syndrome Trellis Sampler" extensions: https://www.cs.ox.ac.uk/people/andrew.ker/docs/nakajima-syndrome-trellis-sampler.pdf
- "On the Information Bottleneck Problems" survey (Goldfeld-Polyanskiy 2020): https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7516564/
