# Online research ledger — Domain J: Eureka cross-domain (the OBSCURE genius)

Per-paper notes; 12 entries. Per operator directive: "obscure but eureka genius translation and application techniques".

---

## J.1 — Holographic Reduced Representations / HRR (Plate, IEEE TNN 1995)
- **Author**: Tony Plate
- **DOI/IEEE**: https://ieeexplore.ieee.org/document/377968/
- **PubMed**: https://pubmed.ncbi.nlm.nih.gov/18263348/
- **PDF**: https://redwood.berkeley.edu/wp-content/uploads/2020/08/Plate-HRR-IEEE-TransNN.pdf
- **Empirical claim**: Symbolic-on-vector binding via CIRCULAR CONVOLUTION; arbitrary variable bindings, sequences, frame-like structures in fixed-width vector.
- **Modern survey**: Kleyko et al., ACM CSUR 2022 https://dl.acm.org/doi/10.1145/3538531
- **Relevance**: TOP-5 OBSCURE-4. The HRR/VSA framework lets us encode (frame_idx ⊗ scene_token) compositionally into ~D-byte vectors. Decoder is correlation (no learned weights needed). **[literature-prediction: 100-200× compression of compositional sidecars relative to per-frame storage]**.
- **Integration cost**: ~2-3 days dev for a sidecar proof-of-concept; research-only L0 lane.

## J.2 — Learning with HRRs (2022)
- **OpenReview**: https://openreview.net/forum?id=RX6PrcpXP-
- **Empirical claim**: HRR can replace softmax attention via binding/unbinding operations.
- **Relevance**: Modern HRR application; suggests HRR can replace per-frame attention layers in our scorer surrogate.

## J.3 — VSA via Category Theory (2025)
- **arXiv**: https://arxiv.org/html/2501.05368v2
- **Relevance**: Theoretical foundations of VSA; bridges to abstract algebra.

## J.4 — Tropical Geometry of Neural Networks (Zhang-Naitzat-Lim, ICML 2018)
- **PDF**: https://www.stat.uchicago.edu/~lekheng/work/tropical.pdf
- **Empirical claim**: PWL neural net = tropical-rational function. Closed-form expressivity bounds via tropical hypersurface vertex count.
- **Relevance**: TOP-5 OBSCURE-2. Closed-form bound on SegNet argmax flip-perturbation.

## J.5 — Real Tropical Geometry of NNs (2024)
- **arXiv**: https://arxiv.org/abs/2403.11871
- **Empirical claim**: Positive tropicalizations of hypersurfaces; tropical semialgebraic sets for NN regions.
- **Relevance**: Adversarially-robust neural net decision boundaries via tropical embedding.

## J.6 — ICASSP 2024 Tutorial: Tropical Geometry for ML (Maragos)
- **Link**: https://robotics.ntua.gr/icassp-2024-tutorial/
- **Relevance**: Pedagogical reference for getting started with tropical geometry tools.

## J.7 — Stochastic Resonance Neurons (Manuylovich et al., Communications Engineering 2024)
- **Nature link**: https://www.nature.com/articles/s44172-024-00314-0
- **arXiv predecessor**: https://arxiv.org/abs/2205.10122 (Manuylovich 2022)
- **Empirical claim**: Bistable-dynamical-system neurons + noise; FEWER neurons for same accuracy; more robust to training noise.
- **Relevance**: TOP-5 OBSCURE-3. Combines naturally with our LangevinOptimizer for compound architecture+optimizer benefit.

## J.8 — Stochastic Multiresonance in NNs (Nature Scientific Reports 2024)
- **Link**: https://www.nature.com/articles/s41598-024-55997-4
- **Empirical claim**: Statistical complexity measures show quadruple stochastic resonance in small-world neural networks.
- **Relevance**: Theoretical companion to J.7.

## J.9 — Tensor Networks Meet Neural Networks survey (Pan et al., 2023-2024)
- **arXiv**: https://arxiv.org/html/2302.09019v3
- **Empirical claim**: CP, Tucker, MPS/Tensor-Train, MPO, TR, HT, PEPS decompositions for NN compression.
- **Relevance**: SPECULATIVE-1 in master synthesis. PEPS polynomial-correlation-decay vs MPS exponential-correlation-decay matters for spatial neural codecs.

## J.10 — Compressing NNs Using Tensor Networks Exponentially Fewer Params (Spj Science 2024)
- **Link**: https://spj.science.org/doi/10.34133/icomputing.0123
- **Relevance**: Concrete empirical: tensor networks → exponentially fewer variational parameters.

## J.11 — Lottery Ticket Hypothesis (Frankle-Carbin, ICLR 2019)
- **OpenReview**: https://openreview.net/forum?id=rJl-b3RcF7
- **Empirical claim**: Randomly-init dense NN contains sparse subnet trainable to match dense.
- **Relevance**: Predecessor of Progressive Fourier Neural Representation (E.6) which finds Lottery Tickets in Fourier space.

## J.12 — A Survey of Lottery Ticket Hypothesis (2024)
- **arXiv**: https://arxiv.org/html/2403.04861v1
- **Relevance**: Modern survey for IMP variants.

## J.13 — Bayesian Lottery Ticket Hypothesis (2026)
- **arXiv**: https://arxiv.org/html/2602.18825
- **Relevance**: Bayesian re-formulation; suggests posterior-over-tickets is the right object.

## J.14 — Concrete Ticket Search (2025)
- **arXiv**: https://arxiv.org/html/2512.07142
- **Empirical claim**: Preserve training dynamics during ticket discovery.
- **Relevance**: Follow-up of #11.

## J.15 — Cross-modal compression (text → bytes; audio → image)
- **Reference**: Various 2024-2026 cross-modal codec work.
- **Relevance**: Speculative — our contest is single-modal (video) but the cross-modal techniques may apply within the renderer+mask+pose tri-modal structure.

---

## How each obscure-genius primitive could plug into our contest

### J-genius-1: HRR sidecar (Plate 1995)
- Per Obscure-4 in master synthesis. Build `tac.substrates.hrr_sidecar`:
  - Encode per-frame (frame_idx, scene_descriptor) via circular convolution into a single D-dim vector
  - Decoder is just correlation/unbinding — NO learnable parameters needed
  - Quantize the D-dim vector for archive
  - Cost: O(D log D) per frame at decode via FFT
- **[literature-prediction: D=4096 floats × 4 bytes = 16KB, vs 1200 frame-latents × 8 floats × 4 bytes = 38KB for our PR106 sidecar → ~60% reduction at LOSSLESS fidelity if scene-tokens are well-chosen]**.
- Research-only L0 lane; needs council review before substrate dispatch.

### J-genius-2: Tropical L∞ bound on SegNet (Domain J.4-J.6)
- Computes the MINIMUM L∞ perturbation needed to flip SegNet argmax at each pixel.
- Direct application: use this as a per-pixel COST MAP for STC encoding (Domain D.1).
- **[literature-prediction: 2-5× improvement in mask attack efficiency vs current proxy-loss-guided attacks]**.

### J-genius-3: Stochastic resonance neurons in our renderer
- Replace some HNeRV-family conv layers with bistable-resonator activations.
- Same accuracy with fewer params → smaller archive.
- Speculative; smoke test before adoption.

### J-genius-4: Lottery ticket compression
- IMP on our renderer → identify subnetwork. Encode mask (which neurons are kept) cheaply via STC (Domain D.1).
- Already partially explored in our internal IMP cycle work.

### J-genius-5: Tensor train decomposition of renderer weights
- MPS/Tensor-Train factorization of conv weight tensors.
- Speculative-1 in master synthesis; defer until budget headroom.

## Follow-up reads:
- ACM CSUR Hyperdimensional Computing survey Part I + II (Kleyko 2022): https://dl.acm.org/doi/10.1145/3538531 + https://dl.acm.org/doi/10.1145/3558000
- Schmidhuber "Compression-as-Intelligence" foundational essays (1980s-2010s, MDL/Kolmogorov complexity)
- Recent (2024-2026) work on quantum-inspired tensor decompositions
- Robotics tropical-geometry talks: https://icarus.csd.auth.gr/tropical-algebra-and-geometry-for-machine-learning-and-optimization-2/
