# Online research ledger — Domain A: INR / Neural Codec (post-2024)

Per-paper notes; 12 entries. All claims `[literature-prediction]` or `[third-party-empirical:<paper>]` per CLAUDE.md Apples-to-apples discipline. Cross-link: master synthesis `online_research_bleeding_edge_synthesis_20260513.md`.

---

## A.1 — HiNeRV (Kwan et al., NeurIPS 2023)
- **Authors**: Ho Man Kwan, Ge Gao, Fan Zhang, Andrew Gower, David Bull (Univ. Bristol)
- **Venue**: NeurIPS 2023
- **arXiv**: https://arxiv.org/abs/2306.09818
- **Repo**: https://github.com/hmkx/HiNeRV
- **Project page**: https://hmkx.github.io/hinerv/
- **Empirical claim**: 36.6dB PSNR @ 0.051 bpp on UVG; 72.3% BD-rate save over HNeRV; 43.4% over DCVC-DC.
- **Architecture**: Depth-wise conv + MLP + bilinear interp; hierarchical positional encoding; both frames AND patches.
- **Relevance to our contract**: Direct predecessor of every HNeRV-family substrate we're working on. The hierarchical encoding is what PR101 partially captures.
- **Integration cost**: HiNeRV's pipeline (train/prune/quant) is a complete substrate-engineering reference; ~5-7 days to port a clean version with archive grammar + inflate runtime.
- **Risk**: substrate-engineering scale per HNeRV-parity lesson 7.

## A.2 — CANeRV (anonymous, 2025)
- **Authors**: Listed in arXiv 2502.06181
- **arXiv**: https://arxiv.org/abs/2502.06181
- **Empirical claim**: First INR to beat H.266/VVC (x266) on HEVC ClassB + UVG datasets; **20% BD-rate save over HiNeRV**.
- **Architecture**: Content-adaptive network structure: Dynamic Sequence-level Adjustment (DSA), Dynamic Frame-level Adjustment (DFA), Hierarchical Structural Adaptation (HSA).
- **Relevance**: Beating VVC at video-INR is the frontier. If portable, this is the strongest candidate for a non-HNeRV substrate.
- **Integration cost**: ~3+ days (no public repo as of search date; would need to re-implement from paper).
- **Risk**: HIGH — author code not surfaced; reimplementation hazard.

## A.3 — DS-NeRV (Yan et al., CVPR 2024)
- **Authors**: Hao Yan, Zhihui Ke, Xiaobo Zhou, Tie Qiu, Xidong Shi, Dadong Jiang (Tianjin Univ.)
- **Venue**: CVPR 2024
- **arXiv**: https://arxiv.org/abs/2403.15679
- **Project page**: https://haoyan14.github.io/DS-NeRV/
- **Empirical claim**: 31.2 PSNR with only 0.35M parameters.
- **Architecture**: Decomposes video into sparse static + dynamic codes; no optical flow / no residual supervision; cross-channel attention fusion for frame decoding.
- **Relevance**: Static/dynamic decomposition is directly aligned with our PR106 r2 latent sidecar structure. Adapt: static codes go in the bulk, dynamic codes go in the sidecar.
- **Integration cost**: ~1.5 days dev (substrate-class atom in registry; archive grammar update; smoke).

## A.4 — C3 (Kim et al., CVPR 2024)
- **Authors**: H. Kim, M. Bauer, L. Theis, J. R. R. A. Martens, E. Dupont (DeepMind)
- **Venue**: CVPR 2024
- **arXiv**: https://arxiv.org/abs/2312.02753
- **Project page**: https://c3-neural-compression.github.io/
- **Paper PDF**: https://openaccess.thecvf.com/content/CVPR2024/papers/Kim_C3_High-Performance_and_Low-Complexity_Neural_Compression_from_a_Single_Image_CVPR_2024_paper.pdf
- **Empirical claim**: Matches VTM (H.266 reference) with <3k MACs/pixel; order of magnitude lower decoding complexity than neural baselines at same RD.
- **Architecture**: Cool-Chic successor + auto-regressive per-channel entropy coder + soft-rounding with annealed temperature + Kumaraswamy noise.
- **Relevance**: Two specific techniques are DIRECTLY usable (soft-rounding + Kumaraswamy noise) — TOP-10 actionable #3 in master synthesis.
- **Integration cost**: ~0.5 day dev (drop into `tac.differentiable_eval_roundtrip`).

## A.5 — FINER (Liu et al., CVPR 2024)
- **Authors**: Zhen Liu, Hao Zhu, et al.
- **Venue**: CVPR 2024
- **arXiv**: https://arxiv.org/abs/2312.02434
- **Repo**: https://github.com/liuzhen0212/FINER
- **Project page**: https://liuzhen0212.github.io/finer/
- **Empirical claim**: Variable-periodic activation `sin((|x|+1)·x)` outperforms SIREN on 2D image fitting / 3D SDF / 5D NeRF.
- **Architecture**: Tunes spectral bias via initialization range of biases.
- **Relevance**: TOP-10 actionable #6. Drop-in activation replacement.
- **Integration cost**: ~0.5 day dev.

## A.6 — FINER++ (Zhu et al., 2024)
- **arXiv**: https://arxiv.org/html/2407.19434
- **Project page**: https://liuzhen0212.github.io/finerpp/
- **Empirical claim**: Family of variable-periodic functions; extends FINER.
- **Relevance**: Follow-up of A.5; revisit after FINER baseline.

## A.7 — Cool-Chic 5.0 (Ladune et al., 2025)
- **Authors**: Théo Ladune, Pierrick Philippe, Wassim Hamidouche, Lu Zhang, Olivier Déforges (Orange)
- **arXiv**: https://arxiv.org/html/2605.02726
- **Repo**: https://github.com/Orange-OpenSource/Cool-Chic
- **Empirical claim**: First overfitted codec to beat VVC (-11% rate); -7% rate over Cool-Chic 4.x; 10× faster encoding; <2000 MACs/pixel.
- **Architecture**: + Inter-Feature Context Extractor + hyperlatent grids + stabilizer linear layers + 2nd-order optimizer for encoding.
- **Relevance**: TOP-10 actionable #4. Hyperprior side-info port to T1-Balle.
- **Integration cost**: ~2 days dev (substrate-engineering); new archive section + new inflate parser.

## A.8 — COMBINER (Guo et al., NeurIPS 2023 Spotlight)
- **Authors**: Zongyu Guo, Gergely Flamich, Jiajun He, Zhibo Chen, José Miguel Hernández-Lobato
- **Venue**: NeurIPS 2023
- **arXiv**: https://arxiv.org/abs/2305.19185
- **Repo**: https://github.com/cambridge-mlg/combiner
- **Empirical claim**: Variational Bayesian INR + relative entropy coding; direct rate-distortion optimization via β-ELBO; bit-budget 16 bits/block.
- **Relevance**: Bayesian approach to bit allocation. Plugs into our `tac.bit_allocator`. Substrate-engineering scale.
- **Integration cost**: ~3 days dev.

## A.9 — RECOMBINER (Cambridge MLG, ICLR 2024)
- **Authors**: Jiajun He, Gergely Flamich, Zongyu Guo, José Miguel Hernández-Lobato
- **Venue**: ICLR 2024
- **arXiv**: https://arxiv.org/abs/2309.17182
- **Repo**: https://github.com/cambridge-mlg/RECOMBINER
- **Empirical claim**: Improves COMBINER via richer variational posterior + learnable positional encodings + hierarchical prior across patches. SOTA on CIFAR-10 low-bitrate.
- **Relevance**: TOP-20 EUREKA #13. Hierarchical Bayesian INR — natural fit for our substrate stacking.
- **Integration cost**: ~3 days dev.

## A.10 — COIN++ (Dupont et al., 2022)
- **Authors**: Emilien Dupont, Hrushikesh Loya, Milad Alizadeh, Adam Goliński, Yee Whye Teh, Arnaud Doucet
- **arXiv**: https://arxiv.org/abs/2201.12904
- **Repo**: https://github.com/EmilienDupont/coin (predecessor COIN)
- **Empirical claim**: Meta-learned base network + modulations as compressed code; 2 orders of magnitude faster encoding.
- **Relevance**: Modulations-as-sidecar is the lineage of ActINR (A.11). Reference for any meta-learned-base approach.
- **Integration cost**: ~2-3 days dev for meta-learning bootstrap.

## A.11 — ActINR / Bias-Modulation INR (Kayabasi et al., CVPR 2025)
- **Authors**: Alper Kayabasi, Anil Kumar Vadathya, Guha Balakrishnan, Vishwanath Saragadam
- **Venue**: CVPR 2025
- **arXiv**: https://arxiv.org/abs/2501.09277
- **Empirical claim**: Shared INR weights across frames + per-frame biases for motion; 10× slow-mo, 4× SR + 2× slow-mo, denoising, inpainting.
- **Relevance**: TOP-20 EUREKA #14. The "biases = motion location" insight is potentially **the obscure-1 win**. Replace PR106 latent sidecar with bias modulations.
- **Integration cost**: ~1.5 days dev (substrate-class atom; archive grammar update).

## A.12 — Rabbit NeRV (RNeRV; Khoury 2025)
- **arXiv**: https://arxiv.org/html/2506.24127 ("How to Design and Train Your Implicit Neural Representation for Video Compression")
- **Empirical claim**: Disentangled component review of NeRV family; new SOTA configuration; hyper-network for real-time encoding.
- **Relevance**: Reference for our HNeRV-family ablations. Hyper-network claim is a META-tool for our parallel-dispatch actuator.
- **Integration cost**: ~3-5 days dev.

## A.13 — WIRE (Saragadam et al., CVPR 2023) — REFERENCE
- **Authors**: Vishwanath Saragadam, Daniel LeJeune, Jasper Tan, Guha Balakrishnan, Ashok Veeraraghavan, Richard Baraniuk
- **Venue**: CVPR 2023
- **arXiv**: https://arxiv.org/abs/2301.05187
- **Repo**: https://github.com/vishwa91/wire
- **Project**: https://vishwa91.github.io/wire
- **Empirical claim**: Complex Gabor wavelet as INR activation; smallest + most spatially compact error.
- **Relevance**: Predecessor of FINER; activation-swap reference; especially good for noisy/sparse data fitting (potentially relevant to our pose sidecar).
- **Integration cost**: ~0.5 day dev (activation swap).

---

## Follow-up reads in domain A:
- NeRV++ (Hammoud et al., 2024): https://arxiv.org/html/2402.18305v1
- MoE-INR (Wang et al., VIS 2025): https://academicweb.nd.edu/~cwang11/papers/vis25-moeinr.pdf
- E-NeRV (Li et al., ECCV 2022): https://arxiv.org/abs/2207.08132
- FFNeRV (Lee et al., ACM MM 2023): https://maincold2.github.io/ffnerv/
- "A Survey of Implicit Neural Representations for Video Compression" (Riahi et al., TechRxiv 2024): https://www.techrxiv.org/users/931236/articles/1302259
- "How to Design and Train Your INR for Video Compression" (Khoury 2025): https://arxiv.org/abs/2506.24127
