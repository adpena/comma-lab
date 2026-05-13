# Online research ledger — Domain F: Brownian / SDE / Langevin training

Per-paper notes; 8 entries.

---

## F.1 — SGLD (Welling-Teh, ICML 2011; foundational)
- **Empirical claim**: Add scaled isotropic Gaussian noise to SGD; samples from posterior (Bayesian inference) and escapes saddle points (optimization).
- **Relevance**: Foundation for our scaffolded LangevinOptimizer.

## F.2 — Non-convex SGLD nonasymptotic analysis (Raginsky-Rakhlin-Telgarsky, 2017)
- **arXiv**: https://arxiv.org/abs/1702.03849
- **Empirical claim**: SGLD with proper temperature schedule provably converges to ε-near-global-optimum in poly time for non-convex.
- **Relevance**: Theoretical backbone for our use of LangevinOptimizer to escape SegNet argmax local minima.

## F.3 — Langevin Dynamics + Lyapunov Potentials Unified View (Mou-Liu-Tarokh 2024)
- **arXiv**: https://arxiv.org/abs/2407.04264
- **Empirical claim**: Lyapunov-based analysis; improved rates; first finite gradient-complexity guarantee for SGLD under Lipschitz + Poincaré.
- **Relevance**: Modern reference for tuning our LangevinOptimizer (temperature, step-size).

## F.4 — Non-Reversible SGLD (2020)
- **arXiv**: https://arxiv.org/abs/2004.02823
- **Empirical claim**: Time-irreversible variant; faster mixing.
- **Relevance**: Worth a smoke test against vanilla SGLD on our pose-marginal landscape.

## F.5 — Variance-Reduced SGLD (SVRG-LD, SRG-LD; 2022-2023)
- **arXiv**: https://arxiv.org/abs/2203.16217
- **Empirical claim**: SVRG/SRG control-variate variants of SGLD; improved convergence rate.
- **Relevance**: Practical efficiency win for any Langevin-based lane.

## F.6 — EDM Karras et al. (NeurIPS 2022)
- **arXiv**: https://arxiv.org/abs/2206.00364
- **Repo**: https://github.com/NVlabs/edm
- **Empirical claim**: Elucidated SDE design space for diffusion. Higher-order Runge-Kutta sampler; preconditioning of score networks. CIFAR-10 FID 1.79.
- **Relevance**: SDE framing of generative training. For us: the SDE view of training-as-Brownian-motion suggests using Heun/RK2 update steps for LangevinOptimizer (instead of Euler-Maruyama) — likely a free 2× sample-efficiency win.

## F.7 — Stochastic Karras VE (Hugging Face Diffusers reference impl.)
- **Link**: https://huggingface.co/docs/diffusers/v0.3.0/en/api/pipelines/stochastic_karras_ve
- **Relevance**: Reference implementation of EDM-style SDE sampling.

## F.8 — Analyzing and Improving Training Dynamics of Diffusion Models (Karras et al. 2024)
- **Empirical claim**: Follow-up to EDM with training-time analysis.
- **Relevance**: Latest from Karras' group; reference for SDE-as-training.

## F.9 — k-diffusion (Crowson; community reference impl.)
- **Repo**: https://github.com/crowsonkb/k-diffusion
- **Relevance**: Standard reference impl. for Karras-style training.

---

## SDE-optimizer design sketch

Our scaffolded LangevinOptimizer (session insight #7) currently uses Euler-Maruyama discretization. Domain-F bleeding edge suggests:

1. **Heun/RK2 step** for SGLD (Karras EDM Algorithm 2): predict + correct → 2× sample efficiency.
2. **Lyapunov-based temperature schedule** (F.3): provably faster mixing than constant-temp.
3. **Variance reduction** (F.5): control variates → 5-10× speedup if computable.
4. **Non-reversible drift term** (F.4): adds rotational drift → faster mixing on multimodal landscapes (likely our SegNet argmax landscape).

These four upgrades stack additively. Combined predicted gain: ~5-20× sample efficiency vs current Euler-Maruyama LangevinOptimizer. **[literature-prediction]**.

## Follow-up reads:
- Welling-Teh 2011 original SGLD paper (canonical reference)
- Chen-Fox-Guestrin Stochastic Gradient HMC (2014; sister method)
- SOC-MartNet (HJB+SDE 2024): https://www.researchgate.net/publication/383165352
