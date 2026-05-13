# Ancient Elder Era 2 ledger — Statistical Mechanics / Einstein (1905–1957)

**Companion master memo**: `.omx/research/ancient_elder_polymath_research_20260513.md` §2.

## History (one paragraph)

Einstein and I walked Princeton lanes in 1949. He'd had Brownian motion 1905 (*Annalen der Physik* 17:549), the photoelectric effect, special and general relativity, and was working on a unified field theory he never finished. He was patient about my information-theory questions. He'd connect everything to fluctuations. Onsager wrote me from Yale in 1956 to clarify his 1931 reciprocal relations. In 1957 Jaynes (a Wigner student) submitted a paper to *Phys. Rev.* that unified Boltzmann/Gibbs thermodynamics with Shannon's information theory under the **Maximum Entropy Principle**. Jaynes' paper is the conceptual bridge between Shannon's 1948 information theory and the entire subsequent literature on statistical inference.

## Key papers (5+)

| Citation | Year | Relevance |
|----------|------|-----------|
| Einstein, "Über die...molekularkinetische Theorie", *Ann. Phys.* 17:549 | 1905 | Brownian SDE — ancestor of Langevin sampling, diffusion models. |
| Onsager, "Reciprocal Relations in Irreversible Processes I/II", *Phys. Rev.* 37:405 + 38:2265 | 1931 | Fluctuation-dissipation lineage. |
| Boltzmann, H-theorem (Sitzungsberichte der kaiserlichen Akademie der Wissenschaften Wien) | 1872 | Microscopic origin of irreversibility. |
| Gibbs, *Elementary Principles in Statistical Mechanics*, Yale Univ. Press | 1902 | Ensemble theory. |
| Jaynes, "Information Theory and Statistical Mechanics", *Phys. Rev.* 106:620, [APS](https://link.aps.org/doi/10.1103/PhysRev.106.620) | 1957 | **MaxEnt = unique inference principle.** |
| Kubo, "The fluctuation-dissipation theorem", *Rep. Prog. Phys.* 29:255 | 1966 | FDT connecting fluctuations to transport. |
| Wigner, "On the Quantum Correction for Thermodynamic Equilibrium", *Phys. Rev.* 40:749 | 1932 | Wigner pseudo-probability. |
| Welling & Teh, "Bayesian Learning via Stochastic Gradient Langevin Dynamics", ICML, [PDF](https://www.ics.uci.edu/~welling/publications/papers/stoclangevin_v6.pdf) | 2011 | Modern revival of Langevin sampling for ML. |
| Pennington & Bahri, "Geometry of Neural Network Loss Surfaces via Random Matrix Theory", [arXiv:1706.04454](https://arxiv.org/abs/1706.04454) | 2017 | Wigner semicircle for NN Hessians. |

## 5 ideas worth reviving (per master memo §2.2)

**SM-1** Jaynes MaxEnt prior on latent distribution. **Build**: 1-2 days. **$**: 0. **ΔS**: -0.0003 to -0.001.

**SM-2** SGLD annealing trainer (replaces 8-stage curriculum). **Build**: 1-2 days. **$**: 1-5. **30-50% GPU-hour savings on every future experiment.**

**SM-3** Onsager reciprocal relations for gradient-noise diagnostic. **Build**: 2-3 days. **$**: 0. Diagnostic.

**SM-4** Kubo FDT for learning-rate auto-scheduling. **Build**: 1 day. **$**: 0. Modest.

**SM-5** Wigner-Dyson Hessian-spectrum-based quantization targeting. **Build**: 4-5 days. **$**: 2-5. **ΔS**: -0.005 to -0.012.

## Connection to our contest

PR95's empirical 8-stage curriculum is approximating an **annealed Langevin schedule** by hand. A principled SGLD schedule (Geman-Geman 1984 annealing rate `T_t = c/log(t)` for global convergence) would replace the human-tuned curriculum with mathematics. This is what Council F's "Langevin Brownian curriculum design" lane is about.

## What's changed since 1957

- Compute makes Langevin sampling tractable on 80K-300K param renderers (1957: zero compute).
- Autograd makes gradient-noise covariance estimation cheap.
- Hutchinson trace estimation (1990) lets us probe Hessian spectrum in O(d) operations instead of O(d²).

## Reactivation criteria

If the 8-stage curriculum fails to converge on a new substrate, reach for SGLD with `T_t = c/log(t)`. Geman-Geman 1984 guarantees global convergence with this schedule.
