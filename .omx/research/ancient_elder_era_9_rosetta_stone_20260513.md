# Ancient Elder Era 9 ledger — Rosetta Stone of Rediscovery

**Companion master memo**: `.omx/research/ancient_elder_polymath_research_20260513.md` §9.

**Purpose**: humility reference. For 25 modern techniques, the older paper that originated the core idea. Use this when future councils debate "novelty" of a proposed primitive.

## The full table

| Modern technique | Originator (first to derive the math) | Year | Citation |
|------------------|---------------------------------------|------|----------|
| Transformer attention | Hopfield networks (modern Hopfield = attention per Ramsauer 2020) | 1982 | Hopfield, *PNAS* 79:2554; Ramsauer, [arXiv:2008.02217](https://arxiv.org/abs/2008.02217) |
| Mixture-of-experts | Selfridge's Pandemonium | 1958 | Selfridge, *HMSO 1959* |
| KL distillation | Kullback-Leibler "On Information and Sufficiency" | 1951 | *Ann. Math. Statist.* 22:79 |
| Diffusion models | Score matching + Brownian motion + nonequilibrium thermo | 1905/2005/2015 | Einstein 1905; Hyvärinen 2005; Sohl-Dickstein 2015 |
| LoRA / DoRA | Householder additive low-rank factorization | 1958 | Householder, *J. ACM* 5:339 |
| Variational autoencoder | Helmholtz machine + variational inference | 1995/1999 | Dayan-Hinton-Neal-Zemel, *Neural Computation* 7:889; Jordan-Ghahramani-Jaakkola-Saul 1999 |
| GAN | Schmidhuber predictability minimization | 1992 | *Neural Computation* 4:863 |
| ResNet skip connections | Highway networks + additive identity shortcuts | 1995/2015 | Bishop 1995 textbook; Srivastava 2015 |
| Batch normalization | LeCun-Bottou-Orr-Müller whitening | 1998 | "Efficient BackProp" |
| Adam optimizer | Polyak momentum + Rprop + RMSProp | 1964/1993/2012 | Polyak 1964 *USSR Comp. Math.* 4:1; Riedmiller-Braun 1993; Tieleman-Hinton 2012 |
| Muon (Newton-Schulz iteration) | Schulz iteration for matrix inverse | 1933 | *Z. Angew. Math. Mech.* 13:57; Higham SIAM 2008 |
| Sparse attention (Longformer, BigBird) | Selfridge cognitive-demon selective attention | 1958 | Selfridge HMSO 1959 |
| Neural ODE | Implicit Euler stepping in differentiable solvers | 1982 | Petzold *SIAM J. Numer. Anal.* |
| Normalizing flows | Box-Cox transformations + Jacobian determinant | 1964/2010 | *J. Royal Stat. Soc. B* 26:211; Tabak-Vanden-Eijnden 2010 |
| Contrastive learning (InfoNCE) | Becker-Hinton self-organizing network | 1992 | *Nature* 355:161 |
| Sinusoidal position embedding | Fourier series | 1822 | Fourier, *Théorie analytique de la chaleur* |
| Layer normalization | Pearson z-score | 1894 | Pearson *Phil. Trans. R. Soc.* |
| GELU / Swish activation | Sigmoid (Cox logistic) + ReLU (Fukushima neocognitron) | 1958/1969 | Cox 1958; Fukushima 1969 |
| ANS differentiable arithmetic coding | Rissanen arithmetic coding + Duda ANS | 1976/2009 | Rissanen 1976 *IBM J. Res. Dev.*; Duda [arXiv:0902.0271](https://arxiv.org/abs/0902.0271) |
| Hyperprior (Ballé 2018) | Mixture-of-Gaussians source models + Gibbs ensemble | 1902/1995 | Gibbs 1902; Bishop 1995 |
| RLHF (PPO + reward model) | Wiener cybernetic feedback control | 1948 | Wiener *Cybernetics* |
| Chain-of-thought / scratchpad | McCarthy Lisp + Polya *How to Solve It* | 1945/1960 | Polya 1945; McCarthy 1960 *Comm. ACM* 3:184 |
| In-context learning | Tversky-Kahneman heuristic-by-analogy | 1974 | *Science* 185:1124 |
| Diffusion classifier-free guidance | Importance sampling (von Neumann-Ulam) | 1949 | von Neumann-Ulam Monte Carlo |
| Mamba / state-space models | Kalman filter + HMMs | 1960/1966 | Kalman 1960 *J. Basic Eng.* 82:35; Baum-Petrie 1966 *Ann. Math. Statist.* 37:1554 |
| Implicit neural representations (NeRF, SIREN) | Radial basis function networks + Kolmogorov-Arnold | 1957/1988 | Kolmogorov-Arnold *Doklady* 114:953; Broomhead-Lowe 1988 *Complex Systems* 2:321 |
| Modern entropy bottleneck (compressive autoencoder) | Tishby information bottleneck | 1999 | Tishby et al. *Proc. 37th Allerton* |
| Self-supervised representation learning | Schmidhuber predictability minimization + Becker-Hinton stereogram | 1992 | (above) |

## Implication

The lab should NOT measure progress by "how many novel ideas have we proposed". Every "novel" idea has a 30-60 year ancestor. **Measure progress by "how many old ideas have we actually implemented and measured on the contest scorer".**

## Reactivation criteria

Use this table at every council deliberation when a "novel" primitive is proposed. Find the older paper; check if the modern derivation is BETTER (clearer, more general) than the original. Often it isn't.
