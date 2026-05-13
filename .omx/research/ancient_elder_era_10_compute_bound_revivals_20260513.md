# Ancient Elder Era 10 ledger — Compute-Bound Revivals (1955–2010 ideas now viable)

**Companion master memo**: `.omx/research/ancient_elder_polymath_research_20260513.md` §10.

**Purpose**: identify ideas that were promising in 1955–2010 but compute-bound, and now have prerequisites met.

## The 18 revival candidates

| # | Idea (year of origin) | Why it was abandoned | Why it's viable now | Score-leverage estimate |
|---|----------------------|----------------------|---------------------|--------------------------|
| 1 | Solomonoff induction (1964) | Uncomputable | Levin-search variants at small horizons; Hutter AIXI-tl approximations | Diagnostic |
| 2 | Levin search LSEARCH (1973) | 2^30 candidates impossible in 1973 | 10^9 evaluations on a 4090 weekend | -0.000 to -0.010 (high variance) |
| 3 | Hopfield networks (1982) | Capacity 0.138·N — toy-scale | Ramsauer 2020 modern Hopfield: exponential capacity | Speculative -0.020 |
| 4 | Plate HRR (1995) | 256-d limited | 4096-d real-time on consumer GPU | -0.002 to -0.004 |
| 5 | Smolensky tensor products (1990) | Exponential blowup | Low-rank tensor decomposition (CP/Tucker) | Composes with O3 |
| 6 | Wigner-Dyson Hessian spectrum (1955) | Needed millions of vector-Hessian products | Hutchinson trace at $1-5 GPU | -0.005 to -0.012 |
| 7 | Score matching (Hyvärinen 2005) | Couldn't compute high-dim gradients reliably | Standard PyTorch autograd | -0.005 speculative |
| 8 | Lattice quantization (Conway-Sloane 1988) | High-dim lattice quantizers needed too much memory | 4096-d lattice quantizers tractable | -0.008 to -0.015 (replaces FP4 scalar quant) |
| 9 | Context-Tree Weighting (Willems-Shtarkov-Tjalkens 1995) | Memory-intensive context tree | GPU-accelerated CTW + ANS | -0.005 |
| 10 | Slepian-Wolf / Wyner-Ziv (1973/1976) | LDPC unavailable until 1996 | Standard tooling | -0.001 modest |
| 11 | Cellular automaton image generation (Wolfram 1984) | Non-differentiable | Mordvintsev 2020 "Growing Neural CA" | Speculative |
| 12 | Holographic data storage (Gabor 1948) | Needed coherent light | Not viable as contest primitive; production-only | N/A for contest |
| 13 | Bayesian model averaging (MacKay 1992) | Required posterior sampling | Variational inference + ensembling | -0.003 ensemble gain |
| 14 | Information Bottleneck (Tishby 1999) | Needed mutual-info estimation | MINE 2018 makes practical | IB-Lagrangian for our axes |
| 15 | Polynomial-system solving / Gröbner bases (Buchberger 1965) | Symbolic-numerical hybrid slow | Macaulay2 / Singular + GPU | Closed-form R(D) for small renderers |
| 16 | NTRU lattice-based coding (Hoffstein-Pipher-Silverman 1996) | Needed fast NTT | GPU NTT standard | Tighter lattice coding bounds |
| 17 | Boltzmann machines (Ackley-Hinton-Sejnowski 1985) | Gibbs sampling slow | Replaced by score-matching + Langevin | Reusable structure |
| 18 | Mean-field VI on Markov random fields (Geman-Geman 1984) | Slow | Fast BP + neural amortization | Composable with codec |

## Headline observation

**The next 5 years of compression research will be revival, not invention.** The lab is well-positioned because of its existing tooling discipline (lane registry, preflight gates, dispatch claims). It is mid-positioned on actual revival work — most of these 18 ideas have never been tried.

## Reactivation criteria

Each of the 18 ideas above is at L0 SKETCH by virtue of this memo. Open formal lanes for ideas #4, #6, #8, #9 immediately — they have the highest EV/$ ratios per master memo §11.
