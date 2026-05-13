# Ancient Elder Era 7 ledger — Convex Analysis / Optimization (1933–2011)

**Companion master memo**: `.omx/research/ancient_elder_polymath_research_20260513.md` §7.

## History (one paragraph)

Dick Dykstra and I corresponded 1980–1985. His algorithm (1983) extended von Neumann's 1933 alternating-projections theorem to non-orthogonal projections in Hilbert space — exactly what's needed for **convex feasibility** = "find a point in the intersection of K convex sets". Bregman 1967 gave us generalized projections via Bregman divergence (modern mirror descent). Gabay-Mercier 1976 derived ADMM (originally for finite-element PDEs). Nesterov 1983 gave us accelerated gradient (O(1/k²) instead of O(1/k)). Boyd et al. 2011 wrote the canonical modern ADMM monograph.

## Key papers (5+)

| Citation | Year | Relevance |
|----------|------|-----------|
| von Neumann, "Functional Operators Vol II", lecture notes (published 1950 Princeton) | 1933 | Alternating projections onto two subspaces. |
| Bregman, "The relaxation method...", *USSR Comp. Math. Math. Phys.* 7:200 | 1967 | Bregman divergences. |
| Gabay & Mercier, "A dual algorithm for...nonlinear variational problems", *Comp. & Math. App.* 2:17 | 1976 | **ADMM birth.** |
| Polyak, "Some methods of speeding up the convergence of iteration methods", *USSR Comp. Math. Math. Phys.* 4:1 | 1964 | Heavy-ball momentum (Adam ancestor). |
| Nesterov, "A method of solving a convex programming problem with convergence rate O(1/k²)", *Soviet Math. Doklady* 27:372 | 1983 | **Nesterov acceleration.** |
| Dykstra, "An algorithm for restricted least squares regression", *J. Amer. Stat. Assoc.* 78:837 | 1983 | Dykstra's algorithm. |
| Nemirovski & Yudin, *Problem Complexity and Method Efficiency in Optimization*, Wiley | 1983 | Mirror descent. |
| Boyd, Parikh, Chu, Peleato, Eckstein, "Distributed Optimization and Statistical Learning via ADMM", *FnT-ML* 3:1 | 2011 | **Modern canonical reference.** |

## 3 ideas worth reviving (per master memo §7.2)

**OP-1** True iterative-consensus ADMM for meta-Lagrangian solver. **Build**: 5-7 days. **$**: 0. Enables better bit allocation across seg/pose/rate (ΔS via better allocation -0.005).

**OP-2** Bregman-divergence mirror descent / Itakura-Saito for entropy-coding mass functions. **Build**: 1 day. **$**: 0. Modest.

**OP-3** Nesterov-accelerated SGD (NAdamW vs AdamW). **Build**: 1-line swap. **$**: 0. 10-15% fewer epochs.

## Connection to our contest

Catalog #94 (`check_admm_naming_matches_iterative_consensus_implementation`) currently flags 27 named-ADMM files in the codebase that are actually Lagrangian + bisection. **True ADMM would be a genuine upgrade** — it converges in 30-50 iterations on convex problems with provable Pareto-frontier coverage.

This is also Boyd's specialty in the grand council; he would endorse OP-1 immediately.

## What's changed since 1933–2011

- GPU acceleration makes 100-200 ADMM iterations on 80K-300K parameter problems take seconds.
- Differentiable ADMM (Boyd's modern unrolled-ADMM papers, 2017+) makes ADMM steps trainable.
- NAdam (Dozat 2016, [https://openreview.net/forum?id=OM0jvwB8jIp57ZJjtNEZ](https://openreview.net/forum?id=OM0jvwB8jIp57ZJjtNEZ)) is standard PyTorch.

## Reactivation criteria

OP-1 reactivates the moment the meta-Lagrangian solver shows convergence struggles (currently uses approximate methods; Boyd repeatedly flags this in council).
OP-3 should fire immediately at the next opportunity — it's free.
