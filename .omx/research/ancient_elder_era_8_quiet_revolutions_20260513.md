# Ancient Elder Era 8 ledger — Quiet Revolutions (1995–2010)

**Companion master memo**: `.omx/research/ancient_elder_polymath_research_20260513.md` §8.

## History (one paragraph)

These were the quiet years on the academic stage but in retrospect the most important compression and generative-modeling primitives were laid down here. Hyvärinen 2005 invented score matching (training `∇_x log p(x)` without computing `Z(θ)`). Cuturi 2013 made Wasserstein differentiable. Sohl-Dickstein 2015 — using Hyvärinen + Einstein 1905 Brownian — invented diffusion models, which then took ~5 more years to dominate generative AI. Tabak-Vanden-Eijnden 2010 sketched the normalizing-flow lineage.

## Key papers (5+)

| Citation | Year | Relevance |
|----------|------|-----------|
| Hyvärinen, "Estimation of Non-Normalized Statistical Models by Score Matching", *JMLR* 6:695, [JMLR](https://jmlr.org/papers/v6/hyvarinen05a.html) | 2005 | **Score matching — diffusion ancestor.** |
| Hinton, "Training Products of Experts by Minimizing Contrastive Divergence", *Neural Computation* 14:1771 | 2002 | Energy-based model training. |
| Tabak & Vanden-Eijnden, "Density estimation by dual ascent of the log-likelihood", *Comm. Math. Sci.* 8:217 | 2010 | Normalizing flow precursor. |
| Cuturi, "Sinkhorn Distances: Lightspeed Computation of Optimal Transport", *NeurIPS*, [link](https://papers.nips.cc/paper_files/paper/2013/hash/af21d0c97db2e27e13572cbf59eb343d-Abstract.html) | 2013 | **Differentiable Wasserstein.** |
| Sohl-Dickstein, Weiss, Maheswaranathan, Ganguli, "Deep Unsupervised Learning using Nonequilibrium Thermodynamics", *ICML*, [PMLR](https://proceedings.mlr.press/v37/sohl-dickstein15.html), [arXiv:1503.03585](https://arxiv.org/abs/1503.03585) | 2015 | Diffusion model. |
| Tishby, Pereira, Bialek, "The Information Bottleneck Method", *Proc. 37th Allerton Conf.* | 1999 | IB Lagrangian. |
| Belghazi et al., "MINE: Mutual Information Neural Estimation", [arXiv:1801.04062](https://arxiv.org/abs/1801.04062) | 2018 | Tractable MI estimation. |

## 3 ideas worth reviving (per master memo §8.2)

**QR-1** Score matching as proxy training objective. **Build**: 1-2 weeks. **$**: 5-10. Speculative — could close proxy-auth gap.

**QR-2** Sinkhorn-Wasserstein for differentiable scorer-equivalence-class targeting (council F triplet E C3). **Build**: 5-7 days. **$**: 0-5. Already partially considered by Council F.

**QR-3** Contrastive divergence for fast posterior approximation in MaxEnt latent coding. **Build**: 3-4 days. **$**: 0-3. Composes with SM-1.

## Connection to our contest

Score matching is the right framing for proxy-auth gap closure: instead of pixel MSE → SegNet/PoseNet, match the score gradient `∂L/∂x̂` directly. Could give crisper proxy training signal.

Sinkhorn-Wasserstein gives us a differentiable distance-to-nearest-equivalence-class-member, which is exactly the φ1 SABOR / φ3 S2SBS objective.

## What's changed since 1995–2010

- Compute makes high-dim score estimation tractable (2005: hard; 2026: standard).
- Differentiable Sinkhorn (Cuturi 2013) makes Wasserstein training feasible.
- MINE (Belghazi 2018) makes mutual-information estimation tractable for IB.

## Reactivation criteria

QR-1 reactivates if the proxy-auth gap persists after differentiable scorer-preprocess + eval_roundtrip fixes (currently CLAUDE.md non-negotiables). The gap-closer of last resort.
