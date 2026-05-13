# Ancient Elder Era 6 ledger — Abandoned Paths (1972–1995)

**Companion master memo**: `.omx/research/ancient_elder_polymath_research_20260513.md` §6.

## History (one paragraph)

1972–1995 was the era of "great ideas, no compute". Hopfield 1982 (PNAS 79:2554) had a working associative-memory network — capacity 0.138·N patterns — but every demo was a 30-node toy. Smolensky's tensor product representations (1990, *AI* 46:159) had exponential blowup that 1990 hardware couldn't bear. Pollack's RAAM (1990, *AI* 46:77) produced beautiful tree representations on 50-symbol grammars then stalled. Plate's HRR (1995, *IEEE TNN* 6:623) was filed under "exotic NLP" — until Ramsauer 2020 showed that the underlying mathematics IS modern attention. **These ideas were never wrong — just compute-bound.**

## Key papers (5+)

| Citation | Year | Relevance |
|----------|------|-----------|
| Hopfield, "Neural networks and physical systems...", *PNAS* 79:2554, [DOI](https://doi.org/10.1073/pnas.79.8.2554) | 1982 | Attractor associative memory. |
| Plate, "Holographic Reduced Representations", *IEEE Trans. Neural Networks* 6:623, [DOI](https://doi.org/10.1109/72.377968), [Berkeley PDF](https://redwood.berkeley.edu/wp-content/uploads/2020/08/Plate-HRR-IEEE-TransNN.pdf) | 1995 | Circular-convolution binding. |
| Pollack, "Recursive distributed representations", *AI* 46:77 | 1990 | Tree-structured representations. |
| Smolensky, "Tensor product variable binding...", *AI* 46:159 | 1990 | Tensor-product bindings. |
| Gabor, "A new microscopic principle", *Nature* 161:777 | 1948 | Holography (distributed memory). |
| van Heerden, "Theory of optical information storage in solids", *Applied Optics* 2:393 | 1963 | Holographic data storage. |
| Shafer, *A Mathematical Theory of Evidence*, Princeton | 1976 | Dempster-Shafer belief functions. |
| Steinbuch, "Die Lernmatrix", *Kybernetik* 1(1):36 | 1961 | Associative memory before Hopfield. |
| Ramsauer et al., "Hopfield Networks is All You Need", [arXiv:2008.02217](https://arxiv.org/abs/2008.02217) | 2020 | **Modern Hopfield = transformer attention update rule.** |
| Łukasiewicz, many-valued logic (1920s) | 1920s | Fuzzy logic ancestor. |

## 5 ideas worth reviving (per master memo §6.2)

**AP-1** Plate HRR for relational latent representation. **Build**: 8-12 days. **$**: 5-10. **ΔS**: -0.002 to -0.004. Medium.

**AP-2** Modern Hopfield attractor decoder for noise-robust latents. **Build**: 10-14 days. **$**: 10-15. **ΔS**: speculative -0.020 (high variance). High-risk high-reward.

**AP-3** Gabor holography for distributed-memory frame storage. Not viable for contest packet; listed for non-contest production.

**AP-4** Smolensky tensor-product for compositional masks. Shared infrastructure with φ3 S2SBS.

**AP-5** Dempster-Shafer belief-function ensemble inflate.py. Speculative.

## Connection to our contest

Plate HRR could encode the **structural decomposition** of per-pair latents — explicit `ego_motion ⊛ pose + scene ⊛ mask` binding. This composition is exactly what the dashcam pose structure suggests.

Modern Hopfield with continuous states has exponential capacity (Demircigil-Salavati 2017). For our 600-pair latents, even a few dozen attractors per pair would give the noise-tolerance needed for aggressive bit reduction.

## What's changed since 1972–1995

- Ramsauer 2020 proved modern Hopfield (continuous states) = transformer attention. Same mathematics.
- Compute makes 4096-d HRR vectors trivial (1995: 256-d was painful).
- Differentiable circular convolution via FFT (Brent-Kung 1981) is standard PyTorch.

## Reactivation criteria

AP-1 / AP-2 should be built in parallel as a "novel-representation" arm — pick whichever is empirically better; both are HNeRV-family-orthogonal.

AP-3 / AP-5 reactivate for non-contest production targets (`comma_ai_production` / `production_edge_adaptive` per CLAUDE.md "Contest vs production target modes" section).
