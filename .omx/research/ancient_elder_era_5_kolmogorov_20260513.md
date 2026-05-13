# Ancient Elder Era 5 ledger — Kolmogorov / Algorithmic Information (1964–2005)

**Companion master memo**: `.omx/research/ancient_elder_polymath_research_20260513.md` §5.

## History (one paragraph)

Kolmogorov visited Princeton in 1962. He didn't speak much English but sent preprints in Russian until 1969 English translations. The big insight: complexity of `x` = length of the shortest program that outputs `x`. Three independent inventors: Solomonoff 1964, Kolmogorov 1965, Chaitin 1975. Then Rissanen 1978 gave us the **practical, computable** cousin: Minimum Description Length (MDL). Then Levin 1973 gave us LSEARCH — an optimal but exponentially slow search algorithm. Then Hutter 2005 packaged it all into AIXI, the theoretically-optimal universal agent (intractable but illuminating).

## Key papers (5+)

| Citation | Year | Relevance |
|----------|------|-----------|
| Solomonoff, "A Formal Theory of Inductive Inference, Parts I and II", *Information and Control* 7:1 + 7:224, [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0019995864902232) | 1964 | Algorithmic probability; universal prior 2^(-K). |
| Kolmogorov, "Three approaches to the quantitative definition of information", *Problems of Information Transmission* 1:1 | 1965 | Complexity definition. |
| Levin, "Universal sequential search problems", *Problems of Information Transmission* 9:265 | 1973 | Levin search. |
| Martin-Löf, "The definition of random sequences", *Information and Control* 9:602 | 1966 | Algorithmic randomness. |
| Chaitin, "A theory of program size formally identical to information theory", *J. ACM* 22:329 | 1975 | Chaitin's omega; halting probability. |
| Rissanen, "Modeling by shortest data description", *Automatica* 14:465 | 1978 | **MDL — the practical bridge to working compressors.** |
| Hutter, *Universal Artificial Intelligence*, Springer | 2005 | AIXI. |
| Vovk, Solomonoff prediction theory | 1998 onward | Online sequence prediction with universal bounds. |

## 3 ideas worth reviving (per master memo §5.2)

**KC-1** MDL-optimal renderer architecture search. **Build**: 2-3 days. **$**: 0. **Provides explicit decoder-vs-latent byte split optimization.**

**KC-2** Levin search (LSEARCH) for codec primitive discovery (≤ 200 byte primitives). **Build**: 10-14 days. **$**: 5-15. High variance.

**KC-3** Algorithmic mutual information for cross-frame redundancy estimation (LZMA-based estimate of `K(pair_i, pair_j) - K(pair_i) - K(pair_j)`). **Build**: 1-2 days. **$**: 0. **Rank 8 by EV/$ — cheapest diagnostic in memo.**

## Connection to our contest

MDL says: choose the architecture and codec jointly to minimize `bits(decoder) + bits(latents | decoder)`. PR101 does this empirically; **a proper MDL solver could rank N candidate architectures pre-dispatch**, saving the lab GPU hours that would otherwise be spent on dominated configurations.

Levin search is the formal cousin of "try every 200-byte primitive" — for small primitive sizes, it's now compute-feasible.

## What's changed since 1964–2005

- Compute makes Levin search with k ≤ 30 bits feasible (2^30 ≈ 10^9 candidates).
- Modern compressors (PAQ, cmix, LZMA2) approximate K(x) tightly enough for diagnostics.
- Variational MDL (Hinton-van Camp 1993) bridges to neural-net Bayesian inference.

## Reactivation criteria

KC-3 is so cheap it should run NOW. KC-1 reactivates the moment a new architecture is proposed. KC-2 reactivates if the lab has unclaimed GPU budget that would otherwise idle.
