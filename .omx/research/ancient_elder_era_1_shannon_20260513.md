# Ancient Elder Era 1 ledger — Shannon Era (1948–1970)

**Companion master memo**: `.omx/research/ancient_elder_polymath_research_20260513.md` §1.

## History (one paragraph)

I sat in the corridor at Bell Labs Building 1 in 1947 watching Claude juggle. Bob Fano's PhD students were running coding experiments at MIT. By 1948 we had the full source-channel duality. Huffman's 1951 PhD thesis at MIT delivered the optimal prefix code (under MIT's Bob Fano). The 1959 paper on rate-distortion is where Claude proved that for ANY distortion measure D, there exists R(D) bits/sample below which D cannot be achieved. The 1970s saw the network-coding extensions (Slepian-Wolf 1973, Wyner-Ziv 1976) — they handled correlated sources and decoder-side information, both of which are EXACTLY our contest's structure.

## Key papers (5+)

| Citation | Year | Relevance |
|----------|------|-----------|
| Shannon, "A Mathematical Theory of Communication", *Bell System Tech. J.* 27:379, [IEEE](https://ieeexplore.ieee.org/document/6773024) | 1948 | Source + channel coding theorems. |
| Huffman, "Method for the construction of minimum-redundancy codes", *Proc. IRE* 40:1098, [IEEE](https://ieeexplore.ieee.org/document/4051119) | 1952 | Optimal prefix code; within 1 bit of H(X). |
| Shannon, "Coding theorems for a discrete source with a fidelity criterion", *IRE Nat. Conv. Record* | 1959 | **R(D) for vector-valued distortion — THE bound for our contest.** |
| Slepian & Wolf, "Noiseless coding of correlated information sources", *IEEE Trans. IT* 19:471, [DOI](https://doi.org/10.1109/TIT.1973.1055037) | 1973 | Distributed source coding — encode separately, decode jointly. |
| Wyner & Ziv, "Rate-distortion with side information at decoder", *IEEE Trans. IT* 22:1, [IEEE](https://ieeexplore.ieee.org/document/1055508) | 1976 | Decoder side info — exactly our temporal-frame structure. |
| Pasco, "Source coding algorithms", PhD thesis Stanford | 1976 | Arithmetic coding. |
| Rissanen, "Generalized Kraft inequality and arithmetic coding", *IBM J. Res. Dev.* 20:198 | 1976 | Arithmetic coding (independent). |
| Willems, Shtarkov, Tjalkens, "The context-tree weighting method", *IEEE Trans. IT* 41:653, [IEEE](https://ieeexplore.ieee.org/document/382012) | 1995 | Near-optimal universal CTW estimator. |

## 5 ideas worth reviving (per master memo §1.2)

**SE-1** Wyner-Ziv decoder-side-info latent codec. **Build**: 3-5 days. **$ dispatch**: 0. **ΔS**: -0.001.

**SE-2** Rissanen MPM / CTW conditional arithmetic coder on latent stream. **Build**: 4-7 days. **$**: 0. **ΔS**: -0.005.

**SE-3** Universal LZMA2 baseline as preflight check (process discipline). **Build**: <1 day. **$**: 0. **ΔS**: 0 (diagnostic).

**SE-4** JSCC — scorer-conditional entropy coder. **Build**: 5-8 days. **$**: 0-5. **ΔS**: -0.012. **RANK 1**.

**SE-5** Shannon-Fano-Elias fixed-block coder for runtime parallelism. **Build**: 2-3 days. **$**: 0. **ΔS**: 0 (operational margin).

## Connection to our contest

The contest scorer is a **vector-valued distortion** (d_seg, d_pose, rate). Shannon 1959 explicitly handles vector D. **NOBODY in the lab has cited Shannon 1959 for this purpose.** The R(D₁, D₂, D₃) frontier IS the theoretical floor.

## What's changed since 1948–1976

- Arithmetic coding is now standard (1976→2009 ANS).
- LDPC codes (Gallager 1962, rediscovered MacKay-Neal 1996) make Slepian-Wolf binning trivial.
- Differentiable arithmetic coders (Townsend 2018) make end-to-end-trained JSCC viable.
- Compute makes context-tree weighting at depth 8 cheap.

## Reactivation criteria

Re-open IDEA AP-1 / AP-3 lane if a sub-0.18 substrate lands; revisit Wyner-Ziv binning for cross-frame residual coding then.
