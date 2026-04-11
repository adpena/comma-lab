# Lossless Factorization Comparison

## split 0 static-frequency baseline

- `frame_major` source: `reports/raw/2026-04-11-commavq-gpt-arithmetic/small/train_split0.bin`
- `frame_major` encoded: `reports/raw/2026-04-11-commavq-gpt-arithmetic/small/train_split0.tfc`
- `frame_major` encoded bytes: `479100091`
- `frame_major` empirical bits/token: `9.873739374489201`

- `position_major` source: `reports/raw/2026-04-11-commavq-gpt-arithmetic/position_major/train_split0.bin`
- `position_major` encoded: `reports/raw/2026-04-11-commavq-gpt-arithmetic/position_major/train_split0.tfc`
- `position_major` encoded bytes: `476398563`
- `position_major` empirical bits/token: `9.886363473549515`

## conclusion

- `position_major` is worse on simple marginal entropy.
- `position_major` is better on the exact static-frequency coded bytes by about `2.7 MB` on split `0`.
- A sampled previous-symbol conditional benchmark over `2,000,000` tokens is much stronger on `position_major` than `frame_major`.
  - `frame_major`: `3,449,158` encoded bytes, ratio `1.1597`
  - `position_major`: `1,414,089` encoded bytes, ratio `2.8287`
- A full exact previous-symbol encode over the finished `position_major` split-0 stream is now on disk:
  - encoded artifact: `reports/raw/2026-04-11-commavq-gpt-arithmetic/position_major/train_split0.tpc`
  - metadata: `reports/raw/2026-04-11-commavq-gpt-arithmetic/position_major/train_split0_prev_symbol_encode.json`
  - original bytes: `768645000`
  - encoded bytes: `232173020`
  - compression ratio: `3.3106559926730506`
  - contexts: `1026`
  - outer header bytes: `4115`
- Reordering alone is not enough, but reordering plus simple conditional coding is already materially better than the static coder on the same sample.

## next implication

- `position_major + previous-symbol conditioning` is now the strongest empirical direction in the lossless arithmetic lane.
- The next exact implementation cut should build on this real file-level conditional coder signal, not on more source-order experiments.
- The most useful comparison still missing is a full exact `frame_major` previous-symbol encode on the same split-0 stream.
