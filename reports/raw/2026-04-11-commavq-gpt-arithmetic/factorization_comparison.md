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
- Reordering alone is not enough; the next meaningful gain likely comes from conditional coding on top of the prepared streams.
