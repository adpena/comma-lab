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
- A full exact previous-symbol encode over the finished `frame_major` split-0 stream is now also on disk:
  - encoded artifact: `reports/raw/2026-04-11-commavq-gpt-arithmetic/small/train_split0.tpc`
  - metadata: `reports/raw/2026-04-11-commavq-gpt-arithmetic/small/train_split0_prev_symbol_encode.json`
  - original bytes: `774005000`
  - encoded bytes: `473320648`
  - compression ratio: `1.635265656105499`
  - contexts: `1026`
  - outer header bytes: `4116`
- A full exact previous-pair benchmark over the finished `position_major` split-0 stream is now on disk:
  - metadata: `reports/raw/2026-04-11-commavq-gpt-arithmetic/position_major/train_split0_prev_pair_benchmark_full.json`
  - original bytes: `768645000`
  - encoded bytes: `278032654`
  - compression ratio: `2.7645853425547635`
  - contexts: `986185`
- A full exact single-file `zstd_dict` benchmark over the finished `position_major` split-0 stream is now on disk:
  - encoded artifact: `reports/raw/2026-04-11-commavq-gpt-arithmetic/position_major/train_split0.zst`
  - original bytes: `768645000`
  - encoded bytes: `393059915`
  - compression ratio: `1.95554156164716`
  - dictionary bytes: `131072`
  - sample count after block splitting: `11730`
- A full exact single-file `zstd_dict` benchmark over the finished `frame_major` split-0 stream is also on disk:
  - encoded artifact: `reports/raw/2026-04-11-commavq-gpt-arithmetic/small/train_split0.zst`
  - original bytes: `774005000`
  - encoded bytes: `441684503`
  - compression ratio: `1.7523933820245443`
  - dictionary bytes: `131072`
  - sample count after block splitting: `11812`
- The first official commavq GPT local-only score sample had to be invalidated after scorer fixes:
  - the earlier sample used the wrong default context (`256` instead of the official `2580`)
  - the fast scorer dropped sliding context across chunk boundaries
  - sampled-run accounting overstated consumed segment tokens
  - a corrected rerun is in progress and should repopulate `reports/raw/2026-04-11-commavq-gpt-arithmetic/small/train_split0_gpt_score_8192.json`
- Reordering alone is not enough, but reordering plus simple conditional coding is already materially better than the static coder on the same sample.

## next implication

- `position_major + previous-symbol conditioning` is now the strongest empirical direction in the lossless arithmetic lane.
- The full exact file-level comparison now matches the earlier sample signal: `position_major` is dramatically better than `frame_major` under the same previous-symbol conditional coder.
- Increasing the conditioning order naively is not automatically better:
  - `position_major + prev-pair` is worse than `position_major + prev-symbol`
  - likely cause: context explosion and fragmented statistics (`986185` contexts)
- Classical shared-dictionary compression is a viable local comparator but still well behind the strongest conditional token coder:
  - `position_major + zstd_dict` is better than `frame_major + zstd_dict`
  - but both are much worse than `position_major + prev-symbol`
- The official GPT lane is now real and local, but the last published sample was invalidated and is being rerun under the corrected scorer.
- The first durable backend probe is now on disk:
  - metadata: `reports/raw/2026-04-11-commavq-gpt-arithmetic/small/train_split0_gpt_score_probe_64.json`
  - `64` tokens scored at the official `2580` context
  - CPU and MPS match on bits/token (`9.263728540522601`)
  - MPS is faster on this probe (`2.355s` vs `4.006s`)
- The stronger official GPT scorer sample is now also on disk:
  - metadata: `reports/raw/2026-04-11-commavq-gpt-arithmetic/small/train_split0_gpt_score_2048_mps.json`
  - device: `mps`
  - scored tokens: `2048`
  - bits/token: `1.7371709589676005`
  - perplexity: `3.333807860550116`
  - this is the strongest measured official-GPT signal in the repo so far
- The exact local-only `zpaq` rerun is now complete:
  - metadata: `reports/raw/2026-04-11-commavq-zpaq-baseline-rerun/zpaq_baseline_result.json`
  - archive bytes: `538876316`
  - original bytes: `960000000`
  - exact match: `5000/5000`
  - compression ratio: `1.7814848630311673`
  - this beats the promoted `lzma` floor locally, but remains `local_only`
- The next exact implementation cut should build on this `position_major` file-level conditional coder signal, not on more source-order experiments.
