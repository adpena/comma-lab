# PR101 FEC7 Selector Entropy Profile

- score_claim: `false`
- dispatch_attempted: `false`
- ready_for_exact_eval_dispatch: `false`
- source_archive: `experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/archive.zip`
- FEC6 selector bytes: `249`
- target saving bytes: `78`
- global entropy floor bytes: `241`
- best charged FEC7 candidate: `fec7_global_pr103_range_u8_hist`
- best charged FEC7 payload bytes: `268`
- best saving vs FEC6 selector: `-19`
- can meet target: `false`

## Charged Candidates

| candidate | bytes | saving vs FEC6 | model bytes | range bytes | meets target |
|---|---:|---:|---:|---:|---|
| fec7_global_pr103_range_u8_hist | 268 | -19 | 16 | 244 | false |
| fec7_split_none_pr103_range | 274 | -25 | 19 | 243 | false |
| fec7_pairmod2_pr84_context_range | 281 | -32 | 32 | 240 | false |
| fec7_pairmod4_pr84_context_range | 313 | -64 | 64 | 240 | false |
| fec7_pairmod8_pr84_context_range | 369 | -120 | 128 | 232 | false |
| fec7_pairmod16_pr84_context_range | 489 | -240 | 256 | 224 | false |
| fec7_pairmod25_pr84_context_range | 621 | -372 | 400 | 212 | false |
| fec7_pairmod50_pr84_context_range | 997 | -748 | 800 | 188 | false |
| fec7_pairmod100_pr84_context_range | 1757 | -1508 | 1600 | 148 | false |

## Blocker

- blocked: `true`
- reason: FEC6 selector bytes are already near the global entropy floor; tested byte-closed FEC7 range/adaptive prototypes charge their model bytes and do not approach the required saving.
- reactivation_criteria: Reopen only with a selector model whose charged model plus range stream is at least target_saving_bytes smaller than FEC6, or with a compliance-reviewed runtime prior that is not source-embedded selector data.

