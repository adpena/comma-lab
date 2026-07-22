---
title: Findings - Task 557 and historical adaptive-arithmetic anchor reconciliation
date_utc: 2026-07-22T04:47:13Z
source_task: 557
consumer_task: 603
feeds_task: 613
research_only: true
verdict: BOTH_ANCHORS_VALID_ONLY_WITH_SUBSTRATE_MODEL_AND_GROUPING_SCOPE
verdict_scope: Historical PR101/PR103 merged-stream arithmetic win and current Task 557 donor rows are different empirical objects; neither licenses a global arithmetic or Brotli rule
main_landing_review_required: true
---

# Reconciled evidence

The coder corpus preserves a historical PR101/PR103 result: merging the eight largest weight streams
and `latent-hi` into one constriction range stream saved about `290 B` relative to the PR100-style
baseline. That is a real arithmetic-on-latent-like anchor, but it includes a particular substrate,
stream grouping, model, and removal of per-stream rounding overhead.

The current committed #557 receipt measures a different donor and different concrete arithmetic
models:

| #557 donor section | Brotli | repository IID arithmetic | repository spatial arithmetic |
|---|---:|---:|---:|
| base weights, 72,695 elements | 63,394 B | 66,322 B | 93,991 B |
| pair code, 38,400 elements | 20,518 B | 35,989 B | 37,432 B |

Thus the current pair-code row favors Brotli by `15,471 B` against repository IID arithmetic and
`16,914 B` against repository spatial arithmetic. The #557 memo explicitly scopes this negative to
those concrete context/sign-magnitude models and does not falsify arithmetic coding as a family.

# Corrected reusable anchor

`CODEC_WINNER_IS_PAYLOAD_MODEL_GROUPING_AND_FRAMING_SPECIFIC; PR103_MERGED_AC_WIN_AND_TASK557_CURRENT_DONOR_BROTLI_WIN_COEXIST; MEASURE_THE_EXACT_FINAL_STREAM`.

The Task #603 survey therefore remeasures each live stream and uses neither a global Brotli rule nor a
global adaptive-arithmetic rule.

# Custody

- Historical lineage: `.omx/research/pr_extended_bit_level_lineage_pr95_pr100_pr101_pr103_20260507_claude.md`.
- `.omx/research/arith_selfcomp_rate_coders_20260719_receipt.json`: `2,057,039 B`, SHA-256
  `8cd25a6bde36676285326ac10d49d041e9f58deb7ffb5f03e518a2185435490d`.
- `.omx/research/arith_selfcomp_rate_coders_20260719_codex.md`: `10,030 B`, SHA-256
  `de62a734a39e152ffb4d1aecb6436ba984bd9450976b1bd5331b42c8a61ab326`.

0.1910828242 [contest-CPU] — unchanged by construction.
