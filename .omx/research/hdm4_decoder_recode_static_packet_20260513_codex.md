# HDM4 Decoder Recode Static Packet - 2026-05-13

## Summary

HDM4 is a byte-closed, lossless decoder-section recode for the PR106-R2
PR101-grammar lowlevel archive. It is the immediate successor to the HDM3
exact-CUDA frontier: same source payload semantics, same latents/sidecar bytes,
same PR106-R2 PR101-grammar runtime path, smaller decoder section.

This memo now records the exact CUDA closure. HDM4 is an internal
score-lowering frontier row with byte-closed `[contest-CUDA]` custody, but it
is still not public/submission promotion authority until the normal
adjudication and CPU-axis policy gates close.

## Packet

- source archive:
  `experiments/results/pr106_r2_pr101_grammar_lowlevel_repack_20260513_codex/pr106_r2_pr101_grammar_hnerv_brotli_repack_candidate.zip`
- source SHA-256: `287e6edc612803a9a9d5de3ce50b421c039704f38bae442a6dcc97a3e8d6ed4d`
- source bytes: `186629`
- candidate archive:
  `experiments/results/pr106_r2_lowlevel_hdm4_archive_candidate_20260513_codex/pr106_r2_lowlevel_hdm4_archive_candidate.zip`
- candidate SHA-256: `218ae16f3f13b722e9752d698667ed8770151e40d44b5756c0ebbccb7682825f`
- candidate bytes: `186492`
- byte delta vs source: `-137`
- byte delta vs HDM3 exact-CUDA frontier: `-123`
- rate-only score delta vs source if components remain equal:
  `-0.000091222677`
- expected rate-only score delta vs HDM3 exact-CUDA frontier if components
  remain equal: `-0.0000819013`

## Deterministic Recipe

- decoder recode key: `hdm4`
- recipe id: `1`
- recipe name: `conv3x3_then_1x1_then_bias_tail_dp4`
- split points: `[6, 9, 26, 28]`
- q Brotli bytes: `169861`
- q chunk bytes: `[130887, 2770, 4398, 31806]`
- raw scale bytes: `112`

The recipe is fixed and deterministic. It was selected by exhaustive dynamic
programming over declared schema-order families and contiguous Brotli
partitions, with the objective of minimizing decoder-section bytes including
runtime header overhead.

## Static Proofs

- manifest:
  `experiments/results/pr106_r2_lowlevel_hdm4_archive_candidate_20260513_codex/manifest.with_tool_run.json`
- runtime adapter proof:
  `experiments/results/pr106_r2_lowlevel_hdm4_archive_candidate_20260513_codex/runtime_adapter_proof.with_tool_run.json`
- strict static compliance:
  `experiments/results/pr106_r2_lowlevel_hdm4_archive_candidate_20260513_codex/pre_submission_compliance.static.json`
- readiness:
  `experiments/results/pr106_r2_lowlevel_hdm4_archive_candidate_20260513_codex/hdm3_exact_eval_packet_readiness.json`

Original static readiness state before dispatch:

- adapter payload identity: passed
- restored decoder section matches source: passed
- restored payload matches source: passed
- latents and sidecar preserved: passed
- strict static compliance: passed
- static packet ready: true
- remaining blockers: `lane_dispatch_claim_missing`, `exact_cuda_auth_eval_missing`

## Exact CUDA Closure

Lane `hnerv_hdm4_q_brotli_split_exact_eval` was claimed and dispatched through
the canonical detached Modal auth-eval path. Recovery landed a terminal
dispatch-claim row; `tools/claim_lane_dispatch.py summary` reports zero active
or stale nonterminal claims.

- exact CUDA result:
  `experiments/results/modal_auth_eval/pr106_r2_lowlevel_hdm4_candidate_pr101_runtime_cuda_20260513_codex/contest_auth_eval.json`
- recovered call id: `fc-01KRG4MNWZTHNMP56ZWKS5S0A1`
- evidence grade: `[contest-CUDA]`
- hardware: `Tesla T4`
- score axis: `contest_cuda`
- n samples: `600`
- canonical score: `0.20642625334307507`
- display-rounded score: `0.21`
- avg SegNet distance: `0.0006426`
- avg PoseNet distance: `0.00003236`
- archive bytes: `186492`
- archive SHA-256:
  `218ae16f3f13b722e9752d698667ed8770151e40d44b5756c0ebbccb7682825f`
- raw aggregate SHA-256:
  `5f65c70f59c78e5a4394dc062fe750cf721619f6d67790c4844d52f14d248993`

Against the HDM3 exact CUDA frontier
(`0.2065081539943091`, `186615` bytes), HDM4 lowers the exact CUDA score by
`-0.000081900651234`. SegNet and PoseNet components match exactly, and the
inflated raw aggregate SHA matches exactly. The observed score delta matches
the 123-byte rate-term delta from the contest formula.

Updated routing artifacts:

- scorecard:
  `experiments/results/hnerv_frontier_scorecard_refresh_20260513_codex/scorecard.json`
- scorecard markdown:
  `experiments/results/hnerv_frontier_scorecard_refresh_20260513_codex/scorecard.md`
- entropy-gap ranking:
  `experiments/results/hnerv_frontier_entropy_gap_ranking_20260513_codex/frontier_entropy_gap_ranking.json`
- entropy-gap ranking markdown:
  `experiments/results/hnerv_frontier_entropy_gap_ranking_20260513_codex/frontier_entropy_gap_ranking.md`

Scorecard audit result:

```text
hnerv frontier scorecard: PASS (11 rows, 2 payload groups, 33 follow-up targets, internal score-lowering=PR106-R2-lowlevel-HDM4 (0.20642625334307507))
```

## Next Exact Step

Do not redispatch HDM4 for the same CUDA axis. The next rate-only action is
promotion/adjudication review of the existing exact CUDA custody, plus a
separate `[contest-CPU]` closure if the operator wants CPU-axis leaderboard
alignment. New score-lowering work should target a byte-different transform of
the HDM4 `inner_decoder_packed_brotli` section or the unchanged
`inner_latents_and_sidecar_brotli` section, with old/new section SHA-256,
charged-byte proof, lane claim, and exact CUDA eval before any new score claim.
