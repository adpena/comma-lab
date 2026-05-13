# HDM4 Decoder Recode Static Packet - 2026-05-13

## Summary

HDM4 is a byte-closed, lossless decoder-section recode for the PR106-R2
PR101-grammar lowlevel archive. It is the immediate successor to the HDM3
exact-CUDA frontier: same source payload semantics, same latents/sidecar bytes,
same PR106-R2 PR101-grammar runtime path, smaller decoder section.

This memo makes no score claim. HDM4 is static-packet-ready only; exact CUDA
auth eval remains required before promotion.

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

Static readiness state:

- adapter payload identity: passed
- restored decoder section matches source: passed
- restored payload matches source: passed
- latents and sidecar preserved: passed
- strict static compliance: passed
- static packet ready: true
- remaining blockers: `lane_dispatch_claim_missing`, `exact_cuda_auth_eval_missing`

## Next Exact Step

Claim lane `hnerv_hdm4_q_brotli_split_exact_eval` before any GPU job, then run
exact CUDA auth eval against the candidate archive and
`submissions/pr106_latent_sidecar_r2_pr101_grammar/inflate.sh`.

If exact CUDA components match the HDM3 frontier, HDM4 should lower the exact
CUDA score from `0.2065081539943091` by about `0.0000819`, solely through the
rate term. If components differ, classify as runtime/harness mismatch until
full custody review is complete.
