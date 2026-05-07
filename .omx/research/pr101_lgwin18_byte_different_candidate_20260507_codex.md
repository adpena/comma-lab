# PR101 lgwin18 Byte-Different Candidate - 2026-05-07

Status: local CPU-prep candidate, not dispatch-ready.

This ledger records a PR101-shaped byte-different archive produced by changing
the Brotli window parameter for the decoder blob while preserving PR101's fixed
decoder slice length. No remote/GPU dispatch was performed.

## Candidate

- Candidate archive:
  `experiments/results/pr101_codecop_lgwin18_candidate_20260507_codex/substituted/pr101_lgwin18_pr101_lgwin18_auto_selectFalse_brotli_lgwin18_brotli_quality11/archive.zip`
- Candidate bytes: `178258`
- Candidate SHA-256:
  `c95c59933f95746f6b8dd5fb7b4450419a25c01b2c9f8dac6e586cd4b3582933`
- Source PR101 archive SHA-256:
  `b83bf3488625dbd73adeddff91712994197ab53098e578e91327a0c6e49efb3e`
- Decoder blob changed:
  `836d1876bffd74f77f30e387a3b4cac1dbb25929cc4d348830d36cfa2a6d48a6`
  to
  `ea7155370f0adaab5d0078ef9158de7abe8363ff7a09afd26cf699ed8b81600d`
- Latent blob preserved:
  `de8a0da594f073efc43849334573ba06438bb37d53f9343ee6367659c0106bbe`
- Sidecar blob preserved:
  `6c2946e323bbbc6f8d906ef6c68989e8acbd8d60332c87da8fe8147f1ea7b12f`

## Evidence

- Manifest:
  `experiments/results/pr101_codecop_lgwin18_candidate_20260507_codex/manifest.json`
- Substitution report:
  `experiments/results/pr101_codecop_lgwin18_candidate_20260507_codex/substituted/pr101_lgwin18_pr101_lgwin18_auto_selectFalse_brotli_lgwin18_brotli_quality11/substitution_report.json`
- Local decoder parity:
  `experiments/results/pr101_codecop_lgwin18_candidate_20260507_codex/local_decoder_parity_check.json`

Local parity result: the candidate decoder blob decodes and re-encodes through
the stock PR101 contract back to the source default decoder blob. This is a
decoder-byte transducer parity check, not a contest score.

## Dispatch State

`ready_for_exact_eval_dispatch=false`.

Remaining blockers:

- exact runtime parity not supplied;
- matching lane dispatch claim not supplied;
- contest CUDA auth eval missing.

The candidate is below the active PR103-on-PR106 byte floor by bytes, but it is
not an exact-score claim. It should enter exact-eval only after runtime parity,
lane claim, and strict packet compiler/compliance surfaces pass.
