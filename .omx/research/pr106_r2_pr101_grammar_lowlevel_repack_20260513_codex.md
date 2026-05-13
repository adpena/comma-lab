# PR106/R2 PR101-Grammar Low-Level Repack Candidate

Date: 2026-05-13
Author: codex
Scope: byte-closed PR106/R2 PR101-grammar PacketIR/HNeRV low-level brotli repack

## Summary

The HNeRV low-level brotli repacker now handles PR106 sidecar wrapper packets
(`0xfe`) by parsing the wrapper with the canonical PR106 PacketIR grammar,
repacking only the inner `0xff` HNeRV payload, and re-emitting the wrapper with
the original sidecar payload and framing metadata preserved.

This produced a new local candidate archive:

- source archive: `submissions/pr106_latent_sidecar_r2_pr101_grammar/archive.zip`
- source SHA-256: `c48631e11a9bb18d051da9100ca4d5773558a8a81ac38dc8f6f4e8b6119d0383`
- source bytes: `186780`
- candidate archive: `experiments/results/pr106_r2_pr101_grammar_lowlevel_repack_20260513_codex/pr106_r2_pr101_grammar_hnerv_brotli_repack_candidate.zip`
- candidate SHA-256: `287e6edc612803a9a9d5de3ce50b421c039704f38bae442a6dcc97a3e8d6ed4d`
- candidate bytes: `186629`
- byte delta: `-151`
- rate-only formula delta if components are unchanged: about `-0.000100545`

No contest score is claimed here. Exact [contest-CUDA] and [contest-CPU]
auth-eval artifacts are still required before score language, ranking, or
promotion.

## Proofs Produced

Artifact directory:
`experiments/results/pr106_r2_pr101_grammar_lowlevel_repack_20260513_codex/`

- `result.json`: candidate build manifest, `score_claim=false`,
  `ready_for_exact_eval_dispatch=false`, `ready_for_archive_preflight=true`.
- `packetir_identity.json`: PR106 PacketIR parse/emit identity passes for the
  candidate archive; every wrapper payload byte is parser-accounted.
- `runtime_consumption.json`: selected PR106/R2 PR101 runtime consumes and
  applies sidecar bytes; mutation changes runtime semantic/corrected-latent
  digests; still no full-frame or score claim.
- `same_runtime_prefix_parity_local_cpu.json`: source and candidate match for a
  1-pair local CPU prefix under the same runtime.
- `same_runtime_full_frame_parity_local_cpu.json`: source and candidate match
  for all 600 pairs / 1200 frames under the same PR106/R2 PR101 runtime on
  local CPU streaming hash:
  `b272a1a4841f8fcc9fe843e0544ea0bb46b8359fe5f8cc9d81acf8bd3b7baf99`.
- `pre_submission_compliance.static.json`: intentionally fails against stale
  source auth-eval/archive-manifest custody from the submission directory. This
  is expected until the candidate has its own exact auth-eval and manifest.

## Exact-Eval Boundary

Existing exact source result:

- PR106/R2 PR101 grammar [contest-CUDA T4]:
  `0.2066181354574151`, archive bytes `186780`, SHA-256
  `c48631e11a9bb18d051da9100ca4d5773558a8a81ac38dc8f6f4e8b6119d0383`.
- PR106/R2 PR101 grammar [contest-CPU GHA Linux x86_64]:
  `0.22806551797550428`, archive bytes `186780`, same SHA-256.

Because the candidate has full-frame same-runtime local parity against the
source runtime, the next valid action is exact auth eval on the candidate bytes,
not further local score inference.

Required before any score claim:

1. Claim lane dispatch with lane id `pr106_r2_pr101_grammar_lowlevel_repack_151b`.
2. Run [contest-CUDA T4] auth eval against the candidate archive and
   `submissions/pr106_latent_sidecar_r2_pr101_grammar/inflate.sh` with the
   submission dir uploaded as the runtime tree.
3. Run [contest-CPU Linux x86_64] closure if CUDA succeeds or if CPU-axis
   submission readiness is being assessed.
4. Re-run strict pre-submission compliance against the candidate-specific
   auth-eval JSON and archive manifest.

## Classification

- Legitimate local byte movement: yes, candidate archive is byte-different and
  byte-smaller.
- Runtime sidecar consumption: yes, proven by mutation digest.
- Full-frame same-runtime local parity: yes.
- Exact contest score: not yet.
- Promotion/submission readiness: false until exact eval and compliance close.

