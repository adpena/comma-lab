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

The candidate now has paired exact [contest-CUDA T4] and [contest-CPU Linux
x86_64] auth-eval custody.

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
- `archive_manifest.candidate.json`: candidate-specific archive manifest for
  the 186,629-byte packet.
- `pre_submission_compliance.cuda_exact_nonfinal.json`: strict nonfinal CUDA
  compliance passed against the recovered exact auth-eval JSON, candidate
  archive manifest, runtime tree match, and terminal Modal dispatch claim.
- `pre_submission_compliance.cpu_exact_nonfinal.json`: strict nonfinal CPU
  compliance passed against the recovered exact auth-eval JSON, candidate
  archive manifest, runtime tree match, and terminal Modal dispatch claim.

Exact CUDA artifact directory:
`experiments/results/modal_auth_eval/pr106_r2_pr101_grammar_lowlevel_repack_cuda_20260513_codex/`

- Modal call id: `fc-01KRFTS57HQQPR7SKNA5JACZMZ`
- lane id: `pr106_r2_pr101_grammar_lowlevel_repack_151b_cuda`
- job id: `modal_pr106_r2_pr101_lowlevel_151b_cuda_20260513T0451Z`
- `contest_auth_eval.json`: exact [contest-CUDA T4] score
  `0.2065174760196528`, final rounded score `0.21`, 600 samples.
- components: `avg_segnet_dist=0.0006426`, `avg_posenet_dist=0.00003236`,
  archive bytes `186629`.
- runtime: uploaded `submissions/pr106_latent_sidecar_r2_pr101_grammar`
  runtime, T4 `Tesla T4`, CUDA `12.4`, driver `580.95.05`.
- inflated output aggregate SHA-256:
  `5f65c70f59c78e5a4394dc062fe750cf721619f6d67790c4844d52f14d248993`.
- terminal claim status:
  `completed_contest_cuda_modal_auth_eval_recovered`.

Exact CPU artifact directory:
`experiments/results/modal_auth_eval_cpu/pr106_r2_pr101_grammar_lowlevel_repack_cpu_20260513_codex/`

- Modal call id: `fc-01KRFTYAENF8Y6CT0TC9QDKS43`
- lane id: `pr106_r2_pr101_grammar_lowlevel_repack_151b_cpu`
- job id: `modal_pr106_r2_pr101_lowlevel_151b_cpu_20260513T0454Z`
- `contest_auth_eval.json`: exact [contest-CPU Linux x86_64] score
  `0.22796397327358284`, final rounded score `0.23`, 600 samples.
- components: `avg_segnet_dist=0.00063196`, `avg_posenet_dist=0.00016402`,
  archive bytes `186629`.
- runtime: uploaded `submissions/pr106_latent_sidecar_r2_pr101_grammar`
  runtime, Linux `x86_64`, CUDA unavailable.
- terminal claim status:
  `completed_contest_cpu_modal_auth_eval_recovered`.

## Exact-Eval Boundary

Existing exact source result:

- PR106/R2 PR101 grammar [contest-CUDA T4]:
  `0.2066181354574151`, archive bytes `186780`, SHA-256
  `c48631e11a9bb18d051da9100ca4d5773558a8a81ac38dc8f6f4e8b6119d0383`.
- PR106/R2 PR101 grammar [contest-CPU GHA Linux x86_64]:
  `0.22806551797550428`, archive bytes `186780`, same SHA-256.

Because the candidate has full-frame same-runtime local parity against the
source runtime, the valid score authority is exact auth eval on the candidate
bytes. Both CUDA and CPU are now measured.

Before this candidate can be called release/submission ready:

1. Build a candidate-specific release surface if submission review is desired:
   candidate `archive.zip`, candidate `archive_manifest.json`, candidate
   `contest_auth_eval.json`, and report text that names the candidate SHA/bytes.
2. Re-run strict `--contest-final` compliance against that release surface.

## Score Movement

CUDA apples-to-apples comparison against the matching source archive/runtime:

- source [contest-CUDA T4]: `0.2066181354574151`, bytes `186780`.
- candidate [contest-CUDA T4]: `0.2065174760196528`, bytes `186629`.
- measured delta: `-0.00010065943776230331`.

This is a rate-only win consistent with the 151-byte archive reduction. It is
not a new representation win and does not change SegNet/PoseNet behavior by
itself.

CPU apples-to-apples comparison against the matching source archive/runtime:

- source [contest-CPU Linux x86_64]: `0.22806463271134514`, bytes `186780`.
- candidate [contest-CPU Linux x86_64]: `0.22796397327358284`, bytes `186629`.
- measured delta: `-0.00010065943776230331`.

The equal CUDA and CPU deltas are expected because the decoded frames and
scorer components stayed fixed per axis; only the archive byte term changed.

## Classification

- Legitimate local byte movement: yes, candidate archive is byte-different and
  byte-smaller.
- Runtime sidecar consumption: yes, proven by mutation digest.
- Full-frame same-runtime local parity: yes.
- Exact contest score: yes on [contest-CUDA T4] and [contest-CPU Linux
  x86_64].
- Promotion/submission readiness: false until contest-final release-surface
  compliance closes.
