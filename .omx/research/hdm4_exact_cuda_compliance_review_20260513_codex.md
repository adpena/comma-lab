<!-- generated_at: 2026-05-13T09:02:00Z, from_state_hash: hdm4_exact_cuda_runtime_hash_restored -->
# HDM4 exact-CUDA compliance review

## Scope

This memo reviews the HDM4 byte-closed candidate against contest-compliance and
custody gates after the exact-CUDA artifact landed.

## Candidate custody

- Lane: `hnerv_hdm4_q_brotli_split_exact_eval`
- Job: `pr106_r2_lowlevel_hdm4_candidate_pr101_runtime_cuda_20260513_codex`
- Archive:
  `experiments/results/pr106_r2_lowlevel_hdm4_archive_candidate_20260513_codex/pr106_r2_lowlevel_hdm4_archive_candidate.zip`
- Archive bytes: `186492`
- Archive SHA-256:
  `218ae16f3f13b722e9752d698667ed8770151e40d44b5756c0ebbccb7682825f`
- Exact-CUDA score artifact:
  `experiments/results/modal_auth_eval/pr106_r2_lowlevel_hdm4_candidate_pr101_runtime_cuda_20260513_codex/contest_auth_eval.json`

## Review result

PacketIR parse/re-emit identity passed for the scored archive:
`packet_ir_identity_passed=true`, single member `0.bin`, payload bytes `186384`,
payload SHA-256
`7197c76fb92b7f1a3ce6f1695360412e02915dd2cc4030bf3397f69fa504f47d`.

The exact runtime file
`submissions/pr106_latent_sidecar_r2_pr101_grammar/inflate.py` was restored to
SHA-256
`60055bced3ab608d0e93ba83e18fa5bc662746cfa273ad50d5960c34028d1fb3`, matching
the Modal exact-CUDA provenance. This matters because even comment-only edits to
`inflate.py` change the runtime-tree custody identity.

`scripts/pre_submission_compliance_check.py --contest-final --strict` against
the source runtime currently fails only the report-custody checks:

- `report_mentions_archive_sha256`: missing `218ae16f3f13b722`
- `report_mentions_archive_size_bytes`: missing `186492`

The generated static release surface is not yet promotable because its wrapper
runtime-tree hash differs from the Modal exact-CUDA runtime tree. Treat that as a
release-surface packaging issue, not a model or PacketIR result.

## Classification

HDM4 is an exact-CUDA measured candidate, not a submission-ready packet. The
remaining work is release-surface custody: materialize a report/manifest/runtime
surface whose runtime tree matches the scored runtime and whose report mentions
the scored archive hash and byte count.

## Next actions

1. Keep scored runtime bytes stable unless intentionally re-running exact eval.
2. Build an HDM4 release-review surface that preserves the scored runtime tree.
3. Re-run strict pre-submission compliance against that surface.
4. Promote only after the compliance JSON is clean and linked from the custody
   ledger.
