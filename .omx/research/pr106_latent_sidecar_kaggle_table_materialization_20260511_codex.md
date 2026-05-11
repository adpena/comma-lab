# PR106 latent sidecar materialization from Kaggle score table (2026-05-11)

Purpose: turn the harvested Kaggle PR106 latent score table into charged bytes
without promoting the result as an auth-eval score.

## Candidate

- source archive:
  `experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/archive.zip`
- source archive bytes: `186239`
- source archive SHA-256:
  `3fefbe5dfdd738179a55ca5c995ff8f63ec2755662d60684706f20d313913f58`
- source `0.bin` bytes: `186131`
- source `0.bin` SHA-256:
  `7f2cc905b7611ae8d7bced72be24e2266b0aa341f90cfeccbb0854fd8fc01eb7`
- candidate archive:
  `experiments/results/pr106_latent_sidecar_from_kaggle_table_20260511_codex/sidecar_archive.zip`
- candidate archive bytes: `186808`
- candidate archive SHA-256:
  `947b85e8a69db295d4dcf80b0b528639c47839f40f289a2c05b70a2064658b48`
- candidate `0.bin` bytes: `186700`
- candidate `0.bin` SHA-256:
  `e4f4022c3b69c3fe9a9d15471bb94c81d2ba116d38aebaac9708b42bea7a5056`
- sidecar bytes: `561`
- sidecar SHA-256:
  `3ee41acab8638fb436ee912faf8b94efe0ce4b266e7ea896018db1d3736b726e`

## Proof

- score_claim: `false`
- ready_for_exact_eval_dispatch: `false`
- source archive SHA match against Kaggle manifest: `false`
- source `0.bin` SHA match against Kaggle manifest: `true`
- inner PR106 payload unchanged inside wrapper: `true`
- nonzero latent deltas: `600`
- non-noop latent dims: `600`
- delta_q range: `[-1, +1]`
- candidate diff no-op status: `non_noop_payload_changed`
- candidate_non_noop: `true`
- archive byte delta vs PR106: `+569`
- rate component delta vs PR106: `+0.0003788737443265155`

The archive SHA mismatch is a ZIP-framing mismatch only. The scorer table was
measured against the same PR106 `0.bin` payload. `build_pr106_latent_sidecar.py`
now validates this explicitly: if archive SHA differs, the manifest must expose
`source_zero_bin_sha256`, and the source archive's `0.bin` must match it.

## Boundary

This candidate is now byte-closed enough for the next exact-eval step, but it
is not a score result. The next valid promotion path is:

1. use `submissions/pr106_latent_sidecar/inflate.sh` with this archive;
2. run exact contest-CUDA auth eval under a fresh dispatch claim;
3. preserve archive SHA, runtime tree SHA, command, hardware, logs, components,
   and JSON result;
4. only after exact CUDA adjudication compare against PR106 or any public
   frontier row.

No CPU, MPS, or Kaggle provider table value should be used to rank, kill, or
submit this lane.
