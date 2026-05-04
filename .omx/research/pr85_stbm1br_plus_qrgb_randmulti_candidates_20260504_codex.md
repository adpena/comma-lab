# PR85 QRGB Transfer Archive Candidates - 2026-05-04

- tool: `experiments/build_pr85_qrgb_transfer_archive_candidates.py`
- score_claim: false
- dispatch_performed: false
- remote_jobs_dispatched: false
- dispatch_unlocked: false
- blocker_class: `source_sha_mismatch`

## Source Anchor

- PR85 source archive bytes: `229756`
- PR85 source archive sha256: `c6f004d444ed32c628611a2f21f567c666af6bcbcceba618cc089ec024a0cda6`
- known PR85 anchor match: False

## Build Scope

- max candidates: `1`
- selected unique pairs: `1`
- archive candidates built: `0`
- fixed-runtime preflights: `0`
- ready after lane claim: `0`

## Candidate Artifacts

- `pr85_qrgb_f2_randglobal_pair_0192`
  - build_status: `blocked`
  - dispatch_unlocked: false
  - archive: `None`
  - bytes: `None`
  - sha256: `None`
  - changed_segments: `None`
  - fixed_runtime_ready: `None`

## Blockers

- `source_sha_mismatch`: selected source archive is not the known PR85 anchor
- `source_sha_mismatch`: planning row source archive SHA does not match selected source archive
- `source_sha_mismatch`: planning row source archive bytes do not match selected source archive
- `source_sha_mismatch`: planning row source archive SHA does not match selected source archive
- `source_sha_mismatch`: planning row source archive bytes do not match selected source archive
- `source_sha_mismatch`: planning row source archive SHA does not match selected source archive
- `source_sha_mismatch`: planning row source archive bytes do not match selected source archive
- `source_sha_mismatch`: planning row source archive SHA does not match selected source archive
- `source_sha_mismatch`: planning row source archive bytes do not match selected source archive
- `source_sha_mismatch`: selected source archive is not the known PR85 anchor

## Compliance Notes

- Local archive builds only; no training, scorer import, GPU dispatch, or score claim.
- Archives remain single-member `x` PR85 bundles and are byte-closed.
- Exact eval is allowed only after a lane claim and only through canonical CUDA auth eval.
