# Contest Submission Packet Checklist

This packet includes a concrete submission directory. It does not copy raw frames.

## Archive Custody

- [x] `archive.zip` exists at `experiments/results/lightning_batch/exact_eval_pr85_stbm1br_stbm_runtime_t4_g4dn2x_20260504T0613Z/archive.zip`.
- [x] SHA-256: `c6f004d444ed32c628611a2f21f567c666af6bcbcceba618cc089ec024a0cda6`.
- [x] Byte size: `229756`.

## Copied Submission

- [x] Submission directory: `experiments/results/lightning_batch/exact_eval_pr85_stbm1br_stbm_runtime_t4_g4dn2x_20260504T0613Z/final_submission_packet_replay_runtime_20260504_codex/submission`.
- [x] Runtime source directory: `experiments/results/pr85_stbm1br_mask_recode_20260504_worker/replay_submission_stbm`.
- [x] Runtime file count: `3`.
- [x] `archive.zip` and `report.txt` copied into the submission directory.

- [x] Runtime `inflate.py` sha256 `11dc16a956128eb855375d2fc1c846506111aa6499015b35f7ea51d93a4beeec` mode `0o644`.
- [x] Runtime `inflate.sh` sha256 `2c37f19e210f97c8926b70594e4d57fd8b0256dadace0cd55c28bbe6995ff027` mode `0o755`.
- [x] Runtime `range_mask_codec.cpp` sha256 `94cd1a86111fb6d34b6e12d37c624bd5938df0fbc6c4c24c8d40c5a83fcb016b` mode `0o644`.

## Frontier Snapshot

- Score authority JSON: `contest_auth_eval.json`.
- Field-supported grade: `A++`.
- Recomputed score field: `0.25369011029397787`.
- Score claim: `false`.
- Ranking claim: `false`.
- Promotion claim: `false`.

## Auth Eval Fields

- [x] `contest_auth_eval.json` samples: `600`.
- [x] SegNet distance field present: `0.00057185`.
- [x] PoseNet distance field present: `0.0001894`.
- [x] Recomputed score field present: `0.25369011029397787`.
- [x] Device field: `cuda`.
- [x] GPU model field: `Tesla T4`.

## Optional Evidence

- [x] `component_trace.json`
- [x] `report.txt`
- [x] `eval_provenance.json`
- [x] `auth_eval.log`
- [x] `contest_auth_eval.adjudicated.json`
- [x] `adjudication_provenance.json`

## Non-Score Supporting Artifacts

- [ ] `planner_ledgers`: none recorded.
- [ ] `visualizations`: none recorded.
- [ ] `next_action_tranches`: none recorded.

## Validation

- [x] `archive_sha256_matches_contest_auth_eval`: archive.zip sha256=c6f004d444ed32c628611a2f21f567c666af6bcbcceba618cc089ec024a0cda6
- [x] `archive_bytes_matches_contest_auth_eval`: archive.zip bytes=229756
- [x] `provenance_archive_bytes_matches_archive`: contest_auth_eval provenance archive_size_bytes matches archive.zip
- [x] `score_recomputes_from_components`: formula=0.25369019992751551 json=0.25369011029397787
- [x] `expected_archive_sha256`: expected=c6f004d444ed32c628611a2f21f567c666af6bcbcceba618cc089ec024a0cda6
- [x] `expected_archive_size_bytes`: expected=229756
- [x] `expected_samples`: expected=600
- [x] `component_trace_cross_check`: component trace cross-check did not report a mismatch
- [x] `component_trace_contest_auth_eval_sha256`: component trace cross-check points at an accepted contest_auth_eval JSON
- [x] `component_trace_n_samples`: trace=600 contest=600
- [x] `component_trace_archive_bytes`: trace=229756 contest=229756
- [x] `eval_provenance_archive_sha256`: eval_provenance archive_sha256 matches archive.zip
- [x] `eval_provenance_archive_bytes`: eval_provenance archive_size_bytes matches archive.zip
- [x] `eval_provenance_device`: eval_provenance device matches contest_auth_eval provenance
- [x] `adjudicated_archive_sha256`: adjudicated JSON archive_sha256 matches archive.zip
- [x] `adjudicated_archive_bytes`: adjudicated JSON archive_size_bytes matches archive.zip
- [x] `adjudicated_n_samples`: adjudicated JSON n_samples matches contest_auth_eval.json

## Evidence Classification

- Field-supported grade: `A++`.
- Basis: `cuda_t4_full_sample_adjudicated_fields`.
- Score claim: `false`.
- Ranking claim: `false`.
- Promotion claim: `false`.
