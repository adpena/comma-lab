# Contest Submission Packet Checklist

This packet is metadata-only. It does not copy raw frames or archive payloads.

## Archive Custody

- [x] `archive.zip` exists at `experiments/results/lightning_batch/exact_eval_qzs3_b32_maskfirst_qp1_fix1_t4_20260502T0331Z/archive.zip`.
- [x] SHA-256: `cf44aa7fdb13b9ab6236aefde1f5f58e915d7dfa99128246235443841396c6ab`.
- [x] Byte size: `276347`.

## Auth Eval Fields

- [x] `contest_auth_eval.json` samples: `600`.
- [x] SegNet distance field present: `0.00061244`.
- [x] PoseNet distance field present: `0.00049637`.
- [x] Recomputed score field present: `0.3157055307844823`.
- [x] Device field: `cuda`.
- [x] GPU model field: `Tesla T4`.

## Optional Evidence

- [x] `component_trace.json`
- [x] `report.txt`
- [x] `eval_provenance.json`
- [x] `contest_auth_eval.adjudicated.json`
- [x] `adjudication_provenance.json`

## Validation

- [x] `archive_sha256_matches_contest_auth_eval`: archive.zip sha256=cf44aa7fdb13b9ab6236aefde1f5f58e915d7dfa99128246235443841396c6ab
- [x] `archive_bytes_matches_contest_auth_eval`: archive.zip bytes=276347
- [x] `provenance_archive_bytes_matches_archive`: contest_auth_eval provenance archive_size_bytes matches archive.zip
- [x] `score_recomputes_from_components`: formula=0.31570565490293501 json=0.31570553078448232
- [x] `expected_archive_sha256`: expected=cf44aa7fdb13b9ab6236aefde1f5f58e915d7dfa99128246235443841396c6ab
- [x] `expected_archive_size_bytes`: expected=276347
- [x] `expected_samples`: expected=600
- [x] `component_trace_cross_check`: component trace cross-check did not report a mismatch
- [x] `component_trace_contest_auth_eval_sha256`: component trace cross-check points at this contest_auth_eval.json
- [x] `component_trace_n_samples`: trace=600 contest=600
- [x] `component_trace_archive_bytes`: trace=276347 contest=276347
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
