# Contest Submission Packet Checklist

This packet includes a concrete submission directory. It does not copy raw frames.

## Archive Custody

- [x] `archive.zip` exists at `experiments/results/lightning_batch/exact_eval_public_pr100_hnerv_lc_v2_adapter_t4_20260504T1213Z/archive.zip`.
- [x] SHA-256: `afd53348f50303bf0ec6a7ffecc1ac037df2f1c70745244b9c45c72e8eb80641`.
- [x] Byte size: `178981`.

## Copied Submission

- [x] Submission directory: `experiments/results/submission_packet_pr100_adapter_20260504/apogee_pr100_hnerv_lc_v2_adapter`.
- [x] Runtime source directory: `experiments/results/public_runtime_adapters_20260504_codex/pr100_runtime_adapter`.
- [x] Runtime file count: `5`.
- [x] `archive.zip` and `report.txt` copied into the submission directory.

- [x] Runtime `hnerv_model.py` sha256 `e63b04ad3df4942b9bc1e31afd8ec84177dfbe83827f67cf7c5a682b05c1b46b` mode `0o644`.
- [x] Runtime `inflate.py` sha256 `2b94d042f090155a19606e98f00f4bf1986974a3c3ad7059ae7b1c4ee23605db` mode `0o644`.
- [x] Runtime `inflate.sh` sha256 `b8dd1979c49171fde4552956d0d07034646778f747ccbdc2029839a302f7a806` mode `0o755`.
- [x] Runtime `schema.py` sha256 `bc434bd596e753dbeae97c0ddce4d9cf98a50cfe862a451834864d83620d6a0a` mode `0o644`.
- [x] Runtime `sidecar.py` sha256 `52955000fe673914307b481bff29377a015bf7a0fe9dc38652e29153c033af75` mode `0o644`.

## Frontier Snapshot

- Score authority JSON: `contest_auth_eval.adjudicated.json`.
- Field-supported grade: `A++`.
- Recomputed score field: `0.22826947142244708`.
- Score claim: `false`.
- Ranking claim: `false`.
- Promotion claim: `false`.

## Auth Eval Fields

- [x] `contest_auth_eval.adjudicated.json` samples: `600`.
- [x] SegNet distance field present: `0.00067623`.
- [x] PoseNet distance field present: `0.00017198`.
- [x] Recomputed score field present: `0.22826947142244708`.
- [x] Device field: `cuda`.
- [x] GPU model field: `Tesla T4`.

## Optional Evidence

- [ ] `component_trace.json`
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

- [x] `archive_crc_ok`: bad_crc_member=None
- [x] `archive_no_duplicate_members`: duplicates=[]
- [x] `archive_packed_payload_singleton`: packed_payload_members=[]
- [x] `archive_member_safe:0.bin`: safe member name
- [x] `archive_local_central_name_match:0.bin`: local='0.bin' central='0.bin'
- [x] `archive_local_central_flag_bits_match:0.bin`: local=0 central=0
- [x] `archive_sha256_matches_contest_auth_eval`: archive.zip sha256=afd53348f50303bf0ec6a7ffecc1ac037df2f1c70745244b9c45c72e8eb80641
- [x] `archive_bytes_matches_contest_auth_eval`: archive.zip bytes=178981
- [x] `provenance_archive_bytes_matches_archive`: contest_auth_eval provenance archive_size_bytes matches archive.zip
- [x] `score_recomputes_from_components`: formula=0.22826957271120643 json=0.22826947142244708
- [x] `expected_archive_sha256`: expected=afd53348f50303bf0ec6a7ffecc1ac037df2f1c70745244b9c45c72e8eb80641
- [x] `expected_archive_size_bytes`: expected=178981
- [x] `expected_samples`: expected=600
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
