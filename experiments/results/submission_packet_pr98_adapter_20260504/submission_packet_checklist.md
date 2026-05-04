# Contest Submission Packet Checklist

This packet includes a concrete submission directory. It does not copy raw frames.

## Archive Custody

- [x] `archive.zip` exists at `experiments/results/lightning_batch/exact_eval_public_pr98_hnerv_adapter_t4_20260504T0958Z/archive.zip`.
- [x] SHA-256: `7ecb0df1c4627d55d88e03eff3d890b7a7a5b047c62515acff20232cf29310eb`.
- [x] Byte size: `178392`.

## Copied Submission

- [x] Submission directory: `experiments/results/submission_packet_pr98_adapter_20260504/apogee_pr98_hnerv_adapter`.
- [x] Runtime source directory: `experiments/results/public_runtime_adapters_20260504_codex/pr98_runtime_adapter`.
- [x] Runtime file count: `19`.
- [x] `archive.zip` and `report.txt` copied into the submission directory.

- [x] Runtime `inflate.py` sha256 `fdfbfec9bebae59424ad3fd56191074369d3958e77ccd8e645559d28d3d252cd` mode `0o644`.
- [x] Runtime `inflate.sh` sha256 `a24c0d11d18eca1e9680f2e46c800222115bf02f8c70e1c78c18aa2f0a8953f7` mode `0o755`.
- [x] Runtime `src/codec.py` sha256 `d9f2825a416e66c4c60b0ed506e87afcc080342809f2af73c3c6efb2b3c16f4a` mode `0o644`.
- [x] Runtime `src/data.py` sha256 `db0016ba8e854daa0d01563108ab1c76b6046578fe5f54af479abd34c600ab42` mode `0o644`.
- [x] Runtime `src/losses.py` sha256 `757725dc64681d90ebd83057c956ca3d72e02e06f778571b599d99fac7b17368` mode `0o644`.
- [x] Runtime `src/model.py` sha256 `e63b04ad3df4942b9bc1e31afd8ec84177dfbe83827f67cf7c5a682b05c1b46b` mode `0o644`.
- [x] Runtime `src/optim.py` sha256 `6c79f7559e4fdf23ec1ab60fcdad29f785f8e223dc6cf426d9cf0bc38555d3c6` mode `0o644`.
- [x] Runtime `src/score.py` sha256 `c8c4421fd26faa5c97ebade2919f9ba02c916bca3806fa95ba5159f221c4ef7f` mode `0o644`.
- [x] Runtime `src/stages/codec_stage.py` sha256 `1ddd7ccac9427a4679f0bf884409cd22173219412f1962f92ed711cef2d54c55` mode `0o644`.
- [x] Runtime `src/stages/common.py` sha256 `ad766d0639269eba70af5e5c26ef71267dfe8350e0335cdc506b198a08febdef` mode `0o644`.
- [x] Runtime `src/stages/stage1_v328_ce.py` sha256 `a43a44fa4cdc17d83e81f94006228268fed78f2b641c73b9562124bfc995c2b7` mode `0o644`.
- [x] Runtime `src/stages/stage2_v331_softplus.py` sha256 `1203339dc5319f81d2f571a11be166a034bcb9b1a9bb346f823d68a9331aec20` mode `0o644`.
- [x] Runtime `src/stages/stage3_v332_smooth.py` sha256 `119fe4e1718502973c38fcd94163ea935e2ce7776ada833c4d41661225a73695` mode `0o644`.
- [x] Runtime `src/stages/stage4_v332_qat.py` sha256 `aaf7f704f0fcf3a7860bca1be589cfd6b90a217b529d77ffc7abc826d42114af` mode `0o644`.
- [x] Runtime `src/stages/stage5_c1a_l7.py` sha256 `6e1c4f8a1abdde16d6086f5c38dc523207ca1ca04b89cc72a0a2f58fe0ea0dbf` mode `0o644`.
- [x] Runtime `src/stages/stage6_lambda_sweep.py` sha256 `fe3893d388d7820d365d0b088a0800df800c6963e2e101e6e451b1172e6d453b` mode `0o644`.
- [x] Runtime `src/stages/stage7_sigma_sweep.py` sha256 `8b8b812bb4bc49e560e6fe2e1dd6b3fecb36c60bcdaeb1a5411e5697653e4443` mode `0o644`.
- [x] Runtime `src/stages/stage8_muon_finetune.py` sha256 `9e036c99b7fc58fa24f954be251bebb5af4cddd3b84fe408b9ebc2b94b58ed5f` mode `0o644`.
- [x] Runtime `src/train.py` sha256 `d88610929db7f42b7a3522a77ded29b2a9a4e7abe63eae7bc5915fa4b8e58b79` mode `0o644`.

## Frontier Snapshot

- Score authority JSON: `contest_auth_eval.adjudicated.json`.
- Field-supported grade: `A++`.
- Recomputed score field: `0.22933111465960354`.
- Score claim: `false`.
- Ranking claim: `false`.
- Promotion claim: `false`.

## Auth Eval Fields

- [x] `contest_auth_eval.adjudicated.json` samples: `600`.
- [x] SegNet distance field present: `0.00068841`.
- [x] PoseNet distance field present: `0.00017394`.
- [x] Recomputed score field present: `0.22933111465960354`.
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

- [x] `planner_ledgers` `.omx/research/public_hnerv_adapter_replays_20260504_codex.md`: `planning_or_proxy_only`, score claim `false`.
- [ ] `visualizations`: none recorded.
- [x] `next_action_tranches` `experiments/results/pr98_channel_ablation_candidates_20260504_codex/candidates_manifest.json`: `roadmap_only`, score claim `false`.

## Validation

- [x] `archive_crc_ok`: bad_crc_member=None
- [x] `archive_no_duplicate_members`: duplicates=[]
- [x] `archive_packed_payload_singleton`: packed_payload_members=[]
- [x] `archive_member_safe:0.bin`: safe member name
- [x] `archive_local_central_name_match:0.bin`: local='0.bin' central='0.bin'
- [x] `archive_local_central_flag_bits_match:0.bin`: local=0 central=0
- [x] `archive_sha256_matches_contest_auth_eval`: archive.zip sha256=7ecb0df1c4627d55d88e03eff3d890b7a7a5b047c62515acff20232cf29310eb
- [x] `archive_bytes_matches_contest_auth_eval`: archive.zip bytes=178392
- [x] `provenance_archive_bytes_matches_archive`: contest_auth_eval provenance archive_size_bytes matches archive.zip
- [x] `score_recomputes_from_components`: formula=0.22933102502497396 json=0.22933111465960354
- [x] `expected_archive_sha256`: expected=7ecb0df1c4627d55d88e03eff3d890b7a7a5b047c62515acff20232cf29310eb
- [x] `expected_archive_size_bytes`: expected=178392
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
