# Final Submission Packaging Hardening - 2026-05-04

## Scope

Grand Council final-packaging lane. Work confined to the contest submission
packet builder, pre-submission compliance gate, focused tests, and this ledger.
No remote jobs were dispatched and no score was claimed from a new eval.

## Hardening

- `scripts/build_contest_submission_packet.py` now accepts `--runtime-dir` and
  builds a concrete `submission/` packet from the exact auth-eval runtime file
  manifest plus `archive.zip` and `report.txt`.
- Runtime files are selected from
  `contest_auth_eval.provenance.inflate_runtime_manifest.files`, validated by
  SHA-256 and byte count before copy, and copied with metadata-preserving
  `shutil.copy2()` so executable bits such as `inflate.sh` survive packaging.
- `scripts/pre_submission_compliance_check.py` now has
  `--require-submission-runtime-match`, implied by `--contest-final`, and checks
  each auth-eval runtime manifest file against the actual packet
  `--submission-dir`.
- The gate now accepts either provenance-nested or top-level auth-eval runtime
  file manifests, verifies duplicate paths are absent, validates SHA/byte field
  shape, and fails closed on missing, unsafe, or mismatched packet runtime files.

## Focused Tests

Command:

```bash
.venv/bin/python -m pytest src/tac/tests/test_build_contest_submission_packet.py src/tac/tests/test_pre_submission_compliance_check.py
```

Result: `31 passed`.

## STBM Replay Runtime Packet Check

Source exact-eval artifact:

```text
experiments/results/lightning_batch/exact_eval_pr85_stbm1br_stbm_runtime_t4_g4dn2x_20260504T0613Z
```

Runtime selected:

```text
experiments/results/pr85_stbm1br_mask_recode_20260504_worker/replay_submission_stbm
```

Built packet:

```text
experiments/results/lightning_batch/exact_eval_pr85_stbm1br_stbm_runtime_t4_g4dn2x_20260504T0613Z/final_submission_packet_replay_runtime_20260504_codex
```

Bound custody:

```text
archive_sha256=c6f004d444ed32c628611a2f21f567c666af6bcbcceba618cc089ec024a0cda6
archive_size_bytes=229756
runtime_tree_sha256=d195f4ecd0743cfd146efafee6729e96ee5428bfb28bbd0ca87cbad055494440
runtime_files=inflate.py, inflate.sh, range_mask_codec.cpp
```

Strengthened gate:

```bash
.venv/bin/python scripts/pre_submission_compliance_check.py \
  --submission-dir experiments/results/lightning_batch/exact_eval_pr85_stbm1br_stbm_runtime_t4_g4dn2x_20260504T0613Z/final_submission_packet_replay_runtime_20260504_codex/submission \
  --auth-eval-json experiments/results/lightning_batch/exact_eval_pr85_stbm1br_stbm_runtime_t4_g4dn2x_20260504T0613Z/contest_auth_eval.json \
  --contest-final \
  --expected-archive-sha256 c6f004d444ed32c628611a2f21f567c666af6bcbcceba618cc089ec024a0cda6 \
  --expected-archive-size-bytes 229756 \
  --expected-runtime-tree-sha256 d195f4ecd0743cfd146efafee6729e96ee5428bfb28bbd0ca87cbad055494440
```

Result: `status=passed`, `failed_checks=[]`. Runtime file match checks passed
for `inflate.py`, `inflate.sh`, and `range_mask_codec.cpp`; `inflate.sh`
remained executable at mode `0o755`.
