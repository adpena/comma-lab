# Track 1 A1 PR101 Archive State Loader - Worker A - 2026-05-08

## Scope

Worker A owned the A1 PR101 archive-to-state_dict loader gap for
`experiments/train_score_gradient_pr101_finetune.py`. No GPU training or exact
eval dispatch was attempted.

## Inspected Surfaces

- `experiments/train_score_gradient_pr101_finetune.py`
  - Existing non-smoke path raised `NotImplementedError` before any PR101
    archive load.
  - Argparse surface already exposed `--pr101-archive`; no new training flags
    were invented.
- `src/tac/pr101_split_brotli_codec.py`
  - Reusable PR101 split-Brotli decoder already exposed
    `decode_decoder_compact`.
  - Canonical fixed schema is `FIXED_STATE_SCHEMA` with 28 tensors.
- `tools/codec_op_param_sweep_manifest.py`
  - Had a private `_load_state_dict_from_pr101_archive` helper, but it was
    buried in a sweep tool and imported another tool module.
- `tools/pr101_archive_substitution_surgery.py`
  - Confirmed PR101 inner layout constants:
    `decoder_blob=162164`, `latent_blob=15387`, member `x`.

## Archive Custody Check

Source archive:
`experiments/results/public_pr101_hnerv_ft_microcodec_intake_20260504_codex/archive.zip`

- archive bytes: 178258
- archive sha256: `b83bf3488625dbd73adeddff91712994197ab53098e578e91327a0c6e49efb3e`
- member: `x`, ZIP_STORED, inner bytes 178158
- inner sha256: `5f1948f9572e65f71c614d2ff15764ee416522e25cb1b06c8b1299c1306e8aaf`
- decoder slice: 162164 bytes,
  `836d1876bffd74f77f30e387a3b4cac1dbb25929cc4d348830d36cfa2a6d48a6`
- latent slice: 15387 bytes,
  `de8a0da594f073efc43849334573ba06438bb37d53f9343ee6367659c0106bbe`
- sidecar slice: 607 bytes,
  `6c2946e323bbbc6f8d906ef6c68989e8acbd8d60332c87da8fe8147f1ea7b12f`

The decoded state_dict has 28 tensors and strict-loads into the A1
`HNeRVDecoder(latent_dim=28, base_channels=36, eval_size=(384, 512))` with no
missing or unexpected keys.

## Implementation

- Added `src/tac/pr101_archive_state_loader.py`.
  - Strict single-member ZIP validation.
  - Requires member `x` and ZIP_STORED by default.
  - Splits decoder, latent, and sidecar byte ranges.
  - Decodes the decoder slice through `decode_decoder_compact`.
  - Validates exact state_dict keys, tensor shapes, and `torch.float32` dtype.
  - Returns a typed `Pr101ArchiveState` with state_dict, byte slices, and
    custody metadata.
- Added `experiments/load_pr101_archive_to_state_dict.py`.
  - Thin CLI for materializing the decoded state_dict and metadata.
- Wired `experiments/train_score_gradient_pr101_finetune.py`.
  - Non-smoke now fail-closes through the canonical loader and then
    strict-loads into `HNeRVDecoder`.
  - The old `pr101_archive_loader_not_yet_implemented` blocker is removed.
  - Build manifest now records PR101 substrate metadata.

## Tests

Commands run:

```bash
.venv/bin/python -m pytest tests/test_pr101_archive_state_loader.py
.venv/bin/python -m pytest src/tac/tests/test_pr101_split_brotli_codec.py tests/test_pr101_archive_state_loader.py
.venv/bin/python -m py_compile src/tac/pr101_archive_state_loader.py experiments/load_pr101_archive_to_state_dict.py experiments/train_score_gradient_pr101_finetune.py tests/test_pr101_archive_state_loader.py
.venv/bin/python experiments/load_pr101_archive_to_state_dict.py --archive experiments/results/public_pr101_hnerv_ft_microcodec_intake_20260504_codex/archive.zip --metadata-output /tmp/pr101_loader_metadata_check.json
.venv/bin/python experiments/train_score_gradient_pr101_finetune.py --smoke --device cpu --smoke-epochs 1 --smoke-steps-per-epoch 1 --batch-size 1 --output /tmp/a1_pr101_loader_smoke_check
```

Results:

- Loader test file: 6 passed.
- Loader plus existing split-Brotli codec tests: 26 passed.
- Py compile: passed.
- CLI metadata extraction: passed and reported the custody values above.
- A1 smoke command: exited 0; synthetic one-step smoke warned that neither
  proxy term decreased, which is not a loader blocker.

## Remaining Blockers

- No expensive GPU job was run in this turn.
- Full A1 training still needs normal lane-claim discipline before remote CUDA
  dispatch.
- A trained checkpoint is not itself an exact-eval candidate. After training,
  the lane still needs archive rebuild, packet custody, exact CUDA auth eval,
  and adversarial review before any score claim.
