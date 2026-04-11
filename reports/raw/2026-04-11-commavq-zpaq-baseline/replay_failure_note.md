## Exact Replay Failure

- replay command: `evaluate_local_submission_contract(...)`
- archive: `reports/raw/2026-04-11-commavq-zpaq-baseline/zpaq_baseline_submission.zip`
- observed failure:
  - `decompress.py` failed during `zpaq extract`
  - one payload attempted to extract to a nested absolute-temp path under the extraction root and exited non-zero

## Root Cause

- `src/tac/lossless/codecs.py::_compress_with_zpaq()` staged payload bytes in a temp directory
- the archive command passed the staged source by absolute temp path
- `zpaq` therefore stored temp-rooted file paths in the archive
- later extraction under a fresh temp root tried to recreate those archived paths and failed

## Fix Landed

- `_compress_with_zpaq()` now archives the staged file by relative `source_name` from inside the staging directory
- regression coverage:
  - `experiments.test_tac_lossless_codecs.TacLosslessCodecsTests.test_zpaq_roundtrip_uses_subprocess_and_restores_file`

## Implication

- any `zpaq` archive created before that fix is not trustworthy for exact replay
- rebuild the `zpaq_baseline` archive from the fixed code before attempting another exact full replay
