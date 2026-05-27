# Codex Findings: ATW Numpy-Portable Inflate

Date: 2026-05-27T16:57:30Z

## Verdict

ATW codec V1 now has a torch-free archive parse and inflate path for the decode
side of the MLX-first / numpy-portable contract. Training and authoring can stay
MLX/PyTorch-side, but shipped inflate consumes the ATW1 archive through numpy
arrays and reconstructs the real decoder plus Wyner-Ziv side-info head without
importing torch or mlx.

## Landed Integration

- Added `parse_archive_numpy(...)` and `ATWCodecArchiveNumpy` for torch-free
  ATW1 section parsing.
- Added a numpy state-dict deserializer that reads the same length-prefixed fp16
  state-dict byte grammar produced by the torch-side archive writer.
- Rewired `atw_codec_v1/inflate.py` to use only numpy + brotli and the canonical
  numpy-portable inflate bridge.
- Preserved contest CLI safety: single-member `0.bin`/`x` reader, fail-closed
  ambiguity check, safe relative file-list raw output paths, and per-video raw
  byte-count assertion for full 600-pair archives.
- Added tests proving no torch/mlx imports in inflate, torch-free parser code,
  numpy parser parity against the torch parser, numpy decode parity against
  `ATWCodec.reconstruct_from_wz_residual`, WZ side-info operational consumption,
  single-member archive reader behavior, and safe raw output path handling.

## Authority Boundary

This is a runtime/portability closure only. It does not claim score, promote an
archive, rank/kill an ATW candidate, or dispatch exact eval. It makes the ATW
inflate runtime eligible for byte-closed replay work once an actual candidate
archive is selected through the normal exact-readiness gate.

## Verification

- `.venv/bin/python -m ruff check src/tac/substrates/atw_codec_v1/archive.py src/tac/substrates/atw_codec_v1/inflate.py src/tac/substrates/atw_codec_v1/tests/test_atw_codec_v1_numpy_inflate.py`
- `.venv/bin/python -m py_compile src/tac/substrates/atw_codec_v1/archive.py src/tac/substrates/atw_codec_v1/inflate.py src/tac/substrates/atw_codec_v1/tests/test_atw_codec_v1_numpy_inflate.py`
- `.venv/bin/python -m pytest src/tac/substrates/atw_codec_v1/tests/test_atw_codec_v1_numpy_inflate.py -q`
  - `7 passed`
- `.venv/bin/python -m pytest src/tac/tests/test_atw_codec_v1_scaffold.py src/tac/substrates/atw_codec_v1/tests/test_atw_codec_v1_numpy_inflate.py -q`
  - `57 passed`
- `.venv/bin/python tools/lane_maturity.py validate`
  - `1439 lane(s) validated cleanly`

## Remaining Work

The next ATW closure is byte-closed replay: build or select an ATW1 archive,
run the numpy inflate path through the same receiver/runtime proof boundary as
the repair materializers, and feed any MLX-local response deltas into the repair
campaign posterior as advisory-only acquisition signal.
