# Codex Findings: Nirvana Numpy-Portable Inflate

Date: 2026-05-27T17:00:17Z

## Verdict

Nirvana now has a numpy-portable archive parser and inflate runtime path aligned
with the MLX-first / numpy-portable contract. The archive format has moved to
NRV1 schema v2 for torch-free decoder state-dict storage, so shipped inflate no
longer needs torch pickle reconstruction.

## Landed Integration

- Exported `NirvanaArchiveNumpy` and `parse_archive_numpy`.
- Replaced the torch-pickle decoder blob with the canonical numpy-portable
  state-dict bridge under brotli.
- Rewired `nirvana/inflate.py` to use numpy + brotli + PIL and the shared
  numpy-portable primitives for linear, depthwise/pointwise conv, pixel shuffle,
  bilinear resize, sigmoid, and patch stitching.
- Added parser parity and no torch/mlx inflate import tests in the existing
  Nirvana test module.

## Authority Boundary

This is a runtime portability closure and parser contract migration. It does
not claim score, promote Nirvana, rank/kill a candidate, spend budget, or
dispatch exact eval. It makes the Nirvana substrate more eligible for byte-closed
local replay once an actual archive/candidate enters the exact-readiness path.

## Verification

- `.venv/bin/python -m ruff check src/tac/substrates/nirvana/__init__.py src/tac/substrates/nirvana/archive.py src/tac/substrates/nirvana/inflate.py src/tac/substrates/nirvana/tests/test_nirvana.py`
- `.venv/bin/python -m py_compile src/tac/substrates/nirvana/__init__.py src/tac/substrates/nirvana/archive.py src/tac/substrates/nirvana/inflate.py src/tac/substrates/nirvana/tests/test_nirvana.py`
- `.venv/bin/python -m pytest src/tac/substrates/nirvana/tests/test_nirvana.py -q`
  - `13 passed`
- `.venv/bin/python tools/lane_maturity.py validate`
  - `1439 lane(s) validated cleanly`

## Remaining Work

The next step is byte-closed replay: generate or select an NRV1-v2 archive,
inflate it through the numpy runtime, capture receiver proof, and feed any
local MLX response signal into the same false-authority repair/stackability
posterior instead of treating Nirvana as a standalone research artifact.
