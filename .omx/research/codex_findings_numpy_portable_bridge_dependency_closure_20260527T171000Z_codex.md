# Codex Findings: numpy-portable bridge dependency closure

UTC: 2026-05-27T17:10:00Z

## Finding

The pushed BoostNeRV numpy-portable runtime commit depended on shared bridge
helpers that were still only present in the worktree. A clean checkout of
`origin/main` would fail to import BoostNeRV because `as_numpy_array` was not
yet in `tac.substrates._shared.numpy_portable_inflate`.

## Landing

- Added `as_numpy_array(...)` as the public duck-typed tensor/array bridge for
  archive packers that must accept torch, MLX, or numpy values without importing
  a training framework at module import time.
- Added shared PyTorch-OIHW/NHWC numpy convolution adapters:
  `conv2d_nhwc_oihw`, `depthwise_conv2d_nhwc_oihw`, and
  `pointwise_conv1x1_nhwc_oihw`.
- Registered the helpers in `DECODE_PRIMITIVES` and `__all__`.
- Added torch-parity tests for the grouped, depthwise, and pointwise adapters.

## Verification

- `.venv/bin/python -m ruff check src/tac/substrates/_shared/numpy_portable_inflate.py src/tac/substrates/_shared/tests/test_numpy_portable_inflate.py`
- `.venv/bin/python -m py_compile src/tac/substrates/_shared/numpy_portable_inflate.py src/tac/substrates/_shared/tests/test_numpy_portable_inflate.py`
- `.venv/bin/python -m pytest src/tac/substrates/_shared/tests/test_numpy_portable_inflate.py -q`
  - `53 passed`
- `.venv/bin/python -m pytest src/tac/substrates/boost_nerv/tests/test_boost_nerv.py src/tac/substrates/boost_nerv/tests/test_boost_nerv_numpy_inflate.py -q`
  - `24 passed`

## Authority Boundary

This is a runtime infrastructure fix. It grants no score, dispatch, promotion,
rank, or kill authority. It only closes the import/runtime dependency required
for clean-checkout numpy-portable substrate inflates.
