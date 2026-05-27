# Codex Findings: Z5 numpy archive lazy-import hardening

UTC: 2026-05-27T17:12:58Z

## Context

The Z5 predictive-coding world-model substrate had been ported to a numpy-portable inflate runtime. The runtime consumed the real predictor, decoder, latent initialization, residuals, and ego-motion side information via `parse_archive_numpy(...)`.

## Finding

`inflate.py` imported `parse_archive_numpy` from `archive.py`, but `archive.py` still imported `torch` at module import time. That creates a false portability boundary: the inflate source itself contains no torch import, but a clean contest runtime without torch can still fail while importing the archive parser module.

## Landing

- Removed the top-level torch import from the Z5 archive grammar.
- Converted archive packing and quantization helpers to use the shared `as_numpy_array(...)` bridge so archive writers can accept torch, MLX-like, or numpy arrays without importing a training framework.
- Kept `parse_archive(...)` torch-compatible through a lazy `importlib.import_module("torch")` call on the training/eval path only.
- Added tests that reject top-level torch/MLX imports in `archive.py`, preserve the torch-free `parse_archive_numpy(...)` path, prove exact numpy-vs-torch parser parity, prove predictor consumption, and cover single-member archive plus safe file-list output behavior.

## Verification

- `.venv/bin/python -m ruff check src/tac/substrates/_shared/numpy_portable_inflate.py src/tac/substrates/_shared/tests/test_numpy_portable_inflate.py src/tac/substrates/z5_predictive_coding_world_model/__init__.py src/tac/substrates/z5_predictive_coding_world_model/archive.py src/tac/substrates/z5_predictive_coding_world_model/inflate.py src/tac/substrates/z5_predictive_coding_world_model/tests/test_z5_numpy_inflate.py`
- `.venv/bin/python -m py_compile src/tac/substrates/_shared/numpy_portable_inflate.py src/tac/substrates/_shared/tests/test_numpy_portable_inflate.py src/tac/substrates/z5_predictive_coding_world_model/__init__.py src/tac/substrates/z5_predictive_coding_world_model/archive.py src/tac/substrates/z5_predictive_coding_world_model/inflate.py src/tac/substrates/z5_predictive_coding_world_model/tests/test_z5_numpy_inflate.py`
- `.venv/bin/python -m pytest src/tac/substrates/_shared/tests/test_numpy_portable_inflate.py src/tac/substrates/z5_predictive_coding_world_model/tests/test_z5_numpy_inflate.py -q`
- `.venv/bin/python -m pytest src/tac/substrates/z5_predictive_coding_world_model/tests/test_z5_substrate.py src/tac/substrates/z5_predictive_coding_world_model/tests/test_parse_z5pcwm1_archive_bytes_canonical.py src/tac/substrates/z5_predictive_coding_world_model/tests/test_z5_numpy_inflate.py -q`

## Authority

This is runtime-custody hardening for a local MLX/numpy-portable substrate. It is not a contest score claim, not a rank/kill decision, and not promotion authority. Promotion still requires byte-closed archive custody plus exact contest CPU/CUDA auth evaluation.
