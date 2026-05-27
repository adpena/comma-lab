# Codex Findings: BoostNeRV numpy-portable inflate hardening

UTC: 2026-05-27T17:06:12Z

## Context

Local `main` already carried `90f294397 boost_nerv: port inflate to numpy-portability via canonical bridge`, which moved BoostNeRV archive parsing and inflate into the MLX-first / numpy-portable decode contract. That landing correctly removed torch/MLX imports from the shipped inflate path and added a parity test for the real base decoder plus boosting heads.

## Finding

The bridge was coherent, but two receiver/runtime-boundary details still needed the same adversarial hardening applied to ATW:

- `main_cli` read only `0.bin`, so archives using the contest-compatible `x` member name would fail while ambiguous `0.bin` plus `x` packets were not rejected.
- File-list output handling used `Path(fname).stem`, which did not preserve safe subdirectories and did not make the file-list path validation explicit.

Both are runtime-custody problems, not score signals. They must stay on the receiver decode-only side and must not grant score or promotion authority.

## Landing

- Added `_read_single_member_archive_bytes(...)` to accept exactly one of `0.bin` or `x`, and fail closed on missing or ambiguous archive members.
- Added `_png_output_dir_numpy(...)` to reject absolute paths, traversal, empty entries, and doubled separators while preserving safe relative subdirectories.
- Extended BoostNeRV numpy inflate tests for member selection, ambiguity rejection, safe path behavior, torch/MLX-free inflate imports, numpy parser parity, and PNG decode parity against the torch model.
- Kept `parse_archive_numpy(...)` as a torch-free parser while `parse_archive(...)` remains the training/eval parity bridge.

## Verification

- `.venv/bin/python -m ruff check src/tac/substrates/boost_nerv/__init__.py src/tac/substrates/boost_nerv/archive.py src/tac/substrates/boost_nerv/inflate.py src/tac/substrates/boost_nerv/tests/test_boost_nerv.py src/tac/substrates/boost_nerv/tests/test_boost_nerv_numpy_inflate.py`
- `.venv/bin/python -m py_compile src/tac/substrates/boost_nerv/__init__.py src/tac/substrates/boost_nerv/archive.py src/tac/substrates/boost_nerv/inflate.py src/tac/substrates/boost_nerv/tests/test_boost_nerv.py src/tac/substrates/boost_nerv/tests/test_boost_nerv_numpy_inflate.py`
- `.venv/bin/python -m pytest src/tac/substrates/boost_nerv/tests/test_boost_nerv.py src/tac/substrates/boost_nerv/tests/test_boost_nerv_numpy_inflate.py -q`

## Authority

This is a runtime-closure and portability hardening artifact. It is not a contest score claim, not a rank/kill signal, and not promotion authority. Any score movement still requires byte-closed archive custody plus exact contest CPU/CUDA auth evaluation.
