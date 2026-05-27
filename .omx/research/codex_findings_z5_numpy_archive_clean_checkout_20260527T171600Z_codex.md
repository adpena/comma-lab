# Codex Findings: Z5 numpy archive clean-checkout closure

UTC: 2026-05-27T17:16:00Z

## Finding

The Z5 numpy-portable inflate path had the same clean-checkout boundary as the
other class-shift substrates: the runtime parser was numpy-only, but the archive
module still imported torch at module load and the archive packer assumed torch
tensor methods. That is acceptable for training, but not for a vendored
numpy-portable runtime tree.

## Landing

- Removed the top-level torch dependency from
  `z5_predictive_coding_world_model/archive.py`.
- Moved training-side torch reconstruction behind dynamic imports in
  `parse_archive(...)` and `_deserialize_state_dict(...)`.
- Switched pack/quantize paths to the shared `as_numpy_array(...)` bridge so
  torch, MLX, and numpy values can enter the archive writer without a framework
  import at module load.
- Added tests proving no top-level torch/MLX archive imports, exact numpy parser
  parity, single-member archive selection, ambiguous archive rejection, and safe
  raw-output path handling.

## Verification

- `.venv/bin/python -m ruff check src/tac/substrates/z5_predictive_coding_world_model/archive.py src/tac/substrates/z5_predictive_coding_world_model/tests/test_z5_numpy_inflate.py`
- `.venv/bin/python -m py_compile src/tac/substrates/z5_predictive_coding_world_model/archive.py src/tac/substrates/z5_predictive_coding_world_model/tests/test_z5_numpy_inflate.py`
- `.venv/bin/python -m pytest src/tac/substrates/z5_predictive_coding_world_model/tests/test_z5_numpy_inflate.py -q`
  - `9 passed`
- `.venv/bin/python -m pytest src/tac/substrates/z5_predictive_coding_world_model/tests -q`
  - `56 passed, 1 existing warning`
- `write_numpy_portable_contest_runtime(..., substrate_pkg_name="z5_predictive_coding_world_model", ...)`
  - `z5_runtime_emit_ok`

## Authority Boundary

This is runtime/parser infrastructure only. It does not claim score movement,
promotion eligibility, rank/kill authority, or exact-eval readiness without a
byte-closed archive and contest CPU/CUDA auth-axis evaluation.
