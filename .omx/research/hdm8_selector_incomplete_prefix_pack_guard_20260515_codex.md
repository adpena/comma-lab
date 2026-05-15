# HDM8 Selector Incomplete Prefix Pack Guard - 2026-05-15

## Scope

Prevent CUDA-prefix selector rows from being accidentally packed as a complete
HDM8 selector archive.

## Problem

The HDM8 CUDA-prefix screen can emit useful per-pair component rows for a
subset such as 32 pairs. Those rows are valid calibration evidence, but they
are not a submission selector: the runtime decodes 600 pairs and would fail
once `pair_index >= len(selector_indices)`.

## Fix

`tools/build_hdm8_film_grain_sidecar_packet.py` now refuses to pack a selector
archive unless the embedded selector contains exactly 600 pair indices.

Regression:

- `src/tac/tests/test_hdm8_film_grain_sidecar.py::test_builder_refuses_to_pack_incomplete_prefix_selector`

Verification:

```bash
.venv/bin/ruff check tools/build_hdm8_film_grain_sidecar_packet.py src/tac/tests/test_hdm8_film_grain_sidecar.py
PYTHONPATH=src:upstream:$PWD .venv/bin/python -m pytest src/tac/tests/test_hdm8_film_grain_sidecar.py -q
```

Result:

- ruff: passed
- pytest: `18 passed`

## Effect

The 32-pair CUDA-prefix sweep remains valid for gate calibration and mode
selection. A byte-closed exact-eval candidate must be built only from a full
600-pair CUDA rowset or another complete selector source.
