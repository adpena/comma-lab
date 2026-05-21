# Codex Findings: MLX-to-PyTorch Dtype Preservation

UTC: 2026-05-21T21:36:00Z

## Verdict

PROCEED. The MLX-to-PyTorch export path no longer silently casts every tensor
to fp32.

## What Landed

- `export_mlx_state_dict_to_torch_pt(...)` now preserves each NumPy tensor dtype
  by default.
- Added explicit `force_float32_names=(...)` for the rare case where a known
  float weight must be normalized to fp32 for a PyTorch runtime.
- The export manifest now records per-tensor:
  - source dtype;
  - export dtype;
  - shape;
  - full SHA-256;
  - whether fp32 casting was forced.
- The manifest includes a `dtype_policy` block so downstream consumers can
  audit whether a tensor was preserved or deliberately cast.

## Why This Matters

The previous exporter converted every tensor through `arr.astype(np.float32)`.
That is unsafe for integer indices, boolean masks, quantized/codebook state, and
other non-float buffers. Those surfaces are exactly the kind of state a local
MLX training lane may need to export into PyTorch for contest-axis validation.

## Verification

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  src/tac/tests/test_mlx_to_pytorch_export.py -q
```

Result: `3 passed in 0.57s`.

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m compileall -q \
  src/tac/local_acceleration/mlx_to_pytorch_export.py \
  src/tac/tests/test_mlx_to_pytorch_export.py
```

Result: pass.

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/review_tracker.py \
  policy-check src/tac/local_acceleration/mlx_to_pytorch_export.py \
  src/tac/tests/test_mlx_to_pytorch_export.py
```

Result: `0 violations`.

```bash
git diff --check -- \
  src/tac/local_acceleration/mlx_to_pytorch_export.py \
  src/tac/tests/test_mlx_to_pytorch_export.py
```

Result: pass.
