# Codex Findings: MLX YUV6 Primitive Parity

Date: 2026-05-22T16:21:18Z

## Verdict

PROCEED. The MLX-native `rgb_to_yuv6` primitive matches upstream
`frame_utils.rgb_to_yuv6` exactly on a deterministic full-resolution fixture.

## Why This Matters

The prior PR101 MLX primitive PV proved the bilinear resize route, but its memo
explicitly left `rgb_to_yuv6 + normalize` as deterministic pure tensor math
rather than a separately measured primitive. This closes that cheap conformance
gap for PoseNet scorer-input generation.

## What Landed

- `src/tac/local_acceleration/mlx_yuv6_primitive_parity.py`
  - MLX-native BT.601 YUV6 conversion.
  - Upstream PyTorch comparator using `upstream/frame_utils.py::rgb_to_yuv6`.
  - False-authority manifest with array SHA-256 custody and delta summary.
- `tools/audit_mlx_yuv6_primitive_parity.py`
  - CLI for deterministic fixture or caller-supplied `.npy` input.
- `src/tac/tests/test_mlx_yuv6_primitive_parity.py`
  - Unit parity, fail-closed shape validation, and CLI manifest coverage.

## Empirical Anchor

Artifact:

- `.omx/research/mlx_yuv6_primitive_parity_20260522T162118Z.json`

Command:

```bash
.venv/bin/python tools/audit_mlx_yuv6_primitive_parity.py \
  --output .omx/research/mlx_yuv6_primitive_parity_20260522T162118Z.json \
  --repo-root . \
  --seed 101 \
  --batch 3 \
  --height 384 \
  --width 512 \
  --epsilon 1e-5 \
  --run-id mlx_yuv6_primitive_parity_20260522T162118Z
```

Result:

- `verdict = PASS_MLX_YUV6_PRIMITIVE_PARITY`
- `input_shape = [3, 3, 384, 512]`
- `output_shape = [3, 6, 192, 256]`
- `max_abs_delta = 0.0`
- `rms_delta = 0.0`
- `score_claim = false`
- `ready_for_exact_eval_dispatch = false`

## Verification

- `.venv/bin/python -m pytest src/tac/tests/test_mlx_yuv6_primitive_parity.py src/tac/tests/test_mlx_preprocess.py -q`
- `.venv/bin/python -m ruff check src/tac/local_acceleration/mlx_yuv6_primitive_parity.py tools/audit_mlx_yuv6_primitive_parity.py src/tac/tests/test_mlx_yuv6_primitive_parity.py`

## Residual Risk

This is primitive-level parity only. It does not prove full FastViT-T12 PoseNet
or EfficientNet-B2-UNet SegNet architecture-level MLX scorer conformance, and
it is not a contest score or promotion artifact.
