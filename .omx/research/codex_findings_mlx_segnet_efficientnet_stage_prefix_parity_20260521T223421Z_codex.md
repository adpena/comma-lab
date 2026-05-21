# Codex Findings: MLX SegNet EfficientNet Stage Prefix Parity

Date: 2026-05-21T22:34:21Z

## Scope

Composed the first SegNet EfficientNet primitives into encoder structure:

- EfficientNet feature-model stem: `conv_stem` + `bn1`
- EfficientNet sequential stage adapters
- SegNet encoder stages 0 and 1

This advances SegNet encoder parity from individual blocks to staged feature
prefixes. It is still not full SegNet parity and not full auth-scorer parity.

## Landed implementation

- `src/tac/local_acceleration/mlx_scorer_adapters.py`
  - `MLXEfficientNetStemAdapter`
  - `MLXEfficientNetStageAdapter`
  - `torch_efficientnet_stem_to_mlx`
  - `torch_efficientnet_stage_to_mlx`
  - `run_mlx_efficientnet_stem_nchw`
  - `run_mlx_efficientnet_stage_nchw`
- `src/tac/tests/test_mlx_scorer_adapters.py`
  - upstream-weight MLX CPU parity for SegNet EfficientNet stem
  - upstream-weight MLX CPU parity for SegNet EfficientNet stage 0
  - upstream-weight MLX CPU parity for SegNet EfficientNet stage 1

## Empirical parity

All values use upstream SegNet weights from `modules.segnet_sd_path`.

| Slice | Device | Input seed | Output shape | max_abs vs PyTorch | mean_abs vs PyTorch |
|---|---|---:|---:|---:|---:|
| EfficientNet stem | MLX CPU | `83` | `(1, 32, 32, 40)` | `1.9073486328125e-06` | `1.0534690630947807e-07` |
| EfficientNet stage 0 | MLX CPU | `89` | `(1, 16, 8, 10)` | `3.0517578125e-05` | `5.6291928558493964e-06` |
| EfficientNet stage 1 | MLX CPU | `97` | `(1, 24, 4, 5)` | `1.8358230590820312e-05` | `3.518847051964258e-06` |

## Verification

```text
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest src/tac/tests/test_mlx_scorer_adapters.py -q
28 passed in 3.85s

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  src/tac/tests/test_local_acceleration.py \
  src/tac/tests/test_mlx_preprocess.py \
  src/tac/tests/test_mlx_cache_audit.py \
  src/tac/tests/test_mlx_scorer_fidelity.py \
  src/tac/tests/test_mlx_scorer_adapters.py \
  src/tac/tests/test_mlx_scorer_state_map.py \
  src/tac/tests/test_mlx_scorer_port_inventory.py -q
84 passed in 10.70s
```

## Remaining boundary

Next SegNet encoder step is stages 2 through 6 plus the exact
`TimmUniversalEncoder` feature-return contract. After that, the U-Net decoder
blocks and segmentation head must be ported and tested.
