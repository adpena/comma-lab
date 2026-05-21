# Codex Findings: MLX SegNet EfficientNet Block Parity

Date: 2026-05-21T22:31:43Z

## Scope

Started the SegNet side of the MLX scorer port by covering the first EfficientNet
encoder block families used by upstream SegNet:

- `timm.layers.norm_act.BatchNormAct2d` activation handling
- `timm.models._efficientnet_blocks.SqueezeExcite`
- `timm.models._efficientnet_blocks.DepthwiseSeparableConv`
- `timm.models._efficientnet_blocks.InvertedResidual`

This is SegNet encoder primitive parity only, not full SegNet parity and not a
full auth-scorer claim.

## Landed implementation

- `src/tac/local_acceleration/mlx_scorer_adapters.py`
  - `MLXBatchNormAct2dAdapter`
  - `MLXEfficientNetSqueezeExciteAdapter`
  - `MLXDepthwiseSeparableConvAdapter`
  - `MLXInvertedResidualAdapter`
  - `torch_efficientnet_block_to_mlx`
  - `run_mlx_efficientnet_block_nchw`
  - `mlx_silu`
- `src/tac/tests/test_mlx_scorer_adapters.py`
  - upstream-weight MLX CPU parity for SegNet `encoder.model.blocks[0][0]`
  - upstream-weight MLX CPU parity for SegNet `encoder.model.blocks[1][0]`
  - MLX GPU drift measurement for `encoder.model.blocks[1][0]`

## Empirical parity

All values use upstream SegNet weights from `modules.segnet_sd_path`.

| Slice | Device | Input seed | Output shape | max_abs vs PyTorch | mean_abs vs PyTorch |
|---|---|---:|---:|---:|---:|
| `blocks[0][0]` DepthwiseSeparableConv | MLX CPU | `73` | `(1, 16, 8, 10)` | `2.47955322265625e-05` | `3.902986463799607e-06` |
| `blocks[0][0]` DepthwiseSeparableConv | MLX GPU | `73` | `(1, 16, 8, 10)` | `0.13507318496704102` | `0.01271683257073164` |
| `blocks[1][0]` InvertedResidual | MLX CPU | `79` | `(1, 24, 4, 5)` | `8.106231689453125e-06` | `1.2522098131739767e-06` |
| `blocks[1][0]` InvertedResidual | MLX GPU | `79` | `(1, 24, 4, 5)` | `0.03046560287475586` | `0.005410439800471067` |

The GPU drift is materially larger than the PoseNet slices and is recorded only
as non-authoritative acceleration signal. MLX CPU remains the parity gate.

## Verification

```text
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest src/tac/tests/test_mlx_scorer_adapters.py -q
25 passed in 3.17s

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  src/tac/tests/test_local_acceleration.py \
  src/tac/tests/test_mlx_preprocess.py \
  src/tac/tests/test_mlx_cache_audit.py \
  src/tac/tests/test_mlx_scorer_fidelity.py \
  src/tac/tests/test_mlx_scorer_adapters.py \
  src/tac/tests/test_mlx_scorer_state_map.py \
  src/tac/tests/test_mlx_scorer_port_inventory.py -q
81 passed in 9.61s
```

## Remaining boundary

Next SegNet conformance slice should compose these primitives into the
EfficientNet encoder stem and staged feature extraction contract, then proceed
to the U-Net decoder blocks and segmentation head.
