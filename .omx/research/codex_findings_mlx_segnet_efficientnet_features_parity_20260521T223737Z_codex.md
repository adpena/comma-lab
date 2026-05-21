# Codex Findings: MLX SegNet EfficientNet Features Parity

Date: 2026-05-21T22:37:37Z

## Scope

Extended SegNet MLX parity from EfficientNet stage prefixes to the full timm
`EfficientNetFeatures` feature-list contract used by
`segmentation_models_pytorch`'s `TimmUniversalEncoder`.

The adapter now covers:

- EfficientNet stem
- all EfficientNet block stages 0 through 6
- `_stage_out_idx` feature collection
- five feature-map outputs from upstream `encoder.model(x)`

This is still encoder-model parity only. `TimmUniversalEncoder` wrapper behavior,
U-Net decoder blocks, segmentation head, and full SegNet logits remain open.

## Landed implementation

- `src/tac/local_acceleration/mlx_scorer_adapters.py`
  - `MLXEfficientNetFeaturesAdapter`
  - `torch_efficientnet_features_to_mlx`
  - `run_mlx_efficientnet_features_nchw`
- `src/tac/tests/test_mlx_scorer_adapters.py`
  - upstream-weight MLX CPU parity for all EfficientNet feature outputs

## Empirical parity

Fixed input: `torch.manual_seed(101)`, shape `(1, 3, 64, 80)`, upstream SegNet
weights from `modules.segnet_sd_path`.

| Feature | Output shape | max_abs vs PyTorch | mean_abs vs PyTorch |
|---:|---:|---:|---:|
| 0 | `(1, 16, 32, 40)` | `3.1948089599609375e-05` | `4.102948878426105e-06` |
| 1 | `(1, 24, 16, 20)` | `0.00028324127197265625` | `2.503556970623322e-05` |
| 2 | `(1, 48, 8, 10)` | `0.00028586387634277344` | `2.0522733393590897e-05` |
| 3 | `(1, 120, 4, 5)` | `6.80088996887207e-05` | `1.2281798262847587e-05` |
| 4 | `(1, 352, 2, 3)` | `2.288818359375e-05` | `3.01904447042034e-06` |

## Verification

```text
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest src/tac/tests/test_mlx_scorer_adapters.py -q
29 passed in 4.40s

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  src/tac/tests/test_local_acceleration.py \
  src/tac/tests/test_mlx_preprocess.py \
  src/tac/tests/test_mlx_cache_audit.py \
  src/tac/tests/test_mlx_scorer_fidelity.py \
  src/tac/tests/test_mlx_scorer_adapters.py \
  src/tac/tests/test_mlx_scorer_state_map.py \
  src/tac/tests/test_mlx_scorer_port_inventory.py -q
85 passed in 11.77s
```

## Remaining boundary

Next SegNet adapter boundary is `TimmUniversalEncoder.forward`: prepend the
input tensor as the scale-1 feature exactly as SMP does, then port
`UnetDecoderBlock` and `SegmentationHead`.
