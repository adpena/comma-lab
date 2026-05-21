# Codex Findings: MLX SegNet TimmUniversalEncoder Parity

Date: 2026-05-21T22:40:46Z

## Scope

Extended SegNet MLX parity from timm `EfficientNetFeatures` to the
`segmentation_models_pytorch` `TimmUniversalEncoder` wrapper contract consumed
by the U-Net decoder.

The wrapper behavior covered here:

- rejects unsupported channel-last and transformer-style paths
- delegates to the MLX EfficientNetFeatures adapter
- prepends the original input tensor as the scale-1 feature when `_is_vgg_style`
  is false, matching SMP `TimmUniversalEncoder.forward`

This is encoder parity. Decoder and segmentation-head parity remain open.

## Landed implementation

- `src/tac/local_acceleration/mlx_scorer_adapters.py`
  - `MLXTimmUniversalEncoderAdapter`
  - `torch_timm_universal_encoder_to_mlx`
  - `run_mlx_timm_universal_encoder_nchw`
- `src/tac/tests/test_mlx_scorer_adapters.py`
  - upstream-weight MLX CPU parity for all six SegNet encoder features

## Empirical parity

Fixed input: `torch.manual_seed(103)`, shape `(1, 3, 64, 80)`, upstream SegNet
weights from `modules.segnet_sd_path`.

| Feature | Output shape | max_abs vs PyTorch | mean_abs vs PyTorch |
|---:|---:|---:|---:|
| 0 | `(1, 3, 64, 80)` | `0.0` | `0.0` |
| 1 | `(1, 16, 32, 40)` | `2.956390380859375e-05` | `3.988937805843307e-06` |
| 2 | `(1, 24, 16, 20)` | `0.00035190582275390625` | `2.397239768470172e-05` |
| 3 | `(1, 48, 8, 10)` | `0.00018155574798583984` | `1.7016714991768822e-05` |
| 4 | `(1, 120, 4, 5)` | `6.67572021484375e-05` | `1.0320183719159104e-05` |
| 5 | `(1, 352, 2, 3)` | `1.0013580322265625e-05` | `1.8727150745689869e-06` |

## Verification

```text
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest src/tac/tests/test_mlx_scorer_adapters.py -q
30 passed in 4.28s

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  src/tac/tests/test_local_acceleration.py \
  src/tac/tests/test_mlx_preprocess.py \
  src/tac/tests/test_mlx_cache_audit.py \
  src/tac/tests/test_mlx_scorer_fidelity.py \
  src/tac/tests/test_mlx_scorer_adapters.py \
  src/tac/tests/test_mlx_scorer_state_map.py \
  src/tac/tests/test_mlx_scorer_port_inventory.py -q
86 passed in 10.67s
```

## Remaining boundary

The next SegNet conformance slice is the decoder:

- `UnetDecoderBlock` interpolation and skip concatenation semantics
- `Conv2dReLU` blocks
- segmentation head logits

After decoder parity, full SegNet logits can be compared to PyTorch on a small
input, then the full scorer wrapper can be built against the byte-closed
scorer-input cache.
