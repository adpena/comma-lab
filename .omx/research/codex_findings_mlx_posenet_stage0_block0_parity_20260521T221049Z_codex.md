# Codex Findings: MLX PoseNet Stage0 Block0 Parity

Date: 2026-05-21T22:10:49Z

## Scope

Extended the MLX scorer adapter surface from the PoseNet FastViT stem into the
first `vision.stages[0].blocks[0]` `RepMixerBlock`, using the upstream
`timm.models.fastvit` forward methods as ground truth:

- `RepMixerBlock.forward`: token mixer, then residual `LayerScale2d(ConvMlp)`.
- `RepMixer.forward`: `x + LayerScale2d(mixer(x) - norm(x))`.
- `ConvMlp.forward`: depthwise `ConvNormAct`, `1x1` conv, GELU-tanh, `1x1` conv.
- `LayerScale2d.forward`: channelwise gamma multiply.

## Landed implementation

- `src/tac/local_acceleration/mlx_scorer_adapters.py`
  - `MLXLayerScale2dAdapter`
  - `MLXConvMlpAdapter`
  - `MLXRepMixerAdapter`
  - `MLXRepMixerBlockAdapter`
  - `torch_conv_mlp_to_mlx`
  - `torch_repmixer_block_to_mlx`
  - `run_mlx_repmixer_block_nchw`
- `src/tac/tests/test_mlx_scorer_adapters.py`
  - upstream-weight MLX CPU parity test for PoseNet `vision.stages[0].blocks[0]`
  - upstream-weight MLX GPU drift measurement test for the same block

## Empirical parity

Fixed input: `torch.manual_seed(41)`, shape `(1, 64, 16, 20)`, upstream
PoseNet weights from `modules.posenet_sd_path`.

| Device | Output shape | max_abs vs PyTorch | mean_abs vs PyTorch |
|---|---:|---:|---:|
| MLX CPU | `(1, 64, 16, 20)` | `3.814697265625e-05` | `1.845528231569915e-06` |
| MLX GPU | `(1, 64, 16, 20)` | `3.814697265625e-05` | `1.9626481844170485e-06` |

This is a block-level adapter parity result, not a full scorer conformance
claim.

## Verification

```text
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest src/tac/tests/test_mlx_scorer_adapters.py -q
12 passed in 1.59s

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  src/tac/tests/test_mlx_scorer_adapters.py \
  src/tac/tests/test_mlx_scorer_state_map.py \
  src/tac/tests/test_mlx_scorer_port_inventory.py -q
17 passed in 5.92s

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  src/tac/tests/test_local_acceleration.py \
  src/tac/tests/test_mlx_preprocess.py \
  src/tac/tests/test_mlx_cache_audit.py \
  src/tac/tests/test_mlx_scorer_fidelity.py \
  src/tac/tests/test_mlx_scorer_adapters.py \
  src/tac/tests/test_mlx_scorer_state_map.py \
  src/tac/tests/test_mlx_scorer_port_inventory.py -q
68 passed in 8.79s
```

## Remaining boundary

The full MLX scorer remains non-authoritative until all upstream PoseNet and
SegNet modules are covered and an end-to-end scorer input/output parity harness
passes against the recovered Modal Linux contest-CPU anchor. The next local
adapter boundary is additional FastViT stage coverage, then EfficientNet SegNet
coverage, then full-model deterministic score parity.
