# Codex Findings: MLX PoseNet FastViT Vision Parity

Date: 2026-05-21T22:22:32Z

## Scope

Extended the MLX PoseNet adapter surface through the full upstream FastViT
vision trunk:

- `vision.stem`
- `vision.stages[0:4]`
- `vision.final_conv`
- `vision.head`

Newly covered terminal modules:

- `timm.layers.squeeze_excite.SEModule`
- `timm.layers.classifier.ClassifierHead` with average pooling

## Landed implementation

- `src/tac/local_acceleration/mlx_scorer_adapters.py`
  - `MLXSEModuleAdapter`
  - `MLXClassifierHeadAdapter`
  - `MLXFastVitVisionAdapter`
  - `torch_fastvit_vision_to_mlx`
  - `run_mlx_fastvit_vision_nchw`
  - explicit MLX `eval`/`synchronize` before NumPy extraction
  - explicit MLX `clear_cache` when entering a temporary device boundary
- `src/tac/tests/test_mlx_scorer_adapters.py`
  - later-stage parity tests for stages 2 and 3
  - full FastViT vision parity test on MLX CPU
  - full FastViT vision GPU drift measurement

## Hidden-state bug caught

The full vision CPU parity test passed in isolation but failed after the broader
MLX test suite had exercised other MLX paths:

```text
test_posenet_fastvit_vision_matches_torch_on_mlx_cpu
max_abs = 0.9210568070411682
```

The adapter runners now force MLX graph evaluation and synchronization before
NumPy conversion, and `temporary_mlx_device` clears MLX cache when entering an
explicit CPU/GPU conformance block. The broader suite then passed reliably.

This is a scorer-conformance relevant hardening: hidden MLX cache/device state
cannot be allowed to masquerade as model drift.

## Empirical parity

All values use upstream PoseNet weights from `modules.posenet_sd_path`.

| Slice | Device | Input seed | Output shape | max_abs vs PyTorch | mean_abs vs PyTorch |
|---|---|---:|---:|---:|---:|
| stage 2 | MLX CPU | `59` | `(1, 256, 4, 5)` | `5.7220458984375e-06` | `5.23999915458262e-07` |
| stage 3 | MLX CPU | `61` | `(1, 512, 2, 3)` | `1.71661376953125e-05` | `8.458881097794801e-07` |
| full FastViT vision | MLX CPU | `67` | `(1, 2048)` | `1.1920928955078125e-07` | `2.4234850570792332e-08` |
| full FastViT vision | MLX GPU | `67` | `(1, 2048)` | `0.008238913491368294` | `0.0018763679545372725` |

The GPU number is recorded as drift measurement only. MLX CPU remains the local
conformance gate for this adapter surface.

## Verification

```text
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest src/tac/tests/test_mlx_scorer_adapters.py -q
20 passed in 2.43s

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  src/tac/tests/test_local_acceleration.py \
  src/tac/tests/test_mlx_preprocess.py \
  src/tac/tests/test_mlx_cache_audit.py \
  src/tac/tests/test_mlx_scorer_fidelity.py \
  src/tac/tests/test_mlx_scorer_adapters.py \
  src/tac/tests/test_mlx_scorer_state_map.py \
  src/tac/tests/test_mlx_scorer_port_inventory.py -q
76 passed in 8.76s
```

## Remaining boundary

PoseNet still needs its non-vision head ported:

- input normalization by `_mean` / `_std`
- `summarizer`
- `ResBlock`
- `AllNorm`
- `Hydra`

After PoseNet end-to-end parity, SegNet/EfficientNet remains the large separate
surface before a full auth-scorer MLX parity claim can be considered.
