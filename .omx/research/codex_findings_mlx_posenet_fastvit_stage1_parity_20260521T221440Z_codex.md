# Codex Findings: MLX PoseNet FastViT Stage1 Parity

Date: 2026-05-21T22:14:40Z

## Scope

Extended the MLX PoseNet adapter surface from an individual `RepMixerBlock` to
FastViT stage execution through `vision.stages[1]`.

Newly covered upstream modules:

- `timm.models.fastvit.ReparamLargeKernelConv`
- `timm.models.fastvit.PatchEmbed`
- `timm.models.fastvit.FastVitStage`

The implementation remains eval-mode only and refuses unsupported branches
instead of silently approximating them.

## Landed implementation

- `src/tac/local_acceleration/mlx_scorer_adapters.py`
  - `MLXReparamLargeKernelConvAdapter`
  - `MLXPatchEmbedAdapter`
  - `MLXFastVitStageAdapter`
  - `torch_patch_embed_to_mlx`
  - `torch_fastvit_stage_to_mlx`
  - `run_mlx_patch_embed_nchw`
  - `run_mlx_fastvit_stage_nchw`
- `src/tac/tests/test_mlx_scorer_adapters.py`
  - MLX CPU parity for PoseNet FastViT stage 0
  - MLX CPU parity for PoseNet stage 1 `PatchEmbed`
  - MLX CPU parity for PoseNet FastViT stage 1
  - MLX GPU drift measurement for PoseNet FastViT stage 1

## Empirical parity

All values use upstream PoseNet weights from `modules.posenet_sd_path`.

| Slice | Device | Input seed | Output shape | max_abs vs PyTorch | mean_abs vs PyTorch |
|---|---|---:|---:|---:|---:|
| stage 0 | MLX CPU | `43` | `(1, 64, 16, 20)` | `6.103515625e-05` | `4.872184035775717e-06` |
| stage 1 PatchEmbed | MLX CPU | `47` | `(1, 128, 8, 10)` | `1.1920928955078125e-06` | `3.322063690802679e-08` |
| stage 1 | MLX CPU | `53` | `(1, 128, 8, 10)` | `4.291534423828125e-06` | `3.758516129437339e-07` |
| stage 1 | MLX GPU | `53` | `(1, 128, 8, 10)` | `4.0531158447265625e-06` | `3.7950297837596736e-07` |

This is still module/stage parity, not a full auth-scorer conformance claim.

## Verification

```text
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest src/tac/tests/test_mlx_scorer_adapters.py -q
16 passed in 1.75s

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  src/tac/tests/test_local_acceleration.py \
  src/tac/tests/test_mlx_preprocess.py \
  src/tac/tests/test_mlx_cache_audit.py \
  src/tac/tests/test_mlx_scorer_fidelity.py \
  src/tac/tests/test_mlx_scorer_adapters.py \
  src/tac/tests/test_mlx_scorer_state_map.py \
  src/tac/tests/test_mlx_scorer_port_inventory.py -q
72 passed in 7.98s
```

## Remaining boundary

The same stage adapter should cover later PoseNet stages structurally, but that
must still be tested at stages 2 and 3. The remaining PoseNet-specific blockers
after stage traversal are `final_conv` with `SEModule` and the classifier head.
SegNet/EfficientNet remains separate work before full auth-scorer parity.
