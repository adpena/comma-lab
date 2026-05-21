# Codex Findings: MLX PoseNet End-to-End Parity

Date: 2026-05-21T22:26:47Z

## Scope

Extended the MLX adapter surface from PoseNet FastViT vision parity to
end-to-end PoseNet `{"pose": ...}` output parity on a small upstream-weight
input.

Newly covered upstream modules:

- PoseNet input normalization via `_mean` / `_std`
- `modules.AllNorm`
- `modules.ResBlock`
- PoseNet `summarizer`
- `modules.Hydra`

## Landed implementation

- `src/tac/local_acceleration/mlx_scorer_adapters.py`
  - `MLXAllNormAdapter`
  - `MLXSequential1dAdapter`
  - `MLXResBlockAdapter`
  - `MLXSummarizerAdapter`
  - `MLXHydraAdapter`
  - `MLXPoseNetAdapter`
  - `torch_posenet_to_mlx`
  - `run_mlx_posenet_nchw`
- `src/tac/tests/test_mlx_scorer_adapters.py`
  - full PoseNet MLX CPU parity test
  - full PoseNet MLX GPU drift measurement

## Bug caught

Initial end-to-end PoseNet parity failed:

```text
max_abs = 0.31035327911376953
```

Root cause: upstream `ResBlock.block_b` begins with `ReLU(inplace=True)`.
Because `ResBlock.forward` later adds `a_out + self.block_b(a_out)`, PyTorch's
in-place ReLU mutates the residual base before the final add. The first MLX
port treated ReLU as pure and therefore added the unmutated `a_out`.

The MLX `ResBlock` adapter now preserves this upstream in-place residual
semantics explicitly.

## Empirical parity

Fixed input: `torch.manual_seed(71)`, shape `(1, 12, 64, 80)`, upstream
PoseNet weights from `modules.posenet_sd_path`.

| Device | Output shape | max_abs vs PyTorch | mean_abs vs PyTorch |
|---|---:|---:|---:|
| MLX CPU | `(1, 12)` | `3.337860107421875e-06` | `1.261010766029358e-06` |
| MLX GPU | `(1, 12)` | `3.814697265625e-06` | `1.809249283724057e-06` |

This is PoseNet module parity, not full auth-scorer parity. SegNet and the
full evaluator aggregation path remain open.

## Verification

```text
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest src/tac/tests/test_mlx_scorer_adapters.py -q
22 passed in 3.03s

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  src/tac/tests/test_local_acceleration.py \
  src/tac/tests/test_mlx_preprocess.py \
  src/tac/tests/test_mlx_cache_audit.py \
  src/tac/tests/test_mlx_scorer_fidelity.py \
  src/tac/tests/test_mlx_scorer_adapters.py \
  src/tac/tests/test_mlx_scorer_state_map.py \
  src/tac/tests/test_mlx_scorer_port_inventory.py -q
78 passed in 9.18s
```

## Remaining boundary

Next highest-value conformance slice is SegNet/EfficientNet adapter coverage,
followed by the evaluator wrapper that computes PoseNet and SegNet distances
over the byte-closed scorer-input cache and compares against the recovered
Modal Linux contest-CPU anchor.
