# Codex Findings: MLX PoseNet Stem Adapter Parity

timestamp_utc: 2026-05-21T22:05:46Z
agent: codex
lane_id: lane_mlx_auth_scorer_training_signal_fidelity_20260521
evidence_grade: macOS-MLX-research-signal
score_claim: false
promotion_eligible: false

## Summary

Codex moved the MLX scorer port from isolated primitive parity into the first real upstream PoseNet scorer subgraph: the three-block FastViT `vision.stem` made of timm `MobileOneBlock`s.

New/extended executable surfaces:

- `MLXConvNormAct2dAdapter`
- `MLXMobileOneBlockAdapter`
- `MLXMobileOneStemAdapter`
- `torch_mobileone_block_to_mlx`
- `torch_mobileone_stem_to_mlx`
- `run_mlx_mobileone_block_nchw`
- `run_mlx_mobileone_stem_nchw`
- `mlx_gelu_tanh`

## Empirical Parity

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  src/tac/tests/test_mlx_scorer_adapters.py \
  -q
```

Observed: `10 passed`.

Covered:

- Upstream-weight PoseNet `vision.stem[0]` parity on MLX CPU.
- Upstream-weight full PoseNet `vision.stem` parity on MLX CPU.
- Separate GPU drift measurement for block0 and full stem.
- Existing Conv2d grouped/depthwise, BatchNorm2d eval, and Linear parity tests.

## Conformance Status

This is still not a full MLX auth scorer. The next unported boundary is the FastViT stage stack after the stem: RepMixer/MobileOne stage blocks, ConvMlp, LayerScale2d, and final pooling/head. The important progress is that the first scorer subgraph is now executable against upstream weights with an explicit CPU parity gate and a separate GPU drift gate.

## Next Action

Port and test the first post-stem FastViT stage block (`vision.stages.0.blocks.0`) using the same rule:

1. MLX CPU parity against upstream PyTorch fixed tensors.
2. MLX GPU drift measured separately.
3. No score or promotion authority until full PoseNet/SegNet component parity is proven and paired auth eval remains authoritative.
