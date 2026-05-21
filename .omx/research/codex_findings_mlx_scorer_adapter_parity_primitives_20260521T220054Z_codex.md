# Codex Findings: MLX Scorer Adapter Parity Primitives

timestamp_utc: 2026-05-21T22:00:54Z
agent: codex
lane_id: lane_mlx_auth_scorer_training_signal_fidelity_20260521
evidence_grade: macOS-MLX-research-signal
score_claim: false
promotion_eligible: false

## Summary

Codex landed the first executable PyTorch-to-MLX scorer adapter primitives:

- `torch_conv2d_to_mlx`
- `torch_batchnorm2d_to_mlx`
- `torch_linear_to_mlx`
- NCHW wrapper runners for Conv2d/BatchNorm2d
- `temporary_mlx_device("cpu"|"gpu")` for explicit parity-axis selection

These are not a full scorer port. They are the first tested adapters that reduce the state-map blocker class from abstract "requires adapter" to executable tensor parity surfaces.

## Parity Result

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  src/tac/tests/test_mlx_scorer_adapters.py \
  -q
```

Observed: `6 passed`.

Covered:

- Conv2d NCHW PyTorch vs MLX NHWC adapter parity for standard, grouped, and depthwise configurations.
- BatchNorm2d eval running-stat parity.
- Linear parity on MLX CPU.
- Linear GPU drift is explicitly measured and bounded as a research signal, not silently treated as exact.

## Important Conformance Finding

On the local M5 Max, a fixed PyTorch `nn.Linear(13, 7)` matched MLX CPU exactly in the test, but MLX GPU differed by roughly `8.8e-4` max absolute error. This means the full scorer port must distinguish:

- MLX CPU: useful for exact adapter parity gates.
- MLX GPU: useful for accelerated training/search only after component drift calibration.

This is directly aligned with the operator requirement: do not claim "1:1 upstream auth scorer" from the fast path until the fast path itself has component-level drift evidence.

## Next Action

Bind these primitives into a PoseNet stem-block adapter test next. That is the first scorer subgraph where Conv2d + BatchNorm + activation + layout all interact.
