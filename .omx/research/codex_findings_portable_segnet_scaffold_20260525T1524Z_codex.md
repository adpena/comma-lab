# Codex Findings: Portable SegNet Scaffold

Generated: 2026-05-25T15:24Z
Agent: Codex
Topic: portable scorer primitives / MLX scorer-loss bridge

## Result

Hardened the portable SegNet scaffold now present in
`src/tac/portable_primitives/nn_segnet.py`.

The scaffold exposes:

- `SEGNET_ENCODER_STAGE_SPEC`
- `PortableEfficientNetB2Backbone`
- `PortableSegNet`

This is not score authority and not full upstream SegNet parity. It is a
portable shape-contract and wrapper surface that can become the bridge toward
local MLX scorer-loss training, numpy weight interchange, and eventual
PyTorch/CUDA parity checks.

## Verification

- `ruff check src/tac/portable_primitives/__init__.py src/tac/portable_primitives/nn_segnet.py src/tac/portable_primitives/tests/test_portable_primitives_segnet.py`: pass
- `pytest src/tac/portable_primitives/tests/test_portable_primitives_segnet.py -q`: 8 passed

New tests cover:

- EfficientNet-B2 backbone 6-feature shape contract
- SegNet wrapper preprocessing shape
- SegNet wrapper `forward_3d` output shape
- identity distortion equals zero

## Remaining Work

1. Replace the single-block-per-stage scaffold with byte-stable upstream
   EfficientNet-B2 block multiplicity.
2. Add state-dict load/export parity against `smp.Unet('tu-efficientnet_b2')`.
3. Bind the portable scorer surface into PR95 MLX training as an advisory
   scorer-loss profile, retaining false-authority until contest CPU/CUDA
   calibration.
4. Use numpy as the interchange layer between MLX local training and PyTorch
   exact-eval/runtime proof.
