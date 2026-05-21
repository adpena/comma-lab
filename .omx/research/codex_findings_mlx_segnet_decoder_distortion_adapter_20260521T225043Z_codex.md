# Codex Findings: MLX SegNet Decoder + Distortion-Component Adapter

Timestamp: 2026-05-21T22:50:43Z
Agent: Codex
Lane: mlx_auth_scorer_port
Evidence grade: local MLX CPU scorer-input parity
Score claim: false
Promotion eligible: false

## Summary

Implemented the next MLX auth-scorer port boundary after SegNet encoder parity:
SMP Unet decoder blocks, full SegNet logits, and a fixed-scorer-input
DistortionNet response wrapper for PoseNet + SegNet.

This is still not a full contest auth-eval replacement. It is a parity-tested
local training/search signal surface on fixed scorer inputs. Byte-closed
auth-eval conformance still requires scorer-input cache identity against a
recovered auth-eval artifact and end-to-end aggregation parity.

## Landed surfaces

- `MLXConv2dReLUAdapter` for SMP decoder `Conv2dReLU`.
- `MLXUnetDecoderBlockAdapter` with explicit target-size nearest interpolation.
- `MLXUnetDecoderAdapter` preserving SMP skip ordering:
  `features[1:][::-1]`, head first, skip list after head.
- `MLXSegmentationHeadAdapter` for raw logits.
- `MLXSegNetAdapter` for encoder -> decoder -> segmentation head logits.
- `MLXDistortionScorerAdapter` for PoseNet + SegNet responses on fixed scorer
  inputs.
- `scorer_distortion_components_numpy(...)` matching upstream component
  formulas:
  - PoseNet: MSE on first 6 pose dimensions.
  - SegNet: argmax disagreement mean over spatial dimensions.

## Empirical verification

Focused scorer-adapter suite:

```text
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest src/tac/tests/test_mlx_scorer_adapters.py -q
36 passed in 6.47s
```

Broader MLX/local-acceleration suite:

```text
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  src/tac/tests/test_local_acceleration.py \
  src/tac/tests/test_mlx_preprocess.py \
  src/tac/tests/test_mlx_cache_audit.py \
  src/tac/tests/test_mlx_scorer_fidelity.py \
  src/tac/tests/test_mlx_scorer_adapters.py \
  src/tac/tests/test_mlx_scorer_state_map.py \
  src/tac/tests/test_mlx_scorer_port_inventory.py -q
92 passed in 13.52s
```

Ruff on touched files:

```text
.venv/bin/python -m ruff check \
  src/tac/local_acceleration/mlx_scorer_adapters.py \
  src/tac/tests/test_mlx_scorer_adapters.py
All checks passed!
```

Full scorer-resolution SegNet logits spot check, random input
`(1, 3, 384, 512)`:

```text
shape=(1, 5, 384, 512)
max_abs=2.4318695068359375e-05
mean_abs=2.3429888642567676e-06
torch_s=0.45137572288513184
mlx_cpu_s=0.3480839729309082
```

## Adversarial notes

- This surface consumes fixed scorer inputs. It deliberately does not claim
  preprocessing parity by itself; that remains under the scorer-input cache hash
  identity contract.
- SegNet decoder nearest interpolation was implemented by explicit target-size
  index selection rather than relying on a scale-factor upsampler, because the
  EfficientNet feature pyramid includes non-integer width growth such as
  `3 -> 5`.
- MLX GPU/MPS remains non-authoritative for exact parity until the observed
  SegNet GPU drift class is resolved. The current hard evidence is MLX CPU.
- The next blocker to full local auth-scorer usability is not another layer
  adapter. It is byte-closed scorer-input cache identity plus full aggregation
  parity against recovered Modal Linux contest-CPU outputs.

## Recommended next action

Build the byte-closed MLX scorer-response harness:

1. Load `segnet_last_rgb.npy`, `posenet_yuv6_pair.npy`, and `pair_indices.npy`
   from a cache whose hash manifest matches recovered auth-eval provenance.
2. Run `MLXDistortionScorerAdapter` over the cache in batches on MLX CPU first.
3. Emit PoseNet/SegNet component arrays, aggregate means, and canonical hashes.
4. Compare to recovered Modal Linux contest-CPU component outputs before using
   this path for local training, candidate search, or optimizer gating.
