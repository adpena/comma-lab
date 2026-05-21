# Codex Findings: MLX Auth-Scorer Training-Signal Fidelity

UTC: 2026-05-21T20:50:55Z

## Verdict

PROCEED_WITH_GUARDRAILS for local MLX substrate training and advisory ranking.

MLX is useful for cheap local training on the M5 Max, but it remains a
non-authoritative research signal. Promotion, rank/kill, leaderboard, and
public score claims still require byte-closed contest auth eval on the
canonical CPU/CUDA axes.

## What Landed

- Added `tac.local_acceleration.mlx_scorer_fidelity`, a fail-closed transfer
  manifest that compares MLX scorer/surrogate payloads against byte-closed
  auth-eval JSON by archive SHA, inflated-output aggregate SHA, sample count,
  score, SegNet contribution, PoseNet contribution, and rate contribution.
- Added `tools/check_mlx_scorer_fidelity.py` so the gate is reachable from an
  operator flow instead of hidden library code.
- Added ARCH-4a SegNet portable primitives in
  `tac.portable_primitives.nn_segnet`: squeeze-excite, MBConv scaffold, SMP-
  shaped UNet decoder blocks, decoder, and segmentation head.
- Exported the SegNet primitives through `tac.portable_primitives`.
- Corrected the false-authority wording in `tac.evaluate`: it is now labeled
  as a legacy TAC post-filter proxy evaluator, not an official submission
  evaluator.

## Subagent Findings Integrated

- No complete MLX-native `upstream/evaluate.py` port exists on current `main`.
- The safest near-term architecture is MLX preprocessing/training/surrogate
  signal plus PyTorch/CUDA auth-eval handoff, not a direct MLX score claim.
- The largest missing scorer half is SegNet EfficientNet-B2-UNet. This landing
  starts that port at the primitive/decoder scaffold layer while deferring full
  EfficientNet-B2 encoder and state_dict parity.
- NumPy is the correct portability spine for weights, cached tensors, and
  fidelity manifests; it is not the hot authority scorer engine.

## Verification

Commands run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  src/tac/tests/test_mlx_scorer_fidelity.py \
  src/tac/portable_primitives/tests/test_portable_primitives_segnet.py \
  src/tac/portable_primitives/tests -q
```

Result: `86 passed in 1.86s`.

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m compileall -q \
  src/tac/local_acceleration/mlx_scorer_fidelity.py \
  src/tac/portable_primitives/nn_segnet.py \
  src/tac/evaluate.py \
  tools/check_mlx_scorer_fidelity.py
```

Result: pass.

```bash
git diff --check -- \
  src/tac/local_acceleration/mlx_scorer_fidelity.py \
  src/tac/tests/test_mlx_scorer_fidelity.py \
  src/tac/portable_primitives/nn_segnet.py \
  src/tac/portable_primitives/tests/test_portable_primitives_segnet.py \
  src/tac/portable_primitives/__init__.py \
  src/tac/evaluate.py \
  tools/check_mlx_scorer_fidelity.py
```

Result: pass.

## Remaining Gaps

- No full MLX-native auth scorer parity claim exists.
- No MLX load path for upstream `posenet.safetensors` / `segnet.safetensors`
  exists yet.
- FastViT/PoseNet remains scaffold-grade for exact timm parity.
- SegNet EfficientNet-B2 encoder is not complete; ARCH-4a only covers portable
  SE/MBConv/UNet decoder primitives.
- No real archive MLX-vs-auth-eval calibration row has landed yet, so the lane
  is L1 scaffold rather than L2 integration.

## Recommended Next Action

Build the MLX preprocessing/cache payload next: archive SHA, inflated aggregate
SHA, non-overlapping pair indices, SegNet last-frame tensors, PoseNet YUV6 pair
tensors, and NumPy hashes. Then run `tools/check_mlx_scorer_fidelity.py`
against a small byte-closed auth-eval calibration packet before using MLX
surrogate rankings to select expensive exact-eval candidates.
