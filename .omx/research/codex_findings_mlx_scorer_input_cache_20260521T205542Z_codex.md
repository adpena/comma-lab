# Codex Findings: MLX Scorer-Input Cache

UTC: 2026-05-21T20:55:42Z

## Verdict

PROCEED as a local training/surrogate artifact.

This landing does not create a score path. It creates a NumPy cache for the
exact tensors the upstream scorer would feed into SegNet and PoseNet after
inflation, so MLX jobs can train against high-signal scorer inputs locally and
then transfer through the PyTorch/CUDA authority path.

## What Landed

- Added `tac.local_acceleration.mlx_preprocess`.
- Added `tools/build_mlx_scorer_input_cache.py`.
- The cache uses upstream-compatible non-overlapping frame pairs.
- SegNet cache surface: last frame only, resized RGB, `(B, 3, 384, 512)`.
- PoseNet cache surface: both frames, resized RGB -> YUV6, `(B, 12, 192, 256)`.
- Cache manifests carry archive/inflated/raw SHA fields and explicit
  non-authoritative flags: `score_claim=false`, `promotion_eligible=false`,
  `rank_or_kill_eligible=false`, and `ready_for_exact_eval_dispatch=false`.

## Verification

Commands run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  src/tac/tests/test_mlx_preprocess.py \
  src/tac/tests/test_mlx_scorer_fidelity.py -q
```

Result: `9 passed in 1.15s`.

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m compileall -q \
  src/tac/local_acceleration/mlx_preprocess.py \
  tools/build_mlx_scorer_input_cache.py \
  src/tac/tests/test_mlx_preprocess.py
```

Result: pass.

```bash
git diff --check -- \
  src/tac/local_acceleration/mlx_preprocess.py \
  src/tac/tests/test_mlx_preprocess.py \
  tools/build_mlx_scorer_input_cache.py
```

Result: pass.

## Remaining Gaps

- The cache is per-raw-file today; multi-file public-test cache assembly should
  land next.
- The cache does not run the scorer. It only preserves the scorer input tensor
  contract for local MLX/PyTorch training and surrogate calibration.
- Full MLX scorer parity still requires upstream state_dict loading and paired
  CUDA/CPU auth-eval transfer calibration.

## Recommended Next Action

Build a small byte-closed calibration packet from an already inflated PR101/FEC6
raw file, run local MLX/PyTorch surrogate predictions over this cache, then
feed those predictions into `tools/check_mlx_scorer_fidelity.py` against the
matching auth-eval JSON before using local rankings to spend on Modal.
