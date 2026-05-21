# Codex Findings: MLX Cache Large Tensor Guard

UTC: 2026-05-21T21:41:00Z

## Verdict

PROCEED. The MLX scorer-input cache builder now refuses accidental eager
multi-GB tensor cache writes unless the operator explicitly acknowledges the
large local artifact.

## What Landed

- `tools/build_mlx_scorer_input_cache.py` now estimates requested pair count
  before the eager full-cache path.
- Eager full `.npy` cache writes fail closed when requested pairs exceed
  `--large-cache-pair-threshold` (default `64`) unless
  `--allow-large-tensor-cache` is provided.
- `--hash-only` remains the intended default path for full auth-surface
  identity artifacts.
- Small local smoke caches remain ergonomic under the default threshold.

## Why This Matters

The full FEC6 scorer-input cache writes roughly multi-GB SegNet/PoseNet tensor
artifacts plus transient raw and float surfaces. That is useful when
intentional, but dangerous as a CLI default. The guard keeps local MLX
development fast while making expensive full-cache materialization explicit.

## Verification

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  src/tac/tests/test_mlx_preprocess.py -q
```

Result: `8 passed in 2.79s`.

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m compileall -q \
  tools/build_mlx_scorer_input_cache.py \
  src/tac/tests/test_mlx_preprocess.py
```

Result: pass.

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/review_tracker.py \
  policy-check tools/build_mlx_scorer_input_cache.py src/tac/tests/test_mlx_preprocess.py
```

Result: `0 violations`.

```bash
git diff --check -- \
  tools/build_mlx_scorer_input_cache.py \
  src/tac/tests/test_mlx_preprocess.py
```

Result: pass.
