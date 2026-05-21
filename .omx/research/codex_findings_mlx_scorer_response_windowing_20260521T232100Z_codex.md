# Codex Findings: MLX scorer-response deterministic pair-window surface

## Summary

Landed a deterministic pair-window mode for the local MLX scorer-response cache
runner. This is a non-authoritative acceleration surface: it lets local training,
profiling, and candidate triage consume exact scorer-input cache slices without
rebuilding caches or implying any contest score claim.

## Code changes

- `src/tac/local_acceleration/mlx_scorer_response.py`
  - Added `start_pair` and `max_pairs` arguments to
    `build_mlx_scorer_response_payload(...)`.
  - Validates `start_pair >= 0`, `max_pairs >= 1` when set, and rejects
    windows outside the cache pair count before scoring.
  - Records `total_cache_pairs`, `start_pair`, `max_pairs`, and `pair_window`
    in the output payload.
- `tools/run_mlx_scorer_response_cache.py`
  - Added CLI flags `--start-pair` and `--max-pairs`.
- `src/tac/tests/test_mlx_scorer_response.py`
  - Added CLI regression for scoring a deterministic pair window.
  - Added validation regression for invalid `max_pairs`.

## Empirical smoke

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/run_mlx_scorer_response_cache.py \
  --reference-cache-dir experiments/results/mlx_scorer_input_cache_reference_video_20260521T2304Z_full600 \
  --candidate-cache-dir experiments/results/mlx_scorer_input_cache_fec6_pr101_20260521T210100Z_full300pairs \
  --archive-size-bytes 178517 \
  --output experiments/results/mlx_scorer_response_fec6_window_smoke_20260521T232018Z_pairs16_4/mlx_response.json \
  --components-dir experiments/results/mlx_scorer_response_fec6_window_smoke_20260521T232018Z_pairs16_4/components \
  --repo-root . \
  --device cpu \
  --batch-pairs 2 \
  --start-pair 16 \
  --max-pairs 4 \
  --progress-every 1
```

Observed payload:

- `n_samples`: 4
- `total_cache_pairs`: 600
- `pair_window`: `[16, 20]`
- `elapsed_seconds`: 4.256556034088135
- `avg_posenet_dist`: 0.00000613186800535459
- `avg_segnet_dist`: 0.0005149841381353326
- `score_claim`: `false`
- `promotion_eligible`: `false`

Component artifacts:

- `posenet_distortion.npy` SHA-256:
  `be8e72c3307af6b58d85d8da16aa33e68e021850f390624988b916d3833c82b0`
- `segnet_distortion.npy` SHA-256:
  `fd2d17a65b141c12fb8515a9644a949466c9aa1f99592b50967a3b3b79663123`

## Verification

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m ruff check \
  src/tac/local_acceleration/mlx_scorer_response.py \
  tools/run_mlx_scorer_response_cache.py \
  src/tac/tests/test_mlx_scorer_response.py

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  src/tac/tests/test_mlx_scorer_response.py -q

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  src/tac/tests/test_mlx_preprocess.py \
  src/tac/tests/test_mlx_scorer_fidelity.py \
  src/tac/tests/test_mlx_cache_audit.py \
  src/tac/tests/test_mlx_to_pytorch_export.py -q
```

Results:

- `ruff`: pass
- `test_mlx_scorer_response.py`: 4 passed
- MLX custody/fidelity/export suite: 29 passed

## Authority status

This surface remains candidate-generation only. It is useful for profiling,
local scorer-response distillation, per-pair curriculum, and fast candidate
triage. It does not replace byte-closed CPU/CUDA auth eval and must pass
`tools/check_mlx_scorer_fidelity.py` against a matching auth-axis payload before
it is used as a training/ranking signal.
