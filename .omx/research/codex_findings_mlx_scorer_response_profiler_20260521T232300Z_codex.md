# Codex Findings: MLX scorer-response profiler

## Summary

Added `tools/profile_mlx_scorer_response_cache.py`, a machine-readable
throughput profiler for fixed MLX scorer-input cache windows. The tool sweeps
`batch_pairs` and `device` values over a deterministic pair window and preserves
the same non-authoritative markers as the scorer-response payload.

This is a local acceleration artifact, not a score artifact. Its output is for
choosing local training/profile settings and surfacing backend numeric drift
before exact CPU/CUDA auth eval.

## Code changes

- New `tools/profile_mlx_scorer_response_cache.py`
  - Inputs: reference cache, candidate cache, archive byte count, device list,
    batch-pair list, pair window, repeat count.
  - Output schema: `mlx_scorer_response_profile.v1`.
  - Emits rows with elapsed seconds, pairs/s, component averages, component
    SHA-256s, and best-throughput row.
  - Hard-codes `score_claim=false`, `promotion_eligible=false`,
    `rank_or_kill_eligible=false`, and `ready_for_exact_eval_dispatch=false`.
- New `src/tac/tests/test_profile_mlx_scorer_response_cache.py`
  - CSV parser coverage.
  - False-authority marker coverage.
  - Best-row selection coverage under a monkeypatched scorer-response builder.

## Empirical profile

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/profile_mlx_scorer_response_cache.py \
  --reference-cache-dir experiments/results/mlx_scorer_input_cache_reference_video_20260521T2304Z_full600 \
  --candidate-cache-dir experiments/results/mlx_scorer_input_cache_fec6_pr101_20260521T210100Z_full300pairs \
  --archive-size-bytes 178517 \
  --output experiments/results/mlx_scorer_response_profile_fec6_20260521T232241Z_pairs16_4_cpu/profile.json \
  --repo-root . \
  --devices cpu \
  --batch-pairs 1,2 \
  --start-pair 16 \
  --max-pairs 4 \
  --repeat 1
```

Observed rows:

| device | batch_pairs | pair_window | elapsed_seconds | pairs_per_second | avg_posenet_dist | avg_segnet_dist |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| cpu | 1 | `[16, 20]` | 4.236191987991333 | 0.94424426733706 | 0.000006131299144840341 | 0.0005149841381353326 |
| cpu | 2 | `[16, 20]` | 3.579463243484497 | 1.1174859826486507 | 0.00000613186800535459 | 0.0005149841381353326 |

Best row: `device=cpu`, `batch_pairs=2`, `pairs_per_second=1.1174859826486507`.

Notable signal: SegNet component SHA was identical across the two batch sizes.
PoseNet component SHA changed at low numeric magnitude, moving the window score
by about `3.6e-7`. This reinforces that the MLX path is suitable for local
ranking/training only after transfer calibration, and that final claims must
remain byte-closed CPU/CUDA auth eval.

## Verification

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m ruff check \
  tools/profile_mlx_scorer_response_cache.py \
  src/tac/tests/test_profile_mlx_scorer_response_cache.py

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  src/tac/tests/test_profile_mlx_scorer_response_cache.py -q
```

Results:

- `ruff`: pass
- profiler tests: 3 passed
