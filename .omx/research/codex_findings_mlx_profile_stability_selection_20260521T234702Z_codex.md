# Codex Findings: MLX profile stability selection

## Summary

Extended the MLX scorer-response profile stability manifest with an explicit
row-selection block. The stability gate still fails closed when any profiled row
violates transfer thresholds, but it now also reports which rows remain eligible
and the fastest stable row under the configured CPU-baseline thresholds.

This converts the profiler from a diagnostic table into an operator-usable
local scheduling signal.

## Code changes

- `src/tac/local_acceleration/mlx_profile_stability.py`
  - Normalizes optional throughput fields (`pairs_per_second`,
    `elapsed_seconds`, `wall_seconds`, `repeat_index`, `start_pair`).
  - Adds `selection.policy = fastest_row_with_no_stability_blockers`.
  - Adds `selection.eligible_row_indices`, `selection.rejected_rows`, and
    `selection.recommended_row`.
- `src/tac/tests/test_mlx_profile_stability.py`
  - Verifies fastest stable row selection, strict-SHA fallback selection, and
    fail-closed behavior under global false-authority blockers.

## Empirical audit on existing FEC6 CPU/GPU profile

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/check_mlx_scorer_response_profile_stability.py \
  --profile experiments/results/mlx_scorer_response_profile_fec6_20260521T2340Z_pairs16_4_cpu_gpu/profile.json \
  --output experiments/results/mlx_scorer_response_profile_fec6_20260521T2340Z_pairs16_4_cpu_gpu/stability_cpu_baseline_with_selection.json \
  --run-id mlx_profile_fec6_pairs16_4_cpu_gpu_selection_20260521 \
  --baseline-device cpu \
  --baseline-batch-pairs 1
```

Observed:

- `verdict`: `FAIL_MLX_PROFILE_STABILITY`
- eligible rows: `[0, 1]`
- rejected rows: GPU rows `2` and `3`
- recommended row:
  - index: `1`
  - device: `cpu`
  - batch_pairs: `2`
  - pair_window: `[16, 20]`
  - pairs_per_second: `1.1846059274864154`
  - canonical_score: `0.1781961810476318`

Interpretation: the existing CPU/GPU profile remains globally unsafe for GPU
selection because the GPU rows exceed CPU-baseline transfer thresholds. The
safe local scorer-response choice for this window is `cpu batch_pairs=2`.

## Verification

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m ruff check \
  src/tac/local_acceleration/mlx_profile_stability.py \
  src/tac/tests/test_mlx_profile_stability.py \
  tools/check_mlx_scorer_response_profile_stability.py

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  src/tac/tests/test_mlx_profile_stability.py -q
```

Results:

- `ruff`: pass
- profile-stability tests: 5 passed

## Authority status

This selection block is local MLX scheduling evidence only. It is not an
auth-eval score, not rank-or-kill evidence, and not sufficient for candidate
promotion.
