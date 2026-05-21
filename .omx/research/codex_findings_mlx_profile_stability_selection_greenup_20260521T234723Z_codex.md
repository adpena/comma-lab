# Codex Findings: MLX profile-stability row selection greenup

## Summary

Reviewed and preserved the new MLX profile-stability selection surface. The
profile-stability manifest now exposes which rows are eligible after stability
checks and recommends the fastest eligible row by `pairs_per_second`.

This closes a small but real signal-loss gap: the prior stability gate could
say pass/fail, but downstream operators still had to infer which stable row to
use from the raw profile rows.

## Behavior

- Global profile blockers reject all rows.
- Per-row stability blockers reject only that row.
- Rows with no blockers remain eligible.
- `selection.recommended_row` chooses the fastest eligible row by
  `pairs_per_second`.
- If throughput is missing, the first eligible row is selected and the reason
  records that throughput was unavailable.

## Verification

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  src/tac/tests/test_mlx_profile_stability.py \
  src/tac/tests/test_profile_mlx_scorer_response_cache.py -q
```

Result: `10 passed in 0.29s`.

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m ruff check \
  src/tac/local_acceleration/mlx_profile_stability.py \
  src/tac/tests/test_mlx_profile_stability.py
```

Result: `All checks passed!`.

## Authority status

The selected row is local MLX profiling signal only. It is not an auth-eval
score, not a contest-axis result, not promotion evidence, and not rank/kill
evidence.

