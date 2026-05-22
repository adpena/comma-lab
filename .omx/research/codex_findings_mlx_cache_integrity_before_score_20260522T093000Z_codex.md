# Codex Findings: MLX Cache Integrity Before Scoring

UTC: 2026-05-22T09:30:00Z
Lane: `lane_mlx_scorer_cache_integrity_verify_before_score_20260522`

## Summary

Closed a cache/auth drift gap in the local MLX scorer path. Before this patch,
`load_scorer_input_cache` loaded `.npy` tensors and validated shapes, but
downstream response payloads copied manifest hashes as identity. A same-shape
array mutation after manifest/audit stamping could therefore score different
tensors while reporting stale cache identity.

## Fix

- Added default-on cache integrity verification in
  `load_scorer_input_cache`.
- Recomputes the same array hash domain used by cache materialization:
  `_array_sha256(dtype_string + json_shape + contiguous_bytes)`.
- Recomputes actual `.npy` artifact bytes and SHA-256.
- Fails before scoring on mismatch with
  `mlx_scorer_input_cache_integrity_failed`.
- Emits `cache_integrity` in MLX scorer-response payloads and embeds the same
  integrity block in each cache identity.

## Regression Test

Added a CLI regression where a valid audited candidate cache is written, then
`segnet_last_rgb.npy` is replaced with same-shape data after manifest stamping.
`tools/run_mlx_scorer_response_cache.py` now exits before scoring and reports:

- `array_sha256_segnet_last_rgb_mismatch`
- `artifact_segnet_last_rgb_sha256_mismatch`

## Verification

- `.venv/bin/ruff check src/tac/local_acceleration/mlx_scorer_response.py tools/run_mlx_scorer_response_cache.py src/tac/tests/test_mlx_scorer_response.py`
- `.venv/bin/python -m pytest -q src/tac/tests/test_mlx_scorer_response.py`
- `.venv/bin/python -m pytest -q src/tac/tests/test_mlx_scorer_response.py src/tac/tests/test_mlx_production_contract.py src/tac/tests/test_mlx_cache_audit.py src/tac/tests/test_materialize_mlx_scorer_cache_from_auth_eval.py`

Result: 64 passed.

## Authority Boundary

This strengthens the local MLX research-signal path only. It does not make MLX
scores authoritative for contest ranking, score claims, promotion, or dispatch
readiness; contest CPU/CUDA auth eval remains required.
