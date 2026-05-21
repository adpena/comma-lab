# Codex Findings: MLX scorer-response profile stability gate

## Summary

Added a fail-closed stability checker for local MLX scorer-response profile
outputs. The profiler already exposes throughput and component hashes; this
landing adds a reusable audit layer that compares all rows against a chosen
baseline row and refuses local MLX profile use if batch/device choices change
score components beyond tight tolerances or if false-authority flags appear.

This is a local acceleration guard. It does not create score authority.

## Code changes

- `src/tac/local_acceleration/mlx_profile_stability.py`
  - New reusable profile stability manifest builder.
  - Checks profile false-authority fields.
  - Requires MLX evidence grade.
  - Compares score, PoseNet average, and SegNet average deltas versus a baseline
    row.
  - Treats component SHA differences as warnings by default, or blockers with
    `require_component_sha_match=true`.
- `tools/check_mlx_scorer_response_profile_stability.py`
  - Thin CLI wrapper over the reusable module.
- `src/tac/tests/test_mlx_profile_stability.py`
  - Covers pass, metric drift fail, false-authority fail, evidence-grade fail,
    strict component-SHA mode, and CLI output.

## Empirical audit on FEC6 profile

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/check_mlx_scorer_response_profile_stability.py \
  --profile experiments/results/mlx_scorer_response_profile_fec6_20260521T232241Z_pairs16_4_cpu/profile.json \
  --output experiments/results/mlx_scorer_response_profile_fec6_20260521T232241Z_pairs16_4_cpu/stability.json \
  --run-id mlx_profile_fec6_pairs16_4_cpu_batch12_stability_20260521 \
  --baseline-device cpu \
  --baseline-batch-pairs 1
```

Observed:

- `passed`: `true`
- `verdict`: `PASS_MLX_PROFILE_STABILITY`
- `blockers`: `[]`
- `warnings`: `["profile_row_posenet_sha256_mismatch:index=1"]`
- `score_delta` for `batch_pairs=2` vs baseline `batch_pairs=1`:
  `3.6323648794356345e-7`
- `posenet_avg_delta`: `5.688605142495362e-10`
- `segnet_avg_delta`: `0.0`

Interpretation: the observed batch-size profile is locally stable under the
default metric thresholds, but PoseNet component bytes differ at low magnitude.
This is exactly why MLX remains candidate-generation/training signal and why
CPU/CUDA auth-eval transfer calibration remains mandatory.

## Verification

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m ruff check \
  src/tac/local_acceleration/mlx_profile_stability.py \
  tools/check_mlx_scorer_response_profile_stability.py \
  src/tac/tests/test_mlx_profile_stability.py

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  src/tac/tests/test_mlx_profile_stability.py -q

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  src/tac/tests/test_mlx_profile_stability.py \
  src/tac/tests/test_profile_mlx_scorer_response_cache.py \
  src/tac/tests/test_mlx_scorer_response.py -q
```

Results:

- `ruff`: pass
- profile stability tests: 5 passed
- focused MLX scorer-response/profile suite: 12 passed

## Authority status

Passing this gate means only that a local MLX profile is internally stable
within configured local tolerances. It does not make the profile a score claim,
does not make it rank-or-kill eligible, and does not replace paired
contest-axis CPU/CUDA auth eval.
