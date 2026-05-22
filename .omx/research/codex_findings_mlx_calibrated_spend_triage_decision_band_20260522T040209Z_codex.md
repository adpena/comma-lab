# Codex Findings: MLX Calibrated Spend-Triage Decision Band

UTC: 2026-05-22T04:02:09Z

## Verdict

PROCEED_WITH_CALIBRATED_TRIAGE. The MLX scorer-response surface now has a conservative decision band for local spend triage. This converts the public-frontier calibration into an explicit rule: MLX pairwise ordering can guide local training and exact-eval spend only when the MLX score gap is wider than the measured calibration uncertainty times a safety factor.

This is not score authority.

## Online Research

Primary-source findings used for the engineering rule:

- NVIDIA CUDA best-practices documentation states that floating-point operation order matters, parallelization can change that order, CUDA-host results should be compared within tolerance, and FMA can differ slightly from separate multiply/add sequences.
- NVIDIA cuBLAS documentation states that bitwise reproducibility is conditional and can be affected by concurrent streams, workspace selection, fixed-point emulation, and atomics; deterministic behavior may require user workspace or `CUBLAS_WORKSPACE_CONFIG`.
- PyTorch deterministic-algorithm documentation states that deterministic mode is not sufficient by itself for full reproducibility and that some CUDA operations require `CUBLAS_WORKSPACE_CONFIG`.
- MLX documentation confirms the NumPy-like, CPU/GPU, unified-memory, lazy-evaluation model and documents compile/lazy evaluation boundaries. That supports MLX as a local training substrate, but it also makes explicit evaluation/canonicalization boundaries important.

Conclusion: the right engineering target is calibrated portability and bounded local decision use, not claiming device-independent score identity.

## Code Changes

- Added `decision_policy` to `mlx_score_calibration.v1`.
- Added configurable `decision_safety_factor` with default `5.0`.
- Added per-pair `mlx_score_gap_abs`, exact-axis gap fields, and `mlx_spend_triage_decision_certified`.
- Added summary counts for certified and uncertain pairwise MLX spend-triage decisions.
- Added `--decision-safety-factor` to `tools/calibrate_mlx_scorer_response_scores.py`.
- Added tests for certified wide gaps and uncertain close gaps.

Authority fields remain explicit false:

- `score_claim=false`
- `promotion_eligible=false`
- `rank_or_kill_eligible=false`
- `ready_for_exact_eval_dispatch=false`

## Frontier Calibration Re-run

Input:

`experiments/results/mlx_public_frontier_calibration_20260522T023053Z/calibration_rows.json`

Output:

`experiments/results/mlx_public_frontier_calibration_20260522T023053Z/calibration_manifest_decision_band.json`

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  tools/calibrate_mlx_scorer_response_scores.py \
  --input experiments/results/mlx_public_frontier_calibration_20260522T023053Z/calibration_rows.json \
  --output experiments/results/mlx_public_frontier_calibration_20260522T023053Z/calibration_manifest_decision_band.json \
  --repo-root . \
  --decision-safety-factor 5.0
```

Result:

- Calibration uncertainty score: `1.7603544242461577e-05`
- Safety factor: `5.0`
- Recommended minimum MLX gap for spend triage: `8.801772121230789e-05`
- PR110/101/103/102 pairwise comparisons certified for spend triage: `6 / 6`
- PR110/101/103/102 uncertain pairwise comparisons: `0 / 6`
- MLX-vs-CPU rank inversions: `0 / 6`
- MLX-vs-CUDA rank inversions: `0 / 6`

Closest measured pair:

- PR103 vs PR102 MLX gap: `0.0005123465712875974`
- Required spend-triage gap: `0.00008801772121230789`
- Verdict: certified for local spend-triage ordering, still not rank/kill authority.

## Verification

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m ruff check \
  src/tac/local_acceleration/mlx_score_calibration.py \
  tools/calibrate_mlx_scorer_response_scores.py \
  src/tac/tests/test_mlx_score_calibration.py
```

Result: `All checks passed!`

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest src/tac/tests/test_mlx_score_calibration.py
```

Result: `5 passed`.

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  src/tac/tests/test_mlx_score_calibration.py \
  src/tac/tests/test_mlx_execution_plan.py \
  src/tac/tests/test_mlx_scorer_response.py \
  src/tac/tests/test_mlx_profile_stability.py \
  src/tac/tests/test_profile_mlx_scorer_response_cache.py \
  src/tac/tests/test_mlx_production_contract.py
```

Result: `43 passed`.

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest $(rg --files src/tac/tests | rg 'mlx')
```

Result: `150 passed`.

```bash
git check-ignore -v experiments/results/mlx_public_frontier_calibration_20260522T023053Z/calibration_manifest_decision_band.json
```

Result: covered by `.gitignore:55:experiments/results/*`.

## Next Action

Wire this decision-band manifest into any local MLX response planner that proposes paid exact-eval spend. The planner should refuse MLX-based ordering when pairwise gaps are below `recommended_min_mlx_gap_for_spend_triage`, and it should continue to require exact CPU/CUDA auth eval for claims, promotion, and close decisions.
