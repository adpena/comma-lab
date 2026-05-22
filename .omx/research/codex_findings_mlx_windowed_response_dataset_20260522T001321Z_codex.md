# Codex Findings: MLX Windowed Scorer-Response Dataset

**UTC:** 2026-05-22T00:13:21Z
**Lane:** MLX local scorer-response acceleration / LL surrogate harvest
**Scope:** pair direct MLX response rows with same-window MLX baselines.

## Finding

The direct `mlx_scorer_response.v1` ingestion path is useful, but a single global baseline is the wrong abstraction for local window harvests. MLX response rows are often generated on small scorer-pair windows, and deltas are only meaningful against a baseline response from the same window.

## Fix

- Added `build_windowed_mlx_response_dataset(...)` in `src/tac/optimization/scorer_response_dataset.py`.
- Added `tools/build_mlx_window_response_dataset.py`.
- Window matching key: `start_pair`, `max_pairs`, and `pair_window`.
- Output remains normal `scorer_response_dataset.v1` with all score/promotion/rank/dispatch authority fields false.
- Each normalized row records:
  - `window_baseline_source_path`
  - `window_baseline_key`
  - the existing MLX source schema/evidence/batch/window/component fields

## Real FEC6 local-window artifact

Baseline response:

`experiments/results/mlx_scorer_response_profile_fec6_20260521T2340Z_pairs16_4_cpu_gpu/baseline_response_cpu_b1_pairs16_20.json`

Candidate response:

`experiments/results/mlx_scorer_response_profile_fec6_20260521T2340Z_pairs16_4_cpu_gpu/next_response_cpu_b2_pairs16_20.json`

Generated dataset:

- `experiments/results/mlx_scorer_response_profile_fec6_20260521T2340Z_pairs16_4_cpu_gpu/windowed_scorer_response_dataset_cpu_b2_vs_b1_pairs16_20.json`
- `experiments/results/mlx_scorer_response_profile_fec6_20260521T2340Z_pairs16_4_cpu_gpu/windowed_scorer_response_dataset_cpu_b2_vs_b1_pairs16_20.md`

Generated LL plan:

- `experiments/results/mlx_scorer_response_profile_fec6_20260521T2340Z_pairs16_4_cpu_gpu/ll_scorer_response_next_probe_plan_windowed_mlx_cpu_b2_vs_b1_pairs16_20.json`
- `experiments/results/mlx_scorer_response_profile_fec6_20260521T2340Z_pairs16_4_cpu_gpu/ll_scorer_response_next_probe_plan_windowed_mlx_cpu_b2_vs_b1_pairs16_20.md`

Observed row:

- `row_count`: `1`
- `family_counts.mlx_scorer_response`: `1`
- `window_baseline_key`: `start=16:max=4:window=16-20`
- `delta_vs_baseline_score`: `3.6323648794356345e-07`
- `score_claim`: `false`
- LL top probe: `ll_mlx_cpu_stable_response_harvest`

## Verification

```bash
.venv/bin/python -m ruff check src/tac/optimization/scorer_response_dataset.py src/tac/tests/test_scorer_response_dataset.py tools/build_mlx_window_response_dataset.py tools/plan_ll_scorer_response_next.py
# All checks passed!

.venv/bin/python -m pytest -q src/tac/tests/test_scorer_response_dataset.py
# 28 passed in 0.24s

.venv/bin/python -m pytest -q src/tac/tests/test_scorer_response_dataset.py src/tac/tests/test_plan_ll_scorer_response_next_cli.py src/tac/tests/test_mlx_execution_plan.py src/tac/tests/test_mlx_profile_stability.py
# 40 passed in 0.67s
```

I also ran the broader MLX/dataset suite:

```bash
.venv/bin/python -m pytest -q src/tac/tests/test_scorer_response_dataset.py src/tac/tests/test_plan_ll_scorer_response_next_cli.py src/tac/tests/test_mlx_execution_plan.py src/tac/tests/test_mlx_profile_stability.py src/tac/tests/test_profile_mlx_scorer_response_cache.py src/tac/tests/test_mlx_scorer_response.py src/tac/tests/test_mlx_batch_invariance.py
```

Result: `57 passed`, `1 failed`. The failure was `test_mlx_scorer_response_cli_can_score_deterministic_pair_window`, which asserted zero PoseNet distance for a synthetic deterministic pair window and observed `0.10896018147468567`. This is outside the files changed in this landing and should be handled in the MLX scorer-response implementation/contract lane before treating that broader suite as green.
