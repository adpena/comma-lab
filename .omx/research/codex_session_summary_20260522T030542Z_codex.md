# Codex Session Summary

UTC: 2026-05-22T03:05:42Z

## Landed

- `313bd0741` — Add MLX singleton response window splitter.
- `edf723203` — Record MLX singleton full-video response harvest.
- `43d0e9ed8` — Add scorer response validation gate.
- `4eef46390` — Guard scorer response validation axes.

## Concrete Artifacts

- Full-video singleton CPU MLX response corpus: `300` one-pair windows for FEC6 PR101.
- Matched reference-vs-reference singleton baseline windows: `300`.
- Scorer-response validation gate: `scorer_response_dataset_validation_gate.v1`.
- Dataset merge utility: `tools/merge_scorer_response_datasets.py`.
- Merged corpus: `321` rows (`300` MLX + `21` decoder-q advisory).

## Current State

The MLX corpus cleared row-count and fold-coverage requirements. The merged MLX+decoder-q corpus cleared family/fold coverage as well, but validation remains blocked:

- `mixed_axis_targets:['[macOS-CPU advisory decoder-q]', '[macOS-MLX research-signal]']`
- `no_prediction_fields_present`

This is intentional. The axis guard prevents a spurious held-out correlation pass caused by mixing MLX per-window targets with macOS full-video advisory targets.

## Verification

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  src/tac/tests/test_scorer_response_dataset.py \
  src/tac/tests/test_mlx_response_windows.py \
  src/tac/tests/test_mlx_scorer_response.py \
  src/tac/tests/test_mlx_score_calibration.py
```

Result: `59 passed`.

## Partner WIP Observed

Unstaged/untracked MLX calibration work is present and was left untouched:

- `src/tac/local_acceleration/mlx_execution_plan.py`
- `src/tac/local_acceleration/mlx_score_calibration.py`
- `src/tac/tests/test_mlx_score_calibration.py`
- `tools/calibrate_mlx_scorer_response_scores.py`

Focused calibration tests passed: `4 passed`.

## Next Action

Build a same-axis second family for `[macOS-MLX research-signal]`, ideally by producing singleton MLX scorer-response rows for a byte-neutral or decoder-q-style candidate cache. Then attach explicit cross-fold prediction fields and rerun `tools/validate_scorer_response_dataset.py` without mixed-axis override.
