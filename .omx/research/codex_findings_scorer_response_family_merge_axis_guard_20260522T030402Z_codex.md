# Codex Findings: Scorer-Response Family Merge + Axis Guard

UTC: 2026-05-22T03:04:02Z

## Verdict

PROCEED_WITH_GUARDS. The scorer-response surface can now merge multiple non-authoritative datasets, and the validation gate now blocks mixed score-axis targets so a trivial family-label predictor cannot masquerade as held-out rigor.

## Code Changes

- Added `merge_scorer_response_datasets`.
- Added `tools/merge_scorer_response_datasets.py`.
- Extended `scorer_response_dataset_validation_gate.v1` with `axis_counts` and `require_single_axis`.
- Added default blocker for mixed or missing target axes.
- Added `--allow-mixed-axis-targets` to the validation CLI for explicit research-only override.

## Empirical Merge

Artifact root:

`experiments/results/scorer_response_family_merge_20260522T0305Z/`

Inputs:

- MLX singleton full-video window dataset: `300` rows, family `mlx_scorer_response`, axis `[macOS-MLX research-signal]`.
- PR110 decoder-q advisory datasets: `21` rows, family `decoder_q`, axis `[macOS-CPU advisory decoder-q]`.

Merged dataset:

`mlx300_decoderq21_merged_dataset.json`

Summary:

- Rows: `321`
- Families: `{"decoder_q": 21, "mlx_scorer_response": 300}`
- Best delta: `0.000388914325026829`, family `decoder_q`
- Worst delta: `0.11924633770300272`, family `mlx_scorer_response`
- Score claim: `false`

## Validation Result

Validation gate:

`mlx300_decoderq21_validation_gate_v2_axis_guard.json`

Status: `blocked`

Blockers:

- `mixed_axis_targets:['[macOS-CPU advisory decoder-q]', '[macOS-MLX research-signal]']`
- `no_prediction_fields_present`

What this clears:

- Row count: `321 >= 50`
- Family count: `2 >= 2`
- Both families contain folds `0..4`
- Global fold coverage is complete

What remains:

- Same-axis target corpus is still missing.
- No explicit prediction field exists for held-out correlation.

## Interpretation

The previous one-family blocker is solved structurally, but the result is not yet a valid held-out predictor corpus because MLX per-window rows and macOS full-video decoder-q advisory rows are different target scales. The axis guard prevents a misleading correlation pass driven purely by family/axis separation.

## Verification

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  src/tac/tests/test_scorer_response_dataset.py \
  src/tac/tests/test_mlx_response_windows.py \
  src/tac/tests/test_mlx_scorer_response.py
```

Result: `55 passed`.

## Next Action

Build a same-axis second family. The cleanest path is to generate MLX singleton response rows for a byte-neutral or decoder-q-style candidate cache so both families live on `[macOS-MLX research-signal]`, then attach explicit cross-fold predictions and re-run the validation gate.
