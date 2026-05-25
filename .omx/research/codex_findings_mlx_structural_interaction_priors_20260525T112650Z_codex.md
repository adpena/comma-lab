# Codex Findings: MLX Structural Interaction Priors

- UTC: 2026-05-25T11:26:50Z
- Lane: `codex_mlx_structural_interaction_priors_20260525`
- Status: scaffold integrated; planning-only; not score authority

## Finding

`mlx_acquisition_batch.v1` grouped rows and preserved compiler hints, but its
`active_interactions` field stayed empty. That meant grouped MLX search lost
structural overlap signal such as shared pair indices, packet members, tensors,
target kinds, and dynamic sparse source/channel axes before the action
functional could consume it.

## Landed Integration

- Added `mlx_acquisition_operation_set_interaction.v1` structural interaction
  rows to MLX acquisition operation sets.
- Interaction rows are deterministic, score-neutral, and carry false-authority
  fields; they do not change `second_order_interaction_effect` until measured
  queue or exact calibration feedback exists.
- Added tests proving interaction rows are emitted by the MLX batch builder and
  survive into inverse-steganalysis action-cell provenance.

## Verification

- `.venv/bin/python -m ruff check src/tac/local_acceleration/mlx_acquisition_batch.py src/tac/tests/test_mlx_acquisition_batch.py src/tac/tests/test_inverse_steganalysis_acquisition.py --no-cache`
- `PYTHONPATH=. .venv/bin/pytest src/tac/tests/test_mlx_acquisition_batch.py src/tac/tests/test_inverse_steganalysis_acquisition.py::test_mlx_grouped_structural_interactions_survive_to_action_cell -q`

## Remaining Work

- Replace zero-delta structural priors with measured interaction deltas from
  queue-owned materializer observation sweeps.
- Feed those measured deltas into the grouped acquisition waterfill instead of
  ranking rows independently.
