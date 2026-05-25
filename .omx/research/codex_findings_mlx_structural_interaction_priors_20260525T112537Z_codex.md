# Codex Findings: MLX Structural Interaction Priors

- UTC: 2026-05-25T11:25:37Z
- Lane: `codex_mlx_structural_interaction_priors_20260525`
- Status: scaffold integrated; planning-only; not score authority

## Finding

Grouped MLX acquisition batches could carry operation-set compiler hints, but
the grouped candidate lost the second-order structure the optimizer needs to
avoid leaf-wise local minima. Operations that share pair indices, tensors,
packet members, archive sections, target kinds, or dynamic sparse gate channels
were still represented as independent rows unless later code rediscovered the
overlap.

## Landed Integration

- Added `mlx_acquisition_operation_set_interaction.v1` rows under each grouped
  MLX operation set.
- Derived deterministic structural priors for shared target kind, operation
  family, packet member, tensor, archive section, pair index, dynamic sparse
  source, and dynamic sparse channel.
- Kept every interaction as an unmeasured zero-delta prior:
  `interaction_delta_score=0.0`, `interaction_extra_saved_bytes=0`, and
  `interaction_model=structural_overlap_unmeasured_zero_delta_prior`.
- Preserved proxy false-authority semantics and dispatch blockers so interaction
  priors can inform acquisition but cannot claim score, promote, or dispatch.
- Proved interaction rows survive from MLX acquisition batch into inverse action
  cell provenance without becoming measured second-order score.

## Verification

- `.venv/bin/python -m ruff check src/tac/local_acceleration/mlx_acquisition_batch.py src/tac/tests/test_inverse_steganalysis_acquisition.py src/tac/tests/test_mlx_acquisition_batch.py`
- `.venv/bin/python -m pytest src/tac/tests/test_mlx_acquisition_batch.py src/tac/tests/test_inverse_steganalysis_acquisition.py -q`

## Remaining Work

- Feed realized queue/materializer observation deltas back into these interaction
  rows so zero-delta priors become calibrated synergy or antagonism terms.
- Connect calibrated interaction terms to the dynamic sparse observation-feedback
  compiler and the water-fill allocator.
- Keep all MLX-derived interaction signal on `[macOS-MLX research-signal]` or
  planning axes until byte-closed candidates pass exact contest auth eval.
