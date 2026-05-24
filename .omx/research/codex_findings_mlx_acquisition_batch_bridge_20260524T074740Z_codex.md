# Codex Findings: MLX Acquisition Batch Bridge

UTC: 2026-05-24T07:47:40Z

## Scope

The operator concern was that local MLX/scorer work was still too row-level and
manual. This pass added a grouped acquisition artifact so strict local MLX
spend-triage selections can enter the inverse-steganalysis action functional as
operation sets.

## Landed

- Added `mlx_acquisition_batch.v1` and
  `mlx_acquisition_operation_set.v1` in
  `src/tac/local_acceleration/mlx_acquisition_batch.py`.
- Added `tools/build_mlx_acquisition_batch.py` to compile strict
  `mlx_effective_spend_triage_candidate_selection.v1` rows into grouped
  operation sets.
- Added `--mlx-acquisition-batch` to
  `tools/build_inverse_steganalysis_action_functional.py`.
- Wired `action_atoms_from_mlx_acquisition_batch(...)` so the action functional
  preserves operation-set provenance, row refs, chosen sequence, pair geometry,
  resource kind, and false-authority blockers.
- Marked lane `lane_codex_mlx_acquisition_batch_autopilot_20260524`
  `impl_complete` with evidence in
  `src/tac/local_acceleration/mlx_acquisition_batch.py`.

## Authority Boundary

The batch is local MLX research signal only. It can rank and group candidate
generation work, but cannot claim score, promote, rank/kill, or dispatch exact
eval. Every batch and operation-set row carries proxy false-authority fields and
requires byte-closed materialization plus exact auth eval before any score claim.

## Remaining Gaps

- Replace the current selection-row grouping with richer MLX training/export
  manifests when native HNeRV/BoostNeRV/NeRV MLX trainers emit canonical
  representation manifests.
- Add learned interaction synthesis across MLX response windows, tensor units,
  archive sections, and packet members.
- Teach the materializer campaign runner to consume `mlx_acquisition_batch.v1`
  directly once grouped local MLX batches should bypass intermediate response
  dataset conversion.

## Verification

```bash
.venv/bin/python -m pytest \
  src/tac/tests/test_byte_shaving_campaign.py \
  src/tac/tests/test_byte_shaving_campaign_queue.py \
  src/tac/tests/test_byte_shaving_materializer_campaign_runner.py \
  src/tac/tests/test_optimizer_candidate_queue.py \
  src/tac/tests/test_inverse_steganalysis_acquisition.py \
  src/tac/tests/test_inverse_steganalysis_action_functional_cli.py \
  src/tac/tests/test_local_training_runtime_profile.py \
  src/tac/tests/test_representation_training_probe_integration.py \
  src/tac/tests/test_local_training_execution_queue.py \
  src/tac/tests/test_mlx_execution_queue.py -q
# 169 passed in 3.02s
```
