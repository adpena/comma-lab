# Codex Findings: Inverse-Steganalysis Water-Bucket Portfolio Gap

Timestamp UTC: 2026-05-24T15:52:03Z

## Findings

1. The operator concern is correct: a bare inverse-action cell is still a leaf
   object. It is useful as a scorer-coordinate signal, but it is not itself a
   byte-closed candidate, archive operation, runtime patch, or final-rate
   actuator.

2. The default water-bucket bridge now refuses to turn a bare selected cell into
   an IAS1 descriptor candidate. Bare cells emit
   `compile_inverse_steganalysis_operation_set` and the non-executable
   `inverse_steganalysis_operation_set_compiler_required` materializer gap.

3. Selected cells with source operation-set provenance remain the executable
   path: they are rehydrated into family-agnostic materializer units such as
   archive-section, tensor, and packet-member operations, then flow into the
   materializer queue and exact-readiness blockers.

4. The bridge emits
   `inverse_steganalysis_water_bucket_materialization_portfolio.v1` so each
   selected water-fill cell carries an explicit actuation mode:
   `source_provenance_operation_set`,
   `high_level_operation_compiler_required`, or
   `leaf_cell_candidate_explicit_opt_in`.

5. The old `materialize_inverse_scorer_cell_candidate` fallback still exists
   only as an explicit diagnostic opt-in (`--allow-leaf-inverse-cell-candidates`)
   and is tagged as a probe rather than portfolio-level actuation.

6. The new high-level compiler gap is fail-closed in the materializer registry:
   it is non-executable, false-authority clean, and blocked until a real
   candidate-family operation compiler, archive grammar, receiver contract, and
   runtime-consumption proof exist.

## Verification

- `.venv/bin/python -m pytest src/tac/tests/test_byte_shaving_campaign.py src/tac/tests/test_byte_shaving_signal_surface_builder.py src/tac/tests/test_byte_shaving_campaign_queue.py src/tac/tests/test_inverse_steganalysis_acquisition.py src/tac/tests/test_inverse_steganalysis_action_functional_cli.py src/tac/tests/test_byte_shaving_materializer_campaign_runner.py src/tac/tests/test_mlx_execution_queue.py -q` (144 passed)
- `.venv/bin/python -m pytest src/tac/tests/test_byte_shaving_campaign_queue.py src/tac/tests/test_inverse_steganalysis_acquisition.py src/tac/tests/test_inverse_steganalysis_action_functional_cli.py src/tac/tests/test_byte_shaving_materializer_campaign_runner.py src/tac/tests/test_mlx_execution_queue.py -q` (117 passed)
- `.venv/bin/ruff check src/tac/optimization/byte_shaving_campaign.py src/comma_lab/scheduler/byte_shaving_materializer_registry.py tools/plan_byte_shaving_campaign.py src/tac/tests/test_byte_shaving_campaign.py src/tac/tests/test_byte_shaving_campaign_queue.py`
- `git diff --check`

## Remaining Production Gap

The next score-moving implementation is not another cell materializer. It is a
portfolio-level operation-set compiler that maps inverse-action cells onto
concrete candidate-family transforms with byte/rate accounting, archive grammar
constraints, runtime-consumption proof, and exact-auth handoff. That compiler
should target HNeRV variants, HNeRV bolt-ons, broader NeRV-family packets, and
non-NeRV archives through the existing family-agnostic materializer surface.
