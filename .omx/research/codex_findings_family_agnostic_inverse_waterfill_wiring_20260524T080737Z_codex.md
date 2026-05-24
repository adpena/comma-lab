# Codex Findings: Family-Agnostic Inverse Waterfill Wiring

UTC: 2026-05-24T08:07:37Z

## Context

Operator reiterated that the inverse-scorer / water-fill system must work for
HNeRV variants, BoostNeRV bolt-ons, broader NeRV-family candidates, and non-NeRV
candidates. The failure mode to avoid is collapsing grouped acquisition signal
back into one generic inverse-scorer-cell leaf operation.

## Landed Wiring

- `mlx_acquisition_batch.v1` operation sets now carry
  `mlx_acquisition_representation_contract.v1` rows.
- Family classes are preserved as `hnerv_variant`, `boostnerv_bolton`,
  `nerv_family`, `non_nerv`, or `unknown`, with source family tokens and
  receiver/materializer contract kinds.
- The inverse action functional provenance now preserves representation
  contracts, receiver contract kinds, materializer contract kinds, and
  family-class metadata.
- `build_signal_surface_from_inverse_action_functional()` now rehydrates
  selected water-bucket cells from byte-shaving or MLX operation-set provenance.
  Original archive-section, tensor, packet-member, and scorer-row operations no
  longer collapse into a generic `scorer_inverse_surface_cell` when provenance is
  available.
- `tools/run_byte_shaving_materializer_campaign.py` now forwards direct
  `--byte-shaving-signal-surface` and `--byte-shaving-campaign-plan` inputs into
  the one-command action-functional/materializer campaign path.
- The materializer registry now has fail-closed family-agnostic contracts for
  additional planner families: `section_header_elide`, `section_reorder`,
  `prune_tensor`, `zip_header_elide`, `member_reorder`, and `member_merge`.
- Adversarial review found two identity bugs before landing:
  explicit inferred `non_nerv` rows could be swallowed by the generic `nerv`
  substring rule, and `chosen_operation_sequence_sha256` omitted family /
  materializer metadata. Both are now guarded by focused regression tests.

## Authority Boundary

All new rows remain planning-only and proxy false-authority. No local MLX row,
family contract, water-bucket cell, or rehydrated operation can claim score,
promotion, rank/kill, or exact-dispatch authority without byte-closed archive
materialization, runtime-consumption proof, and exact auth evaluation.

## Verification

- `.venv/bin/ruff check tools/build_mlx_acquisition_batch.py tools/build_inverse_steganalysis_action_functional.py tools/run_byte_shaving_materializer_campaign.py src/comma_lab/scheduler/byte_shaving_materializer_registry.py src/tac/local_acceleration/mlx_acquisition_batch.py src/tac/local_acceleration/__init__.py src/tac/optimization/byte_shaving_campaign.py src/tac/optimization/inverse_steganalysis_acquisition.py src/tac/tests/test_byte_shaving_campaign.py src/tac/tests/test_byte_shaving_campaign_queue.py src/tac/tests/test_inverse_steganalysis_action_functional_cli.py src/tac/tests/test_byte_shaving_materializer_campaign_runner.py`
- `.venv/bin/python -m pytest src/tac/tests/test_byte_shaving_campaign.py src/tac/tests/test_byte_shaving_campaign_queue.py src/tac/tests/test_byte_shaving_materializer_campaign_runner.py src/tac/tests/test_optimizer_candidate_queue.py src/tac/tests/test_inverse_steganalysis_acquisition.py src/tac/tests/test_inverse_steganalysis_action_functional_cli.py src/tac/tests/test_local_training_runtime_profile.py src/tac/tests/test_representation_training_probe_integration.py src/tac/tests/test_local_training_execution_queue.py src/tac/tests/test_mlx_execution_queue.py -q`
- `.venv/bin/python tools/lane_maturity.py validate`

Additional pre-landing audit greenup after the regression fixes:

- `.venv/bin/python -m pytest -q src/tac/tests/test_byte_shaving_campaign.py src/tac/tests/test_byte_shaving_campaign_queue.py src/tac/tests/test_byte_shaving_materializer_campaign_runner.py src/tac/tests/test_inverse_steganalysis_action_functional_cli.py`
- `.venv/bin/ruff check tools/build_mlx_acquisition_batch.py tools/build_inverse_steganalysis_action_functional.py tools/run_byte_shaving_materializer_campaign.py src/comma_lab/scheduler/byte_shaving_materializer_registry.py src/tac/local_acceleration/mlx_acquisition_batch.py src/tac/local_acceleration/__init__.py src/tac/optimization/byte_shaving_campaign.py src/tac/optimization/inverse_steganalysis_acquisition.py src/tac/tests/test_byte_shaving_campaign.py src/tac/tests/test_byte_shaving_campaign_queue.py src/tac/tests/test_inverse_steganalysis_action_functional_cli.py src/tac/tests/test_byte_shaving_materializer_campaign_runner.py`

Result: 176 broad focused tests passed after audit fixes, 86 tighter
regression tests passed after audit fixes; ruff passed; lane registry
validated.

## Remaining Gaps

- The new family materializer contracts are intentionally non-executable until
  receiver proof and context schemas exist.
- Local MLX/NumPy representation training manifests still need to feed this
  acquisition batch schema directly, not only scorer-response triage selections.
- Work-queue command builders still need concrete archive-section, tensor, and
  packet-member materializer execution branches once executable materializers
  exist.
- Exact auth remains the promotion boundary; MLX remains advisory research
  signal only.
