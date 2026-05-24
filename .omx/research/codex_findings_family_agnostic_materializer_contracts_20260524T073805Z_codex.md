# Codex Findings: Family-Agnostic Materializer Contracts

UTC: 2026-05-24T07:38:05Z

## Scope

This pass addressed the scheduler/materializer gap between the family-agnostic
inverse-steganalysis planner and concrete queue actuation. The target was not
to make HNeRV, BoostNeRV, NeRV-family, or non-NeRV transforms executable in one
step; it was to stop collapsing those operations into vague unregistered
backlog rows.

## Landed

- Added explicit fail-closed materializer contracts for:
  - `archive_section` + `section_entropy_recode`
  - `archive_section` + `section_proceduralize`
  - `tensor` + `quantize_tensor`
  - `tensor` + `factorize_tensor`
  - `tensor` + `shared_codebook_tensor`
  - `packet_member` + `member_recompress`
- Exported the new registry constants through `comma_lab.scheduler`.
- Exported local MLX scorer-response queue and local representation-training
  queue builders through `comma_lab.scheduler` so normal scheduler imports can
  discover them.
- Kept contract-only adapters free of fake implementation-module/function
  pointers; the registry now distinguishes "contract registered" from
  "implementation exists" without phantom code paths.
- Registered lane
  `family_agnostic_materializer_contracts_20260524` and marked
  `impl_complete` with evidence in
  `src/comma_lab/scheduler/byte_shaving_materializer_registry.py`.

## Authority Boundary

All new family-agnostic adapters are non-executable. They emit
`materializer_not_executable:*` and require runtime-consumption proof plus
family-specific archive/tensor/member manifests before candidate archive
materialization. They do not create score authority, promotion authority,
rank/kill authority, or exact-dispatch readiness.

## Remaining Gaps

- Implement actual materializers behind the registered receiver contracts:
  section recode/proceduralize, tensor quantize/factorize/shared-codebook, and
  packet-member recompress.
- Wire HNeRV/BoostNeRV/NeRV/non-NeRV parser manifests into those context fields
  instead of relying on manually shaped operation params.
- Add a canonical MLX acquisition-batch artifact that preserves grouped
  operation sets from local scorer/training evidence into inverse action
  functionals.
- Add exact-readiness refusal tests for each new materializer once executable
  implementations exist.

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

.venv/bin/ruff check \
  src/comma_lab/scheduler/__init__.py \
  src/comma_lab/scheduler/byte_shaving_materializer_registry.py \
  src/comma_lab/scheduler/local_training_queue.py \
  src/comma_lab/scheduler/mlx_execution_queue.py \
  src/tac/local_acceleration/__init__.py \
  src/tac/local_acceleration/mlx_acquisition_batch.py \
  src/tac/local_acceleration/mlx_execution_plan.py \
  src/tac/optimization/inverse_steganalysis_acquisition.py \
  tools/build_inverse_steganalysis_action_functional.py \
  tools/build_local_training_execution_queue.py \
  tools/build_mlx_acquisition_batch.py \
  tools/build_mlx_scorer_response_execution_queue.py \
  tools/build_mlx_window_response_dataset.py \
  tools/plan_mlx_scorer_response_execution.py \
  tools/run_byte_shaving_materializer_campaign.py \
  src/tac/tests/test_byte_shaving_campaign_queue.py \
  src/tac/tests/test_byte_shaving_materializer_campaign_runner.py \
  src/tac/tests/test_inverse_steganalysis_action_functional_cli.py \
  src/tac/tests/test_local_training_execution_queue.py \
  src/tac/tests/test_mlx_execution_queue.py
# All checks passed.
```
