# Codex Findings: Queue Materializer Feedback Bridge

UTC: 2026-05-25T10:44:26Z
Agent: Codex
Lane: `codex_queue_materializer_feedback_bridge_20260525`

## Scope

Adversarial implementation pass on queue-owned family-agnostic materializer
sweeps feeding the inverse-steganalysis action surface. The target bug class was
successful sweep observations being emitted as side artifacts without automatic
planner consumption.

## Findings And Fixes

- Successful materializer feedback artifacts are now surfaced in
  `experiment_queue_observation.v1` under `succeeded_artifact_steps` when the
  postcondition is explicitly a
  `family_agnostic_materializer_empirical_observation.v1` false-authority JSONL
  artifact or the JSON artifact schema is the family materializer observation or
  sweep schema.
- `observations_from_queue_observation(...)` now consumes those
  `succeeded_artifact_steps` and converts referenced family materializer JSONL or
  sweep JSON rows into normalized inverse-steganalysis observations.
- Queue materializer artifact ingestion is bounded and fail-closed: max artifact
  bytes, max rows, invalid JSON/JSONL rejection, required explicit false
  authority, and forbidden truthy authority rejection.
- Step identity is now preserved from both queue observations and performance
  identity maps, fixing a metadata-loss class for `work_ids`, `backlog_keys`,
  `source_unit_ids`, and `source_selection_ids`.
- Materializer observation normalization now infers `rate_positive` from
  positive `saved_bytes` plus positive `observed_rate_gain` when older/direct
  observation rows omit the explicit flag.
- Materializer delta observation merging now prefers `candidate_archive_sha256`,
  then `observation_id`, then artifact path, preventing duplicate JSON/JSONL
  sweep rows from double-counting when a sweep JSON and its JSONL contain the
  same observation.

## Verification

- `PYTHONPATH=. .venv/bin/pytest src/tac/tests/test_inverse_steganalysis_acquisition.py src/tac/tests/test_experiment_queue_observer.py -q`
  - Result: `63 passed`
- `PYTHONPATH=. .venv/bin/pytest src/tac/tests/test_experiment_queue_observer.py src/tac/tests/test_byte_shaving_campaign_queue.py src/tac/tests/test_family_agnostic_materializer_sweep.py src/tac/tests/test_byte_shaving_materializer_campaign_runner.py src/tac/tests/test_inverse_steganalysis_action_functional_cli.py -q`
  - Result: `177 passed`
- `PYTHONPATH=. .venv/bin/pytest src/tac/tests/test_family_agnostic_materializers.py src/tac/tests/test_experiment_queue_observer.py src/tac/tests/test_byte_shaving_materializer_campaign_runner.py src/tac/tests/test_inverse_steganalysis_acquisition.py src/tac/tests/test_byte_shaving_campaign_queue.py src/tac/tests/test_family_agnostic_materializer_sweep.py src/tac/tests/test_inverse_steganalysis_action_functional_cli.py -q`
  - Result: `261 passed`
- `.venv/bin/python -m ruff check src/tac/optimization/inverse_steganalysis_acquisition.py src/comma_lab/scheduler/experiment_queue_observer.py src/tac/tests/test_inverse_steganalysis_acquisition.py src/tac/tests/test_experiment_queue_observer.py --no-cache`
  - Result: pass

## Remaining Work

- Promote this bridge into the next queue-owned acquisition loop by emitting
  grouped materializer sweep specs directly from selected inverse-steg action
  cells.
- Add a run-dir fixture containing both `sweep.json` and `observations.jsonl`
  for duplicate suppression proof against real materializer output.
- Extend the same artifact ingestion pattern to packetIR/compiler materializer
  rows once the packetIR contracts emit family materializer observation schema.

