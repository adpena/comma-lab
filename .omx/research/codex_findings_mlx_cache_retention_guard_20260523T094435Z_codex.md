# Codex Findings: MLX Cache Retention Guard

Timestamp UTC: 2026-05-23T09:44:35Z

## Scope

Hardened the DQS1 local MLX queue path against the disk-pressure failure mode
from rebuildable raw and tensor artifacts. The prior MLX queue wiring could
produce a full-sample `mlx_delta_cache/` but did not automatically produce a
retention plan, and the generic retention certifier trusted MLX cache manifests
without proving the external identity-audit artifact still existed.

## Findings

- `mlx_delta_cache/` compaction must be gated by a dereferenceable identity
  audit stamp, not only by local manifest hashes.
- The previous retention certifier compared `.npy` file SHA-256s against
  tensor-content `array_sha256` values. Those are different hash domains. The
  certifier now validates `.npy` files against `manifest.artifacts.*.sha256`
  while preserving `array_sha256` as tensor identity.
- Local CPU advisory MLX cache stamps now include `local_cpu_advisory_path`, so
  retention certificates can point at the surviving rebuild/source artifact.

## Fix

- `tools/build_mlx_scorer_input_cache_from_local_advisory.py` stamps the cache
  manifest with `local_cpu_advisory_path`.
- `src/comma_lab/artifact_retention.py` requires either
  `auth_eval_identity_audit` or `local_cpu_advisory_cache_identity_audit`, loads
  the referenced audit JSON, verifies the stamp SHA, checks false-authority
  fields, and verifies the audit cache identity matches the cache manifest.
- `src/comma_lab/artifact_retention.py` now validates cache file bytes against
  artifact hashes, not tensor-content hashes.
- `configs/experiment_queues/dqs1_pairset_local_first.yaml` now includes
  `plan_mlx_delta_cache_retention` after `local_mlx_advisory_response`.

## Current DQS1 Queue Observation

`tools/experiment_queue.py validate` now reports:

- `step_count = 8`
- `local_only.max_parallel = 2`
- `resource_limits = {"local_cpu": 1, "local_mlx": 1}`

State reconciliation artifact:
`.omx/research/dqs1_queue_state_reconciliation_20260523T094413Z_codex.json`

- `blocking_orphan_count_before = 0`
- `blocking_orphan_count_after = 0`
- `retired_step_count = 0`
- `after.status_counts = {"queued": 8}`

## Verification

- `.venv/bin/python -m pytest -q src/tac/tests/test_artifact_retention.py src/tac/tests/test_dqs1_local_first_queue_builder.py src/tac/tests/test_mlx_preprocess.py::test_build_mlx_cache_from_local_advisory_cli_stamps_manifest`
- `.venv/bin/ruff check src/comma_lab/artifact_retention.py src/comma_lab/scheduler/dqs1_local_first_queue.py tools/build_dqs1_local_first_queue.py tools/build_mlx_scorer_input_cache_from_local_advisory.py src/tac/tests/test_artifact_retention.py src/tac/tests/test_dqs1_local_first_queue_builder.py src/tac/tests/test_mlx_preprocess.py`
- `.venv/bin/python tools/experiment_queue.py --queue configs/experiment_queues/dqs1_pairset_local_first.yaml validate`
- `.venv/bin/python tools/experiment_queue.py --queue configs/experiment_queues/dqs1_pairset_local_first.yaml run-worker --max-steps 1 --max-idle-cycles 0 --idle-sleep-seconds 0 --no-reload-definition`

## Next Hooks

- Execute the DQS1 queue until the first `plan_mlx_delta_cache_retention` output
  exists, then inspect the retention plan before deleting or moving caches.
- Add an optional cold-store execute policy for MLX caches once the attached SSD
  mount is verified healthy and outside the repo root.
- Feed the harvested MLX response and retention-plan facts into the queue
  observer so live telemetry shows reclaimable tensor bytes.
