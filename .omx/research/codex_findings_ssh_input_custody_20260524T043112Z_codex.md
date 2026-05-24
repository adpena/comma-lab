# Codex Findings - SSH Input Custody Hardening

UTC: 2026-05-24T04:31:12Z
Author: Codex
Lane: `codex_ssh_artifact_mobility_20260524`

## Finding

The queue-owned SSH path still had input-custody gaps after the first artifact
mobility landing. Inverse-scorer chain rows could pass precomputed source and
candidate inflate-output directories to the command without declaring those
directories as mobile inputs. Separately, `experiment_queue.v1` preserved
`telemetry.input_artifact_paths` without normalizing it, and SSH terminal
events recorded rsync path movement but not content identity.

## Landing

- `materialize_inverse_scorer_cell_candidate` chain rows now include
  precomputed `source_inflate_output_dir` and `candidate_inflate_output_dir`
  in `telemetry.input_artifact_paths`.
- `experiment_queue.v1` normalizes `telemetry.input_artifact_paths` the same
  way it normalizes output artifact paths.
- SSH path maps now reject filesystem-root local or remote prefixes, and remote
  artifact paths reject shell and glob metacharacters.
- SSH selection and execution reject symlinked input artifacts.
- SSH input pushes record local file or recursive directory manifests with
  bytes and SHA-256 content identity before the remote command runs.
- Directory input pushes use `rsync --delete`, preventing stale remote files
  from remaining part of the effective worker input.

## Verification

- `.venv/bin/python -m pytest -q src/tac/tests/test_byte_shaving_campaign_queue.py src/tac/tests/test_experiment_queue.py src/tac/tests/test_ssh_experiment_queue_executor.py`
- `.venv/bin/python -m ruff check src/comma_lab/scheduler/byte_shaving_campaign_queue.py src/comma_lab/scheduler/experiment_queue.py src/comma_lab/scheduler/ssh_experiment_queue_executor.py src/tac/tests/test_byte_shaving_campaign_queue.py src/tac/tests/test_experiment_queue.py src/tac/tests/test_ssh_experiment_queue_executor.py`
- `.venv/bin/python -m py_compile src/comma_lab/scheduler/byte_shaving_campaign_queue.py src/comma_lab/scheduler/experiment_queue.py src/comma_lab/scheduler/ssh_experiment_queue_executor.py`

Latest focused SSH/input-custody suite result: `136 passed`. Latest combined
materializer/scheduler custody suite result: `165 passed`; ruff and py_compile
passed.

## Remaining Work

Regenerate the live SSH materializer smoke with a chain-mode directory-input
case so the artifact directory under `experiments/results/` carries the new
input-manifest fields. That smoke should stay non-authoritative for score and
should only graduate after the queue event proves both input push and output
pullback custody.
