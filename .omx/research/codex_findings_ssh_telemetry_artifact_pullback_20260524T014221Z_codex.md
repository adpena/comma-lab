# Codex Findings - SSH Telemetry Artifact Pullback

UTC: 2026-05-24T01:42:21Z
Author: Codex
Lane: `codex_ssh_telemetry_artifact_pullback_20260524T014221Z`

## Finding

The first successful `tertiary` SSH materializer smoke proved queue-owned remote
execution and JSON postcondition pullback, but the observation payload exposed
signal loss: `action_functional.md` was declared in
`step.telemetry.artifact_paths` and existed on the remote, but the SSH executor
only pulled local-visible postcondition paths. The local final event therefore
recorded the Markdown telemetry artifact as missing even though the remote run
had produced it.

Postconditions are the authority gate, but telemetry artifacts are still signal
and must not be silently stranded on a peer machine.

## Landing

`ssh_experiment_queue_executor` now includes mapped telemetry artifact paths in
the artifact mobility contract and rsync pullback set, de-duplicated with
postcondition artifacts. When artifact mobility is required, missing path-map
coverage for telemetry artifacts now produces an explicit
`artifact_pullback_missing_for_telemetry:*` blocker.

## Verification

- `PYTHONPATH=. .venv/bin/python -m ruff check src/comma_lab/scheduler/ssh_experiment_queue_executor.py src/tac/tests/test_ssh_experiment_queue_executor.py`
- `PYTHONPATH=. .venv/bin/python -m pytest -q src/tac/tests/test_ssh_experiment_queue_executor.py::test_ssh_executor_pulls_back_artifacts_before_local_postconditions src/tac/tests/test_ssh_experiment_queue_executor.py::test_ssh_executor_pulls_back_mapped_telemetry_artifacts src/tac/tests/test_ssh_experiment_queue_executor.py::test_ssh_executor_fails_remote_success_when_artifact_pullback_fails`
- `PYTHONPATH=. .venv/bin/python -m py_compile src/comma_lab/scheduler/ssh_experiment_queue_executor.py src/tac/tests/test_ssh_experiment_queue_executor.py`

Result: all passed.

## Remaining Work

After commit and push, rerun the bounded `tertiary` planning-only SSH
materializer smoke. Success requires both the JSON postcondition artifact and
the Markdown telemetry artifact to be present locally after rsync pullback.
