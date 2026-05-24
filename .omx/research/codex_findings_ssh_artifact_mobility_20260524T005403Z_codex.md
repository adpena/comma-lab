# Codex Findings - SSH Artifact Mobility Contract

UTC: 2026-05-24T00:54:03Z
Author: Codex
Lane: `codex_ssh_artifact_mobility_20260524`

## Finding

The queue-owned SSH executor had the right authority boundary for local queue
state, but remote artifact custody was still an implicit operator convention.
A remote command could run on a peer machine while terminal success still
depended on local-visible postconditions, leaving a gap between remote output
paths and the local authority state. That is a signal-loss and false-authority
bug class for distributed materializer campaigns.

## Landing

The SSH executor now carries an explicit artifact mobility contract from
`experiment_queue.v1` through `staircase_dag.v1` into `staircase_dispatch_plan.v1`.
`artifact_mobility` participates in step definition hashes so stale or stripped
plans fail closed. SSH task selection can require either shared-storage
visibility or explicit local-prefix to remote-prefix pullback maps.

When execution is enabled, the SSH executor now claims the queue step locally,
runs the remote command, performs rsync pullbacks for mapped local
postcondition artifacts, and only then evaluates local postconditions through
the canonical queue finalizer. A remote zero exit code without successful
local artifact visibility is recorded as a failed queue step.

The materializer campaign runner exposes the same contract through
`--staircase-ssh-execute`, `--staircase-ssh-artifact-path-map`,
`--staircase-ssh-artifact-shared-path-rationale`, and
`--staircase-ssh-require-artifact-mobility`. SSH execution is mutually
exclusive with top-level local `--execute`, carries its own
`--staircase-ssh-max-steps` bound, and requires an artifact mobility
declaration.

## Verification

- `PYTHONPATH=. .venv/bin/python -m ruff check src/comma_lab/scheduler/experiment_queue.py src/comma_lab/scheduler/staircase_dag.py src/comma_lab/scheduler/ssh_experiment_queue_executor.py src/comma_lab/scheduler/__init__.py tools/run_staircase_ssh_executor.py tools/run_byte_shaving_materializer_campaign.py src/tac/tests/test_ssh_experiment_queue_executor.py src/tac/tests/test_byte_shaving_materializer_campaign_runner.py src/tac/tests/test_staircase_dag.py`
- `PYTHONPATH=. .venv/bin/python -m pytest -q src/tac/tests/test_byte_shaving_materializer_campaign_runner.py src/tac/tests/test_ssh_experiment_queue_executor.py src/tac/tests/test_staircase_dag.py src/tac/tests/test_experiment_queue.py`

Result: `91 passed`.

## Remaining Work

1. Add a first real bounded `tertiary` shared-storage or pullback smoke from a
   generated materializer queue, using noncanonical-state rationale only in an
   isolated run directory if required.
2. Add manifest identity validation for richer remote outputs where the
   postcondition artifact is not itself already a strict `materializer_chain_complete`
   or `json_completion_contract`.
3. Promote repeated path-map and storage-root conventions into a reusable
   `artifact_mobility` helper module if more than SSH consumes them.
4. Extend fleet scheduling to choose between local MLX, local CPU, and SSH CPU
   workers based on resource slots plus storage tier pressure.
