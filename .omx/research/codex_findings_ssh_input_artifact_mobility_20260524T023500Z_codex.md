# Codex Findings - SSH Input Artifact Mobility

UTC: 2026-05-24T02:35:00Z
Author: Codex
Lane: `lane_codex_ssh_input_artifact_mobility_20260524`

## Finding

The SSH executor could pull remote postcondition artifacts back into local
custody, but remote materializer steps also depend on local planning inputs:
scorer-response rows, inverse-scorer surfaces, runtime adapters, source
archives, schema manifests, and source runtimes. Without explicit input
mobility, a remote worker could fail because its command referenced local-only
paths, or worse, rely on stale remote inputs not bound to the current queue
definition.

## Landing

Materializer queue telemetry now carries `input_artifact_paths` for inverse
action-functional rows, inverse-scorer cell candidate rows, and byte-range
materializer chains. The SSH executor validates those inputs during task
selection, requires path-map coverage for declared inputs even if a direct API
caller forgets to request output pullback enforcement, and pushes mapped inputs
with rsync before running the remote command. Malformed
`input_artifact_paths` telemetry blocks selection, and directory input pushes
use rsync `--delete` so stale remote runtime files cannot survive after local
removal.

Input mobility is recorded in the terminal queue event as
`staircase_ssh_input_mobility.v1`. Shared-storage mode skips rsync push only
under the existing explicit shared-storage rationale; missing local inputs still
block selection.

## Verification

- `PYTHONPATH=. .venv/bin/python -m pytest -q src/tac/tests/test_ssh_experiment_queue_executor.py src/tac/tests/test_byte_shaving_campaign_queue.py src/tac/tests/test_byte_shaving_materializer_campaign_runner.py`
- `PYTHONPATH=. .venv/bin/python -m ruff check src/comma_lab/scheduler/ssh_experiment_queue_executor.py src/comma_lab/scheduler/byte_shaving_campaign_queue.py src/tac/tests/test_ssh_experiment_queue_executor.py src/tac/tests/test_byte_shaving_campaign_queue.py`
- `PYTHONPATH=. .venv/bin/python -m py_compile src/comma_lab/scheduler/ssh_experiment_queue_executor.py src/comma_lab/scheduler/byte_shaving_campaign_queue.py tools/run_staircase_ssh_executor.py tools/build_byte_shaving_campaign_queue.py`
- `PYTHONPATH=. .venv/bin/python -m pytest -q src/tac/tests/test_ssh_experiment_queue_executor.py src/tac/tests/test_byte_shaving_campaign_queue.py src/tac/tests/test_byte_shaving_materializer_campaign_runner.py src/tac/tests/test_experiment_queue.py`
- Live SSH smoke:
  `experiments/results/inverse_action_ssh_materializer_smoke_20260524T0246Z/campaign/staircase_ssh_executor_execute.json`

Result: focused scheduler/materializer and queue-authority suite passed with
`142 passed`. The live
`tertiary_m1_macbook_pro_8gb` smoke executed one bounded SSH task with
`execution_mode=bounded_parallel_ssh_executor`, `success_count=1`,
`failure_count=0`, `input_mobility.mode=rsync_push`, and
`artifact_mobility.mode=rsync_pull`. The remote preflight guarded HEAD
`7495fad13b67048dbf7e8ffd9fffb616b631beb7`; the output
`action_functional.json` stayed planning-only and false-authority.

## Remaining Work

Exact CPU/CUDA score authority remains outside this executor. The next
execution step is scaling the same input/output mobility contract across
multi-node materializer batches and the Windows/RTX worker, while keeping queue
identity and local postcondition custody authoritative.
