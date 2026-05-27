# Codex Findings: Experiment Queue Metadata Loss

UTC: 2026-05-27T05:51:32Z

## Finding

`normalize_queue_definition()` preserved experiment-level metadata but silently
dropped queue-level metadata. That was a real signal-loss bug class: queue
builders could attach custody, authority, blocked-work-order, or acquisition
summary metadata at the queue root, then lose it before validation, state
initialization, worker execution, and downstream consumers.

The 5D coverage acquisition queue made the bug visible because the root queue
metadata needs to distinguish:

- locally executable work orders that can refresh the current canvas from local
  anchors;
- externally blocked work orders that still require exact-axis bundles, dispatch
  claims, MLX cache pairs, or scorer-response seeds;
- false-authority status for the whole queue.

Without root metadata preservation, those classifications remained available
only inside individual experiments and were easier for autonomous consumers to
miss.

## Fix

`normalize_queue_definition()` now preserves an optional top-level `metadata`
object using the same strict object normalization used for experiment metadata.

The 5D coverage acquisition queue also records explicit root and refresh
metadata:

- `executable_work_order_ids`
- `blocked_work_order_ids`
- `plan_classes_by_work_order`
- `local_refresh_consumes_work_order_ids`
- `external_blocking_work_order_ids`
- `refresh_semantics`

This keeps the automation honest: emitting an acquisition plan is not treated as
resolving exact-axis or MLX-cache blockers.

## Verification

- `.venv/bin/ruff check src/comma_lab/scheduler/experiment_queue.py src/tac/tests/test_experiment_queue.py src/tac/tests/test_pair_frame_5d_coverage_acquisition_queue.py`:
  pass.
- `.venv/bin/python -m pytest src/tac/tests/test_experiment_queue.py src/tac/tests/test_pair_frame_5d_coverage_acquisition_queue.py -q`:
  72 passed.
- `.venv/bin/python -m pytest src/tac/tests/test_pair_frame_5d_extended_operator_queue.py -q`:
  4 passed.
- Live queue smoke against the current 5D coverage audit:
  - queue validation: `valid=true`, `experiment_count=6`, `step_count=9`
  - acquisition-plan worker: `failure_count=0`, `success_count=5`
  - refresh/refire worker: `failure_count=0`, `success_count=4`
  - refired extended-operator worker: `failure_count=0`, `success_count=8`

## Remaining Work

The coverage acquisition queue is still a local-first planning and refire layer.
It does not itself perform paid or exact-axis dispatch, and it does not create
MLX scorer-response cache pairs. The next durable bridge is to turn the blocked
work-order classes into queue-owned exact-axis and MLX-cache acquisition lanes
with dispatch claims, artifact custody, and false-authority refusal tests.
