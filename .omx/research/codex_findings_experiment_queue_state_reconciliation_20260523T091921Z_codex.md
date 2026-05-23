# Codex Findings: Experiment Queue State Reconciliation

Timestamp UTC: 2026-05-23T09:19:21Z

## Scope

Hardened the local-first experiment queue control surface after the DQS1 queue
observer exposed historical orphaned SQLite rows. The risk was not that those
rows were immediately executable, but that the operator-visible signal did not
distinguish blocking stale work from historical non-current queue state.

## Findings

- `configs/experiment_queues/dqs1_pairset_local_first.yaml` had 123 orphaned
  state rows in the default SQLite queue state.
- Those rows were historical `succeeded` or `skipped` steps, so they were not
  blocking worker launch. The previous operator flow still required manual
  interpretation to prove that.
- Queue state reconciliation should be an append-only, false-authority artifact,
  not an implicit side effect of launching a worker.

## Fix

- Added `tools/experiment_queue.py reconcile-state`.
- The command initializes missing current rows, retires blocking stale orphans
  with an explicit reason, and writes an optional append-only JSON audit artifact.
- The artifact records total orphan counts and explicit
  `blocking_orphan_count_before` / `blocking_orphan_count_after` fields.
- The artifact is marked non-authoritative for score, promotion, rank/kill, and
  exact-eval dispatch.

## DQS1 Result

Artifact:
`.omx/research/dqs1_queue_state_reconciliation_20260523T091900Z_codex.json`

- `blocking_orphan_count_before = 0`
- `blocking_orphan_count_after = 0`
- `retired_step_count = 0`
- `before.orphaned_step_count = 123`
- `after.orphaned_step_count = 123`
- `after.status_counts = {"queued": 5}`

The current DQS1 queue therefore has five current queued steps and no blocking
stale orphaned work. A dry-run worker launch plans
`pairset_drop_two_r029_020_p0259_0430.plan_packet` as the next executable step.

## Integration

- The reconciliation command uses the same canonical queue definition loader,
  SQLite state path resolver, queue summary surface, and orphan retirement
  helper as the worker.
- Regression coverage verifies that blocking stale queued rows are retired,
  non-authoritative fields are present, and the audit artifact records before
  and after queue summaries.

## Next Hooks

- Wire MLX scorer-response/cache steps into the DQS1 local-first queue builder
  so declared `local_mlx` capacity is used by real queue nodes.
- Add queue-level candidate families instead of a single hand-expanded DQS1
  candidate list, using X-ray, atom, master-gradient, canonical-equation, and
  scorer-response calibration signals as acquisition inputs.
- Keep CPU, MLX, contest CPU, and contest CUDA axes separate. This
  reconciliation artifact is workflow authority only, not score authority.
