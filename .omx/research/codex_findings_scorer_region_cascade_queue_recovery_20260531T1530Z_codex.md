# Codex Findings - Scorer-Region Cascade Queue Recovery - 2026-05-31T15:30Z

## Scope

Queue-owned b7106 scorer-region selector cascade:
`scorer_region_selector_cascade_b7106_mg_mlx_full_20260531T132954Z`.

Run root:
`/Volumes/VertigoDataTier/experiments/results/scorer_region_selector_cascade_b7106_mg_mlx_full_20260531T132954Z`.

Research artifacts:
`.omx/research/scorer_region_selector_cascade_b7106_mg_mlx_full_20260531T132954Z`.

## Finding

The live queue had no true candidate failure, but recovery treated MLX-negative
CPU-spend gates as hard failures or unhealthy queued artifacts. The concrete
case was two `mlx_cpu_spend_gate` artifacts with `cpu_gate_allowed=false`; those
are expected acquisition negatives and should close their branch, not stop the
queue.

## Fix Landed

- Added `on_postcondition_failure=skipped` to experiment queue step semantics.
- Made direct execution, external finalize, stale-running recovery, and
  postcondition reconciliation honor the soft-gate policy.
- Added automatic downstream skip propagation for queued descendants of a
  skipped dependency.
- Added process-group isolation and timeout cleanup for queue subprocesses so
  child inflaters/cache builders do not survive failed workers.
- Added MLX response batch-shape normalization as a reusable queue rewriter:
  production scoring stays singleton-batch; cache batch shape remains separate.
- Added process-safe shell inflate left-cache reuse for receiver-output proofs.
- Added supervisor auto-resume for paused queues with runnable work.

## Evidence

Focused tests passed:

```text
12 passed in 8.05s
```

Live queue recovery:

```text
before: {"queued": 341, "succeeded": 71}
after:  {"queued": 328, "skipped": 13, "succeeded": 71}
direct soft-gate skips: 2
downstream dependency skips: 11
```

After two bounded supervisor waves:

```text
status: {"queued": 300, "skipped": 28, "succeeded": 84}
healthy: true
blockers: []
ready_for_exact_eval_dispatch: false
```

Artifacts:

- `soft_gate_reconcile_after_skip_contract_fix.json`
  sha256 `97e67a52b31585a1f25ba0a0227ec917fe9903bfdea55b7988687b2524cf7d4d`
- `soft_gate_observer_after_wave6.json`
  sha256 `3d8378476d8ce225a8ff837a5c7439edde17bc4d08906c380ea801648f2e668c`
- `soft_gate_supervisor_wave5/supervisor_result.json`
  sha256 `00fd3aff1ebf22d2ca9d279c86b1877d0e61ec521a0195e70649efc8d1e5866b`
- `soft_gate_supervisor_wave6/supervisor_result.json`
  sha256 `1d00554b7450b8494baf65d72282988ef8d01ba2b072aad823233dbc41fd82be`
- `queue.soft_gate_locked_cache_batch1_runtime_policy_applied.json`
  sha256 `44624127ae186e63004f96b291dee821863015daada6fa48543b14b5c5c25b58`

## Score Authority

No score claim. Latest MLX research-signal candidates were worse than the
current CPU frontier:

```text
best [macOS-MLX research-signal]: 0.192399501201
current [contest-CPU Linux x86_64]: 0.1919853363
```

Exact auth dispatch correctly remained closed.

## Next Automation Work

The largest remaining throughput loss is duplicate full inflate/cache work:
receiver-output proof and MLX cache materialization both hydrate large raw
outputs. Next implementation should add right-cache/preinflated-output handoff
so P18/P19/P11/P15 chains can reuse identical archive inflations across
receiver proof, MLX acquisition, CPU spot check, and exact-readiness prep.

