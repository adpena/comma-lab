# Codex Findings - IAS1 Exact Auth Harvest And Scheduler DAG Metadata

UTC: 2026-05-23T20:15:56Z
Lanes:
- `lane_inverse_scorer_exact_eval_queue_bridge_20260523`
- `lane_materializer_scheduler_preflight_20260523`

## Finding

The IAS1 inverse-scorer runtime-parity candidate moved from exact-ready custody
into paired Modal auth eval and produced a clear negative anchor on both contest
axes:

- `[contest-CPU Linux x86_64]`: `0.19380912393883232`
- `[contest-CUDA T4]`: `0.2279696105246996`
- Archive SHA-256:
  `2d0850789483e17c7ee68ae8bfe1e33489d1981416f71266cf8a66b19a87e549`
- Archive bytes: `181232`

This does not move either frontier. It is useful labeled signal for the
inverse-scorer planner: the byte-closed/runtime-correct IAS1 top-4 candidate
was materially worse than the DQS1 CPU frontier and the PR106 CUDA frontier
despite passing exact auth eval and full runtime custody.

During recovery, the CUDA result exposed a recovery idempotence bug class:
re-running the canonical recovery tool could append a duplicate terminal claim
row and touch the posterior path again for the same terminal result. That does
not change the harvested score, but it is claim-ledger signal loss/noise and can
make exact-ready audit output harder to reason about.

The materializer scheduler preflight queue also still had an integration gap:
execution steps depended on scheduler cleanup, but the generated staircase DAG
and Dask/local task specs did not expose machine-readable storage-preflight
metadata for downstream executors.

## Fix

- Recovered the IAS1 CUDA and CPU Modal auth-eval calls through
  `tools/recover_modal_auth_eval.py`, closing both active dispatch claims.
- Hardened `tools/recover_modal_auth_eval.py` so if the newest matching
  lane/job claim is already terminal with the same result JSON, a repeat
  recovery skips both duplicate terminal claim closure and duplicate posterior
  update.
- Added a regression test covering duplicate terminal recovery suppression.
- Added `staircase_storage_preflight_dependency.v1` metadata propagation in
  `comma_lab.scheduler.staircase_dag`:
  - detects storage/cleanup scheduler preflight experiments;
  - links downstream nodes that depend on `*.proactive_cleanup`;
  - carries storage-plan and cleanup-plan artifact paths into DAG node metadata;
  - exposes the same false-authority metadata in Dask/local task specs.
- Added a focused materializer execution queue test proving the storage preflight
  dependency survives queue -> DAG -> dispatch-plan conversion.

## Verification

- `tools/recover_modal_auth_eval.py` harvested IAS1 CPU and CUDA exact auth
  results and `tools/claim_lane_dispatch.py summary` reports `active=0`.
- `tools/scan_best_anchor_per_axis.py` still reports frontiers:
  - `[contest-CPU Linux x86_64]` `0.1920282830`
  - `[contest-CUDA T4]` `0.2053300290`
- Focused tests:
  `src/tac/tests/test_byte_shaving_campaign_queue.py`: `31 passed`.
  `src/tac/tests/test_staircase_dag.py`
  plus `src/tac/tests/test_recover_modal_auth_eval_tool.py`: `19 passed`.
  `src/tac/tests/test_recover_modal_auth_eval_tool.py`
  plus `src/tac/tests/test_inverse_scorer_exact_eval_queue.py`: `19 passed`.
- `ruff check` passed on touched Python files.
- `git diff --check` passed on touched files.

## Authority

The IAS1 exact auth results are valid contest-axis score evidence, but they are
regressions and do not promote the candidate. Scheduler/DAG metadata remains
planning-only and false-authority; it does not claim score, promote, rank, kill,
or bypass exact auth eval.
