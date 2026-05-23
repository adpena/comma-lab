# Codex Session Summary

UTC: 2026-05-23T20:15:56Z

## Landed

- Harvested the IAS1 exact-auth pair:
  - `[contest-CPU Linux x86_64]` `0.19380912393883232`
  - `[contest-CUDA T4]` `0.2279696105246996`
- Confirmed both IAS1 exact-auth results regress versus current scanner
  frontiers and remain non-promotional.
- Closed the active IAS1 CPU/CUDA Modal auth-eval dispatch claims.
- Hardened Modal auth-eval recovery against duplicate terminal claim and
  duplicate posterior updates for already-closed lane/job results.
- Propagated scheduler storage-preflight dependency metadata from experiment
  queues into staircase DAG nodes and Dask/local task specs.

## Verification

- `tools/claim_lane_dispatch.py summary`: `active=0`.
- `tools/scan_best_anchor_per_axis.py`: no frontier change.
- Focused tests:
  `src/tac/tests/test_byte_shaving_campaign_queue.py`: `31 passed`.
  `src/tac/tests/test_staircase_dag.py`
  plus `src/tac/tests/test_recover_modal_auth_eval_tool.py`: `19 passed`.
  `src/tac/tests/test_recover_modal_auth_eval_tool.py`
  plus `src/tac/tests/test_inverse_scorer_exact_eval_queue.py`: `19 passed`.
- `ruff check` passed on touched Python files.
- `git diff --check` passed on touched files.

## Next Required Action

Use the IAS1 exact negative as labeled acquisition signal. The immediate
follow-up is to inspect whether the loss came from byte overhead, underpriced
SegNet/PoseNet damage, an IAS1-specific configuration issue, or inverse-scorer
calibration failure, then feed that into the planner rather than hand-selecting
another isolated candidate.
