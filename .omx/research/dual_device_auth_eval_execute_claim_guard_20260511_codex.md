# Dual-device auth eval execute claim guard (2026-05-11)

## Scope

`tools/plan_dual_device_auth_eval.py` can emit paired CPU/CUDA eval commands
for the same archive/runtime. Before this change, `--execute` could run a local
auth eval directly after input closure, leaving the dispatch-claim rule to
operator discipline.

That was too weak. AGENTS requires a lane claim before any eval or training
job, not only remote GPU jobs.

## Change

`--execute` now requires:

- `--lane-id`
- `--instance-job-id`

The planner records:

1. an active claim with status `active_eval_running` before the first eval
   subprocess starts;
2. a forced terminal claim with status `completed_auth_eval_plan_execute` after
   all selected axes pass;
3. a forced terminal claim with status
   `failed_auth_eval_plan_execute_rc<N>` if any eval subprocess exits nonzero.

If the initial claim is refused because of an active same-lane conflict, the
planner exits before running the eval.

## Verification

Focused tests:

```bash
.venv/bin/python -m pytest -q \
  src/tac/tests/test_plan_dual_device_auth_eval.py \
  src/tac/tests/test_analyze_cpu_cuda_eval_drift.py
```

Result: `16 passed in 0.18s`.

## Classification

- score_claim: `false`
- dispatch_attempted: `false`
- remote_or_gpu_eval_started: `false`
- purpose: custody hardening for the next PR103-on-PR106 raw-output rerun after
  the active T1 Modal claim clears
