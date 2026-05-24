# Codex Findings - Materializer Dispatch Plan Axis Repair

UTC: 2026-05-24T04:30:37Z
Author: Codex
Lane: `codex_materializer_dispatch_plan_identity_ev_20260524`

## Finding

The repaired exact-ready score-axis queue set contained 12 distinct
archive/runtime-content/runtime-tree/score-axis identities, but the materializer
exact-eval dispatch planner still treated repeated `candidate_id` as the
dedupe key. That made the review surface call score-affecting alternate
runtimes "duplicate candidate ids" instead of naming the real constraint:
only one active dispatch claim per lane can be live until a terminal claim
closes it.

The planner also inherited a static CUDA score floor. Canonical frontier scan
currently reports `[contest-CUDA]` `0.20533002902019143`, which is stricter
than the legacy static floor and must be the default review-floor when
available.

## Landing

- Dispatch plans now dedupe score-affecting identity by
  archive SHA, runtime-content SHA, runtime-tree SHA, and score axis.
- Same-lane alternate stable identities are preserved in the plan and blocked
  with `same_lane_dispatch_claim_serialization_required:*` until the prior
  lane dispatch has a terminal claim.
- A lane is marked occupied only after a row passes exact-dispatch authority;
  blocked/stale first rows do not suppress a later valid same-lane row.
- Plan rows now carry `dispatch_group_key` and authorized rows carry
  `dispatch_priority_rank`.
- The planner tightens `active_floor_score` from
  `tac.frontier_scan.best_per_axis.contest_cuda` when canonical state has a
  lower CUDA frontier than the static fallback.

## Generated Review Artifact

The per-run dispatch-plan JSON remains ignored by `.gitignore` by design;
durable conclusions are recorded here.

- Plan path: `.omx/research/materializer_exact_eval_dispatch_plan_axis_repair_20260524T043037Z.json`
- Plan SHA-256: `65aff3673b619e4d033b96d52396330c74c1afe4108f2a7f53e41cc62f5e538d`
- Queue path: `.omx/research/materializer_exact_eval_dispatch_plan_axis_repair_20260524T043037Z.experiment_queue.json`
- Queue SHA-256: `e514157f301132e7cd4e89fad99f1e040d0c5c75a646af4debfa797a9ffb0e7a`
- Exact-ready rows inspected: `12`
- Authorized dry-run dispatch rows: `6`
- Same-lane serial blockers: `6`
- Duplicate stable identities: `0`
- Dispatch mode: `dry_run`
- Experiment queue mode: `paused`
- Estimated total cost if later converted to reviewed execute mode: `$1.80`
- Score/promotion/rank authority: `false`
- Dispatch attempted: `false`

## Verification

- `.venv/bin/python -m pytest -q src/tac/tests/test_materializer_chain_harvest_scheduler.py -k 'materializer_dispatch_plan'`
- `.venv/bin/python -m pytest -q src/tac/tests/test_materializer_chain_harvest_scheduler.py src/tac/tests/test_ssh_experiment_queue_executor.py src/tac/tests/test_experiment_queue.py src/tac/tests/test_byte_shaving_campaign_queue.py`
- `.venv/bin/python -m ruff check src/comma_lab/scheduler/materializer_exact_eval_dispatch_plan.py src/comma_lab/scheduler/ssh_experiment_queue_executor.py src/comma_lab/scheduler/experiment_queue.py src/comma_lab/scheduler/byte_shaving_campaign_queue.py src/tac/tests/test_materializer_chain_harvest_scheduler.py src/tac/tests/test_ssh_experiment_queue_executor.py src/tac/tests/test_experiment_queue.py src/tac/tests/test_byte_shaving_campaign_queue.py`
- `.venv/bin/python -m py_compile src/comma_lab/scheduler/materializer_exact_eval_dispatch_plan.py src/comma_lab/scheduler/ssh_experiment_queue_executor.py src/comma_lab/scheduler/experiment_queue.py src/comma_lab/scheduler/byte_shaving_campaign_queue.py src/tac/tests/test_materializer_chain_harvest_scheduler.py src/tac/tests/test_ssh_experiment_queue_executor.py src/tac/tests/test_experiment_queue.py src/tac/tests/test_byte_shaving_campaign_queue.py`

Latest combined suite result: `163 passed`; ruff and py_compile passed.

## Next Step

If operator or autopilot chooses to spend, convert only the six authorized
first-per-lane rows into execute mode after a fresh frontier scan and live claim
check. The six same-lane alternates should be re-planned after terminal claim
rows exist for their lane predecessors.
