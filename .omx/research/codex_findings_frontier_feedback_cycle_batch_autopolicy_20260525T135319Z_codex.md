# Frontier Feedback Cycle Batch Autopolicy

- timestamp_utc: 2026-05-25T13:53:19Z
- author: Codex
- scope: frontier-rate attack feedback loop, DQS1 batch follow-up, materializer/DQS1 observation harvest, queue refresh
- result: implemented
- score_claim: false
- promotion_eligible: false
- rank_or_kill_eligible: false
- ready_for_exact_eval_dispatch: false
- dispatch_attempted: false
- gpu_launched: false

## Finding

The prior frontier feedback refresh and DQS1 local-first tranche surfaces were
correct but still too manual: a partner/operator had to carry the path from
refresh report to local queue execution, then carry harvest paths into
observation JSONL, then carry the observation JSONL into the next queue. That is
exactly where materializer/receiver signal can become orphaned or stale.

This tranche adds a queue-owned feedback cycle:

1. compile frontier materializer and existing DQS1 observations into a bounded
   DQS1 follow-up queue;
2. optionally run that queue through the local DQS1 autopilot with explicit
   bounds;
3. harvest completed DQS1 local results into canonical dynamic observation rows;
4. recompile a post-harvest follow-up queue that suppresses observed candidates.

The cycle remains planning-only. Local macOS/CPU evidence cannot claim score,
promote, rank/kill, dispatch, or launch GPU work.

## Artifacts

- cycle proof:
  `.omx/research/codex_frontier_rate_attack_feedback_cycle_20260525T135319Z/frontier_rate_attack_feedback_cycle.json`
- initial follow-up queue:
  `.omx/research/codex_frontier_rate_attack_feedback_cycle_20260525T135319Z/initial_refresh/dqs1_followup_queue.json`
- initial feedback report:
  `.omx/research/codex_frontier_rate_attack_feedback_cycle_20260525T135319Z/initial_refresh/feedback_refresh_report.json`

The plan-only proof selected four DQS1 follow-up candidates and validated the
generated queue. It did not execute the queue and therefore did not emit harvest
observations in this proof run.

## Verification

- `.venv/bin/ruff check src/comma_lab/scheduler/frontier_rate_attack_feedback.py src/comma_lab/scheduler/frontier_rate_attack_feedback_cycle.py tools/build_frontier_rate_attack_feedback_refresh.py tools/run_frontier_rate_attack_feedback_cycle.py src/tac/tests/test_frontier_rate_attack_feedback.py`
- `.venv/bin/python -m pytest src/tac/tests/test_frontier_rate_attack_feedback.py -q`
- `.venv/bin/python -m pytest src/tac/tests/test_dqs1_local_first_harvest_observations.py src/tac/tests/test_dqs1_local_first_queue_builder.py src/tac/tests/test_frontier_rate_attack_feedback.py -q`
- `tools/run_frontier_rate_attack_feedback_cycle.py` plan-only proof listed above
- false-authority scan over the proof directory found no truthy score,
  promotion, rank/kill, exact-dispatch, dispatch, or GPU authority fields.

## Remaining Work

The next high-EV step is to run this cycle with `--execute-followup` under a
bounded local step budget, then let the same cycle consume its harvested
observation JSONL and emit the post-harvest queue. This is now an implementation
path, not an operator hand-carry path.
