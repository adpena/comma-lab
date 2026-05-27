# Codex Findings: Repair Posterior Follow-Up Queue

UTC: 2026-05-27T20:42:14Z
Agent: Codex

## Verdict

Posterior acquisition routes are now executable queue intent, not only report
metadata. The feedback refresh emits a dedicated
`repair_posterior_acquisition_followup_queue.json` that maps learned repair
posterior policies onto bounded local child-queue actions.

This remains false-authority. The new queue can only run local bounded
follow-up work and cannot claim score, spend budget, promote, rank/kill, or
dispatch exact eval.

## What Changed

- Added `build_repair_posterior_acquisition_followup_queue(...)`.
- Added summary helpers for repair score queues and posterior follow-up queues
  so the CLI and reusable cycle do not hand-roll divergent report contracts.
- Mapped posterior policies to executable queue surfaces:
  targeted response harvest -> targeted correction queue;
  receiver-closed byte credit -> receiver repair queue;
  missing repair artifacts -> repair budget waterfill queue;
  missing local MLX custody -> repair campaign score queue;
  exact-auth handoff remains blocked until contest-axis authority exists.
- Wired the follow-up queue into standalone feedback refresh and reusable
  feedback-cycle artifacts.
- Added operator commands for validate/init/bounded-local run.
- Added the follow-up queue to post-feedback child-queue selection priority
  immediately after repair campaign scoring.
- Added the autonomous many-op parent action so the follow-up queue participates
  in chain-owned child queue execution when present.

## Mathematical Role

This is the first explicit bridge from the posterior action-functional state
back into queue actuation:

`posterior route -> evidence surface -> executable queue -> bounded local work`.

That closes one loop in the multiscale stack: pixel/region/frame/pair response
signals and blocked artifacts become posterior rows; posterior rows become
acquisition routes; routes now select the next local evidence-producing queue.

## Verification

- `.venv/bin/ruff check src/comma_lab/scheduler/repair_campaign_score_queue.py src/comma_lab/scheduler/frontier_rate_attack_feedback_cycle.py tools/build_frontier_rate_attack_feedback_refresh.py src/comma_lab/scheduler/frontier_final_rate_attack_autoloop.py src/tac/tests/test_repair_campaign_score_queue.py src/tac/tests/test_frontier_rate_attack_bootstrap.py`
- `.venv/bin/python -m py_compile src/comma_lab/scheduler/repair_campaign_score_queue.py src/comma_lab/scheduler/frontier_rate_attack_feedback_cycle.py tools/build_frontier_rate_attack_feedback_refresh.py src/comma_lab/scheduler/frontier_final_rate_attack_autoloop.py`
- `.venv/bin/python -m pytest src/tac/tests/test_repair_campaign_score_queue.py -q`
- `.venv/bin/python -m pytest src/tac/tests/test_frontier_rate_attack_bootstrap.py -q`
- `.venv/bin/python -m pytest src/tac/tests/test_frontier_rate_attack_feedback.py -q`
- `.venv/bin/python tools/lane_maturity.py validate`
- Review policy checks on touched source/test/tool files: 0 violations, excluding the large frontier feedback test file which was run for evidence but left unstaged because of pre-existing unrelated review debt.

## Remaining Work

The next concrete actuator is to run a bounded local feedback refresh smoke and
confirm the new follow-up queue appears in the emitted artifact map, validates,
and either executes or freezes with precise missing-evidence blockers.
