# Codex Findings - Post-Recovery Feedback Replan Continuation

generated_at_utc: 2026-05-25T04:21:52Z
lane_id: codex_post_recovery_replan_continuation_20260525
research_only: false

## Finding

Queue-observation recovery could successfully repair the source queue and emit a
fresh `source_observation_after`, but the materializer campaign runner still
planned continuation from the stale pre-recovery policy. That left the most
important recovery signal passive until another operator or agent turn consumed
it.

## Landing

- Added post-recovery feedback replan artifacts in
  `tools/run_byte_shaving_materializer_campaign.py`.
- The runner now writes after-recovery observation, recovery-plan, feedback
  request, followup queue, policy, and continuation queue artifacts when queue
  recovery succeeds.
- Post-recovery feedback uses a distinct
  `inverse_steganalysis_action_functional.after_recovery.*` output stem so it
  cannot overwrite the pre-recovery feedback artifact.
- Top-level run summaries now expose post-recovery attempted/triggered/artifact,
  execution, state-path, policy, continuation, and blocker fields.
- `tools/operator_briefing.py` now counts and routes post-recovery feedback
  followup and continuation queues with explicit queue-state paths.

## Safeguards

- All post-recovery artifacts preserve false-authority fields:
  `score_claim=false`, `promotion_eligible=false`, `rank_or_kill_eligible=false`,
  and `ready_for_exact_eval_dispatch=false`.
- Post-recovery replan triggers only after
  `queue_observation_recovery_execution.success == true` and a mapping-shaped
  `source_observation_after` exists.
- `success` now means the post-recovery followup executed and emitted a
  continuation queue. Merely emitting artifacts is tracked separately as
  `artifacts_emitted`.
- Followup and continuation queues remain local paused queues. Operator next
  commands include explicit queue state paths so work is not silently orphaned
  into a default SQLite file.

## Verification

- `.venv/bin/python -m ruff check tools/run_byte_shaving_materializer_campaign.py tools/operator_briefing.py src/tac/tests/test_byte_shaving_materializer_campaign_runner.py src/tac/tests/test_operator_briefing.py`
- `.venv/bin/python -m pytest src/tac/tests/test_byte_shaving_materializer_campaign_runner.py -q` -> 57 passed
- `.venv/bin/python -m pytest src/tac/tests/test_operator_briefing.py -q` -> 33 passed

## Remaining Work

- Recursively executing the post-recovery continuation queue is intentionally not
  enabled here. The next actuator should consume the paused continuation queue
  through the normal queue/DAG path.
- The same post-recovery feedback pattern should be reused for other local
  recovery lanes that produce fresh canonical observations.
