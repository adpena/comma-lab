# Codex Findings: Queue Observation Recovery Feedback

timestamp_utc: 2026-05-25T03:03Z
agent: codex
research_only: false
score_claim: false
promotion_eligible: false
rank_or_kill_eligible: false
ready_for_exact_eval_dispatch: false
allowed_use: local_queue_feedback_and_inverse_steganalysis_planning_only

## Finding

Queue observation health was still too easy to strand as passive telemetry. The
materializer campaign runner now writes both `queue_observation.json` and a
typed `queue_observation_recovery_plan.v1` artifact. The feedback policy consumes
that plan directly, prioritizes required queue recovery before new acquisition
work, and preserves nonblocking orphan maintenance as advisory signal.

The inverse-steganalysis action-functional handoff now receives queue
observations through `--queue-observation`, so queue health can suppress or steer
the next local acquisition pass when the health row is attributable to a
candidate. Nonblocking historical orphan rows are intentionally not converted
into candidate blockers; they stay in the recovery plan instead. Required queue
recovery blocks queue-owned follow-up execution so the system does not build a
new water-bucket pass on an unhealthy execution proof.

## Engineering Change

- Added `build_queue_observation_recovery_plan(...)` in the feedback replan
  policy layer.
- Added queue observation and recovery-plan artifacts to
  `tools/run_byte_shaving_materializer_campaign.py`.
- Wired queue observation paths into the feedback action-functional command and
  child queue input custody.
- Added `observations_from_queue_observation(...)` consumption in
  `tools/build_inverse_steganalysis_action_functional.py`.
- Exported the new scheduler and inverse-acquisition helpers through the package
  surfaces that existing queue/autopilot code imports.
- Surfaced queue-observation recovery and maintenance counts in
  `tools/operator_briefing.py`.
- Pinned operator-briefing queue commands to their run-scoped `--state` path so
  copied recovery/worker commands do not fall through to default SQLite state.
- Hardened inverse-acquisition queue health handling so blocking queue health
  can suppress candidate cells while nonblocking orphan maintenance does not
  poison unrelated candidates.
- Added explicit follow-up refusal when `queue_observation_recovery_required` is
  true, and made required queue-health recovery non-autopilotable because it can
  emit state-mutating `init`, `rewind`, or `retire-orphans` commands.

## Verification

- `.venv/bin/python -m pytest src/tac/tests/test_inverse_steganalysis_acquisition.py src/tac/tests/test_inverse_steganalysis_action_functional_cli.py src/tac/tests/test_queue_feedback_replan_policy.py src/tac/tests/test_byte_shaving_materializer_campaign_runner.py src/tac/tests/test_operator_briefing.py -q` (158 passed)
- `.venv/bin/python -m ruff check src/comma_lab/scheduler/__init__.py src/comma_lab/scheduler/queue_feedback_replan_policy.py src/tac/optimization/__init__.py src/tac/optimization/inverse_steganalysis_acquisition.py src/tac/tests/test_inverse_steganalysis_acquisition.py src/tac/tests/test_inverse_steganalysis_action_functional_cli.py src/tac/tests/test_queue_feedback_replan_policy.py src/tac/tests/test_byte_shaving_materializer_campaign_runner.py src/tac/tests/test_operator_briefing.py tools/build_inverse_steganalysis_action_functional.py tools/run_byte_shaving_materializer_campaign.py tools/operator_briefing.py`
- `.venv/bin/python - <<'PY'` import probe for `comma_lab.scheduler.build_queue_observation_recovery_plan`, `QUEUE_OBSERVATION_RECOVERY_PLAN_SCHEMA`, and `tac.optimization.observations_from_queue_observation`
- `git diff --check`

## Next Integration Hooks

- Feed `queue_observation_recovery_plan.v1` rows into the campaign planner UI and
  queue observation dashboard so required recovery has first-class operator
  visibility.
- Add a grouped-acquisition rule that treats repeated queue recovery blockers as
  a negative prior for the operation set, not just for the individual candidate.
- Extend the same recovery-plan contract to the next materializer families:
  `packet_member_recompress_v1`, `tensor_factorize_v1`, and PR95/HNeRV archive
  reproduction lanes.
