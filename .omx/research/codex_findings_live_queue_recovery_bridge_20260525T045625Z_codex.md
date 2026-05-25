# Codex Findings: Live Queue Recovery Bridge

- utc: 2026-05-25T04:56:25Z
- lane_id: codex_live_queue_recovery_bridge_20260525
- author: codex
- research_only: false

## Finding

The operator briefing now trusted live byte-shaving queue health, but the failed
campaign still stopped at an observe-only hint. That preserved truthfulness but
left the queue-health signal outside the canonical recovery/replan artifacts for
older campaign runs that were created before queue observation recovery existed.

## Landing

Added `tools/materialize_byte_shaving_queue_recovery.py`, a local-only
materializer for legacy byte-shaving campaign runs. Given a
`materializer_campaign_run.json`, it:

- live-observes the referenced `experiment_queue` with orphan identities;
- builds the canonical `queue_observation_recovery_plan.v1`;
- builds the canonical `queue_feedback_replan_policy.v1`;
- emits a paused local `experiment_queue.v1` recovery queue when required;
- writes a no-score-authority materialization summary;
- never mutates the source queue state unless the emitted recovery queue is
  separately resumed and executed.

`tools/operator_briefing.py` now derives a live recovery plan from live queue
blockers and routes missing recovery artifacts to:

```text
.venv/bin/python tools/materialize_byte_shaving_queue_recovery.py --run-summary <materializer_campaign_run.json> --write
```

Once those artifacts exist, Phase 6c advances to the canonical recovery queue
`init` command instead of repeating the materialization command.

## Real Artifact Advanced

For
`.omx/research/high_level_byte_shaving_runner_smoke_20260524T050723Z/campaign3/materializer_campaign_run.json`,
the bridge materialized:

- `queue_observation.json`
- `queue_observation_recovery_plan.json`
- `queue_feedback_replan_policy.json`
- `queue_observation_recovery_queue.json`
- `queue_observation_recovery_materialization.json`

The materialization summary reports:

- observation healthy: `false`
- blocker: `experiment_queue_observation_failed_steps:1`
- recovery required: `true`
- recovery action count: `1`
- policy decision: `recover_queue_health`
- recovery queue emitted: `true`
- score authority: `false`
- exact-eval dispatch authority: `false`

The recovery queue was initialized locally once and observed healthy/paused with
one queued recovery action; the transient SQLite state was not committed because
this repo does not track runtime `.sqlite` state files.

## Tests

- `.venv/bin/python -m py_compile tools/operator_briefing.py tools/materialize_byte_shaving_queue_recovery.py src/tac/tests/test_operator_briefing.py`
- `.venv/bin/python -m pytest src/tac/tests/test_operator_briefing.py -k "byte_shaving_acquisition_summary or materialize_byte_shaving_queue_recovery"`
  - result: `8 passed, 29 deselected`
- `.venv/bin/python -m pytest src/tac/tests/test_operator_briefing.py`
  - result: `37 passed in 132.64s`
- `.venv/bin/python tools/operator_briefing.py --json`
  - result: Phase 6c now detects emitted recovery artifacts and recommends the paused recovery queue `init` command

## Residual Work

The remaining source failure is not a scheduler-health issue: the failed source
step hit an invalid ZIP template. Rewinding the failed source step would only
make it runnable again; the next productive step is to feed the recovered queue
health plus invalid-template blocker into the post-recovery feedback/replan path
so the next campaign iteration avoids rematerializing the same invalid leaf.
