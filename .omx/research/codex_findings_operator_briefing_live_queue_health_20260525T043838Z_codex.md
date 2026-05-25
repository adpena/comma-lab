# Codex Findings: Operator Briefing Live Queue Health

- utc: 2026-05-25T04:38:38Z
- lane_id: codex_operator_briefing_live_queue_health_20260525
- author: codex
- research_only: false

## Finding

The high-level byte-shaving briefing could report an old materializer campaign
as `READY_LOCAL_QUEUE` from its saved `materializer_campaign_run.json` even when
the current `experiment_queue` SQLite state had already failed. This orphaned
the live failure signal and could route the operator back into
`run-worker --execute` against a queue with no ready steps.

Current live proof after the patch:

- queue:
  `.omx/research/high_level_byte_shaving_runner_smoke_20260524T050723Z/campaign3/materializer_execution_queue.json`
- state:
  `.omx/state/experiment_queue_high_level_byte_shaving_runner_smoke_20260524T050723Z_campaign3.sqlite`
- status: `BLOCKED`
- live blocker: `experiment_queue_observation_failed_steps:1`
- failed step count: `1`
- ready step count: `0`
- next command now points to `observe --tail-lines 20`, not `run-worker --execute`

## Landing

`tools/operator_briefing.py` now direct-imports the canonical read-only queue
observer for byte-shaving materializer campaign rows. For any row with a
`queue_path`, live queue observation wins over embedded/stale observation JSON.
The row records the observation source, live health, mode, blockers, status
counts, queue SHA, and state watermark while preserving the false-authority
contract.

Additional guard: a live queue whose mode is not `running` blocks worker command
routing even if the observer is otherwise healthy. Missing queue definitions,
missing state, failed steps, drift, orphan counts, and observer failures remain
visible as blockers rather than being flattened into stale readiness.

## Tests

- `.venv/bin/python -m py_compile tools/operator_briefing.py src/tac/tests/test_operator_briefing.py`
- `.venv/bin/python -m pytest src/tac/tests/test_operator_briefing.py -k "byte_shaving_acquisition_summary"`
  - result: `7 passed, 29 deselected`
- `.venv/bin/python -m pytest src/tac/tests/test_operator_briefing.py`
  - result: `36 passed in 129.35s`
- `.venv/bin/python tools/operator_briefing.py --json`
  - live Phase 6c summary now reports the current campaign as `BLOCKED`

## Residual Work

This fixes briefing truthfulness and command routing. The next automation step is
to feed the live queue blockers into the existing queue-observation recovery and
post-recovery feedback replan loop so stale campaign rows automatically emit the
next recovery/replan queue instead of stopping at an observe-only operator hint.
