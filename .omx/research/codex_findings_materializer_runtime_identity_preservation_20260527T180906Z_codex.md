# Materializer Runtime Identity Preservation

UTC: 2026-05-27T18:09:06Z

## Finding

The materializer chain harvest path could bind receiver/runtime paths from a work
queue row into the harvested source queue without also carrying the corresponding
runtime tree identity. That left a signal-loss gap: downstream archive and exact
eval gates could see a path-level receiver closure but not the runtime hash that
made the closure reproducible.

## Landing

The harvest overlay now imports canonical runtime tree hash fields, extracts
hashes from top-level and nested runtime identity records, and applies those
hashes to source queue rows when absent. If an existing row already carries a
different runtime tree hash, the harvester records an explicit identity blocker
and forces the row fail-closed for runtime adapter readiness, receiver contract
satisfaction, and exact-eval dispatch.

## Authority

This is a custody and reproducibility guard only. It does not claim score,
promote a candidate, rank or kill a lane, launch GPU work, or mark any row ready
for exact eval dispatch.

## Verification

- `.venv/bin/python -m ruff check src/comma_lab/scheduler/materializer_chain_harvest.py src/tac/tests/test_materializer_chain_harvest_scheduler.py`
- `.venv/bin/python -m py_compile src/comma_lab/scheduler/materializer_chain_harvest.py src/tac/tests/test_materializer_chain_harvest_scheduler.py`
- `.venv/bin/python -m pytest src/tac/tests/test_materializer_chain_harvest_scheduler.py -q`
- `.venv/bin/python -m ruff check src/tac/optimization/repair_campaign_learning_signal.py src/tac/optimization/repair_campaign_posterior.py src/comma_lab/scheduler/repair_campaign_score_queue.py tools/build_repair_campaign_blocked_learning_signals.py tools/append_repair_campaign_blocked_posterior.py src/tac/tests/test_repair_campaign_score_queue.py`
- `.venv/bin/python -m pytest src/tac/tests/test_repair_campaign_score_queue.py src/tac/tests/test_repair_campaign_stackability_queue.py -q`
