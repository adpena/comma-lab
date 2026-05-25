# Codex findings: materializer harvest state provenance

UTC: 2026-05-25T02:30:55Z

## Scope

This pass hardens the materializer-chain harvest boundary so a state-backed
harvest cannot accept explicitly supplied or root-scanned manifests that lack
queue work identity. The experiment queue remains the authority for executed
materializer work; direct manifest paths stay useful only when no state filter
is being used.

## Finding

- `harvest_materializer_chain_manifests()` accepted `chain_manifest_paths`
  while also receiving an experiment queue state DB. Those explicit discoveries
  had no `work_id`, so the succeeded-state filter silently skipped them and a
  manifest outside queue ownership could enter the planning source queue.
- The same missing-identity class applies to root-scanned manifests and any
  work-queue postcondition row without a durable `work_id` when state filtering
  is active.

## Landed integration

- `_state_blockers()` now receives `state_filter_active`; when
  `require_succeeded_state` is true and a state DB was provided, missing
  `work_id` fails closed with
  `experiment_queue_state_work_id_missing_for_manifest`.
- Added a regression for explicit manifest harvest with state supplied and no
  work provenance.
- Registered lane `codex_materializer_harvest_state_provenance_20260525` and
  marked implementation plus strict preflight evidence.

## Verification

- `.venv/bin/python -m pytest src/tac/tests/test_materializer_chain_harvest_scheduler.py -q`
  passed: 41 tests.
- `.venv/bin/python -m ruff check src/comma_lab/scheduler/materializer_chain_harvest.py src/tac/tests/test_materializer_chain_harvest_scheduler.py`
  passed.

No score claim, rank/kill decision, promotion authority, or dispatch authority
is created here. This only tightens queue-owned harvest provenance.
