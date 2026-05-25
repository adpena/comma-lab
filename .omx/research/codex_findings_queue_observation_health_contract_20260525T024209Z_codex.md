# Codex Findings: Queue Observation Health Contract

- UTC: 2026-05-25T02:42:09Z
- Lane: `codex_queue_observation_health_contract_20260525`
- Scope: `experiment_queue.v1` observation, materializer exact-readiness harvest custody, and byte-shaving campaign runner artifacts.
- Authority: infrastructure signal only. No score claim, promotion claim, rank/kill claim, or exact-eval readiness claim is made here.

## Findings

1. `observe` was still operator-facing prose/JSON without a typed health verdict. Downstream planners could count rows manually, but did not have a stable `healthy=false`/`blockers=[...]` contract for missing state, definition drift, orphan rows, failed steps, or stale/corrupt artifacts.
2. `tools/run_byte_shaving_materializer_campaign.py` recorded the final observation without `--include-orphans`, so the run artifact could preserve an orphan count while dropping the actual orphan step identities.
3. The exact-readiness follow-up harvest path had a state-provenance bug class: generated harvest steps could read the canonical default queue state even when the campaign runner was executing against an isolated `--queue-state`. That causes accepted materializer output to lose scheduler provenance and fail closed during harvest.
4. Harvest state filtering accepted ambiguous state arguments too easily: a supplied state path without an explicit queue id could silently broaden provenance interpretation.
5. Sidecar adversarial review found two observer false-green cases: `blocked` step rows were omitted from observed health, and `succeeded` rows whose required artifacts were later missing/corrupt were not rechecked.
6. Sidecar adversarial review also found that the runner test proved the `--include-orphans` flag but not retention of real orphan identities in the saved run artifact.

## Landed Fixes

- `src/comma_lab/scheduler/experiment_queue_observer.py` now emits `healthy`, `blockers`, and `blocker_count`.
- The observer now treats `blocked` rows as health blockers and rechecks succeeded rows for missing/corrupt declared artifacts without tailing logs for every succeeded row.
- `tools/run_byte_shaving_materializer_campaign.py` now calls queue observation with `--include-orphans`.
- `src/comma_lab/scheduler/byte_shaving_campaign_queue.py` now threads both `source_work_queue_path` and `source_state_path` into generated exact-readiness harvest steps and records both in experiment metadata.
- `tools/build_byte_shaving_campaign_queue.py` exposes `--materializer-execution-state`; the campaign runner passes its actual execution state path through that flag.
- `src/comma_lab/scheduler/materializer_chain_harvest.py` and `tools/harvest_materializer_chain_candidates.py` now require an explicit queue id whenever state filtering is supplied.
- The campaign runner E2E now seeds a nonblocking stale row and asserts `observation.orphaned_steps` carries the stale row identity and blocker into `materializer_campaign_run.json`.

## Verification

- `.venv/bin/python -m ruff check src/comma_lab/scheduler/byte_shaving_campaign_queue.py src/comma_lab/scheduler/experiment_queue_observer.py src/comma_lab/scheduler/materializer_chain_harvest.py tools/build_byte_shaving_campaign_queue.py tools/harvest_materializer_chain_candidates.py tools/run_byte_shaving_materializer_campaign.py src/tac/tests/test_byte_shaving_campaign_queue.py src/tac/tests/test_byte_shaving_materializer_campaign_runner.py src/tac/tests/test_experiment_queue_observer.py src/tac/tests/test_materializer_chain_harvest_scheduler.py`
- `PYTHONPATH=. .venv/bin/pytest src/tac/tests/test_byte_shaving_campaign_queue.py src/tac/tests/test_materializer_chain_harvest_scheduler.py src/tac/tests/test_byte_shaving_materializer_campaign_runner.py src/tac/tests/test_experiment_queue_observer.py -q`
- `git diff --check`

## Outstanding Work

- Feed `observation.blockers` into queue feedback replanning and autopilot policy selection so follow-up queues can automatically prefer rewind/retire/rebuild actions over blind reruns.
- Add a queue-dashboard summary that groups blocker families across active campaigns, with orphan details visible by default.
- Extend the same typed-health contract to Dask/SSH executors once local queue authority remains green for another pass.
