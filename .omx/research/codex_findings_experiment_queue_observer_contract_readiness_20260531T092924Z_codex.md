# Codex Findings: Experiment Queue Observer Contract Readiness

## Verdict

The experiment queue observer had a remaining duplicate-readiness path: materializer
rows could carry the shared archive-bound candidate contract while observer-derived
rows still copied raw `ready_for_exact_eval_dispatch` from the source row. That
preserved a false-authority hazard for stale rows and weakened the contract-first
migration.

## Landing

- `src/comma_lab/scheduler/experiment_queue_observer.py` now resolves the selected
  `tac_archive_bound_candidate_contract.v1` payload before observer materializer
  readiness projection.
- Stale or truthy duplicate authority fields beside the contract are surfaced as
  `archive_bound_candidate_contract_invalid:*` blockers.
- Observer materializer rows copy the shared contract payload through and derive
  `ready_for_exact_eval_dispatch` from the validated contract only. Invalid contract
  payloads fail closed to `False`.
- Deferred-runtime-identity allowance now refuses rows whose shared contract is
  stale or invalid, instead of trusting legacy raw readiness fields.
- Materializer payload revalidation can validate candidate archive custody from the
  shared contract itself, so contract-first rows do not depend on duplicate archive
  fields.

## Regression Coverage

`src/tac/tests/test_experiment_queue_observer_contract_readiness.py` creates real
candidate/source archive bytes, hashes them into the shared contract, then verifies:

- valid contract rows still allow expected deferred runtime identity blockers;
- truthy raw `ready_for_exact_eval_dispatch` beside the contract is rejected and not
  propagated;
- stale duplicate `archive_bound_candidate_ready` beside the contract becomes a
  hard contract blocker in both observer row projection and materializer payload
  revalidation.

## Verification

- `.venv/bin/python -m ruff check src/comma_lab/scheduler/experiment_queue_observer.py src/tac/tests/test_experiment_queue_observer_contract_readiness.py`
- `PYTHONPATH=.:src .venv/bin/pytest src/tac/tests/test_experiment_queue_observer.py src/tac/tests/test_experiment_queue_observer_contract_readiness.py -q`
- `git diff --check -- src/comma_lab/scheduler/experiment_queue_observer.py src/tac/tests/test_experiment_queue_observer_contract_readiness.py`

## Remaining Work

Next contract-reader migrations should target inverse-scorer/dynamic sparse gate
exact queues and any remaining candidate-promotion views that still interpret
legacy readiness fields without calling the shared archive-bound contract helper.
