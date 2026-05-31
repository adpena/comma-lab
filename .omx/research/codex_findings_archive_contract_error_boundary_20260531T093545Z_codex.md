# Codex Findings: Archive Contract Error Boundary

## Verdict

The shared archive-bound candidate contract reader already performed stale-field
and false-authority checks, but truthy authority failures could escape as raw
`ValueError` from `require_no_truthy_authority_fields`. Several consumers catch
`ArchiveBoundCandidateContractError` to fail closed, so the shared helper needed
to normalize that error boundary once instead of patching every consumer.

## Landing

- `tac.optimization.archive_bound_candidate_contract` now wraps truthy authority
  violations as `ArchiveBoundCandidateContractError`.
- Contract extraction, embedded-contract validation, contract construction, and
  contract-surface construction all use the same error boundary.
- Existing consumers such as inverse-scorer exact queues, dynamic sparse gate
  oracle, exact-readiness bridge, materializer submission closure, DQS1 feedback
  bridge, repair stack search, byte-shaving signal surface, and cross-family
  portfolio inherit the fail-closed contract error type.

## Regression Coverage

`src/tac/tests/test_archive_bound_candidate_adapter_spine.py` now verifies that a
row carrying a valid shared contract but a truthy raw
`ready_for_exact_eval_dispatch` duplicate raises `ArchiveBoundCandidateContractError`
instead of leaking a generic `ValueError`.

## Verification

- `.venv/bin/python -m ruff check src/tac/optimization/archive_bound_candidate_contract.py src/tac/tests/test_archive_bound_candidate_adapter_spine.py`
- `PYTHONPATH=.:src .venv/bin/pytest src/tac/tests/test_archive_bound_candidate_adapter_spine.py src/tac/tests/test_experiment_queue_observer_contract_readiness.py -q`
- `PYTHONPATH=.:src .venv/bin/pytest src/tac/tests/test_inverse_scorer_exact_eval_queue.py src/tac/tests/test_dynamic_sparse_gate_oracle.py src/tac/tests/test_optimizer_exact_readiness.py -q`
- `PYTHONPATH=.:src .venv/bin/pytest src/tac/tests/test_optimizer_candidate_queue.py -q`

## Remaining Work

The next highest-leverage pass is to migrate any emitter that still produces
family-specific archive/readiness fields without embedding
`tac_archive_bound_candidate_contract.v1`, especially older MLX substrate
trainers and legacy DQS1/public-frontier sidecar outputs.
