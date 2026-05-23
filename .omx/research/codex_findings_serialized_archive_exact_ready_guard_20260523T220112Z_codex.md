# Codex Findings: Serialized Archive Exact-Ready Guard

- timestamp_utc: 2026-05-23T22:01:12Z
- agent: codex
- lane_id: codex_serialized_archive_economics_guard_20260523
- research_only: false

## Finding

The prior serialized archive economics tranche made source/candidate archive
byte deltas explicit for queue and materializer rows, but `exact_readiness`
could still promote a byte-closed row without revalidating the
`serialized_archive_delta` contract against live archive bytes. A stale
exact-ready queue could also keep dispatch authority after the serialized
archive contract was corrupted or drifted from disk.

## Fix Landed

`tac.optimizer.exact_readiness` now validates `serialized_archive_delta`
contracts as first-class exact-ready evidence:

- schema must match `serialized_archive_delta_contract.v1`;
- source and candidate archive bytes must be present;
- declared delta, realized savings, savings-realized flag, and status must
  match recomputed values;
- modeled savings without realized serialized savings remain blocking;
- `require_realized_saving=true` blocks zero-delta and larger candidates;
- candidate archive bytes in the contract must match the live archive file.

The promoted exact-ready row preserves the source contract, and
`tac.optimizer.exact_ready_audit` revalidates that contract against current
disk state so stale exact-ready queues lose authority.

## Verification

- `.venv/bin/python -m pytest src/tac/tests/test_serialized_archive_economics.py src/tac/tests/test_byte_shaving_campaign_queue.py src/tac/tests/test_byte_range_entropy_recode_materializer.py src/tac/tests/test_inverse_scorer_cell_materializer.py src/tac/tests/test_byte_shaving_campaign.py src/tac/tests/test_optimizer_candidate_queue.py src/tac/tests/test_optimizer_exact_readiness.py src/tac/tests/test_experiment_queue.py -q`
  - 210 passed, 1 warning
- `.venv/bin/python -m ruff check src/tac/optimizer/exact_readiness.py src/tac/optimizer/exact_ready_audit.py src/tac/tests/test_optimizer_exact_readiness.py`
  - passed
- `git diff --check`
  - passed

## Remaining Wire-In

The next consumer boundary is materializer-to-candidate promotion: rows should
enter exact-ready promotion only through a helper that requires a complete
serialized archive contract, live archive custody, and local CPU/MLX advisory
evidence with explicit non-authority semantics until auth eval lands.
