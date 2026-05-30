# Codex Findings: Repair Exact Bridge Contract Reader

UTC: 2026-05-30T18:50:06Z

## Landing

Wired `repair_family_exact_ready_bridge` through the shared archive-bound
candidate contract reader when repair handoff rows carry an embedded contract
or contract surface. Stale duplicate receiver/archive fields now fail before
the bridge emits a source optimizer queue or blocked exact-ready queue.

## Verification

- `ruff check` on touched repair bridge files: passed.
- `py_compile` on touched repair bridge files: passed.
- `pytest src/tac/tests/test_repair_family_materializers.py -q`: `28 passed`.
- Combined candidate/portfolio/repair regression:
  `test_archive_bound_candidate_adapter_spine.py`,
  `test_cross_family_candidate_portfolio.py`,
  `test_optimizer_candidate_queue.py`,
  `test_repair_family_materializers.py`: `91 passed`.
- `git diff --check`: passed.

## Remaining Pressure

The next contract-reader migration targets are exact-ready closure and
materializer submission closure paths that still inspect raw readiness fields
before consulting the canonical contract.
