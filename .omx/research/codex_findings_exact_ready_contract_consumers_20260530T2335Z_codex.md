# Codex Findings: Exact-Ready Contract Consumers

UTC: 2026-05-30T23:35Z
Scope: `src/tac/optimizer/exact_ready_audit.py`, `tools/parallel_dispatch_top_k.py`, exact-ready/dispatch tests.

## Finding

After exact-readiness promotion became contract-first, downstream consumers still had raw archive readers. `exact_ready_audit.py` and `parallel_dispatch_top_k.py` could miss custody supplied by the promoted source-contract snapshot, or reinterpret stale raw path/SHA/byte fields separately from the shared contract.

## Landing

- Exact-readiness can now consume `source_archive_bound_candidate_contract` as a selected archive-bound contract snapshot.
- Exact-dispatch source snapshot validation treats the contract as authority: raw duplicate SHA/byte fields are checked when present, but are no longer required when the contract snapshot is complete.
- Exact-ready audit resolves archive path/SHA/bytes from the promoted source-contract snapshot before declaring custody missing.
- Parallel dispatch resolves archive path/SHA/bytes and detached queue repo-root selection from the same source-contract snapshot, so contract-backed rows do not fall back to stale raw-only helpers.

## Verification

- `.venv/bin/ruff check src/tac/optimizer/exact_readiness.py src/tac/optimizer/exact_dispatch_authority.py src/tac/optimizer/exact_ready_audit.py tools/parallel_dispatch_top_k.py src/tac/tests/test_optimizer_exact_readiness.py src/tac/tests/test_dispatch_command_builder_shapes.py`
- `.venv/bin/pytest src/tac/tests/test_optimizer_exact_readiness.py src/tac/tests/test_exact_dispatch_authority.py src/tac/tests/test_optimizer_exact_ready_audit.py src/tac/tests/test_dispatch_command_builder_shapes.py -q` -> 159 passed

## Remaining Gap

The same contract-first migration should be swept across any remaining exact-ready queue viewers and suppression/repair helpers so they read one custody schema instead of duplicating archive readiness interpretation.
