# Codex Findings: Exact Readiness Contract-First Gate

UTC: 2026-05-30T23:20Z
Scope: `src/tac/optimizer/exact_readiness.py`, `src/tac/optimizer/exact_dispatch_authority.py`, exact-readiness/dispatch tests.

## Finding

Exact-readiness already parsed `archive_bound_candidate_contract` when present, but advisory/materializer-origin rows could still present raw archive fields plus `ready_for_exact_eval_dispatch=true` without carrying the shared archive-bound candidate contract. That left an authority seam where MLX/probe/materializer rows could bypass the contract-first custody surface.

## Landing

- Added `archive_bound_contract_required_for_row(...)` to classify MLX, macOS advisory, materializer harvest, repair/probe, entropy-coder, DQS1/PR103/public-frontier, and byte-closed origins as contract-required.
- `readiness_blockers(...)` now fails closed when such rows lack the shared contract, and rejects selected contracts whose archive-bound exact-handoff booleans or file custody are incomplete.
- Promoted exact-ready rows now preserve source contract identity, schema, selected contract, and surface snapshot.
- `exact_dispatch_authority(...)` now rejects direct advisory/materializer contract-required rows and validates the promoted source-contract snapshot against live row archive SHA/bytes.

## Verification

- `.venv/bin/ruff check src/tac/optimizer/exact_readiness.py src/tac/optimizer/exact_dispatch_authority.py src/tac/tests/test_optimizer_exact_readiness.py src/tac/tests/test_exact_dispatch_authority.py`
- `.venv/bin/pytest src/tac/tests/test_optimizer_exact_readiness.py src/tac/tests/test_exact_dispatch_authority.py -q` -> 89 passed
- Recursive adversarial review bundle `2b1a5f182534f0b3`: 3 clean passes, sealed, unresolved critical count 0.

## Remaining Gap

Fermat read-only audit also found duplicate raw custody readers in `exact_ready_audit.py` and `tools/parallel_dispatch_top_k.py`. They should be migrated next to consume the same normalized contract/readiness facts instead of reinterpreting archive path/SHA/bytes independently.
