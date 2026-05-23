# Codex Findings: Materializer Chain Harvest

- timestamp_utc: 2026-05-23T22:16:54Z
- agent: codex
- lane_id: codex_materializer_chain_harvest_20260523
- research_only: false

## Finding

Completed byte-shaving materializer chains were producing validated archive and
artifact manifests, but the generic optimizer candidate queue did not accept
those chain schemas directly. That left a manual handoff gap between local
MLX/CPU materializer execution and the downstream exact-readiness promotion
gate.

## Fix Landed

Added `tac.optimizer.materializer_chain_harvest` as a fail-closed adapter for
`byte_range_entropy_recode_chain_v1` and
`inverse_scorer_cell_candidate_chain_v1` manifests. The adapter validates:

- candidate archive path, SHA-256, and byte count against disk;
- source archive path, SHA-256, and byte count against disk;
- canonical `serialized_archive_delta_contract.v1` against live archive bytes;
- chain artifact records and step artifact records against disk;
- chain completion booleans without treating completion as exact readiness;
- false-authority boundaries, rejecting truthy score/promotion/eval claims.

`tools/build_optimizer_candidate_queue.py` now accepts those completed chain
manifests and emits planning-only queue rows with dispatch blockers preserved.
Rows can rank and feed follow-up gates, but cannot claim score, promote, or
dispatch until the explicit exact-readiness path clears archive/runtime custody.

## Sidecar Audit

Read-only subagent Raman independently identified the same boundary: chain
completion is not exact readiness, SQLite status is not sufficient custody, and
serialized archive economics must be live-byte validated. This patch implements
the source-queue harvest layer; exact-ready promotion remains intentionally
separate.

## Verification

- `.venv/bin/python -m pytest src/tac/tests/test_optimizer_candidate_queue.py -q`
  - 31 passed
- `.venv/bin/python -m ruff check src/tac/optimizer/materializer_chain_harvest.py src/tac/optimizer/candidate_queue.py src/tac/tests/test_optimizer_candidate_queue.py tools/build_optimizer_candidate_queue.py`
  - passed

## Next Gate

Build the scheduler-level harvest CLI that scans materializer work queues and
experiment state, finds completed chain manifests, writes a source queue, and
optionally invokes the existing exact-readiness promoter under an explicit flag.
That should consume this adapter rather than bypassing it.
