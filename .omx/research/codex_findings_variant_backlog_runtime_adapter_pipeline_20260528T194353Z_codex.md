# Codex Findings: Variant Backlog Runtime Adapter Pipeline

Date: 2026-05-28T19:43:53Z
Agent: Codex

## Scope

Closed the entropy variant-signal orphan path as a pipeline contract, not a
standalone tool. Probe-only range/ANS entropy rows now open queue-owned
materializer backlog tasks that point to the smallest byte-closed prototype
candidate available for the same source archive.

## Landed

- Added `tac.optimization.repair_entropy_coder_runtime_adapters` as the reusable
  stdlib-only decode adapter surface for range/LZMA and deterministic rANS
  prototype packets.
- Moved prototype decode proof through that adapter entrypoint so receiver
  proofs exercise adapter decode instead of private local member roundtrip.
- Added `repair_archive_variant_materializer_backlog.v1` rows to the byte
  transform executor and surfaced them through stack search, interaction tensor
  features, learning signals, entropy-stage materializer work orders, chain
  execution bundles, and autonomous loop summaries.
- Updated entropy substrate coverage so range/ANS prototypes are
  `prototype_materialized_runtime_adapter_proven`; exact CPU/CUDA adjudication
  remains the fail-closed blocker.

## Evidence

- `.venv/bin/ruff check --fix ...` on touched files: passed.
- `.venv/bin/python -m py_compile ...` on touched files: passed.
- `.venv/bin/pytest src/tac/tests/test_repair_family_materializers.py -q`:
  26 passed.
- `.venv/bin/pytest src/tac/tests/test_repair_campaign_materialization_queue.py -q`:
  10 passed.
- `tools/review_tracker.py policy-check` on all touched Python files:
  0 violations after three review passes.

## Remaining Blocker

The range/ANS prototypes are now archive-bound and receiver-adapter-proven, but
they still have no score authority and no exact dispatch authority until a
contest CPU/CUDA axis payload signs the candidate. That is intentional and
fail-closed.
