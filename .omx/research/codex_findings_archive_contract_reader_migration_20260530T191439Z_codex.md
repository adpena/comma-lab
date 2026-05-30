# Codex Findings: Archive Contract Reader Migration

## Summary

Migrated additional archive-candidate consumers onto the shared
`archive_bound_candidate_contract` reader so stale duplicate readiness and
custody fields fail closed instead of being interpreted independently.

## Landed

- `materializer_submission_closure` now reads receiver readiness and archive
  SHA/byte/path custody from the shared contract when present, and rejects stale
  duplicate fields before building a submission/runtime closure.
- `candidate_queue` now uses contract-first archive verification and promotion
  views; stale contract rows are demoted and cannot fall back to duplicate
  archive fields.
- `repair_family_stack_search` now validates archive-bound contracts before
  stack scoring and exact-handoff custody; stale contracts add acquisition
  penalty, block archive custody, and prevent exact-handoff counting.
- `exact_readiness` now resolves archive path/SHA/bytes and runtime proof path
  from the shared contract when present, while stale duplicate contract fields
  block exact-ready promotion.

## Verification

- `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest src/tac/tests/test_materializer_submission_closure.py src/tac/tests/test_optimizer_candidate_queue.py src/tac/tests/test_repair_family_materializers.py -q`
- `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest src/tac/tests/test_optimizer_exact_readiness.py -q`
- `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m ruff check src/tac/optimizer/materializer_submission_closure.py src/tac/tests/test_materializer_submission_closure.py src/tac/optimizer/candidate_queue.py src/tac/tests/test_optimizer_candidate_queue.py src/tac/optimization/repair_family_stack_search.py src/tac/tests/test_repair_family_materializers.py`
- `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m ruff check src/tac/optimizer/exact_readiness.py src/tac/tests/test_optimizer_exact_readiness.py`
- `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile src/tac/optimizer/materializer_submission_closure.py src/tac/optimizer/candidate_queue.py src/tac/optimization/repair_family_stack_search.py`
- `git diff --check -- <touched files>`

## Remaining

The next migration target is the inverse-scorer and remaining family-specific
consumers that still inspect receiver/runtime readiness directly when no
normalized archive-bound contract has been promoted into their input row.
