# Contract Migration Backlog Queue - Codex Findings

Timestamp: 2026-05-31T10:36:36Z

## Verdict

The archive-bound contract audit was strong enough to block stale authority, but the migration-required signal was still too easy to treat as a report. This slice makes that signal executable: every grouped missing-contract surface now compiles into a false-authority backlog queue row.

## Landed Surface

- `tac_archive_bound_candidate_contract_migration_backlog_queue.v1`
- `tac_archive_bound_candidate_contract_migration_backlog_row.v1`
- `tools/audit_archive_bound_candidate_contracts.py --output-backlog-json`
- `.omx/research/archive_bound_contract_migration_backlog_queue_20260531T1038Z.json`

## Current Live Audit

- Audit passed: true
- Blocking findings: 0
- Migration-required findings: 9092
- Migration backlog groups: 22
- Contract surfaces: 21/21 valid
- Advisory findings: 1 JSON parse issue in a tracked historical artifact

## Authority Boundary

Backlog rows are not score, dispatch, promotion, or budget authority. They are allowed only to route contract migration, materializer, receiver-proof, exact-blocker, or posterior-ledger repair work. Every row carries:

- shared contract required before acquisition spend
- posterior ledger required before acquisition spend
- budget spend disallowed
- exact dispatch disallowed
- score claim false

## Next Hardening Target

The first concrete follow-up is to consume these rows from the bounded runner/acquisition selector so non-contract candidate paths are automatically converted into the smallest materializer-contract task, not rediscovered through another broad audit.
