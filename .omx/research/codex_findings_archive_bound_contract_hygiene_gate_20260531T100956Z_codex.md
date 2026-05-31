# Codex findings: archive-bound contract hygiene gate

Timestamp: 2026-05-31T10:09:56Z
Lane: `lane_codex_archive_bound_contract_hygiene_gate_20260531`

## Findings

- Added `tac.optimization.archive_bound_candidate_contract_audit` as the shared
  scanner for archive-like JSON/Markdown artifacts. It validates
  `tac_archive_bound_candidate_contract.v1` surfaces, reports stale duplicate
  readiness/custody fields as hard blockers, and leaves missing-contract rows as
  migration-required routing signal.
- Wired `tools/audit_archive_bound_candidate_contracts.py`, operator briefing
  Phase 6j, and all-lanes Gate 34 so archive/MLX/public-frontier emitters are
  visible from normal operator flows.
- Hardened `exact_ready_audit`: direct embedded archive-bound contracts now own
  exact-readiness/archive custody when present; loose
  `ready_for_exact_eval_dispatch` cannot overrule a stale or invalid contract.
- Fixed the `.omx/...` path-normalization bug in stale duplicate detection:
  `lstrip("./")` was incorrectly stripping the dot from `.omx` and creating
  false path mismatches against absolute repo paths.

## Verification

- `ruff` passed on touched Python surfaces.
- `pytest src/tac/tests/test_archive_bound_candidate_contract_audit.py src/tac/tests/test_optimizer_exact_ready_audit.py -q`
  passed with 37 tests.
- Tracked-file archive-contract hygiene scan over `.omx/research` and
  `experiments/results` passed with 21/21 valid contract surfaces, zero blocking
  findings, 9092 migration-required legacy rows, and one advisory read/parse
  finding.
- Operator briefing JSON smoke exposed Phase 6j as `MIGRATION_REQUIRED` with
  zero blocking findings.

## Remaining Work

The migration-required rows are intentionally not suppressed. They identify
legacy DQS1/frontier/exact-ready/public-replay signal that still needs adapter
spine conversion, but they do not block commits until a row carries a stale or
invalid shared contract.
