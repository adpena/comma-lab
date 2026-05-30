# Codex Findings: Archive Contract Reader Fail-Closed

UTC: 2026-05-30T18:45:57Z

## Landing

Centralized archive-bound candidate contract consumption in
`tac.optimization.archive_bound_candidate_contract`:

- added one adapter-package schema constant shared by adapter and consumers;
- added `archive_bound_candidate_contracts_from_payload(...)` for contract,
  surface, adapter-package, and legacy row payloads;
- added stale duplicate-field blockers for readiness booleans and archive
  byte/SHA custody fields;
- moved cross-family portfolio archive contract parsing through the shared
  reader so portfolio acquisition cannot silently choose stale raw fields over
  the canonical contract.

## Authority Boundary

The new reader preserves false-authority semantics with
`require_no_truthy_authority_fields(...)` and rejects mismatched duplicate
readiness/custody fields before acquisition routing. MLX/proxy/advisory rows
remain planning signal only; exact dispatch still requires downstream
contest-axis authority.

## Verification

- `ruff check` on touched Python files: passed.
- `py_compile` on touched Python files: passed.
- Focused portfolio/adapter tests: `24 passed`.
- Broader candidate/portfolio/repair regression:
  `test_archive_bound_candidate_adapter_spine.py`,
  `test_cross_family_candidate_portfolio.py`,
  `test_optimizer_candidate_queue.py`,
  `test_repair_family_materializers.py`: `90 passed`.
- `git diff --check`: passed.
- `review_tracker.py mark-file ... --status reviewed` on touched Python files:
  completed.

Repo-wide `review_tracker.py policy-check` remains dominated by pre-existing
unreviewed entities outside this slice (`14817` reported violations), so it is
not a clean current-slice signal.

## Next Integration Pressure

Remaining high-EV cleanup is to migrate remaining consumers that still inspect
duplicate readiness fields directly, especially exact-ready bridge/materializer
closure paths, onto the shared contract reader and stale-field guard.
