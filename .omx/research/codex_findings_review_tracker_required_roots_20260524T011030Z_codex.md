# Codex Findings - Review Tracker Required Roots Selftest

UTC: 2026-05-24T01:10:30Z
Author: Codex
Lane: `codex_review_tracker_required_roots_20260524`

## Finding

The previous `src/comma_lab/` review-tracker coverage fix closed the immediate
gap, but the guard was still encoded as ordinary scan configuration. A future
edit could remove `src/tac/` or `src/comma_lab/` from `SCAN_PREFIXES` or the
fallback scan roots and reintroduce the same false-green class without a local
selftest failure.

## Landing

`tools/review_tracker.py` now declares required first-class source scan prefixes
and fallback roots, exposes `review_tracker_scan_scope_blockers()`, and runs the
check from both tracked-file discovery and `tools/review_tracker.py selftest`.
The scan-scope regression test asserts the required roots and the empty blocker
set directly.

## Verification

- `PYTHONPATH=. .venv/bin/python -m ruff check tools/review_tracker.py src/tac/tests/test_review_tracker_scan_scope.py`
- `PYTHONPATH=. .venv/bin/python -m pytest -q src/tac/tests/test_review_tracker_scan_scope.py`
- `PYTHONPATH=. .venv/bin/python -m py_compile tools/review_tracker.py`
- `.venv/bin/python tools/review_tracker.py selftest`

Result: focused tests and lint passed. `selftest` reported
`ALL TESTS PASSED`; it still prints the pre-existing duplicate qualified-name
warning for `tac.tests.test_dispatch_advisor::_load_advisor`.

## Remaining Work

This closes the known review-root omission class for `src/tac/` and
`src/comma_lab/`. If another first-class production root is added, it must be
added to `REQUIRED_SOURCE_SCAN_PREFIXES` and `REQUIRED_SOURCE_SCAN_ROOTS` in the
same commit that makes it authoritative.
