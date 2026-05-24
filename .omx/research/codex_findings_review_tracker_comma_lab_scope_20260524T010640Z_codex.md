# Codex Findings - Review Tracker comma_lab Scope Follow-Up

UTC: 2026-05-24T01:06:40Z
Author: Codex
Lane: `codex_review_tracker_comma_lab_scope_20260524`

## Finding

The review gate tracks staged `src/comma_lab/` Python files, but the review
tracker scan roots did not include `src/comma_lab/`. Queue and scheduler
authority code could therefore be staged under a policy surface that expected
review evidence while the tracker had no entities for those files.

## Landing

`tools/review_tracker.py` now includes `src/comma_lab/` in its canonical scan
prefixes and filesystem fallback roots. The scan-scope regression test asserts
that `src/comma_lab/scheduler/*` files are reviewable and appear in deduped
git-output ordering.

The artifact-mobility scheduler remains in `src/comma_lab/scheduler/` because
it owns queue state, lane custody, dispatch planning, and SSH claim/finalize
semantics. A future `tac.deploy` extraction should be limited to reusable pure
artifact transfer helpers once a second provider consumer exists.

## Verification

- `.venv/bin/python -m ruff check tools/review_tracker.py src/tac/tests/test_review_tracker_scan_scope.py`
- `.venv/bin/python -m pytest src/tac/tests/test_review_tracker_scan_scope.py -q`
- `.venv/bin/python -m py_compile tools/review_tracker.py`
- `.venv/bin/python tools/review_tracker.py scan`
- `.venv/bin/python tools/review_tracker.py policy-check src/comma_lab/scheduler/ssh_experiment_queue_executor.py`
- `.venv/bin/python tools/review_tracker.py policy-check src/tac/tests/test_review_tracker_scan_scope.py`

Result: all commands passed. The scan reported `76902 entities across 5155
files`; the SSH scheduler executor policy check reported `36 entities
compliant, 0 violations`.
