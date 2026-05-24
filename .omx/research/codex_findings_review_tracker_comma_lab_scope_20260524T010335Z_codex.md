# Codex Findings - Review Tracker comma_lab Scan Coverage

UTC: 2026-05-24T01:03:35Z
Author: Codex
Lane: `codex_review_tracker_comma_lab_scope_20260524`

## Finding

The review tracker scan surface included `src/tac`, `experiments`, `tools`,
`submissions`, and `scripts`, but not `src/comma_lab`. That created a meta-bug:
commits touching scheduler infrastructure under `src/comma_lab` could rely on
tests and tool wrappers being reviewed while the production scheduler modules
were not guaranteed to be present in the tracker scan scope.

## Landing

`tools/review_tracker.py` now treats `src/comma_lab/` as reviewable Python and
walks `src/comma_lab` during tracked file discovery. The scan-scope tests assert
both path classification and deterministic tracked-file filtering for
`src/comma_lab/scheduler/*`.

While touching the guardrail, the review tracker was also cleaned so the focused
ruff check passes: explicit `zip(..., strict=False)`, `datetime.UTC`, no
semicolon-packed assignments, no unused loop variables, and no nested-if
SIM102 lint issue in greenup matching.

The artifact-mobility scheduler remains in `src/comma_lab/scheduler/` because
it owns queue state, lane custody, dispatch planning, and SSH claim/finalize
semantics. A future `tac.deploy` extraction should be limited to reusable pure
artifact transfer helpers once a second provider consumer exists.

## Verification

- `PYTHONPATH=. .venv/bin/python -m ruff check tools/review_tracker.py src/tac/tests/test_review_tracker_scan_scope.py`
- `PYTHONPATH=. .venv/bin/python -m pytest -q src/tac/tests/test_review_tracker_scan_scope.py`
- `PYTHONPATH=. .venv/bin/python -m py_compile tools/review_tracker.py`
- `.venv/bin/python tools/review_tracker.py scan`
- `.venv/bin/python tools/review_tracker.py policy-check src/comma_lab/scheduler/ssh_experiment_queue_executor.py`
- `.venv/bin/python tools/review_tracker.py policy-check src/tac/tests/test_review_tracker_scan_scope.py`

Result: scan reported `76902 entities across 5155 files`; the SSH scheduler
executor policy check reported `36 entities compliant, 0 violations`.

## Remaining Work

1. Audit whether any other first-class production roots outside `src/tac` and
   `src/comma_lab` should be review-tracked.
2. Add a preflight assertion that known production roots are represented in
   `SCAN_PREFIXES` so this class of omission cannot silently recur.
