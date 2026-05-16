# Result-Review Reactivation Preflight Guard - 2026-05-16

## Summary

Hardened the existing exact-negative forensic review gate so a
`tac_result_review_packet_v1` must carry non-empty `reactivation_criteria`
before it can satisfy `check_evidence_row_has_falsification_scope_when_negative`.

## Bug Class

`tools/build_result_review_packet.py` now defaults conservative reopen criteria,
but a hand-written or legacy review packet could still satisfy preflight by
declaring the right schema, review requirements, runtime custody, score
recomputation, and dispatch-claim fields while omitting the criteria that reopen
the measured config. That is signal loss: a negative result can become sticky
without the exact evidence required to revisit it.

## Fix

`src/tac/preflight.py::_result_review_packet_is_valid_for_forensic_review`
now fails closed unless `reactivation_criteria` is a list with at least one
non-empty string.

## Verification

- `.venv/bin/python -m pytest src/tac/tests/test_preflight_harden_2026_05_08_checks.py src/tac/tests/test_build_result_review_packet.py -q`
- `.venv/bin/python -m pytest src/tac/tests/test_preflight_harden_2026_05_08_checks.py::test_exact_negative_with_legacy_result_review_packet_passes src/tac/tests/test_preflight_harden_2026_05_08_checks.py::test_exact_negative_result_review_packet_requires_reactivation_criteria -q`
- `.venv/bin/python -m py_compile src/tac/preflight.py src/tac/tests/test_preflight_harden_2026_05_08_checks.py`
- `git diff --check -- src/tac/preflight.py src/tac/tests/test_preflight_harden_2026_05_08_checks.py`

Note: repo-wide `ruff check src/tac/preflight.py` still reports many
pre-existing style findings in the historical monolithic preflight module; this
patch did not attempt unrelated cleanup.
