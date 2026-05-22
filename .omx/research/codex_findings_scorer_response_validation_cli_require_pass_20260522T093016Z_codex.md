# Codex Findings: Scorer-Response Validation CLI Require-Pass

UTC: 2026-05-22T09:30:16Z
Lane: `lane_scorer_response_validation_cli_require_pass_20260522`

## Summary

Hardened `tools/validate_scorer_response_dataset.py` for automation use. The
tool can still be used in report-only mode, but callers may now pass
`--require-pass` to fail closed when the validation gate is blocked.

## Fix

- Added `--require-pass`.
- The CLI still writes JSON and Markdown outputs before returning a failing
  exit code, preserving inspectable artifacts for CI/operator review.
- Stdout now includes a machine-readable `passed` field next to `status` and
  `blockers`.
- Added regression tests for:
  - blocked validation gate returning exit code `2` after writing outputs
  - passed validation gate returning exit code `0`

## Authority Boundary

This is a validation/automation guard only. It creates no score authority,
promotion eligibility, rank/kill authority, or exact-eval dispatch readiness.
The validated dataset remains non-authoritative scorer-response signal unless
separate contest auth-eval custody exists.

No new canonical equation is introduced; this is a CLI enforcement wrapper
around `scorer_response_dataset_validation_gate.v1`.

## Verification

- `.venv/bin/ruff check tools/validate_scorer_response_dataset.py src/tac/tests/test_scorer_response_dataset.py`
- `.venv/bin/python -m pytest -q src/tac/tests/test_scorer_response_dataset.py`

Results:

- Ruff: pass
- Scorer-response dataset tests: 73 passed

## Partner WIP

`tools/build_hfv1_sparse_sidecar_candidate.py` is unrelated dirty work and was
left unstaged.
