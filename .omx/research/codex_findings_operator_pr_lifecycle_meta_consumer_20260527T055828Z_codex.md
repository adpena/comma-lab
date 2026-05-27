# Operator PR Lifecycle Landing

UTC: 2026-05-27T05:58:28Z
Agent: codex

## Verdict

The previously untracked Phase 9 Layer 7 operator PR-submission full-lifecycle
CLI was made lint-clean and testable.

The PR lifecycle CLI is an operator-gated orchestrator.  It composes the
canonical submission packet layers and emits `gh` commands as text only; it does
not run release or PR creation commands.

## Fixes Applied

- Ran `ruff --fix` on the untracked lifecycle CLI and its tests.
- Aligned the exit-code contract with the implementation: clean packets stop at
  exit 4 because release and PR creation remain operator-gated; exit 0 is
  reserved for a future no-human-action mode.
- Updated the lifecycle test harness to match the current paired-auth schema:
  `[contest-CUDA; contest-CPU]` axis spelling and canonical paired-axis evidence
  grades.
- Fixed the synthetic partial-CUDA verdict helper so `PAIRED_PARTIAL_CUDA_ONLY`
  carries the archive SHA and CUDA-side fields required by
  `PairedAuthEvalVerdict`.

## Verification

- `.venv/bin/ruff check tools/operator_pr_submission_full_lifecycle.py src/tac/tests/test_operator_pr_submission_full_lifecycle_cli.py`
  passed.
- `.venv/bin/python -m pytest src/tac/tests/test_operator_pr_submission_full_lifecycle_cli.py -q`
  passed: 45 tests.

## Authority

No score claim, promotion, rank/kill, dispatch, release, or PR operation is
performed by this surface.  The lifecycle CLI exits operator-gated for clean
packets.
