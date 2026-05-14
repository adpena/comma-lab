# Catalog #117 commit-serializer prefix fix - 2026-05-14

## Summary

R1 recursive review found that `check_subagent_commit_serializer_uses_lock`
compared 7-character git short SHAs against 9-character `head_after` prefixes
from `.omx/state/commit-serializer.log`. That made valid serialized commits
look unserialized and made the gate unusable as a measurement tool.

This landing fixes the comparison as validated SHA-prefix matching and only
trusts serializer rows whose commit actually succeeded (`commit_rc == 0`).
Failed "nothing to commit" serializer attempts no longer bless commits created
outside the serializer.

## Evidence

- Focused regression: `PYTHONPATH=src:upstream:$PWD .venv/bin/python -m pytest src/tac/tests/test_check_117_commit_serializer_uses_lock.py -q`
- Result: `4 passed in 1.36s`
- New-file lint: `PYTHONPATH=src:upstream:$PWD .venv/bin/python -m ruff check src/tac/tests/test_check_117_commit_serializer_uses_lock.py`
- Result: `All checks passed!`
- Whitespace guard: `git diff --check -- src/tac/preflight.py src/tac/tests/test_check_117_commit_serializer_uses_lock.py`
- Result: clean

## Live Recompute

After the fix, live Catalog #117 reported 6 unserialized commits in the last 10
commits. This is expected measurement movement: the gate is no longer all-noise,
and failed serializer attempts are no longer counted as successful serialization.

First observed live violations:

- `84780ac`
- `4bfea6b`
- `a2112a8`
- `2b7f204`
- `0916332`
- `92eb932`

These require separate operator/main-session allowlisting or serializer-based
cleanup if Catalog #117 is promoted toward strict mode. This landing intentionally
does not rewrite commit history.
