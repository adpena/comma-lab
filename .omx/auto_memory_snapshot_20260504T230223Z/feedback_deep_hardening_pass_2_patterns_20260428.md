# Deep DX hardening pass 2 — patterns + how to detect them

**Trigger:** 2026-04-28, 26 lanes burning $7.23/hr; user mandate "fix all
DX + preflight + runtime + debugging + anti-arbitrariness bugs permanently".
Builds on the 7 HIGH DX bug fixes earlier today.

## Outcome

- 4 commits (audit doc + tools + checks + tests)
- 2 NEW diagnostic CLIs: `tools/audit_archive.py`, `tools/diagnose_lane.py`
  (24 unit tests)
- 3 NEW preflight checks (51, 52, 53), 15 unit tests, warn-only initially:
  - **Check 51 (no-bare-except)**: live count 0 (after `tools/fleet_dashboard_live.py:67` fix)
  - **Check 52 (subprocess-checked)**: live count 31 (cleanup target)
  - **Check 53 (tools-have-argparse)**: live count 7 (review_tracker family)
- Total preflight checks: 50 → **53** (49 strict + 4 warn-only)

## 3 transferable meta-patterns

### 1. Same-line waiver markers (Codex R5-r6 #1 generalized)

Every new check uses ONLY same-line waiver markers — NEVER N-line lookback.
Examples:
- `# noqa: E722` for bare except
- `# subprocess-no-check-OK: <reason>` for unchecked subprocess
- `# no-argparse-OK: <reason>` for tools without argparse

Why: 6-line lookback can waive UNRELATED calls. Same-line is unambiguous.
This pattern was caught + fixed in Round 22 (Codex R5-r6 Finding #1).
All 53 preflight checks now follow it.

**Detection:** any new check that uses `lines[i-N:i]` lookback for waiver
matching MUST be flagged at code review.

### 2. Single-SSH diagnostic gather (avoids N-shells fan-out)

`tools/diagnose_lane.py` makes ONE SSH call that emits sections delimited
by `===<NAME>===` markers (HEARTBEAT / RUN_LOG / SETUP_LOG / LANE_LOG /
AUTH_EVAL / ARCHIVE / SCORE / GPU). Parses sections with a single
`out.split("===")` pass.

Why: N separate SSH calls hit OpenSSH MaxStartups rate limit (CLAUDE.md
note about bat00). One call gathers everything.

**Detection:** any DX tool that does N>3 separate `ssh root@host` calls
should be refactored to use this pattern.

### 3. Hardcoded-fallback in diagnostic tools

`tools/audit_archive.py` tries to import `REQUIRED_ARCHIVE_MEMBERS` +
magic allowlist from `tac.stack_compositions`, but falls back to hardcoded
values on import failure (with WARN to stderr).

Why: operators may run the audit tool on a Vast.ai instance where the
tac package isn't installed (degraded recovery, fresh container,
different Python env). The tool should still work.

**Detection:** any new diagnostic CLI that imports from `tac.*` should
either (a) not require the import to function or (b) document the
required environment in the docstring.

## How to extend

To add a new strict preflight check:

1. Implement `check_X(repo_root, strict=False, verbose=True) -> list[str]`
   following the signature pattern in `src/tac/preflight.py`.
2. Use SAME-LINE waiver markers (no N-line lookback).
3. Skip `upstream/`, `.venv/`, `build/`, `dist/`, `__pycache__/`,
   `tests/` unless the check is specifically about test code.
4. Add 4-5 unit tests (clean-passes / hot-path-fails / waiver-honored /
   strict-raises) using a `tmp_path`-based `fake_repo` fixture.
5. Wire into `preflight_all()` at warn-only first. Document live count
   in a comment.
6. After live cleanup, flip `strict=False` → `strict=True` in the
   `preflight_all()` call.

## What NOT to do (anti-patterns this pass avoided)

- DON'T grep all of /usr or /opt looking for issues — scan only
  `src/tac/`, `scripts/`, `tools/`, `experiments/`.
- DON'T use `__file__` to skip the preflight file via path-string
  matching; use `Path(__file__).resolve() == path.resolve()`.
- DON'T silently fall through on subprocess timeout — distinguish
  TIMEOUT from EXC: classification (matches `verify_vast_instances.py`).
- DON'T promote new checks to strict before verifying live count is 0.

## Reference

- `.omx/research/deep_hardening_pass_2_20260428.md` (full audit)
- `feedback_verify_vast_instances_3_dx_bugs_20260428.md` (prior pass)
- `feedback_zip_dep_bootstrap_trap.md` (silent-cascade root pattern)
- `src/tac/tests/test_preflight_deep_hardening_checks.py` (15 tests)
- `src/tac/tests/test_audit_archive.py` (9 tests)
- `src/tac/tests/test_diagnose_lane.py` (15 tests)
