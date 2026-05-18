#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# no-argparse-OK: git hook entrypoint — controlled via env vars (PREFLIGHT_HOOK_ENABLED/REVIEW_GATE_ENABLED/etc)
"""Pre-commit / pre-push hook: runs preflight + ruff undefined-name check.

This is the cheap-and-fast safety net that catches SHIRAZ-class bugs at
commit time so they never reach a GPU.

Layers (in order):
  1. ruff F821 (undefined name) — catches NameError-class bugs like the
     auth_eval `expected_raw` scope leak that crashed every authoritative
     evaluation for weeks.
  2. tac.preflight --scope dev — the bounded developer validator stack.
     PREFLIGHT_FULL=1 switches to the exhaustive release/custody stack.
  3. Hands off to review_gate_hook for the standard review-tracker check.

Install:
    ln -sf ../../tools/preflight_hook.py .git/hooks/pre-commit
    ln -sf ../../tools/preflight_hook.py .git/hooks/pre-push
    chmod +x tools/preflight_hook.py

Environment overrides:
    PREFLIGHT_HOOK_ENABLED=0   Skip preflight (review gate still runs)
    PREFLIGHT_FULL=1           Run full whole-repo preflight instead of fast mode
    PREFLIGHT_ALLOW_SLOW=1     Explicitly allow slow release/custody preflight
    PREFLIGHT_TIMEOUT_SECONDS  Override preflight subprocess timeout
    REVIEW_GATE_ENABLED=0      Skip review gate
    REVIEW_GATE_OVERRIDE=1     Override review gate (still runs preflight)
    PREFLIGHT_SKIP_RUFF=1      Skip ruff F821 step (e.g., when ruff missing)
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# ANSI colors
RED = "\033[31m"
YELLOW = "\033[33m"
GREEN = "\033[32m"
BOLD = "\033[1m"
RST = "\033[0m"


def _staged_py_files() -> list[str]:
    """Return staged .py files relative to repo root."""
    try:
        out = subprocess.check_output(
            ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
            cwd=REPO_ROOT,
            text=True,
        )
    except subprocess.CalledProcessError:
        return []
    return [f for f in out.splitlines() if f.endswith(".py") and (REPO_ROOT / f).exists()]


def run_ruff_undefined_name(staged: list[str]) -> int:
    """Run ruff on staged .py files, fail on F821 (undefined name) only.

    F821 is the rule that catches scope-leak bugs like the auth_eval
    `expected_raw` NameError. Keep this isolated from project-level broad-lint
    ignores so per-file style carve-outs cannot suppress undefined-name checks.
    """
    if not staged or os.environ.get("PREFLIGHT_SKIP_RUFF") == "1":
        return 0
    try:
        result = subprocess.run(
            [
                ".venv/bin/ruff",
                "check",
                "--isolated",
                "--force-exclude",
                "--select",
                "F821",
                "--ignore-noqa",
                "--exclude",
                "experiments/archive",
                "--exclude",
                "experiments/results",
                "--no-cache",
                *staged,
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        # ruff not installed — soft-fail (preflight still catches drift)
        print(f"{YELLOW}[preflight-hook] ruff missing, skipping F821 check{RST}",
              file=sys.stderr)
        return 0
    if result.returncode != 0:
        print(f"\n{RED}{BOLD}[preflight-hook] BLOCKED: undefined-name (F821) found{RST}")
        print(result.stdout, file=sys.stderr)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        print(f"\n{RED}This is the bug class that hid the auth_eval `expected_raw` "
              f"NameError for weeks.{RST}", file=sys.stderr)
        print("  Fix the undefined name and re-stage.", file=sys.stderr)
        print("  Skip (NOT recommended): PREFLIGHT_SKIP_RUFF=1 git commit ...",
              file=sys.stderr)
        return 1
    return 0


def run_preflight() -> int:
    """Run the bounded preflight validator stack.

    Default hook mode is intentionally fast and source-index friendly:
    `tac.preflight --no-codebase` catches artifact/profile wiring without
    scanning every recovered public-PR source tree and reverse-engineering
    custody mirror on each commit. Operators can still request the full
    whole-repo scan with `PREFLIGHT_FULL=1`, but it keeps the normal 30s DX
    budget unless `PREFLIGHT_ALLOW_SLOW=1` is set for a deliberate release or
    custody sweep.
    """
    if os.environ.get("PREFLIGHT_HOOK_ENABLED", "1") == "0":
        return 0
    cmd = _preflight_command()
    timeout = _preflight_timeout_seconds()
    try:
        result = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError:
        print(f"{YELLOW}[preflight-hook] .venv missing, skipping preflight{RST}",
              file=sys.stderr)
        return 0
    except subprocess.TimeoutExpired as exc:
        print(f"\n{RED}{BOLD}[preflight-hook] BLOCKED: preflight timed out{RST}",
              file=sys.stderr)
        print(f"  command: {' '.join(cmd)}", file=sys.stderr)
        print(f"  timeout: {timeout}s", file=sys.stderr)
        if exc.stdout:
            print(str(exc.stdout)[-4000:], file=sys.stderr)
        if exc.stderr:
            print(str(exc.stderr)[-4000:], file=sys.stderr)
        print(f"\n{RED}The hook must stay bounded during normal development.{RST}", file=sys.stderr)
        print(
            "  Use PREFLIGHT_ALLOW_SLOW=1 only for deliberate release/custody sweeps.",
            file=sys.stderr,
        )
        return 1
    if result.returncode != 0:
        print(f"\n{RED}{BOLD}[preflight-hook] BLOCKED: preflight failed{RST}")
        print(result.stdout, file=sys.stderr)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        print(f"\n{RED}A drift / arity / profile / arch / filename rule fired.{RST}",
              file=sys.stderr)
        print("  Fix the issue, then commit.", file=sys.stderr)
        print("  Skip (NOT recommended): PREFLIGHT_HOOK_ENABLED=0 git commit ...",
              file=sys.stderr)
        return 1
    return 0


def _preflight_command() -> list[str]:
    """Return the preflight command for the current hook mode."""
    cmd = [".venv/bin/python", "-m", "tac.preflight"]
    if os.environ.get("PREFLIGHT_FULL", "0") == "1":
        cmd.extend(["--scope", "all"])
        if os.environ.get("PREFLIGHT_ALLOW_SLOW", "0") == "1":
            cmd.append("--allow-slow-preflight")
    else:
        cmd.append("--no-codebase")
    return cmd


def _preflight_timeout_seconds() -> int:
    """Return a positive preflight timeout with conservative defaults."""
    raw = os.environ.get("PREFLIGHT_TIMEOUT_SECONDS")
    if raw:
        try:
            value = int(raw)
        except ValueError:
            value = 0
        return value if value > 0 else 30
    if (
        os.environ.get("PREFLIGHT_FULL", "0") == "1"
        and os.environ.get("PREFLIGHT_ALLOW_SLOW", "0") == "1"
    ):
        return 600
    return 30


def run_review_gate() -> int:
    """Hand off to the existing review-tracker gate hook."""
    hook = REPO_ROOT / "tools" / "review_gate_hook.py"
    if not hook.exists():
        return 0
    try:
        result = subprocess.run(
            [".venv/bin/python", str(hook)],
            cwd=REPO_ROOT,
        )
    except FileNotFoundError:
        return 0
    return result.returncode


def main() -> int:
    staged = _staged_py_files()

    # Step 1: ruff F821 on staged files only (fast, ~50ms per file)
    rc = run_ruff_undefined_name(staged)
    if rc != 0:
        return rc

    # Step 2: bounded developer preflight (PREFLIGHT_FULL=1 for full release scan)
    rc = run_preflight()
    if rc != 0:
        return rc

    # Step 3: review gate
    return run_review_gate()


if __name__ == "__main__":
    sys.exit(main())
