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
  3. CI-blind tests — the MLX-gated modules GitHub Actions cannot execute (no Linux
     mlx wheel => pytest.importorskip skips them and reports GREEN), restricted to
     those reachable from the staged files. This hook is their ONLY automated surface.
  4. Hands off to review_gate_hook for the standard review-tracker check.

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
    PREFLIGHT_SKIP_CI_BLIND_TESTS=1  Skip the CI-blind test step (NOT recommended)
    PREFLIGHT_CI_BLIND_TIMEOUT_SECONDS  Override that step's subprocess timeout
"""
from __future__ import annotations

import ast
import os
import re
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


def _ci_blind_test_modules() -> list[Path]:
    """Test modules that GitHub Actions CI structurally CANNOT execute.

    `.github/workflows/ci.yml` runs pytest on ubuntu-24.04 and installs `.[dev,runtime]`.
    `mlx` is a SEPARATE optional extra (pyproject `mlx = ["mlx>=0.5"]`) with no Linux
    wheels, so every module whose import guard is `pytest.importorskip("mlx...")` is
    SKIPPED there — and pytest reports a skip as green. Those modules therefore have NO
    automated surface at all: CI cannot run them, and this hook (until this step) ran no
    tests. `test_ddm_tb1_tr1_renderer.py` sat red on main for ~2 days that way.

    Detection is static (never imports the modules it guards): a cheap substring
    prefilter, then an AST confirmation that an actual `importorskip("mlx...")` CALL
    exists. The AST step matters — a test that merely mentions the guard inside a string
    literal (this file's own tests do) is not itself MLX-gated, and a substring-only
    detector would mis-classify it.
    """
    tests_dir = REPO_ROOT / "src" / "tac" / "tests"
    if not tests_dir.is_dir():
        return []
    blind: list[Path] = []
    for path in sorted(tests_dir.glob("test_*.py")):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if "importorskip" not in text or "mlx" not in text:
            continue
        try:
            tree = ast.parse(text)
        except SyntaxError:
            blind.append(path)  # unparseable => assume gated; ties go to running it
            continue
        if _has_mlx_importorskip_call(tree):
            blind.append(path)
    return blind


def _has_mlx_importorskip_call(tree: ast.AST) -> bool:
    """True when the module CALLS importorskip("mlx...") — any of the observed forms
    (`pytest.importorskip(...)`, a bare `importorskip(...)`, with or without `reason=`)."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
        if name != "importorskip" or not node.args:
            continue
        first = node.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str) \
                and first.value.split(".")[0] == "mlx":
            return True
    return False


def _module_reference_tokens(rel_path: str) -> set[str]:
    """Every importable dotted name a test could use to reach `rel_path`.

    `src/tac/witness_dsl/qa84_rowband_grammar.py` ->
        {"tac.witness_dsl.qa84_rowband_grammar", "witness_dsl.qa84_rowband_grammar",
         "qa84_rowband_grammar"}
    All trailing sub-paths are included because `src/` is on sys.path and tests import by
    package path, by sub-package, or by bare module name depending on their own sys.path
    surgery. No heuristics/thresholds: an extra match only costs runtime, never silence.
    """
    if not rel_path:
        return set()
    parts = Path(rel_path).with_suffix("").parts
    if parts and parts[0] == "src":
        parts = parts[1:]
    return {".".join(parts[i:]) for i in range(len(parts))}


def _select_ci_blind_tests(staged: list[str]) -> list[str]:
    """CI-blind test modules that reference any staged module (or are staged themselves)."""
    blind = _ci_blind_test_modules()
    if not blind or not staged:
        return []
    staged_set = {str(REPO_ROOT / s) for s in staged}
    tokens: set[str] = set()
    for rel in staged:
        tokens |= _module_reference_tokens(rel)
    selected: list[str] = []
    # Whole-word match, not substring: a bare stem like "io" or "foo" must appear as a
    # name, not inside another identifier. Over-matching costs only runtime; UNDER-matching
    # costs silence, so ties go to running the test.
    pattern = re.compile("|".join(rf"\b{re.escape(t)}\b" for t in sorted(tokens))) \
        if tokens else None
    for path in blind:
        if str(path) in staged_set:
            selected.append(str(path.relative_to(REPO_ROOT)))
            continue
        if pattern is None:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if pattern.search(text):
            selected.append(str(path.relative_to(REPO_ROOT)))
    return selected


def _ci_blind_timeout_seconds() -> int:
    raw = os.environ.get("PREFLIGHT_CI_BLIND_TIMEOUT_SECONDS")
    if raw:
        try:
            value = int(raw)
        except ValueError:
            value = 0
        return value if value > 0 else 180
    return 180


def run_ci_blind_tests(staged: list[str]) -> int:
    """Run the CI-blind (MLX-gated) tests reachable from the staged files.

    This is the ONLY automated moment on the only machine that can execute them, so it
    BLOCKS on red and on timeout rather than warning: a soft-pass here would re-create
    the exact silence it exists to remove (skip-as-green).
    """
    if os.environ.get("PREFLIGHT_SKIP_CI_BLIND_TESTS") == "1":
        return 0
    selected = _select_ci_blind_tests(staged)
    if not selected:
        return 0
    timeout = _ci_blind_timeout_seconds()
    try:
        result = subprocess.run(
            [".venv/bin/python", "-m", "pytest", *selected,
             "-q", "--no-header", "-m", "not slow"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=timeout,
        )
    except FileNotFoundError:
        print(f"{YELLOW}[preflight-hook] .venv missing, skipping CI-blind tests{RST}",
              file=sys.stderr)
        return 0
    except subprocess.TimeoutExpired:
        print(f"\n{RED}{BOLD}[preflight-hook] BLOCKED: CI-blind tests timed out{RST}",
              file=sys.stderr)
        print(f"  modules: {' '.join(selected)}", file=sys.stderr)
        print(f"  timeout: {timeout}s (PREFLIGHT_CI_BLIND_TIMEOUT_SECONDS overrides)",
              file=sys.stderr)
        return 1
    if result.returncode == 5:
        # pytest EXIT_NOTESTSCOLLECTED: everything in the selection was deselected by
        # `-m "not slow"`. Nothing ran, so nothing is proven — say so instead of
        # reporting the green tick that a bare `!= 0` check would have called a failure
        # and a bare `== 0` check would have called a pass.
        print(f"{YELLOW}[preflight-hook] CI-blind step: no tests collected in "
              f"{len(selected)} selected module(s) (all `slow`?) — nothing verified{RST}",
              file=sys.stderr)
        return 0
    if result.returncode != 0:
        print(f"\n{RED}{BOLD}[preflight-hook] BLOCKED: CI-blind (MLX-gated) test failed{RST}")
        print(result.stdout[-6000:], file=sys.stderr)
        if result.stderr:
            print(result.stderr[-2000:], file=sys.stderr)
        print(f"\n{RED}GitHub Actions CANNOT catch this: mlx has no Linux wheel, so CI "
              f"SKIPS these modules and reports green.{RST}", file=sys.stderr)
        print("  This hook is the only automated surface that runs them. Fix, then commit.",
              file=sys.stderr)
        print("  Skip (NOT recommended): PREFLIGHT_SKIP_CI_BLIND_TESTS=1 git commit ...",
              file=sys.stderr)
        return 1
    print(f"{GREEN}[preflight-hook] CI-blind tests OK ({len(selected)} module(s)){RST}",
          file=sys.stderr)
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
    _echo_preflight_scope_coverage(result.stderr)
    return 0


def _echo_preflight_scope_coverage(stderr: str) -> None:
    """Surface the preflight DENOMINATOR where a committer actually reads it.

    A green hook that examined nothing is the exact bug this reports on
    ("vacuity is indistinguishable from PASS"). The CLI emits a VACUOUS ledger
    line; without echoing it here the committer sees only the hook's silence.
    One concise line, not an alarm: the number is the signal.
    """
    for line in (stderr or "").splitlines():
        if "PREFLIGHT VACUOUS" in line:
            print(
                f"{YELLOW}[preflight-hook] preflight examined 0 gates this "
                f"commit (fast --no-codebase mode). This hook is NOT gate "
                f"coverage.{RST}",
                file=sys.stderr,
            )
            print(
                "  Full developer gate set: PREFLIGHT_FULL=0 "
                ".venv/bin/python -m tac.preflight --scope dev",
                file=sys.stderr,
            )
            return


def _preflight_command() -> list[str]:
    """Return the preflight command for the current hook mode."""
    cmd = [".venv/bin/python", "-m", "tac.preflight"]
    if os.environ.get("PREFLIGHT_FULL", "0") == "1":
        cmd.extend(["--scope", "all"])
        if os.environ.get("PREFLIGHT_ALLOW_SLOW", "0") == "1":
            cmd.append("--allow-slow-preflight")
    else:
        cmd.append("--no-codebase")
        # VACUITY ACKNOWLEDGEMENT (2026-08-01, task #842). `--no-codebase`
        # skips EVERY codebase gate call site, so this hook mode examines
        # 0 gates — MEASURED: 0 of 27 declared in 0.52s. The CLI now refuses
        # rc=3 on an empty scope rather than printing a bare "PREFLIGHT
        # PASSED"; this hook is the one caller with a designed reason to
        # accept that, so it must NAME the emptiness rather than inherit it
        # silently. The verdict still prints VACUOUS and `run_preflight`
        # echoes the coverage number on every commit — acknowledgement
        # suppresses the refusal ONLY, never the report.
        cmd.append("--acknowledge-empty-scope")
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
    env = os.environ.copy()
    if _is_pre_push_invocation():
        env.setdefault("REVIEW_GATE_MODE", "pre-push")
    else:
        env.setdefault("REVIEW_GATE_MODE", "pre-commit")
    try:
        result = subprocess.run(
            [".venv/bin/python", str(hook)],
            cwd=REPO_ROOT,
            env=env,
        )
    except FileNotFoundError:
        return 0
    return result.returncode


def _is_pre_push_invocation() -> bool:
    """Return true when this shared hook script was invoked by pre-push."""

    try:
        hook_path = Path(sys.argv[0])
        return hook_path.name == "pre-push"
    except Exception:
        return False


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

    # Step 3: CI-blind (MLX-gated) tests reachable from the staged files — the set
    # GitHub Actions structurally cannot execute (skip-as-green).
    rc = run_ci_blind_tests(staged)
    if rc != 0:
        return rc

    # Step 4: review gate
    return run_review_gate()


if __name__ == "__main__":
    sys.exit(main())
