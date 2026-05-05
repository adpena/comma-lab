"""Preflight check: runtime-executed file references in shell scripts must resolve.

Problem class: subagent worktrees that build a wrapper script + the helpers it
calls, but only commit the wrapper script. The helpers are then permanently
lost when the worktree is auto-cleaned. Next dispatch through the wrapper
FATALs at runtime when require_file or `$PYBIN -u <path>` hits a missing file.

Real instances caught (2026-05-04 recovery session):
  - scripts/ensure_remote_uv.sh — called by remote_archive_only_eval.sh:78
    via `bash "$WORKSPACE/scripts/ensure_remote_uv.sh" --symlink-system`,
    missing on disk, would FATAL every dispatch through bootstrap_runtime_deps.
  - experiments/line_search_pose_refinement.py — called by
    remote_lane_line_search_c067.sh, missing on disk, would FATAL Lane
    line-search dispatch at first stage.

This check codifies the audit pattern that surfaced both:
  1. Iterate scripts/*.sh (and any other dispatch-relevant shell dirs)
  2. Extract every RUNTIME-executed reference matching:
       - require_file "$WORKSPACE/<path>"  (explicit dependency declaration)
       - "$PYBIN" [args] <path.py>          (executed python invocation)
       - bash "$WORKSPACE/<path.sh>"        (executed shell invocation)
  3. Resolve each reference relative to repo root
  4. Report any reference whose target file does not exist on disk

Excludes:
  - Comment-only references (line starts with # or ##)
  - Module-path invocations like `python -m tac.experiments.X` (resolved
    via Python import system, not file system)
  - References inside skip-blocks marked `# NOT YET IMPLEMENTED` or
    `# placeholder` so aspirational stubs don't false-positive

Promotion plan: starts strict=False (warn-only) so any pre-existing drift
is surfaced first. Operator fixes the violations, then strict-flips.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Dirs to scan for dispatch shell scripts
_SHELL_SCAN_DIRS = ("scripts",)

# Runtime-executed file reference patterns
# 1. require_file "$WORKSPACE/<path>"  →  capture <path>
_REQUIRE_FILE_RE = re.compile(
    r'require_file\s+"\$WORKSPACE/([^"]+)"'
)
# 2. "$PYBIN" [args] <path.py>  →  capture <path.py>
#    Match $PYBIN OR $PY_BIN OR direct .venv/bin/python invocations
_PYBIN_INVOCATION_RE = re.compile(
    r'(?:"?\$\{?PY(?:BIN|_BIN)\}?"?|\.venv/bin/python\d?)\s+(?:-\w+\s+)*'
    r'((?:experiments|src/tac|tools|submissions/[^\s]+)/[A-Za-z_][A-Za-z_0-9/]*\.py)'
)
# 3. bash "$WORKSPACE/<path.sh>" or bash <path.sh>  →  capture <path.sh>
_BASH_INVOCATION_RE = re.compile(
    r'\bbash\s+(?:"\$WORKSPACE/)?((?:scripts|experiments|tools)/[A-Za-z_][A-Za-z_0-9/]*\.sh)"?'
)

# Skip-block markers — references inside these blocks are intentional
# stubs/aspirational placeholders, not lost helpers. Matched as substrings
# anywhere in previous-N comment lines. Relatively loose to forgive
# variations in operator phrasing; cost is occasional false-suppression
# on code lines containing these phrases as string literals (acceptable
# since this check is warn-only initially).
_SKIP_BLOCK_MARKERS = (
    "NOT YET IMPLEMENTED",
    "not yet implemented",
    "placeholder",
    "PLACEHOLDER",
    "until experiments/",
    "REQUIRED_SOURCE_SHA256S",  # docstring example showing the format
    "future-work stub",
    "future work stub",
)


def _line_is_comment(line: str) -> bool:
    s = line.lstrip()
    return s.startswith("#")


def _line_in_skip_block(prev_lines: list[str], window: int = 6) -> bool:
    """Return True if any of the previous `window` lines mentioned a skip
    marker (placeholder / not-yet-implemented). Conservative: only the same
    code block / nearby context matters."""
    for line in prev_lines[-window:]:
        for marker in _SKIP_BLOCK_MARKERS:
            if marker in line:
                return True
    return False


def _scan_script(script_path: Path, repo_root: Path) -> list[tuple[str, int, str]]:
    """Return list of (script_path_rel, line_number, missing_target_path) tuples."""
    violations: list[tuple[str, int, str]] = []
    try:
        text = script_path.read_text()
    except (UnicodeDecodeError, OSError):
        return violations

    lines = text.splitlines()
    script_rel = str(script_path.relative_to(repo_root))

    for line_idx, line in enumerate(lines):
        # Skip comment-only lines
        if _line_is_comment(line):
            continue
        # Skip if recent context suggests this is intentionally aspirational
        if _line_in_skip_block(lines[:line_idx], window=8):
            continue

        # Collect all runtime-executed references on this line
        targets: list[str] = []
        targets.extend(_REQUIRE_FILE_RE.findall(line))
        targets.extend(_PYBIN_INVOCATION_RE.findall(line))
        targets.extend(_BASH_INVOCATION_RE.findall(line))

        for target in targets:
            target_path = repo_root / target
            if not target_path.exists():
                violations.append((script_rel, line_idx + 1, target))

    return violations


def check_shell_script_runtime_refs_resolve(
    repo_root: Path | None = None,
    strict: bool = False,
    verbose: bool = True,
) -> list[str]:
    """Verify that every runtime-executed file reference in dispatch shell
    scripts resolves to an existing file on disk.

    Args:
        repo_root: defaults to REPO_ROOT auto-detected from this module's path.
        strict: if True, raises RuntimeError on any violation. If False, returns
                the violation list (warn-only).
        verbose: if True, prints scan summary + per-violation details to stdout.

    Returns:
        List of human-readable violation messages
        (`<script>:<line>: missing <target>`).
    """
    root = repo_root or REPO_ROOT
    violations: list[str] = []
    n_scanned = 0

    for shell_dir in _SHELL_SCAN_DIRS:
        d = root / shell_dir
        if not d.is_dir():
            continue
        for script in sorted(d.glob("*.sh")):
            n_scanned += 1
            for script_rel, line_no, target in _scan_script(script, root):
                violations.append(f"{script_rel}:{line_no}: missing {target}")

    if verbose:
        if violations:
            print(f"  [shell-runtime-refs] {len(violations)} violation(s) across {n_scanned} files:")
            for v in violations:
                print(f"    • {v}")
        else:
            print(f"  [shell-runtime-refs] OK: {n_scanned} files scanned")

    if strict and violations:
        raise RuntimeError(
            f"check_shell_script_runtime_refs_resolve: {len(violations)} unresolved "
            f"runtime references — dispatch would FATAL at runtime. See above for details."
        )

    return violations


__all__ = [
    "check_shell_script_runtime_refs_resolve",
    "check_test_imports_resolve_to_disk",
    "REPO_ROOT",
]


# ============================================================
# Sister check: test-file `from <module>.X import` references where the
# corresponding source file is missing on disk. Same lost-helper bug class
# as the shell-script runtime-refs check — caught test_qzs3_packer.py's
# ImportError at collection because experiments/repack_quantizr_faithful_qzs3_archive.py
# was lost in a subagent worktree.
#
# Scope: src/tac/tests/*.py imports targeting repo-internal namespaces
#   - from experiments.X import ...     →  experiments/X.py must exist
#   - from tools.X import ...           →  tools/X.py must exist
#   - from submissions.X.Y import ...   →  submissions/X/Y.py must exist
#
# Out of scope:
#   - stdlib imports (collections, json, etc.)
#   - third-party imports (numpy, torch, brotli, etc.)
#   - tac.* imports (covered by Python's own import system at test time)
#   - dotted imports beyond the file boundary (we only verify the leaf .py)
# ============================================================

_TEST_SCAN_DIR = "src/tac/tests"

# Match: from <ns>.X[.Y...] import
# Captures the dotted path so we can resolve to a file on disk.
_FROM_IMPORT_RE = re.compile(
    r'^\s*from\s+((?:experiments|tools|submissions)(?:\.[A-Za-z_][A-Za-z_0-9]*)+)\s+import\s'
)


def _resolve_dotted_to_file(dotted: str, repo_root: Path) -> Path:
    """Map `experiments.foo.bar` → `experiments/foo/bar.py`.

    Note: this is a heuristic — the actual Python resolution may pick up
    `experiments/foo/bar/__init__.py` instead. We check both.
    """
    parts = dotted.split(".")
    # Try as a module file: <parts...>.py
    as_file = repo_root.joinpath(*parts[:-1], parts[-1] + ".py")
    if as_file.is_file():
        return as_file
    # Try as a package init: <parts...>/__init__.py
    as_pkg = repo_root.joinpath(*parts, "__init__.py")
    if as_pkg.is_file():
        return as_pkg
    # Return the file path that "should have existed" (caller treats as missing)
    return as_file


def _scan_test_file(test_path: Path, repo_root: Path) -> list[tuple[str, int, str, str]]:
    """Return list of (test_path_rel, line_number, dotted_module, expected_file)
    for unresolved imports."""
    violations: list[tuple[str, int, str, str]] = []
    try:
        text = test_path.read_text()
    except (UnicodeDecodeError, OSError):
        return violations

    test_rel = str(test_path.relative_to(repo_root))
    for line_idx, line in enumerate(text.splitlines()):
        m = _FROM_IMPORT_RE.match(line)
        if not m:
            continue
        dotted = m.group(1)
        target = _resolve_dotted_to_file(dotted, repo_root)
        if not target.exists():
            violations.append((test_rel, line_idx + 1, dotted, str(target.relative_to(repo_root))))

    return violations


def check_test_imports_resolve_to_disk(
    repo_root: Path | None = None,
    strict: bool = False,
    verbose: bool = True,
) -> list[str]:
    """Verify that every `from <experiments|tools|submissions>.X import`
    in src/tac/tests/*.py resolves to an existing file on disk.

    Catches the same lost-helper class as
    check_shell_script_runtime_refs_resolve, but for test-file imports
    (which fail at pytest collection time, not at runtime).
    """
    root = repo_root or REPO_ROOT
    test_dir = root / _TEST_SCAN_DIR
    violations: list[str] = []
    n_scanned = 0

    if test_dir.is_dir():
        for test_file in sorted(test_dir.rglob("*.py")):
            n_scanned += 1
            for test_rel, line_no, dotted, expected_file in _scan_test_file(test_file, root):
                violations.append(
                    f"{test_rel}:{line_no}: "
                    f"`from {dotted} import ...` -> missing {expected_file}"
                )

    if verbose:
        if violations:
            print(f"  [test-imports-resolve] {len(violations)} violation(s) across {n_scanned} files:")
            for v in violations:
                print(f"    • {v}")
        else:
            print(f"  [test-imports-resolve] OK: {n_scanned} files scanned")

    if strict and violations:
        raise RuntimeError(
            f"check_test_imports_resolve_to_disk: {len(violations)} unresolved "
            f"test-file imports — pytest will ImportError at collection. See above."
        )

    return violations
