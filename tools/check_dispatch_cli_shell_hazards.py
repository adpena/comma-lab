#!/usr/bin/env python3
"""Scan for dispatch CLI and shell portability hazards.

This guard covers sprint-expensive bug classes:

* adjudicator-only flags accidentally passed to ``launch_lightning_batch_job.py``;
* typo flags known to have caused dead dispatches;
* zsh-facing snippets using ``path`` as a shell variable;
* local/macOS snippets using GNU-only ``find -printf``;
* dispatcher Python files passing operator-local paths (``str(REPO_ROOT)``,
  ``"/Users/..."``) as values for remote-execution flags
  (``--repo-dir``/``--archive``/``--inflate-sh``/...). The Lightning
  catastrophe of 2026-05-05 ($1.55 / 8 dispatches lost) happened because
  the dispatcher passed ``str(REPO_ROOT)`` instead of ``args.remote_pact``
  to ``--repo-dir``, generating ``cd /Users/adpena/Projects/pact`` commands
  that ran on a Lightning Studio runner;
* remote runner shell scripts whose ``PYTHONPATH`` exports include operator
  paths (``/Users/`` / ``/home/adpena/``).
* stale docs/reports that still authorize remote launch from prediction-era
  memos (``READY-TO-LAUNCH``, ``No additional approval needed``, or
  unsuperseded standing launch instructions).

Historical ledgers and result artifacts are excluded by default. Python
heredocs inside shell snippets are ignored so ordinary Python variables named
``path`` do not trigger shell warnings.
"""
from __future__ import annotations

import argparse
import ast
import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.repo_io import json_text, repo_relative  # noqa: E402

DEFAULT_SCAN_PATHS = ("scripts", "docs", "reports", "tools")
DEFAULT_EXCLUDES = (
    ".git",
    ".omx",
    "__pycache__",
    ".recovery_quarantine_20260505T004735Z",
    "experiments/results",
    "reports/raw",
    "runtime-rs/target",
)
TEXT_SUFFIXES = {".py", ".sh", ".bash", ".md", ".rst", ".txt", ".zsh"}

# Flags whose values are paths consumed inside a remote runner script.
# Passing the operator's local mac path here generates `cd /Users/adpena/...`
# commands that fail at the first cd on the remote.
REMOTE_PATH_FLAGS = frozenset({
    "--repo-dir",
    "--upstream-dir",
    "--archive",
    "--inflate-sh",
    "--baseline-archive",
    "--perturbation-plan",
    "--video",
    "--pair-weights",
    "--remote-pact",
})

# Local-path prefixes that must NOT appear in a remote runner.
LOCAL_PATH_PREFIXES = ("/Users/", "/home/adpena/")
# Conventional remote roots that ARE allowed.
REMOTE_ROOT_PREFIXES = (
    "/workspace/",
    "/teamspace/",
    "/root/",
    "/tmp/",
    "/opt/",
    "/var/",
)

# Files whose subprocess.run([...]) call lists are scanned for the dispatch
# local-path-leak metabug. Patterns are simple substrings checked against
# the repo-relative POSIX path.
DISPATCHER_FILE_HINTS = (
    "tools/lightning_dispatch",
    "scripts/launch_lightning",
    "scripts/launch_lane_on_vastai",
    "tools/dispatch_",
)

LAUNCHER_STALE_FLAGS = {
    "--required-device": (
        "adjudicator-only; use launch_lightning_batch_job.py exact-eval "
        "--adjudicate and run scripts/adjudicate_contest_auth_eval.py after harvest"
    ),
    "--required-samples": (
        "adjudicator-only; use launch_lightning_batch_job.py exact-eval "
        "--adjudicate and run scripts/adjudicate_contest_auth_eval.py after harvest"
    ),
}
KNOWN_TYPO_FLAGS = {
    "--rmote": "typo for --remote or --remote-path; grep argparse before dispatch",
}
STALE_DISPATCH_AUTHORIZATION_PATTERNS = (
    (
        re.compile(r"\bREADY-TO-LAUNCH\b", re.IGNORECASE),
        "READY-TO-LAUNCH language is stale launch authority; mark the memo "
        "historical/superseded or require current claim + preflight + explicit approval",
    ),
    (
        re.compile(r"\bNo additional approval needed\b", re.IGNORECASE),
        "No-additional-approval language bypasses current dispatch-claim and "
        "operator-authorization gates",
    ),
    (
        re.compile(r"\bStanding instruction\b.*\blaunch\b", re.IGNORECASE),
        "standing launch instructions are stale unless explicitly superseded",
    ),
)


@dataclass(frozen=True)
class Hazard:
    path: str
    line: int
    kind: str
    message: str


def _repo_rel(path: Path, root: Path) -> str:
    return repo_relative(path, root)


def _is_excluded(path: Path, root: Path, excludes: tuple[str, ...]) -> bool:
    return _is_excluded_rel(_repo_rel(path, root), excludes)


def _is_excluded_rel(rel: str, excludes: tuple[str, ...]) -> bool:
    parts = set(Path(rel).parts)
    for item in excludes:
        if "/" in item:
            item = item.rstrip("/")
            if rel == item or rel.startswith(item + "/"):
                return True
        elif item in parts:
            return True
    return False


def iter_scan_files(
    root: Path,
    scan_paths: tuple[str, ...],
    excludes: tuple[str, ...] = DEFAULT_EXCLUDES,
) -> list[Path]:
    files: list[Path] = []
    for rel in scan_paths:
        base = root / rel
        if not base.exists():
            continue
        if base.is_file():
            if not _is_excluded(base, root, excludes) and base.suffix.lower() in TEXT_SUFFIXES:
                files.append(base)
            continue
        for dirpath, dirnames, filenames in os.walk(base, topdown=True):
            dirpath_path = Path(dirpath)
            dirnames[:] = [
                dirname
                for dirname in sorted(dirnames)
                if not _is_excluded_rel(_repo_rel(dirpath_path / dirname, root), excludes)
            ]
            for filename in sorted(filenames):
                candidate = dirpath_path / filename
                if candidate.suffix.lower() not in TEXT_SUFFIXES:
                    continue
                if _is_excluded_rel(_repo_rel(candidate, root), excludes):
                    continue
                if not candidate.is_file():
                    continue
                files.append(candidate)
    return files


_HEREDOC_START_RE = re.compile(r"<<-?\s*['\"]?(?P<tag>[A-Za-z_][A-Za-z0-9_]*)['\"]?")


def strip_heredoc_bodies(text: str) -> list[tuple[int, str]]:
    """Return ``(original_lineno, line)`` pairs excluding heredoc bodies."""
    out: list[tuple[int, str]] = []
    stop_tag: str | None = None
    for lineno, line in enumerate(text.splitlines(), start=1):
        if stop_tag is not None:
            if line.strip() == stop_tag:
                stop_tag = None
            continue
        match = _HEREDOC_START_RE.search(line)
        if match:
            out.append((lineno, line))
            stop_tag = match.group("tag")
            continue
        out.append((lineno, line))
    return out


def _logical_commands(numbered_lines: list[tuple[int, str]]) -> list[tuple[int, str]]:
    commands: list[tuple[int, str]] = []
    current_lineno: int | None = None
    current: list[str] = []
    for lineno, line in numbered_lines:
        stripped = line.rstrip()
        if current_lineno is None:
            current_lineno = lineno
        if stripped.endswith("\\"):
            current.append(stripped[:-1])
            continue
        current.append(stripped)
        commands.append((current_lineno, " ".join(part.strip() for part in current)))
        current_lineno = None
        current = []
    if current:
        commands.append((current_lineno or numbered_lines[-1][0], " ".join(current)))
    return commands


def _find_printf_is_remote_linux_context(command: str) -> bool:
    markers = ("/workspace/", "root@", "ssh ", "SSH_BASE", "SSH_OPTS")
    return any(marker in command for marker in markers) or "remote" in command.lower()


def _add_flag_hazards(rel: str, lineno: int, command: str, hazards: list[Hazard]) -> None:
    if "launch_lightning_batch_job.py" in command:
        for flag, reason in LAUNCHER_STALE_FLAGS.items():
            if re.search(rf"(?<!\S){re.escape(flag)}(?![A-Za-z0-9_-])", command):
                hazards.append(
                    Hazard(
                        rel,
                        lineno,
                        "stale_lightning_launcher_flag",
                        f"{flag} passed to launch_lightning_batch_job.py; {reason}",
                    )
                )
    for flag, reason in KNOWN_TYPO_FLAGS.items():
        if re.search(rf"(?<!\S){re.escape(flag)}(?![A-Za-z0-9_-])", command):
            hazards.append(Hazard(rel, lineno, "known_typo_flag", f"{flag}: {reason}"))


def _is_dispatcher_file(rel: str) -> bool:
    return any(hint in rel for hint in DISPATCHER_FILE_HINTS)


def _classify_remote_value(node: ast.AST) -> str | None:
    """Return a non-empty diagnostic if ``node`` would leak the operator's local
    path into a remote runner. ``None`` means safe.

    Safe forms (whitelisted):
      - ``args.<name with 'remote'>`` (e.g. ``args.remote_pact``)
      - ``DEFAULT_REMOTE_PACT`` / ``DEFAULT_REMOTE_*`` constants
      - ``str(<X>.relative_to(...))`` (repo-relative path)
      - ``f"{remote_pact}/..."`` (JoinedStr with remote_pact prefix)
      - String literals starting with conventional remote roots.

    Unsafe forms (flagged):
      - ``str(REPO_ROOT)``, ``str(PROJECT_ROOT)``, ``str(ROOT)``
      - ``str(REPO_ROOT / "x")``, ``str(some_path / "x")`` of a known root
      - Bare string literals starting with ``/Users/`` / ``/home/adpena/``.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        value = node.value
        if value.startswith(LOCAL_PATH_PREFIXES):
            return f"hardcoded operator path {value!r}"
        return None

    if isinstance(node, ast.Name):
        return None  # plain identifier — assume the wrapper validated it

    if isinstance(node, ast.Attribute):
        # args.remote_pact / args.archive / etc — assume validated upstream
        return None

    if isinstance(node, ast.JoinedStr):
        # f-string. Concatenation of values is safe iff the first non-empty
        # piece resolves to a remote root or a known remote variable.
        for value in node.values:
            if isinstance(value, ast.Constant) and value.value:
                first = str(value.value)
                if first.startswith(LOCAL_PATH_PREFIXES):
                    return f"f-string starts with operator path {first!r}"
                if first.startswith(REMOTE_ROOT_PREFIXES):
                    return None
                # Path-fragment like '{remote_pact}/upstream' is fine — the
                # FormattedValue carries the variable.
                return None
            if isinstance(value, ast.FormattedValue):
                inner = value.value
                if isinstance(inner, ast.Name) and "remote" in inner.id.lower():
                    return None
                if isinstance(inner, ast.Attribute) and "remote" in inner.attr.lower():
                    return None
                # First piece is a non-remote variable — can't statically tell
                return None
        return None

    if isinstance(node, ast.Call):
        # str(...) wrapper
        if isinstance(node.func, ast.Name) and node.func.id == "str" and node.args:
            inner = node.args[0]

            # str(args.remote_pact)
            if isinstance(inner, ast.Attribute) and "remote" in inner.attr.lower():
                return None

            # str(REPO_ROOT) / str(PROJECT_ROOT) / str(ROOT)
            if isinstance(inner, ast.Name) and inner.id in {"REPO_ROOT", "PROJECT_ROOT", "ROOT", "PACT_ROOT"}:
                return f"str({inner.id})"

            # str(REPO_ROOT / "x") chain — Path / "x"
            if isinstance(inner, ast.BinOp) and isinstance(inner.op, ast.Div):
                left = inner.left
                while isinstance(left, ast.BinOp) and isinstance(left.op, ast.Div):
                    left = left.left
                if isinstance(left, ast.Name) and left.id in {"REPO_ROOT", "PROJECT_ROOT", "ROOT", "PACT_ROOT"}:
                    return f"str({left.id} / ...)"

            # str(<X>.relative_to(...)) — explicitly safe
            if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Attribute):
                if inner.func.attr == "relative_to":
                    return None
                # str(X.absolute()) / str(X.resolve()) — leak prone
                if inner.func.attr in {"absolute", "resolve"}:
                    return f"str(X.{inner.func.attr}()) returns operator-local absolute path"
        return None

    return None


def _scan_python_dispatch_path_leaks(rel: str, text: str) -> list[Hazard]:
    """AST-walk a dispatcher .py file for local-mac-path leaks.

    Only files matching :data:`DISPATCHER_FILE_HINTS` are examined.
    """
    if not _is_dispatcher_file(rel):
        return []
    hazards: list[Hazard] = []
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        # subprocess.run / subprocess.check_call / subprocess.Popen / subprocess.check_output
        is_subprocess = (
            isinstance(node.func, ast.Attribute)
            and node.func.attr in {"run", "check_call", "Popen", "check_output"}
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "subprocess"
        )
        if not is_subprocess:
            continue
        if not node.args or not isinstance(node.args[0], ast.List):
            continue
        elements = node.args[0].elts
        for i in range(len(elements) - 1):
            flag_node = elements[i]
            value_node = elements[i + 1]
            if not isinstance(flag_node, ast.Constant) or not isinstance(flag_node.value, str):
                continue
            flag_name = flag_node.value
            if flag_name not in REMOTE_PATH_FLAGS:
                continue
            leak = _classify_remote_value(value_node)
            if leak:
                hazards.append(
                    Hazard(
                        rel,
                        getattr(value_node, "lineno", flag_node.lineno),
                        "dispatch_local_path_leak",
                        f"{flag_name} value {leak} would leak operator-local path "
                        "into the remote runner. Use args.remote_pact, "
                        "DEFAULT_REMOTE_PACT, or X.relative_to(REPO_ROOT) instead.",
                    )
                )
    return hazards


def _scan_shell_pythonpath_leaks(rel: str, numbered: list[tuple[int, str]]) -> list[Hazard]:
    """Detect ``export PYTHONPATH=...`` lines containing operator paths.

    Only remote runner shell scripts are scanned (``scripts/remote_lane_*.sh``,
    ``scripts/lightning_lane_*.sh``).
    """
    if not (rel.startswith("scripts/remote_lane_") or rel.startswith("scripts/lightning_lane_")):
        return []
    hazards: list[Hazard] = []
    pythonpath_re = re.compile(r"\bexport\s+PYTHONPATH\s*=\s*\"?([^\"\n]+)\"?")
    for lineno, line in numbered:
        match = pythonpath_re.search(line)
        if not match:
            continue
        value = match.group(1)
        for segment in value.split(":"):
            segment = segment.strip()
            if not segment or segment.startswith("$") or segment.startswith("${"):
                continue
            if segment.startswith(LOCAL_PATH_PREFIXES):
                hazards.append(
                    Hazard(
                        rel,
                        lineno,
                        "remote_script_local_pythonpath_leak",
                        f"PYTHONPATH segment {segment!r} is the operator's local "
                        "machine path; remote runner cannot resolve it. Strip the "
                        "segment (the canonical pattern is `src:upstream:${PYTHONPATH:-}`).",
                    )
                )
    return hazards


def _scan_stale_dispatch_authorization(rel: str, numbered: list[tuple[int, str]]) -> list[Hazard]:
    """Detect old human-facing launch authority in docs/reports."""

    if not (rel.startswith("docs/") or rel.startswith("reports/")):
        return []
    hazards: list[Hazard] = []
    for lineno, line in numbered:
        lowered = line.lower()
        if "superseded" in lowered or "historical" in lowered:
            continue
        for pattern, message in STALE_DISPATCH_AUTHORIZATION_PATTERNS:
            if pattern.search(line):
                hazards.append(Hazard(rel, lineno, "stale_dispatch_authorization_doc", message))
    return hazards


def scan_text(path: Path, text: str, *, root: Path) -> list[Hazard]:
    rel = _repo_rel(path, root)
    hazards: list[Hazard] = []
    numbered = strip_heredoc_bodies(text)

    for lineno, command in _logical_commands(numbered):
        _add_flag_hazards(rel, lineno, command, hazards)
        if (
            re.search(r"(^|[\s;|&({])find\s+[^#\n]*\s-printf\b", command)
            and not _find_printf_is_remote_linux_context(command)
        ):
            hazards.append(
                Hazard(
                    rel,
                    lineno,
                    "macos_find_printf",
                    "GNU find -printf is not available on macOS; use Python pathlib/stat "
                    "or a POSIX/BSD-safe form for local commands",
                )
            )

    if path.suffix.lower() in {".md", ".rst", ".txt", ".zsh", ".sh", ".bash"}:
        for lineno, line in numbered:
            if re.search(r"(^|[;&|]\s*)(local\s+)?path=", line) or re.search(
                r"\bfor\s+path\s+in\b", line
            ) or re.search(r"\bread\b[^\n#]*\bpath\b", line):
                hazards.append(
                    Hazard(
                        rel,
                        lineno,
                        "zsh_path_special_variable",
                        "avoid shell variable name 'path' in zsh-facing snippets; "
                        "it mutates command lookup",
                    )
                )
        hazards.extend(_scan_shell_pythonpath_leaks(rel, numbered))
        hazards.extend(_scan_stale_dispatch_authorization(rel, numbered))

    if path.suffix.lower() == ".py":
        hazards.extend(_scan_python_dispatch_path_leaks(rel, text))

    return hazards


def scan_paths(
    root: Path,
    scan_paths: tuple[str, ...] = DEFAULT_SCAN_PATHS,
    excludes: tuple[str, ...] = DEFAULT_EXCLUDES,
) -> list[Hazard]:
    hazards: list[Hazard] = []
    for file_path in iter_scan_files(root, scan_paths=scan_paths, excludes=excludes):
        try:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        hazards.extend(scan_text(file_path, text, root=root))
    return hazards


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--scan-path", action="append", default=[])
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--strict", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.repo_root.resolve()
    scan_roots = tuple(args.scan_path) if args.scan_path else DEFAULT_SCAN_PATHS
    hazards = scan_paths(root, scan_paths=scan_roots)

    payload = {
        "schema": "dispatch_cli_shell_hazards_v1",
        "repo_root": str(root),
        "scan_paths": list(scan_roots),
        "hazard_count": len(hazards),
        "hazards": [asdict(hazard) for hazard in hazards],
    }
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json_text(payload), encoding="utf-8")

    for hazard in hazards:
        print(f"{hazard.path}:{hazard.line}: {hazard.kind}: {hazard.message}", file=sys.stderr)
    if hazards and args.strict:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
