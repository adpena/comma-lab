#!/usr/bin/env python3
"""Scan for dispatch CLI and shell portability hazards.

This guard covers sprint-expensive bug classes:

* adjudicator-only flags accidentally passed to ``launch_lightning_batch_job.py``;
* typo flags known to have caused dead dispatches;
* zsh-facing snippets using ``path`` as a shell variable;
* local/macOS snippets using GNU-only ``find -printf``.

Historical ledgers and result artifacts are excluded by default. Python
heredocs inside shell snippets are ignored so ordinary Python variables named
``path`` do not trigger shell warnings.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

DEFAULT_SCAN_PATHS = ("scripts", "docs", "reports")
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


@dataclass(frozen=True)
class Hazard:
    path: str
    line: int
    kind: str
    message: str


def _repo_rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _is_excluded(path: Path, root: Path, excludes: tuple[str, ...]) -> bool:
    rel = _repo_rel(path, root)
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
        candidates = [base] if base.is_file() else sorted(base.rglob("*"))
        for candidate in candidates:
            if not candidate.is_file():
                continue
            if _is_excluded(candidate, root, excludes):
                continue
            if candidate.suffix.lower() not in TEXT_SUFFIXES:
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


def scan_text(path: Path, text: str, *, root: Path) -> list[Hazard]:
    rel = _repo_rel(path, root)
    hazards: list[Hazard] = []
    numbered = strip_heredoc_bodies(text)

    for lineno, command in _logical_commands(numbered):
        _add_flag_hazards(rel, lineno, command, hazards)
        if "find " in command and "-printf" in command and not _find_printf_is_remote_linux_context(command):
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
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    for hazard in hazards:
        print(f"{hazard.path}:{hazard.line}: {hazard.kind}: {hazard.message}", file=sys.stderr)
    if hazards and args.strict:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
