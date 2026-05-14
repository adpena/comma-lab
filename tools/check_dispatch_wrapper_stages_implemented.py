#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""PCC11 — Catch dispatch wrappers whose `# Stage N:` comments aren't implemented.

Council Q5-B4 prescription. The wave_deploy script anti-pattern (comment-only
contract): a wrapper has `# Stage 5: contest-CUDA eval` followed only by
`echo "TODO"` — the comment promises behavior, the code does nothing. This
scanner walks every dispatch wrapper, finds stage labels, and asserts each
labelled stage has at least one non-trivial command within its window.

Scanned files:
    scripts/remote_*.sh
    scripts/wave_*.sh
    scripts/dispatch_*.sh
    scripts/launch_*.sh
    scripts/lightning_*.sh

A "stage label" matches the pattern (regex form, illustrative — see
``STAGE_LABEL_PATTERNS`` below for the canonical compiled regexes):

- ``# Stage N: ...``  (comment-style header)
- ``log "=== Stage N: ..."``  (logged echo of the stage)

A "non-trivial command" within the next N lines (default 50) is any line that:
  - is NOT a comment (#)
  - is NOT empty
  - is NOT a no-op (`:` or `true` or `false`)
  - is NOT a bare `echo` / `printf` / `log` (informational only)
  - DOES contain meaningful work like:
      * Python invocation: `python` / `$PYBIN` / `.venv/bin/python`
      * subprocess: `subprocess.run`, `bash`, `sh`
      * file ops: `cp`, `mv`, `rm`, `mkdir`, `tar`, `zip`, `find`
      * network: `curl`, `wget`, `ssh`, `scp`, `rsync`, `git`
      * compute: `nvidia-smi`, `torch`, `nvcc`

Exit codes:
    0    no violations
    1    violations found (only when --strict)
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT_DEFAULT = Path(__file__).resolve().parents[1]

WRAPPER_GLOBS = (
    "scripts/remote_*.sh",
    "scripts/wave_*.sh",
    "scripts/dispatch_*.sh",
    "scripts/launch_*.sh",
    "scripts/lightning_*.sh",
)

STAGE_LABEL_PATTERNS = (
    re.compile(r"^\s*#+\s*(?:===|---)?\s*Stage\s+(\d+|[A-Z])(?:\s*[:.\-]|\s+)", re.IGNORECASE),
    re.compile(r"""^\s*log\s+["']===\s*Stage\s+(\d+|[A-Z])""", re.IGNORECASE),
)

# Patterns that count as a non-trivial command (any of these).
# Substring-based (not word-boundary) because shell often quotes vars like
# `"$PYBIN" -c "..."` where `\b` doesn't work.
NONTRIVIAL_PATTERNS = (
    re.compile(r"PYBIN"),                                # any reference to $PYBIN
    re.compile(r"\.venv/bin/python"),
    re.compile(r"(?:^|\s|[\"'/])python[3]?\s+\S"),       # python invocation
    re.compile(r"subprocess\.(run|check_call|check_output|Popen)"),
    re.compile(r"(?:^|\s|;|&&|\|\|)(bash|sh|zsh)\s+\S"),
    re.compile(r"(?:^|\s|;|&&|\|\|)(cp|mv|rm|mkdir|tar|zip|unzip|find|ls|cat|sha256sum|md5sum|stat|du)\s+\S"),
    re.compile(r"(?:^|\s|;|&&|\|\|)(curl|wget|ssh|scp|rsync|git|gh)\s+\S"),
    re.compile(r"(?:^|\s)(nvidia-smi|nvcc|ffmpeg|gcc|cc|make|cmake)\b"),
    re.compile(r"(?:^|\s|;|&&|\|\|)(uv|pip|pip3|poetry|conda)\s+\S"),
    re.compile(r"(?:^|\s|;|&&|\|\|)exec\s+\S"),
    re.compile(r"(?:^|\s|;|&&|\|\|)(eval|source)\s+\S"),
    # Conditional with command body (when bash if/while/case has actual logic)
    re.compile(r"^\s*(if|elif|while|until)\s+"),
    re.compile(r"^\s*(case)\s+"),
    # Function definition or call with args
    re.compile(r"^\s*\w+\s*\(\s*\)\s*\{"),
    re.compile(r"^\s*\w+\s+[A-Za-z0-9_$./-]"),  # `bootstrap_runtime_deps "$ARG"` style
    # Assignment with command-substitution (real work)
    re.compile(r"=\$\("),
    re.compile(r"=\`"),
    # Heredoc (typical for embedded python)
    re.compile(r"<<-?\s*['\"]?[A-Z]"),
    # Pipe with non-trivial right-hand side
    re.compile(r"\|\s+(tee|grep|awk|sed|jq)\s+\S"),
)

# Patterns we EXCLUDE from "non-trivial" — informational only.
TRIVIAL_PATTERNS = (
    re.compile(r"^\s*echo\s+\".+\"\s*$"),
    re.compile(r"^\s*printf\s+\".+\"\s*$"),
    re.compile(r"^\s*log\s+\"[^\"]*\"\s*$"),
    re.compile(r"^\s*:\s*$"),
    re.compile(r"^\s*(true|false)\s*$"),
)


@dataclass(frozen=True)
class Violation:
    path: str
    stage_lineno: int
    stage_label: str
    reason: str


def _is_trivial_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return True
    return any(p.match(line) for p in TRIVIAL_PATTERNS)


def _is_nontrivial_command(line: str) -> bool:
    if _is_trivial_line(line):
        return False
    return any(p.search(line) for p in NONTRIVIAL_PATTERNS)


def _scan_file(path: Path, window: int) -> list[Violation]:
    """For each `# Stage N: <label>` in the file, verify a non-trivial command
    exists within the next `window` lines (or before the next stage label,
    whichever is sooner)."""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []
    lines = text.splitlines()
    violations: list[Violation] = []

    # Pre-find all stage label line numbers + extracted stage IDs.
    # Skip stage labels in the file's "header docstring" — typically the first
    # ~50 lines containing only comments + shebang. These are documentation
    # tables, not runtime stage labels.
    docstring_end = 0
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if i > 0 and stripped and not stripped.startswith("#") and not stripped.startswith("set "):
            docstring_end = i
            break
    # Cap docstring header detection at line 60 (heuristic).
    docstring_end = min(docstring_end, 60)

    # Skip stage labels containing intentional-no-op markers.
    # The corpus surfaced these legitimate patterns: SKIPPED (explicit skip),
    # STUB (planned future work explicitly labelled), "ready at" (announcement
    # of artifact built in prior stage), "nothing additional to do" / "verified
    # above" (work-already-done sentinels), "manual" (operator action stages).
    SKIP_MARKERS = re.compile(
        r"\b(SKIPPED|SKIP|NO-?OP|DEFERRED|DISABLED|STUB|TODO|FIXME)\b"
        r"|ready\s+at\b"
        r"|nothing\s+additional\s+to\s+do"
        r"|verified\s+above"
        r"|already\s+(?:the\s+)?(?:final|done|built|present)"
        r"|\bmanual\b",
        re.IGNORECASE,
    )

    stage_lines: list[tuple[int, str, str]] = []  # (lineno, label_text, stage_id)
    for i, line in enumerate(lines):
        if i < docstring_end:
            continue  # in header docstring; not a runtime stage label
        if SKIP_MARKERS.search(line):
            continue  # explicit no-op marker
        for pat in STAGE_LABEL_PATTERNS:
            m = pat.match(line)
            if m:
                stage_lines.append((i, line.strip(), m.group(1).upper()))
                break

    # Dedupe consecutive stage labels with the SAME stage ID within 10 lines —
    # these are typically a header comment followed by the matching log echo.
    # Keep the LAST label of the run; the body follows it.
    deduped: list[tuple[int, str, str]] = []
    i = 0
    while i < len(stage_lines):
        lineno, label, stage_id = stage_lines[i]
        # Look ahead for the same stage_id within 10 lines.
        j = i
        while j + 1 < len(stage_lines):
            next_lineno, _, next_id = stage_lines[j + 1]
            if next_id == stage_id and next_lineno - stage_lines[j][0] <= 60:
                j += 1
            else:
                break
        # Use the LAST label in the run as the canonical stage start.
        deduped.append(stage_lines[j])
        i = j + 1
    stage_lines = deduped

    for idx, (lineno, label, _stage_id) in enumerate(stage_lines):
        # Window is bounded by next stage label or `window` lines, whichever closer.
        next_stage_lineno = stage_lines[idx + 1][0] if idx + 1 < len(stage_lines) else len(lines)
        end = min(lineno + 1 + window, next_stage_lineno, len(lines))
        body = lines[lineno + 1 : end]
        if not any(_is_nontrivial_command(l) for l in body):
            violations.append(Violation(
                path=str(path),
                stage_lineno=lineno + 1,  # 1-based for editor display
                stage_label=label[:120],
                reason=(
                    f"Stage label at line {lineno + 1} has no non-trivial command in "
                    f"the next {end - lineno - 1} lines. The stage is comment-only — "
                    "per CLAUDE.md 'Comment-only contracts FORBIDDEN', back the comment "
                    "with at least one real command (python / subprocess / file ops / "
                    "network / compute), or remove the misleading label."
                ),
            ))
    return violations


def scan(repo_root: Path, window: int = 50) -> list[Violation]:
    violations: list[Violation] = []
    for glob in WRAPPER_GLOBS:
        for path in sorted(repo_root.glob(glob)):
            violations.extend(_scan_file(path, window))
    return violations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT_DEFAULT)
    parser.add_argument("--window", type=int, default=50,
                        help="lines to scan after a stage label (default 50)")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    violations = scan(args.repo_root, args.window)

    if args.json:
        import json
        print(json.dumps({
            "schema": "dispatch_wrapper_stages_implemented_v1",
            "violations": [
                {"path": v.path, "stage_lineno": v.stage_lineno,
                 "stage_label": v.stage_label, "reason": v.reason}
                for v in violations
            ],
            "count": len(violations),
        }, indent=2))
    else:
        for v in violations:
            print(f"{v.path}:{v.stage_lineno}: {v.stage_label}", file=sys.stderr)
            print(f"  → {v.reason}", file=sys.stderr)
        if violations:
            print(f"\n[PCC11] {len(violations)} comment-only stage label(s) found across "
                  f"{len({v.path for v in violations})} wrapper script(s).", file=sys.stderr)
        else:
            print("[PCC11] OK: 0 comment-only stage labels", file=sys.stderr)

    if violations and args.strict:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
