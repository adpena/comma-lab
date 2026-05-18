# SPDX-License-Identifier: MIT
"""Submission ``inflate.py`` source-size audit helpers.

The contest score charges archive bytes, not ``inflate.py`` bytes. This audit is
therefore a maintainability and review-surface guard, not score evidence. It
flags large submission runtimes so compression work is pointed at reusable
helpers, byte-closed archive payloads, and contest-relevant packers instead of
minifying Python source that the scorer never charges.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEFAULT_MAX_INFLATE_PY_LINES = 200
INFLATE_PY_LOC_BUDGET_WAIVER = "INFLATE_PY_LOC_BUDGET_OK:"
_WAIVER_PLACEHOLDERS = ("<rationale>", "<reason>")


@dataclass(frozen=True)
class InflatePyLocBudgetFinding:
    """One ``inflate.py`` whose physical line count exceeds the budget."""

    rel_path: str
    line_count: int
    max_lines: int

    def format(self) -> str:
        return (
            f"{self.rel_path}: {self.line_count} physical lines exceeds "
            f"max_lines={self.max_lines}. This is not score evidence because "
            "the contest rate term charges archive.zip bytes, not inflate.py "
            "source bytes. Extract reusable helpers or add "
            f"`# {INFLATE_PY_LOC_BUDGET_WAIVER}<rationale>` in the first "
            "40 lines for an intentional source-faithful/runtime-closure "
            "exception."
        )


def _physical_line_count(text: str) -> int:
    if not text:
        return 0
    return len(text.splitlines())


def _has_valid_loc_budget_waiver(text: str, *, first_n_lines: int = 40) -> bool:
    for line in text.splitlines()[:first_n_lines]:
        if INFLATE_PY_LOC_BUDGET_WAIVER not in line:
            continue
        tail = line.split(INFLATE_PY_LOC_BUDGET_WAIVER, 1)[1].strip()
        if not tail:
            continue
        lowered = tail.lower()
        if any(placeholder in lowered for placeholder in _WAIVER_PLACEHOLDERS):
            continue
        return True
    return False


def iter_submission_inflate_py_files(repo_root: Path | str) -> list[Path]:
    """Return tracked-submission ``inflate.py`` surfaces under ``submissions/``.

    Public PR intakes and generated experiment results are intentionally outside
    this helper. They are forensics or rebuildable custody, while the direct
    ``submissions/<lane>/inflate.py`` tree is the operator-facing runtime surface
    that preflight should keep small enough to review.
    """

    root = Path(repo_root)
    submissions_root = root / "submissions"
    if not submissions_root.is_dir():
        return []
    targets: list[Path] = []
    for path in submissions_root.rglob("inflate.py"):
        if "__pycache__" in path.parts:
            continue
        if "_intake_" in path.parts:
            continue
        targets.append(path)
    return sorted(targets)


def scan_submission_inflate_py_loc_budget(
    repo_root: Path | str,
    *,
    max_lines: int = DEFAULT_MAX_INFLATE_PY_LINES,
) -> list[InflatePyLocBudgetFinding]:
    """Find submission ``inflate.py`` files above ``max_lines``.

    The count is physical source lines, matching the operator-facing review
    burden and the existing symposium audit numbers. Syntax-invalid files are
    still countable; unreadable files are skipped so this audit does not become
    a filesystem-permission gate.
    """

    root = Path(repo_root).resolve()
    findings: list[InflatePyLocBudgetFinding] = []
    for path in iter_submission_inflate_py_files(root):
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        line_count = _physical_line_count(text)
        if line_count <= max_lines:
            continue
        if _has_valid_loc_budget_waiver(text):
            continue
        rel_path = str(path.relative_to(root))
        findings.append(
            InflatePyLocBudgetFinding(
                rel_path=rel_path,
                line_count=line_count,
                max_lines=max_lines,
            )
        )
    return findings
