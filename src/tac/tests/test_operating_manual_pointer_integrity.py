"""Tests for check_operating_manual_pointer_integrity (operating-manual anti-rot gate).

The gate verifies three invariants (all WARN-ONLY at the preflight_all wire-in):
  1. docs/operating_manual_craft_handoff.md exists;
  2. CLAUDE.md contains the pointer section header "The Operating Manual";
  3. the manual contains all 8 numbered sections ("## 1." .. "## 8.").
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_operating_manual_pointer_integrity,
)

_MANUAL_REL = Path("docs/operating_manual_craft_handoff.md")


def _make_repo(tmp_path: Path, *, manual: bool = True, pointer: bool = True,
               sections: int = 8) -> Path:
    """Build a minimal fake repo satisfying (or violating) the three invariants."""
    root = tmp_path / "repo"
    (root / "docs").mkdir(parents=True)
    if manual:
        body = ["# The Operating Manual — a craft handoff", ""]
        for n in range(1, sections + 1):
            body += [f"## {n}. Section {n}", "", "text", ""]
        (root / _MANUAL_REL).write_text("\n".join(body), encoding="utf-8")
    claude = "# AGENTS\n\n"
    if pointer:
        claude += "## The Operating Manual — craft handoff — NON-NEGOTIABLE READ\n\npointer body\n"
    (root / "CLAUDE.md").write_text(claude, encoding="utf-8")
    return root


def test_live_repo_passes() -> None:
    """The real repo must be clean (manual + pointer + 8 sections all present)."""
    violations = check_operating_manual_pointer_integrity(strict=False, verbose=False)
    assert violations == []


def test_complete_fake_repo_passes(tmp_path: Path) -> None:
    root = _make_repo(tmp_path)
    assert check_operating_manual_pointer_integrity(repo_root=root) == []


def test_missing_manual_flagged(tmp_path: Path) -> None:
    root = _make_repo(tmp_path, manual=False)
    violations = check_operating_manual_pointer_integrity(repo_root=root)
    assert len(violations) == 1
    assert "MISSING" in violations[0]
    assert str(_MANUAL_REL) in violations[0]


def test_missing_numbered_section_flagged(tmp_path: Path) -> None:
    root = _make_repo(tmp_path, sections=7)  # drops "## 8."
    violations = check_operating_manual_pointer_integrity(repo_root=root)
    assert len(violations) == 1
    assert "[8]" in violations[0]


def test_missing_claude_pointer_flagged(tmp_path: Path) -> None:
    root = _make_repo(tmp_path, pointer=False)
    violations = check_operating_manual_pointer_integrity(repo_root=root)
    assert len(violations) == 1
    assert "The Operating Manual" in violations[0]


def test_strict_raises_on_violation(tmp_path: Path) -> None:
    root = _make_repo(tmp_path, manual=False, pointer=False)
    with pytest.raises(PreflightError):
        check_operating_manual_pointer_integrity(repo_root=root, strict=True)


def test_non_strict_returns_all_violations(tmp_path: Path) -> None:
    root = _make_repo(tmp_path, manual=False, pointer=False)
    violations = check_operating_manual_pointer_integrity(repo_root=root, strict=False)
    # missing manual + missing pointer header = 2 independent violations
    assert len(violations) == 2
