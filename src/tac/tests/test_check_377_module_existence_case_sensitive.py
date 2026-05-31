# SPDX-License-Identifier: MIT
"""Tests for Catalog #377 - check_module_existence_check_is_case_sensitive.

Sister of Catalog #205 (canonical select_inflate_device) at the
macOS case-fold runtime-tree-hash divergence surface. Empirical anchor:
PR111-candidate paired-CUDA RATIFICATION 4× DEFER 2026-05-28 (commit
``6bc74e074``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.preflight import (  # noqa: E402
    PreflightError,
    check_module_existence_check_is_case_sensitive,
)


def test_live_repo_passes() -> None:
    """The current repo MUST pass the gate (canonical fix landed in same batch)."""

    violations = check_module_existence_check_is_case_sensitive(strict=False)
    assert violations == [], (
        f"Catalog #377 regression: {len(violations)} violation(s); "
        f"first: {violations[0] if violations else None!r}"
    )


def test_gate_returns_empty_when_target_missing(tmp_path: Path) -> None:
    """Missing experiments/contest_auth_eval.py is silent (sister gates own existence)."""

    violations = check_module_existence_check_is_case_sensitive(
        repo_root=tmp_path, strict=False
    )
    assert violations == []


def test_gate_flags_missing_path_exists_case_sensitive_token(tmp_path: Path) -> None:
    """Drop _path_exists_case_sensitive → flagged."""

    target_dir = tmp_path / "experiments"
    target_dir.mkdir()
    target = target_dir / "contest_auth_eval.py"
    # iterdir token present, but _path_exists_case_sensitive missing.
    target.write_text(
        "from pathlib import Path\n"
        "def _module_exists(name, root):\n"
        "    for child in root.iterdir():\n"
        "        if child.name == name:\n"
        "            return True\n"
        "    return False\n"
    )
    violations = check_module_existence_check_is_case_sensitive(
        repo_root=tmp_path, strict=False
    )
    assert len(violations) == 1
    assert "_path_exists_case_sensitive" in violations[0]


def test_gate_flags_missing_iterdir_token(tmp_path: Path) -> None:
    """Drop iterdir → flagged (case-fold helper not wired)."""

    target_dir = tmp_path / "experiments"
    target_dir.mkdir()
    target = target_dir / "contest_auth_eval.py"
    target.write_text(
        "from pathlib import Path\n"
        "def _path_exists_case_sensitive(p):\n"
        "    return p.exists()  # WRONG: case-folds on macOS\n"
    )
    violations = check_module_existence_check_is_case_sensitive(
        repo_root=tmp_path, strict=False
    )
    assert len(violations) == 1
    assert "iterdir" in violations[0]


def test_gate_strict_raises_preflight_error(tmp_path: Path) -> None:
    """strict=True raises PreflightError with Catalog #377 message."""

    target_dir = tmp_path / "experiments"
    target_dir.mkdir()
    (target_dir / "contest_auth_eval.py").write_text("# no canonical helpers\n")
    with pytest.raises(PreflightError) as excinfo:
        check_module_existence_check_is_case_sensitive(
            repo_root=tmp_path, strict=True
        )
    assert "Catalog #377" in str(excinfo.value)
    assert "PR111" in str(excinfo.value)
    # Verify the canonical anchor empirical receipts surface:
    assert "macOS" in str(excinfo.value) or "1e9bf123" in str(excinfo.value)


def test_gate_strict_silent_on_clean_synthetic(tmp_path: Path) -> None:
    """strict=True with both canonical tokens present → no exception."""

    target_dir = tmp_path / "experiments"
    target_dir.mkdir()
    (target_dir / "contest_auth_eval.py").write_text(
        "from pathlib import Path\n"
        "def _path_exists_case_sensitive(p):\n"
        "    current = Path(p.parts[0])\n"
        "    for part in p.parts[1:]:\n"
        "        names = {c.name for c in current.iterdir()}\n"
        "        if part not in names:\n"
        "            return False\n"
        "        current = current / part\n"
        "    return True\n"
    )
    # Should NOT raise
    violations = check_module_existence_check_is_case_sensitive(
        repo_root=tmp_path, strict=True
    )
    assert violations == []


def test_gate_string_repo_root_accepted(tmp_path: Path) -> None:
    """repo_root accepted as str (auto-converted to Path)."""

    violations = check_module_existence_check_is_case_sensitive(
        repo_root=str(tmp_path), strict=False
    )
    assert violations == []
