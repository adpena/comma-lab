# SPDX-License-Identifier: MIT
"""Regression tests for the package-runtime pyppmd import guard."""

from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight import PreflightError, check_no_unwaived_pyppmd_imports


def _write(root: Path, rel: str, text: str) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_unwaived_pyppmd_import_is_reported(tmp_path: Path) -> None:
    _write(tmp_path, "src/tac/bad_codec.py", "def f():\n    import pyppmd\n")

    violations = check_no_unwaived_pyppmd_imports(
        repo_root=tmp_path, strict=False, verbose=False
    )

    assert len(violations) == 1
    assert "bad_codec.py:2" in violations[0]
    assert "PYPPMD_LGPL_OK" in violations[0]


def test_pyppmd_import_same_line_waiver_is_accepted(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "src/tac/replay_codec.py",
        "def f():\n"
        "    import pyppmd  # PYPPMD_LGPL_OK:public-archive-replay-only\n",
    )

    assert (
        check_no_unwaived_pyppmd_imports(
            repo_root=tmp_path, strict=False, verbose=False
        )
        == []
    )


def test_pyppmd_import_scan_ignores_tests(tmp_path: Path) -> None:
    _write(tmp_path, "src/tac/tests/test_replay.py", "import pyppmd\n")

    assert (
        check_no_unwaived_pyppmd_imports(
            repo_root=tmp_path, strict=False, verbose=False
        )
        == []
    )


def test_unwaived_pyppmd_import_strict_raises(tmp_path: Path) -> None:
    _write(tmp_path, "src/tac/bad_codec.py", "from pyppmd import compress\n")

    with pytest.raises(PreflightError):
        check_no_unwaived_pyppmd_imports(
            repo_root=tmp_path, strict=True, verbose=False
        )


def test_live_pyppmd_import_guard_has_zero_violations() -> None:
    repo_root = Path(__file__).resolve().parents[3]

    assert (
        check_no_unwaived_pyppmd_imports(
            repo_root=repo_root, strict=False, verbose=False
        )
        == []
    )


def test_pyppmd_import_guard_wired_into_preflight_all() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    text = (repo_root / "src/tac/preflight.py").read_text(encoding="utf-8")

    assert "check_no_unwaived_pyppmd_imports(" in text
    assert "strict=True, verbose=verbose" in text
