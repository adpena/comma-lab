# SPDX-License-Identifier: MIT
"""Catalog #328 coverage for submission inflate.py LOC budget auditing."""
from __future__ import annotations

import inspect
from pathlib import Path

import pytest

import tac.preflight as preflight_module
from tac.preflight import (
    PreflightError,
    check_submission_inflate_py_under_loc_budget,
)
from tac.submission_inflate_loc_budget import scan_submission_inflate_py_loc_budget


def _write_inflate(repo: Path, name: str, lines: int, *, waiver: str = "") -> Path:
    target = repo / "submissions" / name / "inflate.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    body = [waiver] if waiver else []
    body.extend(f"# line {i}" for i in range(max(0, lines - len(body))))
    target.write_text("\n".join(body) + "\n", encoding="utf-8")
    return target


def test_scan_submission_inflate_py_loc_budget_flags_large_runtime(tmp_path: Path) -> None:
    _write_inflate(tmp_path, "small", 3)
    _write_inflate(tmp_path, "large", 201)

    findings = scan_submission_inflate_py_loc_budget(tmp_path, max_lines=200)

    assert len(findings) == 1
    assert findings[0].rel_path == "submissions/large/inflate.py"
    assert findings[0].line_count == 201
    assert findings[0].max_lines == 200
    assert "archive.zip bytes, not inflate.py source bytes" in findings[0].format()


def test_scan_submission_inflate_py_loc_budget_accepts_explicit_waiver(
    tmp_path: Path,
) -> None:
    _write_inflate(
        tmp_path,
        "source_faithful",
        240,
        waiver=(
            "# INFLATE_PY_LOC_BUDGET_OK: source-faithful public runtime "
            "kept inline for byte-for-byte replay review"
        ),
    )

    assert scan_submission_inflate_py_loc_budget(tmp_path, max_lines=200) == []


def test_check_328_strict_raises_on_synthetic_violation(tmp_path: Path) -> None:
    _write_inflate(tmp_path, "large", 201)

    with pytest.raises(PreflightError, match="Catalog #328"):
        check_submission_inflate_py_under_loc_budget(
            repo_root=tmp_path,
            max_lines=200,
            strict=True,
            verbose=False,
        )


def test_check_328_wired_into_preflight_all_warn_only() -> None:
    source = inspect.getsource(preflight_module.preflight_all)
    idx = source.find("check_submission_inflate_py_under_loc_budget")
    assert idx >= 0, "Catalog #328 must be wired into preflight_all"
    call_window = source[idx : idx + 240]
    assert "strict=False" in call_window
