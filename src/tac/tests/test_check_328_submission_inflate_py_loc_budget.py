# SPDX-License-Identifier: MIT
"""Reintroduction guards for the permanently retired Catalog #328 LOC cap."""
from __future__ import annotations

import inspect
import json
import subprocess
import sys
from pathlib import Path

import tac.preflight as preflight_module
from tac.preflight import check_submission_inflate_py_under_loc_budget
from tac.submission_inflate_loc_budget import scan_submission_inflate_py_loc_budget


def _write_inflate(repo: Path, *, lines: int = 500) -> Path:
    target = repo / "submissions" / "unsized" / "inflate.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "\n".join(f"# unrestricted interpreter line {index}" for index in range(lines))
        + "\n",
        encoding="utf-8",
    )
    return target


def test_scanner_never_flags_500_line_inflate_py(tmp_path: Path) -> None:
    """Permanent contract: physical source length has no audit authority."""

    _write_inflate(tmp_path, lines=500)
    assert scan_submission_inflate_py_loc_budget(
        tmp_path,
        max_lines=1,
        review_target_lines=1,
    ) == []


def test_compatibility_preflight_symbol_is_noop_even_when_strict(tmp_path: Path) -> None:
    _write_inflate(tmp_path, lines=500)
    assert check_submission_inflate_py_under_loc_budget(
        repo_root=tmp_path,
        max_lines=1,
        strict=True,
        verbose=True,
    ) == []


def test_loc_check_is_not_wired_into_preflight_all() -> None:
    source = inspect.getsource(preflight_module.preflight_all)
    assert "check_submission_inflate_py_under_loc_budget(" not in source


def test_legacy_audit_cli_is_informational_and_never_strict_fails(
    tmp_path: Path,
) -> None:
    _write_inflate(tmp_path, lines=500)
    result = subprocess.run(
        [
            sys.executable,
            "tools/audit_inflate_py_loc_budget.py",
            "--repo-root",
            str(tmp_path),
            "--max-lines",
            "1",
            "--strict",
            "--summary",
        ],
        check=False,
        cwd=Path(__file__).resolve().parents[3],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0
    assert "RETIRED" in (result.stdout + result.stderr)


def test_legacy_audit_json_labels_retired_contract(tmp_path: Path) -> None:
    _write_inflate(tmp_path, lines=500)
    result = subprocess.run(
        [
            sys.executable,
            "tools/audit_inflate_py_loc_budget.py",
            "--repo-root",
            str(tmp_path),
            "--max-lines",
            "1",
            "--strict",
            "--json",
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[3],
        text=True,
        capture_output=True,
    )
    payload = json.loads(result.stdout)
    assert payload["restriction_status"] == "permanently_removed_2026-07-21"
    assert payload["informational_only"] is True
    assert payload["finding_count"] == 0
    assert payload["hard_budget_violation_count"] == 0
    assert payload["default_budget_warning_count"] == 0
