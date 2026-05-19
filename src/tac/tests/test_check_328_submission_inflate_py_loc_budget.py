# SPDX-License-Identifier: MIT
"""Catalog #328 coverage for submission inflate.py LOC budget auditing."""
from __future__ import annotations

import inspect
import subprocess
import sys
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
    assert findings[0].budget_tier == "hard_budget"
    assert findings[0].severity == "violation"
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


def test_scan_submission_inflate_py_loc_budget_flags_default_review_target(
    tmp_path: Path,
) -> None:
    _write_inflate(tmp_path, "medium", 101)

    findings = scan_submission_inflate_py_loc_budget(tmp_path, max_lines=200)

    assert len(findings) == 1
    assert findings[0].budget_tier == "default_budget"
    assert findings[0].severity == "warn"
    assert findings[0].review_target_lines == 100
    assert "INFLATE_LOC_DEFAULT_BUDGET_WAIVED" in findings[0].format()


def test_scan_submission_inflate_py_loc_budget_accepts_default_budget_waiver(
    tmp_path: Path,
) -> None:
    _write_inflate(
        tmp_path,
        "medium",
        150,
        waiver="# INFLATE_LOC_DEFAULT_BUDGET_WAIVED: reviewable runtime split pending",
    )

    assert scan_submission_inflate_py_loc_budget(tmp_path, max_lines=200) == []


def test_scan_submission_inflate_py_loc_budget_default_waiver_not_hard_waiver(
    tmp_path: Path,
) -> None:
    _write_inflate(
        tmp_path,
        "too_large",
        220,
        waiver="# INFLATE_LOC_DEFAULT_BUDGET_WAIVED: reviewable runtime split pending",
    )

    findings = scan_submission_inflate_py_loc_budget(tmp_path, max_lines=200)

    assert len(findings) == 1
    assert findings[0].budget_tier == "hard_budget"


def test_scan_submission_inflate_py_loc_budget_classifies_size_drivers(
    tmp_path: Path,
) -> None:
    target = _write_inflate(tmp_path, "large", 210)
    target.write_text(
        "\n".join(
            [
                "import torch",
                "import brotli",
                "def run(file_list):",
                "    state_dict = torch.load('state.pt')",
                "    for name in file_list:",
                "        pass",
            ]
            + [f"# filler {idx}" for idx in range(204)]
        ),
        encoding="utf-8",
    )

    finding = scan_submission_inflate_py_loc_budget(tmp_path, max_lines=200)[0]

    assert "state_dict_loader" in finding.size_driver_categories
    assert "compressed_payload_decode" in finding.size_driver_categories
    assert "shared_state_dict_loader_with_sha256" in finding.technique_applicability
    assert finding.shared_runtime_helper_adopted is False


def test_scan_submission_inflate_py_loc_budget_ignores_docstring_helper_mentions(
    tmp_path: Path,
) -> None:
    target = _write_inflate(tmp_path, "docstring_only", 210)
    target.write_text(
        '"""Mentions tac.substrates._shared.inflate_runtime but never imports it."""\n'
        + "\n".join(f"# filler {idx}" for idx in range(209)),
        encoding="utf-8",
    )

    finding = scan_submission_inflate_py_loc_budget(tmp_path, max_lines=200)[0]

    assert finding.shared_runtime_helper_adopted is False


def test_scan_submission_inflate_py_loc_budget_detects_real_shared_import(
    tmp_path: Path,
) -> None:
    target = _write_inflate(tmp_path, "shared_import", 210)
    target.write_text(
        "from tac.substrates._shared.inflate_runtime import select_inflate_device\n"
        + "\n".join(f"# filler {idx}" for idx in range(209)),
        encoding="utf-8",
    )

    finding = scan_submission_inflate_py_loc_budget(tmp_path, max_lines=200)[0]

    assert finding.shared_runtime_helper_adopted is True


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


def test_legacy_audit_inflate_py_loc_budget_wrapper_help() -> None:
    result = subprocess.run(
        [sys.executable, "tools/audit_inflate_py_loc_budget.py", "--help"],
        check=True,
        cwd=Path(__file__).resolve().parents[3],
        text=True,
        capture_output=True,
    )
    assert "Audit direct submission" in result.stdout


def test_audit_inflate_py_loc_budget_accepts_explicit_summary_flag() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "tools/audit_inflate_py_loc_budget.py",
            "--repo-root",
            str(Path(__file__).resolve().parents[3]),
            "--summary",
        ],
        check=False,
        cwd=Path(__file__).resolve().parents[3],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0
    assert "[inflate-py-loc-budget]" in (result.stdout + result.stderr)


def test_audit_inflate_py_loc_budget_json_includes_classification_fields() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "tools/audit_inflate_py_loc_budget.py",
            "--repo-root",
            str(Path(__file__).resolve().parents[3]),
            "--json",
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[3],
        text=True,
        capture_output=True,
    )
    assert "default_budget_warning_count" in result.stdout
    assert "hard_budget_violation_count" in result.stdout
    assert "technique_applicability" in result.stdout
