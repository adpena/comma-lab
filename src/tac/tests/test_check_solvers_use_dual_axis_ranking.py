# SPDX-License-Identifier: MIT
"""Tests for the dual-axis solver ranking preflight check."""
from __future__ import annotations

import pytest

from tac.preflight import (
    PreflightError,
    check_solvers_use_dual_axis_ranking,
)


def test_violations_returned_for_primary_only_row(tmp_path) -> None:
    """A solver row with only ``predicted_score_delta`` and no companion
    flags is a violation."""
    solver_dir = tmp_path / "src" / "tac" / "optimization"
    solver_dir.mkdir(parents=True)
    (solver_dir / "fake_solver.py").write_text(
        '"""Fake solver."""\n'
        "def rank():\n"
        "    return [{'predicted_score_delta': 0.1}]\n",
        encoding="utf-8",
    )
    violations = check_solvers_use_dual_axis_ranking(
        repo_root=tmp_path,
        strict=False,
        verbose=False,
    )
    assert any("predicted_score_delta" in v for v in violations)


def test_strict_mode_raises_on_violations(tmp_path) -> None:
    solver_dir = tmp_path / "src" / "tac" / "optimization"
    solver_dir.mkdir(parents=True)
    (solver_dir / "fake_solver.py").write_text(
        "predicted_score_delta = 0.1\n",
        encoding="utf-8",
    )
    with pytest.raises(PreflightError):
        check_solvers_use_dual_axis_ranking(
            repo_root=tmp_path,
            strict=True,
            verbose=False,
        )


def test_companion_key_in_same_row_dismisses_violation(tmp_path) -> None:
    """Dual-axis companion keys must live in the same emitted row."""
    solver_dir = tmp_path / "src" / "tac" / "optimization"
    solver_dir.mkdir(parents=True)
    (solver_dir / "good_solver.py").write_text(
        "def rank():\n"
        "    return [{\n"
        "        'predicted_score_delta': 0.1,\n"
        "        'predicted_cuda_score': 0.21,\n"
        "        'predicted_cpu_score': 0.16,\n"
        "    }]\n",
        encoding="utf-8",
    )
    violations = check_solvers_use_dual_axis_ranking(
        repo_root=tmp_path,
        strict=False,
        verbose=False,
    )
    assert violations == []


def test_companion_key_elsewhere_in_file_does_not_dismiss_bad_row(tmp_path) -> None:
    """A good row must not waive a later primary-only row in the same file."""
    solver_dir = tmp_path / "src" / "tac" / "optimization"
    solver_dir.mkdir(parents=True)
    (solver_dir / "mixed_solver.py").write_text(
        "def rank():\n"
        "    return [\n"
        "        {'predicted_score_delta': 0.1, 'predicted_cpu_score': 0.16},\n"
        "        {'name': 'bad', 'predicted_score_delta': 0.2},\n"
        "    ]\n",
        encoding="utf-8",
    )
    violations = check_solvers_use_dual_axis_ranking(
        repo_root=tmp_path,
        strict=False,
        verbose=False,
    )
    assert len(violations) == 2, violations
    assert all("predicted_score_delta" in v for v in violations)
    assert any("primary-only" in v for v in violations)


def test_subscript_assignment_to_primary_key_is_caught(tmp_path) -> None:
    solver_dir = tmp_path / "src" / "tac" / "optimization"
    solver_dir.mkdir(parents=True)
    (solver_dir / "subscript_solver.py").write_text(
        "def rank(row):\n"
        "    row['predicted_score_delta'] = 0.1\n"
        "    return row\n",
        encoding="utf-8",
    )
    violations = check_solvers_use_dual_axis_ranking(
        repo_root=tmp_path,
        strict=False,
        verbose=False,
    )
    assert len(violations) == 1
    assert "predicted_score_delta" in violations[0]


def test_string_key_assignment_is_not_misclassified_as_prose(tmp_path) -> None:
    solver_dir = tmp_path / "src" / "tac" / "optimization"
    solver_dir.mkdir(parents=True)
    (solver_dir / "assign_solver.py").write_text(
        "def rank(row):\n"
        "    key = 'predicted_score_delta'\n"
        "    return key\n",
        encoding="utf-8",
    )
    violations = check_solvers_use_dual_axis_ranking(
        repo_root=tmp_path,
        strict=False,
        verbose=False,
    )
    assert len(violations) == 1
    assert "predicted_score_delta" in violations[0]


def test_waiver_marker_dismisses_violation(tmp_path) -> None:
    """``# DUAL_AXIS_RANKING_WAIVED:<reason>`` waives a same-line offender."""
    solver_dir = tmp_path / "src" / "tac" / "optimization"
    solver_dir.mkdir(parents=True)
    (solver_dir / "waived_solver.py").write_text(
        "predicted_score_delta = 0.1  # DUAL_AXIS_RANKING_WAIVED:legacy_test_only\n",
        encoding="utf-8",
    )
    violations = check_solvers_use_dual_axis_ranking(
        repo_root=tmp_path,
        strict=False,
        verbose=False,
    )
    assert violations == []


def test_clean_repo_returns_no_violations(tmp_path) -> None:
    """Empty surfaces → no violations."""
    violations = check_solvers_use_dual_axis_ranking(
        repo_root=tmp_path,
        strict=False,
        verbose=False,
    )
    assert violations == []


def test_real_repo_warn_only_does_not_raise() -> None:
    """The real repo is clean enough for strict dual-axis enforcement."""
    violations = check_solvers_use_dual_axis_ranking(
        strict=True,
        verbose=False,
    )
    assert violations == []
