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


def test_companion_key_in_same_file_dismisses_violation(tmp_path) -> None:
    """If the file mentions any dual-axis companion key, all
    primary-key references are presumed paired."""
    solver_dir = tmp_path / "src" / "tac" / "optimization"
    solver_dir.mkdir(parents=True)
    (solver_dir / "good_solver.py").write_text(
        "predicted_score_delta = 0.1\n"
        "predicted_cpu_score = 0.16\n",  # companion key present
        encoding="utf-8",
    )
    violations = check_solvers_use_dual_axis_ranking(
        repo_root=tmp_path,
        strict=False,
        verbose=False,
    )
    assert violations == []


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
    """The check on the real repo runs in warn-only mode without raising
    (live count may be > 0 while the strict-flip is gated on backfill)."""
    # Don't pass repo_root so the default (REPO_ROOT) is used.
    violations = check_solvers_use_dual_axis_ranking(
        strict=False,
        verbose=False,
    )
    # The real-repo violation count is informational; we do not assert on
    # an exact number because it changes as files are backfilled.
    assert isinstance(violations, list)
