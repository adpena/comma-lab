# SPDX-License-Identifier: MIT
"""Catalog #222 tests for canonical scorer loader tuple assignment order."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_scorer_loader_assignment_order,
    preflight_all,
)


def _repo(tmp_path: Path, rel: str, text: str) -> Path:
    for dirname in ("src/tac", "experiments", "tools"):
        (tmp_path / dirname).mkdir(parents=True, exist_ok=True)
    path = tmp_path / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return tmp_path


def test_check222_rejects_reversed_differentiable_loader_assignment(
    tmp_path: Path,
) -> None:
    root = _repo(
        tmp_path,
        "experiments/bad.py",
        "from tac.scorer import load_differentiable_scorers\n"
        "def main():\n"
        "    seg_scorer, pose_scorer = load_differentiable_scorers('upstream')\n",
    )

    violations = check_scorer_loader_assignment_order(
        repo_root=root,
        strict=False,
        verbose=False,
    )

    assert len(violations) == 1
    assert "reverses the canonical scorer loader return order" in violations[0]


def test_check222_rejects_reversed_default_loader_assignment(tmp_path: Path) -> None:
    root = _repo(
        tmp_path,
        "tools/bad.py",
        "def main():\n"
        "    segnet, posenet = scorer.load_default_scorers('upstream')\n",
    )

    violations = check_scorer_loader_assignment_order(
        repo_root=root,
        strict=False,
        verbose=False,
    )

    assert len(violations) == 1
    assert "segnet, posenet" in violations[0]


def test_check222_accepts_canonical_order(tmp_path: Path) -> None:
    root = _repo(
        tmp_path,
        "experiments/ok.py",
        "from tac.scorer import load_differentiable_scorers\n"
        "def main():\n"
        "    pose_scorer, seg_scorer = load_differentiable_scorers('upstream')\n",
    )

    assert check_scorer_loader_assignment_order(
        repo_root=root,
        strict=False,
        verbose=False,
    ) == []


def test_check222_same_line_waiver_accepts_nonstandard_wrapper(tmp_path: Path) -> None:
    root = _repo(
        tmp_path,
        "experiments/ok.py",
        "def custom_loader():\n"
        "    seg_scorer, pose_scorer = load_scorers('upstream')  "
        "# SCORER_LOADER_ORDER_OK: custom wrapper intentionally returns seg first\n",
    )

    assert check_scorer_loader_assignment_order(
        repo_root=root,
        strict=False,
        verbose=False,
    ) == []


def test_check222_strict_raises(tmp_path: Path) -> None:
    root = _repo(
        tmp_path,
        "experiments/bad.py",
        "def main():\n"
        "    seg_scorer, pose_scorer = load_differentiable_scorers('upstream')\n",
    )

    with pytest.raises(PreflightError, match="check_scorer_loader_assignment_order"):
        check_scorer_loader_assignment_order(
            repo_root=root,
            strict=True,
            verbose=False,
        )


def test_check222_live_repo_clean() -> None:
    assert check_scorer_loader_assignment_order(strict=False, verbose=False) == []


def test_check222_wired_into_preflight_all_strict() -> None:
    source = inspect.getsource(preflight_all)
    assert "check_scorer_loader_assignment_order" in source
    callsite = source.split("check_scorer_loader_assignment_order", 1)[1][:120]
    assert "strict=True" in callsite
