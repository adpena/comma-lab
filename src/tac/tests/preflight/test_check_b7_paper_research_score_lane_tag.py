"""Tests for B7 — paper/research lane-tag extension.

``check_scores_have_lane_tag_paper_research`` extends the existing
``check_scores_have_lane_tag`` to docs/paper/**/*.md and
.omx/research/**/*.md surfaces.
"""
from __future__ import annotations

from pathlib import Path

from tac.preflight import check_scores_have_lane_tag_paper_research


def test_b7_flags_untagged_score_in_paper_md(tmp_path: Path) -> None:
    paper = tmp_path / "docs" / "paper" / "x.md"
    paper.parent.mkdir(parents=True, exist_ok=True)
    paper.write_text("score = 0.193\n")
    violations = check_scores_have_lane_tag_paper_research(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert len(violations) == 1


def test_b7_accepts_tagged_score(tmp_path: Path) -> None:
    paper = tmp_path / "docs" / "paper" / "x.md"
    paper.parent.mkdir(parents=True, exist_ok=True)
    paper.write_text("score = 0.193 [contest-CUDA]\n")
    violations = check_scores_have_lane_tag_paper_research(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert violations == []


def test_b7_scans_omx_research_too(tmp_path: Path) -> None:
    research = tmp_path / ".omx" / "research" / "memo.md"
    research.parent.mkdir(parents=True, exist_ok=True)
    research.write_text("score = 0.155\n")
    violations = check_scores_have_lane_tag_paper_research(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert len(violations) == 1


def test_b7_strict_raises(tmp_path: Path) -> None:
    paper = tmp_path / "docs" / "paper" / "x.md"
    paper.parent.mkdir(parents=True, exist_ok=True)
    paper.write_text("score = 0.193\n")
    import pytest
    from tac.preflight import MetaBugViolation
    with pytest.raises(MetaBugViolation):
        check_scores_have_lane_tag_paper_research(
            repo_root=tmp_path, strict=True, verbose=False,
        )


def test_b7_no_violations_when_dirs_missing(tmp_path: Path) -> None:
    violations = check_scores_have_lane_tag_paper_research(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert violations == []
