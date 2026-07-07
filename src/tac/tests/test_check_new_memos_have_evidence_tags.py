"""Tests for check_new_memos_have_evidence_tags (WARN-ONLY evidence-tag gate).

Manual §5 + CLAUDE.md "Forbidden empirical-claim-without-evidence-tag": new
.omx/research memos with score-like claims need evidence vocabulary in the same
paragraph. Scope: .omx/research/*.md ONLY, dated >= 2026-07-08 (filename date;
mtime fallback). Same-line ``# EVIDENCE_TAG_OK:<rationale>`` waiver.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tac.preflight import PreflightError, check_new_memos_have_evidence_tags


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    (tmp_path / ".omx" / "research").mkdir(parents=True)
    return tmp_path


def _memo(repo: Path, name: str, text: str, *, old_mtime: bool = False) -> Path:
    path = repo / ".omx" / "research" / name
    path.write_text(text, encoding="utf-8")
    if old_mtime:
        # 2026-01-01 UTC — safely before the 2026-07-08 cutoff.
        os.utime(path, (1767225600, 1767225600))
    return path


NEW = "memo_20260710.md"
OLD = "memo_20260601.md"


def test_untagged_dseg_claim_flagged(repo: Path) -> None:
    _memo(repo, NEW, "The run landed d_seg = 0.0047 on the witness surface.\n")
    violations = check_new_memos_have_evidence_tags(repo_root=repo)
    assert len(violations) == 1
    assert "d_seg = 0.0047" in violations[0]


@pytest.mark.parametrize(
    "claim",
    [
        "d_pose ≈ 3.4e-5 after conditioning.",
        "Final S = 0.1911 on this arm.",
        "ΔS: 0.004 from the basis lever.",
        "This lever saves 18.5% of archive bytes.",
        "The new config beats the ancestor run.",
    ],
)
def test_claim_pattern_variants_flagged(repo: Path, claim: str) -> None:
    _memo(repo, NEW, claim + "\n")
    assert check_new_memos_have_evidence_tags(repo_root=repo), claim


@pytest.mark.parametrize(
    "vocab",
    [
        "MEASURED on n600",
        "DERIVED from the rate formula",
        "INFERRED from literature",
        "ASSUMED awaiting verification",
        "[empirical:reports/raw/x.json]",
        "[prediction]",
        "[advisory only]",
        "OPERATOR-STATED",
        "estimated",
        "nominal",
    ],
)
def test_evidence_vocabulary_in_paragraph_passes(repo: Path, vocab: str) -> None:
    _memo(repo, NEW, f"d_seg = 0.0047 on the witness ({vocab}).\n")
    assert check_new_memos_have_evidence_tags(repo_root=repo) == []


def test_vocabulary_elsewhere_in_same_paragraph_passes(repo: Path) -> None:
    _memo(
        repo, NEW,
        "The basis lever landed d_seg = 0.0047.\n"
        "This number was MEASURED through R on n600.\n",
    )
    assert check_new_memos_have_evidence_tags(repo_root=repo) == []


def test_vocabulary_in_other_paragraph_does_not_cover(repo: Path) -> None:
    _memo(
        repo, NEW,
        "Everything below was MEASURED carefully.\n\n"
        "The basis lever landed d_seg = 0.0047.\n",
    )
    assert len(check_new_memos_have_evidence_tags(repo_root=repo)) == 1


def test_same_line_waiver_respected_placeholder_rejected(repo: Path) -> None:
    _memo(
        repo, NEW,
        "d_seg = 0.0047 here.  # EVIDENCE_TAG_OK:illustrative-example-not-a-claim\n",
    )
    assert check_new_memos_have_evidence_tags(repo_root=repo) == []
    _memo(repo, "memo2_20260710.md", "d_seg = 0.0047.  # EVIDENCE_TAG_OK:<rationale>\n")
    assert len(check_new_memos_have_evidence_tags(repo_root=repo)) == 1


def test_pre_cutoff_filename_date_exempt(repo: Path) -> None:
    _memo(repo, OLD, "d_seg = 0.0047 with no tag at all.\n")
    assert check_new_memos_have_evidence_tags(repo_root=repo) == []


def test_undated_filename_uses_mtime(repo: Path) -> None:
    _memo(repo, "undated_memo.md", "d_seg = 0.0047 with no tag.\n", old_mtime=True)
    assert check_new_memos_have_evidence_tags(repo_root=repo) == []
    fresh = _memo(repo, "undated_memo_fresh.md", "d_seg = 0.0047 with no tag.\n")
    # Pin mtime to 2026-08-01 UTC (past the 2026-07-08 cutoff) so the test does not
    # depend on the wall clock crossing the cutoff.
    os.utime(fresh, (1785542400, 1785542400))
    assert len(check_new_memos_have_evidence_tags(repo_root=repo)) == 1


def test_scope_is_omx_research_only(repo: Path) -> None:
    # docs/ + equations-registry-style surfaces are OUT of scope by construction.
    (repo / "docs").mkdir()
    (repo / "docs" / "note_20260710.md").write_text("d_seg = 0.0047\n", encoding="utf-8")
    sub = repo / ".omx" / "research" / "subdir"
    sub.mkdir()
    (sub / "nested_20260710.md").write_text("d_seg = 0.0047\n", encoding="utf-8")
    assert check_new_memos_have_evidence_tags(repo_root=repo) == []


def test_one_violation_per_paragraph(repo: Path) -> None:
    _memo(repo, NEW, "d_seg = 0.0047 and also d_pose = 0.002 in one paragraph.\n")
    assert len(check_new_memos_have_evidence_tags(repo_root=repo)) == 1


def test_strict_raises_warn_returns(repo: Path) -> None:
    _memo(repo, NEW, "d_seg = 0.0047 untagged.\n")
    assert check_new_memos_have_evidence_tags(repo_root=repo, strict=False)
    with pytest.raises(PreflightError, match="evidence-tag"):
        check_new_memos_have_evidence_tags(repo_root=repo, strict=True)


def test_clean_repo_no_research_dir(tmp_path: Path) -> None:
    assert check_new_memos_have_evidence_tags(repo_root=tmp_path) == []


def test_plain_prose_numbers_not_flagged(repo: Path) -> None:
    _memo(
        repo, NEW,
        "We ran 600 pairs over 3 stages; the archive is 178417 bytes.\n",
    )
    assert check_new_memos_have_evidence_tags(repo_root=repo) == []
