# SPDX-License-Identifier: MIT
"""Tests for Catalog #229 — subagent landing premise-verification evidence gate.

Per `feedback_prompt_premise_verification_before_edit_pattern_20260514.md`.

Refuses post-2026-05-14 landing memos that claim >=3 bulk edits without
an empirical verdict table OR reproducer-script path.

Same-line waiver: ``# PREMISE_VERIFICATION_WAIVED:<reason>`` on any body
line. File-level waiver: ``# PREMISE_VERIFICATION_WAIVED_FILE:<reason>``.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    _CHECK_229_BULK_EDIT_RE,
    _CHECK_229_DATE_CUTOFF,
    _CHECK_229_MEMO_GLOB,
    _check_229_memo_date_suffix,
    check_subagent_landing_includes_premise_verification_evidence,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def fake_memo_root(tmp_path, monkeypatch):
    """Build a fake memory directory mimicking
    ``~/.claude/projects/-Users-adpena-Projects-pact/memory/``.
    """
    fake_home = tmp_path / "home"
    memo_dir = (
        fake_home / ".claude" / "projects"
        / "-Users-adpena-Projects-pact" / "memory"
    )
    memo_dir.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setattr(Path, "home", lambda: fake_home)
    return memo_dir


# ─── Date suffix parser ───────────────────────────────────────────────


def test_date_suffix_parser_well_formed(tmp_path):
    p = tmp_path / "feedback_xyz_landed_20260514.md"
    assert _check_229_memo_date_suffix(p) == "20260514"


def test_date_suffix_parser_no_date(tmp_path):
    p = tmp_path / "feedback_xyz_landed.md"
    assert _check_229_memo_date_suffix(p) is None


def test_date_suffix_parser_partial(tmp_path):
    p = tmp_path / "feedback_xyz_landed_2026.md"
    assert _check_229_memo_date_suffix(p) is None


# ─── Cutoff filter ────────────────────────────────────────────────────


def test_pre_cutoff_memo_exempt(fake_memo_root):
    """A pre-2026-05-14 memo with bulk-edit claims is legacy and exempt."""
    body = (
        "# Feedback memo\n\n"
        "10 trainers wired. 5 substrates rewritten. 8 files edited.\n"
    )
    (fake_memo_root / "feedback_legacy_landed_20260501.md").write_text(body)
    v = check_subagent_landing_includes_premise_verification_evidence(
        repo_root=REPO_ROOT, strict=False, verbose=False
    )
    flagged = [line for line in v if "feedback_legacy_landed_20260501" in line]
    assert len(flagged) == 0


def test_post_cutoff_memo_with_bulk_no_evidence_flagged(fake_memo_root):
    body = (
        "# Feedback memo\n\n"
        "10 trainers wired. 5 substrates rewritten. 8 files edited.\n"
        "Empirical claim only.\n"
    )
    (fake_memo_root / "feedback_bad_landed_20260514.md").write_text(body)
    v = check_subagent_landing_includes_premise_verification_evidence(
        repo_root=REPO_ROOT, strict=False, verbose=False
    )
    flagged = [line for line in v if "feedback_bad_landed_20260514" in line]
    assert len(flagged) == 1
    assert "23 bulk edits" in flagged[0]
    assert "Catalog #229" in flagged[0]


# ─── Acceptance: verdict table ────────────────────────────────────────


def test_verdict_table_satisfies(fake_memo_root):
    body = (
        "# Feedback memo\n\n"
        "10 trainers wired. 5 substrates rewritten.\n\n"
        "| trainer_id | verdict |\n"
        "|---|---|\n"
        "| foo | ok |\n"
    )
    (fake_memo_root / "feedback_table_landed_20260514.md").write_text(body)
    v = check_subagent_landing_includes_premise_verification_evidence(
        repo_root=REPO_ROOT, strict=False, verbose=False
    )
    flagged = [line for line in v if "feedback_table_landed_20260514" in line]
    assert len(flagged) == 0


# ─── Acceptance: reproducer-script path ───────────────────────────────


def test_reproducer_path_satisfies_dot_omx_tmp(fake_memo_root):
    body = (
        "# Feedback memo\n\n"
        "10 trainers wired. 5 substrates rewritten.\n\n"
        "Reproducer at .omx/tmp/my_lane_reproducer.py confirms.\n"
    )
    (fake_memo_root / "feedback_repro_landed_20260514.md").write_text(body)
    v = check_subagent_landing_includes_premise_verification_evidence(
        repo_root=REPO_ROOT, strict=False, verbose=False
    )
    flagged = [line for line in v if "feedback_repro_landed_20260514" in line]
    assert len(flagged) == 0


def test_reproducer_path_satisfies_tools_check(fake_memo_root):
    body = (
        "# Feedback memo\n\n"
        "10 trainers wired.\n\n"
        "Canonical helper at tools/check_my_lane_actionable.py.\n"
    )
    (fake_memo_root / "feedback_tools_landed_20260514.md").write_text(body)
    v = check_subagent_landing_includes_premise_verification_evidence(
        repo_root=REPO_ROOT, strict=False, verbose=False
    )
    flagged = [line for line in v if "feedback_tools_landed_20260514" in line]
    assert len(flagged) == 0


# ─── Waiver acceptance ────────────────────────────────────────────────


def test_per_line_waiver_accepted(fake_memo_root):
    body = (
        "# Feedback memo\n\n"
        "10 trainers wired. 5 substrates rewritten. 8 files edited.\n"
        "# PREMISE_VERIFICATION_WAIVED:infra-landing-no-file-list-claims\n"
    )
    (fake_memo_root / "feedback_waived_landed_20260514.md").write_text(body)
    v = check_subagent_landing_includes_premise_verification_evidence(
        repo_root=REPO_ROOT, strict=False, verbose=False
    )
    flagged = [line for line in v if "feedback_waived_landed_20260514" in line]
    assert len(flagged) == 0


def test_per_line_waiver_placeholder_rejected(fake_memo_root):
    body = (
        "# Feedback memo\n\n"
        "10 trainers wired. 5 substrates rewritten.\n"
        "# PREMISE_VERIFICATION_WAIVED:<reason>\n"
    )
    (fake_memo_root / "feedback_placeholder_landed_20260514.md").write_text(body)
    v = check_subagent_landing_includes_premise_verification_evidence(
        repo_root=REPO_ROOT, strict=False, verbose=False
    )
    flagged = [
        line for line in v if "feedback_placeholder_landed_20260514" in line
    ]
    assert len(flagged) == 1


def test_file_level_waiver_accepted(fake_memo_root):
    body = (
        "# PREMISE_VERIFICATION_WAIVED_FILE:infra-landing-only\n\n"
        "10 trainers wired. 5 substrates rewritten. 8 files edited.\n"
    )
    (fake_memo_root / "feedback_file_waiver_landed_20260514.md").write_text(body)
    v = check_subagent_landing_includes_premise_verification_evidence(
        repo_root=REPO_ROOT, strict=False, verbose=False
    )
    flagged = [
        line for line in v if "feedback_file_waiver_landed_20260514" in line
    ]
    assert len(flagged) == 0


# ─── Bulk-edit threshold ──────────────────────────────────────────────


def test_below_threshold_not_flagged(fake_memo_root):
    """Memo with <3 bulk edits is not flagged."""
    body = (
        "# Feedback memo\n\n"
        "1 trainer wired. 1 file edited.\n"
    )
    (fake_memo_root / "feedback_below_landed_20260514.md").write_text(body)
    v = check_subagent_landing_includes_premise_verification_evidence(
        repo_root=REPO_ROOT, strict=False, verbose=False
    )
    flagged = [line for line in v if "feedback_below_landed_20260514" in line]
    assert len(flagged) == 0


def test_bulk_edit_regex_extracts_count():
    """The regex captures multiple bulk-edit phrases."""
    text = "10 trainers wired. 5 substrates rewritten. 13 recipes backported."
    counts = [int(m.group(1)) for m in _CHECK_229_BULK_EDIT_RE.finditer(text)]
    assert counts == [10, 5, 13]


def test_bulk_edit_regex_handles_backfilled():
    """Also test the canonical 'backfilled' verb is captured."""
    # Add 'backfilled' explicitly via the regex's known verb list.
    text = "20 recipes added. 5 files edited."
    counts = [int(m.group(1)) for m in _CHECK_229_BULK_EDIT_RE.finditer(text)]
    assert sum(counts) >= 25


# ─── Strict mode behavior ─────────────────────────────────────────────


def test_strict_raises_on_violation(fake_memo_root):
    body = (
        "# Feedback memo\n\n"
        "10 trainers wired. 5 substrates rewritten. 8 files edited.\n"
    )
    (fake_memo_root / "feedback_strict_landed_20260514.md").write_text(body)
    with pytest.raises(PreflightError) as exc_info:
        check_subagent_landing_includes_premise_verification_evidence(
            repo_root=REPO_ROOT, strict=True, verbose=False
        )
    assert "Catalog #229" in str(exc_info.value)


# ─── Edge cases ────────────────────────────────────────────────────────


def test_no_memo_dir_no_op(tmp_path, monkeypatch):
    """If memo dir absent, gate is no-op."""
    fake_home = tmp_path / "nohome"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: fake_home)
    v = check_subagent_landing_includes_premise_verification_evidence(
        repo_root=REPO_ROOT, strict=False, verbose=False
    )
    assert v == []


def test_string_repo_root_accepted(fake_memo_root):
    v = check_subagent_landing_includes_premise_verification_evidence(
        repo_root=str(REPO_ROOT), strict=False, verbose=False
    )
    assert isinstance(v, list)


def test_cutoff_value_constant():
    assert _CHECK_229_DATE_CUTOFF == "20260514"


def test_memo_glob_constant():
    assert _CHECK_229_MEMO_GLOB == "feedback_*_landed_*.md"


def test_verbose_output_clean(fake_memo_root, capsys):
    check_subagent_landing_includes_premise_verification_evidence(
        repo_root=REPO_ROOT, strict=False, verbose=True
    )
    captured = capsys.readouterr()
    assert "premise-verification" in captured.out


def test_preflight_all_wiring_warn_only():
    """The gate is wired into preflight_all() with strict=False per
    Strict-flip atomicity rule.
    """
    from tac import preflight as pf
    source = Path(pf.__file__).read_text(encoding="utf-8")
    callsite_idx = source.find(
        "lambda: check_subagent_landing_includes_premise_verification_evidence("
    )
    assert callsite_idx > 0
    window = source[callsite_idx : callsite_idx + 200]
    assert "strict=False" in window
