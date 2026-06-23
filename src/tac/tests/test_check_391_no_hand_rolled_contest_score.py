# SPDX-License-Identifier: MIT
"""Tests for Catalog #391 — no hand-rolled contest-score arithmetic.

The HAND-ROLLED-SCORE bug class 2026-06-23: the contest score formula
(upstream/evaluate.py:92) was reconstructed across ~20+ files with the
37_545_489 rate denominator scattered and NO single canonical helper. A
subagent dropped the x25 on the rate term. This gate steers NEW
hand-rolled score arithmetic toward tac.contest_score.

Memory: feedback_canonical_contest_score_helper_landed_20260623.md.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_no_hand_rolled_contest_score,
)


def _make_experiments(tmp_path: Path) -> Path:
    (tmp_path / "experiments").mkdir(parents=True)
    return tmp_path


def _run(root: Path, *, strict=False):
    return check_no_hand_rolled_contest_score(
        repo_root=root, strict=strict, verbose=False
    )


# ── Catches each forbidden pattern ──────────────────────────────────────


def test_391_catches_rate_denominator_literal(tmp_path):
    root = _make_experiments(tmp_path)
    (root / "experiments" / "bad_denom.py").write_text(
        "RATE_DENOM = 37_545_489\n"
    )
    v = _run(root)
    hits = [x for x in v if "bad_denom.py" in x]
    assert len(hits) == 1
    assert "rate-denominator literal" in hits[0]


def test_391_catches_unspaced_denominator_literal(tmp_path):
    root = _make_experiments(tmp_path)
    (root / "experiments" / "bad_denom2.py").write_text(
        "N = 37545489\n"
    )
    v = _run(root)
    assert any("bad_denom2.py" in x for x in v)


def test_391_catches_rate_term(tmp_path):
    root = _make_experiments(tmp_path)
    (root / "experiments" / "bad_rate.py").write_text(
        "rate_term = 25 * archive_bytes / DENOM\n"
    )
    v = _run(root)
    hits = [x for x in v if "bad_rate.py" in x]
    assert len(hits) == 1
    assert "rate-term" in hits[0]


def test_391_catches_seg_term(tmp_path):
    root = _make_experiments(tmp_path)
    (root / "experiments" / "bad_seg.py").write_text(
        "seg = 100 * d_seg\n"
    )
    v = _run(root)
    hits = [x for x in v if "bad_seg.py" in x]
    assert len(hits) == 1
    assert "seg-term" in hits[0]


def test_391_catches_pose_term_sqrt(tmp_path):
    root = _make_experiments(tmp_path)
    (root / "experiments" / "bad_pose.py").write_text(
        "import math\n"
        "pose = math.sqrt(10 * d_pose)\n"
    )
    v = _run(root)
    hits = [x for x in v if "bad_pose.py" in x]
    assert len(hits) == 1
    assert "pose-term" in hits[0]


def test_391_catches_pose_term_power_form(tmp_path):
    root = _make_experiments(tmp_path)
    (root / "experiments" / "bad_pose2.py").write_text(
        "pose = (10 * d_pose) ** 0.5\n"
    )
    v = _run(root)
    assert any("bad_pose2.py" in x for x in v)


def test_391_catches_posenet_dist_pose_term(tmp_path):
    root = _make_experiments(tmp_path)
    (root / "experiments" / "bad_pose3.py").write_text(
        "import math\n"
        "pose = math.sqrt(posenet_dist * 10)\n"
    )
    v = _run(root)
    assert any("bad_pose3.py" in x for x in v)


# ── Allows non-violations ───────────────────────────────────────────────


def test_391_allows_import_from_canonical_helper(tmp_path):
    root = _make_experiments(tmp_path)
    (root / "experiments" / "good.py").write_text(
        "from tac.contest_score import compute_contest_score, break_even_d_seg\n"
        "S = compute_contest_score(d_seg, d_pose, archive_bytes)\n"
        "be = break_even_d_seg(0.19110, d_pose, archive_bytes)\n"
    )
    v = _run(root)
    assert not any("good.py" in x for x in v)


def test_391_allows_unrelated_arithmetic(tmp_path):
    root = _make_experiments(tmp_path)
    (root / "experiments" / "unrelated.py").write_text(
        "x = 100 * y + 25 * z\n"  # 100*y not 100*d_seg; 25*z no denom
        "w = 12345\n"
    )
    v = _run(root)
    assert not any("unrelated.py" in x for x in v)


def test_391_allows_other_large_int(tmp_path):
    root = _make_experiments(tmp_path)
    (root / "experiments" / "other_int.py").write_text(
        "BUFFER = 37_545_490\n"  # one off from the denom
    )
    v = _run(root)
    assert not any("other_int.py" in x for x in v)


# ── Waiver respect ──────────────────────────────────────────────────────


def test_391_respects_valid_waiver(tmp_path):
    root = _make_experiments(tmp_path)
    (root / "experiments" / "waived.py").write_text(
        "RATE_DENOM = 37_545_489  # HAND_ROLLED_SCORE_OK: legacy frozen archive grammar\n"
    )
    v = _run(root)
    assert not any("waived.py" in x for x in v)


def test_391_rejects_placeholder_waiver_rationale(tmp_path):
    root = _make_experiments(tmp_path)
    (root / "experiments" / "ph.py").write_text(
        "RATE_DENOM = 37_545_489  # HAND_ROLLED_SCORE_OK: <rationale>\n"
    )
    v = _run(root)
    assert any("ph.py" in x for x in v)


def test_391_rejects_empty_waiver_rationale(tmp_path):
    root = _make_experiments(tmp_path)
    (root / "experiments" / "empty.py").write_text(
        "RATE_DENOM = 37_545_489  # HAND_ROLLED_SCORE_OK:\n"
    )
    v = _run(root)
    assert any("empty.py" in x for x in v)


# ── Allowlist + vendored exclusions ─────────────────────────────────────


def test_391_excludes_canonical_helper(tmp_path):
    root = tmp_path
    (root / "src" / "tac").mkdir(parents=True)
    (root / "src" / "tac" / "contest_score.py").write_text(
        "UNCOMPRESSED_SIZE_BYTES = 37_545_489\n"
        "S = 100 * d_seg\n"
    )
    v = _run(root)
    assert not any("contest_score.py" in x for x in v)


def test_391_excludes_vendored_intake(tmp_path):
    root = _make_experiments(tmp_path)
    vend = root / "experiments" / "results" / "public_pr101_intake_x"
    vend.mkdir(parents=True)
    (vend / "vendored.py").write_text("N = 37_545_489\n")
    v = _run(root)
    assert not any("vendored.py" in x for x in v)


# ── strict mode ─────────────────────────────────────────────────────────


def test_391_strict_raises_on_violation(tmp_path):
    root = _make_experiments(tmp_path)
    (root / "experiments" / "bad.py").write_text("N = 37_545_489\n")
    with pytest.raises(PreflightError):
        _run(root, strict=True)


def test_391_strict_passes_when_clean(tmp_path):
    root = _make_experiments(tmp_path)
    (root / "experiments" / "clean.py").write_text("x = 1\n")
    # Should not raise.
    out = _run(root, strict=True)
    assert out == []


# ── reports the line number ─────────────────────────────────────────────


def test_391_reports_correct_line_number(tmp_path):
    root = _make_experiments(tmp_path)
    (root / "experiments" / "lined.py").write_text(
        "# line 1\n"
        "# line 2\n"
        "N = 37_545_489\n"  # line 3
    )
    v = _run(root)
    hits = [x for x in v if "lined.py" in x]
    assert len(hits) == 1
    assert ":3 " in hits[0]
