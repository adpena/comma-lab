# SPDX-License-Identifier: MIT
"""Tests for ``tools.contest_score_gap_decomposition``."""

from __future__ import annotations

import importlib.util
import pathlib


def _load_tool_module():
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    path = repo_root / "tools" / "contest_score_gap_decomposition.py"
    spec = importlib.util.spec_from_file_location(
        "contest_score_gap_decomposition", path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_pr103_pr106_anchor_decomposition_reproduces_local_score() -> None:
    mod = _load_tool_module()
    dec = mod.decompose_anchor(mod.PR103_PR106_ANCHOR)
    # Within 1e-6 of the contest-CUDA T4 reported score (small float-precision slack).
    assert abs(dec["total"] - mod.PR103_PR106_ANCHOR.reported_score) < 1e-6


def test_back_solve_pose_returns_non_negative() -> None:
    mod = _load_tool_module()
    pose = mod._back_solve_pose(score=0.193, bytes_=178_258, seg=0.00067082)
    assert pose >= 0.0


def test_medal_band_anchors_have_three_entries() -> None:
    mod = _load_tool_module()
    refs = mod.medal_band_anchors()
    assert len(refs) == 3
    labels = {r.label for r in refs}
    assert any("gold" in lbl for lbl in labels)
    assert any("silver" in lbl for lbl in labels)
    assert any("bronze" in lbl for lbl in labels)


def test_gap_components_sum_to_total_gap() -> None:
    """seg_delta + pose_delta + rate_delta must equal total_gap exactly."""
    mod = _load_tool_module()
    target = mod.PR103_PR106_ANCHOR
    refs = mod.medal_band_anchors()
    gaps = mod.gap_analysis(target, refs)
    for ref in refs:
        g = gaps[ref.label]
        sum_delta = g["seg_term_delta"] + g["pose_term_delta"] + g["rate_term_delta"]
        assert abs(sum_delta - g["total_gap"]) < 1e-9


def test_gap_is_positive_pr106_above_medal_band() -> None:
    """Sanity: PR103-on-PR106 (0.20898) > all medal-band entries (0.193-0.195)."""
    mod = _load_tool_module()
    gaps = mod.gap_analysis(mod.PR103_PR106_ANCHOR, mod.medal_band_anchors())
    for label, g in gaps.items():
        assert g["total_gap"] > 0, f"unexpected: gap to {label} is non-positive"


def test_pose_axis_is_dominant_or_at_least_present() -> None:
    """The Path B forensic: pose distortion should be a meaningful share of the gap."""
    mod = _load_tool_module()
    refs = mod.medal_band_anchors()
    gaps = mod.gap_analysis(mod.PR103_PR106_ANCHOR, refs)
    for label, g in gaps.items():
        # At minimum, pose contribution should be positive (PR106 has more pose distortion)
        # AND should account for at least 30% of the gap.
        pose_share_of_gap = g["pose_term_delta"] / g["total_gap"]
        assert pose_share_of_gap > 0.3, (
            f"expected pose to dominate gap to {label}; got pose_share={pose_share_of_gap:.3f}"
        )


def test_render_markdown_emits_table_header() -> None:
    mod = _load_tool_module()
    md = mod.render_markdown(mod.PR103_PR106_ANCHOR, mod.medal_band_anchors())
    assert "Gap to medal-band references" in md
    assert "seg_delta" in md
    assert "pose_delta" in md
    assert "rate_delta" in md


# ---------------------------------------------------------------------------
# Bug-hunter v3: silent zero-pose back-solve (integration seam)
# ---------------------------------------------------------------------------

def test_back_solve_pose_warns_when_underdetermined() -> None:
    """Bug-hunter v3 (MEDIUM, integration seam): when seg_estimate + rate_term
    already exceed reported score, pose back-solves to <= 0 — previously the
    function silently returned 0.0, producing misleading gap-decomposition
    tables where pose_term=0 looked like a measurement instead of a guess.
    Now it emits a UserWarning so operators can see the underdetermined case
    in test logs and tooling output."""
    import warnings

    mod = _load_tool_module()
    # Construct an over-estimated seg that drives pose back-solve negative:
    # for score=0.193, bytes=178258, seg=0.001:
    #   rate_term = 25 * 178258 / 37545489 ~= 0.1187
    #   seg_term = 100 * 0.001 = 0.1
    #   pose_term = 0.193 - 0.1 - 0.1187 = -0.0257  (negative)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", UserWarning)
        result = mod._back_solve_pose(score=0.193, bytes_=178258, seg=0.001)
    assert result == 0.0
    assert any(
        "underdetermined" in str(w.message) or "back-solve" in str(w.message)
        for w in captured
    ), (
        "expected UserWarning on negative pose back-solve; got: "
        f"{[str(w.message) for w in captured]}"
    )


def test_back_solve_pose_no_warn_in_canonical_case() -> None:
    """Conversely, the canonical case (seg=0.00067082, score=0.193,
    bytes=178258) should not emit a warning — pose_term is positive."""
    import warnings

    mod = _load_tool_module()
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", UserWarning)
        result = mod._back_solve_pose(
            score=0.193, bytes_=178258, seg=0.00067082,
        )
    assert result > 0.0
    # No back-solve warnings for the canonical seg estimate.
    backsolve_warnings = [
        w for w in captured if "back-solve" in str(w.message) or "underdetermined" in str(w.message)
    ]
    assert backsolve_warnings == [], (
        f"unexpected back-solve warning: {[str(w.message) for w in backsolve_warnings]}"
    )
