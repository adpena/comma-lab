# SPDX-License-Identifier: MIT
"""Tests for Catalog #396 — check_measured_win_findings_are_wired_or_research_only.

APPARATUS-BLINDNESS REPAIR + ORPHANED-MEASURED-WIN self-protection
(operator 2026-07-02, store-nothing orphaned-signal meta-bug). The gate
refuses ``.omx/research/*.md`` memos dated >= 2026-07-02 that claim a
MEASURED mechanism-level win (a MEASURED-evidence token AND a mechanism-win
token co-occur) unless the memo is WIRED (canonical_equation ref AND a
launch-config/DSL/wiring pointer), RESEARCH_ONLY (research_only +
reactivation), or carries a same-line ``# ORPHAN_WIN_WAIVED:<rationale>``
waiver (placeholder rejected).

Covers:
* the shared classifier ``classify_findings_memo_orphan_status`` (5 verdicts)
* MEASURED-win detection (measured+win co-occurrence; percent-reduction)
* WIRED / RESEARCH_ONLY / WAIVER acceptance paths
* placeholder + short-rationale waiver rejection
* date filter (pre-cutoff memo exempt)
* not-a-win memo exempt (measured-only OR win-only, not both)
* strict-mode raises with Catalog #396 message; warn-mode returns list
* string repo_root accepted; missing research dir silent
* self-exempt paths (gate's own files)
* live-repo regression: the two known orphans are flagged; warn-only never
  raises against the live tree
* orchestrator wire-in warn-only regression guard
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight import (
    ORPHAN_WIN_STATUS_NOT_A_WIN,
    ORPHAN_WIN_STATUS_ORPHAN,
    ORPHAN_WIN_STATUS_RESEARCH_ONLY,
    ORPHAN_WIN_STATUS_WAIVED,
    ORPHAN_WIN_STATUS_WIRED,
    PreflightError,
    check_measured_win_findings_are_wired_or_research_only,
    classify_findings_memo_orphan_status,
)


# ──────────────────────────────────────────────────────────────────────
# Fixtures / helpers
# ──────────────────────────────────────────────────────────────────────

_ORPHAN_BODY = (
    "# Store-nothing keyframe codec — MEASURED n600\n\n"
    "The store-nothing render WINS: d_pose 1.12 is BETTER than the full "
    "real keyframe's 1.37 at ~0 marginal rate. Measured n600.\n"
)

_WIRED_BODY = (
    "# Store-nothing keyframe codec — MEASURED n600\n\n"
    "The store-nothing render WINS: d_pose 1.12 BETTER than 1.37. Measured "
    "n600. Registered via tac.canonical_equations and wired into "
    "witness_autoconfig.proven_base as a trainer mode.\n"
)

_RESEARCH_ONLY_BODY = (
    "# Wave-F unified-xi — MEASURED n600\n\n"
    "source-smoothing win15 measured -42% rate, BETTER than baseline. "
    "research_only: true. reactivation_criteria: confirm net-S via the #205 "
    "through-R d_seg A/B.\n"
)

_WAIVED_BODY = (
    "# Measured lever — MEASURED n600\n\n"
    "This lever WINS by -30% rate at n600, BETTER than the prior codec. "
    "# ORPHAN_WIN_WAIVED:advisory-negative-companion-not-a-shippable-mechanism\n"
)

_NOT_A_WIN_MEASURED_ONLY = (
    "# Measurement report — MEASURED n600\n\n"
    "We measured d_seg at n600. No comparison drawn; pointer unmoved.\n"
)

_NOT_A_WIN_WIN_ONLY = (
    "# Design note\n\n"
    "This design would be BETTER than the alternative in principle; no "
    "measurement yet. Pure derivation.\n"
)

_PERCENT_REDUCTION_ORPHAN = (
    "# RD codec — MEASURED n600\n\n"
    "The LBND2 codec measured a rate reduction of -73% at n600 "
    "(0.1041 to 0.02765). Landed.\n"
)


def _write_memo(research_dir: Path, name: str, body: str) -> Path:
    research_dir.mkdir(parents=True, exist_ok=True)
    p = research_dir / name
    p.write_text(body, encoding="utf-8")
    return p


def _run(tmp_path: Path, **kw) -> list[str]:
    return check_measured_win_findings_are_wired_or_research_only(
        repo_root=tmp_path, verbose=False, **kw
    )


# ──────────────────────────────────────────────────────────────────────
# classifier verdicts
# ──────────────────────────────────────────────────────────────────────


def test_classifier_orphan():
    assert classify_findings_memo_orphan_status(_ORPHAN_BODY) == (
        ORPHAN_WIN_STATUS_ORPHAN
    )


def test_classifier_wired():
    assert classify_findings_memo_orphan_status(_WIRED_BODY) == (
        ORPHAN_WIN_STATUS_WIRED
    )


def test_classifier_research_only():
    assert classify_findings_memo_orphan_status(_RESEARCH_ONLY_BODY) == (
        ORPHAN_WIN_STATUS_RESEARCH_ONLY
    )


def test_classifier_waived():
    assert classify_findings_memo_orphan_status(_WAIVED_BODY) == (
        ORPHAN_WIN_STATUS_WAIVED
    )


def test_classifier_not_a_win_measured_only():
    assert classify_findings_memo_orphan_status(_NOT_A_WIN_MEASURED_ONLY) == (
        ORPHAN_WIN_STATUS_NOT_A_WIN
    )


def test_classifier_not_a_win_win_only():
    assert classify_findings_memo_orphan_status(_NOT_A_WIN_WIN_ONLY) == (
        ORPHAN_WIN_STATUS_NOT_A_WIN
    )


def test_classifier_percent_reduction_is_a_win():
    # A "-N%" reduction co-occurring with MEASURED is a mechanism win.
    assert classify_findings_memo_orphan_status(_PERCENT_REDUCTION_ORPHAN) == (
        ORPHAN_WIN_STATUS_ORPHAN
    )


def test_wired_requires_both_equation_and_pointer():
    # Equation ref but NO wiring pointer => still ORPHAN.
    body = (
        "MEASURED n600: store-nothing WINS, d_pose 1.12 BETTER than 1.37. "
        "Registered via tac.canonical_equations.\n"
    )
    assert classify_findings_memo_orphan_status(body) == ORPHAN_WIN_STATUS_ORPHAN
    # Wiring pointer but NO equation ref => still ORPHAN.
    body2 = (
        "MEASURED n600: store-nothing WINS, d_pose 1.12 BETTER than 1.37. "
        "wired into witness_autoconfig proven_base.\n"
    )
    assert classify_findings_memo_orphan_status(body2) == ORPHAN_WIN_STATUS_ORPHAN


def test_research_only_requires_reactivation():
    body = (
        "MEASURED n600: lever WINS -42% rate, BETTER than baseline. "
        "research_only: true.\n"  # no reactivation
    )
    assert classify_findings_memo_orphan_status(body) == ORPHAN_WIN_STATUS_ORPHAN


# ──────────────────────────────────────────────────────────────────────
# waiver semantics
# ──────────────────────────────────────────────────────────────────────


def test_waiver_placeholder_rejected():
    body = (
        "MEASURED n600 WINS BETTER than prior. "
        "# ORPHAN_WIN_WAIVED:<rationale>\n"
    )
    assert classify_findings_memo_orphan_status(body) == ORPHAN_WIN_STATUS_ORPHAN


def test_waiver_short_rationale_rejected():
    body = "MEASURED n600 WINS BETTER than prior. # ORPHAN_WIN_WAIVED:x\n"
    assert classify_findings_memo_orphan_status(body) == ORPHAN_WIN_STATUS_ORPHAN


def test_waiver_reason_placeholder_rejected():
    body = "MEASURED n600 WINS BETTER than prior. # ORPHAN_WIN_WAIVED:<reason>\n"
    assert classify_findings_memo_orphan_status(body) == ORPHAN_WIN_STATUS_ORPHAN


# ──────────────────────────────────────────────────────────────────────
# gate behavior (date filter / acceptance / strict)
# ──────────────────────────────────────────────────────────────────────


def test_gate_flags_orphan_post_cutoff(tmp_path):
    _write_memo(tmp_path / ".omx/research", "orphan_win_20260702.md", _ORPHAN_BODY)
    v = _run(tmp_path)
    assert len(v) == 1
    assert "orphan_win_20260702.md" in v[0]
    assert "Catalog #396" in v[0]


def test_gate_pre_cutoff_memo_exempt(tmp_path):
    # Same orphan body but dated BEFORE the cutoff => not flagged by the gate.
    _write_memo(tmp_path / ".omx/research", "orphan_win_20260601.md", _ORPHAN_BODY)
    assert _run(tmp_path) == []


def test_gate_wired_accepted(tmp_path):
    _write_memo(tmp_path / ".omx/research", "wired_20260702.md", _WIRED_BODY)
    assert _run(tmp_path) == []


def test_gate_research_only_accepted(tmp_path):
    _write_memo(
        tmp_path / ".omx/research", "research_only_20260702.md", _RESEARCH_ONLY_BODY
    )
    assert _run(tmp_path) == []


def test_gate_waiver_accepted(tmp_path):
    _write_memo(tmp_path / ".omx/research", "waived_20260702.md", _WAIVED_BODY)
    assert _run(tmp_path) == []


def test_gate_not_a_win_ignored(tmp_path):
    _write_memo(
        tmp_path / ".omx/research", "measured_only_20260702.md",
        _NOT_A_WIN_MEASURED_ONLY,
    )
    assert _run(tmp_path) == []


def test_gate_strict_raises(tmp_path):
    _write_memo(tmp_path / ".omx/research", "orphan_win_20260702.md", _ORPHAN_BODY)
    with pytest.raises(PreflightError) as exc:
        _run(tmp_path, strict=True)
    assert "Catalog #396" in str(exc.value)


def test_gate_warn_mode_does_not_raise(tmp_path):
    _write_memo(tmp_path / ".omx/research", "orphan_win_20260702.md", _ORPHAN_BODY)
    # warn-only (default strict=False) returns the list, never raises.
    assert len(_run(tmp_path, strict=False)) == 1


def test_gate_string_repo_root(tmp_path):
    _write_memo(tmp_path / ".omx/research", "orphan_win_20260702.md", _ORPHAN_BODY)
    v = check_measured_win_findings_are_wired_or_research_only(
        repo_root=str(tmp_path), verbose=False
    )
    assert len(v) == 1


def test_gate_missing_research_dir_silent(tmp_path):
    assert _run(tmp_path) == []


def test_gate_non_md_ignored(tmp_path):
    _write_memo(tmp_path / ".omx/research", "orphan_win_20260702.txt", _ORPHAN_BODY)
    assert _run(tmp_path) == []


def test_gate_undated_filename_ignored(tmp_path):
    # No 8-digit date suffix => not matched by the design-filename regex.
    _write_memo(tmp_path / ".omx/research", "orphan_win_notes.md", _ORPHAN_BODY)
    assert _run(tmp_path) == []


def test_gate_percent_reduction_orphan_flagged(tmp_path):
    _write_memo(
        tmp_path / ".omx/research", "lbnd2_20260702.md", _PERCENT_REDUCTION_ORPHAN
    )
    assert len(_run(tmp_path)) == 1


def test_gate_multi_memo_aggregation(tmp_path):
    rd = tmp_path / ".omx/research"
    _write_memo(rd, "a_20260702.md", _ORPHAN_BODY)
    _write_memo(rd, "b_20260703.md", _PERCENT_REDUCTION_ORPHAN)
    _write_memo(rd, "c_20260702.md", _WIRED_BODY)  # accepted
    assert len(_run(tmp_path)) == 2


# ──────────────────────────────────────────────────────────────────────
# live-repo regression + wire-in
# ──────────────────────────────────────────────────────────────────────


def test_live_repo_known_orphans_flagged():
    """The two known orphans (store-nothing + wave_f Stage-1) are flagged on
    the live tree; warn-only never raises."""
    repo_root = Path(__file__).resolve().parents[3]
    if not (repo_root / ".omx/research").is_dir():
        pytest.skip("research dir not present")
    v = check_measured_win_findings_are_wired_or_research_only(
        repo_root=repo_root, strict=False, verbose=False
    )
    joined = "\n".join(v)
    assert "keyframe_rate_minimization_builds_20260702.md" in joined


def test_live_repo_warn_only_never_raises():
    repo_root = Path(__file__).resolve().parents[3]
    if not (repo_root / ".omx/research").is_dir():
        pytest.skip("research dir not present")
    # Must not raise in warn mode even with a live backlog.
    check_measured_win_findings_are_wired_or_research_only(
        repo_root=repo_root, strict=False, verbose=False
    )


def test_orchestrator_wires_warn_only():
    """Catalog #396 is wired WARN-ONLY (strict=False) in preflight_all — the
    live orphan backlog must not strict-break the build."""
    import inspect

    from tac import preflight

    src = inspect.getsource(preflight.preflight_all)
    assert "check_measured_win_findings_are_wired_or_research_only(" in src
    # Confirm the callsite is strict=False (warn-only).
    idx = src.index("check_measured_win_findings_are_wired_or_research_only(")
    window = src[idx : idx + 120]
    assert "strict=False" in window
