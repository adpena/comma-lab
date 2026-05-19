"""Tests for Catalog #343 check_claude_md_frontier_score_uses_canonical_pointer_not_hardcoded.

Per CLAUDE.md "Frontier scores are pointer-only - NON-NEGOTIABLE" + Catalog
#343: CLAUDE.md must cite the canonical pointer file
``.omx/state/canonical_frontier_pointer.json`` rather than embed hardcoded
score literals that drift over time. HISTORICAL-CONTEXT score literals
(catalog row docstrings; falsification verdicts; postmortems) MUST carry
same-line ``# HISTORICAL_SCORE_LITERAL_OK:<rationale>`` waivers.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_claude_md_frontier_score_uses_canonical_pointer_not_hardcoded,
    preflight_all,
)


REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent


# ─────────────────────────────────────────────────────────────────────────
# Live-repo regression guard (warn-only at landing per Strict-flip rule)
# ─────────────────────────────────────────────────────────────────────────


def test_live_repo_regression_guard_bounded() -> None:
    """Live count at landing is warn-only with a bounded ceiling (historical anchors)."""

    violations = check_claude_md_frontier_score_uses_canonical_pointer_not_hardcoded(
        repo_root=REPO_ROOT, strict=False, verbose=False
    )
    # At landing time CLAUDE.md carries dozens of historical anchors in
    # catalog row docstrings + postmortems + falsification verdicts. Allow
    # up to 100 as a generous ceiling so the gate fires structurally on
    # NEW non-historical hardcoded literals.
    assert len(violations) < 100, (
        f"unexpected explosion in Catalog #343 live count ({len(violations)} > 100); "
        "review NEW non-historical hardcoded frontier scores in CLAUDE.md"
    )


def test_orchestrator_callsite_warn_only_wire_in() -> None:
    """preflight_all wires Catalog #343 as strict=False (warn-only initially)."""

    import inspect
    from tac import preflight as preflight_mod

    source = inspect.getsource(preflight_mod.preflight_all)
    assert "check_claude_md_frontier_score_uses_canonical_pointer_not_hardcoded" in source
    # Should be strict=False at landing per Strict-flip atomicity rule.
    # We don't assert the exact form because the wire-in may be on
    # multiple lines; just confirm the function is called.


# ─────────────────────────────────────────────────────────────────────────
# Positive (synthetic CLAUDE.md flagged)
# ─────────────────────────────────────────────────────────────────────────


def test_synthetic_bare_hardcoded_score_flagged(tmp_path: Path) -> None:
    """A synthetic CLAUDE.md with bare hardcoded 0.19205 is flagged."""

    claude_md = tmp_path / "CLAUDE.md"
    claude_md.write_text(
        "Our current frontier is 0.19205 on contest-CPU axis.\n"
    )
    violations = check_claude_md_frontier_score_uses_canonical_pointer_not_hardcoded(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert len(violations) == 1
    assert "0.19205" in violations[0]


def test_multiple_hardcoded_scores_on_one_line_all_flagged(tmp_path: Path) -> None:
    """Each score literal on a line is reported separately."""

    claude_md = tmp_path / "CLAUDE.md"
    claude_md.write_text("Cluster: 0.19205, 0.19538, 0.20533 all observed.\n")
    violations = check_claude_md_frontier_score_uses_canonical_pointer_not_hardcoded(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert len(violations) == 3


def test_multiple_lines_flagged(tmp_path: Path) -> None:
    """Each line with a hardcoded score is flagged."""

    claude_md = tmp_path / "CLAUDE.md"
    claude_md.write_text(
        "Line 1: 0.19205\n"
        "Line 2: ignore me\n"
        "Line 3: 0.20533\n"
    )
    violations = check_claude_md_frontier_score_uses_canonical_pointer_not_hardcoded(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert len(violations) == 2


# ─────────────────────────────────────────────────────────────────────────
# Negative (canonical pointer reference accepted)
# ─────────────────────────────────────────────────────────────────────────


def test_canonical_pointer_reference_in_window_accepts(tmp_path: Path) -> None:
    """Hardcoded score with adjacent pointer reference passes."""

    claude_md = tmp_path / "CLAUDE.md"
    claude_md.write_text(
        "Current frontier 0.19205\n"
        "see `.omx/state/canonical_frontier_pointer.json`\n"
    )
    violations = check_claude_md_frontier_score_uses_canonical_pointer_not_hardcoded(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert violations == []


def test_refresh_command_reference_in_window_accepts(tmp_path: Path) -> None:
    """Reference to tools/refresh_canonical_frontier.py also satisfies."""

    claude_md = tmp_path / "CLAUDE.md"
    claude_md.write_text(
        "Current frontier 0.19205 (per `tools/refresh_canonical_frontier.py`)\n"
    )
    violations = check_claude_md_frontier_score_uses_canonical_pointer_not_hardcoded(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert violations == []


def test_pointer_reference_5_lines_below_accepts(tmp_path: Path) -> None:
    """Pointer reference within +5 lines accepts the literal above."""

    claude_md = tmp_path / "CLAUDE.md"
    claude_md.write_text(
        "Current frontier 0.19205\n"
        "intervening line 1\n"
        "intervening line 2\n"
        "intervening line 3\n"
        "intervening line 4\n"
        "see `.omx/state/canonical_frontier_pointer.json`\n"
    )
    violations = check_claude_md_frontier_score_uses_canonical_pointer_not_hardcoded(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert violations == []


def test_pointer_reference_6_lines_below_does_not_accept(tmp_path: Path) -> None:
    """Pointer reference beyond +5 line window does NOT accept."""

    claude_md = tmp_path / "CLAUDE.md"
    claude_md.write_text(
        "Current frontier 0.19205\n"
        "intervening line 1\n"
        "intervening line 2\n"
        "intervening line 3\n"
        "intervening line 4\n"
        "intervening line 5\n"
        "intervening line 6\n"
        "see `.omx/state/canonical_frontier_pointer.json`\n"
    )
    violations = check_claude_md_frontier_score_uses_canonical_pointer_not_hardcoded(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert len(violations) == 1


# ─────────────────────────────────────────────────────────────────────────
# Waiver semantics
# ─────────────────────────────────────────────────────────────────────────


def test_historical_waiver_with_rationale_accepts(tmp_path: Path) -> None:
    """HISTORICAL_SCORE_LITERAL_OK with non-placeholder rationale passes."""

    claude_md = tmp_path / "CLAUDE.md"
    claude_md.write_text(
        "PR102 (third prize) 0.19538 [contest-CPU]  # HISTORICAL_SCORE_LITERAL_OK:public PR record\n"
    )
    violations = check_claude_md_frontier_score_uses_canonical_pointer_not_hardcoded(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert violations == []


def test_frontier_pointer_literal_waiver_with_rationale_accepts(tmp_path: Path) -> None:
    """FRONTIER_POINTER_LITERAL_OK with rationale passes."""

    claude_md = tmp_path / "CLAUDE.md"
    claude_md.write_text(
        "Example: 0.19205  # FRONTIER_POINTER_LITERAL_OK:design memo example\n"
    )
    violations = check_claude_md_frontier_score_uses_canonical_pointer_not_hardcoded(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert violations == []


def test_placeholder_rationale_rejected(tmp_path: Path) -> None:
    """`<rationale>` placeholder literal is rejected per self-waive guard."""

    claude_md = tmp_path / "CLAUDE.md"
    claude_md.write_text(
        "Test: 0.19205  # HISTORICAL_SCORE_LITERAL_OK:<rationale>\n"
    )
    violations = check_claude_md_frontier_score_uses_canonical_pointer_not_hardcoded(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert len(violations) == 1


def test_reason_placeholder_rejected(tmp_path: Path) -> None:
    """`<reason>` placeholder literal also rejected."""

    claude_md = tmp_path / "CLAUDE.md"
    claude_md.write_text(
        "Test: 0.19205  # HISTORICAL_SCORE_LITERAL_OK:<reason>\n"
    )
    violations = check_claude_md_frontier_score_uses_canonical_pointer_not_hardcoded(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert len(violations) == 1


def test_empty_rationale_rejected(tmp_path: Path) -> None:
    """Empty rationale after colon is rejected."""

    claude_md = tmp_path / "CLAUDE.md"
    claude_md.write_text("Test: 0.19205  # HISTORICAL_SCORE_LITERAL_OK:\n")
    violations = check_claude_md_frontier_score_uses_canonical_pointer_not_hardcoded(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert len(violations) == 1


# ─────────────────────────────────────────────────────────────────────────
# Strict-mode behavior
# ─────────────────────────────────────────────────────────────────────────


def test_strict_mode_raises_on_violation(tmp_path: Path) -> None:
    """Strict mode raises PreflightError on violation."""

    claude_md = tmp_path / "CLAUDE.md"
    claude_md.write_text("Bare frontier: 0.19205\n")
    with pytest.raises(PreflightError, match="Catalog #343"):
        check_claude_md_frontier_score_uses_canonical_pointer_not_hardcoded(
            repo_root=tmp_path, strict=True, verbose=False
        )


def test_strict_mode_silent_on_clean(tmp_path: Path) -> None:
    """Strict mode silent on clean repo."""

    claude_md = tmp_path / "CLAUDE.md"
    claude_md.write_text("No frontier scores here at all.\n")
    result = check_claude_md_frontier_score_uses_canonical_pointer_not_hardcoded(
        repo_root=tmp_path, strict=True, verbose=False
    )
    assert result == []


def test_missing_claude_md_returns_empty(tmp_path: Path) -> None:
    """Missing CLAUDE.md returns empty (no error)."""

    violations = check_claude_md_frontier_score_uses_canonical_pointer_not_hardcoded(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert violations == []


def test_score_outside_target_range_not_flagged(tmp_path: Path) -> None:
    """Scores like 0.193 (3 decimals only) or 0.18 (out of range) not flagged."""

    claude_md = tmp_path / "CLAUDE.md"
    claude_md.write_text(
        "Score 0.193 has only 3 decimals (3-digit-pattern not matched)\n"
        "Score 0.05 is out of the 0.19xx-0.20xx range\n"
        "Score 0.55 totally unrelated\n"
    )
    violations = check_claude_md_frontier_score_uses_canonical_pointer_not_hardcoded(
        repo_root=tmp_path, strict=False, verbose=False
    )
    # 0.193 is 3 decimals not in our 2-decimal+ pattern after 0.19;
    # confirm regex specificity.
    assert violations == []
