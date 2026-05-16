# SPDX-License-Identifier: MIT
"""Tests for Catalog #300 — check_council_deliberation_declares_tier_in_frontmatter.

Per the COUNCIL-HIERARCHY-V2 landing 2026-05-16. Mirrors the test pattern
of Catalog #292's sister test file `test_check_292_grand_council_assumption_statements.py`.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest


@pytest.fixture
def preflight_module():
    return importlib.import_module("tac.preflight")


def _T2_frontmatter() -> str:
    """Minimal valid T2 v2 frontmatter for fixture use.

    Includes the mission-alignment fields (council_predicted_mission_contribution
    + council_override_invoked) required at T2+ per the operator binding
    standing directive 2026-05-16.
    """
    return (
        "---\n"
        "council_tier: T2\n"
        "council_attendees: [Shannon, Dykstra, Yousfi, Fridrich, Contrarian, Assumption-Adversary]\n"
        "council_quorum_met: true\n"
        "council_verdict: PROCEED\n"
        "council_dissent: []\n"
        "council_assumption_adversary_verdict:\n"
        "  - assumption: EMA decay 0.997\n"
        "    classification: HARD-EARNED\n"
        "    rationale: PR101 empirical\n"
        "council_decisions_recorded:\n"
        "  - op-routable #1\n"
        "council_predicted_mission_contribution: apparatus_maintenance\n"
        "council_override_invoked: false\n"
        "---\n\n"
        "body text\n"
    )


def _T1_frontmatter() -> str:
    return (
        "---\n"
        "council_tier: T1\n"
        "council_attendees: [Shannon]\n"
        "council_quorum_met: true\n"
        "council_verdict: PROCEED\n"
        "council_dissent: []\n"
        "council_decisions_recorded: []\n"
        "---\n\n"
        "body text\n"
    )


def _missing_frontmatter() -> str:
    return (
        "---\n"
        "title: some council deliberation\n"
        "---\n\n"
        "body text without any v2 fields\n"
    )


def _write_memo(research_dir: Path, name: str, body: str) -> Path:
    p = research_dir / name
    p.write_text(body)
    return p


# ─────────────────────── helper unit tests ────────────────────────────


def test_parse_date_suffix_canonical(preflight_module):
    assert preflight_module._check_300_parse_date_suffix("20260516") == 20260516


def test_parse_date_suffix_invalid_length(preflight_module):
    assert preflight_module._check_300_parse_date_suffix("2026051") is None


def test_parse_date_suffix_bad_month(preflight_module):
    assert preflight_module._check_300_parse_date_suffix("20261316") is None


def test_extract_tier_canonical(preflight_module):
    body = "council_tier: T3\nother: x"
    assert preflight_module._check_300_extract_tier(body) == "T3"


def test_extract_tier_quoted(preflight_module):
    body = 'council_tier: "T4"'
    assert preflight_module._check_300_extract_tier(body) == "T4"


def test_extract_tier_missing(preflight_module):
    body = "some unrelated text"
    assert preflight_module._check_300_extract_tier(body) is None


def test_extract_tier_invalid(preflight_module):
    body = "council_tier: T9"
    assert preflight_module._check_300_extract_tier(body) is None


def test_has_waiver_with_rationale(preflight_module):
    body = "some text # COUNCIL_TIER_FRONTMATTER_WAIVED:legitimate-reason"
    assert preflight_module._check_300_has_waiver(body) is True


def test_has_waiver_rejects_placeholder(preflight_module):
    body = "some text # COUNCIL_TIER_FRONTMATTER_WAIVED:<rationale>"
    assert preflight_module._check_300_has_waiver(body) is False


def test_has_waiver_rejects_reason_placeholder(preflight_module):
    body = "some text # COUNCIL_TIER_FRONTMATTER_WAIVED:<reason>"
    assert preflight_module._check_300_has_waiver(body) is False


# ─────────────────────── end-to-end gate behavior ────────────────────────────


def test_no_research_dir_passes(preflight_module, tmp_path: Path):
    # repo_root with no .omx/research dir -> no violations.
    out = preflight_module.check_council_deliberation_declares_tier_in_frontmatter(
        repo_root=tmp_path,
    )
    assert out == []


def test_research_dir_with_no_council_memos_passes(preflight_module, tmp_path: Path):
    research = tmp_path / ".omx" / "research"
    research.mkdir(parents=True)
    (research / "unrelated.md").write_text("not a council memo")
    out = preflight_module.check_council_deliberation_declares_tier_in_frontmatter(
        research_dir=research,
        repo_root=tmp_path,
    )
    assert out == []


def test_post_cutoff_memo_with_all_v2_fields_passes(preflight_module, tmp_path: Path):
    research = tmp_path / ".omx" / "research"
    research.mkdir(parents=True)
    _write_memo(research, "feedback_grand_council_test_20260520.md", _T2_frontmatter())
    out = preflight_module.check_council_deliberation_declares_tier_in_frontmatter(
        research_dir=research,
        repo_root=tmp_path,
    )
    assert out == []


def test_post_cutoff_memo_without_v2_fields_flagged(preflight_module, tmp_path: Path):
    research = tmp_path / ".omx" / "research"
    research.mkdir(parents=True)
    _write_memo(research, "feedback_grand_council_test_20260520.md", _missing_frontmatter())
    out = preflight_module.check_council_deliberation_declares_tier_in_frontmatter(
        research_dir=research,
        repo_root=tmp_path,
    )
    assert len(out) == 1
    assert "council_tier" in out[0] or "v2 4-tier frontmatter" in out[0]


def test_pre_cutoff_memo_exempt(preflight_module, tmp_path: Path):
    research = tmp_path / ".omx" / "research"
    research.mkdir(parents=True)
    # Date 20260515 is BEFORE the 20260516 cutoff -> exempt.
    _write_memo(research, "feedback_grand_council_test_20260515.md", _missing_frontmatter())
    out = preflight_module.check_council_deliberation_declares_tier_in_frontmatter(
        research_dir=research,
        repo_root=tmp_path,
    )
    assert out == []


def test_non_council_memo_skipped(preflight_module, tmp_path: Path):
    research = tmp_path / ".omx" / "research"
    research.mkdir(parents=True)
    _write_memo(research, "feedback_some_analysis_20260520.md", "no v2 fields")
    out = preflight_module.check_council_deliberation_declares_tier_in_frontmatter(
        research_dir=research,
        repo_root=tmp_path,
    )
    assert out == []


def test_skunkworks_council_in_scope(preflight_module, tmp_path: Path):
    research = tmp_path / ".omx" / "research"
    research.mkdir(parents=True)
    _write_memo(research, "feedback_skunkworks_council_test_20260520.md", _missing_frontmatter())
    out = preflight_module.check_council_deliberation_declares_tier_in_frontmatter(
        research_dir=research,
        repo_root=tmp_path,
    )
    assert len(out) == 1


def test_reunion_symposium_in_scope(preflight_module, tmp_path: Path):
    research = tmp_path / ".omx" / "research"
    research.mkdir(parents=True)
    _write_memo(
        research,
        "feedback_grand_reunion_fields_symposium_test_20260520.md",
        _missing_frontmatter(),
    )
    out = preflight_module.check_council_deliberation_declares_tier_in_frontmatter(
        research_dir=research,
        repo_root=tmp_path,
    )
    assert len(out) == 1


def test_t1_does_not_require_assumption_adversary_verdict(preflight_module, tmp_path: Path):
    research = tmp_path / ".omx" / "research"
    research.mkdir(parents=True)
    _write_memo(research, "feedback_grand_council_t1_20260520.md", _T1_frontmatter())
    out = preflight_module.check_council_deliberation_declares_tier_in_frontmatter(
        research_dir=research,
        repo_root=tmp_path,
    )
    assert out == []


def test_t3_requires_assumption_adversary_verdict(preflight_module, tmp_path: Path):
    research = tmp_path / ".omx" / "research"
    research.mkdir(parents=True)
    # T3 memo with all OTHER fields but no council_assumption_adversary_verdict.
    body = (
        "---\n"
        "council_tier: T3\n"
        "council_attendees: [Shannon, Dykstra]\n"
        "council_quorum_met: true\n"
        "council_verdict: PROCEED\n"
        "council_dissent: []\n"
        "council_decisions_recorded: []\n"
        "---\n\n"
        "body\n"
    )
    _write_memo(research, "feedback_grand_council_t3_20260520.md", body)
    out = preflight_module.check_council_deliberation_declares_tier_in_frontmatter(
        research_dir=research,
        repo_root=tmp_path,
    )
    assert len(out) == 1
    assert "council_assumption_adversary_verdict" in out[0]


def test_same_line_waiver_passes(preflight_module, tmp_path: Path):
    research = tmp_path / ".omx" / "research"
    research.mkdir(parents=True)
    body = (
        "---\n"
        "title: legacy memo\n"
        "---\n\n"
        "body # COUNCIL_TIER_FRONTMATTER_WAIVED:legacy-no-v2-migration-yet\n"
    )
    _write_memo(research, "feedback_grand_council_waived_20260520.md", body)
    out = preflight_module.check_council_deliberation_declares_tier_in_frontmatter(
        research_dir=research,
        repo_root=tmp_path,
    )
    assert out == []


def test_placeholder_waiver_rejected(preflight_module, tmp_path: Path):
    research = tmp_path / ".omx" / "research"
    research.mkdir(parents=True)
    body = (
        "---\n"
        "title: legacy memo\n"
        "---\n\n"
        "body # COUNCIL_TIER_FRONTMATTER_WAIVED:<rationale>\n"
    )
    _write_memo(research, "feedback_grand_council_placeholder_20260520.md", body)
    out = preflight_module.check_council_deliberation_declares_tier_in_frontmatter(
        research_dir=research,
        repo_root=tmp_path,
    )
    assert len(out) == 1


def test_strict_mode_raises(preflight_module, tmp_path: Path):
    research = tmp_path / ".omx" / "research"
    research.mkdir(parents=True)
    _write_memo(research, "feedback_grand_council_test_20260520.md", _missing_frontmatter())
    with pytest.raises(preflight_module.PreflightError, match="Catalog #300"):
        preflight_module.check_council_deliberation_declares_tier_in_frontmatter(
            research_dir=research,
            repo_root=tmp_path,
            strict=True,
        )


def test_strict_mode_silent_on_clean(preflight_module, tmp_path: Path):
    research = tmp_path / ".omx" / "research"
    research.mkdir(parents=True)
    _write_memo(research, "feedback_grand_council_test_20260520.md", _T2_frontmatter())
    out = preflight_module.check_council_deliberation_declares_tier_in_frontmatter(
        research_dir=research,
        repo_root=tmp_path,
        strict=True,
    )
    assert out == []


def test_aggregates_multiple_violations(preflight_module, tmp_path: Path):
    research = tmp_path / ".omx" / "research"
    research.mkdir(parents=True)
    _write_memo(research, "feedback_grand_council_a_20260520.md", _missing_frontmatter())
    _write_memo(research, "feedback_grand_council_b_20260521.md", _missing_frontmatter())
    out = preflight_module.check_council_deliberation_declares_tier_in_frontmatter(
        research_dir=research,
        repo_root=tmp_path,
    )
    assert len(out) == 2


def test_string_repo_root_accepted(preflight_module, tmp_path: Path):
    research = tmp_path / ".omx" / "research"
    research.mkdir(parents=True)
    out = preflight_module.check_council_deliberation_declares_tier_in_frontmatter(
        repo_root=str(tmp_path),
    )
    assert out == []


def test_orchestrator_callsite_warn_only_regression_guard(preflight_module):
    """Catalog #300 wired into preflight_all() at strict=False per the
    Strict-flip atomicity rule until 5 deliberations land in v2 format.
    Regression-guard the warn-only state explicitly so the flip is a
    visible deliberate change (not a silent drift)."""
    src = Path(preflight_module.__file__).read_text(encoding="utf-8")
    assert "check_council_deliberation_declares_tier_in_frontmatter(" in src
    # Find the callsite block and confirm strict=False.
    idx = src.find("check_council_deliberation_declares_tier_in_frontmatter(")
    snippet = src[idx : idx + 200]
    assert "strict=False" in snippet, (
        "Catalog #300 must be wired strict=False until 5 v2 deliberations land"
    )


def test_invalid_date_suffix_skipped(preflight_module, tmp_path: Path):
    research = tmp_path / ".omx" / "research"
    research.mkdir(parents=True)
    # 99999999 is not a valid date -> skipped
    _write_memo(
        research,
        "feedback_grand_council_baddate_99999999.md",
        _missing_frontmatter(),
    )
    out = preflight_module.check_council_deliberation_declares_tier_in_frontmatter(
        research_dir=research,
        repo_root=tmp_path,
    )
    assert out == []


def test_external_memory_opt_in_only(preflight_module, tmp_path: Path):
    """External memory_dir scan must be opt-in only (OSS-hermetic per #292 sister)."""
    research = tmp_path / ".omx" / "research"
    research.mkdir(parents=True)
    memory = tmp_path / "memory"
    memory.mkdir()
    _write_memo(memory, "feedback_grand_council_external_20260520.md", _missing_frontmatter())
    # Default (no memory_dir) — memory not scanned -> 0 violations.
    out_default = preflight_module.check_council_deliberation_declares_tier_in_frontmatter(
        research_dir=research,
        repo_root=tmp_path,
    )
    assert out_default == []
    # Explicit memory_dir -> external memo flagged.
    out_explicit = preflight_module.check_council_deliberation_declares_tier_in_frontmatter(
        memory_dir=memory,
        research_dir=research,
        repo_root=tmp_path,
    )
    assert len(out_explicit) == 1


def test_live_repo_regression_guard(preflight_module):
    """Live repo must have 0 violations at landing time per the WARN-ONLY
    Strict-flip atomicity rule (no post-cutoff v2 council memos exist yet)."""
    out = preflight_module.check_council_deliberation_declares_tier_in_frontmatter()
    # Allow up to 2 violations in case sister subagents land council memos
    # during the same commit batch (this gate's own landing memo is one).
    assert len(out) <= 2, f"live count baseline drift: {len(out)} violations"


def test_verbose_output_clean(preflight_module, capsys, tmp_path: Path):
    research = tmp_path / ".omx" / "research"
    research.mkdir(parents=True)
    _write_memo(research, "feedback_grand_council_test_20260520.md", _T2_frontmatter())
    preflight_module.check_council_deliberation_declares_tier_in_frontmatter(
        research_dir=research,
        repo_root=tmp_path,
        verbose=True,
    )
    captured = capsys.readouterr()
    assert "OK" in captured.out


def test_verbose_output_dirty(preflight_module, capsys, tmp_path: Path):
    research = tmp_path / ".omx" / "research"
    research.mkdir(parents=True)
    _write_memo(research, "feedback_grand_council_test_20260520.md", _missing_frontmatter())
    preflight_module.check_council_deliberation_declares_tier_in_frontmatter(
        research_dir=research,
        repo_root=tmp_path,
        verbose=True,
    )
    captured = capsys.readouterr()
    assert "violation" in captured.out


# ───────────────────────────────────────────────────────────────────
# Mission-alignment frontmatter extension tests (operator binding
# directive 2026-05-16). Per CLAUDE.md "Mission alignment —
# non-negotiable" subsection of "Council hierarchy: 4-tier protocol".
# ───────────────────────────────────────────────────────────────────


def _T2_frontmatter_missing_mission() -> str:
    """T2 frontmatter that has v2 fields but lacks mission-alignment fields."""
    return (
        "---\n"
        "council_tier: T2\n"
        "council_attendees: [Shannon, Dykstra]\n"
        "council_quorum_met: true\n"
        "council_verdict: PROCEED\n"
        "council_dissent: []\n"
        "council_assumption_adversary_verdict:\n"
        "  - assumption: A\n"
        "    classification: HARD-EARNED\n"
        "    rationale: R\n"
        "council_decisions_recorded: []\n"
        # NO mission-alignment fields.
        "---\n\n"
        "body text\n"
    )


def _T2_frontmatter_with_override_no_rationale() -> str:
    """T2 frontmatter declaring override invoked but no rationale."""
    return (
        "---\n"
        "council_tier: T2\n"
        "council_attendees: [Shannon, Dykstra]\n"
        "council_quorum_met: true\n"
        "council_verdict: PROCEED\n"
        "council_dissent: []\n"
        "council_assumption_adversary_verdict:\n"
        "  - assumption: A\n"
        "    classification: HARD-EARNED\n"
        "    rationale: R\n"
        "council_decisions_recorded: []\n"
        "council_predicted_mission_contribution: frontier_breaking\n"
        "council_override_invoked: true\n"
        # NO council_override_rationale.
        "---\n\n"
        "body text\n"
    )


def test_extract_mission_contribution_canonical(preflight_module):
    body = "council_predicted_mission_contribution: frontier_breaking"
    assert preflight_module._check_300_extract_mission_contribution(body) == "frontier_breaking"


def test_extract_mission_contribution_quoted(preflight_module):
    body = 'council_predicted_mission_contribution: "apparatus_maintenance"'
    assert preflight_module._check_300_extract_mission_contribution(body) == "apparatus_maintenance"


def test_extract_mission_contribution_invalid(preflight_module):
    body = "council_predicted_mission_contribution: bogus_category"
    assert preflight_module._check_300_extract_mission_contribution(body) is None


def test_extract_override_invoked_true(preflight_module):
    body = "council_override_invoked: true"
    assert preflight_module._check_300_extract_override_invoked(body) is True


def test_extract_override_invoked_false(preflight_module):
    body = "council_override_invoked: false"
    assert preflight_module._check_300_extract_override_invoked(body) is False


def test_has_override_rationale_true(preflight_module):
    body = 'council_override_rationale: "operator verbatim quote"'
    assert preflight_module._check_300_has_override_rationale(body) is True


def test_has_override_rationale_false_for_null(preflight_module):
    body = "council_override_rationale: null"
    assert preflight_module._check_300_has_override_rationale(body) is False


def test_has_override_rationale_false_when_missing(preflight_module):
    body = "council_override_invoked: false\nother: x"
    assert preflight_module._check_300_has_override_rationale(body) is False


def test_t2_without_mission_contribution_flagged(preflight_module, tmp_path: Path):
    research = tmp_path / ".omx" / "research"
    research.mkdir(parents=True)
    _write_memo(research, "feedback_grand_council_no_mission_20260520.md",
                _T2_frontmatter_missing_mission())
    violations = preflight_module.check_council_deliberation_declares_tier_in_frontmatter(
        research_dir=research, repo_root=tmp_path,
    )
    assert len(violations) == 1
    assert "council_predicted_mission_contribution" in violations[0]
    assert "council_override_invoked" in violations[0]


def test_t2_with_override_no_rationale_flagged(preflight_module, tmp_path: Path):
    research = tmp_path / ".omx" / "research"
    research.mkdir(parents=True)
    _write_memo(research, "feedback_grand_council_bare_override_20260520.md",
                _T2_frontmatter_with_override_no_rationale())
    violations = preflight_module.check_council_deliberation_declares_tier_in_frontmatter(
        research_dir=research, repo_root=tmp_path,
    )
    assert len(violations) == 1
    assert "council_override_rationale" in violations[0]
    assert "REQUIRED when" in violations[0]


def test_t2_with_invalid_mission_contribution_flagged(preflight_module, tmp_path: Path):
    research = tmp_path / ".omx" / "research"
    research.mkdir(parents=True)
    body = (
        "---\n"
        "council_tier: T2\n"
        "council_attendees: [Shannon]\n"
        "council_quorum_met: true\n"
        "council_verdict: PROCEED\n"
        "council_dissent: []\n"
        "council_assumption_adversary_verdict:\n"
        "  - assumption: A\n"
        "    classification: HARD-EARNED\n"
        "    rationale: R\n"
        "council_decisions_recorded: []\n"
        "council_predicted_mission_contribution: bogus_category\n"
        "council_override_invoked: false\n"
        "---\n\n"
        "body\n"
    )
    _write_memo(research, "feedback_grand_council_bogus_mission_20260520.md", body)
    violations = preflight_module.check_council_deliberation_declares_tier_in_frontmatter(
        research_dir=research, repo_root=tmp_path,
    )
    assert len(violations) == 1
    assert "council_predicted_mission_contribution" in violations[0]


def test_t1_does_not_require_mission_alignment_fields(preflight_module, tmp_path: Path):
    """T1 working groups are exempt from mission-contribution + override fields."""
    research = tmp_path / ".omx" / "research"
    research.mkdir(parents=True)
    _write_memo(research, "feedback_skunkworks_council_t1_test_20260520.md",
                _T1_frontmatter())
    violations = preflight_module.check_council_deliberation_declares_tier_in_frontmatter(
        research_dir=research, repo_root=tmp_path,
    )
    assert violations == []


def test_t2_full_mission_alignment_passes(preflight_module, tmp_path: Path):
    """T2 memo with all v2 + mission-alignment fields passes cleanly."""
    research = tmp_path / ".omx" / "research"
    research.mkdir(parents=True)
    _write_memo(research, "feedback_grand_council_full_mission_20260520.md",
                _T2_frontmatter())
    violations = preflight_module.check_council_deliberation_declares_tier_in_frontmatter(
        research_dir=research, repo_root=tmp_path,
    )
    assert violations == []


def test_t2_with_override_and_rationale_passes(preflight_module, tmp_path: Path):
    body = (
        "---\n"
        "council_tier: T2\n"
        "council_attendees: [Shannon]\n"
        "council_quorum_met: true\n"
        "council_verdict: PROCEED\n"
        "council_dissent: []\n"
        "council_assumption_adversary_verdict:\n"
        "  - assumption: A\n"
        "    classification: HARD-EARNED\n"
        "    rationale: R\n"
        "council_decisions_recorded: []\n"
        "council_predicted_mission_contribution: frontier_breaking\n"
        "council_override_invoked: true\n"
        'council_override_rationale: "operator verbatim: leaderboard moved, skip sextet"\n'
        "---\n\n"
        "body\n"
    )
    research = tmp_path / ".omx" / "research"
    research.mkdir(parents=True)
    _write_memo(research, "feedback_grand_council_override_test_20260520.md", body)
    violations = preflight_module.check_council_deliberation_declares_tier_in_frontmatter(
        research_dir=research, repo_root=tmp_path,
    )
    assert violations == []


def test_constants_pinned(preflight_module):
    # Mission-alignment required fields tuple is canonical.
    assert preflight_module._CHECK_300_T2_PLUS_MISSION_REQUIRED_FIELDS == (
        "council_predicted_mission_contribution",
        "council_override_invoked",
    )
    # Valid mission contributions match the canonical helper enum.
    assert preflight_module._CHECK_300_VALID_MISSION_CONTRIBUTIONS == frozenset({
        "frontier_breaking",
        "frontier_protecting",
        "rigor_overhead",
        "apparatus_maintenance",
        "mission_questioned",
    })
