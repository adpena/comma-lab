# SPDX-License-Identifier: MIT
"""Tests for Catalog #294 ``check_substrate_landing_memo_has_9_dim_checklist_evidence_section``.

Per ``feedback_9_dimension_success_checklist_per_substrate_and_stack_of_stacks_standing_directive_20260515.md``
(operator standing directive 2026-05-15).

The gate refuses substrate landing + stack-of-stacks composition memos dated
>= 2026-05-15 that lack the literal section header
``## 9-dimension success checklist evidence`` (case-insensitive). Pre-cutoff
memos are exempt. Same-line waiver
``# 9_DIM_CHECKLIST_EVIDENCE_WAIVED:<rationale>`` accepted (placeholder
``<rationale>`` / ``<reason>`` literals rejected).

Sister of:
- Catalog #290 (canonical-vs-unique decision per layer — Dimension 5)
- Catalog #291 (per-session META-ASSUMPTION cadence)
- Catalog #292 (per-deliberation council assumption discipline)
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_substrate_landing_memo_has_9_dim_checklist_evidence_section,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write(memory_dir: Path, name: str, body: str) -> Path:
    memory_dir.mkdir(parents=True, exist_ok=True)
    path = memory_dir / name
    path.write_text(body, encoding="utf-8")
    return path


def _frontmatter(name: str = "test-substrate-landing") -> str:
    return (
        "---\n"
        f"name: {name}\n"
        "description: test fixture\n"
        "metadata:\n"
        "  node_type: memory\n"
        "  type: feedback\n"
        "---\n\n"
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_294_no_memory_dir_returns_empty(tmp_path: Path) -> None:
    """Gate returns [] when memory dir does not exist (not an error)."""
    violations = check_substrate_landing_memo_has_9_dim_checklist_evidence_section(
        memory_dir=tmp_path / "missing",
        research_dir=tmp_path / "missing_research",
        strict=False,
        verbose=False,
    )
    assert violations == []


def test_294_empty_memory_dir_returns_empty(tmp_path: Path) -> None:
    """Gate returns [] when memory dir is empty."""
    (tmp_path / "memory").mkdir()
    violations = check_substrate_landing_memo_has_9_dim_checklist_evidence_section(
        memory_dir=tmp_path / "memory",
        research_dir=tmp_path / "missing_research",
        strict=False,
        verbose=False,
    )
    assert violations == []


def test_294_default_scans_repo_research_only_not_external_memory(tmp_path: Path) -> None:
    """Default behavior is OSS-hermetic: only repo-local research is scanned."""
    _write(
        tmp_path / ".omx" / "research",
        "external_substrate_design_20260520.md",
        "# Design\n\nNo required section.\n",
    )
    violations = check_substrate_landing_memo_has_9_dim_checklist_evidence_section(
        repo_root=tmp_path,
        strict=False,
        verbose=False,
    )
    assert len(violations) == 1
    assert "external_substrate_design_20260520.md" in violations[0]


def test_294_pre_cutoff_memo_without_section_exempt(tmp_path: Path) -> None:
    """Pre-2026-05-15 substrate landing memos are exempt from the gate."""
    body = _frontmatter() + "Some substrate content with no 9-dim section.\n"
    _write(
        tmp_path / "memory",
        "feedback_pre_cutoff_substrate_landed_20260514.md",
        body,
    )
    violations = check_substrate_landing_memo_has_9_dim_checklist_evidence_section(
        memory_dir=tmp_path / "memory",
        research_dir=tmp_path / "missing_research",
        strict=False,
        verbose=False,
    )
    assert violations == []


def test_294_post_cutoff_memo_with_section_passes(tmp_path: Path) -> None:
    """Post-cutoff substrate landing memo with the 9-dim section passes."""
    body = (
        _frontmatter()
        + "Body intro.\n\n"
        + "## 9-dimension success checklist evidence\n\n"
        + "Dimension 1 UNIQUENESS: class-shift via X.\n"
        + "Dimension 2 BEAUTY+ELEGANCE: ~600 LOC.\n"
        + "(... etc ...)\n"
    )
    _write(
        tmp_path / "memory",
        "feedback_test_substrate_landed_20260516.md",
        body,
    )
    violations = check_substrate_landing_memo_has_9_dim_checklist_evidence_section(
        memory_dir=tmp_path / "memory",
        research_dir=tmp_path / "missing_research",
        strict=False,
        verbose=False,
    )
    assert violations == []


def test_294_post_cutoff_memo_without_section_flagged(tmp_path: Path) -> None:
    """Post-cutoff substrate landing memo without the section is flagged."""
    body = _frontmatter() + "Body content with no 9-dim evidence section.\n"
    _write(
        tmp_path / "memory",
        "feedback_test_substrate_landed_20260520.md",
        body,
    )
    violations = check_substrate_landing_memo_has_9_dim_checklist_evidence_section(
        memory_dir=tmp_path / "memory",
        research_dir=tmp_path / "missing_research",
        strict=False,
        verbose=False,
    )
    assert len(violations) == 1
    assert "feedback_test_substrate_landed_20260520.md" in violations[0]
    assert "9-dimension" in violations[0] or "9-DIMENSION" in violations[0].upper()


def test_294_research_design_memo_without_section_flagged(tmp_path: Path) -> None:
    """Active ``.omx/research/*_design_<date>.md`` memos are in scope."""
    research = tmp_path / ".omx" / "research"
    _write(
        research,
        "wunderkind_g1_entropy_coded_v2_design_20260520.md",
        "# Z3G2 Design\n\nNo required 9-dim section here.\n",
    )
    violations = check_substrate_landing_memo_has_9_dim_checklist_evidence_section(
        memory_dir=tmp_path / "missing_memory",
        research_dir=research,
        strict=False,
        verbose=False,
    )
    assert len(violations) == 1
    assert "wunderkind_g1_entropy_coded_v2_design_20260520.md" in violations[0]


def test_294_research_design_memo_with_section_passes(tmp_path: Path) -> None:
    """Research design memos pass only with the required header."""
    research = tmp_path / ".omx" / "research"
    _write(
        research,
        "nscs01_nullspace_split_renderer_design_20260520.md",
        "# NSCS01\n\n## 9-dimension success checklist evidence\n\nRows.\n",
    )
    violations = check_substrate_landing_memo_has_9_dim_checklist_evidence_section(
        memory_dir=tmp_path / "missing_memory",
        research_dir=research,
        strict=False,
        verbose=False,
    )
    assert violations == []


def test_294_section_header_case_insensitive(tmp_path: Path) -> None:
    """Section header detection is case-insensitive."""
    for header in [
        "## 9-Dimension Success Checklist Evidence",
        "## 9-DIMENSION SUCCESS CHECKLIST EVIDENCE",
        "## 9-dimension success checklist evidence",
    ]:
        body = _frontmatter() + f"Body.\n\n{header}\n\nEvidence.\n"
        _write(
            tmp_path / "memory",
            "feedback_case_substrate_landed_20260520.md",
            body,
        )
        violations = check_substrate_landing_memo_has_9_dim_checklist_evidence_section(
            memory_dir=tmp_path / "memory",
            research_dir=tmp_path / "missing_research",
            strict=False,
            verbose=False,
        )
        assert violations == [], f"Header `{header}` should be accepted"


def test_294_nscs_naming_pattern_in_scope(tmp_path: Path) -> None:
    """NSCS<N>_landed_<YYYYMMDD>.md memos are in scope per the directive."""
    body = _frontmatter() + "Body.\n"  # no section header
    _write(
        tmp_path / "memory",
        "feedback_nscs01_landed_20260520.md",
        body,
    )
    violations = check_substrate_landing_memo_has_9_dim_checklist_evidence_section(
        memory_dir=tmp_path / "memory",
        research_dir=tmp_path / "missing_research",
        strict=False,
        verbose=False,
    )
    assert len(violations) == 1
    assert "nscs01" in violations[0].lower()


def test_294_stack_of_stacks_composition_memo_in_scope(tmp_path: Path) -> None:
    """Stack-of-stacks composition memos are in scope per the directive."""
    body = _frontmatter() + "Body with composition discussion.\n"
    _write(
        tmp_path / "memory",
        "feedback_nscs01_nscs02_nscs03_stack_of_stacks_landed_20260520.md",
        body,
    )
    violations = check_substrate_landing_memo_has_9_dim_checklist_evidence_section(
        memory_dir=tmp_path / "memory",
        research_dir=tmp_path / "missing_research",
        strict=False,
        verbose=False,
    )
    assert len(violations) == 1
    assert "stack_of_stacks" in violations[0].lower()


def test_294_composition_memo_in_scope(tmp_path: Path) -> None:
    """Generic ``*composition*landed`` memos are in scope per the directive."""
    body = _frontmatter() + "Composition memo body.\n"
    _write(
        tmp_path / "memory",
        "feedback_atw_dasher_composition_landed_20260520.md",
        body,
    )
    violations = check_substrate_landing_memo_has_9_dim_checklist_evidence_section(
        memory_dir=tmp_path / "memory",
        research_dir=tmp_path / "missing_research",
        strict=False,
        verbose=False,
    )
    assert len(violations) == 1
    assert "composition" in violations[0].lower()


def test_294_non_substrate_non_composition_memo_out_of_scope(tmp_path: Path) -> None:
    """Memos that are neither substrate nor composition nor design are skipped."""
    body = _frontmatter() + "Body without section header.\n"
    _write(
        tmp_path / "memory",
        "feedback_unrelated_topic_landed_20260520.md",
        body,
    )
    violations = check_substrate_landing_memo_has_9_dim_checklist_evidence_section(
        memory_dir=tmp_path / "memory",
        research_dir=tmp_path / "missing_research",
        strict=False,
        verbose=False,
    )
    assert violations == []


def test_294_same_line_waiver_with_rationale_accepted(tmp_path: Path) -> None:
    """Same-line waiver with a real rationale accepts the memo."""
    body = (
        _frontmatter()
        + "Body without the 9-dim section.\n\n"
        + "# 9_DIM_CHECKLIST_EVIDENCE_WAIVED:emergency-hotfix-2026-05-16 substrate landed before standing directive existed; backfilled in follow-up\n"
    )
    _write(
        tmp_path / "memory",
        "feedback_waiver_substrate_landed_20260520.md",
        body,
    )
    violations = check_substrate_landing_memo_has_9_dim_checklist_evidence_section(
        memory_dir=tmp_path / "memory",
        research_dir=tmp_path / "missing_research",
        strict=False,
        verbose=False,
    )
    assert violations == []


def test_294_placeholder_rationale_waiver_rejected(tmp_path: Path) -> None:
    """Placeholder ``<rationale>`` literal waiver MUST be rejected."""
    body = (
        _frontmatter()
        + "Body without the 9-dim section.\n\n"
        + "# 9_DIM_CHECKLIST_EVIDENCE_WAIVED:<rationale>\n"
    )
    _write(
        tmp_path / "memory",
        "feedback_placeholder_substrate_landed_20260520.md",
        body,
    )
    violations = check_substrate_landing_memo_has_9_dim_checklist_evidence_section(
        memory_dir=tmp_path / "memory",
        research_dir=tmp_path / "missing_research",
        strict=False,
        verbose=False,
    )
    assert len(violations) == 1


def test_294_placeholder_reason_waiver_rejected(tmp_path: Path) -> None:
    """Placeholder ``<reason>`` literal waiver MUST be rejected."""
    body = (
        _frontmatter()
        + "Body without the 9-dim section.\n\n"
        + "# 9_DIM_CHECKLIST_EVIDENCE_WAIVED:<reason>\n"
    )
    _write(
        tmp_path / "memory",
        "feedback_placeholder2_substrate_landed_20260520.md",
        body,
    )
    violations = check_substrate_landing_memo_has_9_dim_checklist_evidence_section(
        memory_dir=tmp_path / "memory",
        research_dir=tmp_path / "missing_research",
        strict=False,
        verbose=False,
    )
    assert len(violations) == 1


def test_294_empty_waiver_rejected(tmp_path: Path) -> None:
    """Empty (bare) waiver tag MUST be rejected — no rationale = no waiver."""
    body = (
        _frontmatter()
        + "Body without the 9-dim section.\n\n"
        + "# 9_DIM_CHECKLIST_EVIDENCE_WAIVED:   \n"
    )
    _write(
        tmp_path / "memory",
        "feedback_emptywaiver_substrate_landed_20260520.md",
        body,
    )
    violations = check_substrate_landing_memo_has_9_dim_checklist_evidence_section(
        memory_dir=tmp_path / "memory",
        research_dir=tmp_path / "missing_research",
        strict=False,
        verbose=False,
    )
    assert len(violations) == 1


def test_294_strict_mode_raises_on_violation(tmp_path: Path) -> None:
    """Strict mode raises ``PreflightError`` on violation."""
    body = _frontmatter() + "Body.\n"
    _write(
        tmp_path / "memory",
        "feedback_strict_substrate_landed_20260520.md",
        body,
    )
    with pytest.raises(PreflightError) as exc_info:
        check_substrate_landing_memo_has_9_dim_checklist_evidence_section(
            memory_dir=tmp_path / "memory",
            research_dir=tmp_path / "missing_research",
            strict=True,
            verbose=False,
        )
    msg = str(exc_info.value)
    assert "Catalog #294" in msg
    assert "9-dimension" in msg or "9-DIMENSION" in msg.upper()


def test_294_strict_mode_silent_on_clean(tmp_path: Path) -> None:
    """Strict mode is silent (no raise) when corpus is clean."""
    body = (
        _frontmatter()
        + "Body.\n\n## 9-dimension success checklist evidence\n\nAll dims.\n"
    )
    _write(
        tmp_path / "memory",
        "feedback_clean_substrate_landed_20260520.md",
        body,
    )
    # Should not raise
    violations = check_substrate_landing_memo_has_9_dim_checklist_evidence_section(
        memory_dir=tmp_path / "memory",
        research_dir=tmp_path / "missing_research",
        strict=True,
        verbose=False,
    )
    assert violations == []


def test_294_aggregates_multiple_violations(tmp_path: Path) -> None:
    """Multiple violating memos are all reported."""
    for i in range(3):
        body = _frontmatter() + f"Body {i}.\n"
        _write(
            tmp_path / "memory",
            f"feedback_multi{i}_substrate_landed_20260520.md",
            body,
        )
    violations = check_substrate_landing_memo_has_9_dim_checklist_evidence_section(
        memory_dir=tmp_path / "memory",
        research_dir=tmp_path / "missing_research",
        strict=False,
        verbose=False,
    )
    assert len(violations) == 3


def test_294_string_memory_dir_accepted(tmp_path: Path) -> None:
    """Function accepts a string ``memory_dir`` not just Path."""
    body = _frontmatter() + "Body.\n"
    _write(
        tmp_path / "memory",
        "feedback_str_substrate_landed_20260520.md",
        body,
    )
    # Pass as str
    violations = check_substrate_landing_memo_has_9_dim_checklist_evidence_section(
        memory_dir=str(tmp_path / "memory"),
        research_dir=tmp_path / "missing_research",
        strict=False,
        verbose=False,
    )
    assert len(violations) == 1


def test_294_invalid_date_suffix_skipped(tmp_path: Path) -> None:
    """Memos with un-parseable date suffix are skipped (not flagged)."""
    body = _frontmatter() + "Body.\n"
    _write(
        tmp_path / "memory",
        "feedback_baddate_substrate_landed_NOTADATE.md",
        body,
    )
    violations = check_substrate_landing_memo_has_9_dim_checklist_evidence_section(
        memory_dir=tmp_path / "memory",
        research_dir=tmp_path / "missing_research",
        strict=False,
        verbose=False,
    )
    assert violations == []


def test_294_external_memory_only_via_explicit_opt_in(tmp_path: Path) -> None:
    """External memory dir is scanned ONLY when ``memory_dir=...`` is passed.

    OSS-hermetic discipline: clone-stable default scans repo-local research only;
    external scan is explicit opt-in via the kwarg.
    """
    # Create both repo-local and memory_dir violations.
    _write(
        tmp_path / ".omx" / "research",
        "in_repo_substrate_design_20260520.md",
        "no section\n",
    )
    _write(
        tmp_path / "external_memory",
        "feedback_external_substrate_landed_20260520.md",
        "no section\n",
    )

    # Default: only repo-local research is scanned (1 violation).
    violations_default = check_substrate_landing_memo_has_9_dim_checklist_evidence_section(
        repo_root=tmp_path,
        strict=False,
        verbose=False,
    )
    assert len(violations_default) == 1
    assert "in_repo_substrate_design_20260520.md" in violations_default[0]

    # Explicit opt-in: both surfaces scanned (2 violations).
    violations_opt_in = check_substrate_landing_memo_has_9_dim_checklist_evidence_section(
        memory_dir=tmp_path / "external_memory",
        repo_root=tmp_path,
        strict=False,
        verbose=False,
    )
    assert len(violations_opt_in) == 2


def test_294_orchestrator_wired_warn_only(tmp_path: Path) -> None:
    """Regression guard: confirm orchestrator calls Catalog #294 with strict=False.

    This protects against accidental strict-flip before the operator-routed
    backfill wave clears live violations.
    """
    import inspect
    from tac import preflight as _preflight_module

    source = inspect.getsource(_preflight_module.preflight_all)
    # The gate name appears at the orchestrator wire-in site.
    assert (
        "check_substrate_landing_memo_has_9_dim_checklist_evidence_section"
        in source
    ), "Catalog #294 must be wired into preflight_all"
    # And the wire-in must currently be strict=False (warn-only).
    wire_in_idx = source.index(
        "check_substrate_landing_memo_has_9_dim_checklist_evidence_section"
    )
    # Find the next 200 chars after the gate name; must include strict=False.
    snippet = source[wire_in_idx : wire_in_idx + 200]
    assert "strict=False" in snippet, (
        f"Catalog #294 wire-in must currently be strict=False (warn-only) "
        f"per CLAUDE.md 'Strict-flip atomicity rule'. Snippet: {snippet}"
    )


def test_294_live_repo_regression_guard() -> None:
    """Live repo bounded count guard.

    At landing the live count is small (~6 violations in
    ``.omx/research/*_design_20260515.md``). This guard refuses a future
    regression that adds many more violations without intentional backfill.
    """
    violations = check_substrate_landing_memo_has_9_dim_checklist_evidence_section(
        strict=False,
        verbose=False,
    )
    # Bounded: at landing 6 violations. Allow up to 30 as a soft ceiling
    # for future in-flight scaffolds before the backfill wave lands. If you
    # need to land more, the answer is to backfill the section header into
    # the existing memos, not to raise this ceiling.
    assert isinstance(violations, list)
    assert len(violations) <= 30, (
        f"Live repo violation count {len(violations)} exceeds the soft "
        f"ceiling of 30. Either backfill the '## 9-dimension success "
        f"checklist evidence' section into the violating memos OR raise "
        f"this ceiling explicitly with a code review rationale."
    )
