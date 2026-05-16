# SPDX-License-Identifier: MIT
"""Tests for Catalog #292 ``check_grand_council_deliberation_has_explicit_assumption_statements``.

Per ``feedback_or2_grand_council_per_round_assumption_statement_discipline_landed_20260515.md``
+ ``feedback_adversarial_review_apparatus_blind_to_shared_assumption_failure_meta_meta_meta_meta_20260515.md``
Fix 7.

The gate refuses post-cutoff (>= 2026-05-15) grand council deliberation memos
that don't carry the per-round explicit-assumption-statement discipline tokens:

  (a) per-member operating-within phrase, OR
  (b) Assumption-Adversary HARD-EARNED vs CARGO-CULTED evaluation block, OR
  (c) same-line `# COUNCIL_ASSUMPTION_STATEMENT_WAIVED:<rationale>` waiver.

Sister of Catalog #229 (premise-verification), Catalog #290 (substrate
canonical-vs-unique), Catalog #291 (META-ASSUMPTION cadence), Catalog #185
(META-meta-meta drift detector).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_grand_council_deliberation_has_explicit_assumption_statements,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CANONICAL_OPERATING_WITHIN_BODY = (
    "# Council deliberation\n\n"
    "Shannon: the shared assumption I am operating within for this design is "
    "that rate-distortion theory bounds the achievable score.\n\n"
    "Dykstra: my operating assumption for this round is convex feasibility.\n\n"
    "Verdict: PROCEED with explicit framing.\n"
)

CANONICAL_ASSUMPTION_ADVERSARY_BODY = (
    "# Council deliberation\n\n"
    "Phase A: surfaced assumptions A1-A5.\n"
    "Assumption-Adversary evaluation:\n"
    "  A1 = HARD-EARNED (cite source — preserve)\n"
    "  A2 = CARGO-CULTED (eligible for challenge)\n"
    "  A3 = HARD-EARNED\n\n"
    "Verdict: PROCEED with A2 violation hypothesis queued.\n"
)

LEGACY_BODY_WITHOUT_DISCIPLINE = (
    "# Council deliberation\n\nShannon: PROCEED.\nDykstra: PROCEED.\n"
    "Verdict: 5-of-5 unanimous PROCEED.\n"
)


def _write_council_memo(
    memory_dir: Path,
    name: str,
    *,
    body: str = LEGACY_BODY_WITHOUT_DISCIPLINE,
) -> Path:
    memory_dir.mkdir(parents=True, exist_ok=True)
    path = memory_dir / name
    path.write_text(body, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_292_empty_memory_dir_passes(tmp_path: Path) -> None:
    """No council memos -> no violations."""
    (tmp_path / "memory").mkdir()
    violations = check_grand_council_deliberation_has_explicit_assumption_statements(
        memory_dir=tmp_path / "memory",
        research_dir=tmp_path / "research_missing",
        repo_root=tmp_path,
        strict=False,
        verbose=False,
    )
    assert violations == []


def test_292_missing_memory_dir_passes(tmp_path: Path) -> None:
    """Memory dir doesn't exist -> no violations (no work to do)."""
    violations = check_grand_council_deliberation_has_explicit_assumption_statements(
        memory_dir=tmp_path / "missing",
        research_dir=tmp_path / "also_missing",
        repo_root=tmp_path,
        strict=False,
        verbose=False,
    )
    assert violations == []


def test_292_operating_within_phrase_accepted(tmp_path: Path) -> None:
    """A council memo with the operating-within phrase passes."""
    memory = tmp_path / "memory"
    _write_council_memo(
        memory,
        "feedback_grand_council_test_20260515.md",
        body=CANONICAL_OPERATING_WITHIN_BODY,
    )
    violations = check_grand_council_deliberation_has_explicit_assumption_statements(
        memory_dir=memory,
        research_dir=tmp_path / "research_missing",
        repo_root=tmp_path,
        strict=False,
        verbose=False,
    )
    assert violations == []


def test_292_assumption_adversary_block_accepted(tmp_path: Path) -> None:
    """A council memo with the Assumption-Adversary evaluation block passes."""
    memory = tmp_path / "memory"
    _write_council_memo(
        memory,
        "feedback_grand_council_test_20260515.md",
        body=CANONICAL_ASSUMPTION_ADVERSARY_BODY,
    )
    violations = check_grand_council_deliberation_has_explicit_assumption_statements(
        memory_dir=memory,
        research_dir=tmp_path / "research_missing",
        repo_root=tmp_path,
        strict=False,
        verbose=False,
    )
    assert violations == []


def test_292_post_cutoff_legacy_council_flagged(tmp_path: Path) -> None:
    """A 20260515 council memo without discipline tokens is flagged."""
    memory = tmp_path / "memory"
    _write_council_memo(
        memory,
        "feedback_grand_council_legacy_20260515.md",
        body=LEGACY_BODY_WITHOUT_DISCIPLINE,
    )
    violations = check_grand_council_deliberation_has_explicit_assumption_statements(
        memory_dir=memory,
        research_dir=tmp_path / "research_missing",
        repo_root=tmp_path,
        strict=False,
        verbose=False,
    )
    assert len(violations) == 1
    assert "per-round explicit-assumption-statement discipline" in violations[0]
    assert "Catalog #292" not in violations[0]  # the violation msg doesn't say Catalog
    assert "operating-within" in violations[0]


def test_292_pre_cutoff_council_exempt(tmp_path: Path) -> None:
    """A 20260514 (pre-cutoff) council memo WITHOUT discipline tokens passes."""
    memory = tmp_path / "memory"
    _write_council_memo(
        memory,
        "feedback_grand_council_legacy_20260514.md",
        body=LEGACY_BODY_WITHOUT_DISCIPLINE,
    )
    violations = check_grand_council_deliberation_has_explicit_assumption_statements(
        memory_dir=memory,
        research_dir=tmp_path / "research_missing",
        repo_root=tmp_path,
        strict=False,
        verbose=False,
    )
    assert violations == []


def test_292_pre_cutoff_council_with_2026_dec_date_exempt_correctly(
    tmp_path: Path,
) -> None:
    """Edge case: a 2025-12-31 council memo is well before cutoff."""
    memory = tmp_path / "memory"
    _write_council_memo(
        memory,
        "feedback_grand_council_ancient_20251231.md",
        body=LEGACY_BODY_WITHOUT_DISCIPLINE,
    )
    violations = check_grand_council_deliberation_has_explicit_assumption_statements(
        memory_dir=memory,
        research_dir=tmp_path / "research_missing",
        repo_root=tmp_path,
        strict=False,
        verbose=False,
    )
    assert violations == []


def test_292_post_cutoff_council_with_2026_jun_date_in_scope(
    tmp_path: Path,
) -> None:
    """A 2026-06-01 council memo (future, post-cutoff) IS in scope."""
    memory = tmp_path / "memory"
    _write_council_memo(
        memory,
        "feedback_grand_council_future_20260601.md",
        body=LEGACY_BODY_WITHOUT_DISCIPLINE,
    )
    violations = check_grand_council_deliberation_has_explicit_assumption_statements(
        memory_dir=memory,
        research_dir=tmp_path / "research_missing",
        repo_root=tmp_path,
        strict=False,
        verbose=False,
    )
    assert len(violations) == 1


def test_292_skunkworks_council_in_scope(tmp_path: Path) -> None:
    """Skunkworks council memos are in scope."""
    memory = tmp_path / "memory"
    _write_council_memo(
        memory,
        "feedback_skunkworks_council_round_5_20260515.md",
        body=LEGACY_BODY_WITHOUT_DISCIPLINE,
    )
    violations = check_grand_council_deliberation_has_explicit_assumption_statements(
        memory_dir=memory,
        research_dir=tmp_path / "research_missing",
        repo_root=tmp_path,
        strict=False,
        verbose=False,
    )
    assert len(violations) == 1


def test_292_grand_reunion_symposium_in_scope(tmp_path: Path) -> None:
    """Grand reunion symposium memos are in scope (council-grade deliberation)."""
    memory = tmp_path / "memory"
    _write_council_memo(
        memory,
        "feedback_grand_reunion_fields_grade_passion_symposium_20260515.md",
        body=LEGACY_BODY_WITHOUT_DISCIPLINE,
    )
    violations = check_grand_council_deliberation_has_explicit_assumption_statements(
        memory_dir=memory,
        research_dir=tmp_path / "research_missing",
        repo_root=tmp_path,
        strict=False,
        verbose=False,
    )
    assert len(violations) == 1


def test_292_non_council_memo_out_of_scope(tmp_path: Path) -> None:
    """Non-council memos (e.g. landing memos, codex review memos) are
    out of scope of #292; Catalog #229 handles their discipline."""
    memory = tmp_path / "memory"
    _write_council_memo(
        memory,
        "feedback_some_random_landing_20260515.md",
        body=LEGACY_BODY_WITHOUT_DISCIPLINE,
    )
    _write_council_memo(
        memory,
        "feedback_codex_review_finding_20260515.md",
        body=LEGACY_BODY_WITHOUT_DISCIPLINE,
    )
    violations = check_grand_council_deliberation_has_explicit_assumption_statements(
        memory_dir=memory,
        research_dir=tmp_path / "research_missing",
        repo_root=tmp_path,
        strict=False,
        verbose=False,
    )
    assert violations == []


def test_292_same_line_waiver_accepted(tmp_path: Path) -> None:
    """Same-line `# COUNCIL_ASSUMPTION_STATEMENT_WAIVED:<rationale>` waiver."""
    memory = tmp_path / "memory"
    _write_council_memo(
        memory,
        "feedback_grand_council_text_only_20260515.md",
        body=(
            "# Council deliberation\n\n"
            "Verdict: PROCEED.\n"
            "# COUNCIL_ASSUMPTION_STATEMENT_WAIVED:text-only briefing not "
            "design deliberation; sister #229 handles it.\n"
        ),
    )
    violations = check_grand_council_deliberation_has_explicit_assumption_statements(
        memory_dir=memory,
        research_dir=tmp_path / "research_missing",
        repo_root=tmp_path,
        strict=False,
        verbose=False,
    )
    assert violations == []


def test_292_waiver_placeholder_rationale_rejected(tmp_path: Path) -> None:
    """Placeholder `<rationale>` rejected so the docstring example can't
    self-waive."""
    memory = tmp_path / "memory"
    _write_council_memo(
        memory,
        "feedback_grand_council_placeholder_20260515.md",
        body=(
            "# Council deliberation\n\n"
            "Verdict: PROCEED.\n"
            "# COUNCIL_ASSUMPTION_STATEMENT_WAIVED:<rationale>\n"
        ),
    )
    violations = check_grand_council_deliberation_has_explicit_assumption_statements(
        memory_dir=memory,
        research_dir=tmp_path / "research_missing",
        repo_root=tmp_path,
        strict=False,
        verbose=False,
    )
    assert len(violations) == 1


def test_292_waiver_placeholder_reason_rejected(tmp_path: Path) -> None:
    """Placeholder `<reason>` literal also rejected."""
    memory = tmp_path / "memory"
    _write_council_memo(
        memory,
        "feedback_grand_council_placeholder_reason_20260515.md",
        body=(
            "# Council deliberation\n\n"
            "Verdict: PROCEED.\n"
            "# COUNCIL_ASSUMPTION_STATEMENT_WAIVED:<reason>\n"
        ),
    )
    violations = check_grand_council_deliberation_has_explicit_assumption_statements(
        memory_dir=memory,
        research_dir=tmp_path / "research_missing",
        repo_root=tmp_path,
        strict=False,
        verbose=False,
    )
    assert len(violations) == 1


def test_292_waiver_with_non_placeholder_rationale_accepted(
    tmp_path: Path,
) -> None:
    """A waiver with substantive rationale passes."""
    memory = tmp_path / "memory"
    _write_council_memo(
        memory,
        "feedback_grand_council_real_waiver_20260515.md",
        body=(
            "# Council deliberation\n\n"
            "Verdict: PROCEED.\n"
            "# COUNCIL_ASSUMPTION_STATEMENT_WAIVED:operator-approved emergency "
            "deliberation; full discipline applies next round\n"
        ),
    )
    violations = check_grand_council_deliberation_has_explicit_assumption_statements(
        memory_dir=memory,
        research_dir=tmp_path / "research_missing",
        repo_root=tmp_path,
        strict=False,
        verbose=False,
    )
    assert violations == []


def test_292_strict_mode_raises(tmp_path: Path) -> None:
    """Strict mode raises PreflightError citing Catalog #292."""
    memory = tmp_path / "memory"
    _write_council_memo(
        memory,
        "feedback_grand_council_legacy_20260515.md",
        body=LEGACY_BODY_WITHOUT_DISCIPLINE,
    )
    with pytest.raises(PreflightError) as exc:
        check_grand_council_deliberation_has_explicit_assumption_statements(
            memory_dir=memory,
            research_dir=tmp_path / "research_missing",
            repo_root=tmp_path,
            strict=True,
            verbose=False,
        )
    msg = str(exc.value)
    assert "Catalog #292" in msg
    assert "Council conduct" in msg
    assert "Fix-7" in msg


def test_292_strict_mode_silent_on_clean(tmp_path: Path) -> None:
    """Strict mode does not raise on a clean memory dir."""
    memory = tmp_path / "memory"
    _write_council_memo(
        memory,
        "feedback_grand_council_clean_20260515.md",
        body=CANONICAL_OPERATING_WITHIN_BODY,
    )
    violations = check_grand_council_deliberation_has_explicit_assumption_statements(
        memory_dir=memory,
        research_dir=tmp_path / "research_missing",
        repo_root=tmp_path,
        strict=True,
        verbose=False,
    )
    assert violations == []


def test_292_string_repo_root_accepted(tmp_path: Path) -> None:
    """str repo_root + memory_dir + research_dir all accepted."""
    memory = tmp_path / "memory"
    _write_council_memo(
        memory,
        "feedback_grand_council_test_20260515.md",
        body=CANONICAL_OPERATING_WITHIN_BODY,
    )
    violations = check_grand_council_deliberation_has_explicit_assumption_statements(
        memory_dir=str(memory),
        research_dir=str(tmp_path / "research_missing"),
        repo_root=str(tmp_path),
        strict=False,
        verbose=False,
    )
    assert violations == []


def test_292_multiple_violations_aggregated(tmp_path: Path) -> None:
    """Multiple legacy council memos all surface as violations."""
    memory = tmp_path / "memory"
    _write_council_memo(
        memory,
        "feedback_grand_council_a_20260515.md",
        body=LEGACY_BODY_WITHOUT_DISCIPLINE,
    )
    _write_council_memo(
        memory,
        "feedback_grand_council_b_20260515.md",
        body=LEGACY_BODY_WITHOUT_DISCIPLINE,
    )
    _write_council_memo(
        memory,
        "feedback_skunkworks_council_c_20260515.md",
        body=LEGACY_BODY_WITHOUT_DISCIPLINE,
    )
    violations = check_grand_council_deliberation_has_explicit_assumption_statements(
        memory_dir=memory,
        research_dir=tmp_path / "research_missing",
        repo_root=tmp_path,
        strict=False,
        verbose=False,
    )
    assert len(violations) == 3


def test_292_repo_local_research_dir_scanned_by_default(tmp_path: Path) -> None:
    """When memory_dir is None, repo-local research_dir is scanned."""
    research = tmp_path / ".omx" / "research"
    research.mkdir(parents=True)
    (research / "feedback_grand_council_research_20260515.md").write_text(
        LEGACY_BODY_WITHOUT_DISCIPLINE, encoding="utf-8"
    )
    violations = check_grand_council_deliberation_has_explicit_assumption_statements(
        repo_root=tmp_path,
        strict=False,
        verbose=False,
    )
    assert len(violations) == 1


def test_292_hard_earned_cargo_culted_recognized(tmp_path: Path) -> None:
    """The HARD-EARNED vs CARGO-CULTED classification tokens are recognized
    as an Assumption-Adversary evaluation block."""
    memory = tmp_path / "memory"
    _write_council_memo(
        memory,
        "feedback_grand_council_classified_20260515.md",
        body=(
            "# Council deliberation\n\n"
            "Round 1 classification: hard-earned vs cargo-culted\n"
            "A1 hard-earned. A2 cargo-culted.\n"
            "Verdict: PROCEED.\n"
        ),
    )
    violations = check_grand_council_deliberation_has_explicit_assumption_statements(
        memory_dir=memory,
        research_dir=tmp_path / "research_missing",
        repo_root=tmp_path,
        strict=False,
        verbose=False,
    )
    assert violations == []


def test_292_case_insensitive_token_matching(tmp_path: Path) -> None:
    """Tokens are case-insensitive."""
    memory = tmp_path / "memory"
    _write_council_memo(
        memory,
        "feedback_grand_council_caps_20260515.md",
        body=(
            "# Council deliberation\n\n"
            "Shannon: THE SHARED ASSUMPTION I am OPERATING WITHIN here is X.\n"
            "Verdict: PROCEED.\n"
        ),
    )
    violations = check_grand_council_deliberation_has_explicit_assumption_statements(
        memory_dir=memory,
        research_dir=tmp_path / "research_missing",
        repo_root=tmp_path,
        strict=False,
        verbose=False,
    )
    assert violations == []


def test_292_verbose_clean_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verbose mode prints OK on clean."""
    memory = tmp_path / "memory"
    _write_council_memo(
        memory,
        "feedback_grand_council_clean_20260515.md",
        body=CANONICAL_OPERATING_WITHIN_BODY,
    )
    check_grand_council_deliberation_has_explicit_assumption_statements(
        memory_dir=memory,
        research_dir=tmp_path / "research_missing",
        repo_root=tmp_path,
        strict=False,
        verbose=True,
    )
    captured = capsys.readouterr()
    assert "check_grand_council_deliberation_has_explicit_assumption_statements" in captured.out
    assert "OK" in captured.out


def test_292_verbose_violation_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verbose mode prints violation count when violations are found."""
    memory = tmp_path / "memory"
    _write_council_memo(
        memory,
        "feedback_grand_council_legacy_20260515.md",
        body=LEGACY_BODY_WITHOUT_DISCIPLINE,
    )
    check_grand_council_deliberation_has_explicit_assumption_statements(
        memory_dir=memory,
        research_dir=tmp_path / "research_missing",
        repo_root=tmp_path,
        strict=False,
        verbose=True,
    )
    captured = capsys.readouterr()
    assert "1 violation(s)" in captured.out


def test_292_invalid_date_suffix_skipped(tmp_path: Path) -> None:
    """Filenames with invalid date suffixes (e.g. month 13) are skipped."""
    memory = tmp_path / "memory"
    _write_council_memo(
        memory,
        "feedback_grand_council_bad_date_20261301.md",
        body=LEGACY_BODY_WITHOUT_DISCIPLINE,
    )
    violations = check_grand_council_deliberation_has_explicit_assumption_statements(
        memory_dir=memory,
        research_dir=tmp_path / "research_missing",
        repo_root=tmp_path,
        strict=False,
        verbose=False,
    )
    assert violations == []


def test_292_unrelated_md_files_ignored(tmp_path: Path) -> None:
    """Markdown files not matching the council pattern are ignored."""
    memory = tmp_path / "memory"
    (memory).mkdir(parents=True)
    (memory / "README.md").write_text("just a readme", encoding="utf-8")
    (memory / "feedback_some_landing_20260515.md").write_text(
        LEGACY_BODY_WITHOUT_DISCIPLINE, encoding="utf-8"
    )
    violations = check_grand_council_deliberation_has_explicit_assumption_statements(
        memory_dir=memory,
        research_dir=tmp_path / "research_missing",
        repo_root=tmp_path,
        strict=False,
        verbose=False,
    )
    assert violations == []


def test_292_helper_parse_date_suffix() -> None:
    """The internal date-parser helper handles malformed inputs."""
    from tac.preflight import _check_292_parse_date_suffix
    assert _check_292_parse_date_suffix("20260515") == 20260515
    assert _check_292_parse_date_suffix("not_a_date") is None
    assert _check_292_parse_date_suffix("2026") is None
    assert _check_292_parse_date_suffix("20261301") is None
    assert _check_292_parse_date_suffix("20260132") is None


def test_292_helper_has_waiver_accepts_real_rationale() -> None:
    """The internal waiver helper accepts real rationale."""
    from tac.preflight import _check_292_has_waiver
    assert _check_292_has_waiver(
        "# COUNCIL_ASSUMPTION_STATEMENT_WAIVED:operator-approved deferred"
    )


def test_292_helper_has_waiver_rejects_placeholders() -> None:
    """The internal waiver helper rejects placeholder rationales."""
    from tac.preflight import _check_292_has_waiver
    assert not _check_292_has_waiver(
        "# COUNCIL_ASSUMPTION_STATEMENT_WAIVED:<rationale>"
    )
    assert not _check_292_has_waiver(
        "# COUNCIL_ASSUMPTION_STATEMENT_WAIVED:<reason>"
    )


def test_292_constants_pinned() -> None:
    """The cutoff date and token sets are part of the protection contract."""
    from tac.preflight import (
        _CHECK_292_ASSUMPTION_ADVERSARY_TOKENS,
        _CHECK_292_CUTOFF_DATE_SUFFIX_INT,
        _CHECK_292_OPERATING_WITHIN_TOKENS,
    )
    assert _CHECK_292_CUTOFF_DATE_SUFFIX_INT == 20260515
    assert "operating within" in _CHECK_292_OPERATING_WITHIN_TOKENS
    assert "shared assumption i am" in _CHECK_292_OPERATING_WITHIN_TOKENS
    assert "the shared assumption" in _CHECK_292_OPERATING_WITHIN_TOKENS
    assert "assumption-adversary" in _CHECK_292_ASSUMPTION_ADVERSARY_TOKENS
    assert "hard-earned vs cargo-culted" in _CHECK_292_ASSUMPTION_ADVERSARY_TOKENS
    assert "cargo-culted" in _CHECK_292_ASSUMPTION_ADVERSARY_TOKENS


def test_292_live_repo_regression_guard() -> None:
    """Live-repo default-scan (research-only) is clean per WARN-ONLY landing.

    Default scan does NOT include external Claude memory; OSS-hermetic per
    sister Catalog #291 design. Pre-Fix-7 council memos in .claude/memory
    are out-of-scope for default scan.
    """
    violations = check_grand_council_deliberation_has_explicit_assumption_statements(
        strict=False, verbose=False,
    )
    # Default scan = repo-local .omx/research only. Live count at landing: 0.
    # Bounded sentinel: > 5 means a real drift in repo-local council memos.
    assert len(violations) <= 5, (
        f"Catalog #292 unexpected repo-local drift: {len(violations)} violations.\n"
        + "\n".join(violations[:5])
    )


def test_292_orchestrator_callsite_warn_only_regression_guard() -> None:
    """The orchestrator must call this gate with strict=False (warn-only)
    initially per CLAUDE.md 'Strict-flip atomicity rule'."""
    import inspect

    from tac import preflight as pf
    src = inspect.getsource(pf.preflight_all)
    assert "check_grand_council_deliberation_has_explicit_assumption_statements(" in src, (
        "Catalog #292 must be wired into preflight_all"
    )
    idx = src.find(
        "check_grand_council_deliberation_has_explicit_assumption_statements("
    )
    snippet = src[idx : idx + 200]
    assert "strict=False" in snippet, (
        f"Catalog #292 should be warn-only at landing; got: {snippet[:200]}"
    )


def test_292_function_callable_via_preflight_module_globals() -> None:
    """Catalog #185 sister regression: the gate function MUST be importable
    via tac.preflight module globals so the META-meta-meta drift detector
    can resolve it."""
    from tac import preflight as pf
    assert hasattr(
        pf, "check_grand_council_deliberation_has_explicit_assumption_statements"
    )
    assert callable(
        pf.check_grand_council_deliberation_has_explicit_assumption_statements
    )


def test_292_external_memory_dir_opt_in_only(tmp_path: Path) -> None:
    """External memory_dir requires explicit opt-in; default repo-local
    scan does not surprise CI / clean clones with operator-machine state."""
    # Default scan with only research_dir hint (research is missing)
    violations = check_grand_council_deliberation_has_explicit_assumption_statements(
        research_dir=tmp_path / "missing_research",
        repo_root=tmp_path,
        strict=False,
        verbose=False,
    )
    assert violations == []


def test_292_dedup_research_and_memory_path(tmp_path: Path) -> None:
    """If memory_dir == research_dir, the directory is only scanned once."""
    same_dir = tmp_path / "shared"
    _write_council_memo(
        same_dir,
        "feedback_grand_council_legacy_20260515.md",
        body=LEGACY_BODY_WITHOUT_DISCIPLINE,
    )
    violations = check_grand_council_deliberation_has_explicit_assumption_statements(
        memory_dir=same_dir,
        research_dir=same_dir,
        repo_root=tmp_path,
        strict=False,
        verbose=False,
    )
    # 1 council memo, scanned once (not twice).
    assert len(violations) == 1
