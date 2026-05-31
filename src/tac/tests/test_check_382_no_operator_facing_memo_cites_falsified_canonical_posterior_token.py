# SPDX-License-Identifier: MIT
"""Tests for Catalog #382 STRICT preflight gate.

Sister of Catalog #321/#322 WRITE-surface STRICT gates at the READ surface.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    _CHECK_382_DISCIPLINE_CUTOFF_DATE,
    _CHECK_382_KNOWN_PHANTOM_TOKEN_SEEDS,
    _CHECK_382_WAIVER_PLACEHOLDERS,
    _CHECK_382_WAIVER_TOKEN,
    _check_382_extract_cited_phantom_tokens,
    _check_382_extract_date_suffix,
    _check_382_has_valid_waiver,
    check_no_operator_facing_memo_cites_falsified_canonical_posterior_token,
)


# -------- Live-repo regression guard --------


def test_live_repo_catalog_382_count_zero():
    """Catalog #382 live count at landing should be 0 (no flagged memos)."""
    violations = check_no_operator_facing_memo_cites_falsified_canonical_posterior_token(
        strict=False, verbose=False
    )
    # We've added the design memo for THIS gate which is self-exempt.
    # No other post-cutoff memo cites phantom tokens (cutoff = 20260530).
    assert len(violations) == 0, (
        f"Live count should be 0 at landing; got {len(violations)} violations:\n"
        + "\n".join(violations)
    )


def test_live_repo_orchestrator_wires_strict_false():
    """Orchestrator callsite is strict=False (WARN-ONLY at landing)."""
    src_path = Path(__file__).resolve().parent.parent / "preflight.py"
    body = src_path.read_text(encoding="utf-8")
    # Find the orchestrator callsite
    callsite_idx = body.find(
        "check_no_operator_facing_memo_cites_falsified_canonical_posterior_token(\n            strict=False"
    )
    assert callsite_idx > 0, (
        "Orchestrator callsite must be strict=False per CLAUDE.md "
        "'Strict-flip atomicity rule' (WARN-ONLY at landing)"
    )


# -------- Helper unit tests --------


def test_extract_date_suffix_canonical():
    """Date suffix YYYYMMDD extracted from filename."""
    p = Path("feedback_some_landed_20260529.md")
    assert _check_382_extract_date_suffix(p) == "20260529"


def test_extract_date_suffix_multiple_dates_returns_rightmost():
    """Memo with multiple 8-digit sequences returns rightmost (canonical date)."""
    p = Path("feedback_referencing_20260517_landed_20260529.md")
    assert _check_382_extract_date_suffix(p) == "20260529"


def test_extract_date_suffix_no_date_returns_none():
    """Memo without date suffix returns None."""
    p = Path("feedback_no_date_marker.md")
    assert _check_382_extract_date_suffix(p) is None


def test_extract_cited_phantom_tokens_finds_canonical_seeds():
    """Canonical phantom tokens in memo body are extracted."""
    memo_text = "This memo cites synthesis_vs_empirical_phantom_alpha_from_research_sidecar."
    cited = _check_382_extract_cited_phantom_tokens(memo_text)
    assert "synthesis_vs_empirical_phantom_alpha_from_research_sidecar" in cited


def test_extract_cited_phantom_tokens_empty_when_no_phantom():
    """No phantom tokens → empty list."""
    memo_text = "This memo discusses entirely unrelated topics."
    cited = _check_382_extract_cited_phantom_tokens(memo_text)
    assert cited == []


def test_extract_cited_phantom_tokens_case_insensitive():
    """Token matching is case-insensitive per design memo §"_normalize_token"."""
    memo_text = (
        "Memo cites Synthesis_Vs_Empirical_Phantom_Alpha_From_Research_Sidecar."
    )
    cited = _check_382_extract_cited_phantom_tokens(memo_text)
    assert "synthesis_vs_empirical_phantom_alpha_from_research_sidecar" in cited


def test_has_valid_waiver_accepts_substantive_rationale():
    """Waiver with non-placeholder rationale ≥4 chars is accepted."""
    memo_text = (
        "Cites synthesis_vs_empirical_phantom_alpha_from_research_sidecar "
        "# OPERATOR_FACING_MEMO_FALSIFIED_TOKEN_OK:historical reference for "
        "audit trail per Catalog #110/#113"
    )
    assert _check_382_has_valid_waiver(
        memo_text, "synthesis_vs_empirical_phantom_alpha_from_research_sidecar"
    )


def test_has_valid_waiver_rejects_placeholder_rationale():
    """Placeholder waivers per Catalog #287 sister are rejected."""
    memo_text = (
        "Cites synthesis_vs_empirical_phantom_alpha_from_research_sidecar "
        "# OPERATOR_FACING_MEMO_FALSIFIED_TOKEN_OK:<rationale>"
    )
    assert not _check_382_has_valid_waiver(
        memo_text, "synthesis_vs_empirical_phantom_alpha_from_research_sidecar"
    )


def test_has_valid_waiver_rejects_short_rationale():
    """Rationale <4 chars per Catalog #287 sister is rejected."""
    memo_text = (
        "Cites synthesis_vs_empirical_phantom_alpha_from_research_sidecar "
        "# OPERATOR_FACING_MEMO_FALSIFIED_TOKEN_OK:abc"
    )
    assert not _check_382_has_valid_waiver(
        memo_text, "synthesis_vs_empirical_phantom_alpha_from_research_sidecar"
    )


def test_has_valid_waiver_accepts_all_marker():
    """Generic ALL marker covers all cited tokens."""
    memo_text = (
        "Cites phantom_score_directory_naming_lie\n"
        "# OPERATOR_FACING_MEMO_FALSIFIED_TOKEN_OK:ALL Round 2 audit memo all "
        "phantom tokens are intentional per operator approval"
    )
    assert _check_382_has_valid_waiver(memo_text, "phantom_score_directory_naming_lie")


# -------- End-to-end gate behavior --------


def test_gate_no_research_dir_returns_zero(tmp_path):
    """Missing .omx/research returns 0 violations (silent)."""
    violations = check_no_operator_facing_memo_cites_falsified_canonical_posterior_token(
        strict=False, verbose=False, repo_root=tmp_path
    )
    assert violations == []


def test_gate_pre_cutoff_memo_exempt(tmp_path):
    """Pre-cutoff memo (date < 20260530) is exempt per Strict-flip atomicity."""
    research = tmp_path / ".omx" / "research"
    research.mkdir(parents=True)
    memo = research / "old_memo_20260101.md"
    memo.write_text("This memo cites synthesis_vs_empirical_phantom_alpha_from_research_sidecar.")
    violations = check_no_operator_facing_memo_cites_falsified_canonical_posterior_token(
        strict=False, verbose=False, repo_root=tmp_path
    )
    assert violations == []


def test_gate_post_cutoff_phantom_token_flagged(tmp_path):
    """Post-cutoff memo citing phantom token without waiver is flagged."""
    research = tmp_path / ".omx" / "research"
    research.mkdir(parents=True)
    memo = research / "new_memo_20260601.md"
    memo.write_text(
        "This memo cites synthesis_vs_empirical_phantom_alpha_from_research_sidecar "
        "without any waiver."
    )
    violations = check_no_operator_facing_memo_cites_falsified_canonical_posterior_token(
        strict=False, verbose=False, repo_root=tmp_path
    )
    assert len(violations) == 1
    assert "synthesis_vs_empirical_phantom_alpha_from_research_sidecar" in violations[0]


def test_gate_post_cutoff_waiver_accepted(tmp_path):
    """Post-cutoff memo with substantive waiver is accepted."""
    research = tmp_path / ".omx" / "research"
    research.mkdir(parents=True)
    memo = research / "new_memo_20260601.md"
    memo.write_text(
        "This memo cites synthesis_vs_empirical_phantom_alpha_from_research_sidecar "
        "# OPERATOR_FACING_MEMO_FALSIFIED_TOKEN_OK:historical reference per Catalog #110/#113 APPEND-ONLY discipline"
    )
    violations = check_no_operator_facing_memo_cites_falsified_canonical_posterior_token(
        strict=False, verbose=False, repo_root=tmp_path
    )
    assert violations == []


def test_gate_post_cutoff_placeholder_waiver_rejected(tmp_path):
    """Post-cutoff memo with placeholder waiver per Catalog #287 is rejected."""
    research = tmp_path / ".omx" / "research"
    research.mkdir(parents=True)
    memo = research / "new_memo_20260601.md"
    memo.write_text(
        "Cites synthesis_vs_empirical_phantom_alpha_from_research_sidecar "
        "# OPERATOR_FACING_MEMO_FALSIFIED_TOKEN_OK:<rationale>"
    )
    violations = check_no_operator_facing_memo_cites_falsified_canonical_posterior_token(
        strict=False, verbose=False, repo_root=tmp_path
    )
    assert len(violations) == 1


def test_gate_self_exempt_design_memo(tmp_path):
    """The design memo for THIS gate is self-exempt to prevent self-flagging."""
    research = tmp_path / ".omx" / "research"
    research.mkdir(parents=True)
    memo = research / "meta_meta_phantom_score_artifact_recurrence_design_20260601.md"
    memo.write_text(
        "Cites synthesis_vs_empirical_phantom_alpha_from_research_sidecar "
        "phantom_score_directory_naming_lie modal_dispatch_succeeded_but_canonical_ledger_outcome_never_registered_silent_orphan_harvest"
    )
    violations = check_no_operator_facing_memo_cites_falsified_canonical_posterior_token(
        strict=False, verbose=False, repo_root=tmp_path
    )
    assert violations == []


def test_gate_post_cutoff_no_phantom_token_accepted(tmp_path):
    """Post-cutoff memo without phantom tokens is accepted."""
    research = tmp_path / ".omx" / "research"
    research.mkdir(parents=True)
    memo = research / "new_memo_20260601.md"
    memo.write_text("This memo discusses entirely different topics.")
    violations = check_no_operator_facing_memo_cites_falsified_canonical_posterior_token(
        strict=False, verbose=False, repo_root=tmp_path
    )
    assert violations == []


def test_gate_strict_mode_raises_preflight_error(tmp_path):
    """strict=True raises PreflightError with Catalog #382 message."""
    research = tmp_path / ".omx" / "research"
    research.mkdir(parents=True)
    memo = research / "new_memo_20260601.md"
    memo.write_text(
        "Cites synthesis_vs_empirical_phantom_alpha_from_research_sidecar."
    )
    with pytest.raises(PreflightError) as exc_info:
        check_no_operator_facing_memo_cites_falsified_canonical_posterior_token(
            strict=True, verbose=False, repo_root=tmp_path
        )
    msg = str(exc_info.value)
    assert "check_no_operator_facing_memo_cites_falsified_canonical_posterior_token" in msg
    assert "violation" in msg


def test_gate_strict_silent_on_clean(tmp_path):
    """strict=True does not raise when no violations."""
    research = tmp_path / ".omx" / "research"
    research.mkdir(parents=True)
    memo = research / "clean_memo_20260601.md"
    memo.write_text("Clean memo with no phantom token citations.")
    violations = check_no_operator_facing_memo_cites_falsified_canonical_posterior_token(
        strict=True, verbose=False, repo_root=tmp_path
    )
    assert violations == []


def test_gate_multi_violation_aggregation(tmp_path):
    """Multiple memos with violations aggregate into single result."""
    research = tmp_path / ".omx" / "research"
    research.mkdir(parents=True)
    memo1 = research / "memo1_20260601.md"
    memo1.write_text("Cites synthesis_vs_empirical_phantom_alpha_from_research_sidecar.")
    memo2 = research / "memo2_20260601.md"
    memo2.write_text("Cites phantom_score_directory_naming_lie.")
    violations = check_no_operator_facing_memo_cites_falsified_canonical_posterior_token(
        strict=False, verbose=False, repo_root=tmp_path
    )
    assert len(violations) == 2


def test_gate_memo_without_date_suffix_skipped(tmp_path):
    """Memo without YYYYMMDD date suffix is skipped (no date to compare)."""
    research = tmp_path / ".omx" / "research"
    research.mkdir(parents=True)
    memo = research / "memo_no_date.md"
    memo.write_text(
        "Cites synthesis_vs_empirical_phantom_alpha_from_research_sidecar."
    )
    violations = check_no_operator_facing_memo_cites_falsified_canonical_posterior_token(
        strict=False, verbose=False, repo_root=tmp_path
    )
    assert violations == []


def test_gate_string_repo_root_accepted(tmp_path):
    """repo_root accepts both Path and str."""
    research = tmp_path / ".omx" / "research"
    research.mkdir(parents=True)
    violations = check_no_operator_facing_memo_cites_falsified_canonical_posterior_token(
        strict=False, verbose=False, repo_root=str(tmp_path)
    )
    assert violations == []


# -------- Constants tests --------


def test_canonical_seed_tokens_pinned():
    """Seed set of canonical phantom tokens is pinned for traceability."""
    assert "synthesis_vs_empirical_phantom_alpha_from_research_sidecar" in _CHECK_382_KNOWN_PHANTOM_TOKEN_SEEDS
    assert "phantom_score_directory_naming_lie" in _CHECK_382_KNOWN_PHANTOM_TOKEN_SEEDS


def test_canonical_waiver_token_pinned():
    """Canonical waiver token per Catalog #287 sister."""
    assert _CHECK_382_WAIVER_TOKEN == "# OPERATOR_FACING_MEMO_FALSIFIED_TOKEN_OK:"


def test_canonical_placeholder_rationales_pinned():
    """Placeholder rationales per Catalog #287 sister."""
    assert "<rationale>" in _CHECK_382_WAIVER_PLACEHOLDERS
    assert "<reason>" in _CHECK_382_WAIVER_PLACEHOLDERS


def test_canonical_cutoff_date_pinned():
    """Cutoff date pinned per Strict-flip atomicity."""
    assert _CHECK_382_DISCIPLINE_CUTOFF_DATE == "20260530"


# -------- Catalog #185 sister regression --------


def test_catalog_382_function_callable_via_globals():
    """Catalog #382 function is callable via module globals per Catalog #185."""
    import tac.preflight as p
    func = getattr(
        p,
        "check_no_operator_facing_memo_cites_falsified_canonical_posterior_token",
        None,
    )
    assert callable(func), "Catalog #382 function must be importable via tac.preflight"
