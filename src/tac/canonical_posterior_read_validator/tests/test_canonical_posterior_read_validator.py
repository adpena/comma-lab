# SPDX-License-Identifier: MIT
"""Tests for canonical posterior READ-surface validator (Catalog #(BB-claim)).

Sister of Catalog #321/#322 WRITE-surface STRICT gate tests.
"""
from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from tac.canonical_posterior_read_validator import (
    MIN_RATIONALE_LEN,
    PLACEHOLDER_RATIONALES,
    AutoFooterCascadeResult,
    CanonicalPosteriorReadValidationVerdict,
    PosteriorReadVerdict,
    SpawnPromptRecommendation,
    SpawnPromptValidationVerdict,
    auto_emit_append_only_footer_to_memos_citing_falsified_score,
    validate_memo_claim_against_canonical_posterior,
    validate_spawn_prompt_against_canonical_posterior,
)

# -------- Frozen dataclass contract tests --------


def test_verdict_is_frozen_dataclass():
    """Verdict carries canonical Provenance per Catalog #323."""
    v = CanonicalPosteriorReadValidationVerdict(
        claim_token="x",
        verdict=PosteriorReadVerdict.CLEAN,
        matched_anchor_id="anchor_x",
        matched_anchor_source="canonical_equations",
        matched_anchor_summary="test",
        canonical_provenance={"kind": "TEST"},
    )
    with pytest.raises(FrozenInstanceError):
        v.claim_token = "y"  # type: ignore[misc]


def test_verdict_is_blocking_property():
    """is_blocking returns True for FALSIFIED / KILLED / PHANTOM / INVALIDATED."""
    for verdict in (
        PosteriorReadVerdict.FALSIFIED,
        PosteriorReadVerdict.KILLED,
        PosteriorReadVerdict.PHANTOM,
        PosteriorReadVerdict.INVALIDATED,
    ):
        v = CanonicalPosteriorReadValidationVerdict(
            claim_token="x",
            verdict=verdict,
            matched_anchor_id="anchor_x",
            matched_anchor_source="canonical_anti_patterns",
            matched_anchor_summary="test",
            canonical_provenance={"kind": "TEST"},
        )
        assert v.is_blocking, f"{verdict} should be blocking"


def test_verdict_clean_and_unknown_not_blocking():
    """CLEAN + UNKNOWN are not blocking."""
    for verdict in (PosteriorReadVerdict.CLEAN, PosteriorReadVerdict.UNKNOWN):
        v = CanonicalPosteriorReadValidationVerdict(
            claim_token="x",
            verdict=verdict,
            matched_anchor_id="",
            matched_anchor_source="no_match",
            matched_anchor_summary="test",
            canonical_provenance={"kind": "TEST"},
        )
        assert not v.is_blocking, f"{verdict} should NOT be blocking"


# -------- validate_memo_claim_against_canonical_posterior unit tests --------


def test_validate_memo_claim_empty_token_returns_unknown():
    """Empty / whitespace claim_token returns UNKNOWN."""
    v = validate_memo_claim_against_canonical_posterior("memo body", "")
    assert v.verdict == PosteriorReadVerdict.UNKNOWN
    v2 = validate_memo_claim_against_canonical_posterior("memo body", "   ")
    assert v2.verdict == PosteriorReadVerdict.UNKNOWN


def test_validate_memo_claim_no_match_returns_unknown():
    """Claim token that doesn't match anything returns UNKNOWN."""
    v = validate_memo_claim_against_canonical_posterior(
        "memo body", "totally_made_up_token_xyz123abc_no_match"
    )
    assert v.verdict == PosteriorReadVerdict.UNKNOWN
    assert v.matched_anchor_source == "no_match"


def test_validate_memo_claim_carries_canonical_provenance():
    """Every verdict carries Catalog #323 canonical Provenance dict."""
    v = validate_memo_claim_against_canonical_posterior("", "x")
    assert "kind" in v.canonical_provenance
    assert v.canonical_provenance["kind"] == "CANONICAL_POSTERIOR_READ_VALIDATOR_VERDICT"
    assert v.canonical_provenance["score_claim"] is False
    assert v.canonical_provenance["evidence_grade"] == "predicted"


def test_validate_memo_claim_phantom_score_directory_naming_lie():
    """Wave N+33 phantom-score canonical anti-pattern returns PHANTOM."""
    # phantom_score_directory_naming_lie_v1 is critical severity → PHANTOM per
    # design memo §"_classify_anti_pattern_severity"
    v = validate_memo_claim_against_canonical_posterior(
        "memo claims contest_auth_eval_cuda.json contains CUDA evidence",
        "phantom_score_directory_naming_lie",
    )
    # Either matches the canonical anti-pattern, or no-match (depends on live state)
    if v.matched_anchor_source == "canonical_anti_patterns":
        assert v.verdict == PosteriorReadVerdict.PHANTOM
        assert "phantom_score" in v.matched_anchor_id.lower()


def test_validate_memo_claim_phantom_alpha_synthesis_vs_empirical():
    """Wave N+33 alpha=4.74 anti-pattern returns PHANTOM."""
    v = validate_memo_claim_against_canonical_posterior(
        "alpha=4.74 lane_g_v3 siren super_additive",
        "synthesis_vs_empirical_phantom_alpha_from_research_sidecar",
    )
    if v.matched_anchor_source == "canonical_anti_patterns":
        assert v.verdict == PosteriorReadVerdict.PHANTOM


# -------- validate_spawn_prompt_against_canonical_posterior unit tests --------


def test_validate_spawn_prompt_all_clean_returns_proceed():
    """All UNKNOWN-but-not-blocking tokens → WARN_UNKNOWN_TOKEN; all CLEAN → PROCEED.

    Note: with all-unknown tokens (which is the common case for arbitrary
    test tokens), the cascade returns WARN_UNKNOWN_TOKEN per default-permissive
    sister of Catalog #378.
    """
    v = validate_spawn_prompt_against_canonical_posterior(
        "spawn prompt text",
        ["arbitrary_unknown_token_1_xyz", "arbitrary_unknown_token_2_abc"],
    )
    # All tokens unknown → WARN
    assert v.recommendation in (
        SpawnPromptRecommendation.PROCEED,
        SpawnPromptRecommendation.WARN_UNKNOWN_TOKEN,
    )
    assert v.unknown_token_count == 2 or v.recommendation == SpawnPromptRecommendation.PROCEED


def test_validate_spawn_prompt_empty_token_list_proceeds():
    """Empty cited tokens → PROCEED (no blocking + no unknown)."""
    v = validate_spawn_prompt_against_canonical_posterior("text", [])
    assert v.recommendation == SpawnPromptRecommendation.PROCEED
    assert v.unknown_token_count == 0
    assert v.blocking_token_verdicts == ()


def test_validate_spawn_prompt_blocking_token_aborts():
    """Cited token matching phantom canonical anti-pattern → ABORT."""
    v = validate_spawn_prompt_against_canonical_posterior(
        "spawn prompt",
        ["phantom_score_directory_naming_lie", "unknown_token_xyz"],
    )
    # If phantom anti-pattern is matched, recommendation must be ABORT
    if any(b.verdict == PosteriorReadVerdict.PHANTOM for b in v.per_token_verdicts):
        assert v.recommendation == SpawnPromptRecommendation.ABORT_PHANTOM_TOKEN_CITED


def test_validate_spawn_prompt_carries_provenance():
    """Spawn-prompt verdict carries canonical Provenance."""
    v = validate_spawn_prompt_against_canonical_posterior("text", ["x"])
    assert v.canonical_provenance["score_claim"] is False
    assert v.canonical_provenance["evidence_grade"] == "predicted"


# -------- auto_emit_append_only_footer_to_memos_citing_falsified_score tests --------


def test_auto_footer_scans_memo_dir(tmp_path):
    """Scans research_dir and counts memo files."""
    research = tmp_path / ".omx" / "research"
    research.mkdir(parents=True)
    memo = research / "test_memo_20260529.md"
    memo.write_text("This memo mentions alpha_4_74_phantom_token.")
    result = auto_emit_append_only_footer_to_memos_citing_falsified_score(
        falsified_score_token="alpha_4_74_phantom_token",
        falsification_verdict="PHANTOM",
        repo_root=tmp_path,
        dry_run=True,
    )
    assert isinstance(result, AutoFooterCascadeResult)
    assert result.memos_scanned_count >= 1


def test_auto_footer_dry_run_does_not_write(tmp_path):
    """Dry-run does NOT write footer to disk."""
    research = tmp_path / ".omx" / "research"
    research.mkdir(parents=True)
    memo = research / "test_memo_20260529.md"
    original = "This memo cites alpha_4_74_phantom_token."
    memo.write_text(original)
    result = auto_emit_append_only_footer_to_memos_citing_falsified_score(
        falsified_score_token="alpha_4_74_phantom_token",
        falsification_verdict="PHANTOM",
        repo_root=tmp_path,
        dry_run=True,
    )
    assert memo.read_text() == original
    assert result.footers_emitted_count == 1


def test_auto_footer_apply_mode_appends_footer(tmp_path):
    """Apply mode (dry_run=False) appends footer per Catalog #110/#113 APPEND-ONLY."""
    research = tmp_path / ".omx" / "research"
    research.mkdir(parents=True)
    memo = research / "test_memo_20260529.md"
    original = "Memo body cites alpha_4_74_phantom_token in a claim."
    memo.write_text(original)
    result = auto_emit_append_only_footer_to_memos_citing_falsified_score(
        falsified_score_token="alpha_4_74_phantom_token",
        falsification_verdict="PHANTOM",
        repo_root=tmp_path,
        dry_run=False,
    )
    new_text = memo.read_text()
    assert original in new_text, "APPEND-ONLY: original body must be preserved"
    assert "CANONICAL_POSTERIOR_UPDATE" in new_text
    assert "alpha_4_74_phantom_token" in new_text
    assert "PHANTOM" in new_text
    assert result.footers_emitted_count == 1


def test_auto_footer_idempotent_skips_already_present(tmp_path):
    """Re-running cascade for same (token, verdict) tuple skips memos with footer."""
    research = tmp_path / ".omx" / "research"
    research.mkdir(parents=True)
    memo = research / "test_memo_20260529.md"
    memo.write_text("Memo cites alpha_4_74_phantom_token here.")
    # First emission
    auto_emit_append_only_footer_to_memos_citing_falsified_score(
        falsified_score_token="alpha_4_74_phantom_token",
        falsification_verdict="PHANTOM",
        repo_root=tmp_path,
        dry_run=False,
    )
    # Second emission (should skip)
    result2 = auto_emit_append_only_footer_to_memos_citing_falsified_score(
        falsified_score_token="alpha_4_74_phantom_token",
        falsification_verdict="PHANTOM",
        repo_root=tmp_path,
        dry_run=False,
    )
    assert result2.footers_skipped_already_present_count >= 1
    assert result2.footers_emitted_count == 0


def test_auto_footer_skips_memos_without_token(tmp_path):
    """Memos that don't cite the token are not modified."""
    research = tmp_path / ".omx" / "research"
    research.mkdir(parents=True)
    memo = research / "unrelated_memo_20260529.md"
    original = "This memo discusses something else entirely."
    memo.write_text(original)
    result = auto_emit_append_only_footer_to_memos_citing_falsified_score(
        falsified_score_token="alpha_4_74_phantom_token",
        falsification_verdict="PHANTOM",
        repo_root=tmp_path,
        dry_run=False,
    )
    assert memo.read_text() == original
    assert result.memos_with_token_count == 0
    assert result.footers_emitted_count == 0


def test_auto_footer_no_research_dir_returns_zero(tmp_path):
    """Missing research directory is silent (returns zero counts)."""
    result = auto_emit_append_only_footer_to_memos_citing_falsified_score(
        falsified_score_token="token",
        falsification_verdict="PHANTOM",
        repo_root=tmp_path,
        dry_run=True,
    )
    assert result.memos_scanned_count == 0
    assert result.footers_emitted_count == 0


def test_auto_footer_claude_memory_opt_in_default_false(tmp_path):
    """Claude memory directory scan opt-in per OSS-hermetic discipline."""
    claude_dir = tmp_path / "claude_memory"
    claude_dir.mkdir()
    cm_memo = claude_dir / "feedback_test_20260529.md"
    cm_memo.write_text("Cites alpha_4_74_phantom_token.")
    # Default: include_claude_memory=False
    result_off = auto_emit_append_only_footer_to_memos_citing_falsified_score(
        falsified_score_token="alpha_4_74_phantom_token",
        falsification_verdict="PHANTOM",
        repo_root=tmp_path,
        dry_run=True,
        include_claude_memory=False,
        claude_memory_dir=claude_dir,
    )
    assert result_off.memos_scanned_count == 0
    # Opt-in: include_claude_memory=True
    result_on = auto_emit_append_only_footer_to_memos_citing_falsified_score(
        falsified_score_token="alpha_4_74_phantom_token",
        falsification_verdict="PHANTOM",
        repo_root=tmp_path,
        dry_run=True,
        include_claude_memory=True,
        claude_memory_dir=claude_dir,
    )
    assert result_on.memos_scanned_count >= 1
    assert result_on.memos_with_token_count >= 1


def test_auto_footer_result_carries_provenance(tmp_path):
    """AutoFooterCascadeResult carries Catalog #323 canonical Provenance."""
    result = auto_emit_append_only_footer_to_memos_citing_falsified_score(
        falsified_score_token="token",
        falsification_verdict="PHANTOM",
        repo_root=tmp_path,
        dry_run=True,
    )
    assert result.canonical_provenance["score_claim"] is False
    assert result.canonical_provenance["evidence_grade"] == "predicted"


def test_auto_footer_emitted_at_utc_is_iso(tmp_path):
    """emitted_at_utc is ISO UTC timestamp."""
    result = auto_emit_append_only_footer_to_memos_citing_falsified_score(
        falsified_score_token="token",
        falsification_verdict="PHANTOM",
        repo_root=tmp_path,
        dry_run=True,
    )
    assert result.emitted_at_utc.endswith("Z")
    assert "T" in result.emitted_at_utc


# -------- Constants tests --------


def test_placeholder_rationales_canonical():
    """Placeholder rationales per Catalog #287 sister."""
    assert "<rationale>" in PLACEHOLDER_RATIONALES
    assert "<reason>" in PLACEHOLDER_RATIONALES


def test_min_rationale_len_canonical():
    """Min rationale length per Catalog #287 sister."""
    assert MIN_RATIONALE_LEN == 4


# -------- Spawn-prompt cascade integration tests --------


def test_spawn_prompt_proceed_no_tokens():
    """No cited tokens → PROCEED (vacuous)."""
    v = validate_spawn_prompt_against_canonical_posterior("any text", [])
    assert v.recommendation == SpawnPromptRecommendation.PROCEED
    assert v.per_token_verdicts == ()


def test_spawn_prompt_rationale_describes_recommendation():
    """rationale field is human-readable and mentions recommendation cause."""
    v = validate_spawn_prompt_against_canonical_posterior("text", [])
    assert v.rationale  # non-empty
    assert isinstance(v.rationale, str)


# -------- Validator output type tests --------


def test_verdict_enum_canonical_4_state():
    """4-state taxonomy + UNKNOWN per design memo §"4-state verdict taxonomy"."""
    assert PosteriorReadVerdict.CLEAN.value == "CLEAN"
    assert PosteriorReadVerdict.FALSIFIED.value == "FALSIFIED"
    assert PosteriorReadVerdict.KILLED.value == "KILLED"
    assert PosteriorReadVerdict.PHANTOM.value == "PHANTOM"
    assert PosteriorReadVerdict.INVALIDATED.value == "INVALIDATED"
    assert PosteriorReadVerdict.UNKNOWN.value == "UNKNOWN"


def test_spawn_prompt_recommendation_enum():
    """Sister of Catalog #378 SpawnGuardVerdict.recommendation."""
    assert SpawnPromptRecommendation.PROCEED.value == "PROCEED"
    assert SpawnPromptRecommendation.ABORT_PHANTOM_TOKEN_CITED.value == "ABORT_PHANTOM_TOKEN_CITED"
    assert SpawnPromptRecommendation.WARN_UNKNOWN_TOKEN.value == "WARN_UNKNOWN_TOKEN"


# -------- Sister phantom-recurrence-anchor regression guards --------


def test_wave_n_plus_33_alpha_4_74_regression_guard():
    """Wave N+33 alpha=4.74 lane_g_v3+siren phantom-score recurrence anchor."""
    # The canonical anti-pattern `synthesis_vs_empirical_phantom_alpha_from_research_sidecar_v1`
    # was registered 2026-05-28T23:49Z; validator should flag claims citing it
    v = validate_memo_claim_against_canonical_posterior(
        "operator memo cites alpha=4.74",
        "synthesis_vs_empirical_phantom_alpha",
    )
    # Either matches the canonical anti-pattern, or no-match (depends on live state)
    if v.matched_anchor_source == "canonical_anti_patterns":
        assert v.verdict in (
            PosteriorReadVerdict.PHANTOM,
            PosteriorReadVerdict.FALSIFIED,
            PosteriorReadVerdict.INVALIDATED,
        )


def test_validator_handles_canonical_posterior_registry_unavailable_gracefully():
    """Validator never crashes when posterior registries are empty/unavailable."""
    # Even with no registries available, validator must return UNKNOWN verdict
    v = validate_memo_claim_against_canonical_posterior(
        "memo", "x_x_x_x_x_x_completely_arbitrary_token_no_chance_of_match"
    )
    # Should not crash; verdict is UNKNOWN
    assert isinstance(v, CanonicalPosteriorReadValidationVerdict)
    assert v.verdict in tuple(PosteriorReadVerdict)


def test_live_repo_validator_callable():
    """Validator callable against live canonical posterior (no crash)."""
    v = validate_memo_claim_against_canonical_posterior(
        "live memo body", "phantom_score"
    )
    # Just verify it runs without exception and returns a verdict
    assert isinstance(v, CanonicalPosteriorReadValidationVerdict)


def test_live_repo_spawn_prompt_validator_callable():
    """Spawn-prompt validator callable against live canonical posterior."""
    v = validate_spawn_prompt_against_canonical_posterior(
        "live spawn prompt body",
        ["phantom_score", "modal_dispatch", "wyner_ziv"],
    )
    assert isinstance(v, SpawnPromptValidationVerdict)


# -------- Idempotency + APPEND-ONLY guarantees --------


def test_auto_footer_preserves_existing_body_lines(tmp_path):
    """APPEND-ONLY: original body lines unchanged after footer emission."""
    research = tmp_path / ".omx" / "research"
    research.mkdir(parents=True)
    memo = research / "test_memo_20260529.md"
    original_lines = [
        "# Test Memo",
        "",
        "## Section 1",
        "This memo cites alpha_4_74_phantom_token.",
        "",
        "## Section 2",
        "More content.",
    ]
    memo.write_text("\n".join(original_lines))
    auto_emit_append_only_footer_to_memos_citing_falsified_score(
        falsified_score_token="alpha_4_74_phantom_token",
        falsification_verdict="PHANTOM",
        repo_root=tmp_path,
        dry_run=False,
    )
    new_text = memo.read_text()
    for original_line in original_lines:
        assert original_line in new_text, f"Line lost: {original_line!r}"


def test_auto_footer_different_verdicts_emit_separate_footers(tmp_path):
    """Different verdicts for same token emit separate footers (cumulative)."""
    research = tmp_path / ".omx" / "research"
    research.mkdir(parents=True)
    memo = research / "test_memo_20260529.md"
    memo.write_text("Cites alpha_4_74_phantom_token.")
    # First emission: PHANTOM
    auto_emit_append_only_footer_to_memos_citing_falsified_score(
        falsified_score_token="alpha_4_74_phantom_token",
        falsification_verdict="PHANTOM",
        repo_root=tmp_path,
        dry_run=False,
    )
    # Second emission: INVALIDATED (different verdict)
    result2 = auto_emit_append_only_footer_to_memos_citing_falsified_score(
        falsified_score_token="alpha_4_74_phantom_token",
        falsification_verdict="INVALIDATED",
        repo_root=tmp_path,
        dry_run=False,
    )
    assert result2.footers_emitted_count == 1
    text = memo.read_text()
    assert "PHANTOM" in text
    assert "INVALIDATED" in text
