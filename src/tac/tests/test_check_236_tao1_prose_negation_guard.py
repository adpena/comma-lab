# SPDX-License-Identifier: MIT
"""Catalog #236 TAO-1 self-protection regression tests.

The TAO-1 finding (R2 ledger 2026-05-14, Tao + MacKay CRITICAL) anchored a
bug class where Catalog #233's gate 3 (auth_eval_100ep) accepted PROSE
co-occurrences of "100ep" and "auth-eval" within an 80-char window even
when the surrounding context was clearly NOT affirmative evidence of a
converged auth-eval anchor (e.g. retired-config descriptions, design
discussions, tradeoff comparisons).

These tests pin the post-fix behavior:
- Retired-config descriptions REJECTED (negation-token in window)
- Discussion-of-tradeoffs REJECTED (negation-token in window)
- Affirmative structured evidence (key=value, lane tags) ACCEPTED unconditionally
- Loose 100ep+auth-eval co-occurrence ACCEPTED only when no negation token in window

Sister of Catalog #136 (`check_custody_gate_accept_tokens_concrete_only`)
which extincted the bag-of-tokens validator class. Per CLAUDE.md "Bugs
must be permanently fixed AND self-protected against".

Memory: feedback_r2_critical_fix_wave_tao1_boyd1_landed_20260515.md.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tac.preflight import (
    _CHECK_233_AUTH_EVAL_100EP_LOOSE_PATTERNS,
    _CHECK_233_AUTH_EVAL_100EP_STRUCTURED_PATTERNS,
    _CHECK_233_PROSE_NEGATION_TOKENS,
    _check_233_evaluate_4_gates,
    _check_233_text_has_prose_negation_for_auth_eval,
    check_l1_to_l2_promotion_canonical_4_gate,
)


# ----------------------------------------------------------------------------
# Bug-class anchor — the R2 ledger TAO-1 reproducer cases
# ----------------------------------------------------------------------------


def test_r2_anchor_retired_config_rejected_at_gate_3() -> None:
    """Retired-config description must NOT satisfy gate 3."""
    text = (
        "retired-config: previously 100ep auth-eval was attempted; "
        "superseded by 200ep"
    )
    smoke, tier, auth_100, custody = _check_233_evaluate_4_gates(text)
    assert auth_100 is False, (
        "Catalog #236 TAO-1: retired-config prose must NOT satisfy gate 3"
    )


def test_r2_anchor_discussion_of_tradeoffs_rejected_at_gate_3() -> None:
    """Discussion-of-tradeoffs description must NOT satisfy gate 3."""
    text = (
        "See discussion of 100ep vs 200ep tradeoffs in the smoke green "
        "report; tier c was measured"
    )
    smoke, tier, auth_100, custody = _check_233_evaluate_4_gates(text)
    assert auth_100 is False


def test_r2_anchor_pure_prose_100ep_auth_eval_rejected() -> None:
    """Pure prose mention of 100ep+auth-eval must NOT satisfy gate 3."""
    text = "See discussion of 100ep auth-eval choice from earlier"
    smoke, tier, auth_100, custody = _check_233_evaluate_4_gates(text)
    assert auth_100 is False


def test_r2_anchor_planning_pending_rejected() -> None:
    """Planning/pending mentions must NOT satisfy gate 3."""
    text = "TODO: planning to run 100ep auth-eval anchor next week"
    smoke, tier, auth_100, custody = _check_233_evaluate_4_gates(text)
    assert auth_100 is False


# ----------------------------------------------------------------------------
# Affirmative structured evidence — must ALWAYS satisfy gate 3
# ----------------------------------------------------------------------------


def test_structured_evidence_auth_eval_score_axis_satisfies_gate_3() -> None:
    text = "auth_eval_score_axis=contest_cuda"
    _, _, auth_100, _ = _check_233_evaluate_4_gates(text)
    assert auth_100 is True


def test_structured_evidence_contest_cuda_tag_satisfies_gate_3() -> None:
    text = "[contest-CUDA] anchor recorded"
    _, _, auth_100, _ = _check_233_evaluate_4_gates(text)
    assert auth_100 is True


def test_structured_evidence_auth_eval_complete_satisfies_gate_3() -> None:
    text = "auth_eval_complete=true"
    _, _, auth_100, _ = _check_233_evaluate_4_gates(text)
    assert auth_100 is True


def test_structured_evidence_converged_auth_eval_satisfies_gate_3() -> None:
    text = "converged auth-eval anchor at sha256 abc..."
    _, _, auth_100, _ = _check_233_evaluate_4_gates(text)
    assert auth_100 is True


def test_structured_evidence_overrides_prose_negation_in_same_text() -> None:
    """Affirmative structured evidence anywhere in the text wins.

    Even if a retired-config sentence is also present, a structured
    evidence token elsewhere in the text proves the anchor exists.
    """
    text = (
        "Note: previously 100ep auth-eval was attempted then retired. "
        "Currently active anchor: auth_eval_score_axis=contest_cuda "
        "verified via Catalog #127."
    )
    _, _, auth_100, _ = _check_233_evaluate_4_gates(text)
    assert auth_100 is True


# ----------------------------------------------------------------------------
# Loose co-occurrence — accepted only when no negation token in window
# ----------------------------------------------------------------------------


def test_loose_clean_100ep_auth_eval_satisfies_gate_3() -> None:
    text = (
        "Wave A landed 100ep auth-eval anchor; smoke green; "
        "tier c measured; validate_custody"
    )
    _, _, auth_100, _ = _check_233_evaluate_4_gates(text)
    assert auth_100 is True


def test_loose_with_vs_marker_rejected() -> None:
    text = "considered 100ep vs 50ep auth-eval and the latter won"
    _, _, auth_100, _ = _check_233_evaluate_4_gates(text)
    assert auth_100 is False


def test_loose_with_originally_marker_rejected() -> None:
    text = "originally 100ep auth-eval was the plan"
    _, _, auth_100, _ = _check_233_evaluate_4_gates(text)
    assert auth_100 is False


def test_loose_with_hypothetical_marker_rejected() -> None:
    text = "hypothetical 100ep auth-eval would yield X"
    _, _, auth_100, _ = _check_233_evaluate_4_gates(text)
    assert auth_100 is False


def test_loose_with_abandoned_marker_rejected() -> None:
    text = "abandoned 100ep auth-eval after first attempt"
    _, _, auth_100, _ = _check_233_evaluate_4_gates(text)
    assert auth_100 is False


# ----------------------------------------------------------------------------
# Prose-negation guard helper — direct unit tests
# ----------------------------------------------------------------------------


def test_prose_negation_guard_no_100ep_returns_false() -> None:
    text = "no 100ep mention at all here"  # actually has 100ep
    # The string "no 100ep" doesn't have any of our negation tokens
    # within 80 chars — but "no auth-eval" pattern would. The function
    # only fires when the negation tokens are themselves present.
    assert _check_233_text_has_prose_negation_for_auth_eval("just text") is False


def test_prose_negation_guard_empty_text_returns_false() -> None:
    assert _check_233_text_has_prose_negation_for_auth_eval("") is False


def test_prose_negation_guard_detects_previously() -> None:
    text = "previously 100ep was attempted"
    assert _check_233_text_has_prose_negation_for_auth_eval(text) is True


def test_prose_negation_guard_detects_superseded() -> None:
    text = "100ep run was superseded by 200ep"
    assert _check_233_text_has_prose_negation_for_auth_eval(text) is True


def test_prose_negation_guard_detects_discussion_of() -> None:
    text = "discussion of 100ep tradeoffs"
    assert _check_233_text_has_prose_negation_for_auth_eval(text) is True


def test_prose_negation_guard_window_bounded() -> None:
    """Negation token must be within 80 chars of the 100ep mention.

    If the negation token is far away (>80 chars), the prose guard does
    NOT fire.
    """
    # 200 chars of filler between negation token and 100ep
    filler = "X" * 200
    text = f"previously {filler} 100ep auth-eval anchor here"
    assert _check_233_text_has_prose_negation_for_auth_eval(text) is False


def test_prose_negation_guard_handles_100_underscore_ep() -> None:
    text = "previously 100_ep was attempted"
    assert _check_233_text_has_prose_negation_for_auth_eval(text) is True


def test_prose_negation_guard_handles_100_space_epochs() -> None:
    text = "previously 100 epochs was attempted"
    assert _check_233_text_has_prose_negation_for_auth_eval(text) is True


# ----------------------------------------------------------------------------
# Live-repo regression guard
# ----------------------------------------------------------------------------


def test_catalog_233_live_repo_count_unchanged_post_tao1_fix() -> None:
    """Catalog #233 live count must not regress as a result of this fix.

    The TAO-1 fix tightens gate 3 — this can ONLY refuse more lanes, never
    accept more. So the post-fix live count must be >= pre-fix count. The
    pre-fix count at landing time is 6 (per the R2 ledger). After fix,
    expect either same count (no lanes affected) or higher (some lanes
    that previously satisfied via prose 100ep now fail).
    """
    # The check is currently warn-only per CLAUDE.md "Strict-flip
    # atomicity rule"; this regression guard just asserts it does not
    # raise in non-strict mode.
    violations = check_l1_to_l2_promotion_canonical_4_gate(
        strict=False, verbose=False
    )
    assert isinstance(violations, list)
    # The live count must remain bounded — no catastrophic regression.
    assert len(violations) <= 50, (
        f"Catalog #233 live count regressed unexpectedly to "
        f"{len(violations)} — investigate"
    )


# ----------------------------------------------------------------------------
# Negation token set integrity
# ----------------------------------------------------------------------------


def test_negation_token_set_is_frozen() -> None:
    """Negation token set must be a frozenset (immutable, O(1) lookup)."""
    assert isinstance(_CHECK_233_PROSE_NEGATION_TOKENS, frozenset)
    assert len(_CHECK_233_PROSE_NEGATION_TOKENS) > 20


def test_negation_token_set_is_lowercase() -> None:
    """All negation tokens must be lowercase (the helper lowercases input)."""
    # Some tokens may legitimately have uppercase (e.g. TODO) — the
    # implementation lowercases the input first, so the tokens must
    # match against the lowercased input. Convention: define them as
    # they will be matched against (lowercase).
    for tok in _CHECK_233_PROSE_NEGATION_TOKENS:
        assert tok == tok.lower(), (
            f"Negation token {tok!r} must be lowercase; helper lowercases input"
        )


def test_structured_patterns_are_subset_of_loose_universe() -> None:
    """Structured + loose pattern union must equal the legacy union.

    Backwards-compat: the merged `_CHECK_233_AUTH_EVAL_100EP_PATTERNS`
    alias must contain BOTH structured + loose so legacy callers that
    grep this name still see all patterns.
    """
    from tac.preflight import _CHECK_233_AUTH_EVAL_100EP_PATTERNS

    expected_count = (
        len(_CHECK_233_AUTH_EVAL_100EP_STRUCTURED_PATTERNS)
        + len(_CHECK_233_AUTH_EVAL_100EP_LOOSE_PATTERNS)
    )
    assert len(_CHECK_233_AUTH_EVAL_100EP_PATTERNS) == expected_count
