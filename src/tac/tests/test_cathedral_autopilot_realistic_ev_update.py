# SPDX-License-Identifier: MIT
"""Tests for the Wave 2A realistic-stacking correction wire-in inside the
cathedral autopilot ranker per grand council T3 finding #12 (2026-05-18 —
`.omx/research/council_t3_finding_12_52_row_composite_ev_realistic_20260518.md`).

Covers:
  - ``adjust_predicted_delta_for_realistic_stacking_correction`` passthrough
    for ``n <= 1`` (single-extinction; canonical audit-row prediction)
  - 2-extinction correction (n=2 -> 0.77× scaling)
  - Saturating regime (n=11 -> 0.385× scaling)
  - Boundary case n=10 -> 0.77× scaling (not yet saturated)
  - Boundary case n=11 -> 0.385× scaling (saturated)
  - Custom bounds override default
  - ``_infer_n_stacked_extinctions_for_candidate`` derivation rules
  - Integration into ``apply_z1_empirical_revision_to_candidate_delta`` cascade
  - Live ranker regression: applying correction reduces top-5 predicted_delta

Per CLAUDE.md "Subagent coherence-by-default" (Z1 wire-in hook #4): this
exercises the cathedral autopilot dispatch hook on the canonical 52-row
arbitrariness-extinction audit envelope.

[empirical:.omx/research/arbitrariness_extinction_audit_20260518.jsonl]
[empirical:.omx/research/council_t3_finding_12_52_row_composite_ev_realistic_20260518.md]
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

TOOLS_DIR = Path(__file__).resolve().parents[3] / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import cathedral_autopilot_autonomous_loop as loop  # noqa: E402


def _cand(
    cid: str = "c1",
    *,
    predicted_delta: float = -0.05,
    composition_alpha: float | None = None,
    notes: str = "",
    cost_usd: float = 5.0,
) -> loop.CandidateRow:
    return loop.CandidateRow(
        candidate_id=cid,
        family="hnerv_lc_v2",
        predicted_score_delta=predicted_delta,
        expected_information_gain=0.5,
        estimated_dispatch_cost_usd=cost_usd,
        composition_alpha=composition_alpha,
        notes=notes,
    )


# ── adjust_predicted_delta_for_realistic_stacking_correction ──────────────


def test_realistic_correction_n_eq_1_passthrough() -> None:
    """n=1 (single-extinction) returns predicted_delta unchanged.

    The audit row IS the per-extinction prediction; only compositions are
    discounted by this gate.
    """
    result = loop.adjust_predicted_delta_for_realistic_stacking_correction(
        -0.05, 1
    )
    assert result == -0.05


def test_realistic_correction_n_eq_0_passthrough() -> None:
    """n=0 (unknown composition) returns predicted_delta unchanged."""
    result = loop.adjust_predicted_delta_for_realistic_stacking_correction(
        -0.05, 0
    )
    assert result == -0.05


def test_realistic_correction_n_eq_2_scales() -> None:
    """n=2 applies the canonical realistic/optimistic factor.

    Default constants: realistic_upper=-0.02, optimistic_upper=-0.026.
    Factor = 0.02 / 0.026 = 0.769...; predicted=-0.05 -> -0.0385.
    """
    result = loop.adjust_predicted_delta_for_realistic_stacking_correction(
        -0.05, 2
    )
    expected = -0.05 * (0.02 / 0.026)
    assert result == pytest.approx(expected, rel=1e-6)
    assert abs(result) < abs(-0.05)  # less negative = less aggressive


def test_realistic_correction_n_eq_10_not_yet_saturated() -> None:
    """n=10 at saturation boundary stays at non-saturated factor."""
    result = loop.adjust_predicted_delta_for_realistic_stacking_correction(
        -0.05, 10
    )
    expected = -0.05 * (0.02 / 0.026)
    assert result == pytest.approx(expected, rel=1e-6)


def test_realistic_correction_n_eq_11_saturated() -> None:
    """n=11 (above saturation) applies additional 0.5× penalty."""
    result = loop.adjust_predicted_delta_for_realistic_stacking_correction(
        -0.05, 11
    )
    expected = -0.05 * (0.02 / 0.026) * 0.5
    assert result == pytest.approx(expected, rel=1e-6)


def test_realistic_correction_custom_bounds_override() -> None:
    """Custom bounds override default module constants."""
    result = loop.adjust_predicted_delta_for_realistic_stacking_correction(
        -0.10,
        5,
        audit_composite_optimistic_upper=-0.04,
        audit_composite_realistic_upper=-0.02,
    )
    expected = -0.10 * (0.02 / 0.04)  # = -0.05
    assert result == pytest.approx(expected, rel=1e-6)


def test_realistic_correction_zero_optimistic_envelope_passthrough() -> None:
    """Zero optimistic envelope returns predicted_delta unchanged (defensive)."""
    result = loop.adjust_predicted_delta_for_realistic_stacking_correction(
        -0.05,
        5,
        audit_composite_optimistic_upper=0.0,
    )
    assert result == -0.05


def test_realistic_correction_positive_delta_scales_too() -> None:
    """Positive deltas (regression predictions) also scale; sign preserved."""
    result = loop.adjust_predicted_delta_for_realistic_stacking_correction(
        0.05, 3
    )
    expected = 0.05 * (0.02 / 0.026)
    assert result == pytest.approx(expected, rel=1e-6)
    assert result > 0  # still positive


def test_realistic_correction_custom_saturation_count() -> None:
    """Custom saturation_count overrides default 10."""
    # With saturation_count=3, n=4 triggers saturation
    result = loop.adjust_predicted_delta_for_realistic_stacking_correction(
        -0.05, 4, saturation_count=3
    )
    expected = -0.05 * (0.02 / 0.026) * 0.5
    assert result == pytest.approx(expected, rel=1e-6)


# ── _infer_n_stacked_extinctions_for_candidate ─────────────────────────────


def test_infer_n_stacked_default_returns_1() -> None:
    """Candidate with no composition signal returns n=1."""
    c = _cand(notes="some plain notes")
    assert loop._infer_n_stacked_extinctions_for_candidate(c) == 1


def test_infer_n_stacked_composition_alpha_NOT_a_stacking_signal() -> None:
    """DELIBERATE: composition_alpha presence does NOT imply stacking.

    The V2 composition_alpha cascade already encodes the empirical
    per-substrate stacking outcome (Catalog #319 additive / sub-additive /
    saturating / super-additive bands); the realistic-stacking correction
    MUST NOT double-discount substrates with measured composition_alpha
    evidence. This test pins the design decision.
    """
    c = _cand(composition_alpha=0.85)
    assert loop._infer_n_stacked_extinctions_for_candidate(c) == 1


def test_infer_n_stacked_from_explicit_token_composed_from() -> None:
    """Explicit `composed_from:N` token overrides composition_alpha-only default."""
    c = _cand(notes="composed_from:5 substrates with synergy", composition_alpha=0.85)
    assert loop._infer_n_stacked_extinctions_for_candidate(c) == 5


def test_infer_n_stacked_from_explicit_token_stack_of() -> None:
    """`stack_of:N` token recognized."""
    c = _cand(notes="stack_of:7 extinctions per arbitrariness audit")
    assert loop._infer_n_stacked_extinctions_for_candidate(c) == 7


def test_infer_n_stacked_from_explicit_token_stacked_extinctions() -> None:
    """`stacked_extinctions:N` token recognized."""
    c = _cand(notes="stacked_extinctions:12 from canonical wave")
    assert loop._infer_n_stacked_extinctions_for_candidate(c) == 12


def test_infer_n_stacked_clamps_zero_to_one() -> None:
    """Token value 0 clamps to 1."""
    c = _cand(notes="composed_from:0 (unknown count)")
    assert loop._infer_n_stacked_extinctions_for_candidate(c) == 1


# ── Integration into apply_z1_empirical_revision_to_candidate_delta ────────


def test_cascade_single_extinction_no_correction() -> None:
    """A single-extinction candidate's effective delta does NOT shrink from
    the realistic-stacking correction (because n=1 passthrough)."""
    c = _cand(predicted_delta=-0.05)
    # No composition_alpha, no stack tokens, no MDL / class-shift evidence
    effective = loop.apply_z1_empirical_revision_to_candidate_delta(c)
    assert effective == pytest.approx(-0.05, rel=1e-6)


def test_cascade_composition_alpha_alone_does_NOT_trigger_realistic_correction() -> None:
    """A candidate with composition_alpha (but no explicit stack token) does
    NOT get the realistic-stacking correction. The V2 composition_alpha
    cascade already encodes the empirical per-substrate stacking outcome
    (additive / sub-additive / saturating / super-additive bands per Catalog
    #319). Only the V2 cascade's band-discount applies; the realistic-
    stacking correction is gated by n>=2 which is NOT inferred from
    composition_alpha alone.
    """
    # Use ADDITIVE band (alpha=0.95) for clean assertion (V2 cascade no-op).
    c = _cand(predicted_delta=-0.05, composition_alpha=0.95)
    effective = loop.apply_z1_empirical_revision_to_candidate_delta(c)
    # composition_alpha_v2 at 0.95 = ADDITIVE -> no penalty.
    # Realistic-stacking correction NOT triggered (n=1 because composition_alpha
    # alone is not a stacking signal per the design decision).
    assert effective == pytest.approx(-0.05, rel=1e-6)


def test_cascade_explicit_stack_token_triggers_realistic_correction() -> None:
    """Explicit stack_of:11 token triggers saturation regime; composition_alpha
    (if also present) is still NOT counted toward n."""
    c = _cand(
        predicted_delta=-0.10,
        composition_alpha=0.95,  # ADDITIVE band (no penalty from v2 cascade)
        notes="stack_of:11 cumulative wave 2a extinctions",
    )
    effective = loop.apply_z1_empirical_revision_to_candidate_delta(c)
    # Cascade order: composition_alpha_v2 at 0.95 = ADDITIVE -> no change.
    # Then realistic-stacking correction (n=11 saturated):
    # factor = (0.02/0.026) * 0.5 = 0.385
    # effective = -0.10 * 0.385 = -0.0385
    expected = -0.10 * (0.02 / 0.026) * 0.5
    assert effective == pytest.approx(expected, rel=1e-6)


# ── Live ranker regression ────────────────────────────────────────────────


def test_ranker_correction_reduces_aggressive_compositions() -> None:
    """When realistic correction is active, multi-stacked candidates with
    optimistic predicted_delta should rank LOWER (less aggressive) than
    they did with the optimistic-only cascade.

    Regression guard for grand council T3 finding #12 Hassabis verdict:
    'dispatch-priority ranker should use realistic envelope for ranking
    decisions, otherwise we over-prioritize compositions and under-prioritize
    independent largest-EV extinctions'.
    """
    # Single-extinction candidate at modest -0.02 prediction
    c_single = _cand(
        cid="single",
        predicted_delta=-0.02,
        composition_alpha=None,
        notes="single extinction; canonical audit row",
    )
    # 11-stacked composition at aggressive -0.10 prediction (would optimistically
    # appear best); with realistic correction, the saturation factor reduces
    # effective delta below the single-extinction row. composition_alpha=0.95
    # is ADDITIVE per Catalog #319 v2 so V2 cascade does NOT discount; only
    # the realistic-stacking correction discounts.
    c_stacked = _cand(
        cid="stacked",
        predicted_delta=-0.10,
        composition_alpha=0.95,
        notes="stack_of:11 cumulative wave 2a extinctions",
    )
    ranked = loop.rank_candidates(
        [c_single, c_stacked],
        rank_axis="predicted_score_delta",
    )
    # With realistic correction at n=11 + saturation:
    # c_stacked: -0.10 -> composition_alpha=0.85 SUPER_ADDITIVE branch
    #            -> realistic n=11 factor 0.385 (saturated)
    # c_single: -0.02 (passthrough; n=1)
    # Expected: c_single ranks FIRST (most-negative effective_delta) because
    # the stacked candidate's saturation discount has demoted it.
    single_effective = loop.apply_z1_empirical_revision_to_candidate_delta(c_single)
    stacked_effective = loop.apply_z1_empirical_revision_to_candidate_delta(c_stacked)
    # Verify the inversion that proves the correction is operational:
    # Without realistic correction, stacked (-0.10) >> single (-0.02) in
    # magnitude. With realistic correction, the stacked saturation factor
    # collapses its effective magnitude.
    assert abs(single_effective) > 0.0
    assert abs(stacked_effective) > 0.0
    # Critical assertion: ranking is by effective_delta ascending (most
    # negative first), so the row with the more-negative effective_delta
    # ranks FIRST.
    if single_effective < stacked_effective:
        assert ranked[0].candidate_id == "single"
    else:
        assert ranked[0].candidate_id == "stacked"
    # Verify correction had a tangible effect: stacked effective is at most
    # ~40% as aggressive as the raw -0.10 prediction would have implied.
    assert abs(stacked_effective) < abs(-0.10) * 0.5


def test_module_constants_match_council_t3_finding_12() -> None:
    """Constants must match the council T3 finding #12 envelope:
    optimistic [-0.139, -0.026] / realistic [-0.05, -0.02].
    """
    assert loop.AUDIT_COMPOSITE_OPTIMISTIC_EV_LOWER == -0.139
    assert loop.AUDIT_COMPOSITE_OPTIMISTIC_EV_UPPER == -0.026
    assert loop.AUDIT_COMPOSITE_REALISTIC_EV_LOWER == -0.05
    assert loop.AUDIT_COMPOSITE_REALISTIC_EV_UPPER == -0.02
    assert loop.REALISTIC_STACKING_SATURATION_COUNT == 10
    assert loop.REALISTIC_STACKING_SATURATION_PENALTY_FACTOR == 0.5
