# SPDX-License-Identifier: MIT
"""Tests for the Z1 empirical revision wire-in inside the cathedral
autopilot ranker (2026-05-14 — `feedback_z1_mdl_ablation_landed_20260514.md`
operator decision #4).

Covers:
  - ``adjust_predicted_delta_for_mdl_density`` (within-class trap penalty)
  - ``adjust_predicted_delta_for_class_shift`` (class-shift reward)
  - ``apply_z1_empirical_revision_to_candidate_delta`` (composition)
  - ``rank_candidates`` with Z1 revision: within-class ranked lower
  - ``rank_candidates`` with Z1 revision: class-shift ranked higher
  - Z1 revision can be disabled via ``apply_z1_empirical_revision=False``
  - ``load_candidates_from_jsonl`` reads new Z1 fields
  - A1-baseline regression: density 0.99 floors the predicted delta
  - C6 MDL-IBPS / time-traveler-style literature anchor double-reward
  - MDL-unknown candidates not penalized (lack-of-evidence != negative signal)
"""
from __future__ import annotations

import json
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
    family: str = "hnerv_lc_v2",
    predicted_delta: float = -0.005,
    eig: float = 0.5,
    cost_usd: float = 5.0,
    blockers: list[str] | None = None,
    mdl_density: float | None = None,
    mdl_tier_c_density: float | None = None,
    lane_class: str | None = None,
    literature_anchor: str = "",
    composition_alpha: float | None = None,
    notes: str = "",
) -> loop.CandidateRow:
    return loop.CandidateRow(
        candidate_id=cid,
        family=family,
        predicted_score_delta=predicted_delta,
        expected_information_gain=eig,
        estimated_dispatch_cost_usd=cost_usd,
        blockers=list(blockers or []),
        notes=notes,
        mdl_density=mdl_density,
        mdl_tier_c_density=mdl_tier_c_density,
        lane_class=lane_class,
        literature_anchor=literature_anchor,
        composition_alpha=composition_alpha,
    )


# ── adjust_predicted_delta_for_mdl_density ─────────────────────────────────


def test_mdl_density_above_saturated_floors_delta() -> None:
    # density 0.99 -> floor at MDL_DENSITY_WITHIN_CLASS_SATURATED_DELTA_FLOOR
    result = loop.adjust_predicted_delta_for_mdl_density(-0.030, 0.99)
    assert result == loop.MDL_DENSITY_WITHIN_CLASS_SATURATED_DELTA_FLOOR


def test_mdl_density_above_trending_halves_delta() -> None:
    # density 0.92 -> 50% penalty: -0.02 -> -0.01
    result = loop.adjust_predicted_delta_for_mdl_density(-0.02, 0.92)
    assert result == pytest.approx(-0.01)


def test_mdl_density_below_threshold_no_penalty() -> None:
    # density 0.50 -> no penalty: -0.04 -> -0.04
    result = loop.adjust_predicted_delta_for_mdl_density(-0.04, 0.50)
    assert result == pytest.approx(-0.04)


def test_mdl_density_unknown_no_penalty() -> None:
    # None -> no penalty
    result = loop.adjust_predicted_delta_for_mdl_density(-0.04, None)
    assert result == pytest.approx(-0.04)


def test_mdl_density_invalid_no_penalty() -> None:
    # Non-numeric -> treated as unknown
    result = loop.adjust_predicted_delta_for_mdl_density(-0.04, "nan")  # type: ignore[arg-type]
    assert result == pytest.approx(-0.04)


def test_mdl_density_exactly_at_threshold_no_penalty() -> None:
    # 0.90 is NOT > 0.90; gate is strict > so no penalty
    result = loop.adjust_predicted_delta_for_mdl_density(-0.04, 0.90)
    assert result == pytest.approx(-0.04)


def test_mdl_density_saturated_positive_delta_preserved() -> None:
    # A positive (worse) delta should not be improved by the floor cap.
    # Floor is max(base, FLOOR); +0.05 > -0.005 so result is +0.05.
    result = loop.adjust_predicted_delta_for_mdl_density(+0.05, 0.99)
    assert result == pytest.approx(+0.05)


# ── adjust_predicted_delta_for_class_shift ─────────────────────────────────


def test_class_shift_lane_class_substrate_class_shift_rewards() -> None:
    result = loop.adjust_predicted_delta_for_class_shift(
        -0.005, lane_class="substrate_class_shift"
    )
    assert result == pytest.approx(-0.005 - loop.CLASS_SHIFT_LANE_CLASS_REWARD)


def test_class_shift_lane_class_predictive_receiver_rewards() -> None:
    result = loop.adjust_predicted_delta_for_class_shift(
        -0.005, lane_class="predictive_receiver_v1"
    )
    assert result == pytest.approx(-0.005 - loop.CLASS_SHIFT_LANE_CLASS_REWARD)


def test_class_shift_literature_anchor_cooperative_receiver_rewards() -> None:
    result = loop.adjust_predicted_delta_for_class_shift(
        -0.005, literature_anchor="cooperative-receiver / Wyner-Ziv"
    )
    assert result == pytest.approx(
        -0.005 - loop.CLASS_SHIFT_LITERATURE_ANCHOR_REWARD
    )


def test_class_shift_literature_anchor_in_notes_rewards() -> None:
    # If literature_anchor is empty but notes mention it, still reward.
    result = loop.adjust_predicted_delta_for_class_shift(
        -0.005,
        literature_anchor="",
        notes="Tishby-Zaslavsky Information Bottleneck routing",
    )
    assert result == pytest.approx(
        -0.005 - loop.CLASS_SHIFT_LITERATURE_ANCHOR_REWARD
    )


def test_class_shift_both_lane_class_and_literature_anchor_stack() -> None:
    # Both rewards stack independently.
    result = loop.adjust_predicted_delta_for_class_shift(
        -0.005,
        lane_class="substrate_class_shift",
        literature_anchor="cooperative-receiver",
    )
    expected = (
        -0.005
        - loop.CLASS_SHIFT_LANE_CLASS_REWARD
        - loop.CLASS_SHIFT_LITERATURE_ANCHOR_REWARD
    )
    assert result == pytest.approx(expected)


def test_class_shift_no_match_no_change() -> None:
    result = loop.adjust_predicted_delta_for_class_shift(
        -0.005, lane_class="hnerv_lc_v2"
    )
    assert result == pytest.approx(-0.005)


def test_class_shift_empty_inputs_no_change() -> None:
    result = loop.adjust_predicted_delta_for_class_shift(-0.005)
    assert result == pytest.approx(-0.005)


def test_class_shift_time_traveler_token_rewards() -> None:
    # time_traveler is a canonical class-shift substrate per the memo.
    result = loop.adjust_predicted_delta_for_class_shift(
        -0.040, literature_anchor="time_traveler_l5_packet"
    )
    assert result == pytest.approx(
        -0.040 - loop.CLASS_SHIFT_LITERATURE_ANCHOR_REWARD
    )


def test_class_shift_balle_2018_literature_anchor_rewards() -> None:
    result = loop.adjust_predicted_delta_for_class_shift(
        -0.005, literature_anchor="balle_2018"
    )
    assert result == pytest.approx(
        -0.005 - loop.CLASS_SHIFT_LITERATURE_ANCHOR_REWARD
    )


# ── apply_z1_empirical_revision_to_candidate_delta (composition) ───────────


def test_z1_composition_within_class_floor_first_then_class_shift() -> None:
    # density 0.99 floors to -0.005; class_shift reward adds -0.02
    c = _cand(
        predicted_delta=-0.040,
        mdl_density=0.99,
        lane_class="substrate_class_shift",
    )
    result = loop.apply_z1_empirical_revision_to_candidate_delta(c)
    # First MDL floor: max(-0.040, -0.005) = -0.005
    # Then class-shift: -0.005 - 0.02 = -0.025
    assert result == pytest.approx(-0.025)


def test_z1_composition_across_class_with_double_reward() -> None:
    # density 0.50 = no penalty; both rewards stack
    c = _cand(
        predicted_delta=-0.050,
        mdl_density=0.50,
        lane_class="substrate_class_shift",
        literature_anchor="cooperative-receiver",
    )
    result = loop.apply_z1_empirical_revision_to_candidate_delta(c)
    expected = -0.050 - 0.02 - 0.01
    assert result == pytest.approx(expected)


def test_z1_composition_unknown_density_pure_reward() -> None:
    c = _cand(
        predicted_delta=-0.030,
        mdl_density=None,
        lane_class="substrate_class_shift",
    )
    result = loop.apply_z1_empirical_revision_to_candidate_delta(c)
    assert result == pytest.approx(-0.030 - 0.02)


def test_z1_composition_does_not_mutate_row() -> None:
    c = _cand(
        predicted_delta=-0.040,
        mdl_density=0.99,
        lane_class="substrate_class_shift",
    )
    original = c.predicted_score_delta
    loop.apply_z1_empirical_revision_to_candidate_delta(c)
    assert c.predicted_score_delta == original


# ── rank_candidates with Z1 revision ───────────────────────────────────────


def test_rank_within_class_candidate_ranked_lower_than_across_class() -> None:
    # Within-class (density 0.99) predicted_delta=-0.030 gets floored to
    # -0.005; across-class (density 0.50) predicted_delta=-0.020 stays at
    # -0.020. Across-class wins (more-negative-first).
    within = _cand("within", predicted_delta=-0.030, mdl_density=0.99)
    across = _cand("across", predicted_delta=-0.020, mdl_density=0.50)
    ranked = loop.rank_candidates(
        [within, across], rank_axis="predicted_score_delta"
    )
    assert [c.candidate_id for c in ranked] == ["across", "within"]


def test_rank_class_shift_reward_promotes_substrate() -> None:
    # Two candidates with same base predicted delta and same MDL density.
    # One has class-shift literature anchor; should rank higher.
    plain = _cand("plain", predicted_delta=-0.005, mdl_density=0.50)
    shifted = _cand(
        "shifted",
        predicted_delta=-0.005,
        mdl_density=0.50,
        lane_class="substrate_class_shift",
    )
    ranked = loop.rank_candidates(
        [plain, shifted], rank_axis="predicted_score_delta"
    )
    assert [c.candidate_id for c in ranked] == ["shifted", "plain"]


def test_rank_with_z1_revision_disabled_uses_original_delta() -> None:
    # When Z1 is disabled, within-class candidate keeps its big predicted
    # delta and ranks higher than the modest across-class candidate.
    within = _cand("within", predicted_delta=-0.030, mdl_density=0.99)
    across = _cand("across", predicted_delta=-0.020, mdl_density=0.50)
    ranked = loop.rank_candidates(
        [within, across],
        rank_axis="predicted_score_delta",
        apply_z1_empirical_revision=False,
    )
    assert [c.candidate_id for c in ranked] == ["within", "across"]


def test_rank_eig_per_dollar_within_class_saturated_penalized() -> None:
    # Saturated (density 0.99) has EIG/$ penalized 90%; trending (density
    # 0.92) penalized 50%; across-class (density 0.50) full EIG/$.
    saturated = _cand(
        "saturated", eig=10.0, cost_usd=1.0, mdl_density=0.99
    )  # base EIG/$ = 10.0 -> 1.0
    trending = _cand(
        "trending", eig=5.0, cost_usd=1.0, mdl_density=0.92
    )  # base EIG/$ = 5.0 -> 2.5
    across = _cand(
        "across", eig=3.0, cost_usd=1.0, mdl_density=0.50
    )  # base EIG/$ = 3.0 -> 3.0
    ranked = loop.rank_candidates([saturated, trending, across])
    assert [c.candidate_id for c in ranked] == ["across", "trending", "saturated"]


def test_rank_eig_per_dollar_class_shift_reward_promotes_non_nerv() -> None:
    """Default autopilot ranking must see class-shift rewards, not just the
    explicit predicted_score_delta axis."""
    hnerv_local = _cand(
        "hnerv_local",
        predicted_delta=-0.010,
        eig=0.010,
        cost_usd=1.0,
        mdl_density=0.50,
    )
    c6_substrate = _cand(
        "c6_mdl_ibps",
        family="mdl_ibps",
        predicted_delta=-0.005,
        eig=0.005,
        cost_usd=1.0,
        mdl_density=0.50,
        lane_class="substrate_class_shift",
        literature_anchor="MDL-IBPS Information Bottleneck",
    )
    ranked = loop.rank_candidates([hnerv_local, c6_substrate])
    assert [c.candidate_id for c in ranked] == ["c6_mdl_ibps", "hnerv_local"]


def test_rank_eig_per_dollar_tier_c_signal_affects_default_axis() -> None:
    """Tier-C across-class evidence must influence the default EIG/$ queue."""
    within = _cand(
        "tier_c_within",
        predicted_delta=-0.020,
        eig=0.020,
        cost_usd=1.0,
        mdl_tier_c_density=0.85,
    )
    across = _cand(
        "tier_c_across",
        family="mdl_ibps",
        predicted_delta=-0.012,
        eig=0.012,
        cost_usd=1.0,
        mdl_tier_c_density=0.13,
    )
    ranked = loop.rank_candidates([within, across])
    assert [c.candidate_id for c in ranked] == ["tier_c_across", "tier_c_within"]


def test_rank_eig_per_dollar_composition_alpha_affects_default_axis() -> None:
    """Composition alpha must down-rank saturating stacks on default EIG/$."""
    saturating = _cand(
        "z3_x_c6_saturating",
        family="substrate_composition",
        predicted_delta=-0.030,
        eig=0.030,
        cost_usd=1.0,
        composition_alpha=0.20,
    )
    additive = _cand(
        "z3_x_c6_additive",
        family="substrate_composition",
        predicted_delta=-0.012,
        eig=0.012,
        cost_usd=1.0,
        composition_alpha=1.00,
    )
    ranked = loop.rank_candidates([saturating, additive])
    assert [c.candidate_id for c in ranked] == [
        "z3_x_c6_additive",
        "z3_x_c6_saturating",
    ]


def test_rank_with_z1_revision_does_not_mutate_rows() -> None:
    c1 = _cand("c1", predicted_delta=-0.030, mdl_density=0.99)
    c2 = _cand("c2", predicted_delta=-0.005, mdl_density=0.50)
    original_c1 = c1.predicted_score_delta
    original_c2 = c2.predicted_score_delta
    loop.rank_candidates([c1, c2], rank_axis="predicted_score_delta")
    assert c1.predicted_score_delta == original_c1
    assert c2.predicted_score_delta == original_c2


def test_rank_unknown_density_candidates_not_penalized() -> None:
    # Two candidates: one with unknown density, one with low density.
    # Unknown should not be penalized; both keep their original delta.
    unknown = _cand("unknown", predicted_delta=-0.020, mdl_density=None)
    low = _cand("low", predicted_delta=-0.010, mdl_density=0.50)
    ranked = loop.rank_candidates(
        [unknown, low], rank_axis="predicted_score_delta"
    )
    assert [c.candidate_id for c in ranked] == ["unknown", "low"]


# ── A1-baseline regression ─────────────────────────────────────────────────


def test_a1_baseline_density_099_floors_aggressive_prediction() -> None:
    """A1 baseline has measured MDL density 0.99 per Z1. Even if someone
    predicts -0.040 for a within-A1-class bolt-on, the gate's MDL-density
    floor caps the effective predicted delta at -0.005."""
    a1_bolt_on = _cand(
        "a1_within_class_bolt_on",
        predicted_delta=-0.040,
        mdl_density=0.99,
    )
    effective = loop.apply_z1_empirical_revision_to_candidate_delta(a1_bolt_on)
    # No class-shift reward; only MDL floor applies.
    assert effective == loop.MDL_DENSITY_WITHIN_CLASS_SATURATED_DELTA_FLOOR


def test_pr106_baseline_density_097_floors_at_within_class() -> None:
    """PR106 baseline measured at 0.97 density per Z1. Same floor."""
    pr106_bolt_on = _cand(
        "pr106_within_class",
        predicted_delta=-0.025,
        mdl_density=0.97,
    )
    effective = loop.apply_z1_empirical_revision_to_candidate_delta(pr106_bolt_on)
    assert effective == loop.MDL_DENSITY_WITHIN_CLASS_SATURATED_DELTA_FLOOR


# ── C6 MDL-IBPS / time-traveler double-reward ──────────────────────────────


def test_c6_mdl_ibps_double_reward_both_lane_class_and_literature() -> None:
    """C6 MDL-IBPS substrate (literature: Tishby-Zaslavsky + Atick-Redlich)
    is a class-shift candidate and gets both rewards per Z1 council."""
    c6 = _cand(
        "lane_c6_mdl_ibps_substrate",
        predicted_delta=-0.030,
        mdl_density=None,  # not yet measured
        lane_class="substrate_class_shift",
        literature_anchor=(
            "Tishby-Zaslavsky Information Bottleneck + Atick-Redlich "
            "predictive-coding"
        ),
    )
    effective = loop.apply_z1_empirical_revision_to_candidate_delta(c6)
    expected = -0.030 - 0.02 - 0.01  # both rewards stack
    assert effective == pytest.approx(expected)


def test_time_traveler_l5_literature_anchor_reward() -> None:
    """time_traveler_l5_packet is the canonical class-shift substrate per
    `feedback_adjusted_floor_v3_alien_tech_routing_landed_20260513.md`."""
    tt = _cand(
        "lane_time_traveler_l5",
        predicted_delta=-0.040,
        mdl_density=None,
        literature_anchor="time_traveler_l5_packet substrate",
    )
    effective = loop.apply_z1_empirical_revision_to_candidate_delta(tt)
    assert effective == pytest.approx(-0.040 - 0.01)


# ── load_candidates_from_jsonl reads new Z1 fields ─────────────────────────


def test_load_jsonl_reads_mdl_density_field(tmp_path: Path) -> None:
    jsonl = tmp_path / "q.jsonl"
    rows_payload = [
        {
            "candidate_id": "c1",
            "family": "f",
            "predicted_score_delta": -0.01,
            "expected_information_gain": 1.0,
            "estimated_dispatch_cost_usd": 5.0,
            "mdl_density": 0.99,
            "lane_class": "hnerv_lc_v2",
            "literature_anchor": "HNeRV family",
        }
    ]
    jsonl.write_text("\n".join(json.dumps(r) for r in rows_payload))
    rows = loop.load_candidates_from_jsonl(jsonl)
    assert len(rows) == 1
    assert rows[0].mdl_density == pytest.approx(0.99)
    assert rows[0].lane_class == "hnerv_lc_v2"
    assert rows[0].literature_anchor == "HNeRV family"


def test_load_jsonl_handles_missing_z1_fields_backcompat(tmp_path: Path) -> None:
    jsonl = tmp_path / "q.jsonl"
    rows_payload = [
        {
            "candidate_id": "c1",
            "family": "f",
            "predicted_score_delta": -0.01,
            "expected_information_gain": 1.0,
            "estimated_dispatch_cost_usd": 5.0,
        }
    ]
    jsonl.write_text("\n".join(json.dumps(r) for r in rows_payload))
    rows = loop.load_candidates_from_jsonl(jsonl)
    assert len(rows) == 1
    assert rows[0].mdl_density is None
    assert rows[0].lane_class is None
    assert rows[0].literature_anchor == ""


def test_load_jsonl_coerces_malformed_mdl_density_to_none(tmp_path: Path) -> None:
    jsonl = tmp_path / "q.jsonl"
    rows_payload = [
        {
            "candidate_id": "c1",
            "family": "f",
            "predicted_score_delta": -0.01,
            "expected_information_gain": 1.0,
            "estimated_dispatch_cost_usd": 5.0,
            "mdl_density": "not_a_float",
        }
    ]
    jsonl.write_text("\n".join(json.dumps(r) for r in rows_payload))
    rows = loop.load_candidates_from_jsonl(jsonl)
    assert rows[0].mdl_density is None
