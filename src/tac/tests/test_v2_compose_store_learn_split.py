# SPDX-License-Identifier: MIT
"""Tests for tac.v2_compose.store_learn_split — the KNOWN-split encoder (keystone S1)."""

from __future__ import annotations

from pathlib import Path

import pytest

from tac.v2_compose.store_learn_split import (
    DECISION_GENERATE,
    DECISION_LEARN,
    WARP_GROUND_HOMOGRAPHY,
    WARP_IDENTITY,
    WARP_ROTATION_ONLY,
    ClassRecoverability,
    encode_known_split,
    load_reach_kstar,
    load_warp_recoverability_from_grok,
)

_REPO = Path(__file__).resolve().parents[3]


def _measured_like() -> dict[str, ClassRecoverability]:
    """The MEASURED grok recoverability (FEED-iz/ja; advisory) as a synthetic fixture."""
    return {
        "Road": ClassRecoverability("Road", 0.0231, 0.0196, None, 0.151, 0.2296),
        "Lane": ClassRecoverability("Lane", 0.5795, 0.5851, 0.9798, -0.0096, 0.0059),
        "Undriv": ClassRecoverability("Undriv", 0.0024, 0.0026, 0.0085, -0.092, 0.4933),
        "Movable": ClassRecoverability("Movable", 0.0506, 0.0522, 0.3999, -0.030, 0.0155),
        "MyCar": ClassRecoverability("MyCar", 0.0031, 0.0195, 0.0027, -5.252, 0.2557),
    }


def test_known_split_matches_measured_optimum():
    """The rule reproduces the KNOWN split: Road=ground, Undriv=rotonly, MyCar=identity; Lane+Movable=LEARN."""
    plan = encode_known_split(_measured_like(), reach_kstar=47, n_pairs=600)
    assert plan.per_class["Road"].decision == DECISION_GENERATE
    assert plan.per_class["Road"].warp_type == WARP_GROUND_HOMOGRAPHY
    assert plan.per_class["Undriv"].decision == DECISION_GENERATE
    assert plan.per_class["Undriv"].warp_type == WARP_ROTATION_ONLY
    assert plan.per_class["MyCar"].decision == DECISION_GENERATE
    assert plan.per_class["MyCar"].warp_type == WARP_IDENTITY
    assert plan.per_class["Lane"].decision == DECISION_LEARN
    assert plan.per_class["Lane"].warp_type is None
    assert plan.per_class["Movable"].decision == DECISION_LEARN
    assert set(plan.generate_classes) == {"Road", "Undriv", "MyCar"}
    assert set(plan.learn_classes) == {"Lane", "Movable"}
    assert set(plan.residual_target_classes) == {"Lane", "Movable"}


def test_keyframe_count_is_ceil():
    plan = encode_known_split(_measured_like(), reach_kstar=47, n_pairs=600)
    assert plan.keyframe_count == 13  # ceil(600/47)
    plan2 = encode_known_split(_measured_like(), reach_kstar=50, n_pairs=600)
    assert plan2.keyframe_count == 12  # ceil(600/50)


def test_break_even_is_positive_and_below_floor():
    """At the known-store rate the frontier break-even d_seg is positive (~0.0017) and BELOW the
    deterministic bulk floor (0.0185) -> the residual INR must close the gap (FACT 2)."""
    plan = encode_known_split(_measured_like(), reach_kstar=47, n_pairs=600)
    be = plan.break_even["d_seg_to_beat_frontier_at_known_store"]
    assert 0.0 < be < 0.005
    assert plan.break_even["deterministic_bulk_dseg_floor"] > be  # the gap the residual must close


def test_learn_byte_cost_is_open():
    plan = encode_known_split(_measured_like(), reach_kstar=47, n_pairs=600)
    assert plan.predicted_bytes["learn_residual_inr"] is None  # OPEN — the GPU run
    assert plan.predicted_bytes["free_generated"] == 0          # rule-118 FREE
    assert plan.predicted_bytes["store_total"] > 0


def test_threshold_moves_decision():
    """A class just above the recoverability threshold is LEARN; just below is GENERATE."""
    rec = {"X": ClassRecoverability("X", 0.10, 0.10, None, 0.0, 0.1)}
    learn = encode_known_split(rec, reach_kstar=47, n_pairs=600, recoverability_dseg_max=0.05)
    assert learn.per_class["X"].decision == DECISION_LEARN
    gen = encode_known_split(rec, reach_kstar=47, n_pairs=600, recoverability_dseg_max=0.2)
    assert gen.per_class["X"].decision == DECISION_GENERATE


def test_class_index_is_metadata_only_not_decision():
    """The decision must not depend on the class index (CLAUDE.md NO-hardcode-index rule)."""
    rec = _measured_like()
    p_no_map = encode_known_split(rec, reach_kstar=47, n_pairs=600)
    p_scrambled = encode_known_split(
        rec, reach_kstar=47, n_pairs=600,
        class_index_map={"Road": 4, "Lane": 0, "Undriv": 1, "Movable": 3, "MyCar": 2},
    )
    # identical decisions regardless of the (metadata-only) index map
    for c in rec:
        assert p_no_map.per_class[c].decision == p_scrambled.per_class[c].decision
        assert p_no_map.per_class[c].warp_type == p_scrambled.per_class[c].warp_type
    assert p_scrambled.per_class["Road"].cls_index == 4  # recorded, but not used


def test_attribution_confirmation():
    rec = _measured_like()
    confirmed = encode_known_split(
        rec, reach_kstar=47, n_pairs=600,
        attribution={"residual_classes": ["Lane", "Movable"]},
    )
    assert confirmed.confirmed_by_attribution is True
    diverged = encode_known_split(
        rec, reach_kstar=47, n_pairs=600,
        attribution={"residual_classes": ["Road"]},  # Road is GENERATE -> divergence is a FINDING
    )
    assert diverged.confirmed_by_attribution is False  # never overrides; records the finding


def test_empty_and_bad_inputs_raise():
    with pytest.raises(ValueError):
        encode_known_split({}, reach_kstar=47, n_pairs=600)
    with pytest.raises(ValueError):
        encode_known_split(_measured_like(), reach_kstar=0, n_pairs=600)
    with pytest.raises(ValueError):
        encode_known_split(_measured_like(), reach_kstar=47, n_pairs=0)


def test_to_json_roundtrips_shape():
    plan = encode_known_split(_measured_like(), reach_kstar=47, n_pairs=600)
    j = plan.to_json()
    assert j["score_claim"] is False
    assert j["promotable"] is False
    assert set(j["per_class"]) == set(_measured_like())
    assert j["keyframe_count"] == 13


# --- real-data smoke (skipped if the measured JSONs are absent) ---
_GROK = _REPO / "experiments/results/grok_pose_warp_dseg_20260629T181000Z/results.json"
_REACH = _REPO / "experiments/results/screw_reach/reach_n96.json"


@pytest.mark.skipif(not (_GROK.exists() and _REACH.exists()), reason="measured JSONs absent")
def test_real_measured_data_reproduces_known_split():
    rec = load_warp_recoverability_from_grok(_GROK)
    kstar = load_reach_kstar(_REACH)
    plan = encode_known_split(rec, kstar, n_pairs=600)
    assert set(plan.generate_classes) == {"Road", "Undriv", "MyCar"}
    assert set(plan.learn_classes) == {"Lane", "Movable"}
    assert kstar == 47
    assert plan.keyframe_count == 13
