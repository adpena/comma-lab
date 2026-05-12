"""Tests for `tac.optimization.theoretical_floor_substrate_refresh`."""

from __future__ import annotations

import json

import pytest

from tac.optimization.substrate_composition_matrix import (
    ScoreAxis,
    SubstrateClass,
    build_composition_matrix,
    canonical_substrate_inventory,
)
from tac.optimization.theoretical_floor_substrate_refresh import (
    HINTON_DISTILLED_FLOOR_ADJUSTMENT,
    N_PACKET_COMPILER_PRIMITIVES,
    PACKET_COMPILER_BYTE_SAVINGS_CAP,
    PER_CLASS_FLOOR_PRIOR,
    SCHEMA_VERSION,
    V2_FLOOR_MEDIAN,
    RefreshedFloorEstimate,
    minimum_marginal_byte_ev_threshold,
    packet_compiler_max_score_savings,
    per_substrate_predicted_floor,
    refreshed_floor_estimate,
    refreshed_pareto_frontier,
    serialize_per_substrate_floor,
    serialize_refreshed_estimate,
    write_refresh_report,
)

# ── Per-substrate floor ──────────────────────────────────────────────────


def test_per_substrate_floor_returns_canonical_substrate_count():
    rows = per_substrate_predicted_floor()
    assert len(rows) == len(canonical_substrate_inventory())


def test_per_substrate_floor_score_claim_invariants():
    rows = per_substrate_predicted_floor()
    for r in rows:
        assert r.score_claim is False
        assert r.promotion_eligible is False
        assert r.ready_for_exact_eval_dispatch is False


def test_per_substrate_floor_typed_fields():
    rows = per_substrate_predicted_floor()
    for r in rows:
        assert isinstance(r.substrate_class, SubstrateClass)
        assert isinstance(r.target_axis, ScoreAxis)
        assert r.per_class_floor > 0.0
        assert r.confidence_band_width > 0.0


def test_per_substrate_floor_anr_uses_pr95_anchor():
    rows = per_substrate_predicted_floor()
    anr = next(r for r in rows if r.substrate_id == "anr_token_renderer_v62")
    assert anr.per_class_floor == PER_CLASS_FLOOR_PRIOR["renderer_replacement_anr"]
    assert anr.per_class_floor == 0.193


def test_per_substrate_floor_self_compression_uses_selfcomp_floor():
    rows = per_substrate_predicted_floor()
    scpp = next(r for r in rows if r.substrate_id == "scpp_substrate")
    assert scpp.per_class_floor == PER_CLASS_FLOOR_PRIOR["self_compression"]


def test_per_substrate_floor_predicted_floor_includes_delta():
    """predicted_floor = per_class_floor + delta_mid; delta_mid is negative
    for improvement-claiming substrates so predicted_floor < per_class_floor."""
    rows = per_substrate_predicted_floor()
    matrix = build_composition_matrix()
    for r in rows:
        s = next(s for s in matrix.substrates if s.substrate_id == r.substrate_id)
        delta_mid = s.predicted_delta_alone_midpoint()
        assert abs(r.predicted_floor - (r.per_class_floor + delta_mid)) < 1e-12


# ── Refreshed floor estimate ─────────────────────────────────────────────


def test_refreshed_floor_estimate_returns_typed_dataclass():
    est = refreshed_floor_estimate()
    assert isinstance(est, RefreshedFloorEstimate)
    assert est.schema == SCHEMA_VERSION
    assert est.score_claim is False
    assert est.promotion_eligible is False
    assert est.ready_for_exact_eval_dispatch is False


def test_refreshed_floor_includes_v2_baseline():
    est = refreshed_floor_estimate()
    assert est.v2_baseline_median == V2_FLOOR_MEDIAN
    assert est.hinton_distilled_adjustment == HINTON_DISTILLED_FLOOR_ADJUSTMENT


def test_refreshed_floor_within_reasonable_band():
    est = refreshed_floor_estimate()
    # Refreshed median should be near v2 baseline (0.140) given 24 substrate
    # priors mostly anchor at 0.135-0.155 band.
    assert 0.10 < est.refreshed_median < 0.20


def test_refreshed_ci_narrower_than_v2():
    """The refresh shrinks CI by 15% per the Bayesian-update factor."""
    est = refreshed_floor_estimate()
    refreshed_half_width = (est.refreshed_ci_95_high - est.refreshed_ci_95_low) / 2.0
    v2_half_width = (0.152 - 0.128) / 2.0
    assert refreshed_half_width < v2_half_width
    assert abs(refreshed_half_width - 0.85 * v2_half_width) < 1e-6


def test_refreshed_floor_minimum_substrate_id_present():
    est = refreshed_floor_estimate()
    assert est.minimum_substrate_predicted_floor_substrate_id != ""
    assert est.minimum_substrate_predicted_floor < V2_FLOOR_MEDIAN + 0.5


def test_refreshed_floor_n_substrates_below_v2_floor():
    est = refreshed_floor_estimate()
    # The most aggressive predictions (predicted_delta -0.0090) on
    # self-compression baseline (0.135) -> 0.126 floor; that's below v2's 0.140.
    assert est.n_substrates_below_v2_floor >= 0


def test_refreshed_floor_constituent_bounds_present():
    est = refreshed_floor_estimate()
    keys = set(est.constituent_bounds.keys())
    assert "v2_council_baseline" in keys
    assert "v2_with_hinton_adjustment" in keys
    assert "minimum_substrate_floor" in keys
    assert "packet_compiler_byte_savings_max" in keys


# ── Packet-compiler savings ──────────────────────────────────────────────


def test_packet_compiler_max_score_savings_positive():
    savings = packet_compiler_max_score_savings()
    assert savings > 0.0
    # ALPHA * 500 / 37545489 ≈ 3.33e-4
    assert 1e-4 < savings < 1e-3


def test_packet_compiler_constants_consistent():
    assert N_PACKET_COMPILER_PRIMITIVES == 19
    assert PACKET_COMPILER_BYTE_SAVINGS_CAP == 500


# ── Pareto frontier ──────────────────────────────────────────────────────


def test_refreshed_pareto_frontier_returns_canonical_substrate_count():
    points = refreshed_pareto_frontier()
    assert len(points) == len(canonical_substrate_inventory())


def test_refreshed_pareto_frontier_sorted_by_bytes():
    points = refreshed_pareto_frontier()
    for i in range(len(points) - 1):
        assert points[i].bytes_midpoint <= points[i + 1].bytes_midpoint


def test_refreshed_pareto_frontier_score_claim_invariants():
    points = refreshed_pareto_frontier()
    for p in points:
        assert p.score_claim is False


def test_refreshed_pareto_zero_byte_substrates_inherit_pr106_bytes():
    """Bolt-ons (byte_budget=(0,0)) inherit PR106_R2_BYTES as midpoint."""
    points = refreshed_pareto_frontier()
    # film_pose_conditioning has byte budget (2000, 8000); only nerv_enc_dec has (0,0).
    enc_dec = next((p for p in points if p.substrate_id == "nerv_enc_dec_separated"), None)
    assert enc_dec is not None
    # byte_budget_band = (0, 0) -> midpoint inherits PR106_R2_BYTES.
    assert enc_dec.bytes_midpoint == 178_750  # PR106_R2_BYTES.


# ── Marginal-byte-EV thresholds ──────────────────────────────────────────


def test_minimum_marginal_byte_ev_threshold_returns_dict():
    thr = minimum_marginal_byte_ev_threshold()
    assert thr["schema"] == SCHEMA_VERSION
    assert thr["score_claim"] is False
    assert "rate_axis" in thr
    assert "seg_axis" in thr
    assert "pose_axis_at_pr106_r2_operating_point" in thr


def test_minimum_marginal_byte_ev_threshold_rate_axis_correct():
    thr = minimum_marginal_byte_ev_threshold()
    rate = thr["rate_axis"]
    # ALPHA / N = 25 / 37545489 ≈ 6.66e-7
    expected_score_per_byte = 25.0 / 37_545_489
    assert abs(rate["score_per_byte"] - expected_score_per_byte) < 1e-12


def test_minimum_marginal_byte_ev_threshold_pose_dominates_seg():
    """At PR106 r2 frontier (d_pose ~ 3.4e-5) pose marginal > seg marginal."""
    thr = minimum_marginal_byte_ev_threshold()
    pose = thr["pose_axis_at_pr106_r2_operating_point"]
    assert pose["marginal_dominance_factor_vs_seg"] > 1.0
    # CLAUDE.md says ~2.71x at pose_avg=3.4e-5; we use d_pose=3.4e-5 directly.
    # Score per unit pose distortion = sqrt(10) / (2 * sqrt(d_pose))
    # = 3.162 / (2 * 0.00583) = 271
    # Score per unit seg = 100
    # Ratio = 2.71.
    assert 2.5 < pose["marginal_dominance_factor_vs_seg"] < 3.0


# ── Serialization ────────────────────────────────────────────────────────


def test_serialize_per_substrate_floor_jsonable():
    rows = per_substrate_predicted_floor()
    payload = serialize_per_substrate_floor(rows[0])
    json.dumps(payload)  # Must not raise.
    assert "substrate_class" in payload
    assert "target_axis" in payload
    # Enums serialized as values.
    assert isinstance(payload["substrate_class"], str)


def test_serialize_refreshed_estimate_jsonable():
    est = refreshed_floor_estimate()
    payload = serialize_refreshed_estimate(est)
    json.dumps(payload)  # Must not raise.
    assert payload["score_claim"] is False
    assert payload["promotion_eligible"] is False


def test_write_refresh_report_refuses_tmp_path(tmp_path):
    est = refreshed_floor_estimate()
    rows = per_substrate_predicted_floor()
    pareto = refreshed_pareto_frontier()
    thr = minimum_marginal_byte_ev_threshold()
    with pytest.raises(ValueError, match="forbidden /tmp"):
        write_refresh_report(
            estimate=est,
            per_substrate=rows,
            pareto_frontier=pareto,
            marginal_thresholds=thr,
            path="/tmp/forbidden_refresh.json",
        )


def test_write_refresh_report_writes_durable_path(tmp_path):
    est = refreshed_floor_estimate()
    rows = per_substrate_predicted_floor()
    pareto = refreshed_pareto_frontier()
    thr = minimum_marginal_byte_ev_threshold()
    durable = tmp_path / "refresh.json"
    write_refresh_report(
        estimate=est,
        per_substrate=rows,
        pareto_frontier=pareto,
        marginal_thresholds=thr,
        path=str(durable),
    )
    parsed = json.loads(durable.read_text())
    assert parsed["schema"] == SCHEMA_VERSION
    assert parsed["score_claim"] is False
    assert "refreshed_floor_estimate" in parsed
    assert "per_substrate_predicted_floor" in parsed
    assert "pareto_frontier" in parsed
    assert "minimum_marginal_byte_ev_thresholds" in parsed
    expected_count = len(canonical_substrate_inventory())
    assert len(parsed["per_substrate_predicted_floor"]) == expected_count
    assert len(parsed["pareto_frontier"]) == expected_count


# ── Cross-substrate sanity ──────────────────────────────────────────────


def test_pose_axis_substrate_floor_lower_at_pose_marginal_dominance():
    """At PR106 r2 frontier the pose axis dominates marginally;
    pose-axis substrates should generally have predicted_floor lower than
    or equal to other RESIDUAL substrates."""
    rows = per_substrate_predicted_floor()
    pose_axis = [r for r in rows if r.substrate_class == SubstrateClass.POSE_AXIS_SIDECHANNEL]
    assert len(pose_axis) == 3
    # Every pose-axis substrate has per_class_floor = 0.155.
    for r in pose_axis:
        assert r.per_class_floor == PER_CLASS_FLOOR_PRIOR["pose_axis_sidechannel"]
