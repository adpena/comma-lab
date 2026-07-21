# SPDX-License-Identifier: MIT
from __future__ import annotations

import math

import pytest

from tac.witness_control.costate_organ_v2 import (
    DESIGN_REALIZABILITY,
    FULL_KERNEL_VISIBLE,
    aggregate_readback,
    apparatus_validity,
    byte_price_factor,
    closed_form_seg_pullback,
    compose_lambda,
    dual_metric_readback,
    ema_delag_delta,
    exact_resize_adjoint_four_tap,
    pool_aware_rank,
    realizability_factor,
    score_debt,
    visibility_factor,
    xi_transport_factor,
)


def test_exact_debt_is_zero_at_anchor_and_score_law_above_it():
    assert score_debt(current_dseg=0.00015196, current_dpose=0.00010184)["total_s"] == 0.0
    d = score_debt(current_dseg=0.00115196, current_dpose=0.00110184)
    assert d["seg_s"] == pytest.approx(0.1)
    assert d["pose_s"] > 0.0


def test_exact_four_tap_resize_adjoint_and_refusals():
    assert exact_resize_adjoint_four_tap(2.0, (0.1, 0.2, 0.3, 0.4)) == pytest.approx(
        (0.2, 0.4, 0.6, 0.8))
    with pytest.raises(ValueError, match="exactly four"):
        exact_resize_adjoint_four_tap(1.0, (0.5, 0.5))
    with pytest.raises(ValueError, match="sum to one"):
        exact_resize_adjoint_four_tap(1.0, (0.1, 0.1, 0.1, 0.1))


def test_closed_rank4_head_resize_pullback_is_no_vjp_and_frame0_zero():
    row = closed_form_seg_pullback(
        score_costate_s=0.2, pair_head_norm=4.0,
        tap_weights=(0.1, 0.2, 0.3, 0.4), frame_index=1, orientation=-1)
    assert row["camera_cotangent_four_tap"] == pytest.approx((-0.08, -0.16, -0.24, -0.32))
    assert row["head_rank"] == 4 and row["all_class_gauge_null"]
    assert row["vjp_used"] is False
    frame0 = closed_form_seg_pullback(
        score_costate_s=0.2, pair_head_norm=4.0,
        tap_weights=(0.1, 0.2, 0.3, 0.4), frame_index=0)
    assert frame0["camera_cotangent_four_tap"] == (0.0, 0.0, 0.0, 0.0)


def test_visibility_is_frame_and_channel_structural():
    assert visibility_factor(task="seg", frame_index=0, channel="r")["value"] == 0.0
    assert visibility_factor(task="seg", frame_index=1, channel="r")["value"] == pytest.approx(
        FULL_KERNEL_VISIBLE)
    assert visibility_factor(task="pose", frame_index=0, channel="chroma",
                             spatial_scale_px=1.5)["value"] == 0.0
    # Pose-null chroma HF does not erase SegNet's full-resolution chroma path.
    assert visibility_factor(task="seg", frame_index=1, channel="chroma",
                             spatial_scale_px=1.5)["value"] > 0.0


def test_realizability_quantization_and_formulation_gates():
    assert realizability_factor()["value"] == pytest.approx(DESIGN_REALIZABILITY)
    assert realizability_factor(requested=512, survived_clean=289)["value"] == pytest.approx(
        289 / 512)
    assert realizability_factor(formulation_valid=False)["value"] == 0.0
    assert realizability_factor(apparatus_valid=False)["value"] == 0.0


def test_domain_refined_byte_price_pays_only_inside_break_even():
    free = byte_price_factor(realized_recovery_s=0.001, charged_bytes=0)
    assert free["value"] == 1.0 and free["required_registry_event"] == "domain_refined"
    too_large = byte_price_factor(realized_recovery_s=0.001, charged_bytes=10_000)
    assert too_large["value"] == 0.0 and not too_large["pays_rent"]


def test_composition_is_exact_product_and_bounds_factors():
    row = compose_lambda(exact_gap=0.2, visibility=0.5, realizability=0.25,
                         byte_price=0.8)
    assert row.lambda_value == pytest.approx(0.02)
    with pytest.raises(ValueError, match="must be <= 1"):
        compose_lambda(exact_gap=1, visibility=2, realizability=1, byte_price=1)


def test_pool_aware_kkt_caps_competing_rows_in_same_pool():
    ranked = pool_aware_rank([
        {"candidate": "a", "opportunity_pool": "A", "lambda": 0.15,
         "pool_ceiling_s": 0.2},
        {"candidate": "b", "opportunity_pool": "A", "lambda": 0.10,
         "pool_ceiling_s": 0.2},
        {"candidate": "c", "opportunity_pool": "B", "lambda": 0.04,
         "pool_ceiling_s": 0.05},
    ])
    by_name = {r["candidate"]: r for r in ranked}
    assert by_name["a"]["pool_kkt_marginal"] == pytest.approx(0.15)
    assert by_name["b"]["pool_kkt_marginal"] == pytest.approx(0.05)
    assert by_name["c"]["pool_kkt_marginal"] == pytest.approx(0.04)
    assert all(r["same_pool_addition_forbidden"] for r in ranked)


def test_dual_metrics_emit_both_and_preserve_sign_flip():
    out = dual_metric_readback((1.0, 1.0), (1.0, -2.0), (100.0, 1.0))
    assert out["euclidean_cosine"] < 0.0
    assert out["fisher_cosine"] > 0.0
    assert out["sign_flip_informative"] is True
    assert out["blend_forbidden"] is True


def test_apparatus_poison_ema_and_maturity_fail_closed():
    bad = apparatus_validity(flags={"ckpt-every": "1"}, maturity="unknown")
    assert not bad["valid_for_backtest"] and bad["bench_contaminated"]
    assert bad["ema_lag_correction"] == "unknown_not_applied"
    assert bad["maturity"] == "_dev" and not bad["pointer_eligible"]
    prod = apparatus_validity(maturity="_prod", ema_reset_verified=True, topology_event=True)
    assert prod["pointer_eligible"] and not prod["xi_transport_eligible"]


def test_ema_delag_requires_verified_lag_custody():
    unknown = ema_delag_delta(
        observed_delta=0.03, estimated_lag_delta=0.02, reset_verified=False)
    assert unknown["value"] == 0.03 and not unknown["applied"]
    verified = ema_delag_delta(
        observed_delta=0.03, estimated_lag_delta=0.02, reset_verified=True)
    assert verified["value"] == pytest.approx(0.01) and verified["applied"]


def test_xi_transport_is_optional_and_refused_on_topology_events():
    assert xi_transport_factor(transport_cosine=None,
                               sparse_topology_event=False)["value"] == 1.0
    assert xi_transport_factor(transport_cosine=0.7,
                               sparse_topology_event=False)["value"] == pytest.approx(0.7)
    event = xi_transport_factor(transport_cosine=0.9, sparse_topology_event=True)
    assert event["value"] == 0.0 and event["refusal"] == "sparse_topology_event"


def test_aggregate_readback_refuses_site_lambda_and_has_no_learned_params():
    row = aggregate_readback({"d_seg": 0.003146, "d_pose": 0.001})
    assert row["status"] == "SENSE_READY_SITE_UNBOUND"
    assert row["lambda"] is None
    assert row["realizability"]["learned_parameters"] == 0
    assert row["actuation"] == "NONE" and row["pointer_changed"] is False
    assert math.isclose(row["visibility"]["frame_0_seg_lambda"], 0.0)


def test_shadow_row_carries_v2_beside_v1_without_actuation(tmp_path):
    from tac.witness_control.shadow_controller import RunInputs, build_shadow_report

    report = build_shadow_report(RunInputs(
        run_dir=tmp_path, verdicts=[], stage_rows={}, flags={}))
    row = report.to_row()
    assert "factorized_adjoint" in row and "costate_organ_v2" in row
    assert row["costate_organ_v2"]["status"] == "UNAVAILABLE_NO_VERDICT"
    assert row["costate_organ_v2"]["actuation"] == "NONE"


def test_canonical_composition_law_is_backtest_anchored_and_nonpromotable():
    from tac.canonical_equations.costate_organ_exact_anchor_product_20260721 import (
        build_costate_organ_exact_anchor_product_v2,
    )

    equation = build_costate_organ_exact_anchor_product_v2()
    assert equation.equation_id == "costate_organ_exact_anchor_product_v2"
    assert equation.domain_of_validity["formalization_status"] == (
        "ANCHORED_RETROSPECTIVE_DEVELOPMENT")
    assert equation.domain_of_validity["promotion_eligible"] is False
    assert equation.empirical_anchors[0].empirical_output[
        "exact_anchor_v2_spearman"] > equation.empirical_anchors[0].empirical_output[
            "old_decide_spearman"]
