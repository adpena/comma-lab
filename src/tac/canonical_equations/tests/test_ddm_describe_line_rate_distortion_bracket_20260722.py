# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tac.canonical_equations.ddm_describe_line_rate_distortion_bracket_20260722 import (
    EQUATION_ID,
    LEG_SPARSE_PIXEL,
    LEG_STRUCTURED_CARRIER,
    LEG_VALUE_EXACTNESS,
    V7_CROSS_SHA256,
    V7_VERDICT,
    V8_CROSS_SHA256,
    V8_VERDICT,
    V9_CROSS_SHA256,
    V9_VERDICT,
    V12_N600_SHA256,
    V12_VERDICT,
    V13_N600_SHA256,
    V13_PHASE_N64_SHA256,
    V13_PHASE_N600_SHA256,
    V13_VERDICT,
    V14_G4_SHA256,
    V14_N600_SHA256,
    V14_VERDICT,
    _v7_inputs,
    _v8_inputs,
    _v9_inputs,
    _v12_inputs,
    _v13_inputs,
    _v14_inputs,
    build_ddm_describe_line_rate_distortion_bracket_v1,
    evaluate_ddm_describe_line_rate_distortion_bracket,
    populate_ddm_describe_line_rate_distortion_bracket_v1,
)
from tac.canonical_equations.registry import query_equations

REPO = Path(__file__).resolve().parents[4]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_three_measured_legs_and_n600_anchor_predict_scoped_receipt_verdicts() -> None:
    v7 = evaluate_ddm_describe_line_rate_distortion_bracket(_v7_inputs())
    v8 = evaluate_ddm_describe_line_rate_distortion_bracket(_v8_inputs())
    v9 = evaluate_ddm_describe_line_rate_distortion_bracket(_v9_inputs())
    v12 = evaluate_ddm_describe_line_rate_distortion_bracket(_v12_inputs())
    v13 = evaluate_ddm_describe_line_rate_distortion_bracket(_v13_inputs())
    v14 = evaluate_ddm_describe_line_rate_distortion_bracket(_v14_inputs())

    assert v7["leg"] == LEG_VALUE_EXACTNESS and v7["verdict"] == V7_VERDICT
    assert v7["evaluator_gate_green"] is True
    assert v7["rate_multiple_min"] == pytest.approx(215.560765)
    assert v7["rate_multiple_max"] == pytest.approx(856.66327)

    assert v8["leg"] == LEG_SPARSE_PIXEL and v8["verdict"] == V8_VERDICT
    assert v8["selected_fraction_min"] == pytest.approx(0.040182133516)
    assert v8["selected_fraction_max"] == pytest.approx(0.045302073161)
    assert v8["byte_collapse_min"] > 0.93
    assert v8["dseg_floor_min"] == pytest.approx(0.025907576084)

    assert v9["leg"] == LEG_STRUCTURED_CARRIER and v9["verdict"] == V9_VERDICT
    assert v9["rate_gate_green"] is True
    assert v9["correction_symbols_measured"] == 0
    assert v9["marginal_bytes_per_pair_exact"] == "20729/192"
    assert v9["n600_projected_bytes_exact"] == "2628875/24"
    assert v9["n600_projection_status"].endswith("NOT_MEASURED_N600")

    assert v12["verdict"] == V12_VERDICT
    assert v12["n600_projection_status"] == "MEASURED_N600_RECEIVER_CLOSED"
    assert v12["n600_measured"] == {
        "archive_bytes": 106_106,
        "d_seg": pytest.approx(0.034003668891),
        "d_pose": pytest.approx(163.034719422881),
        "receiver_closed": True,
    }
    assert v12["verdict_scope"].startswith("FORMULATION_CORRECT_A_BOUND")
    assert v12["decision_atom_partition"] == {
        "bounded_atoms": 4_096,
        "exact_scorer_measured_atoms": 3_994,
        "strict_receiver_rejected_atoms": 66,
        "prior_higher_ev_conflict_excluded_atoms": 36,
        "valid": True,
    }
    assert v13["verdict"] == V13_VERDICT
    assert v13["n600_projection_status"] == "MEASURED_N600_RECEIVER_CLOSED"
    assert v13["g1_worldsheet_realization"] == {
        "payload_exact": True,
        "mask_level_clean_rest_dseg": pytest.approx(0.00028294881184895833),
        "through_r_movable_dseg": pytest.approx(0.481331895297),
        "receiver_projection_bound": True,
        "operator_addenda_status": "PRE_ADDENDUM_BASELINE_SUCCESSORS_UNMEASURED",
        "phase_receiver_closed": True,
        "phase_result_status": "RAW_Q8_LANE_HELP_TOTAL_HARM",
        "phase_total_dseg_delta": pytest.approx(0.000280736287),
        "phase_lane_dseg_delta": pytest.approx(-0.029228004790),
    }
    assert v13["verdict_scope"].startswith("INSTANCE_G1")
    assert v14["verdict"] == V14_VERDICT
    assert v14["n600_projection_status"] == "MEASURED_N600_RECEIVER_CLOSED"
    assert v14["g1_worldsheet_realization"] == {
        "payload_exact": True,
        "mask_level_clean_rest_dseg": pytest.approx(0.00028294881184895833),
        "through_r_movable_dseg": pytest.approx(0.291615222639),
        "receiver_projection_bound": True,
        "receiver_realization_profile": True,
        "profile_bytes": 23,
        "camera_resolution_placement": True,
        "exact_g1_replacement": True,
        "static_fields_measured": 3,
        "static_joint_positive_count": 1,
        "best_static_joint_score_delta": pytest.approx(-0.005003006483),
        "ar1_bev_status": "BLOCKED_NO_DECODER_FREE_PHYSICAL_BEV_CUSTODY",
    }
    assert v14["verdict_scope"].startswith("INSTANCE_V14")


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        (lambda row: row.update({"leg": "unknown"}), "unknown describe-line leg"),
        (
            lambda row: row["windows"].__setitem__(0, {**row["windows"][0], "receiver_closed": 1}),
            "receiver_closed must be boolean",
        ),
        (
            lambda row: row.update({"pixel_residual_present": True}),
            "forbids pixel_residual_present",
        ),
    ],
)
def test_law_fails_closed(mutator, match: str) -> None:
    row = _v9_inputs()
    mutator(row)
    with pytest.raises(ValueError, match=match):
        evaluate_ddm_describe_line_rate_distortion_bracket(row)


def test_equation_has_six_bound_nonpromotable_anchors_and_real_routes() -> None:
    equation = build_ddm_describe_line_rate_distortion_bracket_v1()
    assert equation.equation_id == EQUATION_ID
    assert len(equation.empirical_anchors) == 6
    assert equation.domain_of_validity["formalization_status"].endswith("G4_STATIC_PROJECTION")
    phase = equation.domain_of_validity["v13_g1_worldsheet_upgrade"]["raw_q8_phase_ablation"]
    assert phase["status"] == "MEASURED_RECEIVER_CLOSED_LANE_HELP_TOTAL_HARM"
    assert phase["n600_total_dseg_delta"] == pytest.approx(0.000280736287)
    assert phase["n600_lane_conditional_dseg_delta"] == pytest.approx(-0.029228004790)
    g3 = equation.domain_of_validity["v13_g1_worldsheet_upgrade"]["g3_allocation_policy"]
    assert g3["debt_shape"] == "BROAD_NOT_HEAVY_TAILED"
    assert g3["per_pair_topk_correction_allowed"] is False
    assert g3["allocation_form"] == "AMORTIZED_SHARED_GRAMMAR_TEMPLATES_PROCESS_PRIORS"
    assert g3["top24_policy"] == "SCREENING_ONLY_FULL_N600_VERDICT_REQUIRED"
    assert g3["event_proxy_pairs"] == [279, 286, 452]
    v14 = equation.domain_of_validity["v14_receiver_realization_upgrade"]
    assert v14["horizon_joint_score_delta"] == pytest.approx(-0.005003006483)
    assert v14["static_cell_forecast_status"] == "ONE_OF_THREE_POSITIVE_TRANSFER_BADLY_MISCALIBRATED"
    assert v14["ar1_bev_status"] == "BLOCKED_NO_DECODER_FREE_PHYSICAL_BEV_CUSTODY"
    assert equation.domain_of_validity["promotion_eligible"] is False
    assert equation.provenance.score_claim_valid is False
    assert equation.canonical_consumers == (
        "tac.optimization.v10_constructive_solver",
        "tac.optimization.direct_description_entropy_priced_member",
        "tac.witness_control.costate_organ_v2",
    )
    assert equation.canonical_producers == (
        "tools.run_direct_description_entropy_priced_member",
        "tools.run_ddm_v9_carrier_compose",
        "tools.measure_ddm_v13_lane_phase_ablation",
        "tools.measure_ddm_v14_realization_fidelity",
        "tools.measure_ddm_v14_g4_receiver_projection",
    )
    for anchor in equation.empirical_anchors:
        assert anchor.predicted_output["verdict"] == anchor.empirical_output["verdict"]
        assert anchor.empirical_output["receipt_sha256_bindings_verified"] is True
        assert anchor.provenance.promotion_eligible is False


def test_bound_cross_receipt_hashes_rederive_from_primary_artifacts() -> None:
    equation = build_ddm_describe_line_rate_distortion_bracket_v1()
    bindings = {
        path: expected
        for anchor in equation.empirical_anchors
        for path, expected in anchor.inputs["receipt_sha256_bindings"].items()
    }
    assert len(bindings) == 17
    assert {path: _sha256(REPO / path) for path in bindings} == bindings
    assert {
        V7_CROSS_SHA256,
        V8_CROSS_SHA256,
        V9_CROSS_SHA256,
        V12_N600_SHA256,
        V13_N600_SHA256,
        V13_PHASE_N64_SHA256,
        V13_PHASE_N600_SHA256,
        V14_N600_SHA256,
        V14_G4_SHA256,
    }.issubset(bindings.values())


def test_population_round_trips_through_isolated_locked_registry(tmp_path: Path) -> None:
    registry = tmp_path / "canonical_equations.jsonl"
    lock = tmp_path / "canonical_equations.jsonl.lock"
    populated = populate_ddm_describe_line_rate_distortion_bracket_v1(
        path=registry,
        lock_path=lock,
        agent="pytest",
        subagent_id="ddm_structured_carriers_law_registration",
    )
    rows = [json.loads(line) for line in registry.read_text().splitlines() if line]
    loaded = query_equations(path=registry)
    assert populated.equation_id == EQUATION_ID
    assert [equation.equation_id for equation in loaded] == [EQUATION_ID]
    assert len(loaded[0].empirical_anchors) == 6
    assert rows[0]["event_type"] == "registered"
    assert "MAIN landing review required" in rows[0]["notes"]
