from __future__ import annotations

import json
from pathlib import Path

from tac.canonical_equations.ddm_dynamic_quantum_calibration_20260724 import (
    build_dynamic_quantum_calibration,
)
from tools.run_ddm_ms7_receiver_edges import (
    _ACTUATOR,
    CONFIG,
    DDMMS7ReceiverEdgesConfigV1,
    _coordinate,
    _endpoint,
)


def test_typed_config_is_strict_and_local_only() -> None:
    config = DDMMS7ReceiverEdgesConfigV1.model_validate_json(CONFIG.read_bytes())
    assert config.scorer_threads == 4
    assert config.scorer_batch_size == 16
    assert config.research_only is True
    assert config.execution_allowed is False
    assert config.score_claim is False
    assert config.main_review_required is True
    assert Path(config.bulk_root).is_absolute()


def test_rg3_actuator_parser_is_exact() -> None:
    match = _ACTUATOR.fullmatch("rg3.class_birth.pair523.class1_2.boundary.static_in_image.band03.fine00.mag1")
    assert match is not None
    assert match.group("family") == "class_birth"
    assert match.group("pair") == "523"


def test_control_coordinate_is_dynamically_calibrated_to_one() -> None:
    direct = json.loads(
        Path(".omx/research/ddm_ms4d_direct_metric_completion_20260724T155932Z/seg_metric_direct_n600.json").read_text()
    )
    block = next(
        row
        for row in direct["direct_blocks"]
        if row["pair_id"] == 523 and row["bucket_id"] == "lane_undrivable__boundary__static_in_image"
    )
    coordinate, calibration = _coordinate(block)
    assert coordinate.signed_quanta == 1
    assert calibration["selected_k_star"] == 1
    assert coordinate.actuator_id.endswith(".mag1")


def test_v19c_integer_endpoint_is_bound() -> None:
    receipt = json.loads(
        Path(
            ".omx/research/ddm_v19c_correction_saturation_20260723T063500Z/ddm_v19c_correction_saturation_receipt.json"
        ).read_text()
    )
    errors, d_pose = _endpoint(receipt)
    assert errors == 2_923_991
    assert d_pose == 163.061210029156


def test_materialized_r0_and_pf3_receipts_close_required_edges() -> None:
    root = Path(".omx/research/ddm_ms7_receiver_edges_and_25bucket_reach_20260724T172249Z")
    r0 = json.loads((root / "r0_25_bucket_reach_table.json").read_text())
    receipt = json.loads((root / "ddm_ms7_receiver_edges_receipt.json").read_text())
    assert r0["row_count"] == 25
    assert r0["mass_paying_row_count"] == 0
    assert r0["unreachable_and_ignored_row_count"] == 25
    assert {row["verdict"] for row in r0["rows"]} == {"UNREACHABLE-AND-IGNORED"}
    assert all(
        row["reach_prices"]["R1_DYNAMIC_EXISTING_COORDINATE_BYTES"] is None
        and row["reach_prices"]["R2_T_RESIDUAL_BYTES"] is None
        for row in r0["rows"]
    )
    pf3 = receipt["pf3"]
    assert set(pf3["five_pf3_edges"]) == {
        "receiver_object_builder",
        "realized_uint8_quantum",
        "same_object_candidate_delta",
        "dimension_rate_home",
        "coder_payload_owner",
    }
    assert pf3["dynamic_quantum_calibration"]["predicted_vs_realized"]["amplitude_validity_check_passed"]
    assert pf3["coder_race"]["winner"]["codec"] == "E4_BROTLI_Q11"
    assert all(row["parseback_exact"] for row in pf3["coder_race"]["rows"] if row["available"])
    by_codec = {row["codec"]: row for row in pf3["coder_race"]["rows"]}
    assert by_codec["CONSTRICTION_ORDER1_CONTEXT_ANS"]["available"]
    assert by_codec["ZSTD19_TRAINED_DICTIONARY"]["available"]
    assert by_codec["G4_FREE_DECODER_DERIVED_SPATIAL_CONTEXT"]["framed_bytes"] is None
    assert receipt["pointer_moved"] is False


def test_canonical_equation_carries_realized_anchor() -> None:
    equation = build_dynamic_quantum_calibration()
    assert equation.equation_id == "dynamic_quantum_calibration_v1"
    assert len(equation.empirical_anchors) == 1
    anchor = equation.empirical_anchors[0]
    assert anchor.inputs["pair_id"] == 523
    assert anchor.predicted_output["selected_k_star"] == 1
    assert anchor.empirical_output["minimum_nonzero_uint8_level"] == 45
    assert anchor.residual == 0.0
