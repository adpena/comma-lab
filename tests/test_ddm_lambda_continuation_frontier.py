# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from tac.canonical_equations.ddm_lambda_continuation_frontier_20260724 import (
    EQUATION_ID,
    RECEIPT_SHA256,
    build_ddm_restricted_realized_lambda_continuation_v1,
)
from tac.canonical_equations.evaluators import has_evaluator, resolve_equation_value
from tac.optimization.ddm_lambda_continuation_frontier import (
    G4_TEMPORAL_CLASSES,
    SCORER_VISIBILITIES,
    SEMANTIC_STRATA,
    CodedStream,
    LambdaContinuationError,
    MeasuredDescription,
    continuation_rows,
    discrete_dual_rows,
    effective_quantum_D,
    geometric_curvature_ladder,
    lower_supported_hull,
    metric_active_continuation_geometry_report,
    normalized_knee,
    publish_immutable_json,
    second_order_metric_geometry_addendum_report,
    typed_dimension_dual_report,
)

REPO = Path(__file__).resolve().parents[1]
RECEIPT = (
    REPO / ".omx/research/ddm_rd1_lambda_continuation_frontier_20260724T011239Z/"
    "ddm_rd1_lambda_continuation_frontier_receipt_v5.json"
)
TYPED_FRONTIER = (
    REPO / ".omx/research/ddm_rd1_lambda_continuation_frontier_20260724T011239Z/"
    "typed_R_D_frontier_rows_v5.json"
)
TYPED_FRONTIER_SHA256 = "96058098a7fabd568d725005c1d5c301bb862f2780616ce04cb4c379ee28e824"
SOURCE_RECEIPT = (
    REPO / ".omx/research/ddm_rd1_lambda_continuation_frontier_20260724T011239Z/"
    "ddm_rd1_lambda_continuation_frontier_receipt_v2.json"
)
DIMENSION_SUPPLEMENT = (
    REPO / ".omx/research/ddm_rd1_lambda_continuation_frontier_20260724T011239Z/"
    "typed_dimension_duals_effective_quantum.json"
)
DIMENSION_SUPPLEMENT_SHA256 = (
    "c92bb7d4c863bae53f87a228cbbe79938966ff4b645e6c631be07e4bd574e2a0"
)
METRIC_SUPPLEMENT = (
    REPO / ".omx/research/ddm_rd1_lambda_continuation_frontier_20260724T011239Z/"
    "metric_active_continuation_geometry.json"
)
METRIC_SUPPLEMENT_SHA256 = (
    "8cfd85fe1ce90e4c59557caab2ba1912c949deb58b0f2c6064d2648b5b4ce6f6"
)
SECOND_ORDER_SUPPLEMENT = (
    REPO / ".omx/research/ddm_rd1_lambda_continuation_frontier_20260724T011239Z/"
    "metric_active_second_order_geometry.json"
)
SECOND_ORDER_SUPPLEMENT_SHA256 = (
    "2eb5b5e9aca12fc4c936bc1a915f0c65201b47f42a3766e547bf08d79e2d6292"
)
SOURCE_CONTINUATION_SHA256 = (
    "0a053b77c6f1ff396a787ccfa49f5c166600e600fe362fda934ad4b8f1da656b"
)


def _stream(candidate_id: str, counted_bytes: int) -> tuple[CodedStream, ...]:
    return (
        CodedStream(
            stream_id=f"{candidate_id}/skeleton",
            stratum="semantic",
            factor_kind="skeleton",
            custody_role="stored_problem",
            counted_bytes=1,
            sha256="a" * 64,
            codec="TEST",
            source_path=f"{candidate_id}.bin",
        ),
        CodedStream(
            stream_id=f"{candidate_id}/fiber",
            stratum="semantic",
            factor_kind="fiber",
            custody_role="solve_exception",
            counted_bytes=counted_bytes - 1,
            sha256="b" * 64,
            codec="TEST",
            source_path=f"{candidate_id}.bin",
        ),
    )


def _point(candidate_id: str, counted_bytes: int, distortion: float) -> MeasuredDescription:
    return MeasuredDescription(
        candidate_id=candidate_id,
        counted_bytes=counted_bytes,
        d_seg=distortion / 100.0,
        d_pose=0.0,
        coded_streams=_stream(candidate_id, counted_bytes),
        source_artifact=f"{candidate_id}.json",
        source_sha256="c" * 64,
        receiver_closure="archive_receiver_closed",
    )


def _toy_domain() -> tuple[MeasuredDescription, ...]:
    return (
        _point("describe_supported", 100, 10.0),
        _point("current_unsupported", 110, 9.8),
        _point("scalar", 116, 9.0),
        _point("composed", 200, 5.0),
        _point("exact", 1000, 0.0),
    )


def test_two_type_stream_partition_and_donor_firewall() -> None:
    point = _point("typed", 20, 2.0)
    assert point.skeleton_bytes == 1
    assert point.fiber_bytes == 19
    assert point.skeleton_bytes + point.fiber_bytes == point.counted_bytes
    with pytest.raises(LambdaContinuationError, match="donor-conditioned"):
        MeasuredDescription(
            candidate_id="donor",
            counted_bytes=20,
            d_seg=0.1,
            d_pose=0.0,
            coded_streams=_stream("donor", 20),
            source_artifact="donor.json",
            source_sha256="d" * 64,
            receiver_closure="archive_receiver_closed",
            donor_conditioned=True,
        )


def test_neighbor_continuation_reaches_full_rank_and_marks_unsupported_control() -> None:
    domain = _toy_domain()
    hull = lower_supported_hull(domain)
    assert [row.candidate_id for row in hull] == [
        "describe_supported",
        "scalar",
        "composed",
        "exact",
    ]
    ladder = geometric_curvature_ladder(hull)
    assert 8 <= len(ladder) <= 12
    rows = continuation_rows(
        domain,
        ladder,
        seed_candidate_id="current_unsupported",
    )
    assert rows[0]["corrector_path"] == [
        "current_unsupported",
        "describe_supported",
    ]
    assert all(row["neighbor_only"] for row in rows)
    assert all(row["restricted_global_rank_verified"] for row in rows)
    assert all(len(row["full_rank_candidate_ids"]) == len(domain) for row in rows)
    assert rows[-1]["selected_candidate_id"] == "exact"
    assert normalized_knee(hull).candidate_id == "composed"
    assert len(discrete_dual_rows(hull)) == 3


def test_dimension_duals_fail_closed_without_joint_g4_rate_home() -> None:
    hull = lower_supported_hull(_toy_domain())
    report = typed_dimension_dual_report(hull)
    expected_per_edge = (
        (len(SEMANTIC_STRATA) + 1)
        * len(SCORER_VISIBILITIES)
        * len(G4_TEMPORAL_CLASSES)
    )
    assert len(report["bucket_rows"]) == (len(hull) - 1) * expected_per_edge
    assert report["actionable_bucket_count"] == 0
    assert {
        (
            row["dual_index"],
            row["stratum"],
            row["scorer_visibility"],
            row["g4_temporal_class"],
        )
        for row in report["bucket_rows"]
    } == {
        (edge, stratum, visibility, temporal_class)
        for edge in range(1, len(hull))
        for stratum in (*SEMANTIC_STRATA, "POSE6_GLOBAL")
        for visibility in SCORER_VISIBILITIES
        for temporal_class in G4_TEMPORAL_CLASSES
    }
    assert all(
        row["lambda_bytes_per_D_dimension"] is None
        and row["actionable_for_train_decision"] is False
        for row in report["bucket_rows"]
    )
    assert all(
        edge["additivity_residual_D"] == pytest.approx(0.0, abs=1e-12)
        and edge["aggregate_lambda_actionable_for_train_decision"] is False
        for edge in report["edge_summaries"]
    )


def test_effective_quantum_has_typed_units_and_rejects_invalid_inputs() -> None:
    assert effective_quantum_D(
        uint8_step=2.0,
        scorer_sensitivity_D_per_uint8_step=0.125,
    ) == pytest.approx(0.25)
    with pytest.raises(LambdaContinuationError):
        effective_quantum_D(
            uint8_step=0.0,
            scorer_sensitivity_D_per_uint8_step=0.125,
        )
    with pytest.raises(LambdaContinuationError):
        effective_quantum_D(
            uint8_step=1.0,
            scorer_sensitivity_D_per_uint8_step=-0.125,
        )


def test_metric_geometry_keeps_discrete_graph_valid_and_l2_control_only() -> None:
    rows = continuation_rows(
        _toy_domain(),
        geometric_curvature_ladder(lower_supported_hull(_toy_domain())),
        seed_candidate_id="current_unsupported",
    )
    report = metric_active_continuation_geometry_report(rows)
    discrete = report["completed_discrete_solve"]
    continuous = report["continuous_proposal_geometry_contract"]
    assert discrete["selection_geometry"] == "ORDERED_FINITE_NEIGHBOR_GRAPH"
    assert discrete["identity_L2_used"] is False
    assert discrete["state_space_distance_used"] is False
    assert continuous["seg"]["identity_L2_allowed"] == "LABELED_CONTROL_ONLY"
    assert continuous["pose"]["parameter_L2_allowed"] is False
    assert continuous["distribution"]["metric"] == "Bregman divergence"
    assert continuous["composite_readback"]["single_metric_verdict_allowed"] is False
    assert continuous["actionable"] is False


def test_second_order_addendum_uses_scorer_coordinates_and_optimal_first() -> None:
    rows = continuation_rows(
        _toy_domain(),
        geometric_curvature_ladder(lower_supported_hull(_toy_domain())),
        seed_candidate_id="current_unsupported",
    )
    base = metric_active_continuation_geometry_report(rows)
    report = second_order_metric_geometry_addendum_report(base)
    discrete = report["completed_discrete_solve"]
    future = report["future_continuous_move_contract"]
    assert discrete["continuous_approximation_order_used"] is None
    assert discrete["gradient_first_order_naive_used"] is False
    assert future["second_order_from_step_one_when_quadratic_measured"] is True
    assert [row["catalog"] for row in future["second_order_inventory"]] == [
        391,
        423,
        552,
    ]
    assert "rank-4" in future["coordinate_system"]["seg"]
    assert future["coordinate_system"]["parameter_coordinates_allowed"] is False
    assert future["ladder_order"][-1].endswith("labeled controls")
    assert future["actionable"] is False


def test_lambda_evaluator_is_import_registered_and_scope_strict() -> None:
    domain = [row.to_dict() for row in _toy_domain()]
    assert has_evaluator(EQUATION_ID)
    assert (
        resolve_equation_value(
            EQUATION_ID,
            {"lambda": 0.0, "candidates": domain},
        )
        == "describe_supported"
    )
    domain[0]["score_claim"] = True
    with pytest.raises(ValueError, match="false-authority"):
        resolve_equation_value(
            EQUATION_ID,
            {"lambda": 0.0, "candidates": domain},
        )


def test_immutable_checkpoint_refuses_drift(tmp_path: Path) -> None:
    path = tmp_path / "lambda_00.json"
    publish_immutable_json(path, {"lambda": 0.0, "selected": "a"})
    publish_immutable_json(path, {"lambda": 0.0, "selected": "a"})
    with pytest.raises(LambdaContinuationError, match="immutable checkpoint differs"):
        publish_immutable_json(path, {"lambda": 0.0, "selected": "b"})


def test_landed_receipt_and_knee_bundle_custody() -> None:
    payload = RECEIPT.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == RECEIPT_SHA256
    receipt = json.loads(payload)
    assert receipt["objective"]["restricted_domain_candidate_count"] == 110
    assert receipt["objective"]["global_uint8_lattice_optimality_claim"] is False
    assert receipt["lambda_points_reused_without_restart"] is True
    assert receipt["source_continuation_sha256"] == SOURCE_CONTINUATION_SHA256
    assert len(receipt["supported_hull"]) == 4
    assert len(receipt["continuation"]) == 10
    assert receipt["pooled_dual_status"] == (
        "VALID_SCALARIZATION_CONTROL_SUPERSEDED_FOR_TRAIN_DECISION_PRICING"
    )
    assert len(receipt["duals"]) == 162
    assert len(
        {
            (
                row["dual_index"],
                row["stratum"],
                row["scorer_visibility"],
                row["g4_temporal_class"],
            )
            for row in receipt["duals"]
        }
    ) == 162
    assert all(
        row["lambda_bytes_per_D_dimension"] is None
        and row["actionable_for_train_decision"] is False
        for row in receipt["duals"]
    )
    assert receipt["effective_quantum_tolerance"]["uniform_tolerance_allowed"] is False
    assert receipt["effective_quantum_tolerance"]["priced_bucket_count"] == 0
    assert all(
        row["effective_quantum_D"] is None
        for row in receipt["effective_quantum_tolerance"]["bucket_rows"]
    )
    assert all(
        row["additivity_residual_D"] == pytest.approx(0.0, abs=1e-12)
        for row in receipt["dimension_dual_edge_summaries"]
    )
    assert any(
        row["utc"] == "2026-07-24T02:04:16Z"
        for row in receipt["directives_consumed"]
    )
    assert any(
        row["utc"] == "2026-07-24T02:27:12Z"
        for row in receipt["directives_consumed"]
    )
    assert any(
        row["utc"] == "2026-07-24T02:28:21Z"
        for row in receipt["directives_consumed"]
    )
    metric_geometry = receipt["metric_active_continuation_geometry"]
    assert metric_geometry["completed_discrete_solve"]["identity_L2_used"] is False
    assert metric_geometry["completed_discrete_solve"]["status"] == (
        "VALID_NO_CONTINUOUS_GEOMETRY_INVOKED"
    )
    assert (
        metric_geometry["continuous_proposal_geometry_contract"]["actionable"]
        is False
    )
    assert metric_geometry["continuous_proposal_geometry_contract"]["seg"][
        "identity_L2_allowed"
    ] == "LABELED_CONTROL_ONLY"
    second_order = receipt["second_order_metric_geometry_addendum"]
    future_geometry = second_order["future_continuous_move_contract"]
    assert future_geometry["second_order_from_step_one_when_quadratic_measured"] is True
    assert future_geometry["coordinate_system"]["parameter_coordinates_allowed"] is False
    assert future_geometry["ladder_order"][-1].endswith("labeled controls")
    assert future_geometry["actionable"] is False
    assert receipt["anchors"]["describe_line_control"]["scalarizable_supported_point"] is False
    cross_axis = receipt["anchors"]["lambda_infinity_exact"]["preserved_exact_row_cross_axis_check"]
    assert cross_axis["status"] == "MEASURED_AXIS_OR_BATCH_GEOMETRY_DRIFT"
    assert receipt["knee"]["candidate_id"] == "statistics_hard_analytic_composed_frame1"
    assert receipt["knee"]["R6_CANDIDATE"] is False
    storage = receipt["storage_preflight"]
    assert storage["status"] == "PASS"
    assert storage["observed_free_bytes_at_least"] >= storage["required_free_bytes"]
    assert "observed_free_bytes" not in storage
    for candidate in receipt["candidate_domain"]:
        assert candidate["skeleton_bytes"] + candidate["fiber_bytes"] == candidate["counted_bytes"]
        assert sum(stream["counted_bytes"] for stream in candidate["coded_streams"]) == candidate["counted_bytes"]
        assert all(stream["factor_kind"] in {"skeleton", "fiber"} for stream in candidate["coded_streams"])
    artifact = receipt["knee"]["full_description_artifact"]
    bundle = REPO / artifact["path"]
    assert hashlib.sha256(bundle.read_bytes()).hexdigest() == artifact["sha256"]
    with zipfile.ZipFile(bundle) as archive:
        manifest = json.loads(archive.read("manifest.json"))
    assert manifest["logical_counted_bytes"] == receipt["knee"]["counted_bytes"]
    assert manifest["bundle_is_custody_container_not_counted_archive"] is True
    source = json.loads(SOURCE_RECEIPT.read_bytes())
    assert source["continuation"] == receipt["continuation"]
    source_continuation_bytes = json.dumps(
        source["continuation"],
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    assert hashlib.sha256(source_continuation_bytes).hexdigest() == (
        SOURCE_CONTINUATION_SHA256
    )
    assert hashlib.sha256(DIMENSION_SUPPLEMENT.read_bytes()).hexdigest() == (
        DIMENSION_SUPPLEMENT_SHA256
    )
    assert hashlib.sha256(METRIC_SUPPLEMENT.read_bytes()).hexdigest() == (
        METRIC_SUPPLEMENT_SHA256
    )
    assert hashlib.sha256(SECOND_ORDER_SUPPLEMENT.read_bytes()).hexdigest() == (
        SECOND_ORDER_SUPPLEMENT_SHA256
    )


def test_typed_rate_distortion_frontier_is_composed_and_false_authority() -> None:
    payload = TYPED_FRONTIER.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == TYPED_FRONTIER_SHA256
    frontier = json.loads(payload)
    assert frontier["source_receipt_sha256"] == RECEIPT_SHA256
    assert frontier["source_continuation_sha256"] == SOURCE_CONTINUATION_SHA256
    assert frontier["lambda_points_reused_without_restart"] is True
    assert len(frontier["rows"]) == 10
    assert frontier["pointer_moved"] is False
    for row in frontier["rows"]:
        assert row["score_claim"] is False
        assert row["pair_count"] == 600
        assert row["skeleton_bytes"] + row["fiber_bytes"] == row["counted_bytes"]
        assert row["D_realized"] == pytest.approx(
            100.0 * row["d_seg"] + (10.0 * row["d_pose"]) ** 0.5
        )
        assert row["S_composed"] == pytest.approx(
            row["D_realized"] + row["rate_term_25R"]
        )


def test_canonical_equation_builds_from_landed_anchor() -> None:
    equation = build_ddm_restricted_realized_lambda_continuation_v1()
    assert equation.equation_id == EQUATION_ID
    assert len(equation.empirical_anchors) == 1
    assert equation.predicted_vs_empirical_residual["finite_domain_full_rank_mismatch_count"] == 0.0
