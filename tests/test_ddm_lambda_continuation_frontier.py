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
    CodedStream,
    LambdaContinuationError,
    MeasuredDescription,
    continuation_rows,
    discrete_dual_rows,
    geometric_curvature_ladder,
    lower_supported_hull,
    normalized_knee,
    publish_immutable_json,
)

REPO = Path(__file__).resolve().parents[1]
RECEIPT = (
    REPO / ".omx/research/ddm_rd1_lambda_continuation_frontier_20260724T011239Z/"
    "ddm_rd1_lambda_continuation_frontier_receipt_v2.json"
)
TYPED_FRONTIER = (
    REPO / ".omx/research/ddm_rd1_lambda_continuation_frontier_20260724T011239Z/"
    "typed_R_D_frontier_rows.json"
)
TYPED_FRONTIER_SHA256 = "7266153f1984e220e69fa8fe04ed674a070cfffe3f176524e7c2985d212c2b1c"


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
    assert len(receipt["supported_hull"]) == 4
    assert len(receipt["continuation"]) == 10
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


def test_typed_rate_distortion_frontier_is_composed_and_false_authority() -> None:
    payload = TYPED_FRONTIER.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == TYPED_FRONTIER_SHA256
    frontier = json.loads(payload)
    assert frontier["source_receipt_sha256"] == RECEIPT_SHA256
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
