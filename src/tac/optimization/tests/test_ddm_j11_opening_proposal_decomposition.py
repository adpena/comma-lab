# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib

import numpy as np
import pytest

from tac.optimization.ddm_j11_opening_proposal_decomposition import (
    OBJECTIVE_GATE_CONTRADICTION_SCHEMA,
    PC1_REBASE_BLOCKER,
    POSE_ACTUATOR_BLOCKER,
    RANGE_A_BLOCKER,
    SEG_ACTUATOR_BLOCKER,
    J11ProposalDecompositionError,
    blocked_component_tables,
    build_source_preserving_pc1_adapter_archive,
    derive_authority_blockers,
    null_projector_from_full_column_rank_sketch,
    null_projector_from_receiver_gram,
    objective_gate_contradiction,
    parse_source_preserving_pc1_adapter_archive,
    receive_source_preserving_pc1_camera_pairs,
)
from tac.optimization.ddm_pc1_pose_stream import PC1PosePacketV1, make_zero_active_packet


def _pose_data() -> dict:
    return {
        "schema": "ddm_pose_metric_custody.v1",
        "metric_surface": "EXACT_POSENET_OUTPUT_MSE_QUADRATIC",
        "output_dimension": 6,
        "pair_count": 600,
        "scorer_batch_size": 32,
        "rows": [{"pair_id": pair_id} for pair_id in range(600)],
    }


def _seg_data() -> dict:
    return {
        "schema": "ddm_seg_metric_custody.direct_scorer_intrinsic.v2",
        "metric_mode": "DIRECT_SCORER_INTRINSIC_NO_ACTUATOR_INPUT",
        "head_rank": 4,
        "pair_count": 600,
        "scorer_batch_size": 32,
        "direct_blocks": [
            {
                "secant_status": "NOT_APPLICABLE_DIRECT_SCORER_INTRINSIC_NO_ACTUATOR",
            }
        ],
    }


def _pc2_receipt() -> dict:
    return {
        "schema": "ddm_pc2_pose_descent_smoke_receipt.v1",
        "parent": {
            "archive_sha256": ("2a2c0367150f8c8c0953dfb5c1485e238bbc9995c37385e149e52ae22f506241"),
            "archive_bytes": 138813,
        },
        "verdict": "PC1_DESCENT_MEASURED_NET_JOINT_NEGATIVE",
        "exact_verdicts": {
            "0": {
                "archive_bytes": 139547,
                "d_seg": 0.02491527133517795,
                "d_pose": 163.04531226928225,
            }
        },
    }


def _baseline() -> dict:
    return {
        "archive_bytes": 138813,
        "d_seg": 0.06974277072482639,
        "d_pose": 35.49982080959101,
    }


def _packet(*, nonzero: bool = False) -> PC1PosePacketV1:
    q_xi = np.zeros((2, 6), dtype=np.int16)
    if nonzero:
        q_xi[0, 0] = 1
    return PC1PosePacketV1(
        active=True,
        pair_count=2,
        xi_scales=(0.001,) * 6,
        residual_scale=0.25,
        q_xi=q_xi,
        q_luma_phase=np.zeros((2, 4), dtype=np.int8),
    )


def test_complete_output_metrics_do_not_imply_receiver_null_projectors() -> None:
    blockers, rebase = derive_authority_blockers(
        pose_data=_pose_data(),
        seg_data=_seg_data(),
        range_a_source=("Project camera-space ``frames`` onto ``range(A)``; invariant A(PX)=A(X)"),
        pc2_receipt=_pc2_receipt(),
        source_baseline=_baseline(),
    )
    assert blockers == (
        POSE_ACTUATOR_BLOCKER,
        SEG_ACTUATOR_BLOCKER,
        RANGE_A_BLOCKER,
        PC1_REBASE_BLOCKER,
    )
    assert rebase["source_preserving"] is False
    assert rebase["delta_archive_bytes"] == 734
    assert rebase["delta_d_pose"] > 127.0


def test_missing_candidate_prices_remain_null_not_zero() -> None:
    singles, pairs = blocked_component_tables(
        ("worldsheet_joint_active_x_+1",),
        blockers=(POSE_ACTUATOR_BLOCKER,),
    )
    assert len(singles) == 2
    assert len(pairs) == 1
    assert all(row["materialized"] is False for row in (*singles, *pairs))
    assert all(row["joint_delta"] is None for row in (*singles, *pairs))
    assert all(row["n600_evidence"] is False for row in (*singles, *pairs))


def test_pc1_rebase_blocker_drops_only_when_active_zero_is_exactly_preserving() -> None:
    pc2 = _pc2_receipt()
    pc2["exact_verdicts"]["0"] = {
        "archive_bytes": 138813,
        "d_seg": 0.06974277072482639,
        "d_pose": 35.49982080959101,
    }
    blockers, rebase = derive_authority_blockers(
        pose_data=_pose_data(),
        seg_data=_seg_data(),
        range_a_source=("Project camera-space ``frames`` onto ``range(A)``; invariant A(PX)=A(X)"),
        pc2_receipt=pc2,
        source_baseline=_baseline(),
    )
    assert PC1_REBASE_BLOCKER not in blockers
    assert rebase["source_preserving"] is True


def test_pc1_source_preserving_adapter_has_literal_active_zero_archive_identity() -> None:
    parent = b"exact-j10-parent-archive-bytes"
    digest = hashlib.sha256(parent).hexdigest()
    zero = make_zero_active_packet(_packet(nonzero=True))
    adapted = build_source_preserving_pc1_adapter_archive(
        parent_archive=parent,
        parent_sha256=digest,
        packet=zero,
    )
    assert adapted is parent
    parsed_parent, parsed_packet, manifest = parse_source_preserving_pc1_adapter_archive(
        adapted,
        expected_parent_archive=parent,
        expected_parent_sha256=digest,
        zero_home_packet=zero,
    )
    assert parsed_parent == parent
    assert np.count_nonzero(parsed_packet.q_xi) == 0
    assert manifest["equation_id"] == "identity_at_active_zero"
    with pytest.raises(J11ProposalDecompositionError, match="packet type"):
        build_source_preserving_pc1_adapter_archive(
            parent_archive=parent,
            parent_sha256=digest,
            packet=type("MalformedInactivePacket", (), {"active": False})(),
        )


def test_pc1_source_preserving_nonzero_adapter_parseback_is_exact() -> None:
    parent = b"exact-j10-parent-archive-bytes"
    digest = hashlib.sha256(parent).hexdigest()
    packet = _packet(nonzero=True)
    adapted = build_source_preserving_pc1_adapter_archive(
        parent_archive=parent,
        parent_sha256=digest,
        packet=packet,
    )
    assert adapted != parent
    parsed_parent, parsed_packet, manifest = parse_source_preserving_pc1_adapter_archive(
        adapted,
        expected_parent_archive=parent,
        expected_parent_sha256=digest,
        zero_home_packet=make_zero_active_packet(packet),
    )
    assert parsed_parent == parent
    assert np.array_equal(parsed_packet.q_xi, packet.q_xi)
    assert manifest["equation_id"] == "parent_plus_pc1_packet_minus_pc1_active_zero"
    assert (
        build_source_preserving_pc1_adapter_archive(
            parent_archive=parsed_parent,
            parent_sha256=digest,
            packet=parsed_packet,
        )
        == adapted
    )


def test_pc1_source_preserving_receiver_active_zero_is_byte_exact() -> None:
    parent = np.arange(2 * 874 * 1164 * 3, dtype=np.uint8).reshape(1, 2, 874, 1164, 3)
    result = receive_source_preserving_pc1_camera_pairs(
        parent_camera=parent,
        packet=make_zero_active_packet(_packet(nonzero=True)),
        pair_ids=(0,),
    )
    assert np.array_equal(result, parent)


def test_full_rank_sketch_certifies_exact_zero_null_projector() -> None:
    projector, certificate = null_projector_from_full_column_rank_sketch(
        np.asarray([[2.0, 0.0], [0.0, 3.0], [1.0, 1.0]]),
        coordinate_ids=("receiver:0", "receiver:1"),
        sketch_id="fixed_linear_sketch_v1",
    )
    assert np.array_equal(projector, np.zeros((2, 2)))
    assert certificate["certified_rank"] == 2
    assert certificate["full_column_rank"] is True


def test_rank_deficient_sketch_cannot_manufacture_a_null_projector() -> None:
    with pytest.raises(ValueError, match="rank-deficient sketch"):
        null_projector_from_full_column_rank_sketch(
            np.asarray([[1.0, 1.0], [2.0, 2.0]]),
            coordinate_ids=("receiver:0", "receiver:1"),
            sketch_id="fixed_linear_sketch_v1",
        )
    with pytest.raises(ValueError, match="custody differs"):
        null_projector_from_full_column_rank_sketch(
            np.asarray([[1.0]]),
            coordinate_ids=("",),
            sketch_id="fixed_linear_sketch_v1",
        )


def test_complete_receiver_gram_authorizes_nontrivial_null_projector() -> None:
    projector, certificate = null_projector_from_receiver_gram(
        np.asarray([[4.0, 0.0], [0.0, 0.0]]),
        coordinate_ids=("receiver:0", "receiver:1"),
        jacobian_id="complete_receiver_jacobian",
    )
    assert np.array_equal(projector, np.asarray([[0.0, 0.0], [0.0, 1.0]]))
    assert certificate["certified_rank"] == 1
    assert certificate["nullity"] == 1


def test_auxiliary_gate_disagreement_is_typed_without_overriding_pure_objective() -> None:
    row = objective_gate_contradiction(
        candidate_id="candidate",
        pure_priced_joint_delta=-0.25,
        auxiliary_gate_id="seg_only_guard",
        auxiliary_gate_admitted=False,
    )
    assert row is not None
    assert row["schema"] == OBJECTIVE_GATE_CONTRADICTION_SCHEMA
    assert row["pure_priced_admitted"] is True
    assert row["authority_effect"] == "NONE_AUXILIARY_GATE_CANNOT_OVERRIDE_REALIZED_JOINT_DELTA"
    with pytest.raises(
        J11ProposalDecompositionError,
        match="contradiction inputs",
    ):
        objective_gate_contradiction(
            candidate_id="candidate",
            pure_priced_joint_delta=-0.25,
            auxiliary_gate_id="seg_only_guard",
            auxiliary_gate_admitted=1,  # type: ignore[arg-type]
        )
