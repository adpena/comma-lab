# SPDX-License-Identifier: MIT
from __future__ import annotations

from tac.optimization.ddm_j11_opening_proposal_decomposition import (
    PC1_REBASE_BLOCKER,
    POSE_ACTUATOR_BLOCKER,
    RANGE_A_BLOCKER,
    SEG_ACTUATOR_BLOCKER,
    blocked_component_tables,
    derive_authority_blockers,
)


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
