# SPDX-License-Identifier: MIT
from __future__ import annotations

import pytest

from tools.measure_ddm_cb1_perclass_carrier_byteclose import (
    CLASS_ORDER,
    CB1Error,
    _delta_row,
    _self_detected_hood_class,
)


def _measurement(
    *,
    candidate_id: str,
    archive_bytes: int,
    d_seg: float,
    d_pose: float,
    objective: float,
    errors: int,
) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "archive_bytes": archive_bytes,
        "d_seg": d_seg,
        "d_pose": d_pose,
        "advisory_objective": objective,
        "per_class": {
            name: {
                "sites": 100,
                "errors_after": errors,
                "d_seg_after": errors / 100,
            }
            for name in CLASS_ORDER
        },
    }


def test_cb1_hood_class_is_rederived_from_spatial_evidence() -> None:
    receipt = {
        "support_derivation": {
            "detected_class_id": 3,
            "class_evidence": [
                {
                    "class_id": index,
                    "bottom_share": 0.9 if index == 3 else 0.1,
                    "static_iou": 0.8 if index == 3 else 0.2,
                }
                for index in range(len(CLASS_ORDER))
            ],
        }
    }
    assert _self_detected_hood_class(receipt) == 3


def test_cb1_hood_class_uses_mc1_product_law() -> None:
    receipt = {
        "support_derivation": {
            "detected_class_id": 2,
            "class_evidence": [
                {"class_id": 0, "bottom_share": 0.90, "static_iou": 0.10},
                {"class_id": 1, "bottom_share": 0.60, "static_iou": 0.60},
                {"class_id": 2, "bottom_share": 0.50, "static_iou": 0.90},
                {"class_id": 3, "bottom_share": 0.10, "static_iou": 0.10},
                {"class_id": 4, "bottom_share": 0.05, "static_iou": 0.05},
            ],
        }
    }
    assert _self_detected_hood_class(receipt) == 2


def test_cb1_hood_class_refuses_nonunique_product_winner() -> None:
    receipt = {
        "support_derivation": {
            "detected_class_id": 0,
            "class_evidence": [
                {
                    "class_id": index,
                    "bottom_share": 0.5 if index < 2 else 0.1,
                    "static_iou": 0.5 if index < 2 else 0.1,
                }
                for index in range(len(CLASS_ORDER))
            ],
        }
    }
    with pytest.raises(CB1Error, match="unique winner"):
        _self_detected_hood_class(receipt)


def test_cb1_c1_row_admits_only_strict_negative_joint_delta() -> None:
    control = _measurement(
        candidate_id="control",
        archive_bytes=100,
        d_seg=0.2,
        d_pose=0.3,
        objective=4.0,
        errors=20,
    )
    winner = _measurement(
        candidate_id="winner",
        archive_bytes=120,
        d_seg=0.19,
        d_pose=0.29,
        objective=3.5,
        errors=19,
    )
    row = _delta_row(candidate=winner, control=control, metadata={"kind": "test"})
    assert row["incremental_archive_bytes"] == 20
    assert row["delta_joint_s"] == -0.5
    assert row["waterfill_eligible"] is True
    assert row["uphill_reason"] is None

    loser = dict(winner)
    loser.update(
        {
            "candidate_id": "loser",
            "advisory_objective": 4.5,
            "d_pose": 0.5,
        }
    )
    rejected = _delta_row(
        candidate=loser,
        control=control,
        metadata={"kind": "test"},
    )
    assert rejected["waterfill_eligible"] is False
    assert rejected["uphill_reason"]["primary_leg"] == "pose_survival"
