from __future__ import annotations

import copy

import pytest

from tac.optimization.ddm_min_description_contract import (
    LayerHome,
    StreamType,
    TypedStreamTag,
)
from tac.optimization.ddm_pf2_bucket_assignment import (
    ASSIGNMENT_ROW_SCHEMA,
    ASSIGNMENT_TABLE_SCHEMA,
    UNRECOVERABLE_STATUS,
    PF2BucketAssignmentError,
    canonical_sha256,
    validate_assignment_table,
)


def _table() -> dict:
    tag = TypedStreamTag(
        type=StreamType.SKELETON,
        layer_home=LayerHome.L1_PROGRAM,
        evaluate_py_recursion_level_cited="L1_program assignment -> L4_scorer_feature",
        counted_bytes=0,
        free_receiver_code=True,
    ).to_dict()
    rows = [
        {
            "schema": ASSIGNMENT_ROW_SCHEMA,
            "bucket_id": f"bucket-{index:04d}",
            "atlas_key": {
                "class_pair": "Road--Lane",
                "class_ids": [0, 1],
                "class_stratum": "cell",
                "visibility": "seg-visible",
                "g4_temporal_class": "STATIC_IN_IMAGE",
                "representation_type": "SKELETON",
            },
            "pair_ids": [] if index else [0, 17],
            "assignment_status": UNRECOVERABLE_STATUS,
            "receiver_actuator_ids": [],
            "direction_ids": [],
            "typed_stream_tag": tag,
        }
        for index in range(1200)
    ]
    result = {
        "schema": ASSIGNMENT_TABLE_SCHEMA,
        "pf2_receipt_sha256": "a" * 64,
        "rows": rows,
    }
    result["table_content_sha256"] = canonical_sha256(result)
    return result


def test_assignment_table_accepts_recovered_membership_with_lost_foreign_key() -> None:
    value = _table()
    validate_assignment_table(value, expected_pf2_sha256="a" * 64)


def test_assignment_table_rejects_pair_order_and_hash_drift() -> None:
    value = _table()
    value["rows"][0]["pair_ids"] = [17, 0]
    value["table_content_sha256"] = canonical_sha256(
        {key: item for key, item in value.items() if key != "table_content_sha256"}
    )
    with pytest.raises(PF2BucketAssignmentError, match="sorted unique"):
        validate_assignment_table(value, expected_pf2_sha256="a" * 64)

    drifted = copy.deepcopy(_table())
    drifted["rows"][1]["bucket_id"] = "changed"
    with pytest.raises(PF2BucketAssignmentError, match="content SHA"):
        validate_assignment_table(drifted, expected_pf2_sha256="a" * 64)
