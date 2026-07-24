from __future__ import annotations

import copy

import numpy as np
import pytest

from tac.optimization.ddm_min_description_contract import (
    LayerHome,
    StreamType,
    TypedStreamTag,
)
from tac.optimization.ddm_pf2_bucket_assignment import (
    ASSIGNMENT_ROW_SCHEMA,
    ASSIGNMENT_TABLE_SCHEMA,
    PARTIAL_MEASUREMENT_STATUS,
    PROBE_RESULT_SCHEMA,
    RECOVERED_STATUS,
    UNRECOVERABLE_STATUS,
    PF2BucketAssignmentError,
    build_measured_assignment_table,
    canonical_sha256,
    intersect_argmax_delta_with_pf2_events,
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
        "foreign_key_vocabulary": {
            "receiver_actuator_stable_ids": ["j2.island.track0.center_x"],
            "direction_ids": [
                "NEGATIVE_ONE_QUANTUM",
                "POSITIVE_ONE_QUANTUM",
            ],
            "exact_join_row_count": 0,
        },
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


def test_exact_argmax_intersection_never_assigns_nearby_pf2_events() -> None:
    before = np.zeros((1, 384, 512), dtype=np.uint8)
    after = before.copy()
    after[0, 7, 11] = 1
    changed = 3 * 384 * 512 + 7 * 512 + 11
    nearby = 3 * 384 * 512 + 7 * 512 + 12
    hits = intersect_argmax_delta_with_pf2_events(
        pair_ids=[3],
        baseline_cells=before,
        perturbed_cells=after,
        bucket_event_ids={
            "exact": np.asarray([changed], dtype=np.uint32),
            "nearby": np.asarray([nearby], dtype=np.uint32),
        },
    )
    assert set(hits) == {"exact"}
    assert hits["exact"]["pair_ids"] == [3]
    assert hits["exact"]["event_ids"].tolist() == [changed]


def test_exact_argmax_intersection_rejects_invalid_event_vectors() -> None:
    cells = np.zeros((1, 384, 512), dtype=np.uint8)
    with pytest.raises(PF2BucketAssignmentError, match="unsigned vectors"):
        intersect_argmax_delta_with_pf2_events(
            pair_ids=[0],
            baseline_cells=cells,
            perturbed_cells=cells,
            bucket_event_ids={"duplicate": np.asarray([3, 3], dtype=np.uint32)},
        )
    with pytest.raises(PF2BucketAssignmentError, match="unsigned vectors"):
        intersect_argmax_delta_with_pf2_events(
            pair_ids=[0],
            baseline_cells=cells,
            perturbed_cells=cells,
            bucket_event_ids={"outside": np.asarray([600 * 384 * 512], dtype=np.uint32)},
        )


def test_measured_table_preserves_partial_probe_custody_and_exact_join() -> None:
    table = _table()
    event_id = 17
    probe = {
        "schema": PROBE_RESULT_SCHEMA,
        "receiver_actuator_id": "j2.island.track0.center_x",
        "direction_id": "POSITIVE_ONE_QUANTUM",
        "status": "MEASURED_ARGMAX_PERTURBATION",
        "checkpoint_sha256": "b" * 64,
        "bucket_hits": [
            {
                "bucket_id": "bucket-0000",
                "pair_ids": [0],
                "event_count": 1,
                "event_ids_sha256": __import__("hashlib")
                .sha256(np.asarray([event_id], dtype="<u4").tobytes())
                .hexdigest(),
            }
        ],
    }
    measured = build_measured_assignment_table(
        base_table=table,
        expected_pf2_sha256="a" * 64,
        probe_results=[probe],
    )
    first = measured["rows"][0]
    assert first["assignment_status"] == RECOVERED_STATUS
    assert first["pair_ids"] == [0]
    assert first["pf2_membership_pair_ids"] == [0, 17]
    assert first["receiver_actuator_ids"] == ["j2.island.track0.center_x"]
    assert measured["rows"][1]["assignment_status"] == PARTIAL_MEASUREMENT_STATUS
    assert measured["coverage"]["completed_probe_count"] == 1
    assert measured["coverage"]["required_probe_count"] == 2
    validate_assignment_table(measured, expected_pf2_sha256="a" * 64)


def test_measured_table_rejects_unknown_bucket_hit() -> None:
    table = _table()
    probe = {
        "schema": PROBE_RESULT_SCHEMA,
        "receiver_actuator_id": "j2.island.track0.center_x",
        "direction_id": "POSITIVE_ONE_QUANTUM",
        "status": "MEASURED_ARGMAX_PERTURBATION",
        "checkpoint_sha256": "b" * 64,
        "bucket_hits": [
            {
                "bucket_id": "not-a-sealed-pf2-bucket",
                "pair_ids": [0],
                "event_count": 1,
                "event_ids_sha256": "c" * 64,
            }
        ],
    }
    with pytest.raises(PF2BucketAssignmentError, match="bucket hit custody"):
        build_measured_assignment_table(
            base_table=table,
            expected_pf2_sha256="a" * 64,
            probe_results=[probe],
        )
