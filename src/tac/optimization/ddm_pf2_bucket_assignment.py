# SPDX-License-Identifier: MIT
"""Exact, fail-closed PF2 bucket-assignment custody.

PF2's construction can recover which source-pair flip events populated each
typed bucket.  It cannot recover a receiver actuator or signed secant: those
foreign keys were never part of the PF2 construction.  This module preserves
that distinction instead of fabricating a causal join.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any, Final

import numpy as np

from tac.optimization.ddm_g4_spatial_stationarity import (
    boundary_mask,
    transition_codes,
)
from tac.optimization.ddm_min_description_contract import (
    LayerHome,
    StreamType,
    TypedStreamTag,
)

ASSIGNMENT_TABLE_SCHEMA: Final = "ddm_ms5_pf2_bucket_assignment_table.v1"
ASSIGNMENT_ROW_SCHEMA: Final = "ddm_ms5_pf2_bucket_assignment_row.v1"
ASSIGNMENT_RECEIPT_SCHEMA: Final = "ddm_ms5_pf2_bucket_assignment_receipt.v1"
PF2_ATLAS_SCHEMA: Final = "ddm_pf2_dimension_conditioned_atlas.v1"
PF2_BUCKET_SCHEMA: Final = "ddm_pf2_typed_split_atlas_bucket.v2"
PF2_BUCKET_COUNT: Final = 1_200
PAIR_COUNT: Final = 600
HEIGHT: Final = 384
WIDTH: Final = 512
ATLAS_KEY_FIELDS: Final = (
    "class_pair",
    "class_ids",
    "class_stratum",
    "visibility",
    "g4_temporal_class",
    "representation_type",
)
RECOVERED_STATUS: Final = "RECOVERED_COMPLETE"
UNRECOVERABLE_STATUS: Final = "ASSIGNMENT_UNRECOVERABLE_PF2_CONSTRUCTION_HAS_NO_ACTUATOR_DIRECTION_FOREIGN_KEY"


class PF2BucketAssignmentError(ValueError):
    """An assignment input or claimed join is not exactly custodied."""


def canonical_bytes(value: Any) -> bytes:
    """Return deterministic JSON bytes used by all MS5 content hashes."""

    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _typed_tag(representation_type: str) -> dict[str, Any]:
    try:
        stream_type = StreamType(representation_type)
    except ValueError as exc:
        raise PF2BucketAssignmentError(
            f"PF2 representation type is outside the five-type vocabulary: {representation_type}"
        ) from exc
    layer = {
        StreamType.SKELETON: LayerHome.L1_PROGRAM,
        StreamType.CONNECTION: LayerHome.L2_CHART,
        StreamType.FIBER: LayerHome.L3_RASTER,
        StreamType.GAUGE: LayerHome.L3_RASTER,
        StreamType.RESIDUAL: LayerHome.L4_SCORER_FEATURE,
    }[stream_type]
    return TypedStreamTag(
        type=stream_type,
        layer_home=layer,
        evaluate_py_recursion_level_cited=(f"{layer.value} PF2 assignment metadata -> L4_scorer_feature measurement"),
        counted_bytes=0,
        free_receiver_code=True,
    ).to_dict()


def _validate_geometry(
    predicted: np.ndarray,
    target: np.ndarray,
    transition_counts: np.ndarray,
) -> None:
    if predicted.shape != (PAIR_COUNT, HEIGHT, WIDTH) or target.shape != predicted.shape:
        raise PF2BucketAssignmentError("PF2 predicted/target geometry differs from n600")
    if predicted.dtype != np.uint8 or target.dtype != np.uint8:
        raise PF2BucketAssignmentError("PF2 predicted/target cells must be uint8")
    if transition_counts.shape != (25, HEIGHT, WIDTH):
        raise PF2BucketAssignmentError("G4 transition-count geometry differs")


def reconstruct_temporal_masks(
    *,
    predicted: np.ndarray,
    target: np.ndarray,
    transition_counts: np.ndarray,
    xi_event_ids: Sequence[int],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Rebuild PF2's exact flip and mutually exclusive G4 temporal masks."""

    _validate_geometry(predicted, target, transition_counts)
    codes = transition_codes(predicted, target)
    flip = predicted != target
    row_index = np.arange(HEIGHT)[:, None]
    col_index = np.arange(WIDTH)[None, :]
    static_image = np.zeros_like(flip)
    for pair_id in range(PAIR_COUNT):
        static_image[pair_id] = flip[pair_id] & (transition_counts[codes[pair_id], row_index, col_index] >= 2)
    xi_proxy = np.zeros_like(flip)
    total_sites = PAIR_COUNT * HEIGHT * WIDTH
    for event_id in xi_event_ids:
        if isinstance(event_id, bool) or not isinstance(event_id, int):
            raise PF2BucketAssignmentError("G4 xi event IDs must be exact integers")
        if not 0 <= event_id < total_sites:
            raise PF2BucketAssignmentError("G4 xi event ID is outside the n600 field")
        pair_id, pixel = divmod(event_id, HEIGHT * WIDTH)
        row, col = divmod(pixel, WIDTH)
        xi_proxy[pair_id, row, col] = True
    xi_proxy &= flip & ~static_image
    transient = flip & ~static_image & ~xi_proxy
    masks = {
        "STATIC_IN_IMAGE": static_image,
        "STATIC_IN_XI_PROXY": xi_proxy,
        "TRANSIENT": transient,
    }
    if any(mask.dtype != np.bool_ for mask in masks.values()):
        raise PF2BucketAssignmentError("temporal masks must be boolean")
    if not np.array_equal(static_image | xi_proxy | transient, flip):
        raise PF2BucketAssignmentError("G4 temporal masks do not partition the flip field")
    if np.any(static_image & xi_proxy) or np.any(static_image & transient) or np.any(xi_proxy & transient):
        raise PF2BucketAssignmentError("G4 temporal masks overlap")
    return flip, masks


def _bucket_event_mask(
    row: Mapping[str, Any],
    *,
    predicted: np.ndarray,
    target: np.ndarray,
    temporal_masks: Mapping[str, np.ndarray],
    boundary: np.ndarray,
) -> np.ndarray:
    """Reconstruct only content PF2 actually assigned to the typed row."""

    if (
        row.get("occupancy_status") != "MEASURED_EXACT_G4_DISCRETE_SKELETON"
        or row.get("visibility") != "seg-visible"
        or row.get("representation_type") != StreamType.SKELETON.value
    ):
        return np.zeros_like(predicted, dtype=bool)
    class_ids = row.get("class_ids")
    if (
        not isinstance(class_ids, list)
        or len(class_ids) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in class_ids)
    ):
        raise PF2BucketAssignmentError("PF2 class IDs are malformed")
    left, right = class_ids
    pair_mask = ((predicted == left) & (target == right)) | ((predicted == right) & (target == left))
    stratum = row.get("class_stratum")
    if stratum == "cell":
        stratum_mask = ~boundary
    elif stratum == "boundary":
        stratum_mask = boundary
    else:
        raise PF2BucketAssignmentError("PF2 class stratum is malformed")
    temporal_class = row.get("g4_temporal_class")
    try:
        temporal_mask = temporal_masks[str(temporal_class)]
    except KeyError as exc:
        raise PF2BucketAssignmentError("PF2 temporal class is malformed") from exc
    return pair_mask & stratum_mask & temporal_mask


def build_assignment_table(
    *,
    pf2_receipt: Mapping[str, Any],
    pf2_receipt_sha256: str,
    predicted: np.ndarray,
    target: np.ndarray,
    transition_counts: np.ndarray,
    xi_event_ids: Sequence[int],
    actuator_vocabulary: Sequence[str],
    direction_vocabulary: Sequence[str],
) -> dict[str, Any]:
    """Re-walk PF2 and emit exact membership plus explicit lost foreign keys."""

    atlas = pf2_receipt.get("typed_split_atlas")
    rows = atlas.get("rows") if isinstance(atlas, Mapping) else None
    if (
        not isinstance(atlas, Mapping)
        or atlas.get("schema") != PF2_ATLAS_SCHEMA
        or atlas.get("bucket_count") != PF2_BUCKET_COUNT
        or not isinstance(rows, list)
        or len(rows) != PF2_BUCKET_COUNT
    ):
        raise PF2BucketAssignmentError("PF2 receipt does not carry the sealed 1,200-row atlas")
    if len(set(actuator_vocabulary)) != len(actuator_vocabulary) or not actuator_vocabulary:
        raise PF2BucketAssignmentError("receiver actuator vocabulary must be nonempty and unique")
    if tuple(direction_vocabulary) != ("NEGATIVE_ONE_QUANTUM", "POSITIVE_ONE_QUANTUM"):
        raise PF2BucketAssignmentError("paired-secant direction vocabulary drifted")

    flip, temporal_masks = reconstruct_temporal_masks(
        predicted=predicted,
        target=target,
        transition_counts=transition_counts,
        xi_event_ids=xi_event_ids,
    )
    boundary = np.stack([boundary_mask(item) for item in target], axis=0)
    output_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    regenerated_mass = 0
    for atlas_row in rows:
        if (
            not isinstance(atlas_row, Mapping)
            or atlas_row.get("schema") != PF2_BUCKET_SCHEMA
            or not isinstance(atlas_row.get("bucket_id"), str)
            or atlas_row["bucket_id"] in seen
        ):
            raise PF2BucketAssignmentError("PF2 bucket rows are malformed or duplicated")
        bucket_id = str(atlas_row["bucket_id"])
        seen.add(bucket_id)
        event_mask = _bucket_event_mask(
            atlas_row,
            predicted=predicted,
            target=target,
            temporal_masks=temporal_masks,
            boundary=boundary,
        )
        pair_ids = np.flatnonzero(np.any(event_mask, axis=(1, 2))).astype(int).tolist()
        event_count = int(np.count_nonzero(event_mask))
        pair_filter = np.zeros(PAIR_COUNT, dtype=bool)
        pair_filter[pair_ids] = True
        regenerated_from_assignment = event_mask & pair_filter[:, None, None]
        if not np.array_equal(regenerated_from_assignment, event_mask):
            raise PF2BucketAssignmentError(f"PF2 bucket {bucket_id} pair-membership round trip differs")
        if event_count != int(atlas_row.get("content_event_count", -1)):
            raise PF2BucketAssignmentError(f"PF2 bucket {bucket_id} event mass differs from reconstruction")
        regenerated_mass += event_count
        output_rows.append(
            {
                "schema": ASSIGNMENT_ROW_SCHEMA,
                "bucket_id": bucket_id,
                "atlas_key": {field: atlas_row[field] for field in ATLAS_KEY_FIELDS},
                "pair_ids": pair_ids,
                "event_count": event_count,
                "pair_membership_status": "RECOVERED_EXACT_FROM_PF2_RAW_FLIP_EVENTS",
                "assignment_status": UNRECOVERABLE_STATUS,
                "receiver_actuator_ids": [],
                "direction_ids": [],
                "unrecoverable_reason": (
                    "PF2 construction partitions predicted-to-target class flips only; "
                    "it predates and carries no foreign key to the J2 lifted receiver "
                    "DOFs or G2F/G2G signed paired-secant observations."
                ),
                "forbidden_join": (
                    "Pixel support overlap, class-label similarity, or duplicating a "
                    "pair-level tensor is not a causal actuator assignment."
                ),
                "typed_stream_tag": _typed_tag(str(atlas_row["representation_type"])),
            }
        )
    expected_mass = int(atlas.get("measured_seg_skeleton_event_total", -1))
    if regenerated_mass != expected_mass or regenerated_mass != int(np.count_nonzero(flip)):
        raise PF2BucketAssignmentError("PF2 bucket mass conservation failed")

    pairs_per_bucket = Counter(len(row["pair_ids"]) for row in output_rows)
    buckets_per_pair = [0] * PAIR_COUNT
    for row in output_rows:
        for pair_id in row["pair_ids"]:
            buckets_per_pair[pair_id] += 1
    bucket_frequency = Counter(buckets_per_pair)
    table = {
        "schema": ASSIGNMENT_TABLE_SCHEMA,
        "pf2_receipt_sha256": pf2_receipt_sha256,
        "bucket_count": len(output_rows),
        "rows": output_rows,
        "round_trip": {
            "bucket_membership_set_equality_count": len(output_rows),
            "bucket_membership_set_equality_required": PF2_BUCKET_COUNT,
            "method": (
                "reconstruct the raw event mask from the landed atlas key, filter "
                "by emitted pair_ids, and require exact boolean-set equality"
            ),
            "regenerated_event_mass": regenerated_mass,
            "atlas_event_mass": expected_mass,
            "mass_conservation": regenerated_mass == expected_mass,
        },
        "coverage": {
            "fully_assigned_bucket_count": 0,
            "membership_recovered_bucket_count": len(output_rows),
            "unrecoverable_bucket_count": len(output_rows),
            "pairs_per_bucket_distribution": {str(key): value for key, value in sorted(pairs_per_bucket.items())},
            "buckets_per_pair_distribution": {str(key): value for key, value in sorted(bucket_frequency.items())},
            "orphan_pair_ids": [pair_id for pair_id, count in enumerate(buckets_per_pair) if count == 0],
            "multi_actuator_bucket_count": 0,
        },
        "foreign_key_vocabulary": {
            "receiver_actuator_stable_ids": list(actuator_vocabulary),
            "direction_ids": list(direction_vocabulary),
            "exact_join_row_count": 0,
        },
        "verdict": "PAIR_MEMBERSHIP_RECOVERED_ACTUATOR_DIRECTION_JOIN_UNRECOVERABLE",
        "verdict_scope": "INSTANCE_CURRENT_PF2_CONSTRUCTION_LINEAGE",
        "evidence_axis": "[macOS-CPU frozen-scorer advisory]",
        "score_claim": False,
        "pointer": "0.1910828242 [contest-CPU]",
        "pointer_moved": False,
        "main_landing_review_required": True,
    }
    table["table_content_sha256"] = canonical_sha256(table)
    return table


def validate_assignment_table(
    table: Mapping[str, Any],
    *,
    expected_pf2_sha256: str,
) -> None:
    """Strictly validate the first-class table without accepting partial shapes."""

    if table.get("schema") != ASSIGNMENT_TABLE_SCHEMA:
        raise PF2BucketAssignmentError("assignment table schema differs")
    if table.get("pf2_receipt_sha256") != expected_pf2_sha256:
        raise PF2BucketAssignmentError("assignment table PF2 binding differs")
    payload = dict(table)
    claimed = payload.pop("table_content_sha256", None)
    if claimed != canonical_sha256(payload):
        raise PF2BucketAssignmentError("assignment table content SHA differs")
    rows = table.get("rows")
    if not isinstance(rows, list) or len(rows) != PF2_BUCKET_COUNT:
        raise PF2BucketAssignmentError("assignment table must contain exactly 1,200 rows")
    bucket_ids: set[str] = set()
    for row in rows:
        if (
            not isinstance(row, Mapping)
            or row.get("schema") != ASSIGNMENT_ROW_SCHEMA
            or not isinstance(row.get("bucket_id"), str)
            or row["bucket_id"] in bucket_ids
        ):
            raise PF2BucketAssignmentError("assignment rows must have unique typed bucket IDs")
        bucket_ids.add(str(row["bucket_id"]))
        atlas_key = row.get("atlas_key")
        if not isinstance(atlas_key, Mapping) or set(atlas_key) != set(ATLAS_KEY_FIELDS):
            raise PF2BucketAssignmentError("assignment atlas key must preserve the six sealed PF2 fields")
        pair_ids = row.get("pair_ids")
        if (
            not isinstance(pair_ids, list)
            or pair_ids != sorted(set(pair_ids))
            or any(
                isinstance(pair_id, bool) or not isinstance(pair_id, int) or not 0 <= pair_id < PAIR_COUNT
                for pair_id in pair_ids
            )
        ):
            raise PF2BucketAssignmentError("assignment pair IDs must be sorted unique 0..599")
        TypedStreamTag.from_dict(row.get("typed_stream_tag"))
        status = row.get("assignment_status")
        if status == RECOVERED_STATUS:
            if not row.get("receiver_actuator_ids") or not row.get("direction_ids"):
                raise PF2BucketAssignmentError("complete assignment lacks actuator/direction IDs")
        elif not isinstance(status, str) or not status.startswith("ASSIGNMENT_UNRECOVERABLE_"):
            raise PF2BucketAssignmentError("assignment status is neither complete nor fail-closed")


__all__ = [
    "ASSIGNMENT_RECEIPT_SCHEMA",
    "ASSIGNMENT_ROW_SCHEMA",
    "ASSIGNMENT_TABLE_SCHEMA",
    "ATLAS_KEY_FIELDS",
    "RECOVERED_STATUS",
    "UNRECOVERABLE_STATUS",
    "PF2BucketAssignmentError",
    "build_assignment_table",
    "canonical_bytes",
    "canonical_sha256",
    "reconstruct_temporal_masks",
    "validate_assignment_table",
]
