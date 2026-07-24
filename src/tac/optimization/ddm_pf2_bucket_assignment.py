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
PROBE_RESULT_SCHEMA: Final = "ddm_ms6_receiver_support_probe.v1"
MEASURED_PROBE_ASSIGNMENT_SCHEMA: Final = "ddm_ms6_measured_probe_assignment.v1"
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
MEASURED_EMPTY_STATUS: Final = "ASSIGNMENT_UNRECOVERABLE_MEASURED_NO_PF2_ARGMAX_EVENT_PERTURBATION"
PARTIAL_MEASUREMENT_STATUS: Final = "ASSIGNMENT_UNRECOVERABLE_MEASUREMENT_SWEEP_PARTIAL"
DIRECTION_IDS: Final = ("NEGATIVE_ONE_QUANTUM", "POSITIVE_ONE_QUANTUM")
PROBE_STATUSES: Final = (
    "MEASURED_ARGMAX_PERTURBATION",
    "MEASURED_EMPTY_RASTER_SUPPORT",
    "MEASURED_EMPTY_NO_OCCUPIED_BUCKET_OVERLAP",
    "MEASURED_EMPTY_ARGMAX_INVARIANT",
    "INFEASIBLE_RECEIVER_QUANTUM",
)


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


def reconstruct_bucket_event_ids(
    *,
    pf2_receipt: Mapping[str, Any],
    predicted: np.ndarray,
    target: np.ndarray,
    transition_counts: np.ndarray,
    xi_event_ids: Sequence[int],
) -> dict[str, np.ndarray]:
    """Return the exact global raw-event IDs owned by every occupied PF2 bucket.

    Event IDs use PF2/G4's canonical row-major ``pair * H * W + y * W + x``
    address.  Empty buckets are retained as empty uint32 arrays.  This is the
    only admissible surface for the MS6 causal intersection; class labels,
    nearby pixels, and pair co-membership are not substitutes.
    """

    atlas = pf2_receipt.get("typed_split_atlas")
    rows = atlas.get("rows") if isinstance(atlas, Mapping) else None
    if (
        not isinstance(atlas, Mapping)
        or atlas.get("schema") != PF2_ATLAS_SCHEMA
        or not isinstance(rows, list)
        or len(rows) != PF2_BUCKET_COUNT
    ):
        raise PF2BucketAssignmentError("PF2 receipt does not carry the sealed 1,200-row atlas")
    flip, temporal_masks = reconstruct_temporal_masks(
        predicted=predicted,
        target=target,
        transition_counts=transition_counts,
        xi_event_ids=xi_event_ids,
    )
    boundary = np.stack([boundary_mask(item) for item in target], axis=0)
    result: dict[str, np.ndarray] = {}
    total = 0
    for row in rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("bucket_id"), str):
            raise PF2BucketAssignmentError("PF2 bucket row is malformed")
        bucket_id = str(row["bucket_id"])
        if bucket_id in result:
            raise PF2BucketAssignmentError("PF2 bucket IDs are duplicated")
        expected = int(row.get("content_event_count", -1))
        if expected == 0:
            result[bucket_id] = np.empty(0, dtype=np.uint32)
            continue
        mask = _bucket_event_mask(
            row,
            predicted=predicted,
            target=target,
            temporal_masks=temporal_masks,
            boundary=boundary,
        )
        event_ids = np.flatnonzero(mask.reshape(-1)).astype(np.uint32)
        if len(event_ids) != expected:
            raise PF2BucketAssignmentError(f"PF2 bucket {bucket_id} event mass differs")
        result[bucket_id] = event_ids
        total += len(event_ids)
    if total != int(atlas.get("measured_seg_skeleton_event_total", -1)):
        raise PF2BucketAssignmentError("PF2 reconstructed event-index mass differs")
    return result


def intersect_argmax_delta_with_pf2_events(
    *,
    pair_ids: Sequence[int],
    baseline_cells: np.ndarray,
    perturbed_cells: np.ndarray,
    bucket_event_ids: Mapping[str, np.ndarray],
) -> dict[str, dict[str, Any]]:
    """Measure exact PF2 raw events changed by one receiver probe.

    The returned event IDs are exact, not spatial-prior assignments.  Callers
    may serialize them compactly, but the SHA is always over canonical little
    endian uint32 bytes.
    """

    pairs = tuple(int(value) for value in pair_ids)
    before = np.asarray(baseline_cells)
    after = np.asarray(perturbed_cells)
    if (
        not pairs
        or len(set(pairs)) != len(pairs)
        or any(not 0 <= value < PAIR_COUNT for value in pairs)
        or before.shape != (len(pairs), HEIGHT, WIDTH)
        or after.shape != before.shape
        or before.dtype != np.uint8
        or after.dtype != np.uint8
    ):
        raise PF2BucketAssignmentError("probe argmax geometry or pair custody differs")
    changed = before != after
    local_by_pair = {pair_id: index for index, pair_id in enumerate(pairs)}
    result: dict[str, dict[str, Any]] = {}
    plane = HEIGHT * WIDTH
    for bucket_id, raw_ids in bucket_event_ids.items():
        ids = np.asarray(raw_ids)
        if (
            ids.ndim != 1
            or not np.issubdtype(ids.dtype, np.unsignedinteger)
            or np.any(ids.astype(np.uint64) >= PAIR_COUNT * plane)
            or not np.array_equal(ids, np.unique(ids))
        ):
            raise PF2BucketAssignmentError("PF2 bucket event IDs must be unsigned vectors")
        hits: list[np.ndarray] = []
        hit_pairs: list[int] = []
        if ids.size:
            event_pairs = ids.astype(np.uint64) // plane
            for pair_id in sorted({int(value) for value in event_pairs}):
                local = local_by_pair.get(pair_id)
                if local is None:
                    continue
                selected = ids[event_pairs == pair_id]
                offsets = selected.astype(np.uint64) % plane
                rows, cols = np.divmod(offsets, WIDTH)
                keep = changed[local, rows.astype(np.intp), cols.astype(np.intp)]
                if np.any(keep):
                    hits.append(selected[keep].astype(np.uint32, copy=False))
                    hit_pairs.append(pair_id)
        if hits:
            event_ids = np.sort(np.concatenate(hits).astype("<u4", copy=False))
            payload = event_ids.tobytes(order="C")
            result[str(bucket_id)] = {
                "pair_ids": hit_pairs,
                "event_count": int(event_ids.size),
                "event_ids": event_ids,
                "event_ids_sha256": hashlib.sha256(payload).hexdigest(),
            }
    return result


def build_measured_assignment_table(
    *,
    base_table: Mapping[str, Any],
    expected_pf2_sha256: str,
    probe_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Merge causal MS6 probe results into the exact MS5 table schema.

    A row becomes ``RECOVERED_COMPLETE`` only after an actual scorer argmax
    change intersects one of its exact PF2 raw events.  Measured-empty and
    infeasible probes remain first-class top-level custody and never create a
    bucket join.
    """

    validate_assignment_table(base_table, expected_pf2_sha256=expected_pf2_sha256)
    vocabulary = base_table.get("foreign_key_vocabulary")
    if not isinstance(vocabulary, Mapping):
        raise PF2BucketAssignmentError("assignment table lacks its foreign-key vocabulary")
    actuators = tuple(str(value) for value in vocabulary.get("receiver_actuator_stable_ids", ()))
    directions = tuple(str(value) for value in vocabulary.get("direction_ids", ()))
    if not actuators or len(set(actuators)) != len(actuators) or directions != DIRECTION_IDS:
        raise PF2BucketAssignmentError("assignment foreign-key vocabulary differs")

    seen: set[tuple[str, str]] = set()
    normalized: list[dict[str, Any]] = []
    by_bucket: dict[str, list[dict[str, Any]]] = {}
    known_bucket_ids = {str(row["bucket_id"]) for row in base_table["rows"]}
    for raw in probe_results:
        if raw.get("schema") != PROBE_RESULT_SCHEMA:
            raise PF2BucketAssignmentError("MS6 probe result schema differs")
        actuator = raw.get("receiver_actuator_id")
        direction = raw.get("direction_id")
        status = raw.get("status")
        checkpoint_sha = raw.get("checkpoint_sha256")
        key = (str(actuator), str(direction))
        if actuator not in actuators or direction not in directions or key in seen:
            raise PF2BucketAssignmentError("MS6 probe identity is unknown or duplicated")
        if status not in PROBE_STATUSES:
            raise PF2BucketAssignmentError("MS6 probe status differs")
        if (
            not isinstance(checkpoint_sha, str)
            or len(checkpoint_sha) != 64
            or any(value not in "0123456789abcdef" for value in checkpoint_sha)
        ):
            raise PF2BucketAssignmentError("MS6 probe checkpoint SHA differs")
        seen.add(key)
        hits = raw.get("bucket_hits", [])
        if not isinstance(hits, list):
            raise PF2BucketAssignmentError("MS6 probe bucket hits must be a list")
        hit_rows: list[dict[str, Any]] = []
        for hit in hits:
            if not isinstance(hit, Mapping):
                raise PF2BucketAssignmentError("MS6 bucket hit is malformed")
            bucket_id = hit.get("bucket_id")
            pair_ids = hit.get("pair_ids")
            event_count = hit.get("event_count")
            event_sha = hit.get("event_ids_sha256")
            if (
                not isinstance(bucket_id, str)
                or bucket_id not in known_bucket_ids
                or not isinstance(pair_ids, list)
                or pair_ids != sorted(set(pair_ids))
                or any(isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < PAIR_COUNT for value in pair_ids)
                or isinstance(event_count, bool)
                or not isinstance(event_count, int)
                or event_count <= 0
                or not isinstance(event_sha, str)
                or len(event_sha) != 64
            ):
                raise PF2BucketAssignmentError("MS6 bucket hit custody differs")
            assignment = {
                "schema": MEASURED_PROBE_ASSIGNMENT_SCHEMA,
                "receiver_actuator_id": actuator,
                "direction_id": direction,
                "pair_ids": pair_ids,
                "perturbed_event_count": event_count,
                "perturbed_event_ids_sha256": event_sha,
            }
            by_bucket.setdefault(bucket_id, []).append(assignment)
            hit_rows.append(dict(hit))
        if status == "MEASURED_ARGMAX_PERTURBATION" and not hit_rows:
            raise PF2BucketAssignmentError("measured perturbation probe has no exact bucket hits")
        if status != "MEASURED_ARGMAX_PERTURBATION" and hit_rows:
            raise PF2BucketAssignmentError("measured-empty/infeasible probe carries bucket hits")
        normalized.append(
            {
                "schema": PROBE_RESULT_SCHEMA,
                "receiver_actuator_id": actuator,
                "direction_id": direction,
                "status": status,
                "bucket_hit_count": len(hit_rows),
                "perturbed_event_count": sum(int(row["event_count"]) for row in hit_rows),
                "checkpoint_sha256": checkpoint_sha,
            }
        )

    rows: list[dict[str, Any]] = []
    for source in base_table["rows"]:
        row = dict(source)
        bucket_id = str(row["bucket_id"])
        assignments = sorted(
            by_bucket.get(bucket_id, []),
            key=lambda value: (value["receiver_actuator_id"], value["direction_id"]),
        )
        row["pf2_membership_pair_ids"] = list(source["pair_ids"])
        row["measured_probe_assignments"] = assignments
        if assignments:
            row["assignment_status"] = RECOVERED_STATUS
            row["receiver_actuator_ids"] = sorted(
                {str(value["receiver_actuator_id"]) for value in assignments}
            )
            row["direction_ids"] = sorted(
                {str(value["direction_id"]) for value in assignments}
            )
            row["pair_ids"] = sorted(
                {int(pair_id) for value in assignments for pair_id in value["pair_ids"]}
            )
            # Counts live on each probe assignment.  Summing them at bucket
            # level would count a raw PF2 event twice when both secant
            # directions perturb it, falsely presenting probe-event incidence
            # as a unique-event cardinality.
            row.pop("perturbed_event_count", None)
            row.pop("unrecoverable_reason", None)
            row.pop("forbidden_join", None)
        else:
            row["pair_ids"] = []
            row["receiver_actuator_ids"] = []
            row["direction_ids"] = []
            row["perturbed_event_count"] = 0
            row["assignment_status"] = (
                MEASURED_EMPTY_STATUS
                if len(seen) == len(actuators) * len(directions)
                else PARTIAL_MEASUREMENT_STATUS
            )
            row["unrecoverable_reason"] = (
                "No completed one-quantum receiver probe changed an argmax at "
                "this bucket's exact PF2 raw-event coordinates."
            )
        rows.append(row)
    recovered = sum(row["assignment_status"] == RECOVERED_STATUS for row in rows)
    infeasible = sum(row["status"] == "INFEASIBLE_RECEIVER_QUANTUM" for row in normalized)
    measured = len(normalized) - infeasible
    table = {
        key: value
        for key, value in base_table.items()
        if key not in {"rows", "coverage", "verdict", "verdict_scope", "table_content_sha256"}
    }
    table["rows"] = rows
    table["coverage"] = {
        **dict(base_table.get("coverage", {})),
        "fully_assigned_bucket_count": recovered,
        "unrecoverable_bucket_count": PF2_BUCKET_COUNT - recovered,
        "measured_probe_count": measured,
        "infeasible_probe_count": infeasible,
        "completed_probe_count": len(normalized),
        "required_probe_count": len(actuators) * len(directions),
        "measured_empty_probe_count": sum(
            row["status"].startswith("MEASURED_EMPTY_") for row in normalized
        ),
        "multi_actuator_bucket_count": sum(
            len(row["receiver_actuator_ids"]) > 1 for row in rows
        ),
    }
    table["foreign_key_vocabulary"] = {
        **dict(vocabulary),
        "exact_join_row_count": recovered,
    }
    table["probe_results"] = normalized
    table["verdict"] = (
        "CAUSAL_ACTUATOR_DIRECTION_JOIN_MEASURED_COMPLETE"
        if len(seen) == len(actuators) * len(directions) and not infeasible
        else "CAUSAL_ACTUATOR_DIRECTION_JOIN_PARTIAL"
    )
    table["verdict_scope"] = "INSTANCE_V19C_ENDPOINT_ONE_QUANTUM_SWEEP"
    table["table_content_sha256"] = canonical_sha256(table)
    validate_assignment_table(table, expected_pf2_sha256=expected_pf2_sha256)
    return table


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
            actuator_ids = row.get("receiver_actuator_ids")
            direction_ids = row.get("direction_ids")
            assignments = row.get("measured_probe_assignments")
            if not actuator_ids or not direction_ids:
                raise PF2BucketAssignmentError("complete assignment lacks actuator/direction IDs")
            if (
                not isinstance(actuator_ids, list)
                or actuator_ids != sorted(set(actuator_ids))
                or any(not isinstance(value, str) or not value for value in actuator_ids)
                or not isinstance(direction_ids, list)
                or direction_ids != sorted(set(direction_ids))
                or any(value not in DIRECTION_IDS for value in direction_ids)
            ):
                raise PF2BucketAssignmentError("complete assignment foreign-key custody differs")
            # Legacy v1 producer fixtures predate MS6 and carry only the three
            # aggregate fields.  Once MS6 custody is present it is mandatory
            # and re-derived strictly; omission remains accepted solely for
            # backward-compatible loading of those already-landed fixtures.
            if assignments is None:
                continue
            if not isinstance(assignments, list) or not assignments:
                raise PF2BucketAssignmentError("complete assignment lacks measured probe rows")
            derived_actuators: set[str] = set()
            derived_directions: set[str] = set()
            derived_pairs: set[int] = set()
            for assignment in assignments:
                if (
                    not isinstance(assignment, Mapping)
                    or assignment.get("schema") != MEASURED_PROBE_ASSIGNMENT_SCHEMA
                    or not isinstance(assignment.get("receiver_actuator_id"), str)
                    or assignment.get("direction_id") not in DIRECTION_IDS
                    or not isinstance(assignment.get("pair_ids"), list)
                    or assignment["pair_ids"] != sorted(set(assignment["pair_ids"]))
                    or not assignment["pair_ids"]
                    or any(
                        isinstance(pair_id, bool)
                        or not isinstance(pair_id, int)
                        or not 0 <= pair_id < PAIR_COUNT
                        for pair_id in assignment["pair_ids"]
                    )
                    or isinstance(assignment.get("perturbed_event_count"), bool)
                    or not isinstance(assignment.get("perturbed_event_count"), int)
                    or assignment["perturbed_event_count"] <= 0
                    or not isinstance(assignment.get("perturbed_event_ids_sha256"), str)
                    or len(assignment["perturbed_event_ids_sha256"]) != 64
                ):
                    raise PF2BucketAssignmentError("complete measured-probe assignment differs")
                derived_actuators.add(str(assignment["receiver_actuator_id"]))
                derived_directions.add(str(assignment["direction_id"]))
                derived_pairs.update(int(value) for value in assignment["pair_ids"])
            if (
                actuator_ids != sorted(derived_actuators)
                or direction_ids != sorted(derived_directions)
                or pair_ids != sorted(derived_pairs)
            ):
                raise PF2BucketAssignmentError("complete assignment aggregates differ")
        elif not isinstance(status, str) or not status.startswith("ASSIGNMENT_UNRECOVERABLE_"):
            raise PF2BucketAssignmentError("assignment status is neither complete nor fail-closed")
        if "pf2_membership_pair_ids" in row:
            membership = row["pf2_membership_pair_ids"]
            if (
                not isinstance(membership, list)
                or membership != sorted(set(membership))
                or any(
                    isinstance(pair_id, bool)
                    or not isinstance(pair_id, int)
                    or not 0 <= pair_id < PAIR_COUNT
                    for pair_id in membership
                )
            ):
                raise PF2BucketAssignmentError("PF2 membership pair IDs differ")


__all__ = [
    "ASSIGNMENT_RECEIPT_SCHEMA",
    "ASSIGNMENT_ROW_SCHEMA",
    "ASSIGNMENT_TABLE_SCHEMA",
    "ATLAS_KEY_FIELDS",
    "DIRECTION_IDS",
    "MEASURED_EMPTY_STATUS",
    "MEASURED_PROBE_ASSIGNMENT_SCHEMA",
    "PARTIAL_MEASUREMENT_STATUS",
    "PROBE_RESULT_SCHEMA",
    "RECOVERED_STATUS",
    "UNRECOVERABLE_STATUS",
    "PF2BucketAssignmentError",
    "build_assignment_table",
    "build_measured_assignment_table",
    "canonical_bytes",
    "canonical_sha256",
    "intersect_argmax_delta_with_pf2_events",
    "reconstruct_bucket_event_ids",
    "reconstruct_temporal_masks",
    "validate_assignment_table",
]
