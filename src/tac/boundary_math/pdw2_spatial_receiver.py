# SPDX-License-Identifier: MIT
"""Scorer-free, coefficient-only PDW2 spatial receiver contract.

The receiver consumes only strict PDW2/PDP2 packets and an explicit quotient
field tensor.  It does not import SegNet, PoseNet, nor any scorer runtime.  The
module exists to separate two contracts:

1. what the packet alone implies about spatial labels is *not* sufficient;
2. what a specific packet plus a real quotient field implies is deterministic.

The canonical blocker is `PDW2_COEFFICIENT_ONLY_SPATIAL_NONIDENTIFIABILITY`.
"""

from __future__ import annotations

import hashlib
import mmap
from dataclasses import dataclass
from typing import Any, Final

import numpy as np

from tac.boundary_math.power_diagram_witness import (
    _build_gauge_fixed_target,
    decode_pdw2,
    encode_pdw2,
    gauge_fixed_assign_f32,
)

EXPECTED_GEOMETRY: Final = (384, 512, 4)
EXPECTED_RANK: Final = 4
PDW2_COEFFICIENT_ONLY_SPATIAL_NONIDENTIFIABILITY: Final = "PDW2_COEFFICIENT_ONLY_SPATIAL_NONIDENTIFIABILITY"
RECEIVER_SCHEMA: Final = "pdw2_spatial_receiver_receipt.v1"
WITNESS_SCHEMA: Final = "pdw2_coefficient_only_nonidentifiability_witness.v1"
CANARY_SCHEMA: Final = "pdw2_spatial_receiver_packet_mutation_canary.v1"


class PDW2SpatialReceiverError(ValueError):
    """Raised when receiver input/protocol contracts are violated."""


@dataclass(frozen=True)
class ReceiverReceipt:
    schema: str
    pdw2_packet_sha256: str
    pdw2_promotion_blocker: str
    packet_to_partition_consumed: bool
    coefficient_only_through_r_equivalent: bool
    through_r_authority: bool
    d_seg: None
    d_pose: None
    score_claim: bool
    promotion_eligible: bool
    pair_count: int
    rank: int
    class_ids: list[int]
    field_shape: list[int]
    field_sha256: str
    partition_shape: list[int]
    partition_label_counts: list[int]
    partition_labels_sha256: str
    mapped_page_eviction_applied: bool
    partition_labels: list[list[list[int]]] | None = None



def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _validate_packet(packet: bytes | bytearray | memoryview) -> Any:
    if not isinstance(packet, (bytes, bytearray, memoryview)) or len(packet) == 0:
        raise PDW2SpatialReceiverError("PDW2 packet must be non-empty bytes")
    try:
        target = decode_pdw2(packet)
    except ValueError as exc:
        raise PDW2SpatialReceiverError(f"invalid PDW2 packet: {exc}") from exc
    if target.rank != EXPECTED_RANK:
        raise PDW2SpatialReceiverError(
            f"PDW2 packet rank must be {EXPECTED_RANK} for this receiver; "
            f"received {target.rank}"
        )
    canonical = encode_pdw2(target)
    if bytes(packet) != canonical:
        raise PDW2SpatialReceiverError("PDW2 packet is not canonical-re-encoded")
    return target


def _validate_field(field: np.ndarray) -> np.ndarray:
    if not isinstance(field, np.ndarray):
        raise TypeError("quotient field must be a np.ndarray")
    if field.dtype != np.float32:
        raise PDW2SpatialReceiverError("quotient field must be float32")
    if field.ndim != 4:
        raise PDW2SpatialReceiverError("quotient field must be 4D: [pair, H, W, rank]")
    if field.shape[1:] != EXPECTED_GEOMETRY:
        raise PDW2SpatialReceiverError(
            f"quotient field geometry must be [N, {EXPECTED_GEOMETRY[0]}, {EXPECTED_GEOMETRY[1]}, {EXPECTED_RANK}]"
        )
    if field.shape[0] <= 0:
        raise PDW2SpatialReceiverError("quotient field must contain at least one pair")
    if field.shape[0] > 600:
        raise PDW2SpatialReceiverError("quotient field pair count cannot exceed 600")
    return field


def _release_memmap_pages(field: np.ndarray) -> bool:
    """Best-effort eviction of already-consumed file-backed pages.

    Pairwise arithmetic bounds heap allocations, but a sequential read of an
    n600 memmap can otherwise leave the full 1.8 GiB mapping resident on macOS.
    ``MADV_DONTNEED`` changes only the process page cache; it never mutates the
    read-only source file and is deliberately advisory on unsupported hosts.
    """

    if not isinstance(field, np.memmap) or getattr(field, "mode", None) != "r":
        return False
    mapping = getattr(field, "_mmap", None)
    if mapping is None or not hasattr(mapping, "madvise") or not hasattr(mmap, "MADV_DONTNEED"):
        return False
    try:
        mapping.madvise(mmap.MADV_DONTNEED)
    except (OSError, ValueError):
        return False
    return True


def _pairwise_labels_and_counts(
    target: Any,
    field: np.ndarray,
    *,
    include_labels: bool,
) -> tuple[list[list[list[int]]] | None, list[int], str, str, bool]:
    class_ids = [int(v) for v in np.asarray(target.class_ids, dtype=np.int64).tolist()]
    counts = np.zeros(len(class_ids), dtype=np.int64)
    field_hasher = hashlib.sha256()
    labels_hasher = hashlib.sha256()
    labels_payload: list[list[list[int]]] | None = [] if include_labels else None
    mapped_page_eviction_applied = False

    for pair in range(field.shape[0]):
        pair_field = np.asarray(field[pair : pair + 1], dtype=np.float32)
        if not np.isfinite(pair_field).all():
            raise PDW2SpatialReceiverError(
                f"quotient field must be finite; non-finite value in pair {pair}"
            )
        field_hasher.update(pair_field.tobytes(order="C"))
        pair_labels = np.asarray(
            gauge_fixed_assign_f32(pair_field, target)[0],
            dtype=np.int64,
        )
        labels_u16 = np.asarray(pair_labels, dtype="<u2")
        labels_hasher.update(labels_u16.tobytes(order="C"))

        if len(class_ids) == 2 and class_ids == [0, 1]:
            counts += np.bincount(pair_labels.reshape(-1), minlength=len(class_ids))
        else:
            for class_index, class_id in enumerate(class_ids):
                counts[class_index] += int(np.count_nonzero(pair_labels == class_id))

        if labels_payload is not None:
            labels_payload.append(pair_labels.astype(int).tolist())
        mapped_page_eviction_applied = (
            _release_memmap_pages(field) or mapped_page_eviction_applied
        )

    return (
        labels_payload,
        [int(row) for row in counts.tolist()],
        labels_hasher.hexdigest(),
        field_hasher.hexdigest(),
        mapped_page_eviction_applied,
    )


def _vector_feature_candidates() -> list[np.ndarray]:
    scales = (-4.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 4.0)
    candidates: list[np.ndarray] = [np.zeros((EXPECTED_RANK,), dtype=np.float32)]
    eye = np.eye(EXPECTED_RANK, dtype=np.float32)
    for axis in range(EXPECTED_RANK):
        for scale in scales:
            candidates.append((scale * eye[axis]).astype(np.float32))
    for a in scales:
        for b in scales:
            candidates.append(np.array((a, b, 0.5 * a, -0.5 * b), dtype=np.float32))
    for a in scales:
        for b in scales:
            candidates.append(np.array((a, a, b, b), dtype=np.float32))
    return candidates


def run_pdw2_spatial_receiver(
    pdw2_packet: bytes | bytearray | memoryview,
    quotient_feature_field: np.ndarray,
    *,
    include_labels: bool = False,
) -> dict[str, Any]:
    """Decode a quotient field from a strict PDW2 packet with fixed f32 arithmetic.

    Returns:
      * partition-label custody hashes (immutable and deterministic)
      * geometry provenance metadata
      * explicit negative score claims (`d_seg`/`d_pose` are both None)

    Args:
      include_labels: include full per-pair labels (small tensors only; default False)
    """

    target = _validate_packet(pdw2_packet)
    field = _validate_field(quotient_feature_field)
    if include_labels and field.shape[0] > 8:
        raise PDW2SpatialReceiverError(
            "including labels for large pair counts is disabled to avoid large "
            "non-authority artifacts; use pair-wise streaming mode"
        )

    (
        partition_labels,
        partition_counts,
        partition_sha256,
        field_sha256,
        mapped_page_eviction_applied,
    ) = _pairwise_labels_and_counts(target, field, include_labels=include_labels)

    class_ids = [int(v) for v in np.asarray(target.class_ids, dtype=np.int64).tolist()]
    return ReceiverReceipt(
        schema=RECEIVER_SCHEMA,
        pdw2_packet_sha256=_sha256(bytes(pdw2_packet)),
        pdw2_promotion_blocker=PDW2_COEFFICIENT_ONLY_SPATIAL_NONIDENTIFIABILITY,
        packet_to_partition_consumed=True,
        coefficient_only_through_r_equivalent=False,
        through_r_authority=False,
        d_seg=None,
        d_pose=None,
        score_claim=False,
        promotion_eligible=False,
        pair_count=int(field.shape[0]),
        rank=int(target.rank),
        class_ids=class_ids,
        field_shape=[int(x) for x in field.shape],
        field_sha256=field_sha256,
        partition_shape=[int(field.shape[0]), int(field.shape[1]), int(field.shape[2])],
        partition_label_counts=partition_counts,
        partition_labels_sha256=partition_sha256,
        mapped_page_eviction_applied=mapped_page_eviction_applied,
        partition_labels=partition_labels,
    ).__dict__


def _vector_fields_from_target(target: Any) -> tuple[np.ndarray, np.ndarray]:
    # Deterministically construct two finite constant feature vectors whose
    # spatial partition differs for this packet, whenever possible.
    representatives: dict[int, np.ndarray] = {}
    for vector in _vector_feature_candidates():
        point = np.asarray(vector, dtype=np.float32).reshape(1, 1, 1, EXPECTED_RANK)
        class_id = int(gauge_fixed_assign_f32(point, target)[0, 0, 0])
        representatives.setdefault(class_id, point)
        if len(representatives) >= 2:
            break
    if len(representatives) < 2:
        raise PDW2SpatialReceiverError(
            "packet-only witness is not constructible by deterministic const fields"
        )

    ordered = [
        vector
        for _, vector in sorted(representatives.items(), key=lambda item: item[0])[:2]
    ]
    return (
        np.broadcast_to(ordered[0], (1, *EXPECTED_GEOMETRY)).copy(),
        np.broadcast_to(ordered[1], (1, *EXPECTED_GEOMETRY)).copy(),
    )


def _labels_for_field(packet: bytes | bytearray | memoryview, field: np.ndarray) -> np.ndarray:
    target = _validate_packet(packet)
    checked = _validate_field(field)
    return np.asarray(gauge_fixed_assign_f32(checked, target), dtype=np.int64)


def build_pdw2_coefficient_only_nonidentifiability_witness(
    pdw2_packet: bytes | bytearray | memoryview,
) -> dict[str, Any]:
    """Construct a deterministic packet-only witness proving multiple partitions.

    The witness is an executable counterexample: one packet, two constant finite
    quotient tensors (under the same bytes), different argmax partitions.
    """
    target = _validate_packet(pdw2_packet)
    field_a, field_b = _vector_fields_from_target(target)
    labels_a = _labels_for_field(pdw2_packet, field_a)
    labels_b = _labels_for_field(pdw2_packet, field_b)
    classes = [int(v) for v in np.asarray(target.class_ids, dtype=np.int64).tolist()]
    if np.array_equal(labels_a, labels_b):
        raise PDW2SpatialReceiverError("witness construction failed to separate classes")

    return {
        "schema": WITNESS_SCHEMA,
        "pdw2_packet_sha256": _sha256(bytes(pdw2_packet)),
        "packet_to_partition_consumed": True,
        "coefficient_only_through_r_equivalent": False,
        "through_r_authority": False,
        "coefficient_only_persisted_witness_verified": True,
        "pdw2_promotion_blocker": PDW2_COEFFICIENT_ONLY_SPATIAL_NONIDENTIFIABILITY,
        "witness_point": {
            "pair": 0,
            "row": 0,
            "col": 0,
            "geometry": EXPECTED_GEOMETRY,
        },
        "witness_feature_vectors": {
            "feature_vector_a": field_a[0, 0, 0].astype(float).tolist(),
            "feature_vector_b": field_b[0, 0, 0].astype(float).tolist(),
            "class_a": int(labels_a[0, 0, 0]),
            "class_b": int(labels_b[0, 0, 0]),
        },
        "class_ids": classes,
        "d_seg": None,
        "d_pose": None,
        "score_claim": False,
        "promotion_eligible": False,
    }


def detect_pdw2_packet_mutation_canary(
    pdw2_packet: bytes | bytearray | memoryview,
    mutated_packet: bytes | bytearray | memoryview,
    quotient_feature_field: np.ndarray,
) -> dict[str, Any]:
    """Compare original/mutated packet partitions on the same field.

    Returns ``mutation_observed`` only when the packet edit changes at least one
    label. Mutation with no label change is explicitly not counted as a canary.
    """
    target_base = _validate_packet(pdw2_packet)
    target_mut = _validate_packet(mutated_packet)
    if not np.array_equal(target_base.class_ids, target_mut.class_ids):
        raise PDW2SpatialReceiverError("base/mutant partitions differ in canonical class ids")

    field = _validate_field(quotient_feature_field)
    mismatch = 0
    for pair in range(field.shape[0]):
        pair_slice = np.asarray(field[pair : pair + 1], dtype=np.float32)
        if not np.isfinite(pair_slice).all():
            raise PDW2SpatialReceiverError(
                f"quotient field must be finite; non-finite value in pair {pair}"
            )
        base_labels = np.asarray(gauge_fixed_assign_f32(pair_slice, target_base)[0], dtype=np.int64)
        mut_labels = np.asarray(gauge_fixed_assign_f32(pair_slice, target_mut)[0], dtype=np.int64)
        mismatch += int(np.count_nonzero(base_labels != mut_labels))
        _release_memmap_pages(field)

    labels_seen = float(mismatch)
    total = float(field.shape[0] * field.shape[1] * field.shape[2])
    return {
        "schema": CANARY_SCHEMA,
        "pdw2_promotion_blocker": PDW2_COEFFICIENT_ONLY_SPATIAL_NONIDENTIFIABILITY,
        "packet_to_partition_consumed": True,
        "coefficient_only_through_r_equivalent": False,
        "through_r_authority": False,
        "mutation_observed": mismatch > 0,
        "mismatch_pixels": mismatch,
        "mismatch_pairs_total": labels_seen / total if total > 0 else 0.0,
        "d_seg": None,
        "d_pose": None,
        "score_claim": False,
        "promotion_eligible": False,
    }


def mutate_pdw2_packet_first_relative_coefficient(
    pdw2_packet: bytes | bytearray | memoryview,
    delta: float,
) -> bytes:
    """Return a canonical packet with a deterministic first relative coefficient moved."""

    target = _validate_packet(pdw2_packet)
    if target.relative_coefficients.size == 0:
        raise PDW2SpatialReceiverError("target has no relative coefficients to mutate")

    perturbed = np.array(target.relative_coefficients, dtype=np.float32, copy=True)
    perturbed[0, 0] = np.float32(perturbed[0, 0] + np.float32(delta))
    rebuilt = _build_gauge_fixed_target(
        target.class_ids,
        perturbed,
        target.adjacency,
        mode=target.mode,
        scale_pivot=target.scale_pivot,
    )
    return encode_pdw2(rebuilt)


__all__ = [
    "CANARY_SCHEMA",
    "EXPECTED_GEOMETRY",
    "EXPECTED_RANK",
    "PDW2_COEFFICIENT_ONLY_SPATIAL_NONIDENTIFIABILITY",
    "RECEIVER_SCHEMA",
    "WITNESS_SCHEMA",
    "PDW2SpatialReceiverError",
    "build_pdw2_coefficient_only_nonidentifiability_witness",
    "detect_pdw2_packet_mutation_canary",
    "mutate_pdw2_packet_first_relative_coefficient",
    "run_pdw2_spatial_receiver",
]
