# SPDX-License-Identifier: MIT
"""Deterministic DDM realized-flip menu compilation and local-statistics codec.

The compiler keeps cross-control evidence separate from prices on the active
V19C endpoint.  A cluster/fix row is therefore only waterfill-eligible after
the exact receiver/scorer path has supplied both its error and byte deltas.
The small statistics and mask codecs below are the counted measurement
payloads used by the MENU1 runner; neither embeds scorer weights.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import brotli
import numpy as np

MENU_SCHEMA = "ddm_realized_flip_menu.v1"
ROW_SCHEMA = "ddm_realized_flip_menu.row.v1"
LOCAL_STATS_SCHEMA = "ddm_local_statistics.v1"
MASK_SCHEMA = "ddm_targeted_cluster_mask.v1"
SCALAR_AFFINE_SCHEMA = "ddm_scalar_affine.v1"
TEMPORAL_AFFINE_SCHEMA = "ddm_temporal_affine.v1"
POSTCHARTER_SCHEMA = "ddm_realized_flip_menu.postcharter.v1"
EVIDENCE_AXIS = "[macOS-CPU frozen-scorer advisory]"
SOURCE_BYTES = 37_545_489
RATE_DUAL = 25.0 / SOURCE_BYTES
N_CLASSES = 5
CAMERA_HW = (874, 1164)
SEG_HW = (384, 512)


class RealizedFlipMenuError(ValueError):
    """A menu, payload, or telescoping-chain invariant failed closed."""


@dataclass(frozen=True, slots=True)
class FixSpec:
    fix_id: str
    mechanism_bucket: str
    pool_id: str
    source_authority: str
    application_stage: str
    cross_control_evidence_only: bool
    shared_delta_bytes: int | None = None
    bridge_required: bool = False


FIX_SPECS: tuple[FixSpec, ...] = (
    FixSpec(
        "local_amplitude_statistics",
        "BN_SE_AMPLITUDE_STATISTICS",
        "paint_amplitude",
        "PT1 83e06ef4; MENU1 exact V19C remeasure required",
        "REALIZE",
        True,
    ),
    FixSpec(
        "hard_camera_placement",
        "SUB_CELL_PLACEMENT",
        "paint_geometry",
        "PT1 83e06ef4; MENU1 exact V19C remeasure required",
        "PROJECT",
        True,
    ),
    FixSpec(
        "analytic_coverage_blend",
        "SUB_CELL_PLACEMENT",
        "paint_geometry",
        "PT1 83e06ef4; same-pool competitor to hard placement",
        "PROJECT",
        True,
    ),
    FixSpec(
        "dv1_semantic_extension",
        "COARSE_DESCRIPTION",
        "semantic_description",
        "SN1/DV1 selected joint section 03897224",
        "PREDICT",
        False,
        shared_delta_bytes=1_610,
    ),
    FixSpec(
        "v19c_receiver_correction",
        "SCORER_REALIZATION_CORRECTION",
        "receiver_correction",
        "V19C 506fb1df; endpoint contribution already absorbed in base",
        "REALIZE",
        True,
    ),
    FixSpec(
        "e2_pose_frame_dedup",
        "TEMPORAL_STREAM_CODE",
        "semantic_stream_code",
        "E2 720ef23a; cross-control pose/frame-policy evidence",
        "CODE",
        True,
        bridge_required=True,
    ),
)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def cluster_id(row: Mapping[str, Any]) -> str:
    """Return a stable content-derived identity for one SN1 solve-menu row."""

    dimensions = {
        key: row[key]
        for key in (
            "source",
            "stratum",
            "ordered_pair",
            "boundary_distance_band",
            "curvature_band",
            "curve_availability",
            "d2_band",
            "g3_tail_bucket",
            "paint_floor_mechanism",
            "temporal_pattern",
        )
    }
    return "sn1_" + sha256_bytes(_canonical_json(dimensions))[:20]


def compile_menu_rows(
    solve_rows: Sequence[Mapping[str, Any]],
    *,
    v19c_residual_errors: int,
    v19c_total_errors: int,
    receipt_sha256: Mapping[str, str],
) -> list[dict[str, Any]]:
    """Cross the complete SN1 menu with the governed fix inventory.

    Historical PT1/E2/V19C effects are not copied into the active price
    columns.  Only a content-identified byte home may be pre-priced, and even
    then the row remains ineligible until an active-base error delta exists.
    """

    if len(solve_rows) != 2_649:
        raise RealizedFlipMenuError(
            f"SN1 solve menu must have 2649 rows, observed {len(solve_rows)}"
        )
    if v19c_residual_errors != 2_265_811:
        raise RealizedFlipMenuError("V19C endpoint residual error custody differs")
    if v19c_total_errors != 2_923_991:
        raise RealizedFlipMenuError("V19C endpoint total error custody differs")
    if len({cluster_id(row) for row in solve_rows}) != len(solve_rows):
        raise RealizedFlipMenuError("SN1 solve-menu dimensions are not unique")
    output: list[dict[str, Any]] = []
    for source_row in solve_rows:
        cid = cluster_id(source_row)
        for fix in FIX_SPECS:
            pool = f"{fix.pool_id}:{cid}"
            output.append(
                {
                    "schema": ROW_SCHEMA,
                    "row_id": f"{cid}:{fix.fix_id}",
                    "cluster_id": cid,
                    "cluster_rank": int(source_row["menu_rank"]),
                    "cluster_errors_before_upper_bound": int(
                        source_row["error_count"]
                    ),
                    "cluster_dimensions": {
                        key: source_row[key]
                        for key in (
                            "source",
                            "stratum",
                            "ordered_pair",
                            "boundary_distance_band",
                            "curvature_band",
                            "curve_availability",
                            "d2_band",
                            "g3_tail_bucket",
                            "paint_floor_mechanism",
                            "temporal_pattern",
                        )
                    },
                    "fix_id": fix.fix_id,
                    "mechanism_bucket": fix.mechanism_bucket,
                    "composition_pool_id": pool,
                    "application_stage": fix.application_stage,
                    "shared_component_id": (
                        f"shared:{fix.fix_id}"
                        if fix.shared_delta_bytes is not None
                        else None
                    ),
                    "delta_errors_realized": None,
                    "delta_counted_bytes": fix.shared_delta_bytes,
                    "byte_partition": {
                        "COUNTED": fix.shared_delta_bytes,
                        "FREE": 0,
                        "NULL": 0,
                        "status": (
                            "PARTIAL_COUNTED_PRICE"
                            if fix.shared_delta_bytes is not None
                            else "COUNTED_PRICE_UNMEASURED"
                        ),
                        "law": "FREE_UNION_NULL_UNION_COUNTED",
                    },
                    "measurement_status": (
                        "PARTIALLY_PRICED_BYTES_ONLY"
                        if fix.shared_delta_bytes is not None
                        else "UNPRICED_ACTIVE_BASE_REMEASURE_REQUIRED"
                    ),
                    "waterfill_eligible": False,
                    "cross_control_evidence_only": fix.cross_control_evidence_only,
                    "bridge_required": fix.bridge_required,
                    "source_authority": fix.source_authority,
                    "base_candidate": "v19c_endpoint_506fb1df",
                    "base_residual_errors": v19c_residual_errors,
                    "base_total_errors": v19c_total_errors,
                    "receipt_sha256": dict(receipt_sha256),
                    "evidence_axis": EVIDENCE_AXIS,
                    "research_only": True,
                    "score_claim": False,
                }
            )
    expected = len(solve_rows) * len(FIX_SPECS)
    if len(output) != expected or len({row["row_id"] for row in output}) != expected:
        raise RealizedFlipMenuError("compiled menu cardinality or row identity differs")
    return output


def fit_local_statistics(
    *,
    source_rgb_u8: np.ndarray,
    target_rgb_u8: np.ndarray,
    semantic_cells: np.ndarray,
    row_bands: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit per-class x camera-row-band channel mean/variance transforms."""

    sufficient = local_statistics_sufficient_statistics(
        source_rgb_u8=source_rgb_u8,
        target_rgb_u8=target_rgb_u8,
        semantic_cells=semantic_cells,
        row_bands=row_bands,
    )
    return solve_local_statistics(sufficient)


def local_statistics_sufficient_statistics(
    *,
    source_rgb_u8: np.ndarray,
    target_rgb_u8: np.ndarray,
    semantic_cells: np.ndarray,
    row_bands: int,
) -> dict[str, np.ndarray]:
    """Return mergeable float64 sufficient statistics for the local codec."""

    source = np.asarray(source_rgb_u8)
    target = np.asarray(target_rgb_u8)
    cells = np.asarray(semantic_cells)
    if (
        source.shape != target.shape
        or source.ndim != 4
        or source.shape[1:] != (*CAMERA_HW, 3)
        or source.dtype != np.uint8
        or target.dtype != np.uint8
    ):
        raise RealizedFlipMenuError("local-statistics RGB geometry differs")
    if (
        cells.shape != (source.shape[0], *SEG_HW)
        or not np.issubdtype(cells.dtype, np.integer)
        or cells.min(initial=0) < 0
        or cells.max(initial=0) >= N_CLASSES
    ):
        raise RealizedFlipMenuError("local-statistics semantic geometry differs")
    if isinstance(row_bands, bool) or not 1 <= row_bands <= 64:
        raise RealizedFlipMenuError("row_bands must be an integer in [1,64]")
    ys = (np.arange(CAMERA_HW[0]) * SEG_HW[0] // CAMERA_HW[0]).clip(
        0, SEG_HW[0] - 1
    )
    xs = (np.arange(CAMERA_HW[1]) * SEG_HW[1] // CAMERA_HW[1]).clip(
        0, SEG_HW[1] - 1
    )
    camera_classes = cells[:, ys[:, None], xs[None, :]]
    camera_bands = (
        np.arange(CAMERA_HW[0], dtype=np.int64) * row_bands // CAMERA_HW[0]
    )
    counts = np.zeros((N_CLASSES, row_bands), dtype=np.int64)
    source_sum = np.zeros((N_CLASSES, row_bands, 3), dtype=np.float64)
    target_sum = np.zeros_like(source_sum)
    source_sumsq = np.zeros_like(source_sum)
    target_sumsq = np.zeros_like(source_sum)
    source_f = source.astype(np.float64)
    target_f = target.astype(np.float64)
    for class_id in range(N_CLASSES):
        class_mask = camera_classes == class_id
        for band in range(row_bands):
            mask = class_mask & (camera_bands[None, :, None] == band)
            count = int(np.count_nonzero(mask))
            counts[class_id, band] = count
            if count == 0:
                continue
            source_values = source_f[mask]
            target_values = target_f[mask]
            source_sum[class_id, band] = source_values.sum(axis=0)
            target_sum[class_id, band] = target_values.sum(axis=0)
            source_sumsq[class_id, band] = np.square(source_values).sum(axis=0)
            target_sumsq[class_id, band] = np.square(target_values).sum(axis=0)
    return {
        "counts": counts,
        "source_sum": source_sum,
        "target_sum": target_sum,
        "source_sumsq": source_sumsq,
        "target_sumsq": target_sumsq,
    }


def solve_local_statistics(
    sufficient: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve one exact aggregate of local-statistics sufficient rows."""

    required = {
        "counts",
        "source_sum",
        "target_sum",
        "source_sumsq",
        "target_sumsq",
    }
    if set(sufficient) != required:
        raise RealizedFlipMenuError("local-statistics sufficient fields differ")
    counts = np.asarray(sufficient["counts"], dtype=np.int64)
    if counts.ndim != 2 or counts.shape[0] != N_CLASSES:
        raise RealizedFlipMenuError("local-statistics sufficient count geometry differs")
    expected = (N_CLASSES, counts.shape[1], 3)
    values = {
        key: np.asarray(sufficient[key], dtype=np.float64)
        for key in required - {"counts"}
    }
    if any(value.shape != expected for value in values.values()):
        raise RealizedFlipMenuError("local-statistics sufficient geometry differs")
    scale = np.ones(expected, dtype=np.float32)
    offset = np.zeros(expected, dtype=np.float32)
    for class_id in range(N_CLASSES):
        for band in range(counts.shape[1]):
            count = int(counts[class_id, band])
            if count == 0:
                continue
            source_mean = values["source_sum"][class_id, band] / count
            target_mean = values["target_sum"][class_id, band] / count
            source_var = np.maximum(
                values["source_sumsq"][class_id, band] / count
                - np.square(source_mean),
                0.0,
            )
            target_var = np.maximum(
                values["target_sumsq"][class_id, band] / count
                - np.square(target_mean),
                0.0,
            )
            source_std = np.sqrt(source_var)
            target_std = np.sqrt(target_var)
            usable = source_std >= 1.0e-6
            scale[class_id, band, usable] = (
                target_std[usable] / source_std[usable]
            )
            offset[class_id, band] = (
                target_mean - scale[class_id, band] * source_mean
            )
    return scale, offset, counts


def encode_local_statistics(scale: np.ndarray, offset: np.ndarray) -> bytes:
    """Encode float16 local statistics with exact parse-back."""

    scale_value = np.asarray(scale, dtype="<f2")
    offset_value = np.asarray(offset, dtype="<f2")
    if (
        scale_value.ndim != 3
        or scale_value.shape[0] != N_CLASSES
        or scale_value.shape[-1] != 3
        or offset_value.shape != scale_value.shape
        or not np.all(np.isfinite(scale_value))
        or not np.all(np.isfinite(offset_value))
    ):
        raise RealizedFlipMenuError("local-statistics payload shape differs")
    header = struct.pack("<8sHHH", b"DDMLS1\0\0", N_CLASSES, scale_value.shape[1], 3)
    payload = header + scale_value.tobytes(order="C") + offset_value.tobytes(
        order="C"
    )
    decoded_scale, decoded_offset = decode_local_statistics(payload)
    if not np.array_equal(decoded_scale, scale_value) or not np.array_equal(
        decoded_offset, offset_value
    ):
        raise RealizedFlipMenuError("local-statistics parse-back differs")
    return payload


def decode_local_statistics(payload: bytes) -> tuple[np.ndarray, np.ndarray]:
    if len(payload) < 14:
        raise RealizedFlipMenuError("local-statistics payload is truncated")
    magic, classes, bands, channels = struct.unpack("<8sHHH", payload[:14])
    if magic != b"DDMLS1\0\0" or classes != N_CLASSES or channels != 3:
        raise RealizedFlipMenuError("local-statistics payload header differs")
    count = classes * bands * channels
    expected = 14 + 2 * count * np.dtype("<f2").itemsize
    if len(payload) != expected:
        raise RealizedFlipMenuError("local-statistics payload byte count differs")
    values = np.frombuffer(payload[14:], dtype="<f2")
    scale = values[:count].reshape(classes, bands, channels)
    offset = values[count:].reshape(classes, bands, channels)
    return scale.copy(), offset.copy()


def apply_local_statistics(
    camera_pairs: np.ndarray,
    semantic_cells: np.ndarray,
    payload: bytes,
) -> np.ndarray:
    """Apply the counted local transform to the scorer-authority second frame."""

    camera = np.asarray(camera_pairs)
    cells = np.asarray(semantic_cells)
    if (
        camera.dtype != np.uint8
        or camera.ndim != 5
        or camera.shape[1:] != (2, *CAMERA_HW, 3)
        or cells.shape != (camera.shape[0], *SEG_HW)
    ):
        raise RealizedFlipMenuError("local-statistics application geometry differs")
    scale, offset = decode_local_statistics(payload)
    bands = scale.shape[1]
    ys = (np.arange(CAMERA_HW[0]) * SEG_HW[0] // CAMERA_HW[0]).clip(
        0, SEG_HW[0] - 1
    )
    xs = (np.arange(CAMERA_HW[1]) * SEG_HW[1] // CAMERA_HW[1]).clip(
        0, SEG_HW[1] - 1
    )
    classes = cells[:, ys[:, None], xs[None, :]].astype(np.intp)
    row_band = (
        np.arange(CAMERA_HW[0], dtype=np.intp) * bands // CAMERA_HW[0]
    )
    local_scale = scale[classes, row_band[None, :, None]]
    local_offset = offset[classes, row_band[None, :, None]]
    result = camera.copy()
    transformed = camera[:, 1].astype(np.float32) * local_scale + local_offset
    result[:, 1] = np.clip(np.rint(transformed), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(result)


def encode_scalar_affine(scale: float, offset: float) -> bytes:
    """Encode one video-derived scalar gain/bias in a 12-byte payload."""

    values = np.asarray([scale, offset], dtype="<f2")
    if not np.all(np.isfinite(values)):
        raise RealizedFlipMenuError("scalar affine must be finite")
    payload = struct.pack("<8s", b"DDMGA1\0\0") + values.tobytes()
    decoded = decode_scalar_affine(payload)
    if decoded != (float(values[0]), float(values[1])):
        raise RealizedFlipMenuError("scalar affine parse-back differs")
    return payload


def decode_scalar_affine(payload: bytes) -> tuple[float, float]:
    if len(payload) != 12 or payload[:8] != b"DDMGA1\0\0":
        raise RealizedFlipMenuError("scalar affine payload differs")
    values = np.frombuffer(payload[8:], dtype="<f2")
    return float(values[0]), float(values[1])


def apply_scalar_affine(camera_pairs: np.ndarray, payload: bytes) -> np.ndarray:
    """Apply one counted scalar gain/bias to frame 1 only."""

    camera = np.asarray(camera_pairs)
    if camera.dtype != np.uint8 or camera.shape[1:] != (2, *CAMERA_HW, 3):
        raise RealizedFlipMenuError("scalar affine camera geometry differs")
    scale, offset = decode_scalar_affine(payload)
    result = camera.copy()
    transformed = result[:, 1].astype(np.float32) * scale + offset
    result[:, 1] = np.clip(np.rint(transformed), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(result)


def encode_temporal_affine(scale: np.ndarray, offset: np.ndarray) -> bytes:
    """Encode piecewise-in-time RGB affine knots as float16."""

    scale_value = np.asarray(scale, dtype="<f2")
    offset_value = np.asarray(offset, dtype="<f2")
    if (
        scale_value.ndim != 2
        or scale_value.shape[1] != 3
        or offset_value.shape != scale_value.shape
        or not 1 <= scale_value.shape[0] <= 600
        or not np.all(np.isfinite(scale_value))
        or not np.all(np.isfinite(offset_value))
    ):
        raise RealizedFlipMenuError("temporal affine knot geometry differs")
    payload = (
        struct.pack("<8sHH", b"DDMTA1\0\0", scale_value.shape[0], 3)
        + scale_value.tobytes()
        + offset_value.tobytes()
    )
    decoded_scale, decoded_offset = decode_temporal_affine(payload)
    if not np.array_equal(scale_value, decoded_scale) or not np.array_equal(
        offset_value, decoded_offset
    ):
        raise RealizedFlipMenuError("temporal affine parse-back differs")
    return payload


def decode_temporal_affine(payload: bytes) -> tuple[np.ndarray, np.ndarray]:
    if len(payload) < 12:
        raise RealizedFlipMenuError("temporal affine payload is truncated")
    magic, knots, channels = struct.unpack("<8sHH", payload[:12])
    expected = 12 + knots * channels * 4
    if (
        magic != b"DDMTA1\0\0"
        or channels != 3
        or not 1 <= knots <= 600
        or len(payload) != expected
    ):
        raise RealizedFlipMenuError("temporal affine payload differs")
    values = np.frombuffer(payload[12:], dtype="<f2")
    count = knots * channels
    return (
        values[:count].reshape(knots, channels).copy(),
        values[count:].reshape(knots, channels).copy(),
    )


def apply_temporal_affine(
    camera_pairs: np.ndarray,
    *,
    pair_ids: Sequence[int],
    pair_count: int,
    payload: bytes,
) -> np.ndarray:
    """Apply deterministic piecewise temporal RGB knots to frame 1 only."""

    camera = np.asarray(camera_pairs)
    ids = np.asarray(pair_ids, dtype=np.int64)
    if (
        camera.dtype != np.uint8
        or camera.shape[1:] != (2, *CAMERA_HW, 3)
        or ids.shape != (camera.shape[0],)
        or pair_count <= 0
        or np.any(ids < 0)
        or np.any(ids >= pair_count)
    ):
        raise RealizedFlipMenuError("temporal affine application geometry differs")
    scale, offset = decode_temporal_affine(payload)
    knot_ids = np.minimum(ids * scale.shape[0] // pair_count, scale.shape[0] - 1)
    result = camera.copy()
    transformed = (
        result[:, 1].astype(np.float32) * scale[knot_ids, None, None]
        + offset[knot_ids, None, None]
    )
    result[:, 1] = np.clip(np.rint(transformed), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(result)


def encode_target_masks(
    rows: Sequence[tuple[int, np.ndarray]],
) -> bytes:
    """Encode sparse scorer-cell masks as independently compressed chunks."""

    header = bytearray(struct.pack("<8sI", b"DDMTM1\0\0", len(rows)))
    body = bytearray()
    expected_start = 0
    for start, mask_value in rows:
        mask = np.asarray(mask_value)
        if (
            mask.dtype != np.bool_
            or mask.ndim != 3
            or mask.shape[1:] != SEG_HW
            or start != expected_start
        ):
            raise RealizedFlipMenuError("target-mask chunk geometry differs")
        packed = np.packbits(mask.reshape(-1), bitorder="little").tobytes()
        compressed = brotli.compress(packed, quality=11, mode=brotli.MODE_GENERIC)
        body.extend(
            struct.pack(
                "<IIIII",
                start,
                start + mask.shape[0],
                mask.size,
                len(packed),
                len(compressed),
            )
        )
        body.extend(compressed)
        expected_start += mask.shape[0]
    payload = bytes(header + body)
    decoded = decode_target_masks(payload)
    if len(decoded) != len(rows) or any(
        start != decoded[index][0]
        or not np.array_equal(np.asarray(mask), decoded[index][1])
        for index, (start, mask) in enumerate(rows)
    ):
        raise RealizedFlipMenuError("target-mask parse-back differs")
    return payload


def decode_target_masks(payload: bytes) -> list[tuple[int, np.ndarray]]:
    if len(payload) < 12:
        raise RealizedFlipMenuError("target-mask payload is truncated")
    magic, chunks = struct.unpack("<8sI", payload[:12])
    if magic != b"DDMTM1\0\0":
        raise RealizedFlipMenuError("target-mask magic differs")
    cursor = 12
    output: list[tuple[int, np.ndarray]] = []
    for _ in range(chunks):
        if cursor + 20 > len(payload):
            raise RealizedFlipMenuError("target-mask chunk header is truncated")
        start, stop, bit_count, packed_bytes, compressed_bytes = struct.unpack(
            "<IIIII", payload[cursor : cursor + 20]
        )
        cursor += 20
        compressed = payload[cursor : cursor + compressed_bytes]
        if len(compressed) != compressed_bytes:
            raise RealizedFlipMenuError("target-mask chunk is truncated")
        cursor += compressed_bytes
        packed = brotli.decompress(compressed)
        if len(packed) != packed_bytes or bit_count != (stop - start) * np.prod(SEG_HW):
            raise RealizedFlipMenuError("target-mask chunk dimensions differ")
        bits = np.unpackbits(
            np.frombuffer(packed, dtype=np.uint8), bitorder="little"
        )[:bit_count]
        output.append(
            (start, bits.astype(bool).reshape(stop - start, *SEG_HW))
        )
    if cursor != len(payload):
        raise RealizedFlipMenuError("target-mask payload has trailing bytes")
    return output


def transition_counts(
    *, before: np.ndarray, after: np.ndarray, target: np.ndarray
) -> dict[str, int]:
    arrays = tuple(np.asarray(value) for value in (before, after, target))
    if any(value.shape != arrays[0].shape for value in arrays[1:]):
        raise RealizedFlipMenuError("transition arrays must share a shape")
    before_wrong = arrays[0] != arrays[2]
    after_wrong = arrays[1] != arrays[2]
    corrected = int(np.count_nonzero(before_wrong & ~after_wrong))
    introduced = int(np.count_nonzero(~before_wrong & after_wrong))
    errors_before = int(np.count_nonzero(before_wrong))
    errors_after = int(np.count_nonzero(after_wrong))
    if errors_after != errors_before - corrected + introduced:
        raise RealizedFlipMenuError("transition conservation failed")
    return {
        "errors_before": errors_before,
        "errors_after": errors_after,
        "errors_corrected": corrected,
        "errors_introduced": introduced,
        "errors_persisting": int(np.count_nonzero(before_wrong & after_wrong)),
        "delta_errors_realized": errors_before - errors_after,
    }


def advisory_objective(*, errors: int, sites: int, d_pose: float, bytes_: int) -> float:
    if sites <= 0 or d_pose < 0.0 or bytes_ < 0:
        raise RealizedFlipMenuError("objective inputs are outside their domains")
    return 100.0 * errors / sites + math.sqrt(10.0 * d_pose) + 25.0 * bytes_ / SOURCE_BYTES


def greedy_telescoping_curve(
    *,
    base: Mapping[str, Any],
    proposals: Iterable[Mapping[str, Any]],
    byte_budget: int,
) -> list[dict[str, Any]]:
    """Admit a dependency-ordered chain only after exact joint remeasurement."""

    if byte_budget <= 0:
        raise RealizedFlipMenuError("byte budget must be positive")
    current = dict(base)
    curve = [{**current, "admitted": True, "admission_reason": "BASE"}]
    for proposal_value in proposals:
        proposal = dict(proposal_value)
        if proposal["parent_candidate_id"] != current["candidate_id"]:
            raise RealizedFlipMenuError("proposal parent is not the current exact state")
        within_budget = int(proposal["archive_bytes"]) <= byte_budget
        improves = float(proposal["advisory_objective"]) < float(
            current["advisory_objective"]
        )
        admitted = within_budget and improves
        curve.append(
            {
                **proposal,
                "admitted": admitted,
                "admission_gates": {
                    "within_byte_budget": within_budget,
                    "strict_joint_improvement": improves,
                },
                "admission_reason": (
                    "STRICT_JOINT_IMPROVEMENT"
                    if admitted
                    else (
                        "BYTE_BUDGET_AND_NO_JOINT_GAIN"
                        if not within_budget and not improves
                        else (
                            "BYTE_BUDGET"
                            if not within_budget
                            else "NO_JOINT_GAIN"
                        )
                    )
                ),
            }
        )
        if admitted:
            current = proposal
    return curve


def _require_advisory_receipt(
    receipt: Mapping[str, Any],
    *,
    schema: str,
    label: str,
) -> None:
    if receipt.get("schema") != schema:
        raise RealizedFlipMenuError(f"{label} schema differs")
    if receipt.get("score_claim") is not False:
        raise RealizedFlipMenuError(f"{label} must be score_claim=false")
    pointer_moved = receipt.get("pointer_moved")
    if pointer_moved is None and isinstance(receipt.get("pointer"), Mapping):
        pointer_moved = receipt["pointer"].get("moved")
    if pointer_moved is not False:
        raise RealizedFlipMenuError(f"{label} cannot claim a pointer move")
    if not str(receipt.get("evidence_axis", "")).startswith("[macOS-CPU"):
        raise RealizedFlipMenuError(f"{label} evidence axis differs")


def _pointer_score_axis(receipt: Mapping[str, Any]) -> tuple[float, str]:
    pointer = receipt.get("pointer")
    if isinstance(pointer, str):
        score_text, axis = pointer.split(" ", 1)
        return float(score_text), axis
    if isinstance(pointer, Mapping):
        if "score" in pointer and "axis" in pointer:
            return float(pointer["score"]), str(pointer["axis"])
        if "contest_cpu" in pointer:
            return float(pointer["contest_cpu"]), "[contest-CPU]"
    raise RealizedFlipMenuError("pointer identity is not typed")


def compile_postcharter_addendum(
    *,
    menu1: Mapping[str, Any],
    mc1: Mapping[str, Any],
    ws1: Mapping[str, Any],
    rd1: Mapping[str, Any],
    e4: Mapping[str, Any],
    input_sha256: Mapping[str, str],
) -> dict[str, Any]:
    """Join later measured rows without laundering base or rate domains."""

    _require_advisory_receipt(
        menu1,
        schema="ddm_menu1_realized_flip_menu_measurement.v1",
        label="MENU1",
    )
    _require_advisory_receipt(
        mc1,
        schema="ddm_mc1_hood_static_reassert_measurement.v1",
        label="MC1",
    )
    _require_advisory_receipt(
        ws1,
        schema="ddm_ws1_seglex96_filtered_warmstart_measurement.v1",
        label="WS1",
    )
    _require_advisory_receipt(
        rd1,
        schema="ddm_rd1_lambda_continuation_frontier_receipt.v4",
        label="RD1",
    )
    _require_advisory_receipt(
        e4,
        schema="ddm_e4_brotli_rate_recovery_receipt.v1",
        label="E4",
    )
    if set(input_sha256) != {"menu1", "mc1", "ws1", "rd1", "e4"} or any(
        len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        for value in input_sha256.values()
    ):
        raise RealizedFlipMenuError("post-charter input hash custody differs")

    box = menu1["box"]
    target_errors = int(box["target_errors"])
    target_d_seg = float(box["d_seg_max"])
    byte_budget = int(box["archive_bytes_max"])
    target_d_pose = float(box["d_pose_max"])
    pointer_score, pointer_axis = _pointer_score_axis(menu1)
    pointer = f"{pointer_score:.10f} {pointer_axis}"
    if any(
        _pointer_score_axis(receipt) != (pointer_score, pointer_axis)
        for receipt in (mc1, ws1, rd1, e4)
    ):
        raise RealizedFlipMenuError("post-charter pointer custody differs")
    menu_curve = {row["candidate_id"]: row for row in menu1["curve"]}
    if len(menu_curve) != len(menu1["curve"]):
        raise RealizedFlipMenuError("MENU1 curve candidate identities differ")
    joint = menu_curve["statistics_hard_analytic_composed_frame1"]
    ws_joint = ws1["warm_start_candidates"]["W_joint"]
    joint_fields = ("candidate_id", "archive_bytes", "errors", "d_seg", "d_pose")
    if any(ws_joint[field] != joint[field] for field in joint_fields):
        raise RealizedFlipMenuError("WS1 W_joint does not bind the MENU1 endpoint")

    mc = mc1["pool_winner"]
    transition = mc["transition_from_parent"]
    if (
        mc["parent_candidate_id"] != joint["candidate_id"]
        or int(transition["errors_before"]) != int(joint["errors"])
        or int(transition["errors_after"]) != int(mc["errors"])
        or int(transition["delta_errors_realized"])
        != int(transition["errors_before"]) - int(transition["errors_after"])
    ):
        raise RealizedFlipMenuError("MC1 telescoping custody differs")
    mc_delta_objective = float(mc["advisory_objective"]) - float(
        joint["advisory_objective"]
    )
    if mc1["waterfill_route"]["admit"] is not False or mc_delta_objective <= 0.0:
        raise RealizedFlipMenuError("MC1 joint-negative admission gate differs")
    mc_partition = dict(mc["byte_partition"])
    if (
        sum(int(mc_partition[key]) for key in ("COUNTED", "FREE", "NULL"))
        != int(mc["archive_bytes"]) - int(joint["archive_bytes"])
    ):
        raise RealizedFlipMenuError("MC1 byte partition does not close")
    mc_partition["law"] = "FREE_UNION_NULL_UNION_COUNTED"
    mc_partition["accounting_role"] = "incremental payload over MENU1 parent"
    mc_row = {
        "schema": ROW_SCHEMA,
        "row_id": "mc1_hood_static_reassert_static_stored_frame1",
        "row_role": "MEASURED_FIX",
        "base_curve_id": "curve:v19c_menu1_joint",
        "price_domain": "V19C_MENU1_EXACT_CHAIN",
        "parent_candidate_id": joint["candidate_id"],
        "candidate_id": mc["candidate_id"],
        "cluster_id": "GLOBAL:MyCar",
        "mechanism_bucket": "MYCAR_STATIC_REASSERT",
        "composition_pool_id": mc["composition_pool_id"],
        "delta_errors_realized": int(transition["delta_errors_realized"]),
        "delta_counted_bytes": int(mc["byte_partition"]["COUNTED"]),
        "byte_partition": mc_partition,
        "collateral_retained": {
            key: int(transition[key])
            for key in (
                "errors_corrected",
                "errors_introduced",
                "errors_persisting",
            )
        },
        "archive_bytes": int(mc["archive_bytes"]),
        "errors": int(mc["errors"]),
        "d_seg": float(mc["d_seg"]),
        "d_pose": float(mc["d_pose"]),
        "advisory_objective": float(mc["advisory_objective"]),
        "delta_advisory_objective": mc_delta_objective,
        "measurement_status": "MEASURED_EXACT_MENU1_PARENT_CHAIN",
        "waterfill_eligible": True,
        "admitted": False,
        "admission_reason": "NO_JOINT_GAIN",
        "evidence_axis": EVIDENCE_AXIS,
        "research_only": True,
        "score_claim": False,
    }

    ws = ws1["warm_start_candidates"]["W_seg"]
    if (
        ws["candidate_id"] != "temporal_affine_16knot_frame1_seglex96_hood_masked"
        or int(ws["errors"]) != sum(
            int(row["errors"]) for row in ws["per_class"].values()
        )
        or int(ws["archive_bytes"]) > byte_budget
    ):
        raise RealizedFlipMenuError("WS1 base-state custody differs")
    ws1_row = {
        "schema": ROW_SCHEMA,
        "row_id": "ws1_seglex96_wseg_base",
        "row_role": "BASE_STATE",
        "base_curve_id": "curve:ws1_seglex96",
        "price_domain": "WS1_SEG_LEXICOGRAPHIC_EXACT_CHAIN",
        "parent_candidate_id": "v19c_base",
        "candidate_id": ws["candidate_id"],
        "mechanism_bucket": "SEG_LEXICOGRAPHIC_WARM_START",
        "composition_pool_id": "base_state:ws1_seglex96",
        "delta_errors_realized_vs_v19c": int(menu_curve["v19c_base"]["errors"])
        - int(ws["errors"]),
        "delta_counted_bytes_vs_v19c": int(ws["archive_bytes"])
        - int(menu_curve["v19c_base"]["archive_bytes"]),
        "byte_partition": {
            "COUNTED": int(ws["delta_payload_bytes"]),
            "FREE": 0,
            "NULL": 0,
            "law": "FREE_UNION_NULL_UNION_COUNTED",
            "accounting_role": "incremental payload over V19C base",
        },
        "archive_bytes": int(ws["archive_bytes"]),
        "errors": int(ws["errors"]),
        "d_seg": float(ws["d_seg"]),
        "d_pose": float(ws["d_pose"]),
        "advisory_objective": float(ws["advisory_objective"]),
        "gap_errors_to_box": int(ws["errors"]) - target_errors,
        "gap_d_seg_to_box": float(ws["d_seg"]) - target_d_seg,
        "per_class": dict(ws["per_class"]),
        "measurement_status": "MEASURED_SEPARATE_BASE_STATE",
        "waterfill_eligible": True,
        "admitted": True,
        "admission_reason": "SEPARATE_BASE_NOT_COMPOSED_WITH_V19C_MENU1_CURVE",
        "evidence_axis": EVIDENCE_AXIS,
        "research_only": True,
        "score_claim": False,
    }

    typed_duals = rd1["duals"]
    if len(typed_duals) != 162:
        raise RealizedFlipMenuError("RD1 typed dual cardinality differs")
    actionable_typed = [
        row for row in typed_duals if row.get("actionable_for_train_decision") is True
    ]
    if actionable_typed:
        raise RealizedFlipMenuError("RD1 typed cells unexpectedly claim actionability")
    aggregate_priors: list[dict[str, Any]] = []
    for control in rd1["aggregate_scalarization_controls"]:
        if (
            control.get("lambda_bytes_per_D") is None
            or control.get("marginal_D_reduction_per_byte") is None
        ):
            raise RealizedFlipMenuError("RD1 aggregate prior is incomplete")
        aggregate_priors.append(
            {
                "row_id": f"rd1_aggregate_dual_{int(control['dual_index']):03d}",
                "row_role": "WATERFILL_PRIOR",
                "left_candidate_id": control["left_candidate_id"],
                "right_candidate_id": control["right_candidate_id"],
                "constraint_group": control["constraint_group"],
                "delta_counted_bytes": int(control["delta_counted_bytes"]),
                "delta_D_realized": float(control["delta_D_realized"]),
                "lambda_bytes_per_D": float(control["lambda_bytes_per_D"]),
                "marginal_D_reduction_per_byte": float(
                    control["marginal_D_reduction_per_byte"]
                ),
                "advisory_only": True,
                "waterfill_eligible": False,
                "blocker": "TYPED_G4_AND_DIMENSION_RATE_HOME_CUSTODY_ABSENT",
                "score_claim": False,
            }
        )
    if len(aggregate_priors) != 3 or len(
        {row["row_id"] for row in aggregate_priors}
    ) != len(aggregate_priors):
        raise RealizedFlipMenuError("RD1 aggregate prior identity differs")

    rate = e4["rate_recovery_vs_e3"]["archive"]
    section_rows = {
        row["section"]: row for row in e4["coder_only_tagged_ab"]["sections"]
    }
    archive_after = e4["archive_custody"]["after_brotli_tagged"]
    if (
        int(rate["after_bytes"]) != int(archive_after["bytes"])
        or int(rate["delta_bytes"])
        != int(rate["after_bytes"]) - int(rate["before_bytes"])
        or e4["section_consumption"]["raw_identity_across_coders"] is not True
        or e4["distortion_trade"]["present"] is not False
    ):
        raise RealizedFlipMenuError("E4 lossless rate custody differs")
    semantic = section_rows["semantic/composed.dds"]
    chart = section_rows["base/chart.ddb"]
    e4_row = {
        "schema": ROW_SCHEMA,
        "row_id": "e4_brotli_q11_e_line_rate",
        "row_role": "RATE_ONLY_CODER",
        "base_curve_id": "curve:e_line_export",
        "price_domain": "E_LINE_EXPORT_ONLY",
        "parent_candidate_id": "e3_lzma1_untagged",
        "candidate_id": "e4_brotli_q11_tagged",
        "mechanism_bucket": "ARCHIVE_LOSSLESS_CODER",
        "composition_pool_id": "e_line_coder",
        "delta_errors_realized": 0,
        "delta_counted_bytes": int(rate["delta_bytes"]),
        "byte_partition": {
            "COUNTED": int(rate["after_bytes"]),
            "FREE": 0,
            "NULL": 0,
            "law": "FREE_UNION_NULL_UNION_COUNTED",
            "accounting_role": "post-coder total; delta stored separately",
        },
        "archive_bytes": int(rate["after_bytes"]),
        "post_coder_section_bytes": {
            "semantic/composed.dds": int(semantic["after_bytes"]),
            "base/chart.ddb": int(chart["after_bytes"]),
        },
        "section_delta_bytes": {
            "semantic/composed.dds": int(semantic["delta_bytes"]),
            "base/chart.ddb": int(chart["delta_bytes"]),
        },
        "semantic_compression_fraction": -float(semantic["delta_bytes"])
        / float(semantic["before_bytes"]),
        "distortion_identity": "DECODED_RAW_BYTE_IDENTICAL",
        "measurement_status": "MEASURED_POST_CODER_COUNTED_BYTES",
        "waterfill_eligible": True,
        "admitted": True,
        "admission_reason": "RATE_IMPROVEMENT_WITHIN_E_LINE_DOMAIN_ONLY",
        "cross_curve_composition_forbidden": True,
        "evidence_axis": e4["evidence_axis"],
        "research_only": True,
        "score_claim": False,
    }

    joint_binding = max(
        joint["per_class"], key=lambda name: int(joint["per_class"][name]["errors"])
    )
    ws1_binding = max(
        ws["per_class"], key=lambda name: int(ws["per_class"][name]["errors"])
    )
    mc_binding = max(
        mc["per_class"], key=lambda name: int(mc["per_class"][name]["errors_after"])
    )
    scorer_curves = (joint, ws)
    any_seg_curve_entered = any(
        float(row["d_seg"]) <= target_d_seg and int(row["archive_bytes"]) <= byte_budget
        for row in scorer_curves
    )
    any_joint_curve_entered = any(
        float(row["d_seg"]) <= target_d_seg
        and float(row["d_pose"]) <= target_d_pose
        and int(row["archive_bytes"]) <= byte_budget
        for row in scorer_curves
    )
    return {
        "schema": POSTCHARTER_SCHEMA,
        "lane_id": "lane_ddm_menu1_realized_flip_menu_20260723",
        "input_sha256": dict(input_sha256),
        "menu_rows": [mc_row, ws1_row, e4_row],
        "waterfill_priors": {
            "rd1_typed_cell_count": len(typed_duals),
            "rd1_actionable_typed_cell_count": 0,
            "typed_cell_status": "NULL_PENDING_JOINT_G4_AND_RATE_HOME_CUSTODY",
            "aggregate_rows": aggregate_priors,
            "advisory_only": True,
        },
        "base_curves": {
            "curve:v19c_menu1_joint": {
                "candidate_id": joint["candidate_id"],
                "archive_bytes": int(joint["archive_bytes"]),
                "errors": int(joint["errors"]),
                "d_seg": float(joint["d_seg"]),
                "d_pose": float(joint["d_pose"]),
                "advisory_objective": float(joint["advisory_objective"]),
                "binding_bucket": joint_binding,
                "gap_errors_to_box": int(joint["errors"]) - target_errors,
                "seg_box_entered": float(joint["d_seg"]) <= target_d_seg,
            },
            "curve:ws1_seglex96": {
                "candidate_id": ws["candidate_id"],
                "archive_bytes": int(ws["archive_bytes"]),
                "errors": int(ws["errors"]),
                "d_seg": float(ws["d_seg"]),
                "d_pose": float(ws["d_pose"]),
                "advisory_objective": float(ws["advisory_objective"]),
                "binding_bucket": ws1_binding,
                "gap_errors_to_box": int(ws["errors"]) - target_errors,
                "seg_box_entered": float(ws["d_seg"]) <= target_d_seg,
            },
            "curve:e_line_export": {
                "candidate_id": e4_row["candidate_id"],
                "archive_bytes": e4_row["archive_bytes"],
                "within_200k": e4_row["archive_bytes"] <= byte_budget,
                "distortion_identity": e4_row["distortion_identity"],
            },
        },
        "residual_routing": {
            "v19c_menu1_joint_binding_bucket": joint_binding,
            "ws1_seglex96_binding_bucket": ws1_binding,
            "mc1_post_reassert_binding_bucket": mc_binding,
            "mc1_admitted": False,
            "next_measurement": (
                "Fisher-margin ranked corrected-inner-Jacobian actuator on each "
                "base curve, preserving curve-local custody"
            ),
        },
        "box": {
            "archive_bytes_max": byte_budget,
            "target_errors": target_errors,
            "d_seg_max": target_d_seg,
            "d_pose_max": target_d_pose,
            "pose_stream_required": bool(box["pose_stream_required"]),
            "any_seg_curve_entered": any_seg_curve_entered,
            "any_joint_curve_entered": any_joint_curve_entered,
            "r6_candidate": any_joint_curve_entered,
        },
        "verdict": "MENU1_POSTCHARTER_JOINED_BOX_NOT_REACHED",
        "verdict_scope": (
            "FORMULATION: measured V19C/MENU1 joint and WS1 seg-lexicographic "
            "base curves plus MC1 static reassert and E4 Brotli Q11 E-line coder; "
            "families and paradigm remain open"
        ),
        "pointer": pointer,
        "pointer_moved": False,
        "evidence_axis": EVIDENCE_AXIS,
        "research_only": True,
        "score_claim": False,
        "promotion_eligible": False,
        "main_landing_review_required": True,
    }


__all__ = [
    "CAMERA_HW",
    "EVIDENCE_AXIS",
    "FIX_SPECS",
    "LOCAL_STATS_SCHEMA",
    "MASK_SCHEMA",
    "MENU_SCHEMA",
    "POSTCHARTER_SCHEMA",
    "RATE_DUAL",
    "ROW_SCHEMA",
    "SEG_HW",
    "RealizedFlipMenuError",
    "advisory_objective",
    "apply_local_statistics",
    "apply_scalar_affine",
    "apply_temporal_affine",
    "cluster_id",
    "compile_menu_rows",
    "compile_postcharter_addendum",
    "decode_local_statistics",
    "decode_scalar_affine",
    "decode_target_masks",
    "decode_temporal_affine",
    "encode_local_statistics",
    "encode_scalar_affine",
    "encode_target_masks",
    "encode_temporal_affine",
    "fit_local_statistics",
    "greedy_telescoping_curve",
    "local_statistics_sufficient_statistics",
    "sha256_bytes",
    "solve_local_statistics",
    "transition_counts",
]
