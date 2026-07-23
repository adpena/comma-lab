# SPDX-License-Identifier: MIT
"""Canonical byte-identity law for the DDM Build #636 runtime exporter."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

EQUATION_ID = "ddm_runtime_export_identity_receiver_closed_v1"
PAINT_JACOBIAN_EQUATION_ID = "ddm_semantic_paint_camera_uint8_jacobian_v1"
REFERENCE_BYTES = 37_545_489


@dataclass(frozen=True, slots=True)
class ExportIdentityResult:
    pair_count: int
    source_sha256: str
    packaged_sha256: str
    byte_identical: bool


def sha256_chunks(chunks: Iterable[bytes]) -> tuple[int, str]:
    """Return exact byte count and SHA-256 for one ordered byte stream."""

    digest = hashlib.sha256()
    total = 0
    for chunk in chunks:
        if not isinstance(chunk, bytes):
            raise TypeError("identity chunks must be immutable bytes")
        digest.update(chunk)
        total += len(chunk)
    return total, digest.hexdigest()


def export_identity(
    *,
    pair_count: int,
    source_bytes: int,
    source_sha256: str,
    packaged_bytes: int,
    packaged_sha256: str,
) -> ExportIdentityResult:
    """Apply ``R_repo(z) == R_package(export(z))`` at exact camera bytes."""

    if isinstance(pair_count, bool) or not isinstance(pair_count, int) or pair_count <= 0:
        raise ValueError("pair_count must be a positive integer")
    for label, value in (("source_bytes", source_bytes), ("packaged_bytes", packaged_bytes)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{label} must be a positive integer")
    for label, value in (("source_sha256", source_sha256), ("packaged_sha256", packaged_sha256)):
        if not (
            isinstance(value, str)
            and len(value) == 64
            and all(character in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"{label} must be a lowercase SHA-256")
    return ExportIdentityResult(
        pair_count=pair_count,
        source_sha256=source_sha256,
        packaged_sha256=packaged_sha256,
        byte_identical=source_bytes == packaged_bytes and source_sha256 == packaged_sha256,
    )


def score_row(*, archive_bytes: int, d_seg: float, d_pose: float) -> dict[str, float]:
    """Recompute the three contest terms without granting score authority."""

    if isinstance(archive_bytes, bool) or not isinstance(archive_bytes, int) or archive_bytes <= 0:
        raise ValueError("archive_bytes must be a positive integer")
    if not (math.isfinite(d_seg) and math.isfinite(d_pose) and d_seg >= 0.0 and d_pose >= 0.0):
        raise ValueError("distortions must be finite and nonnegative")
    seg = 100.0 * d_seg
    pose = math.sqrt(10.0 * d_pose)
    rate = 25.0 * archive_bytes / REFERENCE_BYTES
    return {"seg": seg, "pose": pose, "rate": rate, "total": seg + pose + rate}


def semantic_paint_jacobian_summary(
    labels: np.ndarray,
    palette_rgb_u8: list[list[int]],
    *,
    camera_hw: tuple[int, int],
    frames_per_pair: int,
) -> dict[str, object]:
    """Measure the exact semantic-paint to camera-uint8 Jacobian support.

    The receiver gathers one scorer-grid label with
    ``floor(camera_index * scorer_size / camera_size)`` and paints that role
    colour after base-chart realization.  Consequently a unit palette
    coefficient perturbation changes exactly one output byte per gathered
    camera pixel in that role.  Label reassignment support is the outer product
    of row/column gather multiplicities, repeated across both frames.
    """

    value = np.asarray(labels)
    if value.dtype != np.uint8 or value.ndim != 3 or value.shape[0] <= 0:
        raise ValueError("labels must be a nonempty [pair,h,w] uint8 array")
    if (
        isinstance(frames_per_pair, bool)
        or not isinstance(frames_per_pair, int)
        or frames_per_pair <= 0
    ):
        raise ValueError("frames_per_pair must be a positive integer")
    camera_h, camera_w = camera_hw
    if (
        isinstance(camera_h, bool)
        or isinstance(camera_w, bool)
        or not isinstance(camera_h, int)
        or not isinstance(camera_w, int)
        or camera_h <= 0
        or camera_w <= 0
    ):
        raise ValueError("camera dimensions must be positive integers")
    palette = np.asarray(palette_rgb_u8)
    if (
        palette.dtype.kind not in {"i", "u"}
        or palette.ndim != 2
        or palette.shape[1] != 3
        or palette.shape[0] < 2
        or np.any((palette < 0) | (palette > 255))
        or int(value.max()) >= palette.shape[0]
    ):
        raise ValueError("palette must cover every label with uint8 RGB rows")

    scorer_h, scorer_w = int(value.shape[1]), int(value.shape[2])
    camera_rows = np.floor_divide(
        np.arange(camera_h, dtype=np.int64) * scorer_h, camera_h
    )
    camera_columns = np.floor_divide(
        np.arange(camera_w, dtype=np.int64) * scorer_w, camera_w
    )
    row_multiplicity = np.bincount(camera_rows, minlength=scorer_h)
    column_multiplicity = np.bincount(camera_columns, minlength=scorer_w)
    support = np.multiply.outer(row_multiplicity, column_multiplicity)
    if int(support.sum()) != camera_h * camera_w:
        raise ValueError("camera gather support does not close")

    coefficient_rows = []
    for code in range(1, palette.shape[0]):
        active_camera_pixels = 0
        for start in range(0, value.shape[0], 16):
            mask = value[start : start + 16] == code
            weighted_support = np.broadcast_to(support, mask.shape)
            active_camera_pixels += int(np.sum(weighted_support, where=mask))
        active_camera_pixels *= frames_per_pair
        directions = [
            1 if int(channel) < 255 else -1 for channel in palette[code]
        ]
        coefficient_rows.append(
            {
                "active_camera_pixels": active_camera_pixels,
                "code": code,
                "rgb_u8": [int(channel) for channel in palette[code]],
                "unit_perturbation_direction_rgb": directions,
                "unit_perturbation_linf_uint8": 1,
                "unit_perturbation_output_bytes_changed_per_channel": (
                    active_camera_pixels
                ),
            }
        )
    painted_pixels = sum(
        int(row["active_camera_pixels"]) for row in coefficient_rows
    )
    return {
        "camera_hw": [camera_h, camera_w],
        "coefficient_rows": coefficient_rows,
        "equation_id": PAINT_JACOBIAN_EQUATION_ID,
        "frames_per_pair": frames_per_pair,
        "label_assignment_preimage_camera_pixels": {
            "max_per_cell_all_frames": int(support.max()) * frames_per_pair,
            "min_per_cell_all_frames": int(support.min()) * frames_per_pair,
            "sum_per_pair_all_frames": int(support.sum()) * frames_per_pair,
        },
        "pair_count": int(value.shape[0]),
        "painted_camera_pixels_all_pairs_all_frames": painted_pixels,
        "quantization": "exact_uint8_unit_step_no_clipping",
        "receiver_map": (
            "semantic_label -> floor-index camera gather -> role RGB overwrite"
        ),
        "status": "DERIVED_MAP_AND_MEASURED_NAMED_STATE_SUPPORT",
    }


def describe() -> dict[str, object]:
    return {
        "equation_id": EQUATION_ID,
        "equation": "R_repo(z).camera_bytes == R_package(export(z)).camera_bytes",
        "domain": (
            "one named composed DDM state, exact counted archive bytes, and the reviewed "
            "generic packaged receiver; identity transfers a scorer row only when that same "
            "source state's row has frozen-scorer custody"
        ),
        "empirical_verification_status": (
            "MEASURED_EXACT_N600_BUILD_636_SOURCE_AND_PACKAGED_RAW_SHA256_IDENTICAL"
        ),
        "evidence_axis": "[macOS-CPU frozen-scorer advisory]",
        "verification_receipt": (
            ".omx/research/ddm_e1_runtime_exporter_n600_20260723/"
            "ddm_e1_runtime_verification_receipt.json"
        ),
        "score_claim": False,
        "supporting_equation_ids": [PAINT_JACOBIAN_EQUATION_ID],
    }


__all__ = [
    "EQUATION_ID",
    "PAINT_JACOBIAN_EQUATION_ID",
    "REFERENCE_BYTES",
    "ExportIdentityResult",
    "describe",
    "export_identity",
    "score_row",
    "semantic_paint_jacobian_summary",
    "sha256_chunks",
]
