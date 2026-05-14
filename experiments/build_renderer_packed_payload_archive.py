#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a lossless single-member renderer payload archive.

This is a no-retraining, byte-only transform.  It reads an existing renderer
archive, concatenates the logical runtime members behind a validated header,
Brotli-compresses that payload, and writes a deterministic archive containing
``renderer_payload.bin.br`` or the shorter top-submission-compatible ``p``
member.  The submission inflate path expands the payload
back into the original members before normal renderer inflate.

The resulting archive is not a score claim until exact CUDA auth eval is run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import zipfile
from pathlib import Path
from typing import Any


MAGIC = b"RPK1"
COMPACT_MAGIC = b"RP2\x01"
SCHEMA = "renderer_payload_v1"
COMPACT_SCHEMA = "renderer_payload_fixed3_v1"
PR64_LEN_TABLE_SCHEMA = "renderer_payload_pr64_len_table_v1"
HEADER_STRUCT = "<I"
COMPACT_HEADER_STRUCT = "<B3xIII"
PR64_LEN_TABLE_STRUCT = "<III"
FIXED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
PAYLOAD_MEMBER_NAME = "renderer_payload.bin.br"
SHORT_PAYLOAD_MEMBER_NAME = "p"
ALLOWED_PAYLOAD_MEMBER_NAMES = (PAYLOAD_MEMBER_NAME, SHORT_PAYLOAD_MEMBER_NAME)
PAYLOAD_FORMAT_RPK1_JSON = "rpk1_json"
PAYLOAD_FORMAT_RP2_FIXED3 = "rp2_fixed3"
PAYLOAD_FORMAT_PR64_LEN_TABLE = "pr64_len_table"
PAYLOAD_FORMAT_PUBLIC_PR64_MASK_FIRST_LEN_TABLE = "public_pr64_mask_first_len_table"
ALLOWED_PAYLOAD_FORMATS = (
    PAYLOAD_FORMAT_RPK1_JSON,
    PAYLOAD_FORMAT_RP2_FIXED3,
    PAYLOAD_FORMAT_PR64_LEN_TABLE,
    PAYLOAD_FORMAT_PUBLIC_PR64_MASK_FIRST_LEN_TABLE,
)
ORIGINAL_VIDEO_BYTES = 37_545_489
POSE_FP16_COL_DELTA_CODEC = "pose_fp16_col_delta_v1"
POSE_QPOSE14_COL_DELTA_CODEC = "pose_qpose14_col_delta_v1"
POSE_QP1_CODEC = "pose_qp1_v1"
POSE_FP16_VELOCITY_ONLY_CODEC = "pose_fp16_velocity_only_v1"
POSE_PUBLIC_PR64_VELOCITY_DELTA_CODEC = "pose_public_pr64_velocity_delta_v1"
POSE_FP16_VELOCITY_RESIDUAL_TOPK_CODEC = "pose_fp16_velocity_residual_topk_v1"
COMPACT_POSE_CODEC_IDS = {
    "raw": 0,
    POSE_FP16_COL_DELTA_CODEC: 1,
    POSE_QPOSE14_COL_DELTA_CODEC: 2,
    POSE_QP1_CODEC: 3,
}

DEFAULT_MEMBER_ORDER = (
    "renderer.bin",
    "masks.mkv",
    "grayscale.mkv",
    "masks.alpha4.mkv",
    "masks.amrc",
    "masks.nrv",
    "masks.cmg2",
    "masks.cmg3",
    "masks.cdo1",
    "masks.cdo1.xz",
    "masks.cdo1.zlib",
    "masks.cdo1.br",
    "optimized_poses.bin",
    "optimized_poses.pt",
    "optimized_embedding.pt",
    "poses.pt",
    "corrections.bin",
    "gradient_corrections.bin",
    "mini_segnet.bin",
    "mini_posenet.bin",
    "posenet_targets.bin",
    "zoom_scalars.bin",
    "foveation_params.bin",
    "alpha4_residual_repair.amr1",
    "alpha4_residual_repair.amr1.xz",
    "alpha4_residual_repair.amr1.zlib",
    "alpha4_residual_repair.amr1.br",
)


def _validate_member_name(name: str) -> None:
    path = Path(name)
    if not name or name.startswith("/") or ".." in path.parts or len(path.parts) != 1:
        raise ValueError(f"unsafe archive member path: {name!r}")


def read_source_members(source_archive: Path) -> dict[str, bytes]:
    """Read safe top-level members from a ZIP archive in source order."""
    if not source_archive.exists():
        raise FileNotFoundError(f"source archive not found: {source_archive}")
    try:
        with zipfile.ZipFile(source_archive, "r") as zf:
            infos = [info for info in zf.infolist() if not info.is_dir()]
            if not infos:
                raise ValueError(f"source archive is empty: {source_archive}")
            members: dict[str, bytes] = {}
            for info in infos:
                _validate_member_name(info.filename)
                if info.filename in members:
                    raise ValueError(f"duplicate source archive member: {info.filename}")
                members[info.filename] = zf.read(info)
    except zipfile.BadZipFile as exc:
        raise ValueError(f"not a valid zip archive: {source_archive}") from exc
    return members


def ordered_runtime_members(
    members: dict[str, bytes],
    *,
    explicit_order: list[str] | None = None,
) -> list[tuple[str, bytes]]:
    """Return deterministic runtime members to pack.

    When ``explicit_order`` is omitted, known renderer members are emitted in
    a stable semantic order and any additional safe top-level members follow
    lexicographically.  This keeps the format extensible without relying on
    host ZIP member order.
    """
    order = explicit_order if explicit_order is not None else list(DEFAULT_MEMBER_ORDER)
    missing = [name for name in order if name not in members and explicit_order is not None]
    if missing:
        raise ValueError(f"explicit member order contains missing members: {missing}")

    selected_names: list[str] = []
    for name in order:
        if name in members:
            selected_names.append(name)
    extras = sorted(name for name in members if name not in selected_names)
    for name in extras:
        _validate_member_name(name)
    selected_names.extend(extras)

    if "renderer.bin" not in selected_names:
        raise ValueError("packed renderer archive requires renderer.bin")
    if not any(name in selected_names for name in ("masks.mkv", "grayscale.mkv", "masks.alpha4.mkv", "masks.amrc", "masks.nrv", "masks.cmg2", "masks.cmg3")):
        raise ValueError("packed renderer archive requires one mask payload")
    if not any(name in selected_names for name in ("optimized_poses.bin", "optimized_poses.pt", "zoom_scalars.bin")):
        raise ValueError("packed renderer archive requires pose or zoom payload")

    return [(name, members[name]) for name in selected_names]


def build_renderer_payload(
    ordered_members: list[tuple[str, bytes]],
    *,
    source_archive_sha256: str | None = None,
    pose_codec: str = "raw",
    pose_residual_topk: int = 0,
) -> tuple[bytes, dict[str, Any]]:
    """Build the raw ``RPK1`` payload and return payload plus header."""
    encoded_members: list[tuple[str, bytes, dict[str, Any]]] = []
    for name, data in ordered_members:
        meta_extra: dict[str, Any] = {"codec": "raw"}
        encoded = data
        if name == "optimized_poses.bin" and pose_codec == POSE_FP16_COL_DELTA_CODEC:
            encoded = encode_pose_fp16_col_delta(data)
            meta_extra = {
                "codec": POSE_FP16_COL_DELTA_CODEC,
                "decoded_bytes": len(data),
                "decoded_sha256": hashlib.sha256(data).hexdigest(),
            }
        elif name == "optimized_poses.bin" and pose_codec == POSE_QPOSE14_COL_DELTA_CODEC:
            encoded, reconstructed = encode_pose_qpose14_col_delta(data)
            meta_extra = {
                "codec": POSE_QPOSE14_COL_DELTA_CODEC,
                "decoded_bytes": len(reconstructed),
                "decoded_sha256": hashlib.sha256(reconstructed).hexdigest(),
                "source_decoded_sha256": hashlib.sha256(data).hexdigest(),
                "lossy": True,
                "pose_error_stats": pose_error_stats(data, reconstructed),
            }
        elif name == "optimized_poses.bin" and pose_codec == POSE_QP1_CODEC:
            encoded, reconstructed = encode_pose_qp1(data)
            meta_extra = {
                "codec": POSE_QP1_CODEC,
                "decoded_bytes": len(reconstructed),
                "decoded_sha256": hashlib.sha256(reconstructed).hexdigest(),
                "source_decoded_sha256": hashlib.sha256(data).hexdigest(),
                "lossy": True,
                "pose_error_stats": pose_error_stats(data, reconstructed),
            }
        elif name == "optimized_poses.bin" and pose_codec == POSE_FP16_VELOCITY_ONLY_CODEC:
            encoded, reconstructed = encode_pose_fp16_velocity_only(data)
            meta_extra = {
                "codec": POSE_FP16_VELOCITY_ONLY_CODEC,
                "decoded_bytes": len(reconstructed),
                "decoded_sha256": hashlib.sha256(reconstructed).hexdigest(),
                "source_decoded_sha256": hashlib.sha256(data).hexdigest(),
                "lossy": True,
                "pose_error_stats": pose_error_stats(data, reconstructed),
            }
        elif name == "optimized_poses.bin" and pose_codec == POSE_FP16_VELOCITY_RESIDUAL_TOPK_CODEC:
            encoded, reconstructed = encode_pose_fp16_velocity_residual_topk(
                data,
                topk=pose_residual_topk,
            )
            meta_extra = {
                "codec": POSE_FP16_VELOCITY_RESIDUAL_TOPK_CODEC,
                "decoded_bytes": len(reconstructed),
                "decoded_sha256": hashlib.sha256(reconstructed).hexdigest(),
                "source_decoded_sha256": hashlib.sha256(data).hexdigest(),
                "lossy": True,
                "pose_residual_topk": pose_residual_topk,
                "pose_error_stats": pose_error_stats(data, reconstructed),
            }
        elif name == "optimized_poses.bin" and pose_codec != "raw":
            raise ValueError(f"unsupported pose codec: {pose_codec!r}")
        encoded_members.append((name, encoded, meta_extra))

    header = {
        "schema": SCHEMA,
        "source_archive_sha256": source_archive_sha256,
        "members": [
            {
                "name": name,
                "bytes": len(encoded),
                "sha256": hashlib.sha256(encoded).hexdigest(),
                **meta_extra,
            }
            for name, encoded, meta_extra in encoded_members
        ],
    }
    header_bytes = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload = (
        MAGIC
        + struct.pack(HEADER_STRUCT, len(header_bytes))
        + header_bytes
        + b"".join(encoded for _, encoded, _ in encoded_members)
    )
    return payload, header


def build_compact_renderer_payload(
    ordered_members: list[tuple[str, bytes]],
    *,
    source_archive_sha256: str | None = None,
    pose_codec: str = "raw",
) -> tuple[bytes, dict[str, Any]]:
    """Build a compact fixed-slice payload for renderer/masks/poses.

    This mirrors the public top-submission pattern: one member named ``p`` and
    a tiny binary header with fixed runtime member order.  It is intentionally
    less extensible than RPK1 and should only be used for the canonical
    renderer.bin + masks.mkv + optimized_poses.bin archive shape.
    """
    members = dict(ordered_members)
    required = ("renderer.bin", "masks.mkv", "optimized_poses.bin")
    if tuple(name for name, _ in ordered_members) != required:
        raise ValueError(
            f"{PAYLOAD_FORMAT_RP2_FIXED3} requires exact member order {required}, "
            f"got {tuple(name for name, _ in ordered_members)}"
        )
    if pose_codec not in COMPACT_POSE_CODEC_IDS:
        raise ValueError(
            f"{PAYLOAD_FORMAT_RP2_FIXED3} supports pose codecs "
            f"{tuple(COMPACT_POSE_CODEC_IDS)}, got {pose_codec!r}"
        )

    renderer = members["renderer.bin"]
    masks = members["masks.mkv"]
    source_pose = members["optimized_poses.bin"]
    pose_meta: dict[str, Any] = {"codec": "raw"}
    pose_encoded = source_pose
    if pose_codec == POSE_FP16_COL_DELTA_CODEC:
        pose_encoded = encode_pose_fp16_col_delta(source_pose)
        pose_meta = {
            "codec": POSE_FP16_COL_DELTA_CODEC,
            "decoded_bytes": len(source_pose),
            "decoded_sha256": hashlib.sha256(source_pose).hexdigest(),
        }
    elif pose_codec == POSE_QPOSE14_COL_DELTA_CODEC:
        pose_encoded, reconstructed = encode_pose_qpose14_col_delta(source_pose)
        pose_meta = {
            "codec": POSE_QPOSE14_COL_DELTA_CODEC,
            "decoded_bytes": len(reconstructed),
            "decoded_sha256": hashlib.sha256(reconstructed).hexdigest(),
            "source_decoded_sha256": hashlib.sha256(source_pose).hexdigest(),
            "lossy": True,
            "pose_error_stats": pose_error_stats(source_pose, reconstructed),
        }
    elif pose_codec == POSE_QP1_CODEC:
        pose_encoded, reconstructed = encode_pose_qp1(source_pose)
        pose_meta = {
            "codec": POSE_QP1_CODEC,
            "decoded_bytes": len(reconstructed),
            "decoded_sha256": hashlib.sha256(reconstructed).hexdigest(),
            "source_decoded_sha256": hashlib.sha256(source_pose).hexdigest(),
            "lossy": True,
            "pose_error_stats": pose_error_stats(source_pose, reconstructed),
        }

    header = {
        "schema": COMPACT_SCHEMA,
        "source_archive_sha256": source_archive_sha256,
        "payload_format": PAYLOAD_FORMAT_RP2_FIXED3,
        "members": [
            {
                "name": "renderer.bin",
                "bytes": len(renderer),
                "sha256": hashlib.sha256(renderer).hexdigest(),
                "codec": "raw",
            },
            {
                "name": "masks.mkv",
                "bytes": len(masks),
                "sha256": hashlib.sha256(masks).hexdigest(),
                "codec": "raw",
            },
            {
                "name": "optimized_poses.bin",
                "bytes": len(pose_encoded),
                "sha256": hashlib.sha256(pose_encoded).hexdigest(),
                **pose_meta,
            },
        ],
    }
    payload = (
        COMPACT_MAGIC
        + struct.pack(
            COMPACT_HEADER_STRUCT,
            COMPACT_POSE_CODEC_IDS[pose_codec],
            len(renderer),
            len(masks),
            len(pose_encoded),
        )
        + renderer
        + masks
        + pose_encoded
    )
    return payload, header


def build_pr64_len_table_payload(
    ordered_members: list[tuple[str, bytes]],
    *,
    source_archive_sha256: str | None = None,
    pose_codec: str = "raw",
    pose_residual_topk: int = 0,
) -> tuple[bytes, dict[str, Any]]:
    """Build the public PR64-style single-Brotli length-table payload.

    The decompressed payload is ``<III> + renderer + masks + pose``.  Unlike
    RP2, the table has no magic or codec byte; the runtime decodes pose bytes
    by their own magic when a lossy pose codec is used, otherwise it treats the
    pose member as raw fp16 bytes.
    """

    members = dict(ordered_members)
    required = ("renderer.bin", "masks.mkv", "optimized_poses.bin")
    if tuple(name for name, _ in ordered_members) != required:
        raise ValueError(
            f"{PAYLOAD_FORMAT_PR64_LEN_TABLE} requires exact member order {required}, "
            f"got {tuple(name for name, _ in ordered_members)}"
        )

    renderer = members["renderer.bin"]
    masks = members["masks.mkv"]
    source_pose = members["optimized_poses.bin"]
    pose_meta: dict[str, Any] = {"codec": "raw"}
    pose_encoded = source_pose
    if pose_codec == POSE_FP16_COL_DELTA_CODEC:
        pose_encoded = encode_pose_fp16_col_delta(source_pose)
        pose_meta = {
            "codec": POSE_FP16_COL_DELTA_CODEC,
            "decoded_bytes": len(source_pose),
            "decoded_sha256": hashlib.sha256(source_pose).hexdigest(),
        }
    elif pose_codec == POSE_QPOSE14_COL_DELTA_CODEC:
        pose_encoded, reconstructed = encode_pose_qpose14_col_delta(source_pose)
        pose_meta = {
            "codec": POSE_QPOSE14_COL_DELTA_CODEC,
            "decoded_bytes": len(reconstructed),
            "decoded_sha256": hashlib.sha256(reconstructed).hexdigest(),
            "source_decoded_sha256": hashlib.sha256(source_pose).hexdigest(),
            "lossy": True,
            "pose_error_stats": pose_error_stats(source_pose, reconstructed),
        }
    elif pose_codec == POSE_QP1_CODEC:
        pose_encoded, reconstructed = encode_pose_qp1(source_pose)
        pose_meta = {
            "codec": POSE_QP1_CODEC,
            "decoded_bytes": len(reconstructed),
            "decoded_sha256": hashlib.sha256(reconstructed).hexdigest(),
            "source_decoded_sha256": hashlib.sha256(source_pose).hexdigest(),
            "lossy": True,
            "pose_error_stats": pose_error_stats(source_pose, reconstructed),
        }
    elif pose_codec == POSE_FP16_VELOCITY_ONLY_CODEC:
        pose_encoded, reconstructed = encode_pose_fp16_velocity_only(source_pose)
        pose_meta = {
            "codec": POSE_FP16_VELOCITY_ONLY_CODEC,
            "decoded_bytes": len(reconstructed),
            "decoded_sha256": hashlib.sha256(reconstructed).hexdigest(),
            "source_decoded_sha256": hashlib.sha256(source_pose).hexdigest(),
            "lossy": True,
            "pose_error_stats": pose_error_stats(source_pose, reconstructed),
        }
    elif pose_codec == POSE_PUBLIC_PR64_VELOCITY_DELTA_CODEC:
        pose_encoded, reconstructed = encode_pose_public_pr64_velocity_delta(source_pose)
        pose_meta = {
            "codec": POSE_PUBLIC_PR64_VELOCITY_DELTA_CODEC,
            "decoded_bytes": len(reconstructed),
            "decoded_sha256": hashlib.sha256(reconstructed).hexdigest(),
            "source_decoded_sha256": hashlib.sha256(source_pose).hexdigest(),
            "lossy": True,
            "pose_error_stats": pose_error_stats(source_pose, reconstructed),
        }
    elif pose_codec == POSE_FP16_VELOCITY_RESIDUAL_TOPK_CODEC:
        pose_encoded, reconstructed = encode_pose_fp16_velocity_residual_topk(
            source_pose,
            topk=pose_residual_topk,
        )
        pose_meta = {
            "codec": POSE_FP16_VELOCITY_RESIDUAL_TOPK_CODEC,
            "decoded_bytes": len(reconstructed),
            "decoded_sha256": hashlib.sha256(reconstructed).hexdigest(),
            "source_decoded_sha256": hashlib.sha256(source_pose).hexdigest(),
            "lossy": True,
            "pose_residual_topk": pose_residual_topk,
            "pose_error_stats": pose_error_stats(source_pose, reconstructed),
        }
    elif pose_codec != "raw":
        raise ValueError(f"unsupported {PAYLOAD_FORMAT_PR64_LEN_TABLE} pose codec: {pose_codec!r}")

    header = {
        "schema": PR64_LEN_TABLE_SCHEMA,
        "source_archive_sha256": source_archive_sha256,
        "payload_format": PAYLOAD_FORMAT_PR64_LEN_TABLE,
        "members": [
            {
                "name": "renderer.bin",
                "bytes": len(renderer),
                "sha256": hashlib.sha256(renderer).hexdigest(),
                "codec": "raw",
            },
            {
                "name": "masks.mkv",
                "bytes": len(masks),
                "sha256": hashlib.sha256(masks).hexdigest(),
                "codec": "raw",
            },
            {
                "name": "optimized_poses.bin",
                "bytes": len(pose_encoded),
                "sha256": hashlib.sha256(pose_encoded).hexdigest(),
                **pose_meta,
            },
        ],
    }
    payload = (
        struct.pack(PR64_LEN_TABLE_STRUCT, len(renderer), len(masks), len(pose_encoded))
        + renderer
        + masks
        + pose_encoded
    )
    return payload, header


def build_public_pr64_mask_first_len_table_payload(
    ordered_members: list[tuple[str, bytes]],
    *,
    source_archive_sha256: str | None = None,
    pose_codec: str = "raw",
    pose_residual_topk: int = 0,
) -> tuple[bytes, dict[str, Any]]:
    """Build the public PR64 mask-first single-Brotli length-table payload.

    Public ``unified_brotli`` writes ``<mask_len, model_len, pose_len>``
    followed by ``masks.mkv + renderer.bin + pose``.  This format is preferred
    for deploy candidates once PR64-basin parity is established because it
    keeps our generated payload in the same parser branch as the public floor.
    """

    members = dict(ordered_members)
    required = ("renderer.bin", "masks.mkv", "optimized_poses.bin")
    if tuple(name for name, _ in ordered_members) != required:
        raise ValueError(
            f"{PAYLOAD_FORMAT_PUBLIC_PR64_MASK_FIRST_LEN_TABLE} requires exact member order {required}, "
            f"got {tuple(name for name, _ in ordered_members)}"
        )

    renderer = members["renderer.bin"]
    masks = members["masks.mkv"]
    source_pose = members["optimized_poses.bin"]
    pose_meta: dict[str, Any] = {"codec": "raw"}
    pose_encoded = source_pose
    if pose_codec == POSE_PUBLIC_PR64_VELOCITY_DELTA_CODEC:
        pose_encoded, reconstructed = encode_pose_public_pr64_velocity_delta(source_pose)
        pose_meta = {
            "codec": POSE_PUBLIC_PR64_VELOCITY_DELTA_CODEC,
            "decoded_bytes": len(reconstructed),
            "decoded_sha256": hashlib.sha256(reconstructed).hexdigest(),
            "source_decoded_sha256": hashlib.sha256(source_pose).hexdigest(),
            "lossy": True,
            "pose_error_stats": pose_error_stats(source_pose, reconstructed),
        }
    elif pose_codec == POSE_FP16_COL_DELTA_CODEC:
        pose_encoded = encode_pose_fp16_col_delta(source_pose)
        pose_meta = {
            "codec": POSE_FP16_COL_DELTA_CODEC,
            "decoded_bytes": len(source_pose),
            "decoded_sha256": hashlib.sha256(source_pose).hexdigest(),
        }
    elif pose_codec == POSE_QPOSE14_COL_DELTA_CODEC:
        pose_encoded, reconstructed = encode_pose_qpose14_col_delta(source_pose)
        pose_meta = {
            "codec": POSE_QPOSE14_COL_DELTA_CODEC,
            "decoded_bytes": len(reconstructed),
            "decoded_sha256": hashlib.sha256(reconstructed).hexdigest(),
            "source_decoded_sha256": hashlib.sha256(source_pose).hexdigest(),
            "lossy": True,
            "pose_error_stats": pose_error_stats(source_pose, reconstructed),
        }
    elif pose_codec == POSE_QP1_CODEC:
        pose_encoded, reconstructed = encode_pose_qp1(source_pose)
        pose_meta = {
            "codec": POSE_QP1_CODEC,
            "decoded_bytes": len(reconstructed),
            "decoded_sha256": hashlib.sha256(reconstructed).hexdigest(),
            "source_decoded_sha256": hashlib.sha256(source_pose).hexdigest(),
            "lossy": True,
            "pose_error_stats": pose_error_stats(source_pose, reconstructed),
        }
    elif pose_codec == POSE_FP16_VELOCITY_ONLY_CODEC:
        pose_encoded, reconstructed = encode_pose_fp16_velocity_only(source_pose)
        pose_meta = {
            "codec": POSE_FP16_VELOCITY_ONLY_CODEC,
            "decoded_bytes": len(reconstructed),
            "decoded_sha256": hashlib.sha256(reconstructed).hexdigest(),
            "source_decoded_sha256": hashlib.sha256(source_pose).hexdigest(),
            "lossy": True,
            "pose_error_stats": pose_error_stats(source_pose, reconstructed),
        }
    elif pose_codec == POSE_FP16_VELOCITY_RESIDUAL_TOPK_CODEC:
        pose_encoded, reconstructed = encode_pose_fp16_velocity_residual_topk(
            source_pose,
            topk=pose_residual_topk,
        )
        pose_meta = {
            "codec": POSE_FP16_VELOCITY_RESIDUAL_TOPK_CODEC,
            "decoded_bytes": len(reconstructed),
            "decoded_sha256": hashlib.sha256(reconstructed).hexdigest(),
            "source_decoded_sha256": hashlib.sha256(source_pose).hexdigest(),
            "lossy": True,
            "pose_residual_topk": pose_residual_topk,
            "pose_error_stats": pose_error_stats(source_pose, reconstructed),
        }
    elif pose_codec != "raw":
        raise ValueError(
            f"unsupported {PAYLOAD_FORMAT_PUBLIC_PR64_MASK_FIRST_LEN_TABLE} pose codec: {pose_codec!r}"
        )

    header = {
        "schema": PR64_LEN_TABLE_SCHEMA,
        "source_archive_sha256": source_archive_sha256,
        "payload_format": PAYLOAD_FORMAT_PUBLIC_PR64_MASK_FIRST_LEN_TABLE,
        "members": [
            {
                "name": "renderer.bin",
                "bytes": len(renderer),
                "sha256": hashlib.sha256(renderer).hexdigest(),
                "codec": "raw",
            },
            {
                "name": "masks.mkv",
                "bytes": len(masks),
                "sha256": hashlib.sha256(masks).hexdigest(),
                "codec": "raw",
            },
            {
                "name": "optimized_poses.bin",
                "bytes": len(pose_encoded),
                "sha256": hashlib.sha256(pose_encoded).hexdigest(),
                **pose_meta,
            },
        ],
    }
    payload = (
        struct.pack(PR64_LEN_TABLE_STRUCT, len(masks), len(renderer), len(pose_encoded))
        + masks
        + renderer
        + pose_encoded
    )
    return payload, header


def encode_pose_fp16_col_delta(raw_pose_bytes: bytes, *, pose_dim: int = 6) -> bytes:
    """Encode raw fp16 row-major poses losslessly as column deltas.

    The runtime decoder reconstructs the exact original ``optimized_poses.bin``
    byte stream before existing renderer code loads poses.
    """
    if len(raw_pose_bytes) % (pose_dim * 2) != 0:
        raise ValueError(
            f"raw pose byte length {len(raw_pose_bytes)} is not a multiple of pose_dim*2"
        )
    n_rows = len(raw_pose_bytes) // (pose_dim * 2)
    if n_rows <= 0 or n_rows > 10_000:
        raise ValueError(f"invalid pose row count: {n_rows}")
    values = list(struct.unpack("<" + "H" * (n_rows * pose_dim), raw_pose_bytes))
    out = bytearray(b"PCD1")
    out += struct.pack("<HH", n_rows, pose_dim)
    for col in range(pose_dim):
        prev = values[col]
        out += struct.pack("<H", prev)
        for row in range(1, n_rows):
            cur = values[row * pose_dim + col]
            delta = ((cur - prev + 0x8000) & 0xFFFF) - 0x8000
            out += struct.pack("<h", delta)
            prev = cur
    return bytes(out)


def encode_pose_qpose14_col_delta(
    raw_pose_bytes: bytes,
    *,
    pose_dim: int = 6,
    velocity_offset: float = 20.0,
    velocity_scale: float = 512.0,
    pose_scale: float = 2048.0,
) -> tuple[bytes, bytes]:
    """Encode qpose14-style int16 poses with column deltas.

    This is intentionally lossy.  It preserves the public qpose14 quantization
    contract for all pose dimensions instead of the more aggressive
    velocity-only unified_brotli variant.
    """
    values = _unpack_fp16_values(raw_pose_bytes, pose_dim=pose_dim)
    n_rows = len(values) // pose_dim
    if n_rows <= 0 or n_rows > 10_000:
        raise ValueError(f"invalid pose row count: {n_rows}")

    quantized_words: list[int] = []
    reconstructed_values: list[float] = []
    for row in range(n_rows):
        velocity = float(values[row * pose_dim])
        q0 = int(round((velocity - velocity_offset) * velocity_scale))
        if not 0 <= q0 <= 0xFFFF:
            raise ValueError(f"qpose velocity out of uint16 range at row {row}: {q0}")
        quantized_words.append(q0)
        reconstructed_values.append(q0 / velocity_scale + velocity_offset)

        for dim in range(1, pose_dim):
            value = float(values[row * pose_dim + dim])
            q = int(round(value * pose_scale))
            if not -32768 <= q <= 32767:
                raise ValueError(
                    f"qpose dim {dim} out of int16 range at row {row}: {q}"
                )
            quantized_words.append(q & 0xFFFF)
            reconstructed_values.append(q / pose_scale)

    out = bytearray(b"QP14")
    out += struct.pack("<HH", n_rows, pose_dim)
    for col in range(pose_dim):
        prev = quantized_words[col]
        out += struct.pack("<H", prev)
        for row in range(1, n_rows):
            cur = quantized_words[row * pose_dim + col]
            delta = ((cur - prev + 0x8000) & 0xFFFF) - 0x8000
            out += struct.pack("<h", delta)
            prev = cur
    return bytes(out), _pack_fp16_values(reconstructed_values)


def encode_pose_qp1(raw_pose_bytes: bytes, *, pose_dim: int = 6) -> tuple[bytes, bytes]:
    """Encode PR #67 QP1 velocity-only ZigZag-VLQ poses.

    This is intentionally lossy: only pose column 0 is retained, columns 1-5
    decode to zero. It is the RP2 payload equivalent of the public qpose14
    ``pose_qp1_br`` atom, not a promotion claim by itself.
    """
    values = _unpack_fp16_values(raw_pose_bytes, pose_dim=pose_dim)
    n_rows = len(values) // pose_dim
    rows = [
        [float(values[row * pose_dim + dim]) for dim in range(pose_dim)]
        for row in range(n_rows)
    ]

    from tac.qp1_pose_codec import decode_qp1, encode_qp1

    encoded = encode_qp1(rows)
    decoded = decode_qp1(encoded, pose_dim=pose_dim)
    reconstructed_values = [float(value) for row in decoded for value in row]
    return encoded, _pack_fp16_values(reconstructed_values)


def _unpack_fp16_values(raw_pose_bytes: bytes, *, pose_dim: int = 6) -> tuple[float, ...]:
    if len(raw_pose_bytes) % (pose_dim * 2) != 0:
        raise ValueError(
            f"raw pose byte length {len(raw_pose_bytes)} is not a multiple of pose_dim*2"
        )
    return struct.unpack("<" + "e" * (len(raw_pose_bytes) // 2), raw_pose_bytes)


def _pack_fp16_values(values: list[float]) -> bytes:
    return struct.pack("<" + "e" * len(values), *values)


def encode_pose_fp16_velocity_only(
    raw_pose_bytes: bytes,
    *,
    pose_dim: int = 6,
    velocity_offset: float = 20.0,
    velocity_scale: float = 512.0,
) -> tuple[bytes, bytes]:
    """Encode qpose-style velocity-only poses and return encoded + decoded bytes.

    This is intentionally lossy.  It implements the public unified_brotli
    hypothesis on an existing archive: keep the dominant velocity/log-zoom
    channel, delta-code it, and zero the remaining pose dimensions.
    """
    values = _unpack_fp16_values(raw_pose_bytes, pose_dim=pose_dim)
    n_rows = len(values) // pose_dim
    if n_rows <= 0 or n_rows > 10_000:
        raise ValueError(f"invalid pose row count: {n_rows}")

    velocity_q: list[int] = []
    reconstructed_values: list[float] = []
    for row in range(n_rows):
        velocity = float(values[row * pose_dim])
        q = int(round((velocity - velocity_offset) * velocity_scale))
        if not 0 <= q <= 0xFFFF:
            raise ValueError(f"velocity quantization out of uint16 range at row {row}: {q}")
        velocity_q.append(q)
        reconstructed_values.append(q / velocity_scale + velocity_offset)
        reconstructed_values.extend([0.0] * (pose_dim - 1))

    out = bytearray(b"PVL1")
    out += struct.pack("<HHH", n_rows, pose_dim, velocity_q[0])
    prev = velocity_q[0]
    for row, cur in enumerate(velocity_q[1:], start=1):
        delta = cur - prev
        if not -32768 <= delta <= 32767:
            raise ValueError(f"velocity delta out of int16 range at row {row}: {delta}")
        out += struct.pack("<h", delta)
        prev = cur
    return bytes(out), _pack_fp16_values(reconstructed_values)


def encode_pose_public_pr64_velocity_delta(
    raw_pose_bytes: bytes,
    *,
    pose_dim: int = 6,
    velocity_offset: float = 20.0,
    velocity_scale: float = 512.0,
) -> tuple[bytes, bytes]:
    """Encode public PR64 bare velocity-delta poses.

    The public parser branch carries no magic or row count, so this is scoped
    to the contest's 600-row pose stream. It emits one uint16 anchor followed by
    599 int16 deltas, for exactly 1200 bytes.
    """
    values = _unpack_fp16_values(raw_pose_bytes, pose_dim=pose_dim)
    n_rows = len(values) // pose_dim
    if n_rows != 600:
        raise ValueError(f"public PR64 bare pose codec requires 600 rows, got {n_rows}")

    velocity_q: list[int] = []
    reconstructed_values: list[float] = []
    for row in range(n_rows):
        velocity = float(values[row * pose_dim])
        q = int(round((velocity - velocity_offset) * velocity_scale))
        if not 0 <= q <= 0xFFFF:
            raise ValueError(f"velocity quantization out of uint16 range at row {row}: {q}")
        velocity_q.append(q)
        reconstructed_values.append(q / velocity_scale + velocity_offset)
        reconstructed_values.extend([0.0] * (pose_dim - 1))

    out = bytearray()
    out += struct.pack("<H", velocity_q[0])
    prev = velocity_q[0]
    for row, cur in enumerate(velocity_q[1:], start=1):
        delta = cur - prev
        if not -32768 <= delta <= 32767:
            raise ValueError(f"velocity delta out of int16 range at row {row}: {delta}")
        out += struct.pack("<h", delta)
        prev = cur
    if len(out) != 1200:
        raise RuntimeError(f"public PR64 bare pose codec emitted {len(out)} bytes")
    return bytes(out), _pack_fp16_values(reconstructed_values)


def encode_pose_fp16_velocity_residual_topk(
    raw_pose_bytes: bytes,
    *,
    topk: int,
    pose_dim: int = 6,
    velocity_offset: float = 20.0,
    velocity_scale: float = 512.0,
) -> tuple[bytes, bytes]:
    """Encode velocity plus charged top-K residual atoms for non-velocity dims."""
    if topk < 0:
        raise ValueError(f"topk must be non-negative, got {topk}")
    values = _unpack_fp16_values(raw_pose_bytes, pose_dim=pose_dim)
    half_words = list(struct.unpack("<" + "H" * (len(raw_pose_bytes) // 2), raw_pose_bytes))
    n_rows = len(values) // pose_dim
    if n_rows <= 0 or n_rows > 10_000:
        raise ValueError(f"invalid pose row count: {n_rows}")
    max_atoms = n_rows * max(0, pose_dim - 1)
    if topk > max_atoms:
        raise ValueError(f"topk {topk} exceeds available non-velocity atoms {max_atoms}")

    velocity_q: list[int] = []
    for row in range(n_rows):
        velocity = float(values[row * pose_dim])
        q = int(round((velocity - velocity_offset) * velocity_scale))
        if not 0 <= q <= 0xFFFF:
            raise ValueError(f"velocity quantization out of uint16 range at row {row}: {q}")
        velocity_q.append(q)

    mean_half_words: list[int] = []
    mean_values: list[float] = []
    for dim in range(1, pose_dim):
        mean = sum(float(values[row * pose_dim + dim]) for row in range(n_rows)) / n_rows
        mean_half = struct.unpack("<H", struct.pack("<e", mean))[0]
        mean_half_words.append(mean_half)
        mean_values.append(struct.unpack("<e", struct.pack("<H", mean_half))[0])

    residual_atoms: list[tuple[float, int, int, int]] = []
    for row in range(n_rows):
        for dim in range(1, pose_dim):
            value = float(values[row * pose_dim + dim])
            residual = abs(value - mean_values[dim - 1])
            key = row * pose_dim + dim
            residual_atoms.append((residual, row, dim, half_words[key]))
    residual_atoms.sort(key=lambda item: (-item[0], item[1], item[2]))
    selected_atoms = residual_atoms[:topk]

    out = bytearray(b"PVR1")
    out += struct.pack("<HHHH", n_rows, pose_dim, topk, velocity_q[0])
    for word in mean_half_words:
        out += struct.pack("<H", word)
    prev = velocity_q[0]
    for row, cur in enumerate(velocity_q[1:], start=1):
        delta = cur - prev
        if not -32768 <= delta <= 32767:
            raise ValueError(f"velocity delta out of int16 range at row {row}: {delta}")
        out += struct.pack("<h", delta)
        prev = cur
    for _residual, row, dim, half_word in selected_atoms:
        out += struct.pack("<HH", row * pose_dim + dim, half_word)

    reconstructed = bytearray(n_rows * pose_dim * 2)
    offset = 0
    for row, q in enumerate(velocity_q):
        struct.pack_into("<e", reconstructed, offset, q / velocity_scale + velocity_offset)
        offset += 2
        for dim in range(1, pose_dim):
            struct.pack_into("<H", reconstructed, offset, mean_half_words[dim - 1])
            offset += 2
    for _residual, row, dim, half_word in selected_atoms:
        struct.pack_into("<H", reconstructed, (row * pose_dim + dim) * 2, half_word)
    return bytes(out), bytes(reconstructed)


def pose_error_stats(source_pose_bytes: bytes, reconstructed_pose_bytes: bytes, *, pose_dim: int = 6) -> dict[str, Any]:
    """Return compact source-vs-reconstruction error stats for lossy pose codecs."""
    src = _unpack_fp16_values(source_pose_bytes, pose_dim=pose_dim)
    rec = _unpack_fp16_values(reconstructed_pose_bytes, pose_dim=pose_dim)
    if len(src) != len(rec):
        raise ValueError("pose error stats require equal source/reconstruction length")
    n_rows = len(src) // pose_dim
    max_abs_by_dim: list[float] = []
    rmse_by_dim: list[float] = []
    for dim in range(pose_dim):
        sq = 0.0
        mx = 0.0
        for row in range(n_rows):
            err = float(rec[row * pose_dim + dim]) - float(src[row * pose_dim + dim])
            ae = abs(err)
            if ae > mx:
                mx = ae
            sq += err * err
        max_abs_by_dim.append(mx)
        rmse_by_dim.append((sq / n_rows) ** 0.5)
    return {
        "pose_dim": pose_dim,
        "rows": n_rows,
        "max_abs_by_dim": max_abs_by_dim,
        "rmse_by_dim": rmse_by_dim,
    }


def write_deterministic_payload_archive(
    output_archive: Path,
    compressed_payload: bytes,
    *,
    payload_member_name: str = PAYLOAD_MEMBER_NAME,
) -> None:
    _validate_member_name(payload_member_name)
    output_archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_archive, "w", compression=zipfile.ZIP_STORED) as zf:
        info = zipfile.ZipInfo(payload_member_name, date_time=FIXED_ZIP_TIMESTAMP)
        info.compress_type = zipfile.ZIP_STORED
        info.create_system = 3
        info.external_attr = 0o600 << 16
        info.extra = b""
        info.comment = b""
        zf.writestr(info, compressed_payload)


def build_packed_archive(
    source_archive: Path,
    output_archive: Path,
    *,
    brotli_quality: int = 11,
    pose_codec: str = "raw",
    pose_residual_topk: int = 0,
    payload_member_name: str = PAYLOAD_MEMBER_NAME,
    payload_format: str = PAYLOAD_FORMAT_RPK1_JSON,
) -> dict[str, Any]:
    """Build and verify a deterministic packed renderer archive."""
    if not 0 <= brotli_quality <= 11:
        raise ValueError(f"brotli_quality must be in [0, 11], got {brotli_quality}")
    if payload_member_name not in ALLOWED_PAYLOAD_MEMBER_NAMES:
        raise ValueError(
            f"payload_member_name must be one of {ALLOWED_PAYLOAD_MEMBER_NAMES}, "
            f"got {payload_member_name!r}"
        )
    if payload_format not in ALLOWED_PAYLOAD_FORMATS:
        raise ValueError(
            f"payload_format must be one of {ALLOWED_PAYLOAD_FORMATS}, got {payload_format!r}"
        )
    import brotli

    source_bytes = source_archive.read_bytes()
    source_sha = hashlib.sha256(source_bytes).hexdigest()
    source_members = read_source_members(source_archive)
    ordered = ordered_runtime_members(source_members)
    if payload_format == PAYLOAD_FORMAT_RP2_FIXED3:
        payload, header = build_compact_renderer_payload(
            ordered,
            source_archive_sha256=source_sha,
            pose_codec=pose_codec,
        )
    elif payload_format == PAYLOAD_FORMAT_PR64_LEN_TABLE:
        payload, header = build_pr64_len_table_payload(
            ordered,
            source_archive_sha256=source_sha,
            pose_codec=pose_codec,
            pose_residual_topk=pose_residual_topk,
        )
    elif payload_format == PAYLOAD_FORMAT_PUBLIC_PR64_MASK_FIRST_LEN_TABLE:
        payload, header = build_public_pr64_mask_first_len_table_payload(
            ordered,
            source_archive_sha256=source_sha,
            pose_codec=pose_codec,
            pose_residual_topk=pose_residual_topk,
        )
    else:
        payload, header = build_renderer_payload(
            ordered,
            source_archive_sha256=source_sha,
            pose_codec=pose_codec,
            pose_residual_topk=pose_residual_topk,
        )
    compressed = brotli.compress(payload, quality=brotli_quality, lgwin=24)
    write_deterministic_payload_archive(
        output_archive,
        compressed,
        payload_member_name=payload_member_name,
    )

    # Verify byte-stable payload decode before returning byte-screen stats.
    decoded = brotli.decompress(compressed)
    if decoded != payload:
        raise RuntimeError("Brotli round-trip mismatch for renderer payload")

    output_bytes = output_archive.stat().st_size
    savings_bytes = source_archive.stat().st_size - output_bytes
    return {
        "score_claim": False,
        "evidence_grade": "empirical",
        "source_archive": str(source_archive),
        "source_archive_bytes": source_archive.stat().st_size,
        "source_archive_sha256": source_sha,
        "output_archive": str(output_archive),
        "output_archive_bytes": output_bytes,
        "output_archive_sha256": hashlib.sha256(output_archive.read_bytes()).hexdigest(),
        "savings_bytes": savings_bytes,
        "formula_only_rate_delta": -25.0 * savings_bytes / ORIGINAL_VIDEO_BYTES,
        "payload_member": payload_member_name,
        "payload_format": payload_format,
        "payload_raw_bytes": len(payload),
        "payload_compressed_bytes": len(compressed),
        "brotli_quality": brotli_quality,
        "pose_codec": pose_codec,
        "pose_residual_topk": pose_residual_topk,
        "header": header,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--brotli-quality", type=int, default=11)
    parser.add_argument(
        "--pose-codec",
        choices=(
            "raw",
            POSE_FP16_COL_DELTA_CODEC,
            POSE_QPOSE14_COL_DELTA_CODEC,
            POSE_QP1_CODEC,
            POSE_FP16_VELOCITY_ONLY_CODEC,
            POSE_PUBLIC_PR64_VELOCITY_DELTA_CODEC,
            POSE_FP16_VELOCITY_RESIDUAL_TOPK_CODEC,
        ),
        default="raw",
        help="Optional pose transform decoded before renderer load.",
    )
    parser.add_argument(
        "--pose-residual-topk",
        type=int,
        default=0,
        help=f"Top-K non-velocity residual atoms for {POSE_FP16_VELOCITY_RESIDUAL_TOPK_CODEC}.",
    )
    parser.add_argument(
        "--payload-member-name",
        choices=ALLOWED_PAYLOAD_MEMBER_NAMES,
        default=PAYLOAD_MEMBER_NAME,
        help="Archive member name for the Brotli-compressed RPK1 payload.",
    )
    parser.add_argument(
        "--payload-format",
        choices=ALLOWED_PAYLOAD_FORMATS,
        default=PAYLOAD_FORMAT_RPK1_JSON,
        help="Payload header/contract format.",
    )
    args = parser.parse_args(argv)

    output_archive = args.output_dir / "archive.zip"
    result = build_packed_archive(
        args.source_archive,
        output_archive,
        brotli_quality=args.brotli_quality,
        pose_codec=args.pose_codec,
        pose_residual_topk=args.pose_residual_topk,
        payload_member_name=args.payload_member_name,
        payload_format=args.payload_format,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "packed_renderer_payload_provenance.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
