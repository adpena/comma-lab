#!/usr/bin/env python3
"""Unpack a single renderer payload member into the normal inflate layout.

This is intentionally standalone: the contest inflate environment must be able
to run it before importing the heavier renderer stack or the local ``tac``
package.  The payload format is deterministic and lossless:

``RPK1`` + little-endian uint32 JSON header length + JSON header + raw members.

The header records the ordered logical member names, byte lengths, and SHA-256
digests.  Extraction refuses unsafe paths, duplicate names, length mismatches,
SHA mismatches, and overwrite mismatches.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from fractions import Fraction
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
PAYLOAD_BIN = "renderer_payload.bin"
PAYLOAD_BR = "renderer_payload.bin.br"
PAYLOAD_SHORT_BR = "p"
PUBLIC_PR81_REORDERED_QZS3_MAGIC = b"Q81R"
POSE_FP16_COL_DELTA_CODEC = "pose_fp16_col_delta_v1"
POSE_QPOSE14_COL_DELTA_CODEC = "pose_qpose14_col_delta_v1"
POSE_QP1_CODEC = "pose_qp1_v1"
POSE_QPV1_CODEC = "pose_qpv1_v1"
POSE_FP16_VELOCITY_ONLY_CODEC = "pose_fp16_velocity_only_v1"
POSE_FP16_VELOCITY_RESIDUAL_TOPK_CODEC = "pose_fp16_velocity_residual_topk_v1"
NERV_MAGIC = b"NRV1"
QP19_MAGIC = b"QP19"
QPV1_MAGIC = b"QPV1"
PUBLIC_PR67_QZS3_MODEL_LENS = (
    55_965,
    56_034,
    56_093,
    56_221,
    57_031,
    57_053,
    57_757,
    60_880,
)
PUBLIC_PR75_MASK_LEN = 219_472
PUBLIC_PR75_MODEL_LEN = 56_034
PUBLIC_PR75_MODEL_LENS = (
    55_756,  # PR75 qpose14_r55_segactions_minp observed 2026-05-03
    55_914,  # current PR67 release asset observed 2026-05-03
    PUBLIC_PR75_MODEL_LEN,
)
PUBLIC_PR75_ACTIONS_LEN = 236
PUBLIC_PR75_ACTION_LENS = (
    PUBLIC_PR75_ACTIONS_LEN,
    253,
    255,
    325,  # PR77 qzs3_tile_delta_r147 observed 2026-05-03
    1_095,  # PR79 S2 adaptive-action lossless repack
    1_162,  # PR79 qpose14_r55_segactions_minp_v2 observed 2026-05-03
)
PUBLIC_PR75_FIXED_SLICE_VARIANTS = (
    # (total payload bytes, Brotli(QZS3 renderer) bytes, Brotli(actions) bytes)
    (276_641, 56_034, 236),
    (276_520, 55_914, 236),
    (276_381, 55_756, 255),
    (276_379, 55_756, 253),
    (276_451, 55_756, 325),  # PR77 qzs3_tile_delta_r147 observed 2026-05-03
    (277_221, 55_756, 1_095),  # PR79 S2 adaptive-action lossless repack
    (277_247, 55_756, 1_121),  # PR79 S1 split-action lossless repack
    (277_288, 55_756, 1_162),  # PR79 qpose14_r55_segactions_minp_v2 observed 2026-05-03
)
PUBLIC_PR75_MAX_ACTION_RECORDS = 10_000
PUBLIC_PR75_MAX_PAIR_INDEX_EXCLUSIVE = 10_000
PUBLIC_PR81_RANGE_MASK_BYTES = 159_011
PUBLIC_PR81_SPLIT_MODEL_REORDERED_BYTES = 55_725
PUBLIC_PR81_POSE_STREAM_BYTES = 899
PUBLIC_PR81_ROUTER_ACTION_BYTES = 225
PUBLIC_PR81_PACKED_PAYLOAD_BYTES = (
    PUBLIC_PR81_RANGE_MASK_BYTES
    + PUBLIC_PR81_SPLIT_MODEL_REORDERED_BYTES
    + PUBLIC_PR81_POSE_STREAM_BYTES
    + PUBLIC_PR81_ROUTER_ACTION_BYTES
)
PUBLIC_QMA9_NO_ROUTER_PACKED_PAYLOAD_BYTES = (
    PUBLIC_PR81_RANGE_MASK_BYTES
    + PUBLIC_PR81_SPLIT_MODEL_REORDERED_BYTES
    + PUBLIC_PR81_POSE_STREAM_BYTES
)
SEG_TILE_ACTION_DICT_MAGIC = b"TAD1"
SEG_TILE_ACTION_DICT_HEADER_STRUCT = "<4sHH"
SEG_TILE_ACTION_SPLIT_MAGIC = b"S1"
SEG_TILE_ACTION_SPLIT2_MAGIC = b"S2"
SEG_TILE_ACTION_SPLIT2_ADAPTIVE_ARITH = 1
COMPACT_POSE_CODECS = {
    0: "raw",
    1: POSE_FP16_COL_DELTA_CODEC,
    2: POSE_QPOSE14_COL_DELTA_CODEC,
    3: POSE_QP1_CODEC,
}

_ALLOWED_MEMBER_NAMES = {
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
    "masks.qma9",
    "optimized_poses.pt",
    "optimized_poses.bin",
    "optimized_poses.qp1",
    "optimized_embedding.pt",
    "poses.pt",
    "corrections.bin",
    "gradient_corrections.bin",
    "mini_segnet.bin",
    "mini_posenet.bin",
    "posenet_targets.bin",
    "zoom_scalars.bin",
    "foveation_params.bin",
    "sjkl.bin",
    "seg_tile_actions.bin",
    "seg_tile_actions.br",
    "seg_tile_action_dict.bin",
    "router_actions.3bit",
    "alpha4_residual_repair.amr1",
    "alpha4_residual_repair.amr1.xz",
    "alpha4_residual_repair.amr1.zlib",
    "alpha4_residual_repair.amr1.br",
}


def _safe_member_name(name: str) -> str:
    path = Path(name)
    if not name or name.startswith("/") or ".." in path.parts or len(path.parts) != 1:
        raise ValueError(f"unsafe renderer payload member path: {name!r}")
    if name not in _ALLOWED_MEMBER_NAMES:
        raise ValueError(f"unknown renderer payload member name: {name!r}")
    return name


def _read_payload_bytes(archive_dir: Path) -> bytes:
    candidates = [
        (PAYLOAD_BIN, archive_dir / PAYLOAD_BIN, "raw"),
        (PAYLOAD_BR, archive_dir / PAYLOAD_BR, "brotli"),
        (PAYLOAD_SHORT_BR, archive_dir / PAYLOAD_SHORT_BR, "brotli"),
    ]
    present = [(label, path, codec) for label, path, codec in candidates if path.exists()]
    if len(present) > 1:
        names = ", ".join(label for label, _path, _codec in present)
        raise ValueError(
            f"ambiguous renderer payload containers in archive: {names}. "
            "Use exactly one packed payload member."
        )
    if present:
        label, path, codec = present[0]
        if codec == "raw":
            return path.read_bytes()
        return _decompress_brotli_payload(path, label)
    raise FileNotFoundError(
        f"missing {PAYLOAD_BIN}, {PAYLOAD_BR}, or {PAYLOAD_SHORT_BR} in {archive_dir}"
    )


def _decompress_brotli_payload(path: Path, label: str) -> bytes:
    try:
        import brotli
    except ImportError as exc:
        raise RuntimeError(
            f"{label} exists but brotli is not importable; "
            "inflate.sh should provide brotli before calling "
            "unpack_renderer_payload.py."
        ) from exc
    data = path.read_bytes()
    try:
        return brotli.decompress(data)
    except brotli.error:
        if label == PAYLOAD_SHORT_BR:
            # Public qpose14 PR #63 uses one ZIP member named "p", but that
            # member is a concatenation of Brotli streams rather than a single
            # Brotli stream.  Return the raw container so _parse_payload can
            # apply the public fixed-slice parser below.
            return data
        raise


def _parse_payload(payload: bytes) -> tuple[dict[str, Any], dict[str, bytes]]:
    if payload[:4] == COMPACT_MAGIC:
        return _parse_compact_payload(payload)
    if payload[:4] == NERV_MAGIC:
        return _parse_public_pr67_nerv_qzs3_qp1_payload(payload)
    if len(payload) < len(MAGIC) + struct.calcsize(HEADER_STRUCT):
        raise ValueError("renderer payload is too short for magic/header")
    if payload[:4] != MAGIC:
        parsed_public_qp19 = _try_parse_public_qp19_qpv1_payload(payload)
        if parsed_public_qp19 is not None:
            return parsed_public_qp19
        parsed_public_pr81 = _try_parse_public_pr81_qma9_qzs3_qp1_router_payload(payload)
        if parsed_public_pr81 is not None:
            return parsed_public_pr81
        parsed_public_pr75 = _try_parse_public_pr75_qzs3_qp1_segactions_payload(payload)
        if parsed_public_pr75 is not None:
            return parsed_public_pr75
        parsed_public_pr67 = _try_parse_public_pr67_qzs3_qp1_payload(payload)
        if parsed_public_pr67 is not None:
            return parsed_public_pr67
        parsed_public_pr63 = _try_parse_public_pr63_qpose14_payload(payload)
        if parsed_public_pr63 is not None:
            return parsed_public_pr63
        parsed = _try_parse_pr64_len_table_payload(payload)
        if parsed is not None:
            return parsed
        raise ValueError(f"bad renderer payload magic {payload[:4]!r}; expected {MAGIC!r}")

    header_len = struct.unpack_from(HEADER_STRUCT, payload, 4)[0]
    header_start = 4 + struct.calcsize(HEADER_STRUCT)
    header_end = header_start + header_len
    if header_len <= 0 or header_end > len(payload):
        raise ValueError(f"invalid renderer payload header length: {header_len}")

    header = json.loads(payload[header_start:header_end].decode("utf-8"))
    if header.get("schema") != SCHEMA:
        raise ValueError(f"unsupported renderer payload schema: {header.get('schema')!r}")
    members_meta = header.get("members")
    if not isinstance(members_meta, list) or not members_meta:
        raise ValueError("renderer payload header must contain a non-empty members list")

    offset = header_end
    seen: set[str] = set()
    members: dict[str, bytes] = {}
    for meta in members_meta:
        if not isinstance(meta, dict):
            raise ValueError("renderer payload member metadata must be objects")
        name = _safe_member_name(str(meta.get("name", "")))
        if name in seen:
            raise ValueError(f"duplicate renderer payload member: {name}")
        seen.add(name)

        n_bytes = int(meta.get("bytes", -1))
        if n_bytes < 0:
            raise ValueError(f"negative byte length for renderer payload member {name}")
        end = offset + n_bytes
        if end > len(payload):
            raise ValueError(f"renderer payload member {name} overruns payload")
        data = payload[offset:end]
        offset = end

        expected_sha = str(meta.get("sha256", ""))
        actual_sha = hashlib.sha256(data).hexdigest()
        if expected_sha != actual_sha:
            raise ValueError(
                f"renderer payload member {name} encoded SHA mismatch: "
                f"expected {expected_sha}, got {actual_sha}"
            )
        codec = str(meta.get("codec", "raw"))
        if codec == "raw":
            decoded = data
        elif codec == POSE_FP16_COL_DELTA_CODEC:
            if name != "optimized_poses.bin":
                raise ValueError(f"{POSE_FP16_COL_DELTA_CODEC} may only target optimized_poses.bin")
            decoded = _decode_pose_fp16_col_delta(data)
        elif codec == POSE_QPOSE14_COL_DELTA_CODEC:
            if name != "optimized_poses.bin":
                raise ValueError(f"{POSE_QPOSE14_COL_DELTA_CODEC} may only target optimized_poses.bin")
            decoded = _decode_pose_qpose14_col_delta(data)
        elif codec == POSE_QP1_CODEC:
            if name != "optimized_poses.bin":
                raise ValueError(f"{POSE_QP1_CODEC} may only target optimized_poses.bin")
            decoded = _decode_pose_qp1(data)
        elif codec == POSE_FP16_VELOCITY_ONLY_CODEC:
            if name != "optimized_poses.bin":
                raise ValueError(f"{POSE_FP16_VELOCITY_ONLY_CODEC} may only target optimized_poses.bin")
            decoded = _decode_pose_fp16_velocity_only(data)
        elif codec == POSE_FP16_VELOCITY_RESIDUAL_TOPK_CODEC:
            if name != "optimized_poses.bin":
                raise ValueError(f"{POSE_FP16_VELOCITY_RESIDUAL_TOPK_CODEC} may only target optimized_poses.bin")
            decoded = _decode_pose_fp16_velocity_residual_topk(data)
        else:
            raise ValueError(f"unsupported renderer payload member codec: {codec!r}")

        expected_decoded_bytes = meta.get("decoded_bytes")
        if expected_decoded_bytes is not None and int(expected_decoded_bytes) != len(decoded):
            raise ValueError(
                f"renderer payload member {name} decoded byte mismatch: "
                f"expected {expected_decoded_bytes}, got {len(decoded)}"
            )
        expected_decoded_sha = meta.get("decoded_sha256")
        if expected_decoded_sha is not None:
            actual_decoded_sha = hashlib.sha256(decoded).hexdigest()
            if str(expected_decoded_sha) != actual_decoded_sha:
                raise ValueError(
                    f"renderer payload member {name} decoded SHA mismatch: "
                    f"expected {expected_decoded_sha}, got {actual_decoded_sha}"
                )
        members[name] = decoded

    if offset != len(payload):
        raise ValueError(
            f"renderer payload has {len(payload) - offset} trailing bytes after members"
        )
    return header, members


def _try_parse_pr64_len_table_payload(payload: bytes) -> tuple[dict[str, Any], dict[str, bytes]] | None:
    header_size = struct.calcsize(PR64_LEN_TABLE_STRUCT)
    if len(payload) < header_size:
        return None
    first_len, second_len, pose_len = struct.unpack_from(PR64_LEN_TABLE_STRUCT, payload, 0)
    if first_len <= 0 or second_len <= 0 or pose_len <= 0:
        return None
    if header_size + first_len + second_len + pose_len != len(payload):
        return None

    first_start = header_size
    first_end = first_start + first_len
    second_end = first_end + second_len
    first = payload[first_start:first_end]
    second = payload[first_end:second_end]
    pose_raw = payload[second_end:]

    payload_format = "pr64_len_table"
    if _looks_like_renderer_payload(first) and not _looks_like_renderer_payload(second):
        raw_members = {
            "renderer.bin": first,
            "masks.mkv": second,
            "optimized_poses.bin": pose_raw,
        }
    elif not _looks_like_renderer_payload(first) and _looks_like_renderer_payload(second):
        # Public unified_brotli PR #64 uses <mask_len, model_len, pose_len>
        # followed by mask, Torch-FP4 model, and velocity-delta poses.
        payload_format = "public_pr64_mask_first_len_table"
        raw_members = {
            "renderer.bin": second,
            "masks.mkv": first,
            "optimized_poses.bin": pose_raw,
        }
    else:
        # Backward-compatible local format: our builder writes renderer,
        # masks, pose.  Unit fixtures and some experimental payloads use
        # synthetic renderer bytes that do not carry QZS3/QFAI/Torch magic.
        raw_members = {
            "renderer.bin": first,
            "masks.mkv": second,
            "optimized_poses.bin": pose_raw,
        }

    pose_codec = "raw"
    decoded_pose = pose_raw
    if pose_raw.startswith(b"PCD1"):
        pose_codec = POSE_FP16_COL_DELTA_CODEC
        decoded_pose = _decode_pose_fp16_col_delta(pose_raw)
    elif pose_raw.startswith(b"QP14"):
        pose_codec = POSE_QPOSE14_COL_DELTA_CODEC
        decoded_pose = _decode_pose_qpose14_col_delta(pose_raw)
    elif pose_raw.startswith(b"QP1"):
        pose_codec = POSE_QP1_CODEC
        decoded_pose = _decode_pose_qp1(pose_raw)
    elif pose_raw.startswith(b"PVL1"):
        pose_codec = POSE_FP16_VELOCITY_ONLY_CODEC
        decoded_pose = _decode_pose_fp16_velocity_only(pose_raw)
    elif pose_raw.startswith(b"PVR1"):
        pose_codec = POSE_FP16_VELOCITY_RESIDUAL_TOPK_CODEC
        decoded_pose = _decode_pose_fp16_velocity_residual_topk(pose_raw)
    elif payload_format in {"pr64_len_table", "public_pr64_mask_first_len_table"} and len(pose_raw) == 1200:
        pose_codec = "public_pr64_velocity_delta_uint16_int16"
        decoded_pose = _decode_public_pr64_velocity_delta(pose_raw)

    members = {
        "renderer.bin": raw_members["renderer.bin"],
        "masks.mkv": raw_members["masks.mkv"],
        "optimized_poses.bin": decoded_pose,
    }
    header = {
        "schema": PR64_LEN_TABLE_SCHEMA,
        "payload_format": payload_format,
        "members": [
            {
                "name": "renderer.bin",
                "bytes": len(raw_members["renderer.bin"]),
                "sha256": hashlib.sha256(raw_members["renderer.bin"]).hexdigest(),
                "codec": "raw",
            },
            {
                "name": "masks.mkv",
                "bytes": len(raw_members["masks.mkv"]),
                "sha256": hashlib.sha256(raw_members["masks.mkv"]).hexdigest(),
                "codec": "raw",
            },
            {
                "name": "optimized_poses.bin",
                "bytes": len(pose_raw),
                "sha256": hashlib.sha256(pose_raw).hexdigest(),
                "codec": pose_codec,
                "decoded_bytes": len(decoded_pose),
                "decoded_sha256": hashlib.sha256(decoded_pose).hexdigest(),
            },
        ],
    }
    return header, members


def _looks_like_renderer_payload(data: bytes) -> bool:
    return (
        data.startswith(b"QZS3")
        or data.startswith(b"MQZ1")
        or data.startswith(b"QBF1")
        or data.startswith(b"BFJ1")
        or data.startswith(b"QFAI")
        or data.startswith(b"\x80\x02")
        or data.startswith(b"PK\x03\x04")
    )


def _renderer_payload_codec_label(data: bytes) -> str:
    if data.startswith(PUBLIC_PR81_REORDERED_QZS3_MAGIC):
        return "public_pr81_reordered_qzs3_model_bundle"
    if data.startswith(b"QZS3"):
        return "brotli_qzs3"
    if data.startswith(b"MQZ1"):
        return "brotli_mqz1"
    if data.startswith(b"QBF1"):
        return "brotli_qbf1"
    if data.startswith(b"BFJ1"):
        return "brotli_bfj1"
    if data.startswith(b"QFAI"):
        return "brotli_qfai"
    if data.startswith(b"\x80\x02"):
        return "brotli_pickle_renderer"
    if data.startswith(b"PK\x03\x04"):
        return "brotli_zip_renderer"
    return "brotli_unknown_renderer"


def _looks_like_mask_obu(data: bytes) -> bool:
    return data.startswith(b"\x12\x00\x0a\x0a") or data.startswith(b"\x12\x00")


def _looks_like_qma9_mask(data: bytes) -> bool:
    if len(data) < 20 or not data.startswith(b"QMA9"):
        return False
    frame_count, width, height, body_bytes = struct.unpack_from("<IIII", data, 4)
    return (
        frame_count == 600
        and (width, height) in {(512, 384), (384, 512)}
        and 20 + int(body_bytes) == len(data)
    )


def _try_parse_public_pr81_qma9_qzs3_qp1_router_payload(
    payload: bytes,
) -> tuple[dict[str, Any], dict[str, bytes]] | None:
    """Parse PR81/PR84 QMA9 one-member public payload into runtime members.

    The public PR81 archive is a stored ZIP member named ``p`` with four fixed
    raw slices: QMA9 semantic masks, reordered Brotli QZS3 model bundle,
    Brotli QP1 pose stream, and 3-bit router actions.  PR84 drops the router
    tail but keeps the same first three slices.  The model bundle is
    intentionally wrapped with ``Q81R`` here so inflate can restore the public
    reordered chunk order before loading QZS3.  This parser never decodes
    scorer output; it only exposes charged bytes as typed runtime members.
    """

    has_router = len(payload) == PUBLIC_PR81_PACKED_PAYLOAD_BYTES
    if not has_router and len(payload) != PUBLIC_QMA9_NO_ROUTER_PACKED_PAYLOAD_BYTES:
        return None
    mask_end = PUBLIC_PR81_RANGE_MASK_BYTES
    model_end = mask_end + PUBLIC_PR81_SPLIT_MODEL_REORDERED_BYTES
    pose_end = model_end + PUBLIC_PR81_POSE_STREAM_BYTES
    raw_slices = {
        "masks.qma9": payload[:mask_end],
        "renderer.bin": payload[mask_end:model_end],
        "optimized_poses.qp1": payload[model_end:pose_end],
        "router_actions.3bit": payload[pose_end:],
    }
    if not _looks_like_qma9_mask(raw_slices["masks.qma9"]):
        return None
    if has_router and len(raw_slices["router_actions.3bit"]) != PUBLIC_PR81_ROUTER_ACTION_BYTES:
        return None
    if not has_router and raw_slices["router_actions.3bit"]:
        return None

    import brotli

    try:
        pose_qp1 = brotli.decompress(raw_slices["optimized_poses.qp1"])
    except Exception:
        return None
    if not pose_qp1.startswith(b"QP1"):
        return None
    renderer = (
        PUBLIC_PR81_REORDERED_QZS3_MAGIC
        + raw_slices["renderer.bin"]
    )
    members = {
        "masks.qma9": raw_slices["masks.qma9"],
        "renderer.bin": renderer,
        "optimized_poses.qp1": pose_qp1,
    }
    if has_router:
        members["router_actions.3bit"] = raw_slices["router_actions.3bit"]
    member_entries = [
        {
            "name": "masks.qma9",
            "bytes": len(raw_slices["masks.qma9"]),
            "sha256": hashlib.sha256(raw_slices["masks.qma9"]).hexdigest(),
            "codec": "qma9_adaptive9_binary_range_mask",
            "decoded_bytes": 600 * 384 * 512,
        },
        {
            "name": "renderer.bin",
            "bytes": len(raw_slices["renderer.bin"]),
            "sha256": hashlib.sha256(raw_slices["renderer.bin"]).hexdigest(),
            "codec": "public_pr81_reordered_qzs3_model_bundle",
            "decoded_bytes": len(renderer),
            "decoded_sha256": hashlib.sha256(renderer).hexdigest(),
        },
        {
            "name": "optimized_poses.qp1",
            "bytes": len(raw_slices["optimized_poses.qp1"]),
            "sha256": hashlib.sha256(raw_slices["optimized_poses.qp1"]).hexdigest(),
            "codec": "public_qp1_brotli",
            "decoded_bytes": len(pose_qp1),
            "decoded_sha256": hashlib.sha256(pose_qp1).hexdigest(),
        },
    ]
    if has_router:
        member_entries.append(
            {
                "name": "router_actions.3bit",
                "bytes": len(raw_slices["router_actions.3bit"]),
                "sha256": hashlib.sha256(raw_slices["router_actions.3bit"]).hexdigest(),
                "codec": "public_pr81_packed_3bit_pair_router_actions",
                "decoded_bytes": 600,
            }
        )
    header = {
        "schema": PR64_LEN_TABLE_SCHEMA,
        "payload_format": (
            "public_pr81_qma9_reordered_qzs3_qp1_router_fixed_slices"
            if has_router
            else "public_qma9_reordered_qzs3_qp1_no_router_fixed_slices"
        ),
        "members": member_entries,
    }
    return header, members


def _decode_seg_tile_actions(data: bytes) -> bytes:
    """Decode and validate PR75-style Brotli tile action records.

    Records are either u16 frame + u8 tile + u8 action, or u16 frame +
    u16 tile + u8 action. The current public payload uses the compact
    4-byte form. Recent minp payloads may also use ``SG2`` tile-grouped
    varints or ``S1`` split tile/action streams. Store decoded records as
    charged runtime data so the renderer hot path does not need to know
    whether the source was Brotli.
    """
    import brotli

    if data.startswith(SEG_TILE_ACTION_SPLIT2_MAGIC):
        raw = _decode_split2_seg_tile_actions(data)
        if len(raw) % 4 != 0:
            raise ValueError(
                f"seg_tile_actions S2 decoded length is not raw4-aligned: {len(raw)}"
            )
        return raw
    raw = brotli.decompress(data)
    grid_header = b""
    if raw.startswith(b"TG1"):
        if len(raw) < 5:
            raise ValueError("seg_tile_actions TG1 header is truncated")
        tile_size = int.from_bytes(raw[3:5], "little")
        if tile_size <= 0 or 384 % tile_size != 0 or 512 % tile_size != 0:
            raise ValueError(f"unsupported seg_tile_actions TG1 tile_size: {tile_size}")
        grid_header = raw[:5]
        raw = raw[5:]
    if raw.startswith(SEG_TILE_ACTION_SPLIT_MAGIC):
        raw = _decode_split_seg_tile_actions(raw)
    elif raw.startswith(b"SG2") or (len(raw) % 4 != 0 and len(raw) % 5 != 0):
        records: list[tuple[int, int, int]] = []
        cursor = 3 if raw.startswith(b"SG2") else 0
        while cursor < len(raw):
            tile, cursor = _read_uvarint(
                raw,
                cursor,
                max_value=PUBLIC_PR75_MAX_PAIR_INDEX_EXCLUSIVE - 1,
            )
            count, cursor = _read_uvarint(
                raw,
                cursor,
                max_value=PUBLIC_PR75_MAX_ACTION_RECORDS,
            )
            if count <= 0:
                raise ValueError("seg_tile_actions SG2 group has zero records")
            frame = 0
            for idx in range(count):
                delta, cursor = _read_uvarint(
                    raw,
                    cursor,
                    max_value=PUBLIC_PR75_MAX_PAIR_INDEX_EXCLUSIVE - 1,
                )
                frame = delta if idx == 0 else frame + delta
                if frame >= PUBLIC_PR75_MAX_PAIR_INDEX_EXCLUSIVE:
                    raise ValueError(f"seg_tile_actions SG2 frame out of bounds: {frame}")
                if cursor >= len(raw):
                    raise ValueError("seg_tile_actions SG2 payload ended inside record")
                action = raw[cursor]
                cursor += 1
                records.append((int(frame), int(tile), int(action)))
        use_raw5 = any(tile >= 256 for _, tile, _ in records)
        out = bytearray()
        for frame, tile, action in records:
            out += int(frame).to_bytes(2, "little")
            if use_raw5:
                out += int(tile).to_bytes(2, "little")
            else:
                out.append(int(tile))
            out.append(int(action))
        if use_raw5:
            raw = b"TA5" + bytes(out)
        else:
            raw = bytes(out)
    if len(raw) % 4 == 0:
        record_size = 4
    elif len(raw) % 5 == 0:
        record_size = 5
    else:
        raise ValueError(
            f"seg_tile_actions payload has unsupported decoded length {len(raw)}"
        )
    n_records = len(raw) // record_size
    if n_records <= 0 or n_records > 10_000:
        raise ValueError(f"unreasonable seg_tile_actions record count: {n_records}")
    return grid_header + raw


def _decode_adaptive_arithmetic_actions(
    data: bytes,
    *,
    nbits: int,
    record_count: int,
    action_count: int = 108,
) -> bytes:
    if record_count <= 0 or record_count > PUBLIC_PR75_MAX_ACTION_RECORDS:
        raise ValueError(f"unreasonable S2 action record count: {record_count}")
    if action_count <= 0 or action_count > 256:
        raise ValueError(f"unreasonable S2 action alphabet size: {action_count}")
    expected_bytes = (nbits + 7) // 8
    if nbits <= 0 or len(data) != expected_bytes:
        raise ValueError(
            f"S2 arithmetic body length mismatch: nbits={nbits} "
            f"expected_bytes={expected_bytes} got={len(data)}"
        )

    code_int = int.from_bytes(data, "big") if data else 0
    if code_int >= (1 << nbits):
        raise ValueError("S2 arithmetic code exceeds declared bit length")
    code = Fraction(code_int, 1 << nbits)
    low = Fraction(0, 1)
    high = Fraction(1, 1)
    counts = [1] * action_count
    total = action_count
    out = bytearray()
    for _idx in range(record_count):
        width = high - low
        if width <= 0:
            raise ValueError("S2 arithmetic interval collapsed")
        scaled = (code - low) * total / width
        value = scaled.numerator // scaled.denominator
        if value < 0 or value >= total:
            raise ValueError("S2 arithmetic code fell outside active interval")
        cumulative = 0
        for symbol, count in enumerate(counts):
            if value < cumulative + count:
                break
            cumulative += count
        else:
            raise ValueError("S2 arithmetic symbol lookup failed")
        high = low + width * Fraction(cumulative + count, total)
        low = low + width * Fraction(cumulative, total)
        counts[symbol] += 1
        total += 1
        out.append(symbol)
    return bytes(out)


def _decode_split2_seg_tile_actions(data: bytes) -> bytes:
    """Decode S2 split action streams into runtime raw4 records."""
    import brotli

    offset = len(SEG_TILE_ACTION_SPLIT2_MAGIC)
    mode, offset = _read_uvarint(data, offset, max_value=16)
    if mode != SEG_TILE_ACTION_SPLIT2_ADAPTIVE_ARITH:
        raise ValueError(f"unsupported S2 seg_tile_actions mode: {mode}")
    meta_len, offset = _read_uvarint(data, offset, max_value=len(data))
    action_nbits, offset = _read_uvarint(data, offset, max_value=1_000_000)
    meta_end = offset + meta_len
    if meta_end >= len(data):
        raise ValueError("S2 seg_tile_actions metadata overruns payload")
    meta = brotli.decompress(data[offset:meta_end])
    action_code = data[meta_end:]

    if not meta.startswith(SEG_TILE_ACTION_SPLIT_MAGIC):
        raise ValueError("S2 metadata stream must start with S1 magic")
    cursor = len(SEG_TILE_ACTION_SPLIT_MAGIC)
    group_count, cursor = _read_uvarint(meta, cursor, max_value=256)
    if group_count <= 0:
        raise ValueError("S2 metadata stream has zero groups")

    groups: list[tuple[int, int]] = []
    tile_id = 0
    total_records = 0
    for group_index in range(group_count):
        tile_delta, cursor = _read_uvarint(meta, cursor, max_value=255)
        if group_index == 0:
            tile_id = tile_delta
        else:
            if tile_delta <= 0:
                raise ValueError("S2 tile deltas must increase")
            tile_id += tile_delta
        if tile_id >= 192:
            raise ValueError(f"S2 tile out of bounds: {tile_id}")
        count, cursor = _read_uvarint(meta, cursor, max_value=PUBLIC_PR75_MAX_ACTION_RECORDS)
        if count <= 0:
            raise ValueError("S2 group has zero records")
        total_records += count
        if total_records > PUBLIC_PR75_MAX_ACTION_RECORDS:
            raise ValueError(f"unreasonable S2 record count: {total_records}")
        groups.append((tile_id, count))

    decoded_pairs: list[tuple[int, int]] = []
    for tile_id, count in groups:
        pair_index = 0
        for record_index in range(count):
            delta, cursor = _read_uvarint(
                meta,
                cursor,
                max_value=PUBLIC_PR75_MAX_PAIR_INDEX_EXCLUSIVE - 1,
            )
            pair_index = delta if record_index == 0 else pair_index + delta
            if pair_index >= PUBLIC_PR75_MAX_PAIR_INDEX_EXCLUSIVE:
                raise ValueError(f"S2 pair out of bounds: {pair_index}")
            decoded_pairs.append((pair_index, tile_id))
    if cursor != len(meta):
        raise ValueError(f"S2 metadata stream has {len(meta) - cursor} trailing bytes")

    actions = _decode_adaptive_arithmetic_actions(
        action_code,
        nbits=action_nbits,
        record_count=total_records,
    )
    out = bytearray(total_records * 4)
    out_offset = 0
    for (pair_index, tile_id), action_id in zip(decoded_pairs, actions, strict=True):
        out[out_offset:out_offset + 2] = int(pair_index).to_bytes(2, "little")
        out[out_offset + 2] = int(tile_id)
        out[out_offset + 3] = int(action_id)
        out_offset += 4
    return bytes(out)


def _decode_split_seg_tile_actions(raw: bytes) -> bytes:
    """Decode S1 split tile-group actions into runtime raw4 records.

    Wire format after Brotli:
    ``b"S1" + group_count + (tile_delta, count)* + pair_deltas* + actions*``.
    The pair delta stream is reset per tile group.  Splitting the metadata,
    deltas, and action ids improves Brotli's local model while preserving the
    same decoded action records as SG2.
    """

    offset = len(SEG_TILE_ACTION_SPLIT_MAGIC)
    group_count, offset = _read_uvarint(raw, offset, max_value=256)
    if group_count <= 0:
        raise ValueError("split seg_tile_actions has zero groups")

    groups: list[tuple[int, int]] = []
    tile_id = 0
    total_records = 0
    for group_index in range(group_count):
        tile_delta, offset = _read_uvarint(raw, offset, max_value=255)
        if group_index == 0:
            tile_id = tile_delta
        else:
            if tile_delta <= 0:
                raise ValueError("split seg_tile_actions tile deltas must increase")
            tile_id += tile_delta
        if tile_id >= 192:
            raise ValueError(f"split seg_tile_actions tile out of bounds: {tile_id}")
        count, offset = _read_uvarint(raw, offset, max_value=PUBLIC_PR75_MAX_ACTION_RECORDS)
        if count <= 0:
            raise ValueError("split seg_tile_actions group has zero records")
        total_records += count
        if total_records > PUBLIC_PR75_MAX_ACTION_RECORDS:
            raise ValueError(f"unreasonable split seg_tile_actions count: {total_records}")
        groups.append((tile_id, count))

    decoded_pairs: list[tuple[int, int]] = []
    for tile_id, count in groups:
        pair_index = 0
        for record_index in range(count):
            delta, offset = _read_uvarint(
                raw,
                offset,
                max_value=PUBLIC_PR75_MAX_PAIR_INDEX_EXCLUSIVE - 1,
            )
            pair_index = delta if record_index == 0 else pair_index + delta
            if pair_index >= PUBLIC_PR75_MAX_PAIR_INDEX_EXCLUSIVE:
                raise ValueError(
                    f"split seg_tile_actions pair out of bounds: {pair_index}"
                )
            decoded_pairs.append((pair_index, tile_id))

    actions_end = offset + total_records
    if actions_end != len(raw):
        raise ValueError(
            f"split seg_tile_actions length mismatch: expected {actions_end}, got {len(raw)}"
        )
    out = bytearray(total_records * 4)
    out_offset = 0
    for pair_index, tile_id in decoded_pairs:
        action_id = int(raw[offset])
        offset += 1
        if action_id >= 108:
            raise ValueError(f"split seg_tile_actions action id outside dictionary: {action_id}")
        out[out_offset:out_offset + 2] = int(pair_index).to_bytes(2, "little")
        out[out_offset + 2] = int(tile_id)
        out[out_offset + 3] = action_id
        out_offset += 4
    return bytes(out)


def _decode_seg_tile_action_dict(data: bytes) -> bytes:
    """Decode and validate a charged custom PR75 tile-action dictionary."""
    import brotli

    raw = brotli.decompress(data)
    header_size = struct.calcsize(SEG_TILE_ACTION_DICT_HEADER_STRUCT)
    if len(raw) < header_size:
        raise ValueError("seg_tile_action_dict payload is too short")
    magic, version, count = struct.unpack_from(
        SEG_TILE_ACTION_DICT_HEADER_STRUCT, raw, 0
    )
    if magic != SEG_TILE_ACTION_DICT_MAGIC or version != 1:
        raise ValueError(
            f"unsupported seg_tile_action_dict header: magic={magic!r} version={version}"
        )
    if count <= 0 or count > 256:
        raise ValueError(f"unreasonable seg_tile_action_dict count: {count}")
    expected = header_size + count * 3 * 4
    if len(raw) != expected:
        raise ValueError(
            f"seg_tile_action_dict length mismatch: expected {expected}, got {len(raw)}"
        )
    return raw


def _decode_packed_seg_tile_actions(
    data: bytes,
    *,
    record_count: int,
    dictionary_count: int,
) -> bytes:
    """Decode P5 3-byte action records into the runtime 4-byte record form."""
    import brotli

    if record_count <= 0 or record_count > 10_000:
        raise ValueError(f"unreasonable packed seg_tile_actions count: {record_count}")
    if dictionary_count <= 0 or dictionary_count > 64:
        raise ValueError(
            f"P5 requires a 1..64 action dictionary, got {dictionary_count}"
        )
    packed = brotli.decompress(data)
    expected = record_count * 3
    if len(packed) != expected:
        raise ValueError(
            f"packed seg_tile_actions length mismatch: expected {expected}, got {len(packed)}"
        )
    out = bytearray(record_count * 4)
    out_offset = 0
    for offset in range(0, len(packed), 3):
        word = packed[offset] | (packed[offset + 1] << 8) | (packed[offset + 2] << 16)
        pair_index = word & 0x3FF
        tile_id = (word >> 10) & 0xFF
        action_id = (word >> 18) & 0x3F
        if action_id >= dictionary_count:
            raise ValueError(
                f"packed seg_tile_actions action id {action_id} outside dictionary"
            )
        out[out_offset:out_offset + 2] = pair_index.to_bytes(2, "little")
        out[out_offset + 2] = tile_id
        out[out_offset + 3] = action_id
        out_offset += 4
    return bytes(out)


def _minimal_uvarint_length(value: int) -> int:
    if value < 0:
        raise ValueError(f"negative delta-varint value: {value}")
    if value == 0:
        return 1
    return (value.bit_length() + 6) // 7


def _read_uvarint(
    data: bytes,
    offset: int,
    *,
    max_value: int | None = None,
) -> tuple[int, int]:
    value = 0
    shift = 0
    start = offset
    while offset < len(data):
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            consumed = offset - start
            if consumed != _minimal_uvarint_length(value):
                raise ValueError(
                    f"noncanonical delta-varint at byte {start}: "
                    f"value {value} used {consumed} bytes"
                )
            if max_value is not None and value > max_value:
                raise ValueError(
                    f"delta-varint value {value} at byte {start} exceeds max {max_value}"
                )
            return value, offset
        shift += 7
        if shift > 63:
            break
    raise ValueError(f"truncated or overlong delta-varint at byte {start}")


def _decode_delta_varint_seg_tile_actions(
    data: bytes,
    *,
    record_count: int,
) -> bytes:
    """Decode P6 pair-delta varint action records into P3 runtime records."""
    import brotli

    if record_count <= 0 or record_count > PUBLIC_PR75_MAX_ACTION_RECORDS:
        raise ValueError(f"unreasonable delta-varint seg_tile_actions count: {record_count}")
    packed = brotli.decompress(data)
    out = bytearray(record_count * 4)
    offset = 0
    out_offset = 0
    pair_index = 0
    for _idx in range(record_count):
        delta, offset = _read_uvarint(
            packed,
            offset,
            max_value=PUBLIC_PR75_MAX_PAIR_INDEX_EXCLUSIVE - 1,
        )
        pair_index += delta
        if pair_index >= PUBLIC_PR75_MAX_PAIR_INDEX_EXCLUSIVE:
            raise ValueError(f"delta-varint seg_tile_actions pair out of bounds: {pair_index}")
        if offset + 2 > len(packed):
            raise ValueError("delta-varint seg_tile_actions payload ended inside record")
        tile_id = packed[offset]
        action_id = packed[offset + 1]
        offset += 2
        out[out_offset:out_offset + 2] = pair_index.to_bytes(2, "little")
        out[out_offset + 2] = tile_id
        out[out_offset + 3] = action_id
        out_offset += 4
    if offset != len(packed):
        raise ValueError(
            f"delta-varint seg_tile_actions has {len(packed) - offset} trailing bytes"
        )
    return bytes(out)


def _try_parse_public_pr75_qzs3_qp1_segactions_payload(
    payload: bytes,
) -> tuple[dict[str, Any], dict[str, bytes]] | None:
    """Parse PR75/qpose14_r55_segactions_minp-style single-blob payloads.

    Supported wire forms:
    - fixed public slice: mask, Brotli(QZS3 model), Brotli(seg actions),
      Brotli(QP1 pose) for the 276641-byte public PR75 blob;
    - self-describing P3: ``b"P3" + <u32 mask_len, u16 model_len,
      u16 actions_len> + mask + model + actions + pose``.
    - self-describing P4: ``b"P4" + <u32 mask_len, u16 model_len,
      u16 dict_len, u16 actions_len> + mask + model + dict + actions + pose``.
    - self-describing P5: ``b"P5" + <u32 mask_len, u16 model_len,
      u16 dict_len, u16 actions_len, u16 record_count>`` followed by the same
      slices as P4, except actions are packed 3-byte records.
    - self-describing P6: ``b"P6" + <u32 mask_len, u16 model_len,
      u16 actions_len, u16 record_count>`` followed by mask, model,
      Brotli(delta-varint actions), and pose.  The delta-varint stream expands
      to the same fixed-dictionary 4-byte runtime records as P3.

    The decoded runtime members are ordinary masks/renderer/poses plus a
    raw ``seg_tile_actions.bin`` payload that inflate applies to fake2.
    """
    import brotli

    is_self_describing_pr75 = payload[:2] in {b"P3", b"P4", b"P5", b"P6"}
    has_action_dict = False
    packed_actions = False
    fixed_slice_candidates: tuple[tuple[int, int], ...] = ()
    record_count = 0
    if payload.startswith(b"P3"):
        header_size = 2 + struct.calcsize("<IHH")
        if len(payload) <= header_size:
            return None
        mask_len, model_len, actions_len = struct.unpack_from("<IHH", payload, 2)
        dict_len = 0
        cursor = header_size
        if min(mask_len, model_len, actions_len) <= 0:
            return None
        if cursor + mask_len + model_len + actions_len >= len(payload):
            return None
    elif payload.startswith(b"P4"):
        header_size = 2 + struct.calcsize("<IHHH")
        if len(payload) <= header_size:
            return None
        mask_len, model_len, dict_len, actions_len = struct.unpack_from(
            "<IHHH", payload, 2
        )
        cursor = header_size
        has_action_dict = True
        if min(mask_len, model_len, dict_len, actions_len) <= 0:
            return None
        if cursor + mask_len + model_len + dict_len + actions_len >= len(payload):
            return None
    elif payload.startswith(b"P5"):
        header_size = 2 + struct.calcsize("<IHHHH")
        if len(payload) <= header_size:
            return None
        mask_len, model_len, dict_len, actions_len, record_count = struct.unpack_from(
            "<IHHHH", payload, 2
        )
        cursor = header_size
        has_action_dict = True
        packed_actions = True
        if min(mask_len, model_len, dict_len, actions_len, record_count) <= 0:
            return None
        if cursor + mask_len + model_len + dict_len + actions_len >= len(payload):
            return None
    elif payload.startswith(b"P6"):
        header_size = 2 + struct.calcsize("<IHHH")
        if len(payload) <= header_size:
            return None
        mask_len, model_len, actions_len, record_count = struct.unpack_from(
            "<IHHH", payload, 2
        )
        cursor = header_size
        dict_len = 0
        if min(mask_len, model_len, actions_len, record_count) <= 0:
            return None
        if cursor + mask_len + model_len + actions_len >= len(payload):
            return None
    elif len(payload) > (
        PUBLIC_PR75_MASK_LEN + min(PUBLIC_PR75_MODEL_LENS) + PUBLIC_PR75_ACTIONS_LEN
    ):
        mask_len = PUBLIC_PR75_MASK_LEN
        model_len = PUBLIC_PR75_MODEL_LEN
        actions_len = PUBLIC_PR75_ACTIONS_LEN
        cursor = 0
        dict_len = 0
        fixed_slice_candidates = tuple(
            (variant_model_len, variant_actions_len)
            for variant_total_len, variant_model_len, variant_actions_len in PUBLIC_PR75_FIXED_SLICE_VARIANTS
            if len(payload) == variant_total_len
        )
        if not fixed_slice_candidates:
            fixed_slice_candidates = tuple(
                (variant_model_len, variant_actions_len)
                for variant_model_len in PUBLIC_PR75_MODEL_LENS
                for variant_actions_len in PUBLIC_PR75_ACTION_LENS
            )
    else:
        return None

    def try_decode_slices(
        candidate_model_len: int,
        candidate_actions_len: int,
    ) -> tuple[dict[str, bytes], bytes, bytes, bytes | None, bytes, bytes] | None:
        nonlocal last_self_describing_error
        mask_start = cursor
        mask_end = mask_start + mask_len
        model_end = mask_end + candidate_model_len
        dict_end = model_end + dict_len
        actions_end = dict_end + candidate_actions_len
        if actions_end >= len(payload):
            return None
        candidate_raw_slices = {
            "masks.mkv": payload[mask_start:mask_end],
            "renderer.bin": payload[mask_end:model_end],
            "seg_tile_action_dict.bin": payload[model_end:dict_end],
            "seg_tile_actions.bin": payload[dict_end:actions_end],
            "optimized_poses.bin": payload[actions_end:],
        }
        try:
            candidate_masks = brotli.decompress(candidate_raw_slices["masks.mkv"])
            candidate_renderer = brotli.decompress(candidate_raw_slices["renderer.bin"])
            candidate_action_dict = (
                _decode_seg_tile_action_dict(candidate_raw_slices["seg_tile_action_dict.bin"])
                if has_action_dict
                else None
            )
            if packed_actions:
                header_size = struct.calcsize(SEG_TILE_ACTION_DICT_HEADER_STRUCT)
                _magic, _version, dictionary_count = struct.unpack_from(
                    SEG_TILE_ACTION_DICT_HEADER_STRUCT, candidate_action_dict, 0
                )
                candidate_actions = _decode_packed_seg_tile_actions(
                    candidate_raw_slices["seg_tile_actions.bin"],
                    record_count=record_count,
                    dictionary_count=dictionary_count,
                )
            elif payload.startswith(b"P6"):
                candidate_actions = _decode_delta_varint_seg_tile_actions(
                    candidate_raw_slices["seg_tile_actions.bin"],
                    record_count=record_count,
                )
            else:
                candidate_actions = _decode_seg_tile_actions(
                    candidate_raw_slices["seg_tile_actions.bin"]
                )
            candidate_pose_qp1 = brotli.decompress(candidate_raw_slices["optimized_poses.bin"])
        except Exception as exc:
            if is_self_describing_pr75:
                last_self_describing_error = str(exc)
            return None
        if (
            not _looks_like_mask_obu(candidate_masks)
            or not _looks_like_renderer_payload(candidate_renderer)
            or not candidate_pose_qp1.startswith(b"QP1")
        ):
            if is_self_describing_pr75:
                last_self_describing_error = (
                    "decoded mask, renderer, actions, or pose stream failed validation"
                )
            return None
        return (
            candidate_raw_slices,
            candidate_masks,
            candidate_renderer,
            candidate_action_dict,
            candidate_actions,
            candidate_pose_qp1,
        )

    candidates = fixed_slice_candidates or ((model_len, actions_len),)
    parsed = None
    last_self_describing_error = ""
    for candidate_model_len, candidate_actions_len in candidates:
        parsed = try_decode_slices(candidate_model_len, candidate_actions_len)
        if parsed is not None:
            break
    if parsed is None:
        if is_self_describing_pr75:
            detail = last_self_describing_error or (
                "decoded mask, renderer, actions, or pose stream failed validation"
            )
            raise ValueError(
                f"invalid self-describing PR75 payload: {detail}"
            )
        return None
    raw_slices, masks, renderer, action_dict, actions, pose_qp1 = parsed

    members = {
        "renderer.bin": renderer,
        "masks.mkv": masks,
        "optimized_poses.qp1": pose_qp1,
        "seg_tile_actions.bin": actions,
    }
    if action_dict is not None:
        members["seg_tile_action_dict.bin"] = action_dict
    action_wire = _decode_seg_tile_action_wire(raw_slices["seg_tile_actions.bin"])
    if action_wire.startswith(SEG_TILE_ACTION_SPLIT_MAGIC):
        action_codec = "brotli_seg_tile_actions_split_s1_v1"
    elif action_wire.startswith(SEG_TILE_ACTION_SPLIT2_MAGIC):
        action_codec = "seg_tile_actions_split_s2_adaptive_arith_v1"
    elif payload.startswith(b"P6"):
        action_codec = "brotli_seg_tile_actions_delta_varint_v1"
    else:
        action_codec = "brotli_seg_tile_actions_v1"
    header = {
        "schema": PR64_LEN_TABLE_SCHEMA,
        "payload_format": _public_pr75_payload_format(payload),
        "members": [
            {
                "name": "renderer.bin",
                "bytes": len(raw_slices["renderer.bin"]),
                "sha256": hashlib.sha256(raw_slices["renderer.bin"]).hexdigest(),
                "codec": _renderer_payload_codec_label(renderer),
                "decoded_bytes": len(renderer),
                "decoded_sha256": hashlib.sha256(renderer).hexdigest(),
            },
            {
                "name": "masks.mkv",
                "bytes": len(raw_slices["masks.mkv"]),
                "sha256": hashlib.sha256(raw_slices["masks.mkv"]).hexdigest(),
                "codec": "brotli_av1_obu",
                "decoded_bytes": len(masks),
                "decoded_sha256": hashlib.sha256(masks).hexdigest(),
            },
            {
                "name": "seg_tile_actions.bin",
                "bytes": len(raw_slices["seg_tile_actions.bin"]),
                "sha256": hashlib.sha256(raw_slices["seg_tile_actions.bin"]).hexdigest(),
                "codec": action_codec,
                "decoded_bytes": len(actions),
                "decoded_sha256": hashlib.sha256(actions).hexdigest(),
            },
        ],
    }
    if action_dict is not None:
        header["members"].append(
            {
                "name": "seg_tile_action_dict.bin",
                "bytes": len(raw_slices["seg_tile_action_dict.bin"]),
                "sha256": hashlib.sha256(raw_slices["seg_tile_action_dict.bin"]).hexdigest(),
                "codec": "brotli_seg_tile_action_dict_v1",
                "decoded_bytes": len(action_dict),
                "decoded_sha256": hashlib.sha256(action_dict).hexdigest(),
            }
        )
    header["members"].append(
        {
                "name": "optimized_poses.qp1",
                "bytes": len(raw_slices["optimized_poses.bin"]),
                "sha256": hashlib.sha256(raw_slices["optimized_poses.bin"]).hexdigest(),
                "codec": "public_qp1_brotli",
                "decoded_bytes": len(pose_qp1),
                "decoded_sha256": hashlib.sha256(pose_qp1).hexdigest(),
        }
    )
    return header, members


def _try_parse_public_qp19_qpv1_payload(
    payload: bytes,
) -> tuple[dict[str, Any], dict[str, bytes]] | None:
    """Parse PR77's self-describing QP19 mask/renderer/QPV1 pose container."""

    if not payload.startswith(QP19_MAGIC):
        return None
    import brotli

    header_size = 18
    if len(payload) < header_size:
        raise ValueError("QP19 payload is too short")
    version = payload[4]
    flags = payload[5]
    if version != 1:
        raise ValueError(f"unsupported QP19 payload version: {version}")
    mask_len, model_len, pose_len = struct.unpack_from("<III", payload, 6)
    if min(mask_len, model_len, pose_len) <= 0:
        raise ValueError("QP19 payload has nonpositive member length")
    mask_start = header_size
    mask_end = mask_start + int(mask_len)
    model_end = mask_end + int(model_len)
    pose_end = model_end + int(pose_len)
    if pose_end != len(payload):
        raise ValueError(f"QP19 payload length mismatch: header={pose_end} actual={len(payload)}")
    raw_slices = {
        "masks.mkv": payload[mask_start:mask_end],
        "renderer.bin": payload[mask_end:model_end],
        "optimized_poses.bin": payload[model_end:pose_end],
    }
    try:
        masks = brotli.decompress(raw_slices["masks.mkv"])
        renderer = brotli.decompress(raw_slices["renderer.bin"])
        pose_qpv1 = brotli.decompress(raw_slices["optimized_poses.bin"])
    except Exception as exc:
        raise ValueError(f"invalid QP19 Brotli member: {exc}") from exc
    if not _looks_like_mask_obu(masks):
        raise ValueError("QP19 decoded masks failed mask magic validation")
    if not _looks_like_renderer_payload(renderer):
        raise ValueError("QP19 decoded renderer failed renderer magic validation")
    if not pose_qpv1.startswith(QPV1_MAGIC):
        raise ValueError(f"QP19 decoded pose failed QPV1 magic: {pose_qpv1[:4]!r}")
    poses = _decode_pose_qpv1(pose_qpv1)
    members = {
        "renderer.bin": renderer,
        "masks.mkv": masks,
        "optimized_poses.bin": poses,
    }
    header = {
        "schema": PR64_LEN_TABLE_SCHEMA,
        "payload_format": "public_pr77_qp19_qzs3_qpv1_v1",
        "payload_header": {
            "kind": "QP19",
            "version": int(version),
            "flags": int(flags),
            "bytes": header_size,
        },
        "members": [
            {
                "name": "renderer.bin",
                "bytes": len(raw_slices["renderer.bin"]),
                "sha256": hashlib.sha256(raw_slices["renderer.bin"]).hexdigest(),
                "codec": _renderer_payload_codec_label(renderer),
                "decoded_bytes": len(renderer),
                "decoded_sha256": hashlib.sha256(renderer).hexdigest(),
            },
            {
                "name": "masks.mkv",
                "bytes": len(raw_slices["masks.mkv"]),
                "sha256": hashlib.sha256(raw_slices["masks.mkv"]).hexdigest(),
                "codec": "brotli_av1_obu",
                "decoded_bytes": len(masks),
                "decoded_sha256": hashlib.sha256(masks).hexdigest(),
            },
            {
                "name": "optimized_poses.bin",
                "bytes": len(raw_slices["optimized_poses.bin"]),
                "sha256": hashlib.sha256(raw_slices["optimized_poses.bin"]).hexdigest(),
                "codec": "public_qpv1_brotli",
                "decoded_bytes": len(poses),
                "decoded_sha256": hashlib.sha256(poses).hexdigest(),
            },
        ],
    }
    return header, members


def _decode_seg_tile_action_wire(data: bytes) -> bytes:
    if data.startswith(SEG_TILE_ACTION_SPLIT2_MAGIC):
        return data
    import brotli

    return brotli.decompress(data)


def _public_pr75_payload_format(payload: bytes) -> str:
    if payload.startswith(b"P3"):
        return "public_pr75_qzs3_qp1_segactions_p3"
    if payload.startswith(b"P4"):
        return "public_pr75_qzs3_qp1_segactions_p4_custom_dict"
    if payload.startswith(b"P5"):
        return "public_pr75_qzs3_qp1_segactions_p5_packed_custom_dict"
    if payload.startswith(b"P6"):
        return "public_pr75_qzs3_qp1_segactions_p6_delta_varint"
    return "public_pr75_qzs3_qp1_segactions_fixed_slices"


def _decode_public_qpose14_uint16(
    data: bytes,
    *,
    pose_dim: int = 6,
    velocity_offset: float = 20.0,
    velocity_scale: float = 512.0,
    pose_scale: float = 2048.0,
) -> bytes:
    if len(data) % (pose_dim * 2) != 0:
        raise ValueError(
            f"public qpose14 payload length {len(data)} is not divisible by pose_dim*2"
        )
    n_rows = len(data) // (pose_dim * 2)
    words = struct.unpack("<" + "H" * (n_rows * pose_dim), data)
    out_values: list[float] = []
    for row in range(n_rows):
        for col in range(pose_dim):
            word = words[row * pose_dim + col]
            if col == 0:
                out_values.append(word / velocity_scale + velocity_offset)
            else:
                signed = word - 0x10000 if word >= 0x8000 else word
                out_values.append(signed / pose_scale)
    return struct.pack("<" + "e" * len(out_values), *out_values)


def _decode_public_pr64_velocity_delta(
    data: bytes,
    *,
    pose_dim: int = 6,
    velocity_offset: float = 20.0,
    velocity_scale: float = 512.0,
) -> bytes:
    if len(data) < 2 or (len(data) - 2) % 2:
        raise ValueError("public PR64 velocity-delta payload has invalid length")
    n_rows = ((len(data) - 2) // 2) + 1
    velocity_q = int(struct.unpack_from("<H", data, 0)[0])
    out_values: list[float] = []
    offset = 2
    for row in range(n_rows):
        if row > 0:
            delta = struct.unpack_from("<h", data, offset)[0]
            offset += 2
            velocity_q += int(delta)
        out_values.append(velocity_q / velocity_scale + velocity_offset)
        out_values.extend([0.0] * (pose_dim - 1))
    return struct.pack("<" + "e" * len(out_values), *out_values)


def _try_parse_public_pr63_qpose14_payload(
    payload: bytes,
) -> tuple[dict[str, Any], dict[str, bytes]] | None:
    mask_len = 219_472
    model_len = 66_841
    if len(payload) <= mask_len + model_len:
        return None
    raw_slices = {
        "masks.mkv": payload[:mask_len],
        "renderer.bin": payload[mask_len:mask_len + model_len],
        "optimized_poses.bin": payload[mask_len + model_len:],
    }
    try:
        import brotli
        masks = brotli.decompress(raw_slices["masks.mkv"])
        renderer = brotli.decompress(raw_slices["renderer.bin"])
        pose_qpose14 = brotli.decompress(raw_slices["optimized_poses.bin"])
    except Exception:
        return None
    if not _looks_like_mask_obu(masks) or not _looks_like_renderer_payload(renderer):
        return None
    poses = _decode_public_qpose14_uint16(pose_qpose14)
    members = {
        "renderer.bin": renderer,
        "masks.mkv": masks,
        "optimized_poses.bin": poses,
    }
    header = {
        "schema": PR64_LEN_TABLE_SCHEMA,
        "payload_format": "public_pr63_qpose14_fixed_slices",
        "members": [
            {
                "name": "renderer.bin",
                "bytes": len(raw_slices["renderer.bin"]),
                "sha256": hashlib.sha256(raw_slices["renderer.bin"]).hexdigest(),
                "codec": "brotli_torch_fp4",
                "decoded_bytes": len(renderer),
                "decoded_sha256": hashlib.sha256(renderer).hexdigest(),
            },
            {
                "name": "masks.mkv",
                "bytes": len(raw_slices["masks.mkv"]),
                "sha256": hashlib.sha256(raw_slices["masks.mkv"]).hexdigest(),
                "codec": "brotli_av1_obu",
                "decoded_bytes": len(masks),
                "decoded_sha256": hashlib.sha256(masks).hexdigest(),
            },
            {
                "name": "optimized_poses.bin",
                "bytes": len(raw_slices["optimized_poses.bin"]),
                "sha256": hashlib.sha256(raw_slices["optimized_poses.bin"]).hexdigest(),
                "codec": "public_qpose14_uint16_brotli",
                "decoded_bytes": len(poses),
                "decoded_sha256": hashlib.sha256(poses).hexdigest(),
            },
        ],
    }
    return header, members


def _public_pr67_model_lens(payload_len: int) -> list[int]:
    """Return plausible PR67 fixed model-Brotli slice lengths.

    Public submissions used brittle payload-length buckets.  Locally generated
    QZS3/QP1 line-search archives keep the mask/model slices fixed while the
    pose Brotli slice changes, so try all plausible model lengths and let the
    Brotli/QZS3/QP1 validation below select the real contract.
    """

    candidates: list[int] = []

    def add(length: int) -> None:
        if length not in candidates:
            candidates.append(length)

    # Repo-generated QZS3-from-PR63 Torch-FP4 payloads use the same PR67
    # fixed-slice contract but Brotli-compress the converted QZS3 renderer to
    # 55,965 bytes.  Public/repacked PR67 renderers use nearby fixed renderer
    # slice lengths while pose Brotli lengths vary under line search and segment
    # mixing.  Try the whole known renderer-slice set for this family and let
    # Brotli + QZS3 + QP1 validation below select the real contract; never use
    # total payload length as the sole boundary authority.
    if 276_050 <= payload_len <= 277_200:
        for model_len in PUBLIC_PR67_QZS3_MODEL_LENS[:-1]:
            add(model_len)
    if 276_430 <= payload_len <= 276_470:
        add(56_093)
    if 276_550 <= payload_len <= 276_610:
        add(56_221)
    if 278_100 <= payload_len <= 278_130:
        add(57_757)
    if 277_400 <= payload_len <= 277_430:
        add(57_053)
    if 277_350 <= payload_len <= 277_399:
        add(57_031)
    if payload_len == 281_240:
        add(60_880)
    return candidates


def _parse_nerv_member_length(payload: bytes) -> int:
    """Return the exact self-described masks.nrv byte length.

    The unpacker must stay standalone in the contest inflate environment, so it
    mirrors only the small NRV header contract instead of importing tac.
    """

    v1_header_len = 4 + 2 * 6 + 8
    v2_header_len = v1_header_len + 8
    if len(payload) < v1_header_len:
        raise ValueError(
            "PR67/C067 NeRV single-blob payload has malformed masks.nrv: "
            f"blob length {len(payload)} is too small for an NRV header"
        )
    if payload[:4] != NERV_MAGIC:
        raise ValueError(
            "PR67/C067 NeRV single-blob payload has malformed masks.nrv: "
            f"bad magic {payload[:4]!r}, expected {NERV_MAGIC!r}"
        )
    version, num_freqs, hidden_dim, num_classes, depth, weight_dtype = struct.unpack_from(
        "<HHHHHH",
        payload,
        4,
    )
    if version not in (1, 2):
        raise ValueError(
            "PR67/C067 NeRV single-blob payload has malformed masks.nrv: "
            f"unsupported NRV version {version}"
        )
    if not (
        1 <= num_freqs <= 64
        and 1 <= hidden_dim <= 65_535
        and 1 <= num_classes <= 256
    ):
        raise ValueError(
            "PR67/C067 NeRV single-blob payload has malformed masks.nrv: "
            "unreasonable NRV shape "
            f"num_freqs={num_freqs}, hidden_dim={hidden_dim}, num_classes={num_classes}"
        )
    if not (1 <= depth <= 256) or weight_dtype not in (0, 1):
        raise ValueError(
            "PR67/C067 NeRV single-blob payload has malformed masks.nrv: "
            f"unreasonable NRV depth/dtype depth={depth}, weight_dtype={weight_dtype}"
        )
    payload_size = struct.unpack_from("<Q", payload, 4 + 2 * 6)[0]
    scale_table_size = 0
    header_len = v1_header_len
    if version == 2:
        if len(payload) < v2_header_len:
            raise ValueError(
                "PR67/C067 NeRV single-blob payload has malformed masks.nrv: "
                f"blob length {len(payload)} is too small for an NRV2 header"
            )
        scale_table_size = struct.unpack_from("<Q", payload, v1_header_len)[0]
        header_len = v2_header_len
    total = header_len + int(payload_size) + int(scale_table_size)
    if total <= header_len:
        raise ValueError(
            "PR67/C067 NeRV single-blob payload has malformed masks.nrv: "
            "NRV payload has no encoded weights"
        )
    if total > len(payload):
        raise ValueError(
            "PR67/C067 NeRV single-blob payload has malformed masks.nrv: "
            f"declared masks.nrv length {total} exceeds container length {len(payload)}"
        )
    return total


def _parse_public_pr67_nerv_qzs3_qp1_payload(
    payload: bytes,
) -> tuple[dict[str, Any], dict[str, bytes]]:
    """Parse a raw PR67/C067-style NeRV single-blob payload.

    Supported wire order is exactly masks.nrv, Brotli(QZS3 renderer), then
    Brotli(QP1 poses). The masks.nrv member supplies its own byte length; the
    renderer boundary must be proven by Brotli plus QZS3/QP1 content checks.
    """

    nrv_len = _parse_nerv_member_length(payload)
    if len(payload) <= nrv_len:
        raise ValueError(
            "PR67/C067 NeRV single-blob payload is missing renderer/pose streams "
            "after masks.nrv"
        )

    import brotli

    parsed: tuple[dict[str, bytes], bytes, bytes] | None = None
    for model_len in PUBLIC_PR67_QZS3_MODEL_LENS:
        if len(payload) <= nrv_len + model_len:
            continue
        raw_slices = {
            "masks.nrv": payload[:nrv_len],
            "renderer.bin": payload[nrv_len:nrv_len + model_len],
            "optimized_poses.bin": payload[nrv_len + model_len:],
        }
        try:
            renderer = brotli.decompress(raw_slices["renderer.bin"])
            pose_qp1 = brotli.decompress(raw_slices["optimized_poses.bin"])
        except Exception:
            continue
        if not renderer.startswith(b"QZS3") or not pose_qp1.startswith(b"QP1"):
            continue
        parsed = (raw_slices, renderer, pose_qp1)
        break
    if parsed is None:
        raise ValueError(
            "PR67/C067 NeRV single-blob payload has masks.nrv but no valid "
            "Brotli QZS3 renderer + Brotli QP1 pose boundary. Full parsing "
            "requires either the supported masks.nrv/QZS3/QP1 wire contract "
            "or a self-describing RPK1 member table."
        )

    raw_slices, renderer, pose_qp1 = parsed
    poses = _decode_pose_qp1(pose_qp1)
    members = {
        "renderer.bin": renderer,
        "masks.nrv": raw_slices["masks.nrv"],
        "optimized_poses.bin": poses,
    }
    header = {
        "schema": PR64_LEN_TABLE_SCHEMA,
        "payload_format": "public_pr67_nerv_qzs3_qp1_fixed_slices",
        "members": [
            {
                "name": "masks.nrv",
                "bytes": len(raw_slices["masks.nrv"]),
                "sha256": hashlib.sha256(raw_slices["masks.nrv"]).hexdigest(),
                "codec": "raw_nrv",
            },
            {
                "name": "renderer.bin",
                "bytes": len(raw_slices["renderer.bin"]),
                "sha256": hashlib.sha256(raw_slices["renderer.bin"]).hexdigest(),
                "codec": "brotli_qzs3",
                "decoded_bytes": len(renderer),
                "decoded_sha256": hashlib.sha256(renderer).hexdigest(),
            },
            {
                "name": "optimized_poses.bin",
                "bytes": len(raw_slices["optimized_poses.bin"]),
                "sha256": hashlib.sha256(raw_slices["optimized_poses.bin"]).hexdigest(),
                "codec": "public_qp1_brotli",
                "decoded_bytes": len(poses),
                "decoded_sha256": hashlib.sha256(poses).hexdigest(),
            },
        ],
    }
    return header, members


def _try_parse_public_pr67_qzs3_qp1_payload(
    payload: bytes,
) -> tuple[dict[str, Any], dict[str, bytes]] | None:
    mask_len = 219_472
    model_lens = _public_pr67_model_lens(len(payload))
    if not model_lens:
        return None
    import brotli

    parsed: tuple[dict[str, bytes], bytes, bytes, bytes] | None = None
    for model_len in model_lens:
        if len(payload) <= mask_len + model_len:
            continue
        raw_slices = {
            "masks.mkv": payload[:mask_len],
            "renderer.bin": payload[mask_len:mask_len + model_len],
            "optimized_poses.bin": payload[mask_len + model_len:],
        }
        try:
            masks = brotli.decompress(raw_slices["masks.mkv"])
            renderer = brotli.decompress(raw_slices["renderer.bin"])
            pose_qp1 = brotli.decompress(raw_slices["optimized_poses.bin"])
        except Exception:
            continue
        if (
            not _looks_like_mask_obu(masks)
            or not renderer.startswith(b"QZS3")
            or not pose_qp1.startswith(b"QP1")
        ):
            continue
        parsed = (raw_slices, masks, renderer, pose_qp1)
        break
    if parsed is None:
        return None
    raw_slices, masks, renderer, pose_qp1 = parsed
    poses = _decode_pose_qp1(pose_qp1)
    members = {
        "renderer.bin": renderer,
        "masks.mkv": masks,
        "optimized_poses.bin": poses,
    }
    header = {
        "schema": PR64_LEN_TABLE_SCHEMA,
        "payload_format": "public_pr67_qzs3_qp1_fixed_slices",
        "members": [
            {
                "name": "renderer.bin",
                "bytes": len(raw_slices["renderer.bin"]),
                "sha256": hashlib.sha256(raw_slices["renderer.bin"]).hexdigest(),
                "codec": "brotli_qzs3",
                "decoded_bytes": len(renderer),
                "decoded_sha256": hashlib.sha256(renderer).hexdigest(),
            },
            {
                "name": "masks.mkv",
                "bytes": len(raw_slices["masks.mkv"]),
                "sha256": hashlib.sha256(raw_slices["masks.mkv"]).hexdigest(),
                "codec": "brotli_av1_obu",
                "decoded_bytes": len(masks),
                "decoded_sha256": hashlib.sha256(masks).hexdigest(),
            },
            {
                "name": "optimized_poses.bin",
                "bytes": len(raw_slices["optimized_poses.bin"]),
                "sha256": hashlib.sha256(raw_slices["optimized_poses.bin"]).hexdigest(),
                "codec": "public_qp1_brotli",
                "decoded_bytes": len(poses),
                "decoded_sha256": hashlib.sha256(poses).hexdigest(),
            },
        ],
    }
    return header, members


def _parse_compact_payload(payload: bytes) -> tuple[dict[str, Any], dict[str, bytes]]:
    header_size = len(COMPACT_MAGIC) + struct.calcsize(COMPACT_HEADER_STRUCT)
    if len(payload) < header_size:
        raise ValueError("compact renderer payload is too short")
    codec_id, renderer_len, masks_len, pose_len = struct.unpack_from(
        COMPACT_HEADER_STRUCT,
        payload,
        len(COMPACT_MAGIC),
    )
    codec = COMPACT_POSE_CODECS.get(codec_id)
    if codec is None:
        raise ValueError(f"unsupported compact pose codec id: {codec_id}")
    lengths = {
        "renderer.bin": renderer_len,
        "masks.mkv": masks_len,
        "optimized_poses.bin": pose_len,
    }
    if any(n <= 0 for n in lengths.values()):
        raise ValueError(f"compact renderer payload has invalid lengths: {lengths}")

    offset = header_size
    raw_members: dict[str, bytes] = {}
    for name, n_bytes in lengths.items():
        end = offset + n_bytes
        if end > len(payload):
            raise ValueError(f"compact renderer payload member {name} overruns payload")
        raw_members[name] = payload[offset:end]
        offset = end
    if offset != len(payload):
        raise ValueError(
            f"compact renderer payload has {len(payload) - offset} trailing bytes"
        )

    if codec == "raw":
        decoded_pose = raw_members["optimized_poses.bin"]
    elif codec == POSE_FP16_COL_DELTA_CODEC:
        decoded_pose = _decode_pose_fp16_col_delta(raw_members["optimized_poses.bin"])
    elif codec == POSE_QPOSE14_COL_DELTA_CODEC:
        decoded_pose = _decode_pose_qpose14_col_delta(raw_members["optimized_poses.bin"])
    elif codec == POSE_QP1_CODEC:
        decoded_pose = _decode_pose_qp1(raw_members["optimized_poses.bin"])
    else:
        raise ValueError(f"unsupported compact pose codec: {codec!r}")

    members = {
        "renderer.bin": raw_members["renderer.bin"],
        "masks.mkv": raw_members["masks.mkv"],
        "optimized_poses.bin": decoded_pose,
    }
    header = {
        "schema": COMPACT_SCHEMA,
        "payload_format": "rp2_fixed3",
        "members": [
            {
                "name": "renderer.bin",
                "bytes": len(raw_members["renderer.bin"]),
                "sha256": hashlib.sha256(raw_members["renderer.bin"]).hexdigest(),
                "codec": "raw",
            },
            {
                "name": "masks.mkv",
                "bytes": len(raw_members["masks.mkv"]),
                "sha256": hashlib.sha256(raw_members["masks.mkv"]).hexdigest(),
                "codec": "raw",
            },
            {
                "name": "optimized_poses.bin",
                "bytes": len(raw_members["optimized_poses.bin"]),
                "sha256": hashlib.sha256(raw_members["optimized_poses.bin"]).hexdigest(),
                "codec": codec,
                "decoded_bytes": len(decoded_pose),
                "decoded_sha256": hashlib.sha256(decoded_pose).hexdigest(),
            },
        ],
    }
    return header, members


def _decode_pose_fp16_col_delta(data: bytes) -> bytes:
    if len(data) < 8 or data[:4] != b"PCD1":
        raise ValueError("bad pose_fp16_col_delta_v1 payload magic")
    n_rows, n_cols = struct.unpack_from("<HH", data, 4)
    if n_rows <= 0 or n_cols <= 0 or n_rows > 10_000 or n_cols > 64:
        raise ValueError(f"invalid pose delta shape: rows={n_rows}, cols={n_cols}")
    expected = 8 + n_cols * (2 + max(0, n_rows - 1) * 2)
    if len(data) != expected:
        raise ValueError(
            f"pose delta payload length mismatch: expected {expected}, got {len(data)}"
        )

    offset = 8
    columns: list[list[int]] = []
    for _ in range(n_cols):
        prev = struct.unpack_from("<H", data, offset)[0]
        offset += 2
        col = [prev]
        for _r in range(n_rows - 1):
            delta = struct.unpack_from("<h", data, offset)[0]
            offset += 2
            prev = (prev + delta) & 0xFFFF
            col.append(prev)
        columns.append(col)

    out = bytearray(n_rows * n_cols * 2)
    pos = 0
    for row in range(n_rows):
        for col in range(n_cols):
            struct.pack_into("<H", out, pos, columns[col][row])
            pos += 2
    return bytes(out)


def _decode_pose_qpose14_col_delta(
    data: bytes,
    *,
    velocity_offset: float = 20.0,
    velocity_scale: float = 512.0,
    pose_scale: float = 2048.0,
) -> bytes:
    if len(data) < 8 or data[:4] != b"QP14":
        raise ValueError("bad pose_qpose14_col_delta_v1 payload magic")
    n_rows, n_cols = struct.unpack_from("<HH", data, 4)
    if n_rows <= 0 or n_cols <= 0 or n_rows > 10_000 or n_cols > 64:
        raise ValueError(f"invalid qpose delta shape: rows={n_rows}, cols={n_cols}")
    expected = 8 + n_cols * (2 + max(0, n_rows - 1) * 2)
    if len(data) != expected:
        raise ValueError(
            f"qpose delta payload length mismatch: expected {expected}, got {len(data)}"
        )

    offset = 8
    columns: list[list[int]] = []
    for _ in range(n_cols):
        prev = struct.unpack_from("<H", data, offset)[0]
        offset += 2
        col = [prev]
        for _r in range(n_rows - 1):
            delta = struct.unpack_from("<h", data, offset)[0]
            offset += 2
            prev = (prev + delta) & 0xFFFF
            col.append(prev)
        columns.append(col)

    out_values: list[float] = []
    for row in range(n_rows):
        for col in range(n_cols):
            word = columns[col][row]
            if col == 0:
                out_values.append(word / velocity_scale + velocity_offset)
            else:
                signed = word - 0x10000 if word >= 0x8000 else word
                out_values.append(signed / pose_scale)
    return struct.pack("<" + "e" * len(out_values), *out_values)


def _decode_pose_qp1(
    data: bytes,
    *,
    pose_dim: int = 6,
    velocity_offset: float = 20.0,
    velocity_scale: float = 512.0,
) -> bytes:
    """Decode PR #67 QP1 velocity-only ZigZag-VLQ poses to fp16 bytes."""
    if len(data) < 5 or data[:3] != b"QP1":
        raise ValueError("bad pose_qp1_v1 payload magic")

    first = struct.unpack_from("<H", data, 3)[0]
    vals = [first]
    cursor = 5
    while cursor < len(data):
        shift = 0
        acc = 0
        while True:
            if cursor >= len(data):
                raise ValueError("truncated QP1 VLQ payload")
            byte = data[cursor]
            cursor += 1
            acc |= (byte & 0x7F) << shift
            if byte < 0x80:
                break
            shift += 7
        delta = (acc >> 1) ^ -(acc & 1)
        vals.append((vals[-1] + delta) & 0xFFFF)

    out_values: list[float] = []
    for word in vals:
        out_values.append(word / velocity_scale + velocity_offset)
        out_values.extend([0.0] * (pose_dim - 1))
    return struct.pack("<" + "e" * len(out_values), *out_values)


def _read_zigzag_varint(data: bytes, cursor: int) -> tuple[int, int]:
    shift = 0
    acc = 0
    while True:
        if cursor >= len(data):
            raise ValueError("truncated ZigZag-VLQ payload")
        byte = data[cursor]
        cursor += 1
        acc |= (byte & 0x7F) << shift
        if byte < 0x80:
            break
        shift += 7
        if shift > 63:
            raise ValueError("overlong ZigZag-VLQ payload")
    return (acc >> 1) ^ -(acc & 1), cursor


def _decode_pose_qpv1(data: bytes, *, pose_dim: int = 6) -> bytes:
    """Decode PR77 QPV1 multidimensional pose streams to fp16 bytes."""

    if len(data) < 7 or data[:4] != QPV1_MAGIC:
        raise ValueError("bad pose_qpv1_v1 payload magic")
    count = struct.unpack_from("<H", data, 4)[0]
    dim_count = data[6]
    if count <= 0 or dim_count <= 0:
        raise ValueError(f"invalid QPV1 pose shape: rows={count}, dims={dim_count}")
    cursor = 7
    out_values = [0.0] * (count * pose_dim)
    seen: set[int] = set()
    for _idx in range(dim_count):
        if cursor + 13 > len(data):
            raise ValueError("truncated QPV1 dimension header")
        dim = data[cursor]
        cursor += 1
        if dim in seen:
            raise ValueError(f"duplicate QPV1 pose dimension: {dim}")
        if dim >= pose_dim:
            raise ValueError(f"QPV1 pose dimension {dim} outside pose_dim {pose_dim}")
        seen.add(dim)
        offset = struct.unpack_from("<f", data, cursor)[0]
        cursor += 4
        scale = struct.unpack_from("<f", data, cursor)[0]
        cursor += 4
        if not scale:
            raise ValueError(f"invalid QPV1 scale for dim {dim}: {scale}")
        value = struct.unpack_from("<i", data, cursor)[0]
        cursor += 4
        values = [value]
        while len(values) < count:
            delta, cursor = _read_zigzag_varint(data, cursor)
            values.append(values[-1] + delta)
        for row, q_value in enumerate(values):
            out_values[row * pose_dim + dim] = float(offset) + float(q_value) / float(scale)
    if cursor != len(data):
        raise ValueError(f"QPV1 payload has {len(data) - cursor} trailing bytes")
    return struct.pack("<" + "e" * len(out_values), *out_values)


def _decode_pose_fp16_velocity_only(
    data: bytes,
    *,
    velocity_offset: float = 20.0,
    velocity_scale: float = 512.0,
) -> bytes:
    if len(data) < 10 or data[:4] != b"PVL1":
        raise ValueError("bad pose_fp16_velocity_only_v1 payload magic")
    n_rows, n_cols, first_velocity_q = struct.unpack_from("<HHH", data, 4)
    if n_rows <= 0 or n_cols <= 0 or n_rows > 10_000 or n_cols > 64:
        raise ValueError(f"invalid velocity-only pose shape: rows={n_rows}, cols={n_cols}")
    expected = 10 + max(0, n_rows - 1) * 2
    if len(data) != expected:
        raise ValueError(
            f"velocity-only pose payload length mismatch: expected {expected}, got {len(data)}"
        )

    velocity_q = first_velocity_q
    out_values: list[float] = []
    offset = 10
    for row in range(n_rows):
        if row > 0:
            delta = struct.unpack_from("<h", data, offset)[0]
            offset += 2
            velocity_q = (velocity_q + delta) & 0xFFFF
        out_values.append(velocity_q / velocity_scale + velocity_offset)
        out_values.extend([0.0] * (n_cols - 1))
    return struct.pack("<" + "e" * len(out_values), *out_values)


def _decode_pose_fp16_velocity_residual_topk(
    data: bytes,
    *,
    velocity_offset: float = 20.0,
    velocity_scale: float = 512.0,
) -> bytes:
    if len(data) < 12 or data[:4] != b"PVR1":
        raise ValueError("bad pose_fp16_velocity_residual_topk_v1 payload magic")
    n_rows, n_cols, topk, first_velocity_q = struct.unpack_from("<HHHH", data, 4)
    if n_rows <= 0 or n_cols <= 1 or n_rows > 10_000 or n_cols > 64:
        raise ValueError(f"invalid residual pose shape: rows={n_rows}, cols={n_cols}")
    max_atoms = n_rows * (n_cols - 1)
    if topk > max_atoms:
        raise ValueError(f"residual pose topk {topk} exceeds available atoms {max_atoms}")

    means_start = 12
    means_end = means_start + (n_cols - 1) * 2
    deltas_end = means_end + max(0, n_rows - 1) * 2
    expected = deltas_end + topk * 4
    if len(data) != expected:
        raise ValueError(
            f"residual pose payload length mismatch: expected {expected}, got {len(data)}"
        )
    mean_words = [
        struct.unpack_from("<H", data, means_start + (dim - 1) * 2)[0]
        for dim in range(1, n_cols)
    ]

    velocity_q = first_velocity_q
    out = bytearray(n_rows * n_cols * 2)
    delta_offset = means_end
    for row in range(n_rows):
        if row > 0:
            delta = struct.unpack_from("<h", data, delta_offset)[0]
            delta_offset += 2
            velocity_q = (velocity_q + delta) & 0xFFFF
        row_offset = row * n_cols * 2
        struct.pack_into("<e", out, row_offset, velocity_q / velocity_scale + velocity_offset)
        for dim in range(1, n_cols):
            struct.pack_into("<H", out, row_offset + dim * 2, mean_words[dim - 1])

    atom_offset = deltas_end
    seen: set[int] = set()
    for _ in range(topk):
        key, half_word = struct.unpack_from("<HH", data, atom_offset)
        atom_offset += 4
        row, dim = divmod(key, n_cols)
        if row >= n_rows or dim <= 0 or dim >= n_cols:
            raise ValueError(f"residual pose atom out of bounds: row={row}, dim={dim}")
        if key in seen:
            raise ValueError(f"duplicate residual pose atom key: {key}")
        seen.add(key)
        struct.pack_into("<H", out, key * 2, half_word)
    return bytes(out)


def unpack_renderer_payload(archive_dir: Path | str) -> dict[str, Any]:
    """Extract a packed renderer payload into normal renderer members."""
    archive_dir = Path(archive_dir)
    if not archive_dir.is_dir():
        raise FileNotFoundError(f"archive dir does not exist: {archive_dir}")

    payload = _read_payload_bytes(archive_dir)
    header, members = _parse_payload(payload)

    for name, data in members.items():
        out_path = archive_dir / name
        if out_path.exists():
            existing = out_path.read_bytes()
            if existing != data:
                raise ValueError(
                    f"renderer payload would overwrite mismatched member {name}: "
                    f"existing_sha256={hashlib.sha256(existing).hexdigest()} "
                    f"payload_sha256={hashlib.sha256(data).hexdigest()}"
                )
            continue
        out_path.write_bytes(data)

    return {
        "schema": header["schema"],
        "payload_format": header.get("payload_format"),
        "source_archive_sha256": header.get("source_archive_sha256"),
        "payload_bytes": len(payload),
        "members": [
            {
                "name": name,
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
            for name, data in members.items()
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive_dir", type=Path)
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional path for extraction summary JSON.",
    )
    args = parser.parse_args(argv)
    try:
        summary = unpack_renderer_payload(args.archive_dir)
    except Exception as exc:  # pragma: no cover - CLI diagnostic surface
        print(f"FATAL: renderer payload unpack failed: {exc}", file=sys.stderr)
        return 1
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    names = ", ".join(m["name"] for m in summary["members"])
    print(
        f"[renderer payload] unpacked {len(summary['members'])} member(s): {names}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
