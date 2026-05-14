#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Deterministically profile contest archive byte budgets.

This is an empirical archive-inspection tool only. It does not inflate frames,
load scorers, import the contest runtime, dispatch jobs, or make score claims.
All score-affecting conclusions still require exact CUDA auth eval on the
identical archive bytes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import lzma
import math
import struct
import zlib
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA = "archive_bit_budget_profile_v1"
TOOL = "experiments/archive_bit_budget_profiler.py"
ORIGINAL_VIDEO_BYTES = 37_545_489
RATE_LAMBDA = 25.0 / ORIGINAL_VIDEO_BYTES
CUDA_SCORE_TRUTH = "archive.zip -> inflate.sh -> upstream/evaluate.py on CUDA"
NO_SCORE_CLAIM = (
    "This profile is byte/compressibility evidence only. It is not score "
    "evidence, not promotion evidence, and not a method retirement signal."
)
ALLOWED_SINGLE_FILE_NAMES = {
    "renderer.bin",
    "masks.mkv",
    "grayscale.mkv",
    "masks.alpha4.mkv",
    "masks.amrc",
    "masks.nrv",
    "masks.cmg2",
    "masks.cmg3",
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
    "seg_tile_action_dict.bin",
    "alpha4_residual_repair.amr1",
    "alpha4_residual_repair.amr1.xz",
    "alpha4_residual_repair.amr1.zlib",
    "alpha4_residual_repair.amr1.br",
    "p",
}


@dataclass(frozen=True)
class Segment:
    name: str
    offset: int
    encoded_bytes: bytes
    codec: str
    decoded_bytes_estimate: int | None = None
    notes: str | None = None


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _safe_zip_name(name: str) -> str:
    path = Path(name)
    if not name or name.startswith("/") or ".." in path.parts:
        raise ValueError(f"unsafe ZIP member path: {name!r}")
    return name


def _safe_payload_member_name(name: str) -> str:
    _safe_zip_name(name)
    if len(Path(name).parts) != 1:
        raise ValueError(f"unsafe renderer payload member path: {name!r}")
    if name not in ALLOWED_SINGLE_FILE_NAMES:
        raise ValueError(f"unknown renderer payload member name: {name!r}")
    return name


def _entropy_bits_per_byte(data: bytes) -> float:
    if not data:
        return 0.0
    counts = [0] * 256
    for value in data:
        counts[value] += 1
    total = float(len(data))
    entropy = 0.0
    for count in counts:
        if count:
            p = count / total
            entropy -= p * math.log2(p)
    return entropy


def _entropy_record(data: bytes) -> dict[str, Any]:
    entropy = _entropy_bits_per_byte(data)
    entropy_bytes = int(math.ceil(entropy * len(data) / 8.0))
    return {
        "bytes": int(len(data)),
        "entropy_bits_per_byte": round(entropy, 12),
        "zero_order_entropy_bytes": entropy_bytes,
        "zero_order_entropy_ratio": None if not data else round(entropy_bytes / len(data), 12),
        "unique_byte_count": len(set(data)),
    }


def _try_brotli_compress(data: bytes, *, quality: int) -> bytes | None:
    try:
        import brotli
    except ImportError:
        return None
    return brotli.compress(data, quality=quality)


def _try_brotli_decompress(data: bytes) -> bytes | None:
    try:
        import brotli
    except ImportError:
        return None
    try:
        return brotli.decompress(data)
    except brotli.error:
        return None


def _compression_probe(data: bytes) -> dict[str, Any]:
    probes: dict[str, Any] = {
        "input_bytes": len(data),
        "zlib_9_bytes": len(zlib.compress(data, level=9)),
        "lzma_preset9_bytes": len(lzma.compress(data, preset=9)),
    }
    for quality in (5, 9, 11):
        compressed = _try_brotli_compress(data, quality=quality)
        if compressed is not None:
            probes[f"brotli_q{quality}_bytes"] = len(compressed)
    best_codec, best_bytes = min(
        ((key, int(value)) for key, value in probes.items() if key.endswith("_bytes")),
        key=lambda item: (item[1], item[0]),
    )
    probes["best_probe"] = {"codec": best_codec, "bytes": best_bytes}
    probes["best_probe_delta_vs_input"] = best_bytes - len(data)
    probes["best_probe_ratio"] = None if not data else round(best_bytes / len(data), 12)
    return probes


def _wire_type_guess(name: str, data: bytes) -> dict[str, Any]:
    brotli_decoded = _try_brotli_decompress(data)
    decoded_magic = b"" if brotli_decoded is None else brotli_decoded[:8]
    if data.startswith(b"P6"):
        guess = "public_pr75_p6_segactions_payload"
    elif data.startswith(b"P5"):
        guess = "public_pr75_p5_segactions_payload"
    elif data.startswith(b"P4"):
        guess = "public_pr75_p4_segactions_payload"
    elif data.startswith(b"P3"):
        guess = "public_pr75_p3_segactions_payload"
    elif data.startswith(b"RPK1"):
        guess = "rpk1_renderer_payload"
    elif data.startswith(b"RP2\x01"):
        guess = "rp2_fixed3_renderer_payload"
    elif data.startswith(b"QZS3"):
        guess = "qzs3_renderer"
    elif data.startswith(b"MQZ1"):
        guess = "mqz_renderer"
    elif data.startswith(b"QFAI"):
        guess = "qfaithful_renderer"
    elif data.startswith(b"QBF1"):
        guess = "qbf1_renderer"
    elif data.startswith(b"QP14"):
        guess = "qpose14_pose"
    elif data.startswith(b"QP1"):
        guess = "qp1_pose"
    elif data.startswith(b"PVL1"):
        guess = "pose_velocity_only"
    elif data.startswith(b"PVR1"):
        guess = "pose_velocity_residual"
    elif data.startswith(b"PK\x03\x04"):
        guess = "nested_zip_or_torch"
    elif data.startswith(b"\x12\x00\x0a\x0a") or data.startswith(b"\x12\x00"):
        guess = "av1_obu_mask_stream"
    elif data.startswith(b"\xfd7zXZ\x00"):
        guess = "xz_stream"
    elif data.startswith(b"\x89PNG\r\n\x1a\n"):
        guess = "png"
    elif brotli_decoded is not None and _looks_like_mask_obu(brotli_decoded):
        guess = "brotli_av1_obu_mask_stream"
    elif decoded_magic.startswith(b"QZS3"):
        guess = "brotli_qzs3_renderer"
    elif decoded_magic.startswith(b"QP14"):
        guess = "brotli_qpose14_pose"
    elif decoded_magic.startswith(b"QP1"):
        guess = "brotli_qp1_pose"
    elif data.startswith(b"{") or data.startswith(b"["):
        guess = "json_like"
    elif name.endswith(".mkv"):
        guess = "mkv_or_av1_mask_stream"
    elif name.endswith((".pt", ".pth")):
        guess = "torch_checkpoint_or_pickle"
    elif name.endswith(".bin"):
        guess = "binary_payload"
    elif name == "p":
        guess = "single_member_payload"
    else:
        guess = "unknown"
    return {
        "guess": guess,
        "magic_hex": data[:8].hex(),
        "name_hint": Path(name).suffix.lstrip(".") or None,
        "brotli_decodable": brotli_decoded is not None,
        "brotli_decoded_bytes": None if brotli_decoded is None else len(brotli_decoded),
        "brotli_decoded_magic_hex": None if brotli_decoded is None else brotli_decoded[:8].hex(),
    }


def _compression_name(compress_type: int) -> str:
    names = {
        zipfile.ZIP_STORED: "stored",
        zipfile.ZIP_DEFLATED: "deflated",
        zipfile.ZIP_BZIP2: "bzip2",
        zipfile.ZIP_LZMA: "lzma",
    }
    return names.get(compress_type, f"zip_method_{compress_type}")


def _zip_member_overhead(info: zipfile.ZipInfo) -> dict[str, int]:
    name_bytes = info.filename.encode("utf-8")
    extra = info.extra or b""
    comment = info.comment or b""
    return {
        "local_header_bytes": 30 + len(name_bytes) + len(extra),
        "central_directory_header_bytes": 46 + len(name_bytes) + len(extra) + len(comment),
    }


def _looks_like_renderer_payload(data: bytes) -> bool:
    return data.startswith((b"QZS3", b"MQZ1", b"QBF1", b"QFAI", b"PK\x03\x04", b"\x80\x02"))


def _looks_like_mask_obu(data: bytes) -> bool:
    return data.startswith(b"\x12\x00\x0a\x0a") or data.startswith(b"\x12\x00")


def _public_pr67_model_lens(payload_len: int) -> list[int]:
    candidates: list[int] = []
    known_qzs3_qp1_model_lens = (55_965, 56_093, 56_221, 57_031, 57_053, 57_757, 60_880)
    for n_bytes in known_qzs3_qp1_model_lens:
        if payload_len > 219_472 + n_bytes:
            candidates.append(n_bytes)
    return sorted(set(candidates))


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


PUBLIC_PR75_FIXED_SLICE_VARIANTS = (
    # (total payload bytes, Brotli(QZS3 renderer) bytes, Brotli(actions) bytes)
    (276_641, 56_034, 236),
    (276_520, 55_914, 236),
    (276_381, 55_756, 255),
    (276_379, 55_756, 253),
    (276_451, 55_756, 325),  # PR77 qzs3_tile_delta_r147 observed 2026-05-03
)


def _parse_public_pr75_segments(payload: bytes) -> tuple[str, list[Segment]] | None:
    """Parse PR75/qpose14_r55_segactions single-member payloads.

    This mirrors the standalone inflate unpacker enough for byte attribution.
    It validates the compressed mask/model/pose magics but does not execute the
    runtime tile-action transform or import scorer/runtime modules.
    """
    if payload.startswith(b"P3"):
        header_size = 2 + struct.calcsize("<IHH")
        if len(payload) <= header_size:
            return None
        mask_len, model_len, actions_len = struct.unpack_from("<IHH", payload, 2)
        dict_len = 0
        record_count = None
        cursor = header_size
    elif payload.startswith(b"P4"):
        header_size = 2 + struct.calcsize("<IHHH")
        if len(payload) <= header_size:
            return None
        mask_len, model_len, dict_len, actions_len = struct.unpack_from("<IHHH", payload, 2)
        record_count = None
        cursor = header_size
    elif payload.startswith(b"P5"):
        header_size = 2 + struct.calcsize("<IHHHH")
        if len(payload) <= header_size:
            return None
        mask_len, model_len, dict_len, actions_len, record_count = struct.unpack_from(
            "<IHHHH",
            payload,
            2,
        )
        cursor = header_size
    elif payload.startswith(b"P6"):
        header_size = 2 + struct.calcsize("<IHHH")
        if len(payload) <= header_size:
            return None
        mask_len, model_len, actions_len, record_count = struct.unpack_from(
            "<IHHH",
            payload,
            2,
        )
        dict_len = 0
        cursor = header_size
    else:
        mask_len = 219_472
        fixed_candidates = [
            (candidate_model_len, candidate_actions_len)
            for candidate_total_len, candidate_model_len, candidate_actions_len in PUBLIC_PR75_FIXED_SLICE_VARIANTS
            if len(payload) == candidate_total_len
        ]
        if not fixed_candidates:
            fixed_candidates = [(56_034, 236), (55_914, 236), (55_756, 255)]
        dict_len = 0
        record_count = None
        cursor = 0
        if len(payload) <= mask_len + min(model for model, _actions in fixed_candidates):
            return None

    candidates = fixed_candidates if "fixed_candidates" in locals() else [(model_len, actions_len)]
    parsed: tuple[int, int, int, int, bytes, bytes, bytes, bytes, bytes, bytes, bytes | None] | None = None
    for candidate_model_len, candidate_actions_len in candidates:
        if min(mask_len, candidate_model_len, candidate_actions_len) <= 0 or dict_len < 0:
            continue
        if record_count is not None and record_count <= 0:
            continue
        mask_start = cursor
        mask_end = mask_start + mask_len
        model_end = mask_end + candidate_model_len
        dict_end = model_end + dict_len
        actions_end = dict_end + candidate_actions_len
        if actions_end >= len(payload):
            continue
        mask = payload[mask_start:mask_end]
        renderer = payload[mask_end:model_end]
        action_dict = payload[model_end:dict_end]
        actions = payload[dict_end:actions_end]
        pose = payload[actions_end:]
        mask_decoded = _try_brotli_decompress(mask)
        renderer_decoded = _try_brotli_decompress(renderer)
        pose_decoded = _try_brotli_decompress(pose)
        if mask_decoded is None or renderer_decoded is None or pose_decoded is None:
            continue
        if not _looks_like_mask_obu(mask_decoded):
            continue
        if not renderer_decoded.startswith(b"QZS3"):
            continue
        if not pose_decoded.startswith(b"QP1"):
            continue
        parsed = (
            mask_start,
            mask_end,
            model_end,
            dict_end,
            mask,
            renderer,
            action_dict,
            actions,
            pose,
            renderer_decoded,
            pose_decoded,
        )
        break
    if parsed is None:
        return None
    (
        mask_start,
        mask_end,
        model_end,
        dict_end,
        mask,
        renderer,
        action_dict,
        actions,
        pose,
        renderer_decoded,
        pose_decoded,
    ) = parsed
    actions_end = dict_end + len(actions)
    mask_decoded = _try_brotli_decompress(mask)
    if mask_decoded is None:
        return None

    segments = [
        Segment(
            "masks.mkv",
            mask_start,
            mask,
            "brotli_av1_obu",
            len(mask_decoded),
            "PR75 mask segment",
        ),
        Segment(
            "renderer.bin",
            mask_end,
            renderer,
            "brotli_qzs3",
            len(renderer_decoded),
            "PR75 renderer segment",
        ),
    ]
    if dict_len:
        decoded_dict = _try_brotli_decompress(action_dict)
        segments.append(
            Segment(
                "seg_tile_action_dict.bin",
                model_end,
                action_dict,
                "brotli_seg_tile_action_dict_v1",
                None if decoded_dict is None else len(decoded_dict),
                "PR75 tile-action dictionary segment",
            )
        )
    decoded_actions = _try_brotli_decompress(actions)
    if payload.startswith(b"P6"):
        action_codec = "seg_tile_actions_delta_varint_v1"
        decoded_action_bytes = None if record_count is None else int(record_count) * 4
    elif payload.startswith(b"P5"):
        action_codec = "brotli_seg_tile_actions_packed_v1"
        decoded_action_bytes = None if record_count is None else int(record_count) * 4
    else:
        action_codec = "brotli_seg_tile_actions_v1"
        decoded_action_bytes = _decoded_pr75_action_bytes_estimate(decoded_actions)
    segments.extend(
        [
            Segment(
                "seg_tile_actions.bin",
                dict_end,
                actions,
                action_codec,
                decoded_action_bytes,
                "PR75 charged tile-action segment",
            ),
            Segment(
                "optimized_poses.qp1",
                actions_end,
                pose,
                "public_qp1_brotli",
                len(pose_decoded),
                "PR75 QP1 pose segment",
            ),
        ]
    )
    return _public_pr75_payload_format(payload), segments


def _read_profile_uvarint(data: bytes, offset: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while offset < len(data):
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, offset
        shift += 7
        if shift > 63:
            break
    raise ValueError("truncated or overlong uvarint")


def _decoded_pr75_action_bytes_estimate(decoded_actions: bytes | None) -> int | None:
    if decoded_actions is None:
        return None
    if decoded_actions.startswith(b"SG2") or (
        len(decoded_actions) % 4 != 0 and len(decoded_actions) % 5 != 0
    ):
        try:
            cursor = 3 if decoded_actions.startswith(b"SG2") else 0
            records = 0
            while cursor < len(decoded_actions):
                _tile, cursor = _read_profile_uvarint(decoded_actions, cursor)
                count, cursor = _read_profile_uvarint(decoded_actions, cursor)
                for _idx in range(count):
                    _delta, cursor = _read_profile_uvarint(decoded_actions, cursor)
                    if cursor >= len(decoded_actions):
                        return None
                    cursor += 1
                    records += 1
            return records * 4
        except ValueError:
            return None
    return len(decoded_actions)


def _parse_rpk1_segments(payload: bytes) -> tuple[str, list[Segment]] | None:
    if not payload.startswith(b"RPK1") or len(payload) < 8:
        return None
    header_len = struct.unpack_from("<I", payload, 4)[0]
    header_start = 8
    header_end = header_start + header_len
    if header_len <= 0 or header_end > len(payload):
        return None
    try:
        header = json.loads(payload[header_start:header_end].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    members = header.get("members")
    if not isinstance(members, list):
        return None
    offset = header_end
    segments: list[Segment] = []
    for item in members:
        if not isinstance(item, Mapping):
            return None
        name = _safe_payload_member_name(str(item.get("name", "")))
        n_bytes = int(item.get("bytes", -1))
        if n_bytes < 0 or offset + n_bytes > len(payload):
            return None
        encoded = payload[offset: offset + n_bytes]
        segments.append(
            Segment(
                name=name,
                offset=offset,
                encoded_bytes=encoded,
                codec=str(item.get("codec", "raw")),
                decoded_bytes_estimate=(
                    None if item.get("decoded_bytes") is None else int(item["decoded_bytes"])
                ),
                notes="RPK1 header-declared member",
            )
        )
        offset += n_bytes
    if offset != len(payload):
        return None
    return "rpk1_renderer_payload", segments


def _parse_rp2_segments(payload: bytes) -> tuple[str, list[Segment]] | None:
    if not payload.startswith(b"RP2\x01") or len(payload) < 20:
        return None
    codec_id, renderer_len, masks_len, pose_len = struct.unpack_from("<B3xIII", payload, 4)
    pose_codecs = {
        0: "raw",
        1: "pose_fp16_col_delta_v1",
        2: "pose_qpose14_col_delta_v1",
        3: "pose_qp1_v1",
    }
    codec = pose_codecs.get(codec_id)
    if codec is None:
        return None
    lengths = [
        ("renderer.bin", renderer_len, "raw"),
        ("masks.mkv", masks_len, "raw"),
        ("optimized_poses.bin", pose_len, codec),
    ]
    offset = 20
    segments: list[Segment] = []
    for name, n_bytes, member_codec in lengths:
        if n_bytes <= 0 or offset + n_bytes > len(payload):
            return None
        encoded = payload[offset: offset + n_bytes]
        segments.append(
            Segment(
                name=name,
                offset=offset,
                encoded_bytes=encoded,
                codec=member_codec,
                notes="RP2 fixed3 header-declared member",
            )
        )
        offset += n_bytes
    if offset != len(payload):
        return None
    return "rp2_fixed3_renderer_payload", segments


def _parse_pr64_len_table(payload: bytes) -> tuple[str, list[Segment]] | None:
    if len(payload) < 12:
        return None
    first_len, second_len, pose_len = struct.unpack_from("<III", payload, 0)
    if min(first_len, second_len, pose_len) <= 0:
        return None
    if 12 + first_len + second_len + pose_len != len(payload):
        return None
    first = payload[12: 12 + first_len]
    second = payload[12 + first_len: 12 + first_len + second_len]
    pose = payload[-pose_len:]
    if _looks_like_renderer_payload(first) and not _looks_like_renderer_payload(second):
        ordered = [
            ("renderer.bin", 12, first, "raw"),
            ("masks.mkv", 12 + first_len, second, "raw"),
        ]
        payload_format = "pr64_len_table_renderer_first"
    else:
        ordered = [
            ("masks.mkv", 12, first, "raw"),
            ("renderer.bin", 12 + first_len, second, "raw"),
        ]
        payload_format = "public_pr64_mask_first_len_table"
    pose_codec = "raw"
    if pose.startswith(b"QP1"):
        pose_codec = "pose_qp1_v1"
    elif pose.startswith(b"QP14"):
        pose_codec = "pose_qpose14_col_delta_v1"
    elif pose.startswith(b"PVL1"):
        pose_codec = "pose_fp16_velocity_only_v1"
    return payload_format, [
        *(Segment(name, offset, data, codec, notes="PR64 length-table member") for name, offset, data, codec in ordered),
        Segment(
            "optimized_poses.bin",
            12 + first_len + second_len,
            pose,
            pose_codec,
            notes="PR64 length-table member",
        ),
    ]


def _parse_public_pr63_or_pr67(payload: bytes) -> tuple[str, list[Segment]] | None:
    mask_len = 219_472
    candidates: list[tuple[str, int]] = [("public_pr63_qpose14_fixed_slices", 66_841)]
    candidates.extend(("public_pr67_qzs3_qp1_fixed_slices", n) for n in _public_pr67_model_lens(len(payload)))
    for payload_format, model_len in candidates:
        if len(payload) <= mask_len + model_len:
            continue
        mask = payload[:mask_len]
        renderer = payload[mask_len: mask_len + model_len]
        pose = payload[mask_len + model_len:]
        mask_decoded = _try_brotli_decompress(mask)
        renderer_decoded = _try_brotli_decompress(renderer)
        pose_decoded = _try_brotli_decompress(pose)
        if mask_decoded is None or renderer_decoded is None or pose_decoded is None:
            continue
        if not _looks_like_mask_obu(mask_decoded):
            continue
        if payload_format.endswith("qzs3_qp1_fixed_slices") and not renderer_decoded.startswith(b"QZS3"):
            continue
        if payload_format.endswith("qzs3_qp1_fixed_slices") and not pose_decoded.startswith(b"QP1"):
            continue
        if payload_format.endswith("qpose14_fixed_slices") and not _looks_like_renderer_payload(renderer_decoded):
            continue
        return payload_format, [
            Segment(
                "masks.mkv",
                0,
                mask,
                "brotli_av1_obu",
                len(mask_decoded),
                "public fixed-slice mask segment",
            ),
            Segment(
                "renderer.bin",
                mask_len,
                renderer,
                "brotli_qzs3" if payload_format.endswith("qzs3_qp1_fixed_slices") else "brotli_torch_fp4",
                len(renderer_decoded),
                "public fixed-slice renderer segment",
            ),
            Segment(
                "optimized_poses.bin",
                mask_len + model_len,
                pose,
                "public_qp1_brotli" if payload_format.endswith("qzs3_qp1_fixed_slices") else "public_qpose14_uint16_brotli",
                len(pose_decoded),
                "public fixed-slice pose segment",
            ),
        ]
    return None


def _payload_segments(payload: bytes) -> dict[str, Any] | None:
    for parser in (
        _parse_rpk1_segments,
        _parse_rp2_segments,
        _parse_pr64_len_table,
        _parse_public_pr75_segments,
        _parse_public_pr63_or_pr67,
    ):
        parsed = parser(payload)
        if parsed is not None:
            payload_format, segments = parsed
            return _segment_record(payload_format, segments)
    brotli_decoded = _try_brotli_decompress(payload)
    if brotli_decoded is not None:
        parsed = _parse_pr64_len_table(brotli_decoded)
        if parsed is not None:
            payload_format, segments = parsed
            return _segment_record(
                f"{payload_format}_outer_brotli",
                segments,
                source_payload_bytes=len(payload),
                decoded_payload_bytes=len(brotli_decoded),
            )
    return None


def _segment_record(
    payload_format: str,
    segments: Sequence[Segment],
    *,
    source_payload_bytes: int | None = None,
    decoded_payload_bytes: int | None = None,
) -> dict[str, Any]:
    out: list[dict[str, Any]] = []
    for index, segment in enumerate(segments):
        data = segment.encoded_bytes
        probe = _compression_probe(data)
        out.append(
            {
                "index": index,
                "name": segment.name,
                "offset": segment.offset,
                "encoded_bytes": len(data),
                "sha256": _sha256_bytes(data),
                "codec": segment.codec,
                "decoded_bytes_estimate": segment.decoded_bytes_estimate,
                "payload_type_guess": _wire_type_guess(segment.name, data),
                "entropy": _entropy_record(data),
                "compression_probe": probe,
                "rate_score_contribution": round(RATE_LAMBDA * len(data), 12),
                "notes": segment.notes,
            }
        )
    return {
        "payload_format": payload_format,
        "source_payload_bytes": source_payload_bytes,
        "decoded_payload_bytes": decoded_payload_bytes,
        "segment_count": len(out),
        "segments": out,
    }


def _self_compression_signal(name: str, current_bytes: int, data: bytes, *, directly_replaceable: bool) -> dict[str, Any]:
    probe = _compression_probe(data)
    best = int(probe["best_probe"]["bytes"])
    savings = max(0, current_bytes - best)
    if current_bytes < 128:
        priority = "low"
        reason = "small byte surface"
    elif savings <= 0:
        priority = "low"
        reason = "generic nested probes do not beat current byte count"
    elif name in {"masks.mkv", "p"}:
        priority = "high"
        reason = "large stream, but any deployed saving must preserve scorer geometry"
    elif name == "renderer.bin":
        priority = "medium"
        reason = "renderer bytes may be reducible only with loader parity and runtime custody"
    else:
        priority = "medium"
        reason = "candidate byte savings require a charged decoder contract"
    return {
        "best_probe": probe["best_probe"],
        "best_probe_delta_vs_current_bytes": best - current_bytes,
        "best_probe_savings_bytes": savings,
        "best_probe_savings_rate_score": round(RATE_LAMBDA * savings, 12),
        "directly_deployable": False,
        "directly_replaceable_zip_member": directly_replaceable,
        "priority": priority,
        "reason": reason,
        "compliance_note": (
            "A self-compression opportunity is dispatchable only after the archive "
            "contains all decoder/runtime bits and exact CUDA auth eval validates "
            "the identical bytes."
        ),
    }


def profile_archive(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"archive not found: {path}")
    archive_bytes = path.stat().st_size
    archive_sha = _sha256_file(path)
    members: list[dict[str, Any]] = []
    with zipfile.ZipFile(path, "r") as zf:
        infos = zf.infolist()
        seen: set[str] = set()
        for index, info in enumerate(infos):
            name = _safe_zip_name(info.filename)
            if name in seen:
                raise ValueError(f"duplicate ZIP member name: {name!r}")
            seen.add(name)
            if info.is_dir():
                raise ValueError(f"directory ZIP members are not contest payload members: {name!r}")
            data = zf.read(info)
            overhead = _zip_member_overhead(info)
            member_zip_bytes_estimate = int(info.compress_size) + overhead["local_header_bytes"] + overhead["central_directory_header_bytes"]
            segments = _payload_segments(data) if name == "p" or len(infos) == 1 else None
            probe = _compression_probe(data)
            members.append(
                {
                    "index": index,
                    "name": name,
                    "compress_type": int(info.compress_type),
                    "compress_type_name": _compression_name(info.compress_type),
                    "compressed_bytes": int(info.compress_size),
                    "uncompressed_bytes": int(info.file_size),
                    "crc32": f"{info.CRC:08x}",
                    "header_offset": int(info.header_offset),
                    **overhead,
                    "member_zip_bytes_estimate": member_zip_bytes_estimate,
                    "member_rate_score_estimate": round(RATE_LAMBDA * member_zip_bytes_estimate, 12),
                    "sha256_uncompressed": _sha256_bytes(data),
                    "payload_type_guess": _wire_type_guess(name, data),
                    "entropy_uncompressed": _entropy_record(data),
                    "compression_probe_uncompressed": probe,
                    "self_compression_signal": _self_compression_signal(
                        name,
                        int(info.compress_size) if info.compress_type == zipfile.ZIP_STORED else int(info.file_size),
                        data,
                        directly_replaceable=info.compress_type == zipfile.ZIP_STORED,
                    ),
                    "fixed_slice_or_payload_anatomy": segments,
                }
            )
    zip_payload_bytes = sum(int(item["compressed_bytes"]) for item in members)
    zip_member_overhead_bytes = sum(
        int(item["local_header_bytes"]) + int(item["central_directory_header_bytes"])
        for item in members
    )
    return {
        "path": str(path),
        "bytes": archive_bytes,
        "sha256": archive_sha,
        "rate_score_contribution": round(RATE_LAMBDA * archive_bytes, 12),
        "zip_payload_compressed_bytes": zip_payload_bytes,
        "zip_member_header_bytes": zip_member_overhead_bytes,
        "zip_global_overhead_bytes": archive_bytes - zip_payload_bytes - zip_member_overhead_bytes,
        "member_count": len(members),
        "members": members,
    }


def _rank_opportunities(archives: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for archive in archives:
        archive_path = str(archive["path"])
        for member in archive["members"]:
            signal = member["self_compression_signal"]
            rows.append(
                {
                    "archive": archive_path,
                    "scope": "zip_member",
                    "name": member["name"],
                    "bytes": member["compressed_bytes"],
                    "best_probe_savings_bytes": signal["best_probe_savings_bytes"],
                    "priority": signal["priority"],
                    "reason": signal["reason"],
                    "directly_deployable": signal["directly_deployable"],
                }
            )
            anatomy = member.get("fixed_slice_or_payload_anatomy") or {}
            for segment in anatomy.get("segments") or []:
                segment_signal = _self_compression_signal(
                    str(segment["name"]),
                    int(segment["encoded_bytes"]),
                    b"",
                    directly_replaceable=False,
                )
                segment_signal["best_probe"] = segment["compression_probe"]["best_probe"]
                best = int(segment["compression_probe"]["best_probe"]["bytes"])
                savings = max(0, int(segment["encoded_bytes"]) - best)
                rows.append(
                    {
                        "archive": archive_path,
                        "scope": "payload_segment",
                        "name": segment["name"],
                        "bytes": segment["encoded_bytes"],
                        "best_probe_savings_bytes": savings,
                        "priority": "high" if segment["name"] == "masks.mkv" and savings > 0 else "medium",
                        "reason": "segment-level planning signal only; requires charged repack/decoder",
                        "directly_deployable": False,
                    }
                )
    return sorted(
        rows,
        key=lambda item: (
            {"high": 0, "medium": 1, "low": 2}.get(str(item["priority"]), 9),
            -int(item["best_probe_savings_bytes"]),
            str(item["archive"]),
            str(item["scope"]),
            str(item["name"]),
        ),
    )


def build_report(archives: Iterable[Path]) -> dict[str, Any]:
    archive_profiles = [profile_archive(path) for path in archives]
    return {
        "schema": SCHEMA,
        "tool": TOOL,
        "score_claim": False,
        "promotion_eligible": False,
        "evidence_grade": "empirical_byte_profile",
        "canonical_score_source_required": CUDA_SCORE_TRUTH,
        "compliance_note": NO_SCORE_CLAIM,
        "contest_formula_rate_term": {
            "original_video_bytes": ORIGINAL_VIDEO_BYTES,
            "rate_lambda": RATE_LAMBDA,
            "archive_rate_contribution": "25 * archive_bytes / 37,545,489",
        },
        "archives": archive_profiles,
        "ranked_self_compression_opportunities": _rank_opportunities(archive_profiles),
    }


def _markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Archive Bit Budget Profile",
        "",
        NO_SCORE_CLAIM,
        "",
        f"- schema: `{report['schema']}`",
        f"- score claim: `{report['score_claim']}`",
        f"- promotion eligible: `{report['promotion_eligible']}`",
        f"- score truth: `{report['canonical_score_source_required']}`",
        "",
        "## Archives",
        "",
        "| archive | bytes | rate contribution | sha256 | members |",
        "|---|---:|---:|---|---:|",
    ]
    for archive in report["archives"]:
        lines.append(
            f"| `{archive['path']}` | {archive['bytes']} | {archive['rate_score_contribution']} | "
            f"`{archive['sha256']}` | {archive['member_count']} |"
        )
    lines.extend(
        [
            "",
            "## ZIP Members",
            "",
            "| archive | member | method | compressed bytes | uncompressed bytes | type guess | best probe delta | rate estimate |",
            "|---|---|---|---:|---:|---|---:|---:|",
        ]
    )
    for archive in report["archives"]:
        for member in archive["members"]:
            lines.append(
                f"| `{Path(str(archive['path'])).name}` | `{member['name']}` | {member['compress_type_name']} | "
                f"{member['compressed_bytes']} | {member['uncompressed_bytes']} | "
                f"{member['payload_type_guess']['guess']} | "
                f"{member['compression_probe_uncompressed']['best_probe_delta_vs_input']} | "
                f"{member['member_rate_score_estimate']} |"
            )
    lines.extend(
        [
            "",
            "## Payload Segments",
            "",
            "| archive | member | payload format | segment | encoded bytes | codec | type guess | best probe delta |",
            "|---|---|---|---|---:|---|---|---:|",
        ]
    )
    for archive in report["archives"]:
        for member in archive["members"]:
            anatomy = member.get("fixed_slice_or_payload_anatomy")
            if not anatomy:
                continue
            for segment in anatomy["segments"]:
                lines.append(
                    f"| `{Path(str(archive['path'])).name}` | `{member['name']}` | "
                    f"{anatomy['payload_format']} | `{segment['name']}` | "
                    f"{segment['encoded_bytes']} | {segment['codec']} | "
                    f"{segment['payload_type_guess']['guess']} | "
                    f"{segment['compression_probe']['best_probe_delta_vs_input']} |"
                )
    lines.extend(
        [
            "",
            "## Candidate Self-Compression Opportunities",
            "",
            "| priority | scope | name | bytes | best probe savings | directly deployable | reason |",
            "|---|---|---|---:|---:|---|---|",
        ]
    )
    for row in report["ranked_self_compression_opportunities"]:
        lines.append(
            f"| {row['priority']} | {row['scope']} | `{row['name']}` | {row['bytes']} | "
            f"{row['best_probe_savings_bytes']} | {row['directly_deployable']} | {row['reason']} |"
        )
    lines.append("")
    return "\n".join(lines)


def _csv_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for archive in report["archives"]:
        for member in archive["members"]:
            rows.append(
                {
                    "row_type": "zip_member",
                    "archive": archive["path"],
                    "member": member["name"],
                    "payload_format": "",
                    "segment": "",
                    "bytes": member["compressed_bytes"],
                    "uncompressed_bytes": member["uncompressed_bytes"],
                    "codec": member["compress_type_name"],
                    "type_guess": member["payload_type_guess"]["guess"],
                    "sha256": member["sha256_uncompressed"],
                    "rate_score_contribution": member["member_rate_score_estimate"],
                    "best_probe_codec": member["compression_probe_uncompressed"]["best_probe"]["codec"],
                    "best_probe_bytes": member["compression_probe_uncompressed"]["best_probe"]["bytes"],
                    "best_probe_delta": member["compression_probe_uncompressed"]["best_probe_delta_vs_input"],
                }
            )
            anatomy = member.get("fixed_slice_or_payload_anatomy")
            if not anatomy:
                continue
            for segment in anatomy["segments"]:
                rows.append(
                    {
                        "row_type": "payload_segment",
                        "archive": archive["path"],
                        "member": member["name"],
                        "payload_format": anatomy["payload_format"],
                        "segment": segment["name"],
                        "bytes": segment["encoded_bytes"],
                        "uncompressed_bytes": segment.get("decoded_bytes_estimate") or "",
                        "codec": segment["codec"],
                        "type_guess": segment["payload_type_guess"]["guess"],
                        "sha256": segment["sha256"],
                        "rate_score_contribution": segment["rate_score_contribution"],
                        "best_probe_codec": segment["compression_probe"]["best_probe"]["codec"],
                        "best_probe_bytes": segment["compression_probe"]["best_probe"]["bytes"],
                        "best_probe_delta": segment["compression_probe"]["best_probe_delta_vs_input"],
                    }
                )
    return rows


def _write_csv(report: Mapping[str, Any], path: Path) -> None:
    rows = _csv_rows(report)
    fieldnames = [
        "row_type",
        "archive",
        "member",
        "payload_format",
        "segment",
        "bytes",
        "uncompressed_bytes",
        "codec",
        "type_guess",
        "sha256",
        "rate_score_contribution",
        "best_probe_codec",
        "best_probe_bytes",
        "best_probe_delta",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", nargs="+", type=Path, help="contest archive ZIP(s) to profile")
    parser.add_argument("--output-json", type=Path, help="write deterministic JSON report")
    parser.add_argument("--output-csv", type=Path, help="write flat CSV rows for members and payload segments")
    parser.add_argument("--output-md", type=Path, help="write Markdown report")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    report = build_report(args.archives)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_bytes(_json_bytes(report))
    if args.output_csv is not None:
        _write_csv(report, args.output_csv)
    if args.output_md is not None:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(_markdown_report(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "archive_count": len(report["archives"]),
                "schema": report["schema"],
                "score_claim": report["score_claim"],
                "promotion_eligible": report["promotion_eligible"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
