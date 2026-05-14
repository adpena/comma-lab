#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Byte-level qpose/QZS3 archive-family forensics.

This is a local reverse-engineering tool only. It does not run the scorer,
does not require CUDA, does not dispatch jobs, and never claims score. The
output is meant to make charged bytes, decoded runtime members, public action
records, and QP1 pose deltas explicit for C-102/PR75/PR77/PR79-style archives.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import lzma
import math
import struct
import sys
import zipfile
import zlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import brotli


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "experiments/results/pr79_archive_binary_forensics_20260503_worker"
)
DEFAULT_UNPACKER = REPO_ROOT / "submissions/robust_current/unpack_renderer_payload.py"
DEFAULT_PR75_PROFILER = REPO_ROOT / "experiments/profile_pr75_minp_archive.py"
CONTEST_ORIGINAL_BYTES = 37_545_489
RATE_LAMBDA = 25.0 / CONTEST_ORIGINAL_BYTES
TARGET_SCORE = 0.31
SCHEMA = "public_qpose_payload_family_forensics_v1"
TOOL = "experiments/analyze_public_qpose_payload_family.py"
MASK_BR_LEN = 219_472
QP19_MAGIC = b"QP19"


@dataclass(frozen=True)
class ArchiveSpec:
    label: str
    path: Path
    role: str


DEFAULT_ARCHIVES = (
    ArchiveSpec(
        "c102",
        REPO_ROOT
        / "experiments/results/lightning_batch/"
        "exact_eval_c091_next_cem_pose_waterfill_top192_s0125_m06_t4_20260503T1238Z/archive.zip",
        "internal_c102_exact_archive",
    ),
    ArchiveSpec(
        "pr75",
        REPO_ROOT / "experiments/results/top_submission_reverse_engineering_20260503_pr75_minp/archive.zip",
        "public_pr75_minp",
    ),
    ArchiveSpec(
        "pr77",
        REPO_ROOT / "experiments/results/top_submission_reverse_engineering_20260503_pr77/archive.zip",
        "public_pr77_tile_delta",
    ),
    ArchiveSpec(
        "pr79",
        REPO_ROOT / "experiments/results/top_submission_reverse_engineering_20260503_pr79/archive.zip",
        "public_pr79_minp_v2",
    ),
)
DEFAULT_C102_EVAL = (
    REPO_ROOT
    / "experiments/results/lightning_batch/"
    "exact_eval_c091_next_cem_pose_waterfill_top192_s0125_m06_t4_20260503T1238Z/contest_auth_eval.json"
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def zip_overhead_profile(path: Path, info: zipfile.ZipInfo) -> dict[str, Any]:
    raw = path.read_bytes()
    eocd_sig = b"PK\x05\x06"
    eocd_offset = raw.rfind(eocd_sig, max(0, len(raw) - 66_000))
    if eocd_offset < 0:
        raise ValueError(f"could not find EOCD in {path}")
    eocd = struct.unpack_from("<IHHHHIIH", raw, eocd_offset)
    cd_size = int(eocd[5])
    cd_offset = int(eocd[6])
    comment_len = int(eocd[7])
    eocd_bytes = len(raw) - eocd_offset
    if eocd_bytes != 22 + comment_len:
        raise ValueError(f"EOCD/comment length mismatch for {path}")

    local = struct.unpack_from("<IHHHHHIIIHH", raw, info.header_offset)
    if local[0] != 0x04034B50:
        raise ValueError(f"bad local ZIP header for {path}")
    local_name_len = int(local[9])
    local_extra_len = int(local[10])
    local_header_bytes = 30 + local_name_len + local_extra_len
    data_start = int(info.header_offset) + local_header_bytes
    data_end = data_start + int(info.compress_size)
    data_descriptor_bytes = max(0, cd_offset - data_end)
    total_payload_bytes = int(info.compress_size)
    return {
        "archive_bytes": int(path.stat().st_size),
        "payload_compressed_bytes": total_payload_bytes,
        "total_overhead_bytes": int(path.stat().st_size) - total_payload_bytes,
        "local_header_bytes": local_header_bytes,
        "central_directory_bytes": cd_size,
        "eocd_and_comment_bytes": eocd_bytes,
        "data_descriptor_bytes": data_descriptor_bytes,
        "member_name_bytes_local": local_name_len,
        "member_extra_bytes_local": local_extra_len,
        "comment_bytes": comment_len,
        "minimal_single_stored_member_overhead_bytes": 100,
        "over_minimal_bytes": int(path.stat().st_size) - total_payload_bytes - 100,
    }


def read_single_payload_zip(path: Path) -> tuple[bytes, dict[str, Any]]:
    with zipfile.ZipFile(path, "r") as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
        names = [info.filename for info in infos]
        if names != ["p"]:
            raise ValueError(f"{path} must contain exactly one member 'p'; got {names!r}")
        info = infos[0]
        payload = zf.read(info)
        zip_meta = {
            "member_name": info.filename,
            "member_compress_type": int(info.compress_type),
            "member_file_size": int(info.file_size),
            "member_compress_size": int(info.compress_size),
            "member_crc": int(info.CRC),
            "member_header_offset": int(info.header_offset),
            "overhead": zip_overhead_profile(path, info),
        }
        return payload, zip_meta


def split_payload_layout(payload: bytes, unpacker: Any) -> dict[str, Any]:
    """Return charged payload slices with offsets and the boundary authority."""

    def add_segment(
        segments: list[dict[str, Any]],
        *,
        name: str,
        offset: int,
        data: bytes,
        codec: str,
    ) -> None:
        segments.append(
            {
                "name": name,
                "offset": int(offset),
                "charged_bytes": int(len(data)),
                "charged_sha256": _sha256_bytes(data),
                "charged_prefix_hex": data[:8].hex(),
                "codec": codec,
                "raw_bytes_ref": data,
            }
        )

    segments: list[dict[str, Any]] = []
    header: dict[str, Any] | None = None
    cursor = 0
    payload_format = "public_pr75_qzs3_qp1_segactions_fixed_slices"
    boundary_authority = "fixed_slice_length_table"
    if payload.startswith(QP19_MAGIC):
        header_size = 18
        if len(payload) < header_size:
            raise ValueError("QP19 payload is too short")
        version = payload[4]
        flags = payload[5]
        if version != 1:
            raise ValueError(f"unsupported QP19 payload version: {version}")
        mask_len, model_len, pose_len = struct.unpack_from("<III", payload, 6)
        cursor = header_size
        header = {
            "kind": "QP19",
            "version": int(version),
            "flags": int(flags),
            "bytes": header_size,
            "mask_br_bytes": int(mask_len),
            "renderer_br_bytes": int(model_len),
            "pose_br_bytes": int(pose_len),
        }
        if min(mask_len, model_len, pose_len) <= 0:
            raise ValueError("QP19 payload has nonpositive member length")
        if cursor + mask_len + model_len + pose_len != len(payload):
            raise ValueError("QP19 payload length mismatch")
        payload_format = "public_pr77_qp19_qzs3_qpv1_v1"
        boundary_authority = "self_describing_qp19_header"
        add_segment(
            segments,
            name="masks.mkv",
            offset=cursor,
            data=payload[cursor:cursor + mask_len],
            codec="brotli_av1_obu",
        )
        model_start = cursor + mask_len
        add_segment(
            segments,
            name="renderer.bin",
            offset=model_start,
            data=payload[model_start:model_start + model_len],
            codec="brotli_qzs3",
        )
        pose_start = model_start + model_len
        add_segment(
            segments,
            name="optimized_poses.qpv1",
            offset=pose_start,
            data=payload[pose_start:pose_start + pose_len],
            codec="public_qpv1_brotli",
        )
        plain_segments = []
        for segment in segments:
            plain = dict(segment)
            plain.pop("raw_bytes_ref")
            plain_segments.append(plain)
        return {
            "payload_format": payload_format,
            "boundary_authority": boundary_authority,
            "header": header,
            "segments": segments,
            "segments_public": plain_segments,
        }
    if payload.startswith(b"P3"):
        mask_len, model_len, actions_len = struct.unpack_from("<IHH", payload, 2)
        header_size = 2 + struct.calcsize("<IHH")
        cursor = header_size
        header = {
            "kind": "P3",
            "bytes": header_size,
            "mask_br_bytes": int(mask_len),
            "renderer_br_bytes": int(model_len),
            "actions_br_bytes": int(actions_len),
        }
        payload_format = "public_pr75_qzs3_qp1_segactions_p3"
        boundary_authority = "self_describing_p3_header"
        model_action = (int(model_len), int(actions_len))
        dict_len = 0
    elif payload.startswith(b"P4"):
        mask_len, model_len, dict_len, actions_len = struct.unpack_from("<IHHH", payload, 2)
        header_size = 2 + struct.calcsize("<IHHH")
        cursor = header_size
        header = {
            "kind": "P4",
            "bytes": header_size,
            "mask_br_bytes": int(mask_len),
            "renderer_br_bytes": int(model_len),
            "dict_br_bytes": int(dict_len),
            "actions_br_bytes": int(actions_len),
        }
        payload_format = "public_pr75_qzs3_qp1_segactions_p4_custom_dict"
        boundary_authority = "self_describing_p4_header"
        model_action = (int(model_len), int(actions_len))
    elif payload.startswith(b"P5"):
        mask_len, model_len, dict_len, actions_len, record_count = struct.unpack_from(
            "<IHHHH", payload, 2
        )
        header_size = 2 + struct.calcsize("<IHHHH")
        cursor = header_size
        header = {
            "kind": "P5",
            "bytes": header_size,
            "mask_br_bytes": int(mask_len),
            "renderer_br_bytes": int(model_len),
            "dict_br_bytes": int(dict_len),
            "actions_br_bytes": int(actions_len),
            "record_count": int(record_count),
        }
        payload_format = "public_pr75_qzs3_qp1_segactions_p5_packed_custom_dict"
        boundary_authority = "self_describing_p5_header"
        model_action = (int(model_len), int(actions_len))
    elif payload.startswith(b"P6"):
        mask_len, model_len, actions_len, record_count = struct.unpack_from("<IHHH", payload, 2)
        header_size = 2 + struct.calcsize("<IHHH")
        cursor = header_size
        header = {
            "kind": "P6",
            "bytes": header_size,
            "mask_br_bytes": int(mask_len),
            "renderer_br_bytes": int(model_len),
            "actions_br_bytes": int(actions_len),
            "record_count": int(record_count),
        }
        payload_format = "public_pr75_qzs3_qp1_segactions_p6_delta_varint"
        boundary_authority = "self_describing_p6_header"
        model_action = (int(model_len), int(actions_len))
        dict_len = 0
    else:
        mask_len = int(getattr(unpacker, "PUBLIC_PR75_MASK_LEN", MASK_BR_LEN))
        dict_len = 0
        exact = [
            (model_len, actions_len)
            for total, model_len, actions_len in getattr(
                unpacker,
                "PUBLIC_PR75_FIXED_SLICE_VARIANTS",
                (),
            )
            if int(total) == len(payload)
        ]
        if not exact:
            raise ValueError(f"unsupported fixed-slice payload length {len(payload)}")
        model_action = (int(exact[0][0]), int(exact[0][1]))

    model_len, actions_len = model_action
    mask_start = cursor
    mask_end = mask_start + int(mask_len)
    model_end = mask_end + model_len
    dict_end = model_end + int(dict_len)
    actions_end = dict_end + actions_len
    if actions_end >= len(payload):
        raise ValueError("payload slices overrun or leave no pose stream")

    add_segment(
        segments,
        name="masks.mkv",
        offset=mask_start,
        data=payload[mask_start:mask_end],
        codec="brotli_av1_obu",
    )
    add_segment(
        segments,
        name="renderer.bin",
        offset=mask_end,
        data=payload[mask_end:model_end],
        codec="brotli_qzs3",
    )
    if dict_len:
        add_segment(
            segments,
            name="seg_tile_action_dict.bin",
            offset=model_end,
            data=payload[model_end:dict_end],
            codec="brotli_seg_tile_action_dict_v1",
        )
    action_codec = (
        "brotli_seg_tile_actions_delta_varint_v1"
        if payload.startswith(b"P6")
        else "brotli_public_seg_tile_actions"
    )
    add_segment(
        segments,
        name="seg_tile_actions.bin",
        offset=dict_end,
        data=payload[dict_end:actions_end],
        codec=action_codec,
    )
    add_segment(
        segments,
        name="optimized_poses.qp1",
        offset=actions_end,
        data=payload[actions_end:],
        codec="public_qp1_brotli",
    )

    plain_segments = []
    for segment in segments:
        plain = dict(segment)
        plain.pop("raw_bytes_ref")
        plain_segments.append(plain)
    return {
        "payload_format": payload_format,
        "boundary_authority": boundary_authority,
        "header": header,
        "segments": segments,
        "segments_public": plain_segments,
    }


def _entropy(data: bytes) -> dict[str, Any]:
    if not data:
        return {
            "bytes": 0,
            "unique_byte_count": 0,
            "entropy_bits_per_byte": 0.0,
            "zero_order_entropy_bytes": 0,
            "zero_order_entropy_ratio": 0.0,
        }
    counts = Counter(data)
    n = len(data)
    entropy_bits = -sum((count / n) * math.log2(count / n) for count in counts.values())
    zero_order_bytes = math.ceil(entropy_bits * n / 8.0)
    return {
        "bytes": int(n),
        "unique_byte_count": int(len(counts)),
        "entropy_bits_per_byte": entropy_bits,
        "zero_order_entropy_bytes": int(zero_order_bytes),
        "zero_order_entropy_ratio": zero_order_bytes / n,
    }


def _compression_probe(data: bytes) -> dict[str, Any]:
    probes = {
        "input_bytes": len(data),
        "brotli_q5_bytes": len(brotli.compress(data, quality=5)),
        "brotli_q9_bytes": len(brotli.compress(data, quality=9)),
        "brotli_q11_bytes": len(brotli.compress(data, quality=11)),
        "zlib_9_bytes": len(zlib.compress(data, level=9)),
        "lzma_preset9_bytes": len(lzma.compress(data, preset=9 | lzma.PRESET_EXTREME)),
    }
    best_name, best_bytes = min(probes.items(), key=lambda item: (item[1], item[0]))
    return {
        **{key: int(value) for key, value in probes.items()},
        "best_probe": {"codec": best_name, "bytes": int(best_bytes)},
        "best_probe_delta_vs_input": int(best_bytes) - len(data),
    }


def _brotli_reencode_probe(decoded: bytes, charged_bytes: int) -> dict[str, Any]:
    candidates = {
        "brotli_q5": brotli.compress(decoded, quality=5),
        "brotli_q9": brotli.compress(decoded, quality=9),
        "brotli_q11": brotli.compress(decoded, quality=11),
    }
    best_name, best = min(candidates.items(), key=lambda item: (len(item[1]), item[0]))
    return {
        "current_charged_bytes": int(charged_bytes),
        "brotli_q5_bytes": int(len(candidates["brotli_q5"])),
        "brotli_q9_bytes": int(len(candidates["brotli_q9"])),
        "brotli_q11_bytes": int(len(candidates["brotli_q11"])),
        "best_reencode": {"codec": best_name, "bytes": int(len(best))},
        "best_reencode_delta_vs_current": int(len(best)) - int(charged_bytes),
        "semantics_preserving_if_runtime_accepts_brotli_stream": True,
    }


def _decode_qp1_words(raw: bytes) -> list[int]:
    if len(raw) < 5 or raw[:3] != b"QP1":
        raise ValueError(f"bad QP1 stream prefix: {raw[:3]!r}")
    values = [int.from_bytes(raw[3:5], "little")]
    cursor = 5
    while cursor < len(raw):
        acc = 0
        shift = 0
        while True:
            if cursor >= len(raw):
                raise ValueError("truncated QP1 VLQ payload")
            byte = raw[cursor]
            cursor += 1
            acc |= (byte & 0x7F) << shift
            if byte < 0x80:
                break
            shift += 7
            if shift > 63:
                raise ValueError("overlong QP1 VLQ payload")
        delta = (acc >> 1) ^ -(acc & 1)
        values.append((values[-1] + delta) & 0xFFFF)
    return values


def _decode_runtime_raw4_records(raw: bytes) -> list[tuple[int, int, int]]:
    if len(raw) % 4:
        raise ValueError(f"runtime action stream length {len(raw)} is not divisible by 4")
    out: list[tuple[int, int, int]] = []
    for offset in range(0, len(raw), 4):
        out.append((int.from_bytes(raw[offset : offset + 2], "little"), raw[offset + 2], raw[offset + 3]))
    return out


def _summarize_qp1_words(words: list[int], raw: bytes) -> dict[str, Any]:
    return {
        "codec": "QP1_col0_delta_varint",
        "raw_bytes": int(len(raw)),
        "raw_sha256": _sha256_bytes(raw),
        "row_count": int(len(words)),
        "q0_min": int(min(words)),
        "q0_max": int(max(words)),
        "q0_first": int(words[0]),
        "q0_last": int(words[-1]),
        "raw_prefix_hex": raw[:8].hex(),
    }


def _top_counts(counter: Counter[Any], n: int = 12) -> list[dict[str, Any]]:
    return [{"value": key, "count": int(count)} for key, count in counter.most_common(n)]


def _summarize_records(records: list[tuple[int, int, int]], runtime_raw: bytes) -> dict[str, Any]:
    pair_counts = Counter(pair for pair, _tile, _action in records)
    tile_counts = Counter(tile for _pair, tile, _action in records)
    action_counts = Counter(action for _pair, _tile, action in records)
    pairs = sorted(pair_counts)
    return {
        "record_count": int(len(records)),
        "runtime_record_bytes": int(len(runtime_raw)),
        "runtime_record_sha256": _sha256_bytes(runtime_raw),
        "unique_pair_count": int(len(pair_counts)),
        "pair_min": int(min(pairs)) if pairs else None,
        "pair_max": int(max(pairs)) if pairs else None,
        "unique_tile_count": int(len(tile_counts)),
        "unique_action_count": int(len(action_counts)),
        "top_pairs": _top_counts(pair_counts),
        "top_tiles": _top_counts(tile_counts),
        "top_actions": _top_counts(action_counts),
        "first_records": [
            {"pair": int(pair), "tile": int(tile), "action": int(action)}
            for pair, tile, action in records[:12]
        ],
        "last_records": [
            {"pair": int(pair), "tile": int(tile), "action": int(action)}
            for pair, tile, action in records[-12:]
        ],
    }


def analyze_archive(spec: ArchiveSpec, *, unpacker: Any, profiler: Any) -> dict[str, Any]:
    payload, zip_meta = read_single_payload_zip(spec.path)
    layout = split_payload_layout(payload, unpacker)
    robust_header, robust_members = unpacker._parse_payload(payload)  # noqa: SLF001
    segment_map = {segment["name"]: segment for segment in layout["segments"]}

    decoded_streams: dict[str, Any] = {}
    raw_decoded: dict[str, bytes] = {}
    for name in ("masks.mkv", "renderer.bin", "seg_tile_actions.bin", "optimized_poses.qp1"):
        segment = segment_map.get(name)
        if segment is None:
            continue
        decoded = brotli.decompress(segment["raw_bytes_ref"])
        raw_decoded[name] = decoded
        decoded_streams[name] = {
            "name": name,
            "charged_bytes": int(segment["charged_bytes"]),
            "charged_sha256": segment["charged_sha256"],
            "codec": segment["codec"],
            "decoded_bytes": int(len(decoded)),
            "decoded_sha256": _sha256_bytes(decoded),
            "decoded_prefix_hex": decoded[:8].hex(),
            "charged_entropy": _entropy(segment["raw_bytes_ref"]),
            "charged_recompression_probe": _compression_probe(segment["raw_bytes_ref"]),
            "decoded_brotli_reencode_probe": _brotli_reencode_probe(decoded, int(segment["charged_bytes"])),
        }

    action_wire_kind, wire_records = profiler.decode_seg_tile_actions_raw(
        raw_decoded["seg_tile_actions.bin"]
    )
    runtime_actions = robust_members["seg_tile_actions.bin"]
    runtime_records = _decode_runtime_raw4_records(runtime_actions)
    runtime_from_wire = profiler.encode_runtime_action_records(wire_records)
    if runtime_from_wire != runtime_actions:
        raise ValueError(f"{spec.label}: wire-decoded records do not match robust runtime actions")

    pose_raw = raw_decoded["optimized_poses.qp1"]
    pose_words = _decode_qp1_words(pose_raw)
    robust_members_public = {
        name: {
            "decoded_bytes": int(len(data)),
            "decoded_sha256": _sha256_bytes(data),
            "decoded_prefix_hex": data[:8].hex(),
        }
        for name, data in robust_members.items()
    }

    stripped_p3_parse: dict[str, Any] | None = None
    if layout["header"] and layout["header"].get("kind") == "P3":
        stripped = payload[int(layout["header"]["bytes"]) :]
        try:
            stripped_header, stripped_members = unpacker._parse_payload(stripped)  # noqa: SLF001
            stripped_p3_parse = {
                "ok": True,
                "stripped_payload_bytes": int(len(stripped)),
                "stripped_payload_sha256": _sha256_bytes(stripped),
                "payload_format": stripped_header.get("payload_format"),
                "decoded_sha256_equal": {
                    name: _sha256_bytes(robust_members[name]) == _sha256_bytes(stripped_members[name])
                    for name in robust_members
                    if name in stripped_members
                },
            }
        except Exception as exc:
            stripped_p3_parse = {
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }

    plain_layout = {
        "payload_format": layout["payload_format"],
        "boundary_authority": layout["boundary_authority"],
        "header": layout["header"],
        "segments": layout["segments_public"],
        "stripped_p3_fixed_slice_parse": stripped_p3_parse,
    }
    return {
        "label": spec.label,
        "role": spec.role,
        "archive": {
            "path": str(spec.path),
            "bytes": int(spec.path.stat().st_size),
            "sha256": _sha256_file(spec.path),
            "zip": zip_meta,
        },
        "payload": {
            "member": "p",
            "bytes": int(len(payload)),
            "sha256": _sha256_bytes(payload),
            "prefix_hex": payload[:8].hex(),
            "entropy": _entropy(payload),
            "compression_probe": _compression_probe(payload),
        },
        "layout": plain_layout,
        "decoded_streams": decoded_streams,
        "robust_current_parse": {
            "ok": True,
            "payload_format": robust_header.get("payload_format"),
            "members": robust_header.get("members", []),
            "decoded_members": robust_members_public,
        },
        "actions": {
            "wire_kind": action_wire_kind,
            "wire_raw_bytes": int(len(raw_decoded["seg_tile_actions.bin"])),
            "wire_raw_sha256": _sha256_bytes(raw_decoded["seg_tile_actions.bin"]),
            **_summarize_records(runtime_records, runtime_actions),
        },
        "pose": _summarize_qp1_words(pose_words, pose_raw),
        "_action_records_ref": runtime_records,
        "_pose_words_ref": pose_words,
        "_pose_raw_ref": pose_raw,
        "_payload_ref": payload,
    }


def identity_matrix(archives: list[dict[str, Any]], stream_name: str, key: str) -> list[dict[str, Any]]:
    rows = []
    for archive in archives:
        stream = archive["decoded_streams"].get(stream_name)
        rows.append(
            {
                "archive": archive["label"],
                "stream": stream_name,
                "bytes": stream.get("decoded_bytes" if key == "decoded_sha256" else "charged_bytes") if stream else None,
                "sha256": stream.get(key) if stream else None,
            }
        )
    return rows


def pairwise_action_diffs(archives: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    counters = {
        archive["label"]: Counter(archive["_action_records_ref"])
        for archive in archives
    }
    labels = [archive["label"] for archive in archives]
    for i, left in enumerate(labels):
        for right in labels[i + 1 :]:
            l_counter = counters[left]
            r_counter = counters[right]
            common = l_counter & r_counter
            only_l = l_counter - r_counter
            only_r = r_counter - l_counter
            out[f"{left}_vs_{right}"] = {
                "left_record_count": int(sum(l_counter.values())),
                "right_record_count": int(sum(r_counter.values())),
                "common_record_count": int(sum(common.values())),
                "left_only_record_count": int(sum(only_l.values())),
                "right_only_record_count": int(sum(only_r.values())),
                "left_sequence_sha256": archives[labels.index(left)]["actions"]["runtime_record_sha256"],
                "right_sequence_sha256": archives[labels.index(right)]["actions"]["runtime_record_sha256"],
                "sequence_equal": archives[labels.index(left)]["actions"]["runtime_record_sha256"]
                == archives[labels.index(right)]["actions"]["runtime_record_sha256"],
            }
    return out


def pose_diffs(archives: list[dict[str, Any]]) -> dict[str, Any]:
    labels = [archive["label"] for archive in archives]
    base = archives[0]
    out: dict[str, Any] = {
        "base": base["label"],
        "raw_streams": {},
        "word_diffs_vs_base": {},
    }
    base_words = base["_pose_words_ref"]
    base_raw = base["_pose_raw_ref"]
    for archive in archives:
        raw = archive["_pose_raw_ref"]
        words = archive["_pose_words_ref"]
        byte_diff_positions = [
            idx
            for idx in range(max(len(base_raw), len(raw)))
            if (base_raw[idx] if idx < len(base_raw) else None)
            != (raw[idx] if idx < len(raw) else None)
        ]
        word_diff_rows = [
            idx
            for idx in range(max(len(base_words), len(words)))
            if (base_words[idx] if idx < len(base_words) else None)
            != (words[idx] if idx < len(words) else None)
        ]
        out["raw_streams"][archive["label"]] = {
            "raw_bytes": int(len(raw)),
            "raw_sha256": _sha256_bytes(raw),
            "byte_diff_count_vs_base": int(len(byte_diff_positions)),
            "first_byte_diff_positions_vs_base": byte_diff_positions[:20],
        }
        out["word_diffs_vs_base"][archive["label"]] = {
            "row_count": int(len(words)),
            "q0_word_diff_count": int(len(word_diff_rows)),
            "first_q0_diff_rows": word_diff_rows[:20],
        }
    out["all_public_pose_streams_equal"] = len({_sha256_bytes(a["_pose_raw_ref"]) for a in archives[1:]}) == 1
    out["all_streams_equal"] = len({_sha256_bytes(a["_pose_raw_ref"]) for a in archives}) == 1
    out["labels"] = labels
    return out


def load_c102_eval(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    score = float(data["canonical_score"])
    archive_bytes = int(data["archive_size_bytes"])
    gap = score - TARGET_SCORE
    return {
        "path": str(path),
        "canonical_score": score,
        "target_score": TARGET_SCORE,
        "score_gap_to_target": gap,
        "archive_size_bytes": archive_bytes,
        "archive_sha256": data.get("provenance", {}).get("archive_sha256"),
        "avg_segnet_dist": float(data["avg_segnet_dist"]),
        "avg_posenet_dist": float(data["avg_posenet_dist"]),
        "score_seg_contribution": float(data["score_seg_contribution"]),
        "score_pose_contribution": float(data["score_pose_contribution"]),
        "score_rate_contribution": float(data["score_rate_contribution"]),
        "n_samples": int(data["n_samples"]),
        "byte_savings_needed_if_components_unchanged": int(math.ceil(gap / RATE_LAMBDA)),
        "rate_lambda": RATE_LAMBDA,
    }


def _break_even(eval_summary: dict[str, Any], byte_delta: int) -> dict[str, Any]:
    needed_gain = max(0.0, float(eval_summary["score_gap_to_target"]) + byte_delta * RATE_LAMBDA)
    pose_contrib = float(eval_summary["score_pose_contribution"])
    pose_dist = float(eval_summary["avg_posenet_dist"])
    pose_contrib_target = max(0.0, pose_contrib - needed_gain)
    pose_dist_target = (pose_contrib_target * pose_contrib_target) / 10.0
    return {
        "archive_byte_delta_vs_c102": int(byte_delta),
        "rate_score_delta": byte_delta * RATE_LAMBDA,
        "required_total_component_score_gain_to_reach_0_31": needed_gain,
        "seg_only_dist_reduction_required": needed_gain / 100.0,
        "pose_only_posenet_dist_ceiling": pose_dist_target,
        "pose_only_posenet_dist_reduction_required": max(0.0, pose_dist - pose_dist_target),
    }


def build_opportunities(archives: list[dict[str, Any]], eval_summary: dict[str, Any]) -> dict[str, Any]:
    by_label = {archive["label"]: archive for archive in archives}
    c102 = by_label["c102"]
    c102_action_bytes = c102["decoded_streams"]["seg_tile_actions.bin"]["charged_bytes"]
    c102_pose_bytes = c102["decoded_streams"]["optimized_poses.qp1"]["charged_bytes"]
    lossless: list[dict[str, Any]] = []
    stripped = c102["layout"].get("stripped_p3_fixed_slice_parse")
    if stripped and stripped.get("ok") and all(stripped.get("decoded_sha256_equal", {}).values()):
        savings = int(c102["layout"]["header"]["bytes"])
        lossless.append(
            {
                "rank": 1,
                "name": "strip_c102_p3_header_use_existing_fixed_slice_parser",
                "class": "lossless_semantics_preserving",
                "byte_savings": savings,
                "score_delta_if_components_unchanged": -savings * RATE_LAMBDA,
                "new_archive_bytes_estimate": int(c102["archive"]["bytes"]) - savings,
                "target_0_31_status": "insufficient_alone",
                "break_even": _break_even(eval_summary, -savings),
                "evidence": stripped,
                "implementation_note": "Remove the 10-byte P3 header and store the same mask/model/actions/QP1 slices; robust_current fixed-slice fallback parses the stripped payload to identical decoded members.",
            }
        )

    for stream_name, stream in c102["decoded_streams"].items():
        delta = int(stream["decoded_brotli_reencode_probe"]["best_reencode_delta_vs_current"])
        if delta < 0:
            savings = -delta
            lossless.append(
                {
                    "rank": len(lossless) + 1,
                    "name": f"rebrotli_c102_{stream_name}",
                    "class": "lossless_semantics_preserving",
                    "byte_savings": savings,
                    "score_delta_if_components_unchanged": -savings * RATE_LAMBDA,
                    "new_archive_bytes_estimate": int(c102["archive"]["bytes"]) - savings,
                    "target_0_31_status": "insufficient_alone",
                    "break_even": _break_even(eval_summary, -savings),
                    "evidence": stream["decoded_brotli_reencode_probe"],
                    "implementation_note": "Replace only the Brotli stream with one that decompresses to identical decoded bytes.",
                }
            )
    lossless.sort(key=lambda item: (-int(item["byte_savings"]), item["name"]))
    for idx, item in enumerate(lossless, start=1):
        item["rank"] = idx

    risky: list[dict[str, Any]] = []
    for label in ("pr77", "pr79"):
        action_delta = (
            int(by_label[label]["decoded_streams"]["seg_tile_actions.bin"]["charged_bytes"])
            - int(c102_action_bytes)
        )
        risky.append(
            {
                "name": f"transplant_{label}_action_records",
                "class": "risky_component_affecting",
                "archive_byte_delta_vs_c102": action_delta,
                "record_count": by_label[label]["actions"]["record_count"],
                "extra_records_vs_c102_multiset": pairwise_action_diffs([c102, by_label[label]])[
                    f"c102_vs_{label}"
                ]["right_only_record_count"],
                "break_even": _break_even(eval_summary, action_delta),
                "evidence": {
                    "c102_action_sha256": c102["actions"]["runtime_record_sha256"],
                    f"{label}_action_sha256": by_label[label]["actions"]["runtime_record_sha256"],
                },
                "dispatch_status": "do_not_dispatch_from_byte_forensics",
            }
        )
    public_pose_delta = (
        int(by_label["pr75"]["decoded_streams"]["optimized_poses.qp1"]["charged_bytes"])
        - int(c102_pose_bytes)
    )
    risky.append(
        {
            "name": "replace_c102_qp1_pose_with_public_pr75_pr77_pr79_pose",
            "class": "risky_component_affecting",
            "archive_byte_delta_vs_c102": public_pose_delta,
            "break_even": _break_even(eval_summary, public_pose_delta),
            "evidence": {
                "c102_pose_sha256": c102["pose"]["raw_sha256"],
                "public_pose_sha256": by_label["pr75"]["pose"]["raw_sha256"],
                "public_pose_streams_equal": True,
            },
            "dispatch_status": "do_not_dispatch_from_byte_forensics",
        }
    )
    risky.append(
        {
            "name": "remove_c102_seg_tile_actions_stream",
            "class": "risky_component_affecting",
            "archive_byte_delta_vs_c102": -int(c102_action_bytes),
            "break_even": _break_even(eval_summary, -int(c102_action_bytes)),
            "evidence": {
                "removed_record_count": c102["actions"]["record_count"],
                "removed_action_charged_bytes": int(c102_action_bytes),
            },
            "dispatch_status": "not_recommended_without_component_trace",
        }
    )
    risky.sort(
        key=lambda item: (
            item["break_even"]["required_total_component_score_gain_to_reach_0_31"],
            item["archive_byte_delta_vs_c102"],
            item["name"],
        )
    )
    return {
        "lossless_semantics_preserving_ranked": lossless,
        "risky_component_affecting_ranked_by_break_even": risky,
        "unchanged_component_byte_target": {
            "byte_savings_needed": eval_summary["byte_savings_needed_if_components_unchanged"],
            "best_lossless_savings_found": max((int(item["byte_savings"]) for item in lossless), default=0),
            "lossless_path_to_0_31_found": max((int(item["byte_savings"]) for item in lossless), default=0)
            >= int(eval_summary["byte_savings_needed_if_components_unchanged"]),
        },
    }


def strip_internal_refs(archive: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in archive.items() if not key.endswith("_ref")}


def write_action_csvs(out_dir: Path, archives: list[dict[str, Any]]) -> None:
    for archive in archives:
        path = out_dir / f"action_records_{archive['label']}.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["index", "pair", "tile", "action"])
            for idx, (pair, tile, action) in enumerate(archive["_action_records_ref"]):
                writer.writerow([idx, pair, tile, action])

    counters = {
        archive["label"]: Counter(archive["_action_records_ref"])
        for archive in archives
    }
    all_records = sorted(set().union(*(set(counter) for counter in counters.values())))
    with (out_dir / "action_record_multiset_union_diff.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.writer(handle)
        labels = [archive["label"] for archive in archives]
        writer.writerow(["pair", "tile", "action", *[f"count_{label}" for label in labels]])
        for record in all_records:
            writer.writerow([*record, *[counters[label][record] for label in labels]])


def write_pose_csvs(out_dir: Path, archives: list[dict[str, Any]]) -> None:
    labels = [archive["label"] for archive in archives]
    max_rows = max(len(archive["_pose_words_ref"]) for archive in archives)
    with (out_dir / "pose_qp1_q0_word_diff.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row", *labels, "all_equal"])
        for row in range(max_rows):
            values = [
                archive["_pose_words_ref"][row] if row < len(archive["_pose_words_ref"]) else None
                for archive in archives
            ]
            writer.writerow([row, *values, len(set(values)) == 1])

    base = archives[0]
    base_raw = base["_pose_raw_ref"]
    with (out_dir / "pose_qp1_byte_diff_vs_c102.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["archive", "byte_offset", "c102_byte", "other_byte"])
        for archive in archives[1:]:
            raw = archive["_pose_raw_ref"]
            for idx in range(max(len(base_raw), len(raw))):
                left = base_raw[idx] if idx < len(base_raw) else None
                right = raw[idx] if idx < len(raw) else None
                if left != right:
                    writer.writerow([archive["label"], idx, left, right])


def write_identity_csv(out_dir: Path, archives: list[dict[str, Any]]) -> None:
    with (out_dir / "stream_identity_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "archive",
                "stream",
                "charged_bytes",
                "charged_sha256",
                "decoded_bytes",
                "decoded_sha256",
            ]
        )
        for archive in archives:
            for name, stream in archive["decoded_streams"].items():
                writer.writerow(
                    [
                        archive["label"],
                        name,
                        stream["charged_bytes"],
                        stream["charged_sha256"],
                        stream["decoded_bytes"],
                        stream["decoded_sha256"],
                    ]
                )


def build_markdown(profile: dict[str, Any]) -> str:
    archives = profile["archives"]
    opportunities = profile["opportunities"]
    eval_summary = profile["c102_eval_summary"]
    lines: list[str] = []
    lines.append("# PR79 Archive Binary Forensics Worker")
    lines.append("")
    lines.append("Local byte/archive reverse-engineering only. No GPU dispatch and no score claim.")
    lines.append("")
    lines.append("## Archive And ZIP Overhead")
    lines.append("")
    lines.append("| archive | archive bytes | payload bytes | zip overhead | payload format | header bytes | sha256 |")
    lines.append("|---|---:|---:|---:|---|---:|---|")
    for archive in archives:
        header = archive["layout"]["header"]
        header_bytes = int(header["bytes"]) if header else 0
        lines.append(
            f"| {archive['label']} | {archive['archive']['bytes']} | {archive['payload']['bytes']} | "
            f"{archive['archive']['zip']['overhead']['total_overhead_bytes']} | "
            f"{archive['layout']['payload_format']} | {header_bytes} | "
            f"`{archive['archive']['sha256']}` |"
        )
    lines.append("")
    lines.append("## Charged Slice Layout")
    lines.append("")
    lines.append("| archive | stream | offset | charged bytes | decoded bytes | charged sha equal group | decoded sha equal group |")
    lines.append("|---|---:|---:|---:|---:|---|---|")
    for archive in archives:
        for segment in archive["layout"]["segments"]:
            stream = archive["decoded_streams"][segment["name"]]
            charged_group = stream["charged_sha256"][:12]
            decoded_group = stream["decoded_sha256"][:12]
            lines.append(
                f"| {archive['label']} | {segment['name']} | {segment['offset']} | "
                f"{stream['charged_bytes']} | {stream['decoded_bytes']} | `{charged_group}` | `{decoded_group}` |"
            )
    lines.append("")
    lines.append("## Action Records")
    lines.append("")
    lines.append("| archive | wire kind | records | runtime bytes | unique pairs | unique tiles | unique actions |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for archive in archives:
        actions = archive["actions"]
        lines.append(
            f"| {archive['label']} | {actions['wire_kind']} | {actions['record_count']} | "
            f"{actions['runtime_record_bytes']} | {actions['unique_pair_count']} | "
            f"{actions['unique_tile_count']} | {actions['unique_action_count']} |"
        )
    lines.append("")
    lines.append("## QP1 Pose Diff")
    lines.append("")
    lines.append("| archive | raw bytes | q0 row diffs vs C-102 | byte diffs vs C-102 | sha256 |")
    lines.append("|---|---:|---:|---:|---|")
    pose = profile["pose_diffs"]
    for archive in archives:
        label = archive["label"]
        lines.append(
            f"| {label} | {pose['raw_streams'][label]['raw_bytes']} | "
            f"{pose['word_diffs_vs_base'][label]['q0_word_diff_count']} | "
            f"{pose['raw_streams'][label]['byte_diff_count_vs_base']} | "
            f"`{pose['raw_streams'][label]['raw_sha256']}` |"
        )
    lines.append("")
    lines.append("## Break-Even To 0.31")
    lines.append("")
    lines.append(
        f"C-102 canonical score input: {eval_summary['canonical_score']:.15f}; "
        f"gap to {eval_summary['target_score']:.2f}: {eval_summary['score_gap_to_target']:.15f}. "
        f"With components unchanged, required archive-byte savings: "
        f"{eval_summary['byte_savings_needed_if_components_unchanged']} bytes."
    )
    lines.append("")
    lines.append("## Lossless/Semantics-Preserving Opportunities")
    lines.append("")
    lines.append("| rank | opportunity | byte savings | new archive bytes | target status |")
    lines.append("|---:|---|---:|---:|---|")
    for item in opportunities["lossless_semantics_preserving_ranked"]:
        lines.append(
            f"| {item['rank']} | {item['name']} | {item['byte_savings']} | "
            f"{item['new_archive_bytes_estimate']} | {item['target_0_31_status']} |"
        )
    if not opportunities["lossless_semantics_preserving_ranked"]:
        lines.append("| - | none found | 0 | - | insufficient |")
    lines.append("")
    lines.append("## Risky Component-Affecting Transforms")
    lines.append("")
    lines.append("| transform | byte delta | required score gain | seg-only dist reduction | pose-only dist reduction |")
    lines.append("|---|---:|---:|---:|---:|")
    for item in opportunities["risky_component_affecting_ranked_by_break_even"]:
        be = item["break_even"]
        lines.append(
            f"| {item['name']} | {be['archive_byte_delta_vs_c102']} | "
            f"{be['required_total_component_score_gain_to_reach_0_31']:.9f} | "
            f"{be['seg_only_dist_reduction_required']:.9f} | "
            f"{be['pose_only_posenet_dist_reduction_required']:.9f} |"
        )
    lines.append("")
    lines.append("CSV companions: `action_record_multiset_union_diff.csv`, `pose_qp1_q0_word_diff.csv`, `pose_qp1_byte_diff_vs_c102.csv`, and `stream_identity_matrix.csv`.")
    lines.append("")
    return "\n".join(lines)


def write_outputs(profile: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    archives_with_refs = profile.pop("_archives_with_refs")
    write_action_csvs(out_dir, archives_with_refs)
    write_pose_csvs(out_dir, archives_with_refs)
    write_identity_csv(out_dir, archives_with_refs)
    (out_dir / "family_profile.json").write_bytes(_json_bytes(profile))
    (out_dir / "family_profile.md").write_text(build_markdown(profile), encoding="utf-8")


def build_profile(
    *,
    archive_specs: tuple[ArchiveSpec, ...],
    unpacker_path: Path,
    profiler_path: Path,
    c102_eval_path: Path,
) -> dict[str, Any]:
    unpacker = _load_module(unpacker_path, "public_qpose_family_unpacker")
    profiler = _load_module(profiler_path, "public_qpose_family_pr75_profiler")
    archives_with_refs = [
        analyze_archive(spec, unpacker=unpacker, profiler=profiler)
        for spec in archive_specs
    ]
    archives = [strip_internal_refs(archive) for archive in archives_with_refs]
    eval_summary = load_c102_eval(c102_eval_path)
    profile: dict[str, Any] = {
        "schema": SCHEMA,
        "tool": TOOL,
        "score_claim": False,
        "evidence_grade": "empirical_reverse_engineering_forensics",
        "notes": [
            "Local byte/runtime grammar profile only; exact CUDA auth eval remains score truth.",
            "Public fixed-slice ZIP permissiveness is studied as forensics, not used as score evidence.",
            "No GPU dispatch was performed by this tool.",
        ],
        "contest_formula_rate_term": {
            "archive_rate_contribution": "25 * archive_bytes / 37,545,489",
            "original_video_bytes": CONTEST_ORIGINAL_BYTES,
            "rate_lambda": RATE_LAMBDA,
        },
        "c102_eval_summary": eval_summary,
        "archives": archives,
        "identity": {
            "mask_charged": identity_matrix(archives, "masks.mkv", "charged_sha256"),
            "mask_decoded": identity_matrix(archives, "masks.mkv", "decoded_sha256"),
            "renderer_charged": identity_matrix(archives, "renderer.bin", "charged_sha256"),
            "renderer_decoded": identity_matrix(archives, "renderer.bin", "decoded_sha256"),
            "pose_charged": identity_matrix(archives, "optimized_poses.qp1", "charged_sha256"),
            "pose_decoded": identity_matrix(archives, "optimized_poses.qp1", "decoded_sha256"),
        },
        "action_pairwise_diffs": pairwise_action_diffs(archives_with_refs),
        "pose_diffs": pose_diffs(archives_with_refs),
        "opportunities": build_opportunities(archives_with_refs, eval_summary),
        "_archives_with_refs": archives_with_refs,
    }
    return profile


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--robust-unpacker", type=Path, default=DEFAULT_UNPACKER)
    parser.add_argument("--pr75-profiler", type=Path, default=DEFAULT_PR75_PROFILER)
    parser.add_argument("--c102-eval-json", type=Path, default=DEFAULT_C102_EVAL)
    parser.add_argument(
        "--archive",
        action="append",
        default=None,
        metavar="LABEL=PATH",
        help="Override archive list. Repeat as needed; roles are set to custom.",
    )
    return parser


def parse_archive_specs(values: list[str] | None) -> tuple[ArchiveSpec, ...]:
    if not values:
        return DEFAULT_ARCHIVES
    specs = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"--archive must be LABEL=PATH, got {value!r}")
        label, raw_path = value.split("=", 1)
        if not label:
            raise ValueError(f"archive label must be nonempty: {value!r}")
        specs.append(ArchiveSpec(label, Path(raw_path), "custom"))
    return tuple(specs)


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    profile = build_profile(
        archive_specs=parse_archive_specs(args.archive),
        unpacker_path=args.robust_unpacker,
        profiler_path=args.pr75_profiler,
        c102_eval_path=args.c102_eval_json,
    )
    write_outputs(profile, args.output_dir)
    print(json.dumps({k: v for k, v in profile.items() if k != "_archives_with_refs"}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
