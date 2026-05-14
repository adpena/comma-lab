#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Plan local PR65/Henosis stream-transfer candidates on the C091 frontier.

This tool is intentionally local-only.  It verifies strict ZIP custody for the
PR65/Henosis compact archive and C091/C089 PR75-style archives, extracts PR65
pose and postprocess streams as component-basin signal, and emits deterministic
candidate archives plus manifests.  It never dispatches GPU work and never
edits Lightning or dispatch state.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import struct
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import brotli
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tac.qp1_pose_codec import decode_qp1, encode_qp1


TOOL = "experiments/plan_pr65_henosis_stream_transfer.py"
SCHEMA_VERSION = 1
ORIGINAL_VIDEO_BYTES = 37_545_489
RATE_SCORE_PER_BYTE = 25.0 / ORIGINAL_VIDEO_BYTES
FIXED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
QPOST_BUILDER_PATH = REPO_ROOT / "experiments/build_qzs3_postprocess_candidate.py"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "experiments/results/pr65_henosis_stream_transfer_20260503_codex"
DEFAULT_PR65_ARCHIVE = (
    REPO_ROOT
    / "experiments/results/top_submission_delta_reverse_engineering_20260503/"
    "sources/pr65_henosis_archive.zip"
)
DEFAULT_C091_ARCHIVE = (
    REPO_ROOT
    / "experiments/results/lightning_batch/"
    "exact_eval_pr75_minp_public_replay_t4_20260503T1049Z/archive.zip"
)
DEFAULT_C089_ARCHIVE = (
    REPO_ROOT
    / "experiments/results/lightning_batch/"
    "exact_eval_c067_pr75_qp1_top40_p6_t4_awsfix1_20260503T0630Z/archive.zip"
)
DEFAULT_PR65_TRACE = (
    REPO_ROOT
    / "experiments/results/vast_harvest/public_external_component_trace_20260502T0642Z/"
    "pr65_torch25_compat_adapter/component_trace.json"
)
DEFAULT_C091_TRACE = (
    REPO_ROOT
    / "experiments/results/lightning_batch/"
    "exact_eval_pr75_minp_public_replay_t4_20260503T1049Z/component_trace.json"
)
DEFAULT_C089_TRACE = (
    REPO_ROOT
    / "experiments/results/lightning_batch/"
    "exact_eval_c067_pr75_qp1_top40_p6_t4_awsfix1_20260503T0630Z/component_trace.json"
)
EXPECTED_PR65_SHA256 = "b331cb4f6df9d8929db966b943b8c73624cdf3b6db71acbde361570852e59e68"
EXPECTED_C091_SHA256 = "03a2afd5fe92c93a9b7b7e43625158a73b455f0cfbca82d278008a728db78746"
EXPECTED_C089_SHA256 = "0ec53e5b871149ed6eea56c0b9bcca3baec998d5bfad4f371979e0c90e62cea8"
C091_SCORE = 0.31516575028285976
C091_BYTES = 276_481
C091_POSE = 0.00049371
C091_SEG = 0.00060804
PR65_OFFICIAL_POSE = 0.00049291
PR65_OFFICIAL_SEG = 0.00060138
PR65_OFFICIAL_BYTES = 284_425
SUB314_TARGET = 0.314
PR75_MASK_LEN = 219_472
KNOWN_FIXED_PR75_VARIANTS: tuple[tuple[int, int, int, int], ...] = (
    (276_641, 56_034, 236, 0),
    (276_520, 55_914, 236, 0),
    (276_381, 55_756, 255, 0),
    (276_379, 55_756, 253, 0),
    (276_451, 55_756, 325, 0),
)
QPOST_STREAM_NAMES = (
    "post",
    "shift",
    "frac",
    "frac2",
    "frac3",
    "bias",
    "region",
    "randmulti",
)


class HenosisTransferError(ValueError):
    """Raised when PR65/Henosis stream-transfer planning fails a guard."""


@dataclass(frozen=True)
class ZipInventory:
    archive: Path
    archive_bytes: int
    archive_sha256: str
    members: dict[str, bytes]
    member_inventory: list[dict[str, Any]]


@dataclass(frozen=True)
class Pr75Archive:
    label: str
    path: Path
    archive_bytes: int
    archive_sha256: str
    payload: bytes
    payload_sha256: str
    payload_format: str
    encoded_streams: dict[str, bytes]
    decoded_members: dict[str, bytes]
    action_record_count: int
    zip_inventory: dict[str, Any]


@dataclass(frozen=True)
class QPostSpec:
    candidate_id: str
    source_archive_id: str
    include_streams: tuple[str, ...]
    rank_metric: str
    top_pairs: int
    risk_family: str


DEFAULT_QPOST_SPECS: tuple[QPostSpec, ...] = (
    QPostSpec(
        "c091_pr65_bias_segadv_top032",
        "c091",
        ("bias",),
        "seg",
        32,
        "bias_only_existing_c089_exact_negative",
    ),
    QPostSpec(
        "c091_pr65_bias_combined_top064",
        "c091",
        ("bias",),
        "combined",
        64,
        "bias_only_existing_c089_exact_negative_scale",
    ),
    QPostSpec(
        "c091_pr65_bias_region_segadv_top032",
        "c091",
        ("bias", "region"),
        "seg",
        32,
        "region_bias_high_risk_nonbias_stream",
    ),
    QPostSpec(
        "c091_pr65_post_bias_segadv_top016",
        "c091",
        ("post", "bias"),
        "seg",
        16,
        "post_bias_high_risk_nonbias_stream",
    ),
    QPostSpec(
        "c091_pr65_pose_qp1_c089_actions_p6_bias_segadv_top032",
        "c091_pr65_pose_qp1_c089_actions_p6",
        ("bias",),
        "seg",
        32,
        "pose_transfer_plus_bias_existing_c089_exact_negative",
    ),
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(payload))


def _safe_member_name(name: str) -> str:
    path = Path(name)
    hidden = name.startswith(".") or name.startswith("__MACOSX/") or "/." in name
    resource_fork = name.startswith("._") or "/._" in name
    if hidden or resource_fork:
        raise HenosisTransferError(f"hidden/system ZIP member is forbidden: {name!r}")
    if not name or name.startswith("/") or ".." in path.parts or len(path.parts) != 1:
        raise HenosisTransferError(f"unsafe ZIP member path: {name!r}")
    return name


def _local_header_name(path: Path, info: zipfile.ZipInfo) -> str:
    with path.open("rb") as handle:
        handle.seek(info.header_offset)
        header = handle.read(30)
        if len(header) != 30 or header[:4] != b"PK\x03\x04":
            raise HenosisTransferError(f"{path}: bad local file header at {info.header_offset}")
        name_len, extra_len = struct.unpack_from("<HH", header, 26)
        raw_name = handle.read(name_len)
        if len(raw_name) != name_len:
            raise HenosisTransferError(f"{path}: truncated local file name at {info.header_offset}")
        _ = handle.read(extra_len)
    encoding = "utf-8" if info.flag_bits & 0x800 else "cp437"
    return raw_name.decode(encoding)


def strict_zip_inventory(
    archive: Path,
    *,
    expected_members: Sequence[str] | None = None,
    expected_sha256: str | None = None,
) -> ZipInventory:
    archive = archive.resolve()
    if not archive.is_file():
        raise HenosisTransferError(f"missing archive: {archive}")
    archive_sha = _sha256_path(archive)
    if expected_sha256 and archive_sha != expected_sha256:
        raise HenosisTransferError(
            f"{archive}: SHA mismatch, expected {expected_sha256}, got {archive_sha}"
        )

    members: dict[str, bytes] = {}
    inventory: list[dict[str, Any]] = []
    with zipfile.ZipFile(archive, "r") as zf:
        for info in zf.infolist():
            if info.is_dir():
                raise HenosisTransferError(f"{archive}: directory entries are forbidden: {info.filename!r}")
            name = _safe_member_name(info.filename)
            if name in members:
                raise HenosisTransferError(f"{archive}: duplicate ZIP member: {name!r}")
            local_name = _local_header_name(archive, info)
            if local_name != name:
                raise HenosisTransferError(
                    f"{archive}: central/local name mismatch: {name!r} != {local_name!r}"
                )
            data = zf.read(info)
            members[name] = data
            inventory.append(
                {
                    "name": name,
                    "bytes": len(data),
                    "sha256": _sha256_bytes(data),
                    "compress_type": int(info.compress_type),
                    "compress_size": int(info.compress_size),
                    "file_size": int(info.file_size),
                    "date_time": list(info.date_time),
                    "external_attr": int(info.external_attr),
                    "local_header_name": local_name,
                    "local_central_name_match": True,
                }
            )
    names = list(members)
    if expected_members is not None and names != list(expected_members):
        raise HenosisTransferError(
            f"{archive}: expected ZIP members {list(expected_members)!r}, got {names!r}"
        )
    return ZipInventory(
        archive=archive,
        archive_bytes=archive.stat().st_size,
        archive_sha256=archive_sha,
        members=members,
        member_inventory=inventory,
    )


def _segment_summary(encoded: bytes) -> dict[str, Any]:
    try:
        decoded = brotli.decompress(encoded)
        codec = "brotli"
        decoded_error = None
    except brotli.error as exc:
        decoded = b""
        codec = "brotli_decode_error"
        decoded_error = str(exc)
    return {
        "bytes": len(encoded),
        "sha256": _sha256_bytes(encoded),
        "magic_hex": encoded[:12].hex(),
        "codec": codec,
        "decoded_bytes": len(decoded),
        "decoded_sha256": _sha256_bytes(decoded) if decoded else None,
        "decoded_magic_hex": decoded[:12].hex() if decoded else None,
        "decoded_error": decoded_error,
    }


def parse_pr65_henosis_archive(
    pr65_archive: Path,
    *,
    expected_sha256: str | None = EXPECTED_PR65_SHA256,
) -> dict[str, Any]:
    inventory = strict_zip_inventory(
        pr65_archive,
        expected_members=("x",),
        expected_sha256=expected_sha256,
    )
    raw = inventory.members["x"]
    if len(raw) < 30:
        raise HenosisTransferError("PR65 compact member is too short for v4 header")
    lengths = [int.from_bytes(raw[i : i + 3], "little") for i in range(0, 30, 3)]
    l_mask, l_model, l_pose, l_post, l_shift, l_frac, l_frac2, l_frac3, l_bias, l_region = lengths
    if not (l_mask > 1000 and l_model > 1000 and l_pose > 100):
        raise HenosisTransferError(f"implausible PR65 core lengths: {lengths[:3]}")
    if any(value <= 0 for value in (l_post, l_shift, l_frac, l_frac2, l_frac3, l_bias, l_region)):
        raise HenosisTransferError(f"PR65 qpost stream length must be positive: {lengths[3:]}")

    names = ("mask", "model", "pose", "post", "shift", "frac", "frac2", "frac3", "bias", "region")
    pos = 30
    segments: dict[str, bytes] = {}
    for name, n_bytes in zip(names, lengths):
        end = pos + n_bytes
        if end > len(raw):
            raise HenosisTransferError(f"PR65 segment {name!r} overruns x payload")
        segments[name] = raw[pos:end]
        pos = end
    if pos >= len(raw):
        raise HenosisTransferError("PR65 compact member is missing randmulti tail")
    segments["randmulti"] = raw[pos:]
    qpost_encoded_bytes = sum(len(segments[name]) for name in QPOST_STREAM_NAMES)
    core_encoded_bytes = sum(len(segments[name]) for name in ("mask", "model", "pose"))
    return {
        "archive": str(inventory.archive),
        "archive_bytes": inventory.archive_bytes,
        "archive_sha256": inventory.archive_sha256,
        "zip_inventory": {
            "strict_zip": True,
            "members": inventory.member_inventory,
            "zip_overhead_bytes": inventory.archive_bytes - len(raw),
        },
        "payload_header": {
            "schema": "henosis_x_compact_bundle",
            "header_bytes": 30,
            "lengths_24le": dict(zip(names, lengths)),
            "payload_bytes": len(raw),
            "core_encoded_bytes": core_encoded_bytes,
            "qpost_encoded_bytes": qpost_encoded_bytes,
            "randmulti_tail_bytes": len(segments["randmulti"]),
        },
        "segments": {name: _segment_summary(data) for name, data in segments.items()},
        "_segments_bytes": segments,
    }


def _read_uvarint(data: bytes, cursor: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while cursor < len(data):
        byte = data[cursor]
        cursor += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, cursor
        shift += 7
        if shift > 63:
            break
    raise HenosisTransferError("truncated or overlong uvarint")


def _uleb128(value: int) -> bytes:
    if value < 0:
        raise HenosisTransferError(f"cannot encode negative uvarint: {value}")
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def decode_delta_varint_actions(data: bytes, *, record_count: int) -> bytes:
    raw = brotli.decompress(data)
    out = bytearray(record_count * 4)
    cursor = 0
    pair_index = 0
    out_cursor = 0
    for _ in range(record_count):
        delta, cursor = _read_uvarint(raw, cursor)
        pair_index += delta
        if pair_index >= 10_000:
            raise HenosisTransferError(f"P6 action pair index out of bounds: {pair_index}")
        if cursor + 2 > len(raw):
            raise HenosisTransferError("P6 action stream ended inside record")
        tile_id = raw[cursor]
        action_id = raw[cursor + 1]
        cursor += 2
        out[out_cursor : out_cursor + 2] = pair_index.to_bytes(2, "little")
        out[out_cursor + 2] = tile_id
        out[out_cursor + 3] = action_id
        out_cursor += 4
    if cursor != len(raw):
        raise HenosisTransferError("P6 action stream has trailing bytes")
    return bytes(out)


def decode_fixed_pr75_actions(data: bytes) -> bytes:
    raw = brotli.decompress(data)
    if raw.startswith(b"SG2") or (len(raw) % 4 != 0 and len(raw) % 5 != 0):
        out = bytearray()
        cursor = 3 if raw.startswith(b"SG2") else 0
        while cursor < len(raw):
            tile, cursor = _read_uvarint(raw, cursor)
            count, cursor = _read_uvarint(raw, cursor)
            if count <= 0:
                raise HenosisTransferError("SG2 action group has zero records")
            frame = 0
            for idx in range(count):
                delta, cursor = _read_uvarint(raw, cursor)
                frame = delta if idx == 0 else frame + delta
                if frame >= 10_000:
                    raise HenosisTransferError(f"SG2 action frame out of bounds: {frame}")
                if cursor >= len(raw):
                    raise HenosisTransferError("SG2 action stream ended inside record")
                action = raw[cursor]
                cursor += 1
                out += int(frame).to_bytes(2, "little") + bytes([int(tile), int(action)])
        raw = bytes(out)
    if len(raw) % 4 != 0:
        raise HenosisTransferError(f"fixed action records are not 4-byte aligned: {len(raw)}")
    return raw


def encode_delta_varint_actions(raw_actions: bytes) -> bytes:
    if len(raw_actions) % 4:
        raise HenosisTransferError(f"action records must be 4-byte aligned, got {len(raw_actions)}")
    out = bytearray()
    previous_pair = 0
    for idx in range(0, len(raw_actions), 4):
        pair_index = int.from_bytes(raw_actions[idx : idx + 2], "little")
        tile_id = raw_actions[idx + 2]
        action_id = raw_actions[idx + 3]
        delta = pair_index if idx == 0 else pair_index - previous_pair
        if delta < 0:
            raise HenosisTransferError("P6 action delta encoding requires nondecreasing pair indices")
        out.extend(_uleb128(delta))
        out.append(tile_id)
        out.append(action_id)
        previous_pair = pair_index
    return bytes(out)


def _fixed_pr75_lengths(payload: bytes) -> tuple[int, int, int] | None:
    for total_len, renderer_len, actions_len, _unused in KNOWN_FIXED_PR75_VARIANTS:
        if len(payload) == total_len:
            return PR75_MASK_LEN, renderer_len, actions_len
    return None


def parse_pr75_payload(label: str, archive: Path, *, expected_sha256: str | None) -> Pr75Archive:
    inventory = strict_zip_inventory(archive, expected_members=("p",), expected_sha256=expected_sha256)
    payload = inventory.members["p"]
    record_count: int | None = None
    if payload.startswith(b"P6"):
        if len(payload) < 12:
            raise HenosisTransferError(f"{label}: P6 payload too short")
        mask_len, renderer_len, actions_len, record_count = struct.unpack_from("<IHHH", payload, 2)
        cursor = 2 + struct.calcsize("<IHHH")
        payload_format = "public_pr75_qzs3_qp1_segactions_p6_delta_varint"
    else:
        fixed_lengths = _fixed_pr75_lengths(payload)
        if fixed_lengths is None:
            raise HenosisTransferError(
                f"{label}: unsupported PR75 payload prefix={payload[:4]!r} len={len(payload)}"
            )
        mask_len, renderer_len, actions_len = fixed_lengths
        cursor = 0
        payload_format = "public_pr75_qzs3_qp1_segactions_fixed_slices"

    mask_end = cursor + mask_len
    renderer_end = mask_end + renderer_len
    actions_end = renderer_end + actions_len
    if actions_end >= len(payload):
        raise HenosisTransferError(f"{label}: stream lengths leave no pose payload")
    encoded = {
        "mask": payload[cursor:mask_end],
        "renderer": payload[mask_end:renderer_end],
        "actions": payload[renderer_end:actions_end],
        "pose": payload[actions_end:],
    }
    decoded_mask = brotli.decompress(encoded["mask"])
    decoded_renderer = brotli.decompress(encoded["renderer"])
    decoded_pose_qp1 = brotli.decompress(encoded["pose"])
    if not decoded_pose_qp1.startswith(b"QP1"):
        raise HenosisTransferError(f"{label}: decoded pose stream is not QP1")
    if payload.startswith(b"P6"):
        if record_count is None:
            raise HenosisTransferError(f"{label}: P6 record count missing")
        decoded_actions = decode_delta_varint_actions(encoded["actions"], record_count=record_count)
    else:
        decoded_actions = decode_fixed_pr75_actions(encoded["actions"])
        record_count = len(decoded_actions) // 4
    decoded = {
        "masks.mkv": decoded_mask,
        "renderer.bin": decoded_renderer,
        "seg_tile_actions.bin": decoded_actions,
        "optimized_poses.qp1": decoded_pose_qp1,
    }
    return Pr75Archive(
        label=label,
        path=inventory.archive,
        archive_bytes=inventory.archive_bytes,
        archive_sha256=inventory.archive_sha256,
        payload=payload,
        payload_sha256=_sha256_bytes(payload),
        payload_format=payload_format,
        encoded_streams=encoded,
        decoded_members=decoded,
        action_record_count=int(record_count),
        zip_inventory={
            "strict_zip": True,
            "members": inventory.member_inventory,
            "zip_overhead_bytes": inventory.archive_bytes - len(payload),
        },
    )


def _decoded_member_summary(decoded: Mapping[str, bytes]) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "bytes": len(data),
            "sha256": _sha256_bytes(data),
            "magic_hex": data[:12].hex(),
        }
        for name, data in sorted(decoded.items())
    }


def _encoded_stream_summary(encoded: Mapping[str, bytes], *, action_record_count: int) -> dict[str, dict[str, Any]]:
    out = {
        name: {
            "bytes": len(data),
            "sha256": _sha256_bytes(data),
            "magic_hex": data[:12].hex(),
        }
        for name, data in sorted(encoded.items())
    }
    out["actions"]["record_count"] = action_record_count
    return out


def public_pr75_anatomy(source: Pr75Archive) -> dict[str, Any]:
    return {
        "label": source.label,
        "path": str(source.path),
        "archive_bytes": source.archive_bytes,
        "archive_sha256": source.archive_sha256,
        "payload_bytes": len(source.payload),
        "payload_sha256": source.payload_sha256,
        "payload_format": source.payload_format,
        "zip_inventory": source.zip_inventory,
        "encoded_streams": _encoded_stream_summary(
            source.encoded_streams,
            action_record_count=source.action_record_count,
        ),
        "decoded_members": _decoded_member_summary(source.decoded_members),
    }


def _brotli_best(raw: bytes, *, source: bytes | None = None) -> tuple[bytes, dict[str, int] | str]:
    best = source
    best_params: dict[str, int] | str = "source" if source is not None else {}
    for quality in (11, 10, 9, 6, 4, 2, 0):
        for mode in (0, 1, 2):
            for lgwin in (10, 16, 22, 24):
                candidate = brotli.compress(raw, quality=quality, mode=mode, lgwin=lgwin)
                if best is None or len(candidate) < len(best):
                    best = candidate
                    best_params = {"quality": quality, "mode": mode, "lgwin": lgwin}
    if best is None:
        raise HenosisTransferError("no Brotli candidate produced")
    if brotli.decompress(best) != raw:
        raise HenosisTransferError("selected Brotli stream failed round-trip")
    return best, best_params


def decode_pr65_p1d1_pose(encoded_pose: bytes) -> np.ndarray:
    pose_bytes = brotli.decompress(encoded_pose)
    if not pose_bytes.startswith(b"P1D1"):
        raise HenosisTransferError(f"PR65 pose stream is not P1D1: {pose_bytes[:4]!r}")
    pos = 4
    count = pose_bytes[pos]
    pos += 1
    dims: list[int] = []
    lengths: list[int] = []
    for _ in range(count):
        dims.append(int(pose_bytes[pos]))
        pos += 1
        lengths.append(int.from_bytes(pose_bytes[pos : pos + 2], "little"))
        pos += 2
    pose = np.zeros((600, 6), dtype=np.float32)
    for dim, n_bytes in zip(dims, lengths):
        stream = pose_bytes[pos : pos + n_bytes]
        pos += n_bytes
        vals = np.empty(600, dtype=np.uint32)
        acc = 0
        shift = 0
        out_idx = 0
        for byte in stream:
            acc |= (int(byte) & 0x7F) << shift
            if byte & 0x80:
                shift += 7
            else:
                if out_idx >= 600:
                    raise HenosisTransferError("P1D1 pose stream has too many samples")
                vals[out_idx] = acc
                out_idx += 1
                acc = 0
                shift = 0
        if out_idx != 600:
            raise HenosisTransferError(f"P1D1 dim {dim} decoded {out_idx} samples, expected 600")
        delta = ((vals.astype(np.int32) >> 1) ^ -(vals.astype(np.int32) & 1)).astype(np.int32)
        q = np.cumsum(delta)
        if dim == 0:
            pose[:, 0] = q.astype(np.float32) / 512.0 + 20.0
        else:
            q = q.clip(-32768, 32767).astype(np.int16)
            pose[:, int(dim)] = q.astype(np.float32) / 2048.0
    if pos != len(pose_bytes):
        raise HenosisTransferError("P1D1 pose stream has trailing bytes")
    return pose


def _pose_stats(label: str, poses: np.ndarray) -> dict[str, Any]:
    return {
        "label": label,
        "shape": list(poses.shape),
        "mean": [float(v) for v in poses.mean(axis=0)],
        "std": [float(v) for v in poses.std(axis=0)],
        "min": [float(v) for v in poses.min(axis=0)],
        "max": [float(v) for v in poses.max(axis=0)],
        "sha256_float32_le": _sha256_bytes(poses.astype("<f4", copy=False).tobytes()),
    }


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, FIXED_ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o644 << 16
    info.create_system = 3
    return info


def _write_single_p_archive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(_zip_info("p"), payload)


def _build_p6_payload(
    *,
    mask_br: bytes,
    renderer_br: bytes,
    actions_br: bytes,
    action_record_count: int,
    pose_br: bytes,
) -> bytes:
    if max(len(renderer_br), len(actions_br), len(pose_br)) > 0xFFFF:
        raise HenosisTransferError("P6 u16 stream length limit exceeded")
    return (
        b"P6"
        + struct.pack("<IHHH", len(mask_br), len(renderer_br), len(actions_br), int(action_record_count))
        + mask_br
        + renderer_br
        + actions_br
        + pose_br
    )


def build_p6_repack_candidate(
    *,
    candidate_id: str,
    source: Pr75Archive,
    actions_source: Pr75Archive | None = None,
    pose_qp1: bytes,
    output_dir: Path,
    pose_source_label: str,
) -> dict[str, Any]:
    action_source = actions_source or source
    raw_actions = action_source.decoded_members["seg_tile_actions.bin"]
    delta_actions = encode_delta_varint_actions(raw_actions)
    actions_br, actions_params = _brotli_best(
        delta_actions,
        source=action_source.encoded_streams["actions"] if action_source.payload.startswith(b"P6") else None,
    )
    pose_br, pose_params = _brotli_best(pose_qp1, source=None)
    payload = _build_p6_payload(
        mask_br=source.encoded_streams["mask"],
        renderer_br=source.encoded_streams["renderer"],
        actions_br=actions_br,
        action_record_count=len(raw_actions) // 4,
        pose_br=pose_br,
    )
    candidate_dir = output_dir / candidate_id
    archive = candidate_dir / "archive.zip"
    _write_single_p_archive(archive, payload)
    parsed = parse_pr75_payload(candidate_id, archive, expected_sha256=None)
    changed_streams = [
        name
        for name, member in {
            "mask": "masks.mkv",
            "renderer": "renderer.bin",
            "actions": "seg_tile_actions.bin",
            "pose": "optimized_poses.qp1",
        }.items()
        if _sha256_bytes(parsed.decoded_members[member]) != _sha256_bytes(source.decoded_members[member])
    ]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "tool": TOOL,
        "candidate_id": candidate_id,
        "candidate_family": "p6_repack_pose_transfer",
        "score_claim": False,
        "promotion_eligible": False,
        "remote_dispatch": {"dispatched": False, "lightning_state_touched": False},
        "source_archive_id": source.label,
        "source_archive": str(source.path),
        "source_archive_bytes": source.archive_bytes,
        "source_archive_sha256": source.archive_sha256,
        "action_source_archive_id": action_source.label,
        "action_source_archive": str(action_source.path),
        "action_source_archive_sha256": action_source.archive_sha256,
        "archive": str(archive),
        "archive_bytes": parsed.archive_bytes,
        "archive_sha256": parsed.archive_sha256,
        "payload_bytes": len(payload),
        "payload_sha256": _sha256_bytes(payload),
        "payload_format": parsed.payload_format,
        "pose_source_label": pose_source_label,
        "archive_byte_delta_vs_c091": parsed.archive_bytes - C091_BYTES,
        "formula_rate_score_delta_vs_c091": (parsed.archive_bytes - C091_BYTES) * RATE_SCORE_PER_BYTE,
        "decoded_stream_closure": {
            "status": "passed",
            "changed_decoded_streams_vs_source": changed_streams,
            "candidate_decoded_members": _decoded_member_summary(parsed.decoded_members),
        },
        "encoded_streams": _encoded_stream_summary(
            parsed.encoded_streams,
            action_record_count=parsed.action_record_count,
        ),
        "brotli_params": {
            "actions_delta_varint": actions_params,
            "pose_qp1": pose_params,
        },
    }
    manifest.update(_candidate_break_even(archive_bytes=parsed.archive_bytes))
    _write_json(candidate_dir / "manifest.json", manifest)
    return manifest


def _load_qpost_builder() -> Any:
    spec = importlib.util.spec_from_file_location("pr65_henosis_qpost_builder", QPOST_BUILDER_PATH)
    if spec is None or spec.loader is None:
        raise HenosisTransferError(f"cannot load qpost builder: {QPOST_BUILDER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_trace(path: Path | None) -> dict[int, dict[str, float]]:
    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text())
    samples = payload.get("samples", [])
    if not isinstance(samples, list):
        raise HenosisTransferError(f"component trace missing samples list: {path}")
    out: dict[int, dict[str, float]] = {}
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        pair = int(sample["pair_index"])
        seg = float(sample.get("score_seg_contribution_exact", 0.0))
        pose = float(sample.get("score_pose_contribution_first_order", 0.0))
        combined = float(sample.get("score_combined_contribution_first_order", seg + pose))
        out[pair] = {"seg": seg, "pose": pose, "combined": combined}
    return out


def rank_transfer_pairs(
    *,
    source_trace: Mapping[int, Mapping[str, float]],
    pr65_trace: Mapping[int, Mapping[str, float]],
    metric: str,
    top_pairs: int,
) -> list[dict[str, Any]]:
    if metric not in {"seg", "pose", "combined"}:
        raise HenosisTransferError(f"unsupported rank metric: {metric}")
    rows: list[dict[str, Any]] = []
    for pair in sorted(set(source_trace) & set(pr65_trace)):
        source = source_trace[pair]
        target = pr65_trace[pair]
        advantages = {
            key: float(source.get(key, 0.0)) - float(target.get(key, 0.0))
            for key in ("seg", "pose", "combined")
        }
        rows.append(
            {
                "pair_index": pair,
                "rank_metric": metric,
                "rank_advantage": advantages[metric],
                "rank_metric_positive": advantages[metric] > 0.0,
                "advantage": advantages,
                "source_contribution": {key: float(source.get(key, 0.0)) for key in ("seg", "pose", "combined")},
                "pr65_trace_contribution": {key: float(target.get(key, 0.0)) for key in ("seg", "pose", "combined")},
            }
        )
    rows.sort(key=lambda row: (-float(row["rank_advantage"]), int(row["pair_index"])))
    return rows[:top_pairs]


def official_pr65_math() -> dict[str, Any]:
    c091_component = 100.0 * C091_SEG + math.sqrt(10.0 * C091_POSE)
    pr65_component = 100.0 * PR65_OFFICIAL_SEG + math.sqrt(10.0 * PR65_OFFICIAL_POSE)
    pr65_rate = 25.0 * PR65_OFFICIAL_BYTES / ORIGINAL_VIDEO_BYTES
    pr65_score = pr65_component + pr65_rate
    return {
        "c091_anchor": {
            "score": C091_SCORE,
            "archive_bytes": C091_BYTES,
            "pose": C091_POSE,
            "seg": C091_SEG,
            "component_score": c091_component,
            "rate_score": 25.0 * C091_BYTES / ORIGINAL_VIDEO_BYTES,
        },
        "pr65_official_fields_user_supplied": {
            "archive_bytes": PR65_OFFICIAL_BYTES,
            "pose": PR65_OFFICIAL_POSE,
            "seg": PR65_OFFICIAL_SEG,
            "component_score": pr65_component,
            "rate_score": pr65_rate,
            "score_recomputed": pr65_score,
            "score_delta_vs_c091": pr65_score - C091_SCORE,
            "component_gain_vs_c091": c091_component - pr65_component,
            "byte_delta_vs_c091": PR65_OFFICIAL_BYTES - C091_BYTES,
            "rate_penalty_vs_c091": (PR65_OFFICIAL_BYTES - C091_BYTES) * RATE_SCORE_PER_BYTE,
        },
        "sub314": {
            "target": SUB314_TARGET,
            "c091_component_gain_required_at_same_bytes": C091_SCORE - SUB314_TARGET,
        },
    }


def _candidate_break_even(
    *,
    archive_bytes: int,
    selected_pair_records: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    delta_bytes = archive_bytes - C091_BYTES
    rate_delta = delta_bytes * RATE_SCORE_PER_BYTE
    selected_adv = {"seg": 0.0, "pose": 0.0, "combined": 0.0}
    if selected_pair_records:
        for record in selected_pair_records:
            advantage = record.get("advantage", {})
            if isinstance(advantage, Mapping):
                for key in selected_adv:
                    selected_adv[key] += max(0.0, float(advantage.get(key, 0.0)))
    return {
        "score_claim": False,
        "archive_byte_delta_vs_c091": delta_bytes,
        "formula_rate_score_delta_vs_c091": rate_delta,
        "component_gain_required_to_beat_c091": rate_delta,
        "component_gain_required_for_sub314": (C091_SCORE - SUB314_TARGET) + rate_delta,
        "selected_pair_trace_positive_advantage_sum": selected_adv,
        "selected_trace_combined_advantage_minus_sub314_requirement": (
            selected_adv["combined"] - ((C091_SCORE - SUB314_TARGET) + rate_delta)
        ),
        "trace_is_promotable_evidence": False,
    }


def build_qpost_candidate(
    *,
    spec: QPostSpec,
    source_archive: Path,
    pr65_archive: Path,
    output_dir: Path,
    selected_pair_records: Sequence[Mapping[str, Any]],
    qpost_builder: Any,
) -> dict[str, Any]:
    candidate_dir = output_dir / spec.candidate_id
    archive = candidate_dir / "archive.zip"
    pair_indices = tuple(int(row["pair_index"]) for row in selected_pair_records)
    if not pair_indices:
        raise HenosisTransferError(f"{spec.candidate_id}: no positive transfer pairs selected")
    builder_meta = qpost_builder.build_candidate(
        source_archive,
        pr65_archive,
        archive,
        include_streams=tuple(spec.include_streams),
        pair_indices=pair_indices,
    )
    inventory = strict_zip_inventory(archive, expected_members=("p", "qpost.bin"))
    qpost_bytes = inventory.members["qpost.bin"]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "tool": TOOL,
        "candidate_id": spec.candidate_id,
        "candidate_family": "pr65_qpost_sidecar",
        "score_claim": False,
        "promotion_eligible": False,
        "remote_dispatch": {"dispatched": False, "lightning_state_touched": False},
        "source_archive_id": spec.source_archive_id,
        "source_archive": str(source_archive.resolve()),
        "pr65_archive": str(pr65_archive.resolve()),
        "archive": str(archive),
        "archive_bytes": inventory.archive_bytes,
        "archive_sha256": inventory.archive_sha256,
        "strict_zip_inventory": {
            "strict_zip": True,
            "members": inventory.member_inventory,
        },
        "include_streams": list(spec.include_streams),
        "rank_metric": spec.rank_metric,
        "top_pairs_requested": spec.top_pairs,
        "selected_pair_count": len(pair_indices),
        "selected_positive_pair_count": sum(
            1 for row in selected_pair_records if bool(row.get("rank_metric_positive"))
        ),
        "selected_pair_indices": list(pair_indices),
        "selected_pair_rank_records": list(selected_pair_records),
        "builder_meta": builder_meta,
        "qpost_bin": {
            "bytes": len(qpost_bytes),
            "sha256": _sha256_bytes(qpost_bytes),
            "magic_hex": qpost_bytes[:12].hex(),
        },
        "risk_family": spec.risk_family,
    }
    manifest.update(_candidate_break_even(archive_bytes=inventory.archive_bytes, selected_pair_records=selected_pair_records))
    manifest["dispatch_recommendation"] = {
        "class": "non_dispatchable_planning_artifact",
        "reason": (
            "This slice does not dispatch GPU work. PR65 qpost bias has existing C089/C091-adjacent "
            "exact-negative evidence, and component traces are only pair-ranking signal."
        ),
        "required_before_exact_eval": [
            "operator selects candidate after reviewing qpost exact-negative artifacts",
            "claim lane with tools/claim_lane_dispatch.py",
            "run experiments/contest_auth_eval.py --device cuda on exact archive bytes",
        ],
    }
    manifest["exact_eval_command_template"] = (
        ".venv/bin/python -u experiments/contest_auth_eval.py "
        f"--archive {archive} --inflate-sh submissions/robust_current/inflate.sh "
        "--upstream-dir upstream --device cuda --keep-work-dir "
        f"--work-dir {output_dir / 'exact_eval_work' / spec.candidate_id}"
    )
    _write_json(candidate_dir / "manifest.json", manifest)
    return manifest


def _copy_public_fields(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if not key.startswith("_")}


def build_candidate_matrix(
    *,
    pr65_archive: Path = DEFAULT_PR65_ARCHIVE,
    c091_archive: Path = DEFAULT_C091_ARCHIVE,
    c089_archive: Path = DEFAULT_C089_ARCHIVE,
    pr65_trace: Path | None = DEFAULT_PR65_TRACE,
    c091_trace: Path | None = DEFAULT_C091_TRACE,
    c089_trace: Path | None = DEFAULT_C089_TRACE,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    qpost_specs: Sequence[QPostSpec] = DEFAULT_QPOST_SPECS,
    expected_pr65_sha256: str | None = EXPECTED_PR65_SHA256,
    expected_c091_sha256: str | None = EXPECTED_C091_SHA256,
    expected_c089_sha256: str | None = EXPECTED_C089_SHA256,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    pr65 = parse_pr65_henosis_archive(pr65_archive, expected_sha256=expected_pr65_sha256)
    c091 = parse_pr75_payload("c091_pr75_minp_public_replay", c091_archive, expected_sha256=expected_c091_sha256)
    c089 = parse_pr75_payload("c089_pr75_qp1_top40_p6", c089_archive, expected_sha256=expected_c089_sha256)

    pr65_pose = decode_pr65_p1d1_pose(pr65["_segments_bytes"]["pose"])
    c091_pose = decode_qp1(c091.decoded_members["optimized_poses.qp1"])
    pr65_pose_qp1 = encode_qp1(pr65_pose)

    c091_p6_control = build_p6_repack_candidate(
        candidate_id="c091_renderer_pose_c089_actions_p6_control",
        source=c091,
        actions_source=c089,
        pose_qp1=c091.decoded_members["optimized_poses.qp1"],
        output_dir=output_dir,
        pose_source_label="c091_original_qp1",
    )
    pose_transfer = build_p6_repack_candidate(
        candidate_id="c091_pr65_pose_qp1_c089_actions_p6",
        source=c091,
        actions_source=c089,
        pose_qp1=pr65_pose_qp1,
        output_dir=output_dir,
        pose_source_label="pr65_p1d1_reencoded_qp1_velocity_only",
    )
    source_archives: dict[str, Path] = {
        "c091": c091.path,
        "c091_pr65_pose_qp1_c089_actions_p6": Path(pose_transfer["archive"]),
    }

    source_trace = _load_trace(c091_trace)
    pr65_rank_trace = _load_trace(pr65_trace)
    c089_rank_trace = _load_trace(c089_trace)
    rankings = {
        metric: rank_transfer_pairs(
            source_trace=source_trace,
            pr65_trace=pr65_rank_trace,
            metric=metric,
            top_pairs=128,
        )
        for metric in ("seg", "pose", "combined")
    }
    qpost_builder = _load_qpost_builder()
    qpost_candidates: list[dict[str, Any]] = []
    for spec in qpost_specs:
        source_path = source_archives.get(spec.source_archive_id)
        if source_path is None:
            raise HenosisTransferError(f"{spec.candidate_id}: unknown source archive id {spec.source_archive_id!r}")
        selected = rankings[spec.rank_metric][: spec.top_pairs]
        qpost_candidates.append(
            build_qpost_candidate(
                spec=spec,
                source_archive=source_path,
                pr65_archive=pr65_archive,
                output_dir=output_dir,
                selected_pair_records=selected,
                qpost_builder=qpost_builder,
            )
        )

    all_candidates = [c091_p6_control, pose_transfer, *qpost_candidates]
    for candidate in all_candidates:
        candidate.update(
            _candidate_break_even(
                archive_bytes=int(candidate["archive_bytes"]),
                selected_pair_records=candidate.get("selected_pair_rank_records", []),
            )
        )
        candidate.setdefault(
            "dispatch_recommendation",
            {
                "class": "non_dispatchable_planning_artifact",
                "reason": "Local byte/planning candidate only; exact CUDA eval requires a separate dispatch claim.",
                "required_before_exact_eval": [
                    "claim lane with tools/claim_lane_dispatch.py",
                    "run experiments/contest_auth_eval.py --device cuda on exact archive bytes",
                ],
            },
        )

    summary = {
        "schema_version": SCHEMA_VERSION,
        "tool": TOOL,
        "score_claim": False,
        "promotion_eligible": False,
        "remote_dispatch": {"dispatched": False, "lightning_state_touched": False},
        "output_dir": str(output_dir),
        "official_math": official_pr65_math(),
        "source_anatomy": {
            "pr65_henosis": _copy_public_fields(pr65),
            "c091_pr75_minp_public_replay": public_pr75_anatomy(c091),
            "c089_pr75_qp1_top40_p6": public_pr75_anatomy(c089),
        },
        "pose_transfer": {
            "pr65_pose_stats": _pose_stats("pr65_p1d1", pr65_pose),
            "c091_pose_stats": _pose_stats("c091_qp1", c091_pose),
            "pr65_pose_qp1_bytes": len(pr65_pose_qp1),
            "pr65_pose_qp1_sha256": _sha256_bytes(pr65_pose_qp1),
            "pr65_pose_qp1_brotli_bytes": len(brotli.compress(pr65_pose_qp1, quality=11)),
            "pr65_pose_qp1_is_velocity_only": True,
        },
        "trace_rankings": {
            "trace_inputs": {
                "source_c091_trace": str(c091_trace) if c091_trace else None,
                "pr65_trace": str(pr65_trace) if pr65_trace else None,
                "c089_trace": str(c089_trace) if c089_trace else None,
            },
            "source_trace_pair_count": len(source_trace),
            "pr65_trace_pair_count": len(pr65_rank_trace),
            "c089_trace_pair_count": len(c089_rank_trace),
            "top_by_metric": {metric: rows[:16] for metric, rows in rankings.items()},
            "trace_is_promotable_evidence": False,
        },
        "candidates": all_candidates,
        "non_dispatchable_guards": [
            "No GPU dispatch was performed by this tool.",
            "No Lightning state or .omx/state dispatch records are written.",
            "All candidates have score_claim=false and promotion_eligible=false.",
            "Component traces rank atoms only; exact CUDA auth eval is required for any score claim.",
            "PR65 qpost bias has existing exact-negative evidence on nearby C089/PR75 candidates, so qpost sidecars are non-dispatchable by default.",
        ],
    }
    _write_json(output_dir / "candidate_matrix.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pr65-archive", type=Path, default=DEFAULT_PR65_ARCHIVE)
    parser.add_argument("--c091-archive", type=Path, default=DEFAULT_C091_ARCHIVE)
    parser.add_argument("--c089-archive", type=Path, default=DEFAULT_C089_ARCHIVE)
    parser.add_argument("--pr65-trace", type=Path, default=DEFAULT_PR65_TRACE)
    parser.add_argument("--c091-trace", type=Path, default=DEFAULT_C091_TRACE)
    parser.add_argument("--c089-trace", type=Path, default=DEFAULT_C089_TRACE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    summary = build_candidate_matrix(
        pr65_archive=args.pr65_archive,
        c091_archive=args.c091_archive,
        c089_archive=args.c089_archive,
        pr65_trace=args.pr65_trace,
        c091_trace=args.c091_trace,
        c089_trace=args.c089_trace,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
