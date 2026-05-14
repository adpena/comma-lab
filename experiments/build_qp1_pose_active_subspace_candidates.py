#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build local QP1 pose active-subspace candidates for the current PR75/C-089 stack.

This is a local candidate/screen builder only.  It preserves the source
archive's mask, renderer, and PR75 action slices byte-for-byte, rewrites only
the Brotli-compressed QP1 pose stream, and records enough custody for a later
exact CUDA auth eval.  It does not dispatch jobs and does not claim score.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import brotli
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from tac.qp1_pose_codec import (
    QP1_MAGIC,
    QPV1_MAGIC,
    QPV1Payload,
    VELOCITY_OFFSET,
    VELOCITY_SCALE,
    decode_qp1,
    decode_qpv1,
    parse_qpv1,
)


FIXED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
MEMBER_NAME = "p"
QP19_MAGIC = b"QP19"
RATE_SCORE_PER_BYTE = 25.0 / 37_545_489
SCHEMA_VERSION = 1
DEFAULT_SOURCE_ARCHIVE = Path(
    "experiments/results/lightning_batch/"
    "exact_eval_c067_pr75_qp1_top40_p6_t4_awsfix1_20260503T0630Z/archive.zip"
)
DEFAULT_COMPONENT_TRACE = Path(
    "experiments/results/lightning_batch/"
    "exact_eval_c067_pr75_qp1_top40_p6_t4_awsfix1_20260503T0630Z/component_trace.json"
)
DEFAULT_REFERENCE_BASE_ARCHIVE = Path(
    "experiments/results/public_floor_qzs3_qp1_packer_20260502/"
    "pr63_qzs3_qp1_fixedslice/archive.zip"
)
DEFAULT_REFERENCE_ACTIVE_ARCHIVE = Path(
    "experiments/results/vast_harvest/"
    "line_search_qzs3_qp1_pr67_active_subspace_c057_fix2_20260502T0240Z_latest/"
    "archive.zip"
)
DEFAULT_OUTPUT_DIR = Path("experiments/results/qp1_pose_active_subspace_worker_20260503")
PRODUCER = "experiments/build_qp1_pose_active_subspace_candidates.py"


class QP1ActiveSubspaceError(ValueError):
    """Raised when source custody or candidate construction fails closed."""


@dataclass(frozen=True)
class ArchiveParts:
    payload_format: str
    member_name: str
    payload: bytes
    prefix_before_pose: bytes
    mask_br: bytes
    renderer_br: bytes
    pose_br: bytes
    actions_br: bytes = b""
    action_dict_br: bytes = b""
    record_count: int | None = None
    pose_codec: str = "QP1"
    payload_header: dict[str, Any] | None = None


@dataclass(frozen=True)
class TraceSample:
    pair_index: int
    pose_score: float
    seg_score: float
    combined_score: float


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    kind: str
    metric: str
    topk: int
    alpha: float
    max_abs_delta_q: int
    trust: float
    risk: float
    wall_clock_minutes: float


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_single_member_name(name: str) -> str:
    if not name or name.startswith("/") or "\\" in name or "\x00" in name:
        raise QP1ActiveSubspaceError(f"unsafe archive member path: {name!r}")
    path = Path(name)
    if len(path.parts) != 1 or any(part in {"", ".", ".."} for part in path.parts):
        raise QP1ActiveSubspaceError(f"unsafe archive member path: {name!r}")
    if name.startswith(".") or name == "__MACOSX":
        raise QP1ActiveSubspaceError(f"hidden/system archive member path: {name!r}")
    return name


def _assert_local_header_name_matches(archive: Path, info: zipfile.ZipInfo) -> None:
    with archive.open("rb") as handle:
        handle.seek(info.header_offset)
        fixed = handle.read(30)
        if len(fixed) != 30 or fixed[:4] != b"PK\x03\x04":
            raise QP1ActiveSubspaceError("invalid ZIP local file header")
        name_len, extra_len = struct.unpack_from("<HH", fixed, 26)
        local_name = handle.read(name_len).decode("utf-8")
        if local_name != info.filename:
            raise QP1ActiveSubspaceError(
                f"ZIP central/local name mismatch: central={info.filename!r} "
                f"local={local_name!r}"
            )
        if extra_len:
            handle.read(extra_len)


def read_single_member_payload(path: Path, *, member_name: str = MEMBER_NAME) -> bytes:
    member_name = _safe_single_member_name(member_name)
    with zipfile.ZipFile(path, "r") as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
        names = [info.filename for info in infos]
        if names != [member_name]:
            raise QP1ActiveSubspaceError(
                f"expected single member {member_name!r}; got {names!r}"
            )
        info = infos[0]
        if info.compress_type != zipfile.ZIP_STORED:
            raise QP1ActiveSubspaceError("source member must be ZIP_STORED")
        _safe_single_member_name(info.filename)
        _assert_local_header_name_matches(path, info)
        return zf.read(info)


def _looks_like_mask_obu(data: bytes) -> bool:
    return data.startswith(b"\x12\x00\x0a\x0a") or data.startswith(b"\x12\x00")


def _brotli_decompress(data: bytes, label: str) -> bytes:
    try:
        return brotli.decompress(data)
    except brotli.error as exc:
        raise QP1ActiveSubspaceError(f"{label} is not a valid Brotli stream") from exc


def _pose_codec_label(raw: bytes) -> str:
    if raw.startswith(QP1_MAGIC):
        return "QP1"
    if raw.startswith(QPV1_MAGIC):
        return "QPV1"
    raise QP1ActiveSubspaceError(f"decoded pose failed known magic: {raw[:4]!r}")


def _pose_float32_sha256_for_raw(raw: bytes) -> str:
    if raw.startswith(QP1_MAGIC):
        decoded = decode_qp1(raw)
    elif raw.startswith(QPV1_MAGIC):
        decoded = decode_qpv1(raw)
    else:
        raise QP1ActiveSubspaceError(f"unsupported pose stream magic: {raw[:4]!r}")
    return _sha256_bytes(np.asarray(decoded, dtype=np.float32).tobytes(order="C"))


def parse_pr75_payload(payload: bytes, *, member_name: str = MEMBER_NAME) -> ArchiveParts:
    """Parse PR75 P3/P4/P5/P6 payloads and fixed public PR75 slices."""
    if payload.startswith(QP19_MAGIC):
        return parse_qp19_payload(payload, member_name=member_name)

    if payload.startswith(b"P3"):
        header_size = 2 + struct.calcsize("<IHH")
        mask_len, model_len, actions_len = struct.unpack_from("<IHH", payload, 2)
        dict_len = 0
        record_count = None
        payload_format = "public_pr75_qzs3_qp1_segactions_p3"
    elif payload.startswith(b"P4"):
        header_size = 2 + struct.calcsize("<IHHH")
        mask_len, model_len, dict_len, actions_len = struct.unpack_from(
            "<IHHH", payload, 2
        )
        record_count = None
        payload_format = "public_pr75_qzs3_qp1_segactions_p4_custom_dict"
    elif payload.startswith(b"P5"):
        header_size = 2 + struct.calcsize("<IHHHH")
        mask_len, model_len, dict_len, actions_len, record_count = struct.unpack_from(
            "<IHHHH", payload, 2
        )
        payload_format = "public_pr75_qzs3_qp1_segactions_p5_packed_custom_dict"
    elif payload.startswith(b"P6"):
        header_size = 2 + struct.calcsize("<IHHH")
        mask_len, model_len, actions_len, record_count = struct.unpack_from(
            "<IHHH", payload, 2
        )
        dict_len = 0
        payload_format = "public_pr75_qzs3_qp1_segactions_p6_delta_varint"
    else:
        return _parse_fixed_slice_payload(payload, member_name=member_name)

    if min(mask_len, model_len, actions_len) <= 0 or dict_len < 0:
        raise QP1ActiveSubspaceError("invalid nonpositive PR75 slice length")
    cursor = header_size
    mask_start = cursor
    mask_end = mask_start + int(mask_len)
    model_end = mask_end + int(model_len)
    dict_end = model_end + int(dict_len)
    actions_end = dict_end + int(actions_len)
    if actions_end >= len(payload):
        raise QP1ActiveSubspaceError(
            f"PR75 payload too short for pose slice: actions_end={actions_end}, "
            f"payload={len(payload)}"
        )
    mask_br = payload[mask_start:mask_end]
    renderer_br = payload[mask_end:model_end]
    action_dict_br = payload[model_end:dict_end]
    actions_br = payload[dict_end:actions_end]
    pose_br = payload[actions_end:]

    masks = _brotli_decompress(mask_br, "masks.mkv")
    renderer = _brotli_decompress(renderer_br, "renderer.bin")
    pose_raw = _brotli_decompress(pose_br, "optimized_poses.qp1")
    if not _looks_like_mask_obu(masks):
        raise QP1ActiveSubspaceError(f"decoded masks failed magic: {masks[:4]!r}")
    if not renderer.startswith(b"QZS3"):
        raise QP1ActiveSubspaceError(f"decoded renderer failed QZS3 magic: {renderer[:4]!r}")
    pose_codec = _pose_codec_label(pose_raw)
    if action_dict_br:
        _brotli_decompress(action_dict_br, "seg_tile_action_dict.bin")
    _brotli_decompress(actions_br, "seg_tile_actions.bin")

    return ArchiveParts(
        payload_format=payload_format,
        member_name=member_name,
        payload=payload,
        prefix_before_pose=payload[:actions_end],
        mask_br=mask_br,
        renderer_br=renderer_br,
        pose_br=pose_br,
        actions_br=actions_br,
        action_dict_br=action_dict_br,
        record_count=record_count,
        pose_codec=pose_codec,
    )


def parse_qp19_payload(payload: bytes, *, member_name: str = MEMBER_NAME) -> ArchiveParts:
    """Parse PR77's self-describing QP19 container.

    QP19 stores mask, renderer, and pose Brotli streams. The pose stream may be
    PR77's multidimensional QPV1 stream or a legacy QP1 stream; unknown pose
    magic fails closed.
    """

    if not payload.startswith(QP19_MAGIC):
        raise QP1ActiveSubspaceError(f"bad QP19 magic: {payload[:4]!r}")
    header_size = 18
    if len(payload) < header_size:
        raise QP1ActiveSubspaceError("QP19 payload is too short")
    version = payload[4]
    flags = payload[5]
    if version != 1:
        raise QP1ActiveSubspaceError(f"unsupported QP19 payload version: {version}")
    mask_len, model_len, pose_len = struct.unpack_from("<III", payload, 6)
    if min(mask_len, model_len, pose_len) <= 0:
        raise QP1ActiveSubspaceError("invalid nonpositive QP19 slice length")
    mask_start = header_size
    mask_end = mask_start + int(mask_len)
    model_end = mask_end + int(model_len)
    pose_end = model_end + int(pose_len)
    if pose_end != len(payload):
        raise QP1ActiveSubspaceError(
            f"QP19 payload length mismatch: header={pose_end} actual={len(payload)}"
        )
    mask_br = payload[mask_start:mask_end]
    renderer_br = payload[mask_end:model_end]
    pose_br = payload[model_end:pose_end]
    masks = _brotli_decompress(mask_br, "masks.mkv")
    renderer = _brotli_decompress(renderer_br, "renderer.bin")
    pose_raw = _brotli_decompress(pose_br, "optimized_poses.qp")
    if not _looks_like_mask_obu(masks):
        raise QP1ActiveSubspaceError(f"decoded masks failed magic: {masks[:4]!r}")
    if not renderer.startswith(b"QZS3"):
        raise QP1ActiveSubspaceError(f"decoded renderer failed QZS3 magic: {renderer[:4]!r}")
    pose_codec = _pose_codec_label(pose_raw)
    return ArchiveParts(
        payload_format="public_pr77_qp19_qzs3_pose_v1",
        member_name=member_name,
        payload=payload,
        prefix_before_pose=payload[:model_end],
        mask_br=mask_br,
        renderer_br=renderer_br,
        pose_br=pose_br,
        pose_codec=pose_codec,
        payload_header={
            "kind": "QP19",
            "version": int(version),
            "flags": int(flags),
            "bytes": header_size,
            "mask_br_bytes": int(mask_len),
            "renderer_br_bytes": int(model_len),
            "pose_br_bytes": int(pose_len),
        },
    )


def _parse_fixed_slice_payload(payload: bytes, *, member_name: str) -> ArchiveParts:
    # Try known fixed-slice layouts in strict magic-validating order.
    layouts = [
        ("public_pr77_qzs3_qp1_tile_delta_fixed_slices", 219_472, 55_756, 325),
        ("public_pr75_minp_qzs3_qp1_segactions_fixed_slices", 219_472, 55_756, 255),
        ("public_pr75_minp_qzs3_qp1_segactions_fixed_slices", 219_472, 55_756, 253),
        ("public_c067_qzs3_qp1_fixed_slices", 219_472, 55_965, 0),
        ("public_pr75_qzs3_qp1_segactions_fixed_slices", 219_472, 56_034, 236),
        ("public_pr67_qzs3_qp1_fixed_slices", 219_472, 56_093, 0),
    ]
    last_error: Exception | None = None
    for payload_format, mask_len, model_len, actions_len in layouts:
        if len(payload) <= mask_len + model_len + actions_len:
            continue
        mask_br = payload[:mask_len]
        renderer_br = payload[mask_len:mask_len + model_len]
        actions_br = payload[mask_len + model_len:mask_len + model_len + actions_len]
        pose_br = payload[mask_len + model_len + actions_len:]
        try:
            masks = _brotli_decompress(mask_br, "masks.mkv")
            renderer = _brotli_decompress(renderer_br, "renderer.bin")
            if actions_br:
                _brotli_decompress(actions_br, "seg_tile_actions.bin")
            pose_raw = _brotli_decompress(pose_br, "optimized_poses.qp1")
            if _looks_like_mask_obu(masks) and renderer.startswith(b"QZS3"):
                pose_codec = _pose_codec_label(pose_raw)
                return ArchiveParts(
                    payload_format=payload_format,
                    member_name=member_name,
                    payload=payload,
                    prefix_before_pose=payload[:mask_len + model_len + actions_len],
                    mask_br=mask_br,
                    renderer_br=renderer_br,
                    pose_br=pose_br,
                    actions_br=actions_br,
                    pose_codec=pose_codec,
                )
        except QP1ActiveSubspaceError as exc:
            last_error = exc
    raise QP1ActiveSubspaceError(
        "payload is not a supported PR75/P3-P6 or fixed QZS3/QP1 layout"
        + (f": {last_error}" if last_error else "")
    )


def load_archive_parts(path: Path, *, member_name: str = MEMBER_NAME) -> ArchiveParts:
    return parse_pr75_payload(read_single_member_payload(path, member_name=member_name))


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(_safe_single_member_name(name), FIXED_ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o644 << 16
    info.create_system = 3
    return info


def write_single_member_archive(path: Path, *, payload: bytes, member_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(_zip_info(member_name), payload)


def build_candidate_payload(source_parts: ArchiveParts, candidate_pose_br: bytes) -> bytes:
    if source_parts.payload.startswith(QP19_MAGIC):
        header = bytearray(source_parts.prefix_before_pose)
        if len(header) < 18:
            raise QP1ActiveSubspaceError("QP19 source prefix is too short")
        struct.pack_into("<I", header, 14, len(candidate_pose_br))
        return bytes(header) + candidate_pose_br
    return source_parts.prefix_before_pose + candidate_pose_br


def decode_qp1_words(payload: bytes) -> list[int]:
    if not payload.startswith(QP1_MAGIC) or len(payload) < 5:
        raise QP1ActiveSubspaceError("invalid QP1 stream")
    vals = [struct.unpack_from("<H", payload, 3)[0]]
    cursor = 5
    while cursor < len(payload):
        shift = 0
        acc = 0
        while True:
            if cursor >= len(payload):
                raise QP1ActiveSubspaceError("truncated QP1 VLQ")
            byte = payload[cursor]
            cursor += 1
            acc |= (byte & 0x7F) << shift
            if byte < 0x80:
                break
            shift += 7
        delta = (acc >> 1) ^ -(acc & 1)
        vals.append((vals[-1] + delta) & 0xFFFF)
    return vals


def _zigzag_encode(delta: int) -> int:
    return delta << 1 if delta >= 0 else ((-delta) << 1) - 1


def _vlq_encode(value: int) -> bytes:
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def encode_qp1_words(words: list[int]) -> bytes:
    if not words:
        raise QP1ActiveSubspaceError("cannot encode empty QP1 words")
    for word in words:
        if int(word) < 0 or int(word) > 0xFFFF:
            raise QP1ActiveSubspaceError(f"QP1 word out of uint16 range: {word}")
    out = bytearray(QP1_MAGIC)
    out.extend(struct.pack("<H", int(words[0])))
    previous = int(words[0])
    for word in words[1:]:
        current = int(word)
        out.extend(_vlq_encode(_zigzag_encode(current - previous)))
        previous = current
    return bytes(out)


def _pose_float32_sha256(qp1_raw: bytes) -> str:
    return _pose_float32_sha256_for_raw(qp1_raw)


def _words_sha256(words: list[int]) -> str:
    arr = np.asarray(words, dtype="<u2")
    return _sha256_bytes(arr.tobytes(order="C"))


def _signed_words_sha256(words: list[int]) -> str:
    arr = np.asarray(words, dtype="<i4")
    return _sha256_bytes(arr.tobytes(order="C"))


def load_trace_samples(path: Path | None) -> tuple[list[TraceSample], dict[str, Any]]:
    if path is None or not path.exists():
        return [], {}
    payload = json.loads(path.read_text())
    raw_samples = payload.get("samples") or payload.get("top_pose_samples") or []
    samples = [
        TraceSample(
            pair_index=int(sample["pair_index"]),
            pose_score=float(sample.get("score_pose_contribution_first_order", 0.0)),
            seg_score=float(sample.get("score_seg_contribution_exact", 0.0)),
            combined_score=float(sample.get("score_combined_contribution_first_order", 0.0)),
        )
        for sample in raw_samples
        if "pair_index" in sample
    ]
    return samples, payload


def _ranked_indices(
    samples: list[TraceSample],
    *,
    metric: str,
    topk: int,
    n_words: int,
    fallback_words: list[int],
) -> list[int]:
    if samples:
        def score(sample: TraceSample) -> float:
            if metric == "pose":
                return sample.pose_score
            if metric == "seg":
                return sample.seg_score
            return sample.combined_score

        ranked = sorted(
            (sample for sample in samples if 0 <= sample.pair_index < n_words),
            key=lambda sample: (-score(sample), sample.pair_index),
        )
        out: list[int] = []
        seen: set[int] = set()
        for sample in ranked:
            if sample.pair_index not in seen:
                out.append(sample.pair_index)
                seen.add(sample.pair_index)
            if len(out) >= topk:
                return out
    jerk: list[tuple[int, int]] = []
    for index in range(1, n_words - 1):
        value = abs(fallback_words[index - 1] - 2 * fallback_words[index] + fallback_words[index + 1])
        jerk.append((value, index))
    return [index for _value, index in sorted(jerk, key=lambda item: (-item[0], item[1]))[:topk]]


def _sample_score_lookup(samples: list[TraceSample], metric: str) -> dict[int, float]:
    lookup: dict[int, float] = {}
    for sample in samples:
        if metric == "pose":
            value = sample.pose_score
        elif metric == "seg":
            value = sample.seg_score
        else:
            value = sample.combined_score
        lookup[sample.pair_index] = max(0.0, float(value))
    return lookup


def _clamp_delta(value: int, limit: int) -> int:
    return max(-limit, min(limit, int(value)))


def _apply_neighbor_pull(
    words: list[int],
    indices: list[int],
    *,
    alpha: float,
    max_abs_delta_q: int,
) -> tuple[list[int], list[dict[str, Any]]]:
    out = list(words)
    changes: list[dict[str, Any]] = []
    for index in indices:
        if index <= 0 or index >= len(words) - 1:
            continue
        target = int(round((int(words[index - 1]) + int(words[index + 1])) / 2.0))
        raw_delta = target - int(words[index])
        delta = int(round(raw_delta * alpha))
        if delta == 0 and raw_delta != 0:
            delta = 1 if raw_delta > 0 else -1
        delta = _clamp_delta(delta, max_abs_delta_q)
        if delta == 0:
            continue
        new_word = max(0, min(0xFFFF, int(out[index]) + delta))
        if new_word == int(out[index]):
            continue
        out[index] = new_word
        changes.append({
            "pair_index": index,
            "basis": "hard_pair_neighbor_pull",
            "q_before": int(words[index]),
            "q_after": int(new_word),
            "delta_q": int(new_word) - int(words[index]),
            "velocity_before": int(words[index]) / VELOCITY_SCALE + VELOCITY_OFFSET,
            "velocity_after": int(new_word) / VELOCITY_SCALE + VELOCITY_OFFSET,
            "neighbor_target_q": target,
        })
    return out, changes


def _apply_reference_delta(
    words: list[int],
    indices: list[int],
    *,
    reference_delta: list[int],
    alpha: float,
    max_abs_delta_q: int,
) -> tuple[list[int], list[dict[str, Any]]]:
    out = list(words)
    changes: list[dict[str, Any]] = []
    for index in indices:
        if index < 0 or index >= len(words) or index >= len(reference_delta):
            continue
        raw_delta = int(reference_delta[index])
        delta = int(round(raw_delta * alpha))
        if delta == 0 and raw_delta != 0:
            delta = 1 if raw_delta > 0 else -1
        delta = _clamp_delta(delta, max_abs_delta_q)
        if delta == 0:
            continue
        new_word = max(0, min(0xFFFF, int(out[index]) + delta))
        if new_word == int(out[index]):
            continue
        out[index] = new_word
        changes.append({
            "pair_index": index,
            "basis": "reference_active_subspace_delta",
            "q_before": int(words[index]),
            "q_after": int(new_word),
            "delta_q": int(new_word) - int(words[index]),
            "reference_delta_q": raw_delta,
            "velocity_before": int(words[index]) / VELOCITY_SCALE + VELOCITY_OFFSET,
            "velocity_after": int(new_word) / VELOCITY_SCALE + VELOCITY_OFFSET,
        })
    return out, changes


def _qpv1_value_to_float(payload: QPV1Payload, dim: int, value: int) -> float:
    for stream in payload.streams:
        if stream.dim == dim:
            return float(stream.offset) + float(value) / float(stream.scale)
    raise QP1ActiveSubspaceError(f"QPV1 dim {dim} not found")


def _qpv1_primary_words(payload: QPV1Payload) -> list[int]:
    if not payload.streams:
        raise QP1ActiveSubspaceError("QPV1 payload has no dimension streams")
    primary = min(payload.streams, key=lambda stream: stream.dim)
    return list(primary.values)


def _apply_qpv1_neighbor_pull(
    payload: QPV1Payload,
    indices: list[int],
    *,
    alpha: float,
    max_abs_delta_q: int,
) -> tuple[dict[int, list[int]], list[dict[str, Any]]]:
    out = payload.values_by_dim()
    changes: list[dict[str, Any]] = []
    for stream in payload.streams:
        words = list(stream.values)
        dim_words = out[stream.dim]
        for index in indices:
            if index <= 0 or index >= len(words) - 1:
                continue
            target = int(round((int(words[index - 1]) + int(words[index + 1])) / 2.0))
            raw_delta = target - int(words[index])
            delta = int(round(raw_delta * alpha))
            if delta == 0 and raw_delta != 0:
                delta = 1 if raw_delta > 0 else -1
            delta = _clamp_delta(delta, max_abs_delta_q)
            if delta == 0:
                continue
            new_word = int(dim_words[index]) + delta
            if new_word < -(2**31) or new_word > 2**31 - 1:
                continue
            if new_word == int(dim_words[index]):
                continue
            before = int(dim_words[index])
            dim_words[index] = new_word
            changes.append({
                "pair_index": index,
                "dim": int(stream.dim),
                "basis": "qpv1_hard_pair_neighbor_pull",
                "q_before": before,
                "q_after": int(new_word),
                "delta_q": int(new_word) - before,
                "pose_before": _qpv1_value_to_float(payload, stream.dim, before),
                "pose_after": _qpv1_value_to_float(payload, stream.dim, int(new_word)),
                "neighbor_target_q": target,
            })
    return out, changes


def _apply_qpv1_reference_delta(
    payload: QPV1Payload,
    indices: list[int],
    *,
    reference_delta: dict[int, list[int]],
    alpha: float,
    max_abs_delta_q: int,
) -> tuple[dict[int, list[int]], list[dict[str, Any]]]:
    out = payload.values_by_dim()
    changes: list[dict[str, Any]] = []
    for stream in payload.streams:
        if stream.dim not in reference_delta:
            continue
        deltas = reference_delta[stream.dim]
        dim_words = out[stream.dim]
        for index in indices:
            if index < 0 or index >= payload.count or index >= len(deltas):
                continue
            raw_delta = int(deltas[index])
            delta = int(round(raw_delta * alpha))
            if delta == 0 and raw_delta != 0:
                delta = 1 if raw_delta > 0 else -1
            delta = _clamp_delta(delta, max_abs_delta_q)
            if delta == 0:
                continue
            before = int(dim_words[index])
            new_word = before + delta
            if new_word < -(2**31) or new_word > 2**31 - 1 or new_word == before:
                continue
            dim_words[index] = new_word
            changes.append({
                "pair_index": index,
                "dim": int(stream.dim),
                "basis": "qpv1_reference_active_subspace_delta",
                "q_before": before,
                "q_after": int(new_word),
                "delta_q": int(new_word) - before,
                "reference_delta_q": raw_delta,
                "pose_before": _qpv1_value_to_float(payload, stream.dim, before),
                "pose_after": _qpv1_value_to_float(payload, stream.dim, int(new_word)),
            })
    return out, changes


def default_candidate_specs(*, has_reference_delta: bool) -> list[CandidateSpec]:
    specs: list[CandidateSpec] = []
    if has_reference_delta:
        specs.extend([
            CandidateSpec("ref_active_pose_top16_s025", "reference_delta", "pose", 16, 0.25, 4, 0.08, 1.25, 8.0),
            CandidateSpec("ref_active_combined_top32_s0125", "reference_delta", "combined", 32, 0.125, 3, 0.06, 1.35, 8.0),
        ])
    specs.extend([
        CandidateSpec("neighbor_pose_top12_a025", "neighbor_pull", "pose", 12, 0.25, 3, 0.025, 1.70, 5.0),
        CandidateSpec("neighbor_combined_top20_a020", "neighbor_pull", "combined", 20, 0.20, 2, 0.020, 1.85, 5.0),
        CandidateSpec("neighbor_pose_top32_a0125", "neighbor_pull", "pose", 32, 0.125, 2, 0.018, 2.00, 5.0),
    ])
    return specs


def _load_reference_delta(
    base_archive: Path | None,
    active_archive: Path | None,
    *,
    expected_len: int,
    pose_codec: str = "QP1",
) -> tuple[list[int] | dict[int, list[int]] | None, dict[str, Any] | None]:
    if base_archive is None or active_archive is None:
        return None, None
    if not base_archive.exists() or not active_archive.exists():
        return None, None
    try:
        base_parts = load_archive_parts(base_archive)
        active_parts = load_archive_parts(active_archive)
        if base_parts.pose_codec != pose_codec or active_parts.pose_codec != pose_codec:
            return None, {
                "usable": False,
                "reason": "reference pose codec mismatch",
                "base_pose_codec": base_parts.pose_codec,
                "active_pose_codec": active_parts.pose_codec,
                "expected_pose_codec": pose_codec,
                "base_archive": str(base_archive),
                "active_archive": str(active_archive),
            }
        base_raw = _brotli_decompress(base_parts.pose_br, "reference_base_pose")
        active_raw = _brotli_decompress(active_parts.pose_br, "reference_active_pose")
        if pose_codec == "QP1":
            base_words = decode_qp1_words(base_raw)
            active_words = decode_qp1_words(active_raw)
        elif pose_codec == "QPV1":
            base_payload = parse_qpv1(base_raw)
            active_payload = parse_qpv1(active_raw)
            if [stream.dim for stream in base_payload.streams] != [
                stream.dim for stream in active_payload.streams
            ]:
                return None, {
                    "usable": False,
                    "reason": "reference QPV1 dimensions mismatch",
                    "base_archive": str(base_archive),
                    "active_archive": str(active_archive),
                }
            if any(
                (
                    base_stream.offset != active_stream.offset
                    or base_stream.scale != active_stream.scale
                    or base_stream.dim != active_stream.dim
                )
                for base_stream, active_stream in zip(
                    base_payload.streams, active_payload.streams, strict=True
                )
            ):
                return None, {
                    "usable": False,
                    "reason": "reference QPV1 stream metadata mismatch",
                    "base_archive": str(base_archive),
                    "active_archive": str(active_archive),
                }
            if base_payload.count != expected_len or active_payload.count != expected_len:
                return None, {
                    "usable": False,
                    "reason": "reference QPV1 pose length mismatch",
                    "base_len": base_payload.count,
                    "active_len": active_payload.count,
                    "expected_len": expected_len,
                }
            delta_by_dim = {
                base_stream.dim: [
                    int(a) - int(b)
                    for a, b in zip(active_stream.values, base_stream.values, strict=True)
                ]
                for base_stream, active_stream in zip(
                    base_payload.streams, active_payload.streams, strict=True
                )
            }
            flat_delta = [value for values in delta_by_dim.values() for value in values]
            return delta_by_dim, {
                "usable": any(value != 0 for value in flat_delta),
                "pose_codec": pose_codec,
                "base_archive": str(base_archive),
                "base_archive_sha256": _sha256_file(base_archive),
                "active_archive": str(active_archive),
                "active_archive_sha256": _sha256_file(active_archive),
                "nonzero_delta_count": sum(1 for value in flat_delta if value != 0),
                "max_abs_delta_q": max((abs(value) for value in flat_delta), default=0),
                "delta_dims": sorted(delta_by_dim),
                "delta_words_sha256": _words_sha256([value & 0xFFFF for value in flat_delta]),
            }
        else:
            raise QP1ActiveSubspaceError(f"unsupported reference pose codec: {pose_codec}")
    except QP1ActiveSubspaceError as exc:
        return None, {
            "usable": False,
            "reason": f"reference archive parse failed: {exc}",
            "base_archive": str(base_archive),
            "active_archive": str(active_archive),
        }
    if len(base_words) != expected_len or len(active_words) != expected_len:
        return None, {
            "usable": False,
            "reason": "reference pose length mismatch",
            "base_len": len(base_words),
            "active_len": len(active_words),
            "expected_len": expected_len,
        }
    delta = [int(a) - int(b) for a, b in zip(active_words, base_words, strict=True)]
    return delta, {
        "usable": any(value != 0 for value in delta),
        "base_archive": str(base_archive),
        "base_archive_sha256": _sha256_file(base_archive),
        "active_archive": str(active_archive),
        "active_archive_sha256": _sha256_file(active_archive),
        "pose_codec": pose_codec,
        "nonzero_delta_count": sum(1 for value in delta if value != 0),
        "max_abs_delta_q": max((abs(value) for value in delta), default=0),
        "delta_words_sha256": _words_sha256([value & 0xFFFF for value in delta]),
    }


def _candidate_exact_eval_command(archive_path: Path) -> list[str]:
    return [
        ".venv/bin/python",
        "experiments/contest_auth_eval.py",
        "--archive",
        str(archive_path),
        "--upstream-dir",
        "upstream",
        "--device",
        "cuda",
    ]


def _roundtrip_gates(
    candidate_archive: Path,
    *,
    source_parts: ArchiveParts,
    candidate_words: list[int],
    candidate_pose_raw: bytes,
) -> dict[str, Any]:
    candidate_parts = load_archive_parts(candidate_archive, member_name=source_parts.member_name)
    roundtrip_raw = _brotli_decompress(candidate_parts.pose_br, "candidate_pose")
    if source_parts.pose_codec == "QP1":
        words_match_policy = decode_qp1_words(roundtrip_raw) == candidate_words
        canonical_reencode = encode_qp1_words(candidate_words) == roundtrip_raw
    elif source_parts.pose_codec == "QPV1":
        parsed = parse_qpv1(roundtrip_raw)
        words_match_policy = _qpv1_primary_words(parsed) == candidate_words
        canonical_reencode = parsed.to_bytes() == roundtrip_raw
    else:
        words_match_policy = False
        canonical_reencode = False
    gates = {
        "single_member_zip": True,
        "payload_format_preserved": candidate_parts.payload_format == source_parts.payload_format,
        "pose_codec_preserved": candidate_parts.pose_codec == source_parts.pose_codec,
        "mask_slice_preserved": candidate_parts.mask_br == source_parts.mask_br,
        "renderer_slice_preserved": candidate_parts.renderer_br == source_parts.renderer_br,
        "actions_slice_preserved": candidate_parts.actions_br == source_parts.actions_br,
        "action_dict_slice_preserved": candidate_parts.action_dict_br == source_parts.action_dict_br,
        "candidate_pose_brotli_decodes": roundtrip_raw == candidate_pose_raw,
        "candidate_pose_words_match_policy": words_match_policy,
        "candidate_pose_canonical_reencode": canonical_reencode,
        "no_sidecars_in_archive": True,
    }
    gates["all_passed"] = all(bool(value) for value in gates.values())
    return gates


def build_candidates(
    *,
    source_archive: Path,
    output_dir: Path,
    component_trace: Path | None = None,
    reference_base_archive: Path | None = None,
    reference_active_archive: Path | None = None,
    specs: list[CandidateSpec] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    source_archive = source_archive.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        raise QP1ActiveSubspaceError(f"output dir is not empty; pass --force: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    source_parts = load_archive_parts(source_archive)
    source_pose_raw = _brotli_decompress(source_parts.pose_br, "source_pose")
    source_qpv1: QPV1Payload | None = None
    if source_parts.pose_codec == "QP1":
        source_words = decode_qp1_words(source_pose_raw)
        source_canonical = encode_qp1_words(source_words)
        if source_canonical != source_pose_raw:
            raise QP1ActiveSubspaceError("source QP1 stream is not canonical under local encoder")
        pose_rows = len(source_words)
        ranking_words = source_words
    elif source_parts.pose_codec == "QPV1":
        source_qpv1 = parse_qpv1(source_pose_raw)
        source_words = []
        pose_rows = source_qpv1.count
        ranking_words = _qpv1_primary_words(source_qpv1)
        if source_qpv1.to_bytes() != source_pose_raw:
            raise QP1ActiveSubspaceError("source QPV1 stream is not canonical under local encoder")
    else:
        raise QP1ActiveSubspaceError(f"unsupported source pose codec: {source_parts.pose_codec}")
    samples, trace_payload = load_trace_samples(component_trace)
    reference_delta, reference_summary = _load_reference_delta(
        reference_base_archive,
        reference_active_archive,
        expected_len=pose_rows,
        pose_codec=source_parts.pose_codec,
    )
    has_reference_delta = bool(reference_delta) and bool(reference_summary and reference_summary.get("usable"))
    if specs is None:
        specs = default_candidate_specs(has_reference_delta=has_reference_delta)

    source_score = trace_payload.get("score_recomputed_from_components")
    source_archive_bytes = source_archive.stat().st_size
    source_pose_float_sha = _pose_float32_sha256(source_pose_raw)
    source_custody = {
        "archive_path": str(source_archive),
        "archive_bytes": source_archive_bytes,
        "archive_sha256": _sha256_file(source_archive),
        "payload_format": source_parts.payload_format,
        "payload_bytes": len(source_parts.payload),
        "payload_sha256": _sha256_bytes(source_parts.payload),
        "mask_br_sha256": _sha256_bytes(source_parts.mask_br),
        "renderer_br_sha256": _sha256_bytes(source_parts.renderer_br),
        "actions_br_sha256": _sha256_bytes(source_parts.actions_br),
        "action_dict_br_sha256": _sha256_bytes(source_parts.action_dict_br),
        "pose_br_bytes": len(source_parts.pose_br),
        "pose_br_sha256": _sha256_bytes(source_parts.pose_br),
        "pose_codec": source_parts.pose_codec,
        "payload_header": source_parts.payload_header,
        "pose_stream_bytes": len(source_pose_raw),
        "pose_stream_sha256": _sha256_bytes(source_pose_raw),
        "pose_words_sha256": (
            _words_sha256(source_words)
            if source_words
            else _signed_words_sha256([
                value
                for stream in (source_qpv1.streams if source_qpv1 else ())
                for value in stream.values
            ])
        ),
        "pose_float32_semantic_sha256": source_pose_float_sha,
        "pose_rows": pose_rows,
        "velocity_offset": VELOCITY_OFFSET,
        "velocity_scale": VELOCITY_SCALE,
        "non_velocity_columns": (
            "zeroed_by_qp1_contract"
            if source_parts.pose_codec == "QP1"
            else "encoded_by_qpv1_dimension_streams"
        ),
    }

    candidates: list[dict[str, Any]] = []
    for spec in specs:
        indices = _ranked_indices(
            samples,
            metric=spec.metric,
            topk=spec.topk,
            n_words=pose_rows,
            fallback_words=ranking_words,
        )
        if source_parts.pose_codec == "QP1" and spec.kind == "reference_delta":
            if reference_delta is None:
                continue
            candidate_words, changes = _apply_reference_delta(
                source_words,
                indices,
                reference_delta=reference_delta,  # type: ignore[arg-type]
                alpha=spec.alpha,
                max_abs_delta_q=spec.max_abs_delta_q,
            )
            candidate_pose_raw = encode_qp1_words(candidate_words)
        elif source_parts.pose_codec == "QP1" and spec.kind == "neighbor_pull":
            candidate_words, changes = _apply_neighbor_pull(
                source_words,
                indices,
                alpha=spec.alpha,
                max_abs_delta_q=spec.max_abs_delta_q,
            )
            candidate_pose_raw = encode_qp1_words(candidate_words)
        elif source_parts.pose_codec == "QPV1" and spec.kind == "reference_delta":
            if reference_delta is None or source_qpv1 is None:
                continue
            qpv1_values, changes = _apply_qpv1_reference_delta(
                source_qpv1,
                indices,
                reference_delta=reference_delta,  # type: ignore[arg-type]
                alpha=spec.alpha,
                max_abs_delta_q=spec.max_abs_delta_q,
            )
            candidate_pose_raw = source_qpv1.with_values(qpv1_values).to_bytes()
            candidate_words = _qpv1_primary_words(parse_qpv1(candidate_pose_raw))
        elif source_parts.pose_codec == "QPV1" and spec.kind == "neighbor_pull":
            if source_qpv1 is None:
                continue
            qpv1_values, changes = _apply_qpv1_neighbor_pull(
                source_qpv1,
                indices,
                alpha=spec.alpha,
                max_abs_delta_q=spec.max_abs_delta_q,
            )
            candidate_pose_raw = source_qpv1.with_values(qpv1_values).to_bytes()
            candidate_words = _qpv1_primary_words(parse_qpv1(candidate_pose_raw))
        else:
            raise QP1ActiveSubspaceError(f"unsupported candidate kind: {spec.kind}")
        if not changes:
            continue

        candidate_pose_br = brotli.compress(candidate_pose_raw, quality=11)
        candidate_payload = build_candidate_payload(source_parts, candidate_pose_br)
        candidate_dir = output_dir / spec.name
        archive_path = candidate_dir / "archive.zip"
        manifest_path = candidate_dir / "manifest.json"
        write_single_member_archive(
            archive_path,
            payload=candidate_payload,
            member_name=source_parts.member_name,
        )
        gates = _roundtrip_gates(
            archive_path,
            source_parts=source_parts,
            candidate_words=candidate_words,
            candidate_pose_raw=candidate_pose_raw,
        )
        candidate_archive_bytes = archive_path.stat().st_size
        archive_delta = candidate_archive_bytes - source_archive_bytes
        rate_delta = archive_delta * RATE_SCORE_PER_BYTE
        score_lookup = _sample_score_lookup(samples, spec.metric)
        selected_trace_mass = sum(score_lookup.get(change["pair_index"], 0.0) for change in changes)
        mean_abs_delta_q = sum(abs(change["delta_q"]) for change in changes) / len(changes)
        magnitude_factor = min(1.0, mean_abs_delta_q / max(1.0, float(spec.max_abs_delta_q)))
        expected_component_reduction = selected_trace_mass * spec.trust * magnitude_factor
        expected_score_delta = rate_delta - expected_component_reduction
        expected_score = (
            float(source_score) + expected_score_delta
            if source_score is not None and math.isfinite(float(source_score))
            else None
        )
        risk_adjusted_yield = (
            (-expected_score_delta) / (spec.risk * spec.wall_clock_minutes)
            if spec.risk > 0 and spec.wall_clock_minutes > 0
            else 0.0
        )

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "tool": PRODUCER,
            "candidate_id": spec.name,
            "score_claim": False,
            "promotion_eligible": False,
            "evidence_grade": "empirical_local_policy_screen",
            "required_score_truth": (
                "archive.zip -> inflate.sh -> upstream/evaluate.py via "
                "experiments/contest_auth_eval.py --device cuda"
            ),
            "source": source_custody,
            "candidate": {
                "archive_path": str(archive_path),
                "archive_bytes": candidate_archive_bytes,
                "archive_sha256": _sha256_file(archive_path),
                "payload_bytes": len(candidate_payload),
                "payload_sha256": _sha256_bytes(candidate_payload),
                "payload_format": source_parts.payload_format,
                "pose_br_bytes": len(candidate_pose_br),
                "pose_br_sha256": _sha256_bytes(candidate_pose_br),
                "pose_codec": source_parts.pose_codec,
                "pose_stream_bytes": len(candidate_pose_raw),
                "pose_stream_sha256": _sha256_bytes(candidate_pose_raw),
                "pose_words_sha256": (
                    _words_sha256(candidate_words)
                    if source_parts.pose_codec == "QP1"
                    else _signed_words_sha256(candidate_words)
                ),
                "pose_float32_semantic_sha256": _pose_float32_sha256(candidate_pose_raw),
                "changed_pair_count": len(changes),
                "changed_pairs": changes,
                "no_sidecars": True,
            },
            "preservation": {
                "mask_br_sha256": _sha256_bytes(source_parts.mask_br),
                "renderer_br_sha256": _sha256_bytes(source_parts.renderer_br),
                "actions_br_sha256": _sha256_bytes(source_parts.actions_br),
                "action_dict_br_sha256": _sha256_bytes(source_parts.action_dict_br),
                "mask_renderer_actions_preserved_byte_for_byte": True,
            },
            "policy": {
                "kind": spec.kind,
                "metric": spec.metric,
                "topk": spec.topk,
                "alpha": spec.alpha,
                "max_abs_delta_q": spec.max_abs_delta_q,
                "selected_indices": indices,
                "trust": spec.trust,
                "risk": spec.risk,
                "wall_clock_minutes": spec.wall_clock_minutes,
                "reference_delta": reference_summary,
            },
            "local_roundtrip_gates": gates,
            "ranking": {
                "archive_delta_bytes": archive_delta,
                "formula_rate_score_delta": rate_delta,
                "selected_trace_mass": selected_trace_mass,
                "expected_component_reduction_proxy": expected_component_reduction,
                "expected_score_delta_proxy": expected_score_delta,
                "expected_score_proxy": expected_score,
                "risk_adjusted_score_reduction_per_minute_proxy": risk_adjusted_yield,
                "ranking_basis": (
                    "Trace-mass and QP1 local-basis proxy only; exact CUDA auth eval "
                    "is required before any score claim."
                ),
            },
            "exact_eval_command_template": (
                _candidate_exact_eval_command(archive_path) if gates["all_passed"] else None
            ),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        candidates.append(manifest)

    candidates.sort(
        key=lambda item: (
            -float(item["ranking"]["risk_adjusted_score_reduction_per_minute_proxy"]),
            float(item["ranking"]["expected_score_delta_proxy"]),
            item["candidate_id"],
        )
    )
    for rank, candidate in enumerate(candidates, start=1):
        candidate["rank"] = rank
        Path(candidate["candidate"]["archive_path"]).with_name("manifest.json").write_text(
            json.dumps(candidate, indent=2, sort_keys=True) + "\n"
        )

    summary = {
        "schema_version": SCHEMA_VERSION,
        "tool": PRODUCER,
        "score_claim": False,
        "scope": "local_policy_screen_no_remote_dispatch",
        "source": source_custody,
        "component_trace": str(component_trace) if component_trace else None,
        "component_trace_sha256": _sha256_file(component_trace) if component_trace and component_trace.exists() else None,
        "reference_delta": reference_summary,
        "rate_score_per_byte": RATE_SCORE_PER_BYTE,
        "candidate_count": len(candidates),
        "top_candidates": [
            {
                "rank": candidate["rank"],
                "candidate_id": candidate["candidate_id"],
                "archive": candidate["candidate"]["archive_path"],
                "archive_bytes": candidate["candidate"]["archive_bytes"],
                "archive_sha256": candidate["candidate"]["archive_sha256"],
                "changed_pair_count": candidate["candidate"]["changed_pair_count"],
                "archive_delta_bytes": candidate["ranking"]["archive_delta_bytes"],
                "expected_score_delta_proxy": candidate["ranking"]["expected_score_delta_proxy"],
                "expected_score_proxy": candidate["ranking"]["expected_score_proxy"],
                "risk_adjusted_score_reduction_per_minute_proxy": candidate["ranking"][
                    "risk_adjusted_score_reduction_per_minute_proxy"
                ],
                "risk": candidate["policy"]["risk"],
                "local_roundtrip_gates_passed": candidate["local_roundtrip_gates"]["all_passed"],
                "exact_eval_command_template": candidate["exact_eval_command_template"],
            }
            for candidate in candidates
        ],
    }
    (output_dir / "candidate_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def _parse_spec(raw: str) -> CandidateSpec:
    parts = raw.split(":")
    if len(parts) != 8:
        raise argparse.ArgumentTypeError(
            "--candidate must be NAME:KIND:METRIC:TOPK:ALPHA:MAX_DELTA:TRUST:RISK"
        )
    name, kind, metric, topk, alpha, max_delta, trust, risk = parts
    if kind not in {"reference_delta", "neighbor_pull"}:
        raise argparse.ArgumentTypeError(f"unsupported candidate kind: {kind}")
    if metric not in {"pose", "seg", "combined"}:
        raise argparse.ArgumentTypeError(f"unsupported metric: {metric}")
    return CandidateSpec(
        name=name,
        kind=kind,
        metric=metric,
        topk=int(topk),
        alpha=float(alpha),
        max_abs_delta_q=int(max_delta),
        trust=float(trust),
        risk=float(risk),
        wall_clock_minutes=6.0,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", type=Path, default=DEFAULT_SOURCE_ARCHIVE)
    parser.add_argument("--component-trace", type=Path, default=DEFAULT_COMPONENT_TRACE)
    parser.add_argument("--reference-base-archive", type=Path, default=DEFAULT_REFERENCE_BASE_ARCHIVE)
    parser.add_argument("--reference-active-archive", type=Path, default=DEFAULT_REFERENCE_ACTIVE_ARCHIVE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--candidate", action="append", type=_parse_spec, default=[])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    summary = build_candidates(
        source_archive=args.source_archive,
        output_dir=args.output_dir,
        component_trace=args.component_trace,
        reference_base_archive=args.reference_base_archive,
        reference_active_archive=args.reference_active_archive,
        specs=args.candidate or None,
        force=args.force,
    )
    print(json.dumps({
        "output_dir": str(args.output_dir),
        "candidate_count": summary["candidate_count"],
        "top_candidates": summary["top_candidates"][:5],
        "score_claim": summary["score_claim"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
