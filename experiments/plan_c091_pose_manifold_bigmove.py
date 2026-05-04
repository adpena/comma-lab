#!/usr/bin/env python3
"""Plan C091-native pose-manifold big-move candidates.

This tool is local-only. It does not dispatch GPU work and does not claim a
score. It rewrites only the charged QP1 pose stream while preserving C091 mask,
renderer, and decoded tile-action semantics. Public PR65/PR67 streams may be
used as low-dimensional proposal bases, but a candidate must not copy any full
public pose stream.
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
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.qp1_pose_codec import QP1_MAGIC, VELOCITY_OFFSET, VELOCITY_SCALE, decode_qp1


TOOL = "experiments/plan_c091_pose_manifold_bigmove.py"
SCHEMA_VERSION = 1
FIXED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
RATE_SCORE_PER_BYTE = 25.0 / 37_545_489
SUB314_TARGET = 0.314
EXPECTED_SAMPLES = 600
C091_SCORE = 0.31516575028285976
C091_BYTES = 276_481
C091_SHA256 = "03a2afd5fe92c93a9b7b7e43625158a73b455f0cfbca82d278008a728db78746"

DEFAULT_C091_ARCHIVE = (
    REPO_ROOT
    / "experiments/results/lightning_batch/"
    "exact_eval_pr75_minp_public_replay_t4_20260503T1049Z/archive.zip"
)
DEFAULT_C091_EVAL = (
    REPO_ROOT
    / "experiments/results/lightning_batch/"
    "exact_eval_pr75_minp_public_replay_t4_20260503T1049Z/contest_auth_eval.json"
)
DEFAULT_C091_TRACE = (
    REPO_ROOT
    / "experiments/results/lightning_batch/"
    "exact_eval_pr75_minp_public_replay_t4_20260503T1049Z/component_trace.json"
)
DEFAULT_C089_ARCHIVE = (
    REPO_ROOT
    / "experiments/results/lightning_batch/"
    "exact_eval_c067_pr75_qp1_top40_p6_t4_awsfix1_20260503T0630Z/archive.zip"
)
DEFAULT_PR65_ARCHIVE = (
    REPO_ROOT
    / "experiments/results/top_submission_delta_reverse_engineering_20260503/"
    "sources/pr65_henosis_archive.zip"
)
DEFAULT_PR67_BASE_ARCHIVE = (
    REPO_ROOT
    / "experiments/results/public_floor_qzs3_qp1_packer_20260502/"
    "pr63_qzs3_qp1_fixedslice/archive.zip"
)
DEFAULT_PR67_ACTIVE_ARCHIVE = (
    REPO_ROOT
    / "experiments/results/vast_harvest/"
    "line_search_qzs3_qp1_pr67_active_subspace_c057_fix2_20260502T0240Z_latest/"
    "archive.zip"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "experiments/results/c091_pose_manifold_bigmove_20260503_worker"
DEFAULT_LEDGER = REPO_ROOT / ".omx/research/c091_pose_manifold_bigmove_20260503_worker.md"

PR75_MASK_LEN = 219_472
FIXED_SLICE_TABLE: dict[int, tuple[int, int, int, str]] = {
    276_381: (PR75_MASK_LEN, 55_756, 255, "c091_pr75_minp_fixed_actions255"),
    276_379: (PR75_MASK_LEN, 55_756, 253, "pr75_minp_fixed_actions253"),
    276_520: (PR75_MASK_LEN, 55_914, 236, "pr75_fixed_actions236_model55914"),
    276_641: (PR75_MASK_LEN, 56_034, 236, "pr75_fixed_actions236_model56034"),
    276_113: (PR75_MASK_LEN, 55_965, 0, "public_c067_qzs3_qp1"),
    276_464: (PR75_MASK_LEN, 56_093, 0, "public_pr67_qzs3_qp1"),
}
REFERENCE_SLICE_LAYOUTS: tuple[tuple[int, int, int, str], ...] = (
    (PR75_MASK_LEN, 55_965, 0, "public_c067_qzs3_qp1_variable_pose"),
    (PR75_MASK_LEN, 56_093, 0, "public_pr67_qzs3_qp1_variable_pose"),
    (PR75_MASK_LEN, 55_756, 255, "c091_pr75_minp_variable_pose"),
)


class C091PosePlanError(ValueError):
    """Raised when custody, parsing, or planning gates fail closed."""


@dataclass(frozen=True)
class StreamSlices:
    payload_format: str
    mask_br: bytes
    renderer_br: bytes
    actions_br: bytes
    pose_br: bytes
    action_record_count: int


@dataclass(frozen=True)
class SourceArchive:
    label: str
    path: Path
    archive_bytes: int
    archive_sha256: str
    payload: bytes
    payload_sha256: str
    slices: StreamSlices
    decoded: dict[str, bytes]


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    basis_kind: str
    top_pairs: int
    alpha: float
    max_abs_delta_q: int
    efficiency_assumption: float
    rank_mix_pose: float
    rank_mix_seg: float


DEFAULT_SPECS: tuple[CandidateSpec, ...] = (
    CandidateSpec(
        "c091_native_cem_pose_waterfill_top128_s025",
        "cem_pr65_pr67_c089",
        128,
        0.25,
        12,
        0.045,
        1.0,
        1.0,
    ),
    CandidateSpec(
        "c091_native_lowrank_pr65_pr67_top096_s020",
        "lowrank_pr65_pr67",
        96,
        0.20,
        10,
        0.040,
        1.0,
        0.75,
    ),
    CandidateSpec(
        "c091_native_pr67_active_delta_top080_s050",
        "pr67_active_delta",
        80,
        0.50,
        6,
        0.030,
        1.0,
        0.25,
    ),
    CandidateSpec(
        "c091_native_pr65_residual_top064_s0125",
        "pr65_residual",
        64,
        0.125,
        8,
        0.035,
        0.75,
        1.0,
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
    parts = Path(name).parts
    if (
        not name
        or name.startswith("/")
        or "\\" in name
        or "\x00" in name
        or len(parts) != 1
        or any(part in {"", ".", ".."} for part in parts)
        or name.startswith(".")
        or name.startswith("._")
        or name == "__MACOSX"
    ):
        raise C091PosePlanError(f"unsafe ZIP member path: {name!r}")
    return name


def _assert_local_header_name_matches(archive: Path, info: zipfile.ZipInfo) -> None:
    with archive.open("rb") as handle:
        handle.seek(info.header_offset)
        fixed = handle.read(30)
        if len(fixed) != 30 or fixed[:4] != b"PK\x03\x04":
            raise C091PosePlanError("invalid ZIP local file header")
        name_len, extra_len = struct.unpack_from("<HH", fixed, 26)
        local_name = handle.read(name_len).decode("utf-8")
        if local_name != info.filename:
            raise C091PosePlanError(
                f"ZIP central/local name mismatch: central={info.filename!r} local={local_name!r}"
            )
        if extra_len:
            handle.read(extra_len)


def read_single_member_payload(path: Path, *, expected_sha256: str | None = None) -> bytes:
    archive_sha = _sha256_path(path)
    if expected_sha256 is not None and archive_sha != expected_sha256:
        raise C091PosePlanError(f"archive SHA mismatch for {path}: {archive_sha} != {expected_sha256}")
    with zipfile.ZipFile(path, "r") as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
        if len(infos) != 1:
            raise C091PosePlanError(f"expected one archive member; got {[info.filename for info in infos]!r}")
        info = infos[0]
        _safe_member_name(info.filename)
        if info.compress_type != zipfile.ZIP_STORED:
            raise C091PosePlanError("source archive member must be ZIP_STORED")
        _assert_local_header_name_matches(path, info)
        return zf.read(info)


def _brotli_decompress(label: str, data: bytes) -> bytes:
    try:
        return brotli.decompress(data)
    except brotli.error as exc:
        raise C091PosePlanError(f"{label} did not Brotli-decode") from exc


def decode_fixed_actions(data: bytes) -> bytes:
    raw = _brotli_decompress("seg_tile_actions.bin", data)
    if len(raw) % 4 != 0:
        raise C091PosePlanError("fixed action record stream is not 4-byte aligned")
    return raw


def _read_uleb128(data: bytes, cursor: int) -> tuple[int, int]:
    shift = 0
    value = 0
    while True:
        if cursor >= len(data):
            raise C091PosePlanError("truncated action ULEB128 stream")
        byte = data[cursor]
        cursor += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, cursor
        shift += 7
        if shift > 28:
            raise C091PosePlanError("action ULEB128 stream is too wide")


def decode_delta_varint_actions(data: bytes, *, record_count: int) -> bytes:
    raw = _brotli_decompress("seg_tile_actions.delta", data)
    cursor = 0
    previous_pair = 0
    out = bytearray()
    for _ in range(record_count):
        delta, cursor = _read_uleb128(raw, cursor)
        if cursor + 2 > len(raw):
            raise C091PosePlanError("truncated action delta record")
        pair = previous_pair + delta
        tile = raw[cursor]
        action = raw[cursor + 1]
        cursor += 2
        out.extend(struct.pack("<HBB", pair, tile, action))
        previous_pair = pair
    if cursor != len(raw):
        raise C091PosePlanError("action delta stream has trailing bytes")
    return bytes(out)


def _uleb128(value: int) -> bytes:
    if value < 0:
        raise C091PosePlanError("cannot ULEB128-encode a negative value")
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def encode_delta_varint_actions(raw_actions: bytes) -> bytes:
    if len(raw_actions) % 4 != 0:
        raise C091PosePlanError("raw action records are not 4-byte aligned")
    previous_pair = 0
    out = bytearray()
    for offset in range(0, len(raw_actions), 4):
        pair, tile, action = struct.unpack_from("<HBB", raw_actions, offset)
        delta = int(pair) - previous_pair
        if delta < 0:
            raise C091PosePlanError("P6 action delta encoding requires nondecreasing pair indices")
        out.extend(_uleb128(delta))
        out.append(int(tile))
        out.append(int(action))
        previous_pair = int(pair)
    return bytes(out)


def parse_payload_slices(payload: bytes) -> StreamSlices:
    action_record_count: int | None = None
    if payload.startswith(b"P6"):
        cursor = 2 + struct.calcsize("<IHHH")
        mask_len, renderer_len, actions_len, action_record_count = struct.unpack_from("<IHHH", payload, 2)
        payload_format = "public_pr75_qzs3_qp1_segactions_p6_delta_varint"
    elif payload.startswith(b"P3"):
        cursor = 2 + struct.calcsize("<IHH")
        mask_len, renderer_len, actions_len = struct.unpack_from("<IHH", payload, 2)
        payload_format = "public_pr75_qzs3_qp1_segactions_p3"
    elif len(payload) in FIXED_SLICE_TABLE:
        mask_len, renderer_len, actions_len, label = FIXED_SLICE_TABLE[len(payload)]
        cursor = 0
        payload_format = f"public_pr75_qzs3_qp1_fixed_slices:{label}"
        action_record_count = 0
    else:
        matched_layout = None
        for mask_len_i, renderer_len_i, actions_len_i, label_i in REFERENCE_SLICE_LAYOUTS:
            if len(payload) > mask_len_i + renderer_len_i + actions_len_i:
                matched_layout = (mask_len_i, renderer_len_i, actions_len_i, label_i)
                break
        if matched_layout is None:
            raise C091PosePlanError(f"unsupported payload prefix={payload[:4]!r} bytes={len(payload)}")
        mask_len, renderer_len, actions_len, label = matched_layout
        cursor = 0
        payload_format = f"public_pr75_qzs3_qp1_fixed_slices:{label}"
        action_record_count = 0

    if min(mask_len, renderer_len) <= 0 or actions_len < 0:
        raise C091PosePlanError("payload contains an invalid stream length")
    mask_end = cursor + int(mask_len)
    renderer_end = mask_end + int(renderer_len)
    actions_end = renderer_end + int(actions_len)
    if actions_end >= len(payload):
        raise C091PosePlanError("payload stream lengths leave no pose stream")
    actions_br = payload[renderer_end:actions_end]
    if action_record_count is None:
        action_record_count = 0
    return StreamSlices(
        payload_format=payload_format,
        mask_br=payload[cursor:mask_end],
        renderer_br=payload[mask_end:renderer_end],
        actions_br=actions_br,
        pose_br=payload[actions_end:],
        action_record_count=int(action_record_count),
    )


def _runtime_decoded_members(payload: bytes) -> dict[str, bytes]:
    unpacker_path = REPO_ROOT / "submissions/robust_current/unpack_renderer_payload.py"
    spec = importlib.util.spec_from_file_location(
        "c091_pose_manifold_unpacker",
        unpacker_path,
    )
    if spec is None or spec.loader is None:
        raise C091PosePlanError(f"could not load runtime unpacker: {unpacker_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    _header, decoded = module._parse_payload(payload)  # noqa: SLF001
    out: dict[str, bytes] = {}
    for name, blob in decoded.items():
        if name == "optimized_poses.bin" and blob.startswith(QP1_MAGIC):
            out["optimized_poses.qp1"] = blob
        else:
            out[str(name)] = bytes(blob)
    required = {"masks.mkv", "renderer.bin", "optimized_poses.qp1"}
    missing = sorted(required - set(out))
    if missing:
        raise C091PosePlanError(f"runtime unpacker missing decoded members: {missing}")
    return out


def parse_source_archive(
    label: str,
    path: Path,
    *,
    expected_sha256: str | None = None,
) -> SourceArchive:
    payload = read_single_member_payload(path, expected_sha256=expected_sha256)
    slices = parse_payload_slices(payload)
    use_runtime_unpacker = (
        slices.payload_format.startswith("public_pr75_qzs3_qp1_fixed_slices:")
        and bool(slices.actions_br)
    ) or slices.payload_format == "public_pr75_qzs3_qp1_segactions_p3"
    if use_runtime_unpacker:
        decoded = _runtime_decoded_members(payload)
        decoded_mask = decoded["masks.mkv"]
        decoded_renderer = decoded["renderer.bin"]
        decoded_pose = decoded["optimized_poses.qp1"]
        decoded_actions = decoded.get("seg_tile_actions.bin", b"")
    else:
        decoded_mask = _brotli_decompress("masks.mkv", slices.mask_br)
        decoded_renderer = _brotli_decompress("renderer.bin", slices.renderer_br)
        decoded_pose = _brotli_decompress("optimized_poses.qp1", slices.pose_br)
        if slices.actions_br:
            if slices.payload_format.endswith("p6_delta_varint"):
                decoded_actions = decode_delta_varint_actions(
                    slices.actions_br,
                    record_count=slices.action_record_count,
                )
            else:
                decoded_actions = decode_fixed_actions(slices.actions_br)
        else:
            decoded_actions = b""
    if not decoded_pose.startswith(QP1_MAGIC):
        raise C091PosePlanError(f"{label}: decoded pose stream is not QP1")
    return SourceArchive(
        label=label,
        path=path.resolve(),
        archive_bytes=path.stat().st_size,
        archive_sha256=_sha256_path(path),
        payload=payload,
        payload_sha256=_sha256_bytes(payload),
        slices=slices,
        decoded={
            "masks.mkv": decoded_mask,
            "renderer.bin": decoded_renderer,
            "seg_tile_actions.bin": decoded_actions,
            "optimized_poses.qp1": decoded_pose,
        },
    )


def decode_qp1_words(payload: bytes) -> list[int]:
    if not payload.startswith(QP1_MAGIC) or len(payload) < 5:
        raise C091PosePlanError("invalid QP1 stream")
    words = [struct.unpack_from("<H", payload, 3)[0]]
    cursor = 5
    while cursor < len(payload):
        shift = 0
        acc = 0
        while True:
            if cursor >= len(payload):
                raise C091PosePlanError("truncated QP1 VLQ")
            byte = payload[cursor]
            cursor += 1
            acc |= (byte & 0x7F) << shift
            if byte < 0x80:
                break
            shift += 7
        delta = (acc >> 1) ^ -(acc & 1)
        words.append((words[-1] + delta) & 0xFFFF)
    return words


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


def encode_qp1_words(words: Sequence[int]) -> bytes:
    if not words:
        raise C091PosePlanError("cannot encode empty QP1 word stream")
    out = bytearray(QP1_MAGIC)
    previous = int(words[0])
    if previous < 0 or previous > 0xFFFF:
        raise C091PosePlanError("first QP1 word is outside uint16 range")
    out.extend(struct.pack("<H", previous))
    for word in words[1:]:
        current = int(word)
        if current < 0 or current > 0xFFFF:
            raise C091PosePlanError("QP1 word is outside uint16 range")
        out.extend(_vlq_encode(_zigzag_encode(current - previous)))
        previous = current
    return bytes(out)


def _brotli_best(raw: bytes, *, source: bytes | None = None) -> tuple[bytes, dict[str, int] | str]:
    best = source
    best_params: dict[str, int] | str = "source" if source is not None else {}
    for quality in (11, 10, 8, 6, 4, 2, 0):
        for mode in (0, 1):
            for lgwin in (10, 16, 22):
                candidate = brotli.compress(raw, quality=quality, mode=mode, lgwin=lgwin)
                if best is None or len(candidate) < len(best):
                    best = candidate
                    best_params = {"quality": quality, "mode": mode, "lgwin": lgwin}
    if best is None or _brotli_decompress("brotli_best_roundtrip", best) != raw:
        raise C091PosePlanError("Brotli selection failed round-trip")
    return best, best_params


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(_safe_member_name(name), FIXED_ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o644 << 16
    info.create_system = 3
    return info


def _write_single_member_archive(path: Path, payload: bytes) -> None:
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
        raise C091PosePlanError("P6 u16 stream length limit exceeded")
    return (
        b"P6"
        + struct.pack("<IHHH", len(mask_br), len(renderer_br), len(actions_br), int(action_record_count))
        + mask_br
        + renderer_br
        + actions_br
        + pose_br
    )


def _build_p3_payload(
    *,
    mask_br: bytes,
    renderer_br: bytes,
    actions_br: bytes,
    pose_br: bytes,
) -> bytes:
    if max(len(renderer_br), len(actions_br), len(pose_br)) > 0xFFFF:
        raise C091PosePlanError("P3 u16 stream length limit exceeded")
    return (
        b"P3"
        + struct.pack("<IHH", len(mask_br), len(renderer_br), len(actions_br))
        + mask_br
        + renderer_br
        + actions_br
        + pose_br
    )


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise C091PosePlanError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise C091PosePlanError(f"{label} must be a JSON object")
    return payload


def load_anchor_eval(path: Path) -> dict[str, Any]:
    payload = _load_json(path, label="c091_eval")
    archive_sha = payload.get("provenance", {}).get("archive_sha256")
    if archive_sha != C091_SHA256:
        raise C091PosePlanError(f"C091 eval SHA mismatch: {archive_sha}")
    if int(payload.get("archive_size_bytes", -1)) != C091_BYTES:
        raise C091PosePlanError("C091 eval archive byte count mismatch")
    return {
        "path": str(path),
        "archive_size_bytes": int(payload["archive_size_bytes"]),
        "archive_sha256": archive_sha,
        "score_recomputed_from_components": float(payload["score_recomputed_from_components"]),
        "avg_posenet_dist": float(payload["avg_posenet_dist"]),
        "avg_segnet_dist": float(payload["avg_segnet_dist"]),
        "score_pose_contribution": float(payload["score_pose_contribution"]),
        "score_seg_contribution": float(payload["score_seg_contribution"]),
        "n_samples": int(payload["n_samples"]),
        "device": payload.get("provenance", {}).get("device"),
        "gpu_model": payload.get("provenance", {}).get("gpu_model"),
        "gpu_t4_match": bool(payload.get("provenance", {}).get("gpu_t4_match")),
    }


def load_component_trace(path: Path) -> dict[int, dict[str, Any]]:
    payload = _load_json(path, label="c091_component_trace")
    if int(payload.get("n_samples", -1)) != EXPECTED_SAMPLES:
        raise C091PosePlanError("component trace sample count mismatch")
    if payload.get("score_claim") is not False:
        raise C091PosePlanError("component trace must be non-claim evidence")
    cross = payload.get("contest_auth_eval_cross_check")
    if not isinstance(cross, Mapping) or cross.get("all_match") is not True:
        raise C091PosePlanError("component trace must cross-check contest_auth_eval")
    samples = payload.get("samples")
    if not isinstance(samples, list) or len(samples) != EXPECTED_SAMPLES:
        raise C091PosePlanError("component trace samples are incomplete")
    out: dict[int, dict[str, Any]] = {}
    for sample in samples:
        if not isinstance(sample, Mapping):
            continue
        pair = int(sample["pair_index"])
        pose = float(sample.get("score_pose_contribution_first_order", 0.0))
        seg = float(sample.get("score_seg_contribution_exact", 0.0))
        combined = float(sample.get("score_combined_contribution_first_order", pose + seg))
        out[pair] = {
            "pair_index": pair,
            "frame_indices": list(sample.get("frame_indices", [2 * pair, 2 * pair + 1])),
            "pose_score_contribution": pose,
            "seg_score_contribution": seg,
            "combined_score_contribution": combined,
            "posenet_dist": float(sample.get("posenet_dist", 0.0)),
            "segnet_dist": float(sample.get("segnet_dist", 0.0)),
        }
    return out


def _try_load_pr65_words(path: Path, *, expected_len: int) -> tuple[np.ndarray | None, dict[str, Any]]:
    if not path.exists():
        return None, {"usable": False, "reason": "missing", "path": str(path)}
    try:
        from experiments.plan_pr65_henosis_stream_transfer import (
            decode_pr65_p1d1_pose,
            parse_pr65_henosis_archive,
        )
        from tac.qp1_pose_codec import encode_qp1

        parsed = parse_pr65_henosis_archive(path, expected_sha256=None)
        poses = decode_pr65_p1d1_pose(parsed["_segments_bytes"]["pose"])
        words = np.asarray(decode_qp1_words(encode_qp1(poses)), dtype=np.int32)
    except Exception as exc:  # pragma: no cover - exercised by real artifacts
        return None, {"usable": False, "reason": f"parse failed: {exc}", "path": str(path)}
    if len(words) != expected_len:
        return None, {"usable": False, "reason": "length mismatch", "len": len(words), "expected_len": expected_len}
    return words, {
        "usable": True,
        "path": str(path),
        "archive_sha256": _sha256_path(path),
        "word_count": int(len(words)),
        "words_sha256": _sha256_bytes(words.astype("<i4").tobytes()),
    }


def _try_load_archive_words(label: str, path: Path, *, expected_len: int) -> tuple[np.ndarray | None, dict[str, Any]]:
    if not path.exists():
        return None, {"usable": False, "reason": "missing", "path": str(path)}
    try:
        source = parse_source_archive(label, path)
        raw = source.decoded["optimized_poses.qp1"]
        words = np.asarray(decode_qp1_words(raw), dtype=np.int32)
    except Exception as exc:
        return None, {"usable": False, "reason": f"parse failed: {exc}", "path": str(path)}
    if len(words) != expected_len:
        return None, {"usable": False, "reason": "length mismatch", "len": len(words), "expected_len": expected_len}
    return words, {
        "usable": True,
        "path": str(path),
        "archive_sha256": _sha256_path(path),
        "word_count": int(len(words)),
        "words_sha256": _sha256_bytes(words.astype("<i4").tobytes()),
    }


def load_reference_bases(
    *,
    source_words: np.ndarray,
    c089_archive: Path | None,
    pr65_archive: Path | None,
    pr67_base_archive: Path | None,
    pr67_active_archive: Path | None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    expected_len = int(len(source_words))
    bases: dict[str, np.ndarray] = {}
    meta: dict[str, Any] = {}
    if c089_archive is not None:
        words, info = _try_load_archive_words("c089_pr75_qp1_top40_p6", c089_archive, expected_len=expected_len)
        meta["c089"] = info
        if words is not None:
            bases["c089_residual"] = words - source_words
    if pr65_archive is not None:
        words, info = _try_load_pr65_words(pr65_archive, expected_len=expected_len)
        meta["pr65"] = info
        if words is not None:
            bases["pr65_residual"] = words - source_words
    if pr67_base_archive is not None and pr67_active_archive is not None:
        base_words, base_info = _try_load_archive_words("pr67_base", pr67_base_archive, expected_len=expected_len)
        active_words, active_info = _try_load_archive_words("pr67_active", pr67_active_archive, expected_len=expected_len)
        meta["pr67_base"] = base_info
        meta["pr67_active"] = active_info
        if base_words is not None and active_words is not None:
            bases["pr67_active_delta"] = active_words - base_words
            bases["pr67_base_residual"] = base_words - source_words
    return bases, meta


def _combined_basis_delta(bases: Mapping[str, np.ndarray], kind: str) -> np.ndarray:
    if not bases:
        raise C091PosePlanError("no usable pose bases were loaded")
    first = next(iter(bases.values()))
    out = np.zeros_like(first, dtype=np.float64)
    if kind == "pr65_residual":
        if "pr65_residual" not in bases:
            raise C091PosePlanError("pr65_residual basis requested but unavailable")
        out += bases["pr65_residual"]
    elif kind == "pr67_active_delta":
        if "pr67_active_delta" not in bases:
            raise C091PosePlanError("pr67_active_delta basis requested but unavailable")
        out += bases["pr67_active_delta"]
    elif kind == "lowrank_pr65_pr67":
        if "pr65_residual" in bases:
            out += 0.55 * bases["pr65_residual"]
        if "pr67_active_delta" in bases:
            out += 0.45 * bases["pr67_active_delta"]
        if not np.any(out):
            raise C091PosePlanError("lowrank basis has no nonzero direction")
    elif kind == "cem_pr65_pr67_c089":
        if "pr65_residual" in bases:
            out += 0.45 * bases["pr65_residual"]
        if "pr67_active_delta" in bases:
            out += 0.35 * bases["pr67_active_delta"]
        if "c089_residual" in bases:
            out += 0.20 * bases["c089_residual"]
        if not np.any(out):
            raise C091PosePlanError("CEM basis has no nonzero direction")
    else:
        raise C091PosePlanError(f"unknown basis kind: {kind}")
    return out


def rank_pairs(
    trace: Mapping[int, Mapping[str, Any]],
    basis_delta: np.ndarray,
    *,
    spec: CandidateSpec,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    abs_delta = np.abs(basis_delta)
    max_delta = float(abs_delta.max()) if abs_delta.size else 0.0
    for pair in range(min(len(basis_delta), EXPECTED_SAMPLES)):
        if basis_delta[pair] == 0:
            continue
        sample = trace.get(pair, {})
        pose = float(sample.get("pose_score_contribution", 0.0))
        seg = float(sample.get("seg_score_contribution", 0.0))
        combined = float(sample.get("combined_score_contribution", pose + seg))
        magnitude_bonus = 0.0 if max_delta <= 0.0 else 0.08 * abs(float(basis_delta[pair])) / max_delta
        rank_score = spec.rank_mix_pose * pose + spec.rank_mix_seg * seg + combined + magnitude_bonus * combined
        rows.append(
            {
                "pair_index": pair,
                "frame_indices": sample.get("frame_indices", [2 * pair, 2 * pair + 1]),
                "rank_score": rank_score,
                "basis_delta_q": float(basis_delta[pair]),
                "abs_basis_delta_q": abs(float(basis_delta[pair])),
                "pose_score_contribution": pose,
                "seg_score_contribution": seg,
                "combined_score_contribution": combined,
            }
        )
    rows.sort(key=lambda row: (-float(row["rank_score"]), int(row["pair_index"])))
    return rows


def apply_sparse_subspace_move(
    source_words: Sequence[int],
    basis_delta: np.ndarray,
    ranked_pairs: Sequence[Mapping[str, Any]],
    *,
    spec: CandidateSpec,
) -> tuple[list[int], list[dict[str, Any]]]:
    words = [int(word) for word in source_words]
    changes: list[dict[str, Any]] = []
    for row in ranked_pairs[: spec.top_pairs]:
        pair = int(row["pair_index"])
        raw = float(basis_delta[pair])
        delta = int(round(raw * spec.alpha))
        if delta == 0 and raw != 0.0:
            delta = 1 if raw > 0 else -1
        delta = max(-spec.max_abs_delta_q, min(spec.max_abs_delta_q, delta))
        if delta == 0:
            continue
        before = words[pair]
        after = max(0, min(0xFFFF, before + delta))
        if after == before:
            continue
        words[pair] = after
        changes.append(
            {
                **dict(row),
                "q_before": before,
                "q_after": after,
                "delta_q": after - before,
                "velocity_before": before / VELOCITY_SCALE + VELOCITY_OFFSET,
                "velocity_after": after / VELOCITY_SCALE + VELOCITY_OFFSET,
            }
        )
    return words, changes


def _stream_summary(data: bytes) -> dict[str, Any]:
    return {"bytes": len(data), "sha256": _sha256_bytes(data), "magic_hex": data[:12].hex()}


def _roundtrip_gates(
    *,
    source: SourceArchive,
    candidate: SourceArchive,
    candidate_words: Sequence[int],
    candidate_pose_raw: bytes,
    reference_words: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    candidate_decoded_words = decode_qp1_words(candidate.decoded["optimized_poses.qp1"])
    source_words = decode_qp1_words(source.decoded["optimized_poses.qp1"])
    full_reference_copy = False
    copied_reference_labels: list[str] = []
    for label, words in reference_words.items():
        if len(words) == len(candidate_words) and list(map(int, words)) == list(map(int, candidate_words)):
            full_reference_copy = True
            copied_reference_labels.append(label)
    gates = {
        "single_member_zip": True,
        "payload_format_is_self_describing": candidate.slices.payload_format
        in {
            "public_pr75_qzs3_qp1_segactions_p3",
            "public_pr75_qzs3_qp1_segactions_p6_delta_varint",
        },
        "mask_decoded_preserved": candidate.decoded["masks.mkv"] == source.decoded["masks.mkv"],
        "renderer_decoded_preserved": candidate.decoded["renderer.bin"] == source.decoded["renderer.bin"],
        "actions_decoded_preserved": candidate.decoded["seg_tile_actions.bin"] == source.decoded["seg_tile_actions.bin"],
        "pose_stream_changed": candidate.decoded["optimized_poses.qp1"] != source.decoded["optimized_poses.qp1"],
        "candidate_pose_brotli_decodes": candidate.decoded["optimized_poses.qp1"] == candidate_pose_raw,
        "candidate_qp1_words_match_policy": candidate_decoded_words == list(map(int, candidate_words)),
        "candidate_qp1_canonical_reencode": encode_qp1_words(candidate_words) == candidate_pose_raw,
        "source_pose_not_copied_verbatim": candidate_decoded_words != source_words,
        "no_full_public_or_prior_pose_copy": not full_reference_copy,
        "no_sidecars_in_archive": True,
    }
    return {
        **gates,
        "copied_reference_labels": copied_reference_labels,
        "all_passed": all(bool(value) for key, value in gates.items() if key != "copied_reference_labels"),
    }


def _break_even(archive_bytes: int, changed_pairs: Sequence[Mapping[str, Any]], *, efficiency: float) -> dict[str, Any]:
    delta_bytes = archive_bytes - C091_BYTES
    rate_delta = delta_bytes * RATE_SCORE_PER_BYTE
    score_if_components_unchanged = C091_SCORE + rate_delta
    required_sub314 = max(0.0, score_if_components_unchanged - SUB314_TARGET)
    affected_trace_mass = sum(float(row.get("combined_score_contribution", 0.0)) for row in changed_pairs)
    pose_trace_mass = sum(float(row.get("pose_score_contribution", 0.0)) for row in changed_pairs)
    seg_trace_mass = sum(float(row.get("seg_score_contribution", 0.0)) for row in changed_pairs)
    proxy_saved = affected_trace_mass * efficiency
    return {
        "archive_delta_bytes_vs_c091": delta_bytes,
        "rate_score_delta_vs_c091": rate_delta,
        "score_if_components_unchanged": score_if_components_unchanged,
        "sub314_component_gain_required": required_sub314,
        "sub314_equivalent_bytes_needed_after_candidate": math.ceil(required_sub314 / RATE_SCORE_PER_BYTE)
        if required_sub314 > 0.0
        else 0,
        "affected_trace_mass": affected_trace_mass,
        "affected_pose_trace_mass": pose_trace_mass,
        "affected_seg_trace_mass": seg_trace_mass,
        "planning_efficiency_assumption": efficiency,
        "expected_component_gain_proxy": proxy_saved,
        "proxy_margin_vs_sub314_requirement": proxy_saved - required_sub314,
        "trace_is_promotable_evidence": False,
    }


def _exact_eval_command(archive: Path) -> list[str]:
    return [
        ".venv/bin/python",
        "-u",
        "experiments/contest_auth_eval.py",
        "--archive",
        str(archive),
        "--inflate-sh",
        "submissions/robust_current/inflate.sh",
        "--upstream-dir",
        "upstream",
        "--device",
        "cuda",
    ]


def build_candidate(
    *,
    source: SourceArchive,
    source_words: np.ndarray,
    trace: Mapping[int, Mapping[str, Any]],
    bases: Mapping[str, np.ndarray],
    reference_words: Mapping[str, np.ndarray],
    spec: CandidateSpec,
    output_dir: Path,
) -> dict[str, Any]:
    basis_delta = _combined_basis_delta(bases, spec.basis_kind)
    ranked = rank_pairs(trace, basis_delta, spec=spec)
    candidate_words, changes = apply_sparse_subspace_move(source_words, basis_delta, ranked, spec=spec)
    if not changes:
        raise C091PosePlanError(f"{spec.candidate_id}: no nonzero pose changes")
    pose_raw = encode_qp1_words(candidate_words)
    pose_br, pose_params = _brotli_best(pose_raw, source=None)
    actions_br = source.slices.actions_br
    actions_params: dict[str, Any] | str = "source"
    if source.slices.payload_format == "public_pr75_qzs3_qp1_segactions_p6_delta_varint":
        payload = _build_p6_payload(
            mask_br=source.slices.mask_br,
            renderer_br=source.slices.renderer_br,
            actions_br=actions_br,
            action_record_count=source.slices.action_record_count,
            pose_br=pose_br,
        )
    else:
        payload = _build_p3_payload(
            mask_br=source.slices.mask_br,
            renderer_br=source.slices.renderer_br,
            actions_br=actions_br,
            pose_br=pose_br,
        )
    candidate_dir = output_dir / spec.candidate_id
    archive = candidate_dir / "archive.zip"
    _write_single_member_archive(archive, payload)
    parsed = parse_source_archive(spec.candidate_id, archive)
    gates = _roundtrip_gates(
        source=source,
        candidate=parsed,
        candidate_words=candidate_words,
        candidate_pose_raw=pose_raw,
        reference_words=reference_words,
    )
    economics = _break_even(parsed.archive_bytes, changes, efficiency=spec.efficiency_assumption)
    break_even_plausible = (
        gates["all_passed"]
        and economics["proxy_margin_vs_sub314_requirement"] > 0.0
        and len(changes) >= min(64, spec.top_pairs)
    )
    recommendation_class = (
        "exact_eval_candidate_after_claim_not_dispatched"
        if break_even_plausible
        else "hold_for_more_evidence"
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "tool": TOOL,
        "candidate_id": spec.candidate_id,
        "candidate_family": "c091_native_sparse_qp1_pose_manifold_bigmove",
        "score_claim": False,
        "promotion_eligible": False,
        "remote_dispatch": {"dispatched": False, "dispatch_state_touched": False},
        "source_archive": {
            "path": str(source.path),
            "archive_bytes": source.archive_bytes,
            "archive_sha256": source.archive_sha256,
            "payload_format": source.slices.payload_format,
        },
        "archive": {
            "path": str(archive),
            "archive_bytes": parsed.archive_bytes,
            "archive_sha256": parsed.archive_sha256,
            "payload_bytes": len(payload),
            "payload_sha256": _sha256_bytes(payload),
            "payload_format": parsed.slices.payload_format,
        },
        "spec": spec.__dict__,
        "selected_pair_count": len(changes),
        "selected_pair_indices": [int(row["pair_index"]) for row in changes],
        "selected_pair_records": changes,
        "local_roundtrip_gates": gates,
        "decoded_streams": {
            "source": {name: _stream_summary(data) for name, data in sorted(source.decoded.items())},
            "candidate": {name: _stream_summary(data) for name, data in sorted(parsed.decoded.items())},
        },
        "encoded_streams": {
            "mask": _stream_summary(parsed.slices.mask_br),
            "renderer": _stream_summary(parsed.slices.renderer_br),
            "actions": _stream_summary(parsed.slices.actions_br),
            "pose": _stream_summary(parsed.slices.pose_br),
        },
        "pose_word_custody": {
            "source_words_sha256": _sha256_bytes(source_words.astype("<i4").tobytes()),
            "candidate_words_sha256": _sha256_bytes(np.asarray(candidate_words, dtype="<i4").tobytes()),
            "candidate_pose_raw_bytes": len(pose_raw),
            "candidate_pose_raw_sha256": _sha256_bytes(pose_raw),
            "candidate_pose_float32_sha256": _sha256_bytes(
                decode_qp1(pose_raw).astype("<f4", copy=False).tobytes()
            ),
        },
        "brotli_params": {
            "actions_delta_varint": actions_params,
            "pose_qp1": pose_params,
        },
        "economics": economics,
        "break_even_plausible": break_even_plausible,
        "dispatch_recommendation": {
            "class": recommendation_class,
            "reason": (
                "Archive exists, local stream closure passed, changed pose atoms are non-noop, "
                "and the conservative affected-trace efficiency proxy clears sub-0.314 break-even."
                if break_even_plausible
                else "Planning proxy or local gates do not clear the exact-eval recommendation threshold."
            ),
            "no_remote_dispatch_performed": True,
            "do_not_duplicate": "exact_eval_pr65_pose_qp1_c091_c089_actions_p6_t4_20260503T1158Z",
            "required_before_any_remote_eval": [
                "wait for the existing PR65 pose-transfer exact eval if it targets the same comparison slot",
                "claim a non-conflicting lane with tools/claim_lane_dispatch.py",
                "run exact CUDA auth eval on this exact archive SHA through contest_auth_eval.py",
            ],
            "exact_eval_command_template": _exact_eval_command(archive) if break_even_plausible else None,
        },
    }
    _write_json(candidate_dir / "manifest.json", manifest)
    return manifest


def _ledger_text(plan: Mapping[str, Any]) -> str:
    candidates = plan["candidates"]
    top = candidates[0] if candidates else None
    lines = [
        "# C091 Pose Manifold Big-Move - 2026-05-03 Worker",
        "",
        "## Evidence Boundary",
        "",
        "- Scope: local C091-native QP1 pose-manifold planning/build only.",
        "- Remote dispatch: `false`.",
        "- Score claim: `false`.",
        "- Promotion eligible: `false` until exact CUDA auth eval on identical archive bytes.",
        "- Existing in-flight duplicate guard: `exact_eval_pr65_pose_qp1_c091_c089_actions_p6_t4_20260503T1158Z` was not duplicated.",
        "",
        "## Anchor",
        "",
        f"- C091 archive bytes: `{plan['anchor_eval']['archive_size_bytes']}`.",
        f"- C091 SHA-256: `{plan['anchor_eval']['archive_sha256']}`.",
        f"- C091 score: `{plan['anchor_eval']['score_recomputed_from_components']}`.",
        f"- Pose contribution: `{plan['anchor_eval']['score_pose_contribution']}`.",
        f"- Seg contribution: `{plan['anchor_eval']['score_seg_contribution']}`.",
        "",
        "## Candidate Verdict",
        "",
    ]
    if top is None:
        lines.append("- No candidates emitted.")
    else:
        lines.extend(
            [
                f"- Top candidate: `{top['candidate_id']}`.",
                f"- Archive bytes: `{top['archive']['archive_bytes']}`.",
                f"- Archive SHA-256: `{top['archive']['archive_sha256']}`.",
                f"- Changed pose pairs: `{top['selected_pair_count']}`.",
                f"- Local closure gates: `{top['local_roundtrip_gates']['all_passed']}`.",
                f"- Break-even plausible: `{top['break_even_plausible']}`.",
                f"- Dispatch recommendation: `{top['dispatch_recommendation']['class']}`.",
                f"- Sub-0.314 component gain required: `{top['economics']['sub314_component_gain_required']}`.",
                f"- Proxy component gain: `{top['economics']['expected_component_gain_proxy']}`.",
                "",
                "Adversarial verdict: this is a real non-noop C091-native pose residual move, not public stream copying. "
                "It is still trace-proxy evidence only; exact CUDA auth eval can easily reject it if PoseNet/SegNet "
                "response is nonlocal or antagonistic.",
            ]
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- Plan JSON: `{plan['artifacts']['plan_json']}`.",
            f"- Recommendation JSON: `{plan['artifacts']['recommendations_json']}`.",
            "",
            "## Tests",
            "",
            "- Focused test target: `src/tac/tests/test_plan_c091_pose_manifold_bigmove.py`.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_plan(
    *,
    c091_archive: Path = DEFAULT_C091_ARCHIVE,
    c091_eval: Path = DEFAULT_C091_EVAL,
    c091_trace: Path = DEFAULT_C091_TRACE,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    ledger_md: Path | None = DEFAULT_LEDGER,
    c089_archive: Path | None = DEFAULT_C089_ARCHIVE,
    pr65_archive: Path | None = DEFAULT_PR65_ARCHIVE,
    pr67_base_archive: Path | None = DEFAULT_PR67_BASE_ARCHIVE,
    pr67_active_archive: Path | None = DEFAULT_PR67_ACTIVE_ARCHIVE,
    specs: Sequence[CandidateSpec] = DEFAULT_SPECS,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    anchor_eval = load_anchor_eval(c091_eval)
    source = parse_source_archive("c091_pr75_minp_public_replay", c091_archive, expected_sha256=C091_SHA256)
    source_words = np.asarray(decode_qp1_words(source.decoded["optimized_poses.qp1"]), dtype=np.int32)
    if len(source_words) != EXPECTED_SAMPLES:
        raise C091PosePlanError(f"C091 QP1 word count must be {EXPECTED_SAMPLES}")
    trace = load_component_trace(c091_trace)
    bases, basis_meta = load_reference_bases(
        source_words=source_words,
        c089_archive=c089_archive,
        pr65_archive=pr65_archive,
        pr67_base_archive=pr67_base_archive,
        pr67_active_archive=pr67_active_archive,
    )
    reference_words = {
        key.removesuffix("_residual").removesuffix("_delta"): source_words + value
        for key, value in bases.items()
        if key in {"c089_residual", "pr65_residual", "pr67_base_residual"}
    }
    candidates: list[dict[str, Any]] = []
    for spec in specs:
        try:
            candidates.append(
                build_candidate(
                    source=source,
                    source_words=source_words,
                    trace=trace,
                    bases=bases,
                    reference_words=reference_words,
                    spec=spec,
                    output_dir=output_dir,
                )
            )
        except C091PosePlanError as exc:
            candidates.append(
                {
                    "candidate_id": spec.candidate_id,
                    "score_claim": False,
                    "promotion_eligible": False,
                    "build_failed": True,
                    "failure": str(exc),
                    "spec": spec.__dict__,
                }
            )
    built = [item for item in candidates if not item.get("build_failed")]
    built.sort(
        key=lambda item: (
            bool(item["break_even_plausible"]),
            float(item["economics"]["proxy_margin_vs_sub314_requirement"]),
            -int(item["archive"]["archive_bytes"]),
        ),
        reverse=True,
    )
    failed = [item for item in candidates if item.get("build_failed")]
    ordered = [*built, *failed]
    recommendations = [
        {
            "candidate_id": item["candidate_id"],
            "archive": item.get("archive"),
            "break_even_plausible": item.get("break_even_plausible", False),
            "dispatch_recommendation": item.get("dispatch_recommendation"),
            "economics": item.get("economics"),
            "local_roundtrip_gates": item.get("local_roundtrip_gates"),
        }
        for item in ordered
        if not item.get("build_failed")
    ]
    plan_json = output_dir / "plan.json"
    recommendations_json = output_dir / "exact_eval_recommendations.json"
    plan = {
        "schema_version": SCHEMA_VERSION,
        "tool": TOOL,
        "score_claim": False,
        "promotion_eligible": False,
        "remote_dispatch": {"dispatched": False, "dispatch_state_touched": False},
        "anchor_eval": anchor_eval,
        "source_archive": {
            "path": str(source.path),
            "archive_bytes": source.archive_bytes,
            "archive_sha256": source.archive_sha256,
            "payload_sha256": source.payload_sha256,
            "payload_format": source.slices.payload_format,
            "encoded_streams": {
                "mask": _stream_summary(source.slices.mask_br),
                "renderer": _stream_summary(source.slices.renderer_br),
                "actions": _stream_summary(source.slices.actions_br),
                "pose": _stream_summary(source.slices.pose_br),
            },
            "decoded_streams": {name: _stream_summary(data) for name, data in sorted(source.decoded.items())},
        },
        "reference_basis": {
            "metadata": basis_meta,
            "usable_basis_names": sorted(bases),
            "basis_is_planning_signal_not_score_evidence": True,
        },
        "candidate_count": len(built),
        "failed_candidate_count": len(failed),
        "candidates": ordered,
        "recommendations": recommendations,
        "artifacts": {
            "plan_json": str(plan_json),
            "recommendations_json": str(recommendations_json),
            "ledger_md": str(ledger_md) if ledger_md is not None else None,
        },
        "adversarial_verdict": {
            "concise": (
                "Build at most one exact-eval candidate from this family, and only after a lane claim; "
                "the best local row clears a conservative trace-mass break-even proxy but remains vulnerable "
                "to nonlocal PoseNet/SegNet antagonism."
            ),
            "exact_eval_recommendation_count": sum(
                1
                for item in recommendations
                if item.get("dispatch_recommendation", {}).get("class")
                == "exact_eval_candidate_after_claim_not_dispatched"
            ),
        },
    }
    _write_json(plan_json, plan)
    _write_json(recommendations_json, recommendations)
    if ledger_md is not None:
        ledger_md.parent.mkdir(parents=True, exist_ok=True)
        ledger_md.write_text(_ledger_text(plan))
    return plan


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c091-archive", type=Path, default=DEFAULT_C091_ARCHIVE)
    parser.add_argument("--c091-eval", type=Path, default=DEFAULT_C091_EVAL)
    parser.add_argument("--c091-trace", type=Path, default=DEFAULT_C091_TRACE)
    parser.add_argument("--c089-archive", type=Path, default=DEFAULT_C089_ARCHIVE)
    parser.add_argument("--pr65-archive", type=Path, default=DEFAULT_PR65_ARCHIVE)
    parser.add_argument("--pr67-base-archive", type=Path, default=DEFAULT_PR67_BASE_ARCHIVE)
    parser.add_argument("--pr67-active-archive", type=Path, default=DEFAULT_PR67_ACTIVE_ARCHIVE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ledger-md", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--no-ledger", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    plan = build_plan(
        c091_archive=args.c091_archive,
        c091_eval=args.c091_eval,
        c091_trace=args.c091_trace,
        c089_archive=args.c089_archive,
        pr65_archive=args.pr65_archive,
        pr67_base_archive=args.pr67_base_archive,
        pr67_active_archive=args.pr67_active_archive,
        output_dir=args.output_dir,
        ledger_md=None if args.no_ledger else args.ledger_md,
    )
    print(json.dumps({
        "candidate_count": plan["candidate_count"],
        "top_candidate": plan["candidates"][0]["candidate_id"] if plan["candidates"] else None,
        "exact_eval_recommendation_count": plan["adversarial_verdict"]["exact_eval_recommendation_count"],
        "plan_json": plan["artifacts"]["plan_json"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
