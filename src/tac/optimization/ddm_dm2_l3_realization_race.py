# SPDX-License-Identifier: MIT
"""Bounded L3 RGB-realization race for the SHA-bound DDM DM1 demand rows.

The scorer target is never shipped.  It is used offline to select a counted
camera-space RGB delta which is replayed through the existing exact factor-2
resize preimage and the frozen scorers.  This module emits research-only local
CPU evidence; it does not call ``evaluate.py`` or emit a candidate archive.
"""

from __future__ import annotations

import json
import lzma
import math
import os
import struct
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np

from tac.optimization.ddm_dm1_solved_value_pricing import (
    SolvedValueRecord,
    _class_id,
    _class_pair_from_bucket,
    _event_support,
    _stratum_from_bucket,
    _winner_symbols,
    canonical_json_bytes,
    checked_json,
    sha256_bytes,
)
from tac.optimization.ddm_min_description_contract import LayerHome, StreamType
from tac.optimization.resize_full_kernel import FullResizeKernel
from tac.optimization.solve_diff_operator_mining import (
    SolveDiffMiningConfigV1,
    _load_production_inputs,
    _open_production_inputs,
    realize_solve_camera,
    sha256_file,
)

SCHEMA = "ddm_dm2_l3_realization_race.v1"
ROW_SCHEMA = "ddm_dm2_l3_realization_row.v1"
CONFIG_SCHEMA = "ddm_dm2_l3_realization_race_config.v1"
RGB_RECORD_MAGIC = b"DM2RGB1\0"
RGB_JOINT_MAGIC = b"DM2JNT1\0"
CODEC_MAGIC = b"DM2COD1\0"
CODECS = ("zlib9", "lzma9")
AXIS = "[macOS-CPU frozen-scorer advisory]"
POINTER = "0.1910828242 [contest-CPU] UNMOVED"
CAMERA_HW = (874, 1164)
SCORER_HW = (384, 512)
FRAME_INDEX = 1
_REPO = Path(__file__).resolve().parents[3]
_RGB_HEADER = struct.Struct("<8sHBI")
_JOINT_HEADER = struct.Struct("<8sH")
_LENGTH = struct.Struct("<I")
_CODEC_HEADER = struct.Struct("<8sBQ32s")


class DM2RealizationError(ValueError):
    """Raised on custody, parseback, geometry, or false-authority drift."""


def _encode_varint(value: int) -> bytes:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DM2RealizationError("varint requires a nonnegative integer")
    output = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        output.append(byte | (0x80 if value else 0))
        if not value:
            return bytes(output)


def _decode_varint(payload: bytes, offset: int) -> tuple[int, int]:
    value = 0
    shift = 0
    cursor = offset
    while cursor < len(payload) and shift <= 63:
        byte = payload[cursor]
        cursor += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, cursor
        shift += 7
    raise DM2RealizationError("RGB record contains a truncated/oversized varint")


@dataclass(frozen=True)
class RGBDeltaRecord:
    """Exact sparse camera RGB delta relative to one SHA-bound base pair."""

    pair_id: int
    frame_index: int
    flat_indices: tuple[int, ...]
    deltas: tuple[tuple[int, int, int], ...]

    def __post_init__(self) -> None:
        if not 0 <= self.pair_id <= 599 or self.frame_index not in (0, 1):
            raise DM2RealizationError("RGB record pair/frame key is out of range")
        if len(self.flat_indices) != len(self.deltas):
            raise DM2RealizationError("RGB record coordinate/delta counts differ")
        if tuple(sorted(set(self.flat_indices))) != self.flat_indices:
            raise DM2RealizationError("RGB record coordinates must be unique canonical order")
        if any(not 0 <= index < CAMERA_HW[0] * CAMERA_HW[1] for index in self.flat_indices):
            raise DM2RealizationError("RGB record coordinate escapes the camera plane")
        if any(
            len(delta) != 3
            or delta == (0, 0, 0)
            or any(isinstance(value, bool) or not -255 <= value <= 255 for value in delta)
            for delta in self.deltas
        ):
            raise DM2RealizationError("RGB record contains an invalid signed RGB delta")

    @classmethod
    def from_frames(
        cls,
        pair_id: int,
        before: np.ndarray,
        after: np.ndarray,
        *,
        frame_index: int = FRAME_INDEX,
    ) -> RGBDeltaRecord:
        left = np.asarray(before)
        right = np.asarray(after)
        if (
            left.dtype != np.uint8
            or right.dtype != np.uint8
            or left.shape != (*CAMERA_HW, 3)
            or right.shape != left.shape
        ):
            raise DM2RealizationError("RGB record frames must be camera-shaped uint8 HWC")
        delta = right.astype(np.int16) - left.astype(np.int16)
        changed = np.flatnonzero(np.any(delta != 0, axis=2))
        values = delta.reshape(-1, 3)[changed]
        return cls(
            int(pair_id),
            int(frame_index),
            tuple(int(value) for value in changed),
            tuple(tuple(int(channel) for channel in row) for row in values),
        )

    def encode(self) -> bytes:
        output = bytearray(
            _RGB_HEADER.pack(
                RGB_RECORD_MAGIC,
                self.pair_id,
                self.frame_index,
                len(self.flat_indices),
            )
        )
        previous = -1
        for index, delta in zip(self.flat_indices, self.deltas, strict=True):
            output.extend(_encode_varint(index - previous - 1))
            output.extend(struct.pack("<hhh", *delta))
            previous = index
        return bytes(output)

    @classmethod
    def decode(cls, payload: bytes) -> RGBDeltaRecord:
        if len(payload) < _RGB_HEADER.size:
            raise DM2RealizationError("RGB record header is truncated")
        magic, pair_id, frame_index, count = _RGB_HEADER.unpack_from(payload)
        if magic != RGB_RECORD_MAGIC:
            raise DM2RealizationError("RGB record magic differs")
        cursor = _RGB_HEADER.size
        indices: list[int] = []
        deltas: list[tuple[int, int, int]] = []
        previous = -1
        for _ in range(count):
            gap, cursor = _decode_varint(payload, cursor)
            index = previous + gap + 1
            if cursor + 6 > len(payload):
                raise DM2RealizationError("RGB record delta is truncated")
            delta = struct.unpack_from("<hhh", payload, cursor)
            cursor += 6
            indices.append(index)
            deltas.append(tuple(int(value) for value in delta))
            previous = index
        if cursor != len(payload):
            raise DM2RealizationError("RGB record has trailing bytes")
        record = cls(pair_id, frame_index, tuple(indices), tuple(deltas))
        if record.encode() != payload:
            raise DM2RealizationError("RGB record parse/re-encode differs")
        return record

    def apply(self, base: np.ndarray) -> np.ndarray:
        frame = np.asarray(base)
        if frame.dtype != np.uint8 or frame.shape != (*CAMERA_HW, 3):
            raise DM2RealizationError("RGB record base must be camera-shaped uint8")
        output = frame.astype(np.int16)
        if self.flat_indices:
            flat = output.reshape(-1, 3)
            flat[np.asarray(self.flat_indices, dtype=np.intp)] += np.asarray(
                self.deltas, dtype=np.int16
            )
        if np.any(output < 0) or np.any(output > 255):
            raise DM2RealizationError("RGB record application leaves the uint8 lattice")
        return output.astype(np.uint8)


def encode_joint_rgb_records(records: Sequence[RGBDeltaRecord]) -> bytes:
    ordered = tuple(sorted(records, key=lambda row: (row.pair_id, row.frame_index)))
    if len({(row.pair_id, row.frame_index) for row in ordered}) != len(ordered):
        raise DM2RealizationError("joint RGB record contains duplicate pair/frame keys")
    output = bytearray(_JOINT_HEADER.pack(RGB_JOINT_MAGIC, len(ordered)))
    for record in ordered:
        raw = record.encode()
        output.extend(_LENGTH.pack(len(raw)))
        output.extend(raw)
    return bytes(output)


def decode_joint_rgb_records(payload: bytes) -> tuple[RGBDeltaRecord, ...]:
    if len(payload) < _JOINT_HEADER.size:
        raise DM2RealizationError("joint RGB record header is truncated")
    magic, count = _JOINT_HEADER.unpack_from(payload)
    if magic != RGB_JOINT_MAGIC:
        raise DM2RealizationError("joint RGB record magic differs")
    cursor = _JOINT_HEADER.size
    rows = []
    for _ in range(count):
        if cursor + _LENGTH.size > len(payload):
            raise DM2RealizationError("joint RGB record length is truncated")
        length = _LENGTH.unpack_from(payload, cursor)[0]
        cursor += _LENGTH.size
        stop = cursor + length
        if stop > len(payload):
            raise DM2RealizationError("joint RGB record member is truncated")
        rows.append(RGBDeltaRecord.decode(payload[cursor:stop]))
        cursor = stop
    if cursor != len(payload):
        raise DM2RealizationError("joint RGB record has trailing bytes")
    result = tuple(rows)
    if encode_joint_rgb_records(result) != payload:
        raise DM2RealizationError("joint RGB record parse/re-encode differs")
    return result


def _codec_payload(raw: bytes, codec: str) -> bytes:
    if codec == "zlib9":
        return zlib.compress(raw, level=9)
    if codec == "lzma9":
        return lzma.compress(raw, preset=9)
    raise DM2RealizationError(f"unsupported DM2 coder {codec!r}")


def encode_coded_rgb(raw: bytes, codec: str) -> bytes:
    compressed = _codec_payload(raw, codec)
    codec_id = CODECS.index(codec)
    return _CODEC_HEADER.pack(CODEC_MAGIC, codec_id, len(raw), sha256(raw).digest()) + compressed


def decode_coded_rgb(container: bytes) -> tuple[str, bytes]:
    if len(container) < _CODEC_HEADER.size:
        raise DM2RealizationError("coded RGB container header is truncated")
    magic, codec_id, raw_length, expected_digest = _CODEC_HEADER.unpack_from(container)
    if magic != CODEC_MAGIC or codec_id >= len(CODECS):
        raise DM2RealizationError("coded RGB container header differs")
    codec = CODECS[codec_id]
    compressed = container[_CODEC_HEADER.size :]
    try:
        raw = zlib.decompress(compressed) if codec == "zlib9" else lzma.decompress(compressed)
    except (zlib.error, lzma.LZMAError) as exc:
        raise DM2RealizationError("coded RGB payload is malformed") from exc
    if len(raw) != raw_length or sha256(raw).digest() != expected_digest:
        raise DM2RealizationError("coded RGB payload custody differs")
    if encode_coded_rgb(raw, codec) != container:
        raise DM2RealizationError("coded RGB parse/re-encode differs")
    return codec, raw


def price_rgb_raw(raw: bytes) -> tuple[dict[str, dict[str, Any]], str]:
    prices: dict[str, dict[str, Any]] = {}
    for codec in CODECS:
        container = encode_coded_rgb(raw, codec)
        parsed_codec, parsed = decode_coded_rgb(container)
        prices[codec] = {
            "container_bytes": len(container),
            "container_sha256": sha256_bytes(container),
            "payload_bytes": len(container) - _CODEC_HEADER.size,
            "parseback_exact": parsed_codec == codec and parsed == raw,
        }
    winner = min(CODECS, key=lambda name: (prices[name]["container_bytes"], CODECS.index(name)))
    return prices, winner


def dilated_support_mask(flat_indices: np.ndarray, radius: int) -> np.ndarray:
    flat = np.asarray(flat_indices)
    if flat.ndim != 1 or flat.dtype.kind not in "iu" or len(flat) == 0:
        raise DM2RealizationError("support must be a nonempty flat integer vector")
    if isinstance(radius, bool) or not isinstance(radius, int) or radius < 0:
        raise DM2RealizationError("support radius must be a nonnegative integer")
    if np.any(flat < 0) or np.any(flat >= SCORER_HW[0] * SCORER_HW[1]):
        raise DM2RealizationError("support escapes the scorer plane")
    mask = np.zeros(SCORER_HW, dtype=bool)
    for index in np.unique(flat.astype(np.int64)):
        row, col = divmod(int(index), SCORER_HW[1])
        mask[
            max(0, row - radius) : min(SCORER_HW[0], row + radius + 1),
            max(0, col - radius) : min(SCORER_HW[1], col + radius + 1),
        ] = True
    return mask


def candidate_scorer_plane(
    base: np.ndarray,
    target: np.ndarray,
    support: np.ndarray,
    *,
    scope: str,
    radius: int | None,
    quantum: int | None,
) -> np.ndarray:
    """Construct one exact integer scorer-plane candidate.

    ``quantum=None`` is exact target substitution.  Otherwise the target
    direction is clipped to the named fixed integer quantum; no sub-uint8 step
    is created.
    """

    left = np.asarray(base)
    right = np.asarray(target)
    if (
        left.dtype != np.uint8
        or right.dtype != np.uint8
        or left.shape != (*SCORER_HW, 3)
        or right.shape != left.shape
    ):
        raise DM2RealizationError("candidate planes must be scorer-shaped uint8 RGB")
    if scope == "local":
        if radius is None:
            raise DM2RealizationError("local candidate requires a support radius")
        mask = dilated_support_mask(support, radius)
    elif scope == "global":
        if radius is not None:
            raise DM2RealizationError("global candidate cannot carry a support radius")
        mask = np.ones(SCORER_HW, dtype=bool)
    else:
        raise DM2RealizationError("candidate scope must be local or global")
    delta = right.astype(np.int16) - left.astype(np.int16)
    if quantum is not None:
        if isinstance(quantum, bool) or not isinstance(quantum, int) or not 1 <= quantum <= 255:
            raise DM2RealizationError("candidate quantum must be an integer in [1,255]")
        delta = np.clip(delta, -quantum, quantum)
    output = left.astype(np.int16)
    output[mask] += delta[mask]
    return np.clip(output, 0, 255).astype(np.uint8)


def _seg_forward(segnet: Any, camera_pair: np.ndarray) -> np.ndarray:
    import torch

    pair = np.asarray(camera_pair)
    if pair.dtype != np.uint8 or pair.shape != (2, *CAMERA_HW, 3):
        raise DM2RealizationError("SegNet input must be one uint8 camera pair")
    tensor = torch.from_numpy(pair[None]).permute(0, 1, 4, 2, 3).float()
    with torch.inference_mode():
        logits = segnet(segnet.preprocess_input(tensor))
    if tuple(logits.shape) != (1, 5, *SCORER_HW):
        raise DM2RealizationError(f"SegNet logits geometry differs: {tuple(logits.shape)}")
    return logits[0].cpu().numpy().astype(np.float32)


def _pose_forward(posenet: Any, camera_pair: np.ndarray) -> np.ndarray:
    import torch

    pair = np.asarray(camera_pair)
    if pair.dtype != np.uint8 or pair.shape != (2, *CAMERA_HW, 3):
        raise DM2RealizationError("PoseNet input must be one uint8 camera pair")
    tensor = torch.from_numpy(pair[None]).permute(0, 1, 4, 2, 3).float()
    with torch.inference_mode():
        pose = posenet(posenet.preprocess_input(tensor))["pose"][0, :6]
    output = pose.cpu().numpy().astype(np.float64)
    if output.shape != (6,) or not np.all(np.isfinite(output)):
        raise DM2RealizationError("PoseNet first-six output geometry differs")
    return output


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise DM2RealizationError(f"refusing to overwrite unequal artifact: {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _read_config(config_path: str | Path) -> tuple[dict[str, Any], bytes]:
    raw = Path(config_path).read_bytes()
    config = json.loads(raw)
    if config.get("schema") != CONFIG_SCHEMA:
        raise DM2RealizationError("DM2 config schema differs")
    required = {
        "research_only": True,
        "execution_allowed": False,
        "score_claim": False,
        "promotion_eligible": False,
        "archive_emitted": False,
        "pointer_moved": False,
        "pointer": POINTER,
        "evidence_axis": AXIS,
        "main_review_required": True,
    }
    if any(config.get(key) != value for key, value in required.items()):
        raise DM2RealizationError("DM2 false-authority contract differs")
    if config.get("torch_threads") != 4 or config.get("row_count") != 25:
        raise DM2RealizationError("DM2 thread/row bound differs")
    return config, raw


def _bound_inputs(config: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, Any], SolveDiffMiningConfigV1]:
    for path_key, sha_key in (
        ("authority_file", "authority_sha256"),
        ("dm1_receipt_path", "dm1_receipt_sha256"),
        ("dm1_config_path", "dm1_config_sha256"),
        ("source_config_path", "source_config_sha256"),
        ("start_receipt_path", "start_receipt_sha256"),
        ("segnet_weights_path", "segnet_weights_sha256"),
        ("posenet_weights_path", "posenet_weights_sha256"),
        ("upstream_modules_path", "upstream_modules_sha256"),
    ):
        if sha256_file(config[path_key]) != config[sha_key]:
            raise DM2RealizationError(f"{path_key} SHA-256 mismatch")
    dm1 = checked_json(config["dm1_receipt_path"], config["dm1_receipt_sha256"])
    dm1_config = checked_json(config["dm1_config_path"], config["dm1_config_sha256"])
    source_config = SolveDiffMiningConfigV1.model_validate_json(
        Path(config["source_config_path"]).read_bytes()
    )
    if (
        dm1.get("schema") != "ddm_dm1_solved_value_pricing.v1"
        or dm1.get("row_count") != 25
        or len(dm1.get("rows", ())) != 25
        or dm1.get("joint_shared_context", {}).get("exact_counted_bytes")
        != config["semantic_joint_bytes"]
    ):
        raise DM2RealizationError("DM1 receipt binding differs")
    if dm1_config.get("source_config_sha256") != config["source_config_sha256"]:
        raise DM2RealizationError("DM1-to-source-config binding differs")
    for key in (
        "pf2_event_index_path",
        "pf2_event_index_sha256",
        "pf2_index_receipt_path",
        "pf2_index_receipt_sha256",
    ):
        if dm1_config.get(key) != config.get(key):
            raise DM2RealizationError(f"DM2 {key} differs from DM1 custody")
    if sha256_file(config["pf2_event_index_path"]) != config["pf2_event_index_sha256"]:
        raise DM2RealizationError("PF2 event-index SHA-256 mismatch")
    if sha256_file(config["pf2_index_receipt_path"]) != config["pf2_index_receipt_sha256"]:
        raise DM2RealizationError("PF2 index-receipt SHA-256 mismatch")
    index_receipt = checked_json(
        config["pf2_index_receipt_path"], config["pf2_index_receipt_sha256"]
    )
    if index_receipt.get("index_sha256") != config["pf2_event_index_sha256"]:
        raise DM2RealizationError("PF2 receipt-to-index binding differs")
    return dm1, index_receipt, source_config


def _candidate_specs(
    config: Mapping[str, Any],
    *,
    first_local_success_radius: int | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [
        {"scope": "local", "radius": int(radius), "quantum": None}
        for radius in config["local_radius_ladder"]
    ]
    if first_local_success_radius is not None:
        rows.extend(
            {
                "scope": "local",
                "radius": first_local_success_radius,
                "quantum": int(quantum),
            }
            for quantum in config["fixed_quantum_ladder"]
        )
    rows.extend(
        {"scope": "global", "radius": None, "quantum": int(quantum)}
        for quantum in config["fixed_quantum_ladder"]
    )
    rows.append({"scope": "global", "radius": None, "quantum": None})
    unique: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for row in rows:
        key = (row["scope"], row["radius"], row["quantum"])
        if key not in seen:
            seen.add(key)
            unique.append(row)
    return unique


def _candidate_id(spec: Mapping[str, Any]) -> str:
    radius = "all" if spec["radius"] is None else f"r{spec['radius']}"
    quantum = "target" if spec["quantum"] is None else f"q{spec['quantum']}"
    return f"{spec['scope']}_{radius}_{quantum}"


def _expected_record(
    row: Mapping[str, Any],
    logits: np.ndarray,
    support: np.ndarray,
) -> tuple[bytes, bytes]:
    left_name, right_name = _class_pair_from_bucket(str(row["bucket_id"]))
    left, right = _class_id(left_name), _class_id(right_name)
    winners, relations, _summary = _winner_symbols(logits, support, left, right)
    stratum = _stratum_from_bucket(str(row["bucket_id"]))
    record = SolvedValueRecord(
        pair_id=int(row["pair_id"]),
        bucket_id=str(row["bucket_id"]),
        class_left=left,
        class_right=right,
        stream_type=StreamType.SKELETON if stratum == "boundary" else StreamType.FIBER,
        layer_home=LayerHome.L4_SCORER_FEATURE,
        support_sha256=str(row["support"]["sha256_uint32le"]),
        winners=winners,
        margin_relations=relations,
        flat_indices=tuple(int(value) for value in support) if stratum == "boundary" else (),
    )
    if sha256_bytes(record.encode()) != row["semantic_record"]["raw_sha256"]:
        raise DM2RealizationError("recomputed target semantic record differs from DM1")
    return winners, relations


def _score_delta(
    *,
    delta_errors: int,
    delta_pose_sse: float,
    realized_bytes: int,
    config: Mapping[str, Any],
) -> dict[str, float]:
    sites = int(config["global_seg_sites"])
    pose_coordinates = int(config["global_pose_coordinates"])
    base_pose = float(config["base_global_d_pose"])
    delta_d_seg = delta_errors / sites
    delta_d_pose = delta_pose_sse / pose_coordinates
    pose_after = base_pose + delta_d_pose
    if pose_after < 0.0:
        raise DM2RealizationError("local Pose delta makes global d_pose negative")
    seg_score = 100.0 * delta_d_seg
    pose_score = math.sqrt(10.0 * pose_after) - math.sqrt(10.0 * base_pose)
    rate_score = 25.0 * realized_bytes / int(config["source_video_bytes"])
    return {
        "delta_d_seg": delta_d_seg,
        "delta_d_pose": delta_d_pose,
        "seg_score_delta": seg_score,
        "pose_score_delta": pose_score,
        "rate_score_delta": rate_score,
        "joint_score_delta": seg_score + pose_score + rate_score,
    }


def _measure_row(
    *,
    row: Mapping[str, Any],
    event_index: Any,
    index_receipt: Mapping[str, Any],
    context: Any,
    source_config: SolveDiffMiningConfigV1,
    kernel: FullResizeKernel,
    segnet: Any,
    posenet: Any,
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    pair_id = int(row["pair_id"])
    chunk = _load_production_inputs(context, source_config, [pair_id], kernel)
    base_planes = chunk.predictor_planes[0]
    target_planes = chunk.solved_planes[0]
    base_camera = np.stack(
        [realize_solve_camera(base_planes[index], kernel) for index in range(2)]
    )
    target_camera = np.stack(
        [realize_solve_camera(target_planes[index], kernel) for index in range(2)]
    )
    base_logits = _seg_forward(segnet, base_camera)
    target_logits = _seg_forward(segnet, target_camera)
    bucket_id = str(row["bucket_id"])
    array_key = index_receipt["bucket_arrays"].get(bucket_id)
    if not isinstance(array_key, str):
        raise DM2RealizationError(f"PF2 receipt lacks bucket mapping for {bucket_id}")
    support = _event_support(event_index, bucket_id, array_key, pair_id)
    if str(row["support"]["sha256_uint32le"]) != sha256(
        np.asarray(support, dtype="<u4").tobytes()
    ).hexdigest():
        raise DM2RealizationError("PF2 support SHA differs from DM1")
    left_name, right_name = _class_pair_from_bucket(bucket_id)
    left, right = _class_id(left_name), _class_id(right_name)
    expected = _expected_record(row, target_logits, support)
    base_symbols = _winner_symbols(base_logits, support, left, right)[:2]
    base_cells = np.argmax(base_logits, axis=0).astype(np.uint8)
    gt_labels = np.asarray(chunk.labels[0], dtype=np.uint8)
    gt_pose = np.asarray(chunk.poses[0], dtype=np.float64)
    base_pose = _pose_forward(posenet, base_camera)
    base_pose_sse = float(np.square(base_pose - gt_pose).sum(dtype=np.float64))

    search_rows: list[dict[str, Any]] = []
    successful: list[dict[str, Any]] = []
    seen_planes: set[str] = set()
    first_local_success_radius: int | None = None

    initial_specs = [
        {"scope": "local", "radius": int(radius), "quantum": None}
        for radius in config["local_radius_ladder"]
    ]
    for phase, specs in (
        ("local_target", initial_specs),
        ("remaining", []),
    ):
        if phase == "remaining":
            specs = _candidate_specs(
                config, first_local_success_radius=first_local_success_radius
            )
        for spec in specs:
            candidate_id = _candidate_id(spec)
            candidate_plane = candidate_scorer_plane(
                base_planes[FRAME_INDEX],
                target_planes[FRAME_INDEX],
                support,
                scope=str(spec["scope"]),
                radius=spec["radius"],
                quantum=spec["quantum"],
            )
            plane_sha = sha256(candidate_plane.tobytes()).hexdigest()
            if plane_sha in seen_planes:
                continue
            seen_planes.add(plane_sha)
            candidate_camera = base_camera.copy()
            candidate_camera[FRAME_INDEX] = realize_solve_camera(candidate_plane, kernel)
            candidate_logits = _seg_forward(segnet, candidate_camera)
            observed = _winner_symbols(candidate_logits, support, left, right)[:2]
            semantic_exact = observed == expected
            changed_cells = int(
                np.count_nonzero(candidate_plane != base_planes[FRAME_INDEX])
            )
            candidate_cells = np.argmax(candidate_logits, axis=0).astype(np.uint8)
            label_flips = candidate_cells != base_cells
            search = {
                "candidate_id": candidate_id,
                "scope": spec["scope"],
                "radius": spec["radius"],
                "quantum_u8": spec["quantum"],
                "semantic_record_exact": semantic_exact,
                "changed_scorer_channel_values": changed_cells,
                "argmax_flips_vs_base": int(np.count_nonzero(label_flips)),
            }
            search_rows.append(search)
            if semantic_exact:
                if spec["scope"] == "local" and first_local_success_radius is None:
                    first_local_success_radius = int(spec["radius"])
                record = RGBDeltaRecord.from_frames(
                    pair_id, base_camera[FRAME_INDEX], candidate_camera[FRAME_INDEX]
                )
                raw = record.encode()
                if record.apply(base_camera[FRAME_INDEX]).tobytes() != candidate_camera[
                    FRAME_INDEX
                ].tobytes():
                    raise DM2RealizationError("RGB record did not reconstruct candidate")
                prices, winner = price_rgb_raw(raw)
                delta = candidate_camera[FRAME_INDEX].astype(np.int16) - base_camera[
                    FRAME_INDEX
                ].astype(np.int16)
                successful.append(
                    {
                        "candidate_id": candidate_id,
                        "spec": dict(spec),
                        "candidate_plane": candidate_plane,
                        "candidate_camera": candidate_camera,
                        "candidate_cells": candidate_cells,
                        "record": record,
                        "record_raw": raw,
                        "prices": prices,
                        "winning_codec": winner,
                        "exact_counted_bytes": prices[winner]["container_bytes"],
                        "changed_rgb_pixels": len(record.flat_indices),
                        "changed_channel_values": int(np.count_nonzero(delta)),
                        "l1_rgb_delta": int(np.abs(delta).sum(dtype=np.int64)),
                        "l2_rgb_delta": float(
                            np.sqrt(np.square(delta, dtype=np.float64).sum())
                        ),
                    }
                )
        if phase == "local_target" and first_local_success_radius is None:
            continue

    if not successful:
        raise DM2RealizationError("full target positive control failed semantic exactness")
    ranked = sorted(
        successful,
        key=lambda item: (
            int(item["exact_counted_bytes"]),
            int(item["changed_rgb_pixels"]),
            int(item["l1_rgb_delta"]),
            str(item["candidate_id"]),
        ),
    )
    pose_screen = []
    for candidate in ranked[: int(config["pose_selection_candidates"])]:
        pose = _pose_forward(posenet, candidate["candidate_camera"])
        pose_sse = float(np.square(pose - gt_pose).sum(dtype=np.float64))
        candidate["pose6"] = pose
        candidate["pose_sse"] = pose_sse
        candidate["delta_pose_sse"] = pose_sse - base_pose_sse
        pose_screen.append(
            {
                "candidate_id": candidate["candidate_id"],
                "exact_counted_bytes": candidate["exact_counted_bytes"],
                "delta_pair_pose_mse": (pose_sse - base_pose_sse) / 6.0,
                "pose_nonharm": pose_sse <= base_pose_sse,
            }
        )
    nonharm = [candidate for candidate in ranked[: len(pose_screen)] if candidate["pose_sse"] <= base_pose_sse]
    selected = nonharm[0] if nonharm else ranked[0]
    if "pose6" not in selected:
        pose = _pose_forward(posenet, selected["candidate_camera"])
        selected["pose6"] = pose
        selected["pose_sse"] = float(np.square(pose - gt_pose).sum(dtype=np.float64))
        selected["delta_pose_sse"] = selected["pose_sse"] - base_pose_sse

    support_mask = np.zeros(SCORER_HW, dtype=bool)
    support_mask.reshape(-1)[support] = True
    candidate_cells = selected["candidate_cells"]
    changed = candidate_cells != base_cells
    outside = ~support_mask
    base_correct = base_cells == gt_labels
    candidate_correct = candidate_cells == gt_labels
    harmful = int(np.count_nonzero(outside & base_correct & ~candidate_correct))
    helpful = int(np.count_nonzero(outside & ~base_correct & candidate_correct))
    neutral = int(
        np.count_nonzero(outside & changed & ~base_correct & ~candidate_correct)
    )
    delta_errors = int(np.count_nonzero(~candidate_correct) - np.count_nonzero(~base_correct))
    off_target_delta_errors = int(
        np.count_nonzero(outside & ~candidate_correct)
        - np.count_nonzero(outside & ~base_correct)
    )
    score = _score_delta(
        delta_errors=delta_errors,
        delta_pose_sse=float(selected["delta_pose_sse"]),
        realized_bytes=int(selected["exact_counted_bytes"]),
        config=config,
    )
    collateral_seg_score = 100.0 * off_target_delta_errors / int(config["global_seg_sites"])
    collateral_score = collateral_seg_score + score["pose_score_delta"]
    collateral_byte_equivalent = (
        max(0.0, collateral_score)
        * int(config["source_video_bytes"])
        / 25.0
    )
    semantic_bytes = int(row["exact_counted_bytes"])
    effective_cost = float(selected["exact_counted_bytes"]) + collateral_byte_equivalent
    stream = str(row["adjudicated_typed_home"]["type"])
    outcome = {
        "schema": ROW_SCHEMA,
        "row_index": int(row["row_index"]),
        "pair_id": pair_id,
        "bucket_id": bucket_id,
        "stream_type": stream,
        "stratum": row["stratum"],
        "support_count_n": len(support),
        "support_sha256_uint32le": row["support"]["sha256_uint32le"],
        "semantic_record_sha256": row["semantic_record"]["raw_sha256"],
        "semantic_bytes_dm1": semantic_bytes,
        "base_semantic_record_exact": base_symbols == expected,
        "realization_status": "SUCCESS_EXACT_L4_RECORD_THROUGH_L3_RGB",
        "selected_candidate": {
            "candidate_id": selected["candidate_id"],
            **selected["spec"],
            "application_stage": "camera_874x1164_pre_R_exact_factor2_integer_preimage",
            "quantization_policy": "integer fixed quantum or exact target substitution",
        },
        "search": {
            "candidate_count": len(search_rows),
            "semantic_success_count": len(successful),
            "first_local_success_radius": first_local_success_radius,
            "local_target_success": first_local_success_radius is not None,
            "rows": search_rows,
            "verdict_scope": (
                "best measured member of the preregistered local-radius/fixed-quantum "
                "candidate menu; not a global minimum-preimage certificate"
            ),
        },
        "rgb_record": {
            "raw_bytes": len(selected["record_raw"]),
            "raw_sha256": sha256_bytes(selected["record_raw"]),
            "parseback_exact": True,
            "prices": selected["prices"],
            "winning_codec": selected["winning_codec"],
            "exact_counted_bytes": selected["exact_counted_bytes"],
            "changed_rgb_pixels": selected["changed_rgb_pixels"],
            "changed_channel_values": selected["changed_channel_values"],
            "l1_rgb_delta": selected["l1_rgb_delta"],
            "l2_rgb_delta_euclidean_control": selected["l2_rgb_delta"],
        },
        "collateral": {
            "off_target_argmax_flips": int(np.count_nonzero(outside & changed)),
            "harmful_off_target_flips": harmful,
            "helpful_off_target_flips": helpful,
            "neutral_wrong_to_wrong_off_target_flips": neutral,
            "off_target_delta_errors": off_target_delta_errors,
            "seg_score_delta": collateral_seg_score,
            "pose_score_delta": score["pose_score_delta"],
            "joint_collateral_score_delta": collateral_score,
            "positive_collateral_byte_equivalent_at_rate_dual": collateral_byte_equivalent,
        },
        "pose": {
            "base_pose6": base_pose.tolist(),
            "candidate_pose6": selected["pose6"].tolist(),
            "gt_pose6": gt_pose.tolist(),
            "base_pair_mse": base_pose_sse / 6.0,
            "candidate_pair_mse": selected["pose_sse"] / 6.0,
            "delta_pair_mse": selected["delta_pose_sse"] / 6.0,
            "pose_nonharm": selected["pose_sse"] <= base_pose_sse,
            "selection_screen": pose_screen,
            "metric": "exact frozen PoseNet first-six-output MSE",
        },
        "joint_score_accounting": score,
        "ratio": {
            "realized_bytes_per_semantic_byte": selected["exact_counted_bytes"] / semantic_bytes,
            "effective_realized_plus_positive_collateral_bytes": effective_cost,
            "effective_bytes_per_semantic_byte": effective_cost / semantic_bytes,
        },
        "first_rung": (
            "Fit a corrected-inner-Jacobian or compact parabolic shearlet support proposal "
            "and rerun this exact hard admission before interpreting the global write."
            if selected["spec"]["scope"] == "global"
            else "Compose this exact local RGB record with the other successful rows on the pair and remeasure non-telescoping Seg/Pose survival."
        ),
        "verdict_scope": (
            "INSTANCE x exact DM1 row x preregistered factor2 local/global fixed-quantum "
            "menu; no formulation, family, paradigm, score, or promotion verdict"
        ),
        "evidence_axis": AXIS,
        "score_claim": False,
        "pointer": POINTER,
    }
    internal = {
        "row": row,
        "support": support,
        "selected_spec": selected["spec"],
    }
    return outcome, internal


def _implementation_custody() -> dict[str, dict[str, Any]]:
    paths = (
        Path(__file__).resolve(),
        _REPO / "tools/measure_ddm_dm2_l3_realization_race.py",
        _REPO / "src/tac/optimization/ddm_dm1_solved_value_pricing.py",
        _REPO / "src/tac/optimization/resize_full_kernel.py",
        _REPO / "src/tac/optimization/uint8_lattice_feasibility.py",
        _REPO / "src/tac/scorer.py",
    )
    output = {}
    for path in paths:
        if not path.is_file():
            raise DM2RealizationError(f"implementation custody path absent: {path}")
        output[str(path.relative_to(_REPO))] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return output


def _summarize_rows(rows: Sequence[Mapping[str, Any]], stream_type: str) -> dict[str, Any]:
    selected = [row for row in rows if row["stream_type"] == stream_type]
    realized = [int(row["rgb_record"]["exact_counted_bytes"]) for row in selected]
    semantic = [int(row["semantic_bytes_dm1"]) for row in selected]
    effective = [
        float(row["ratio"]["effective_realized_plus_positive_collateral_bytes"])
        for row in selected
    ]
    return {
        "row_count": len(selected),
        "semantic_bytes_independent_sum": sum(semantic),
        "realized_bytes_independent_sum": sum(realized),
        "realized_per_semantic_ratio_of_sums": sum(realized) / sum(semantic),
        "effective_per_semantic_ratio_of_sums": sum(effective) / sum(semantic),
        "local_selected_rows": sum(
            row["selected_candidate"]["scope"] == "local" for row in selected
        ),
        "global_selected_rows": sum(
            row["selected_candidate"]["scope"] == "global" for row in selected
        ),
        "pose_nonharm_rows": sum(row["pose"]["pose_nonharm"] for row in selected),
    }


def _summarize_n(rows: Sequence[Mapping[str, Any]], edges: Sequence[int]) -> list[dict[str, Any]]:
    bounds = tuple(int(value) for value in edges)
    if not bounds or bounds != tuple(sorted(set(bounds))) or bounds[0] != 1:
        raise DM2RealizationError("support n-bin edges must start at 1 and increase")
    output = []
    for index, lo in enumerate(bounds):
        hi = bounds[index + 1] if index + 1 < len(bounds) else None
        selected = [
            row
            for row in rows
            if int(row["support_count_n"]) >= lo
            and (hi is None or int(row["support_count_n"]) < hi)
        ]
        if not selected:
            continue
        semantic = sum(int(row["semantic_bytes_dm1"]) for row in selected)
        realized = sum(int(row["rgb_record"]["exact_counted_bytes"]) for row in selected)
        output.append(
            {
                "n_interval": f"[{lo},{hi})" if hi is not None else f"[{lo},inf)",
                "row_count": len(selected),
                "semantic_bytes_independent_sum": semantic,
                "realized_bytes_independent_sum": realized,
                "realized_per_semantic_ratio_of_sums": realized / semantic,
            }
        )
    return output


def materialize(config_path: str | Path, output_dir: str | Path) -> Mapping[str, Any]:
    """Run or resume the bounded 25-row measurement and emit a small receipt."""

    config, config_raw = _read_config(config_path)
    dm1, index_receipt, source_config = _bound_inputs(config)
    kernel = FullResizeKernel.build()
    context = _open_production_inputs(source_config)

    import torch

    from tac.scorer import load_default_scorers

    torch.set_num_threads(4)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(int(config["seed"]))
    torch.use_deterministic_algorithms(True)
    posenet, segnet = load_default_scorers(config["upstream_dir"], device="cpu")

    root = Path(output_dir)
    stage = root / "stage_checkpoints"
    config_sha = sha256_bytes(config_raw)
    implementation_sha = sha256_file(Path(__file__).resolve())
    accepted_checkpoint_custody = {(config_sha, implementation_sha)}
    for prior in config.get("accepted_row_checkpoint_custody", ()):
        if not isinstance(prior, Mapping):
            raise DM2RealizationError("accepted row-checkpoint custody must be mappings")
        prior_config_sha = prior.get("typed_config_sha256")
        prior_implementation_sha = prior.get("implementation_sha256")
        if not (
            isinstance(prior_config_sha, str)
            and len(prior_config_sha) == 64
            and isinstance(prior_implementation_sha, str)
            and len(prior_implementation_sha) == 64
            and prior.get("scope") == "row_measurement_only_before_joint_composition_fix"
        ):
            raise DM2RealizationError("accepted row-checkpoint custody is malformed")
        accepted_checkpoint_custody.add(
            (prior_config_sha, prior_implementation_sha)
        )
    row_checkpoint_custody_used: set[tuple[str, str]] = set()
    output_rows: list[Mapping[str, Any]] = []
    internals: list[Mapping[str, Any]] = []
    with np.load(config["pf2_event_index_path"], allow_pickle=False) as event_index:
        for row in dm1["rows"]:
            row_index = int(row["row_index"])
            checkpoint = stage / f"row_{row_index:02d}.json"
            if checkpoint.is_file():
                resumed = json.loads(checkpoint.read_bytes())
                if (
                    resumed.get("schema")
                    != "ddm_dm2_l3_realization_row_checkpoint.v1"
                    or (
                        resumed.get("typed_config_sha256"),
                        resumed.get("implementation_sha256"),
                    )
                    not in accepted_checkpoint_custody
                    or resumed.get("row", {}).get("row_index") != row_index
                ):
                    raise DM2RealizationError(
                        f"row checkpoint custody differs: {checkpoint}"
                    )
                row_checkpoint_custody_used.add(
                    (
                        str(resumed["typed_config_sha256"]),
                        str(resumed["implementation_sha256"]),
                    )
                )
                bucket_id = str(row["bucket_id"])
                array_key = index_receipt["bucket_arrays"].get(bucket_id)
                if not isinstance(array_key, str):
                    raise DM2RealizationError(
                        f"PF2 receipt lacks bucket mapping for {bucket_id}"
                    )
                support = _event_support(
                    event_index, bucket_id, array_key, int(row["pair_id"])
                )
                output_rows.append(resumed["row"])
                selected = resumed["row"]["selected_candidate"]
                internals.append(
                    {
                        "row": row,
                        "support": support,
                        "selected_spec": {
                            "scope": selected["scope"],
                            "radius": selected["radius"],
                            "quantum": selected["quantum"],
                        },
                    }
                )
                continue
            measured, internal = _measure_row(
                row=row,
                event_index=event_index,
                index_receipt=index_receipt,
                context=context,
                source_config=source_config,
                kernel=kernel,
                segnet=segnet,
                posenet=posenet,
                config=config,
            )
            checkpoint_payload = canonical_json_bytes(
                {
                    "schema": "ddm_dm2_l3_realization_row_checkpoint.v1",
                    "typed_config_sha256": config_sha,
                    "implementation_sha256": implementation_sha,
                    "row": measured,
                }
            )
            _atomic_write(checkpoint, checkpoint_payload)
            row_checkpoint_custody_used.add((config_sha, implementation_sha))
            output_rows.append(measured)
            internals.append(internal)

    # Non-telescoping joint composition: merge every independently selected
    # movement by taking the farther same-direction movement toward the one
    # shared solved plane, then remeasure exact Seg/Pose on each affected pair.
    by_pair: dict[int, list[Mapping[str, Any]]] = {}
    for internal in internals:
        by_pair.setdefault(int(internal["row"]["pair_id"]), []).append(internal)
    joint_records: list[RGBDeltaRecord] = []
    joint_delta_errors = 0
    joint_delta_pose_sse = 0.0
    joint_off_target_delta_errors = 0
    joint_semantics_exact = True
    joint_pair_rows = []
    joint_fallback_pairs: list[int] = []
    with np.load(config["pf2_event_index_path"], allow_pickle=False) as event_index:
        for pair_id, pair_rows in sorted(by_pair.items()):
            chunk = _load_production_inputs(context, source_config, [pair_id], kernel)
            base_planes = chunk.predictor_planes[0]
            target_planes = chunk.solved_planes[0]
            base_camera = np.stack(
                [realize_solve_camera(base_planes[index], kernel) for index in range(2)]
            )
            joint_plane = base_planes[FRAME_INDEX].copy()
            progress = np.zeros_like(joint_plane, dtype=np.int16)
            union_support = np.zeros(SCORER_HW, dtype=bool)
            for internal in pair_rows:
                support = np.asarray(internal["support"], dtype=np.uint32)
                spec = internal["selected_spec"]
                candidate = candidate_scorer_plane(
                    base_planes[FRAME_INDEX],
                    target_planes[FRAME_INDEX],
                    support,
                    scope=str(spec["scope"]),
                    radius=spec["radius"],
                    quantum=spec["quantum"],
                )
                movement = candidate.astype(np.int16) - base_planes[FRAME_INDEX].astype(np.int16)
                take = np.abs(movement) > np.abs(progress)
                joint_plane[take] = candidate[take]
                progress[take] = movement[take]
                union_support.reshape(-1)[support] = True
            candidate_camera = base_camera.copy()
            candidate_camera[FRAME_INDEX] = realize_solve_camera(joint_plane, kernel)
            base_logits = _seg_forward(segnet, base_camera)
            candidate_logits = _seg_forward(segnet, candidate_camera)
            target_camera = np.stack(
                [realize_solve_camera(target_planes[index], kernel) for index in range(2)]
            )
            target_logits = _seg_forward(segnet, target_camera)
            def semantics_exact(
                logits: np.ndarray,
                scoped_pair_rows: Sequence[Mapping[str, Any]] = pair_rows,
                exact_target_logits: np.ndarray = target_logits,
            ) -> bool:
                exact = True
                for internal in scoped_pair_rows:
                    row = internal["row"]
                    support = np.asarray(internal["support"], dtype=np.uint32)
                    left_name, right_name = _class_pair_from_bucket(
                        str(row["bucket_id"])
                    )
                    left, right = _class_id(left_name), _class_id(right_name)
                    expected = _expected_record(row, exact_target_logits, support)
                    observed = _winner_symbols(logits, support, left, right)[:2]
                    exact &= observed == expected
                return exact

            initial_pair_exact = semantics_exact(candidate_logits)
            composition_mode = "farthest_same_direction_union"
            if not initial_pair_exact:
                # Independent row preimages can interfere after union. Preserve
                # the failed composition result as a measured fact, then fall
                # back to the already-bound solved target plane. This provides
                # an exact constructive upper bound, not a minimum certificate.
                candidate_camera[FRAME_INDEX] = target_camera[FRAME_INDEX]
                candidate_logits = target_logits
                composition_mode = (
                    "full_solved_target_positive_control_after_union_conflict"
                )
                joint_fallback_pairs.append(pair_id)
            pair_exact = semantics_exact(candidate_logits)
            if not pair_exact:
                raise DM2RealizationError(
                    f"full-target positive control failed semantics for pair {pair_id}"
                )
            joint_semantics_exact &= pair_exact
            base_cells = np.argmax(base_logits, axis=0)
            candidate_cells = np.argmax(candidate_logits, axis=0)
            labels = np.asarray(chunk.labels[0])
            base_errors = base_cells != labels
            candidate_errors = candidate_cells != labels
            pair_delta_errors = int(
                np.count_nonzero(candidate_errors) - np.count_nonzero(base_errors)
            )
            off_target = ~union_support
            pair_off_target_delta = int(
                np.count_nonzero(off_target & candidate_errors)
                - np.count_nonzero(off_target & base_errors)
            )
            base_pose = _pose_forward(posenet, base_camera)
            candidate_pose = _pose_forward(posenet, candidate_camera)
            gt_pose = np.asarray(chunk.poses[0], dtype=np.float64)
            pair_pose_sse_delta = float(
                np.square(candidate_pose - gt_pose).sum(dtype=np.float64)
                - np.square(base_pose - gt_pose).sum(dtype=np.float64)
            )
            record = RGBDeltaRecord.from_frames(
                pair_id, base_camera[FRAME_INDEX], candidate_camera[FRAME_INDEX]
            )
            if record.apply(base_camera[FRAME_INDEX]).tobytes() != candidate_camera[
                FRAME_INDEX
            ].tobytes():
                raise DM2RealizationError("joint RGB record parseback differs")
            joint_records.append(record)
            joint_delta_errors += pair_delta_errors
            joint_off_target_delta_errors += pair_off_target_delta
            joint_delta_pose_sse += pair_pose_sse_delta
            joint_pair_rows.append(
                {
                    "pair_id": pair_id,
                    "row_indices": [int(item["row"]["row_index"]) for item in pair_rows],
                    "composition_mode": composition_mode,
                    "initial_union_semantic_records_exact": initial_pair_exact,
                    "semantic_records_exact": pair_exact,
                    "delta_errors": pair_delta_errors,
                    "off_target_delta_errors": pair_off_target_delta,
                    "delta_pose_sse_6d": pair_pose_sse_delta,
                    "changed_rgb_pixels": len(record.flat_indices),
                    "rgb_record_raw_sha256": sha256_bytes(record.encode()),
                }
            )
    joint_raw = encode_joint_rgb_records(joint_records)
    if decode_joint_rgb_records(joint_raw) != tuple(sorted(joint_records, key=lambda row: (row.pair_id, row.frame_index))):
        raise DM2RealizationError("joint RGB record parseback differs")
    joint_prices, joint_winner = price_rgb_raw(joint_raw)
    joint_bytes = int(joint_prices[joint_winner]["container_bytes"])
    joint_score = _score_delta(
        delta_errors=joint_delta_errors,
        delta_pose_sse=joint_delta_pose_sse,
        realized_bytes=joint_bytes,
        config=config,
    )
    joint_collateral_seg = (
        100.0 * joint_off_target_delta_errors / int(config["global_seg_sites"])
    )
    joint_collateral_score = joint_collateral_seg + joint_score["pose_score_delta"]
    joint_collateral_bytes = (
        max(0.0, joint_collateral_score)
        * int(config["source_video_bytes"])
        / 25.0
    )
    semantic_joint = int(config["semantic_joint_bytes"])
    effective_joint = joint_bytes + joint_collateral_bytes

    result = {
        "schema": SCHEMA,
        "run_id": config["run_id"],
        "lane_id": config["lane_id"],
        "source_commit": config["source_commit"],
        "config_path": str(config_path),
        "config_sha256": config_sha,
        "row_count": len(output_rows),
        "rows": output_rows,
        "aggregate": {
            "semantic_records_joint_exact_after_composition": joint_semantics_exact,
            "semantic_bytes_dm1_joint": semantic_joint,
            "realized_rgb_joint": {
                "raw_bytes": len(joint_raw),
                "raw_sha256": sha256_bytes(joint_raw),
                "record_count": len(joint_records),
                "parseback_exact": True,
                "prices": joint_prices,
                "winning_codec": joint_winner,
                "exact_counted_bytes": joint_bytes,
            },
            "collateral": {
                "off_target_delta_errors": joint_off_target_delta_errors,
                "seg_score_delta": joint_collateral_seg,
                "pose_score_delta": joint_score["pose_score_delta"],
                "joint_collateral_score_delta": joint_collateral_score,
                "positive_collateral_byte_equivalent_at_rate_dual": joint_collateral_bytes,
            },
            "joint_score_accounting": joint_score,
            "ratio": {
                "realized_bytes_per_semantic_byte": joint_bytes / semantic_joint,
                "effective_realized_plus_positive_collateral_bytes": effective_joint,
                "effective_bytes_per_semantic_byte": effective_joint / semantic_joint,
                "bound_status": (
                    "CONSTRUCTIVE_UPPER_BOUND_WITH_FULL_TARGET_FALLBACK"
                    if joint_fallback_pairs
                    else "MEASURED_SELECTED_MENU_COMPOSITION"
                ),
            },
            "pair_rows": joint_pair_rows,
            "fallback_pair_ids": joint_fallback_pairs,
            "fallback_pair_count": len(joint_fallback_pairs),
            "non_telescope_policy": "joint camera composition was freshly remeasured; independent row deltas were not summed",
        },
        "decomposition": {
            "SKELETON": _summarize_rows(output_rows, "SKELETON"),
            "FIBER": _summarize_rows(output_rows, "FIBER"),
            "support_n_bins": _summarize_n(output_rows, config["support_n_bin_edges"]),
        },
        "context_only": {
            "reference_613_tangent_bytes": int(config["reference_613_tangent_bytes"]),
            "joint_realized_bytes_fraction_of_reference": (
                joint_bytes / int(config["reference_613_tangent_bytes"])
            ),
            "box_arithmetic_performed": False,
            "interpretation": (
                "The #613 value is a tangent/context reference, not an additive pool. "
                "No residual slack or promotion claim is computed."
            ),
            "family_d_vs_b_implication": (
                "FORMULATION implication only: any family-(d) emitter for these exact rows "
                "must cross the same measured L3/uint8/Seg/Pose gate; this does not rank or "
                "kill family (d) or family (b)."
            ),
        },
        "custody": {
            "authority_file": config["authority_file"],
            "authority_sha256": config["authority_sha256"],
            "dm1_receipt_path": config["dm1_receipt_path"],
            "dm1_receipt_sha256": config["dm1_receipt_sha256"],
            "dm1_config_path": config["dm1_config_path"],
            "dm1_config_sha256": config["dm1_config_sha256"],
            "source_config_path": config["source_config_path"],
            "source_config_sha256": config["source_config_sha256"],
            "start_receipt_path": config["start_receipt_path"],
            "start_receipt_sha256": config["start_receipt_sha256"],
            "pf2_event_index_path": config["pf2_event_index_path"],
            "pf2_event_index_sha256": config["pf2_event_index_sha256"],
            "pf2_index_receipt_path": config["pf2_index_receipt_path"],
            "pf2_index_receipt_sha256": config["pf2_index_receipt_sha256"],
            "segnet_weights_sha256": config["segnet_weights_sha256"],
            "posenet_weights_sha256": config["posenet_weights_sha256"],
            "upstream_modules_sha256": config["upstream_modules_sha256"],
            "implementation": _implementation_custody(),
            "row_checkpoint_custody_used": [
                {
                    "typed_config_sha256": pair[0],
                    "implementation_sha256": pair[1],
                }
                for pair in sorted(row_checkpoint_custody_used)
            ],
            "torch_threads": 4,
            "deterministic_algorithms": True,
            "seed": int(config["seed"]),
        },
        "verdict_scope": (
            "INSTANCE x SHA-bound 25-row demand set x exact factor2 local/global "
            "fixed-quantum candidate menu, with a full solved-target constructive "
            "upper bound on pair-level union conflicts. No global minimum-preimage "
            "certificate, contest score, promotion, family closure, or frontier "
            "mutation."
        ),
        "research_only": True,
        "execution_allowed": False,
        "score_claim": False,
        "promotion_eligible": False,
        "archive_emitted": False,
        "pointer_moved": False,
        "pointer": POINTER,
        "evidence_axis": AXIS,
        "main_review_required": True,
    }
    if not joint_semantics_exact:
        result["aggregate"]["ratio"] = {
            "status": "NULL_NONTELESCOPING_COMPOSITION_FAILED_EXACT_SEMANTICS"
        }
    receipt_path = root / "ddm_dm2_l3_realization_race_receipt.json"
    _atomic_write(receipt_path, canonical_json_bytes(result))
    return result


__all__ = [
    "AXIS",
    "CODECS",
    "CONFIG_SCHEMA",
    "POINTER",
    "SCHEMA",
    "DM2RealizationError",
    "RGBDeltaRecord",
    "candidate_scorer_plane",
    "decode_coded_rgb",
    "decode_joint_rgb_records",
    "dilated_support_mask",
    "encode_coded_rgb",
    "encode_joint_rgb_records",
    "materialize",
    "price_rgb_raw",
]
