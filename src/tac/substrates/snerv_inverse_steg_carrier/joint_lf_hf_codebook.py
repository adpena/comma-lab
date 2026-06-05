# SPDX-License-Identifier: MIT
"""Joint LF/HF factorized codebook receiver payload.

This is a small deterministic receiver primitive for the LF/HF replacement DAG:
selected receiver frames are tiled into RGB blocks, quantized into a learned
codebook, and replayed with NumPy only.  It is false-authority evidence for
implementation, receiver replay, and byte telemetry, not score authority.
"""

from __future__ import annotations

import hashlib
import json
import lzma
import math
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np

SCHEMA = "snerv_joint_lf_hf_factorized_codebook_payload.v1"
PROOF_SCHEMA = "snerv_joint_lf_hf_factorized_codebook_receiver_proof.v1"
MAGIC = b"SJLC1"
HEADER_LEN_FMT = "<I"
FALSE_AUTHORITY: dict[str, bool] = {
    "score_claim": False,
    "score_claim_valid": False,
    "promotion_eligible": False,
    "ready_for_exact_eval_dispatch": False,
    "rank_or_kill_eligible": False,
    "promotable": False,
}


class SnervJointLfHfCodebookError(ValueError):
    """Raised when a joint LF/HF codebook payload is malformed."""


@dataclass(frozen=True)
class JointLfHfCodebookPacket:
    """Encoded joint LF/HF codebook packet."""

    packet: bytes
    header: dict[str, Any]
    payload_sha256: str
    payload_bytes: int

    def as_jsonable(self) -> dict[str, Any]:
        return {
            "schema": self.header["schema"],
            "payload_sha256": self.payload_sha256,
            "payload_bytes": int(self.payload_bytes),
            "header": dict(self.header),
            **FALSE_AUTHORITY,
        }


def encode_joint_lf_hf_factorized_codebook_payload(
    frames_b2chw: np.ndarray,
    *,
    pair_indices: Sequence[int],
    block_hw: tuple[int, int] = (2, 2),
    codebook_size: int = 32,
    quant_step: float = 1.0,
) -> JointLfHfCodebookPacket:
    """Encode selected pair frames as block indices into a quantized codebook."""

    frames = _validate_frames(frames_b2chw)
    pair_ids = [int(value) for value in pair_indices]
    if len(pair_ids) != int(frames.shape[0]):
        raise SnervJointLfHfCodebookError(
            "pair_indices length must match frames batch; "
            f"got {len(pair_ids)} indices for {frames.shape[0]} frame pairs"
        )
    block_h, block_w = (int(block_hw[0]), int(block_hw[1]))
    if block_h <= 0 or block_w <= 0:
        raise SnervJointLfHfCodebookError("block_hw values must be positive")
    step = float(quant_step)
    if not math.isfinite(step) or step <= 0.0:
        raise SnervJointLfHfCodebookError("quant_step must be positive")
    k = int(codebook_size)
    if k <= 0 or k > 65535:
        raise SnervJointLfHfCodebookError("codebook_size must be in [1, 65535]")
    h, w = int(frames.shape[-2]), int(frames.shape[-1])
    if h % block_h or w % block_w:
        raise SnervJointLfHfCodebookError(
            f"frame shape {tuple(frames.shape)} must be divisible by block_hw={block_hw}"
        )
    blocks = _frames_to_blocks(frames, block_h=block_h, block_w=block_w)
    quant_blocks = np.rint(blocks / step).astype(np.int32)
    unique, inverse, counts = np.unique(
        quant_blocks,
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    order = np.lexsort(unique.T[::-1])
    count_order = np.argsort(-counts[order], kind="stable")
    selected_unique_rows = order[count_order[: min(k, len(order))]]
    codebook_i32 = unique[selected_unique_rows]
    if len(codebook_i32) == len(unique):
        remap = np.empty(len(unique), dtype=np.uint16)
        remap[selected_unique_rows] = np.arange(len(codebook_i32), dtype=np.uint16)
        indices = remap[inverse]
    else:
        indices = _nearest_codebook_indices(quant_blocks, codebook_i32)
    codebook_i16 = _int16_checked(codebook_i32, "joint LF/HF codebook")
    indices_u16 = np.asarray(indices, dtype="<u2")
    codebook_raw = np.ascontiguousarray(codebook_i16, dtype="<i2").tobytes()
    index_raw = np.ascontiguousarray(indices_u16, dtype="<u2").tobytes()
    raw = codebook_raw + index_raw
    compressed = lzma.compress(raw, format=lzma.FORMAT_XZ, preset=9 | lzma.PRESET_EXTREME)
    header = {
        "schema": SCHEMA,
        "codec": "quantized_block_codebook_int16_indices_uint16_lzma",
        "pair_indices": pair_ids,
        "shape_b2chw": [int(v) for v in frames.shape],
        "block_hw": [block_h, block_w],
        "block_vector_length": int(codebook_i16.shape[1]),
        "block_count": int(blocks.shape[0]),
        "block_grid_hw": [h // block_h, w // block_w],
        "requested_codebook_size": k,
        "codebook_entry_count": int(codebook_i16.shape[0]),
        "quant_step": step,
        "codebook_dtype": "int16_le",
        "index_dtype": "uint16_le",
        "codebook_raw_bytes": len(codebook_raw),
        "index_raw_bytes": len(index_raw),
        "raw_bytes": len(raw),
        "raw_sha256": _sha256(raw),
        "compressed_bytes": len(compressed),
        "compressed_sha256": _sha256(compressed),
        "receiver_payload_implemented": True,
        "numpy_receiver_decode": True,
        "section_native_byte_telemetry_present": True,
        **FALSE_AUTHORITY,
    }
    header_raw = _json_bytes(header)
    packet = MAGIC + struct.pack(HEADER_LEN_FMT, len(header_raw)) + header_raw + compressed
    return JointLfHfCodebookPacket(
        packet=packet,
        header={
            **header,
            "header_bytes": len(MAGIC) + struct.calcsize(HEADER_LEN_FMT) + len(header_raw),
            "packet_bytes": len(packet),
            "packet_sha256": _sha256(packet),
        },
        payload_sha256=_sha256(packet),
        payload_bytes=len(packet),
    )


def decode_joint_lf_hf_factorized_codebook_payload(packet: bytes) -> np.ndarray:
    """Decode a joint LF/HF codebook packet to ``(B,2,C,H,W)`` frames."""

    header, compressed = inspect_joint_lf_hf_factorized_codebook_payload(packet)
    try:
        raw = lzma.decompress(compressed)
    except lzma.LZMAError as exc:
        raise SnervJointLfHfCodebookError(
            "joint LF/HF codebook payload decompression failed"
        ) from exc
    if len(raw) != int(header["raw_bytes"]):
        raise SnervJointLfHfCodebookError("joint codebook raw byte count mismatch")
    if _sha256(raw) != str(header["raw_sha256"]):
        raise SnervJointLfHfCodebookError("joint codebook raw sha256 mismatch")
    codebook_bytes = int(header["codebook_raw_bytes"])
    index_bytes = int(header["index_raw_bytes"])
    if codebook_bytes + index_bytes != len(raw):
        raise SnervJointLfHfCodebookError("joint codebook sections do not sum")
    entry_count = int(header["codebook_entry_count"])
    vector_len = int(header["block_vector_length"])
    block_count = int(header["block_count"])
    codebook = np.frombuffer(raw[:codebook_bytes], dtype="<i2").copy()
    indices = np.frombuffer(raw[codebook_bytes:], dtype="<u2").copy()
    if codebook.size != entry_count * vector_len:
        raise SnervJointLfHfCodebookError("codebook section shape mismatch")
    if indices.size != block_count:
        raise SnervJointLfHfCodebookError("index section shape mismatch")
    if indices.size and int(np.max(indices)) >= entry_count:
        raise SnervJointLfHfCodebookError("codebook index out of range")
    codebook = codebook.reshape(entry_count, vector_len).astype(np.float32)
    blocks = codebook[indices.astype(np.int64)] * float(header["quant_step"])
    return _blocks_to_frames(
        blocks,
        frame_shape=tuple(int(v) for v in header["shape_b2chw"]),
        block_h=int(header["block_hw"][0]),
        block_w=int(header["block_hw"][1]),
    )


def inspect_joint_lf_hf_factorized_codebook_payload(
    packet: bytes,
) -> tuple[dict[str, Any], bytes]:
    """Return payload header and compressed body after structural validation."""

    blob = bytes(packet)
    if not blob.startswith(MAGIC):
        raise SnervJointLfHfCodebookError("bad joint LF/HF codebook payload magic")
    offset = len(MAGIC)
    header_len_size = struct.calcsize(HEADER_LEN_FMT)
    if len(blob) < offset + header_len_size:
        raise SnervJointLfHfCodebookError("truncated joint LF/HF codebook header")
    (header_len,) = struct.unpack(HEADER_LEN_FMT, blob[offset : offset + header_len_size])
    offset += header_len_size
    header_end = offset + int(header_len)
    if header_end > len(blob):
        raise SnervJointLfHfCodebookError(
            "declared joint LF/HF codebook header exceeds payload size"
        )
    header = json.loads(blob[offset:header_end].decode("utf-8"))
    if header.get("schema") != SCHEMA:
        raise SnervJointLfHfCodebookError(
            f"unsupported joint LF/HF codebook payload schema: {header.get('schema')!r}"
        )
    compressed = blob[header_end:]
    if len(compressed) != int(header["compressed_bytes"]):
        raise SnervJointLfHfCodebookError("compressed byte count mismatch")
    if _sha256(compressed) != str(header["compressed_sha256"]):
        raise SnervJointLfHfCodebookError("compressed sha256 mismatch")
    return dict(header), compressed


def build_joint_lf_hf_factorized_codebook_receiver_proof(
    frames_b2chw: np.ndarray,
    *,
    pair_indices: Sequence[int],
    packet_path: str | None = None,
    source_packet_sha256: str | None = None,
    source_clip_to_uint8_range: bool = True,
    block_hw: tuple[int, int] = (2, 2),
    codebook_size: int = 32,
    quant_step: float = 1.0,
    payload_path: str | None = None,
) -> tuple[dict[str, Any], bytes]:
    """Build a false-authority receiver decode proof for a joint codebook."""

    source = _validate_frames(frames_b2chw)
    encoded = encode_joint_lf_hf_factorized_codebook_payload(
        source,
        pair_indices=pair_indices,
        block_hw=block_hw,
        codebook_size=codebook_size,
        quant_step=quant_step,
    )
    decoded = decode_joint_lf_hf_factorized_codebook_payload(encoded.packet)
    diff = decoded - source
    abs_diff = np.abs(diff)
    decoded_stats = _array_stats(decoded)
    blockers = ["snerv_joint_lf_hf_factorized_codebook_false_authority"]
    if float(decoded_stats.get("std") or 0.0) <= 1.0e-6:
        blockers.append("snerv_joint_lf_hf_codebook_receiver_output_near_constant")
    proof = {
        "schema": PROOF_SCHEMA,
        "generated_utc": datetime.now(UTC).isoformat(),
        "packet_path": packet_path,
        "source_packet_sha256": source_packet_sha256,
        "payload_path": payload_path,
        "payload_bytes": int(encoded.payload_bytes),
        "payload_sha256": encoded.payload_sha256,
        "source_clip_to_uint8_range": bool(source_clip_to_uint8_range),
        "pair_indices": [int(value) for value in pair_indices],
        "sample_shape_b2chw": [int(v) for v in source.shape],
        "payload_header": dict(encoded.header),
        "receiver_payload_implemented": True,
        "receiver_decode_proven": True,
        "numpy_receiver_decode": True,
        "section_native_byte_telemetry_present": True,
        "codebook_raw_bytes": int(encoded.header["codebook_raw_bytes"]),
        "index_raw_bytes": int(encoded.header["index_raw_bytes"]),
        "compressed_payload_bytes": int(encoded.header["compressed_bytes"]),
        "codebook_entry_count": int(encoded.header["codebook_entry_count"]),
        "block_count": int(encoded.header["block_count"]),
        "roundtrip_abs_error_stats": _abs_array_stats(abs_diff),
        "decoded_receiver_stats": decoded_stats,
        "closed_campaign_blockers": [
            "snerv_joint_lf_hf_factorized_codebook_not_implemented",
            "snerv_joint_lf_hf_codebook_numpy_receiver_missing",
            "snerv_joint_lf_hf_codebook_section_byte_telemetry_missing",
        ],
        "blockers": blockers,
        **FALSE_AUTHORITY,
    }
    return proof, encoded.packet


def _frames_to_blocks(frames: np.ndarray, *, block_h: int, block_w: int) -> np.ndarray:
    b, t, c, h, w = (int(v) for v in frames.shape)
    grid_h = h // int(block_h)
    grid_w = w // int(block_w)
    return (
        frames.reshape(b, t, c, grid_h, block_h, grid_w, block_w)
        .transpose(0, 1, 3, 5, 2, 4, 6)
        .reshape(b * t * grid_h * grid_w, c * block_h * block_w)
        .astype(np.float32)
    )


def _blocks_to_frames(
    blocks: np.ndarray,
    *,
    frame_shape: tuple[int, int, int, int, int],
    block_h: int,
    block_w: int,
) -> np.ndarray:
    b, t, c, h, w = (int(v) for v in frame_shape)
    grid_h = h // int(block_h)
    grid_w = w // int(block_w)
    expected_blocks = b * t * grid_h * grid_w
    if int(blocks.shape[0]) != expected_blocks:
        raise SnervJointLfHfCodebookError(
            f"decoded block count {blocks.shape[0]} != expected {expected_blocks}"
        )
    return (
        np.asarray(blocks, dtype=np.float32)
        .reshape(b, t, grid_h, grid_w, c, block_h, block_w)
        .transpose(0, 1, 4, 2, 5, 3, 6)
        .reshape(b, t, c, h, w)
        .astype(np.float32)
    )


def _nearest_codebook_indices(blocks: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    out = np.empty(int(blocks.shape[0]), dtype=np.uint16)
    codebook64 = np.asarray(codebook, dtype=np.float64)
    for start in range(0, int(blocks.shape[0]), 4096):
        chunk = np.asarray(blocks[start : start + 4096], dtype=np.float64)
        distances = np.sum((chunk[:, None, :] - codebook64[None, :, :]) ** 2, axis=2)
        out[start : start + len(chunk)] = np.argmin(distances, axis=1).astype(np.uint16)
    return out


def _int16_checked(values: np.ndarray, context: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.int64)
    if np.any(arr < np.iinfo(np.int16).min) or np.any(arr > np.iinfo(np.int16).max):
        raise SnervJointLfHfCodebookError(f"{context} exceeds int16 range")
    return arr.astype("<i2")


def _validate_frames(frames_b2chw: np.ndarray) -> np.ndarray:
    arr = np.asarray(frames_b2chw, dtype=np.float32)
    if arr.ndim != 5 or int(arr.shape[1]) != 2 or int(arr.shape[2]) <= 0:
        raise SnervJointLfHfCodebookError(
            "frames must have shape (pairs,2,channels,height,width), "
            f"got {tuple(arr.shape)}"
        )
    if any(int(value) <= 0 for value in arr.shape):
        raise SnervJointLfHfCodebookError("frame dimensions must be positive")
    if not np.all(np.isfinite(arr)):
        raise SnervJointLfHfCodebookError("frames must be finite")
    return np.ascontiguousarray(arr, dtype=np.float32)


def _array_stats(array: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(array, dtype=np.float32)
    if arr.size == 0:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
            "saturation_fraction": None,
        }
    saturated = np.count_nonzero((arr <= 0.5) | (arr >= 254.5))
    return {
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr, dtype=np.float64)),
        "std": float(np.std(arr, dtype=np.float64)),
        "saturation_fraction": float(saturated) / float(arr.size),
    }


def _abs_array_stats(array: np.ndarray) -> dict[str, Any]:
    arr = np.abs(np.asarray(array, dtype=np.float32))
    if arr.size == 0:
        return {"count": 0, "mean_abs": None, "max_abs": None}
    return {
        "count": int(arr.size),
        "mean_abs": float(np.mean(arr, dtype=np.float64)),
        "max_abs": float(np.max(arr)),
    }


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()
