# SPDX-License-Identifier: MIT
"""Receiver payload for LF-conditioned HF residual prototypes.

This is a false-authority prototype primitive: it proves that the receiver can
consume an LF anchor plus a byte-charged HF residual stream with deterministic
NumPy replay.  It does not claim score authority or source-forward parity.
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

SCHEMA = "snerv_lf_conditioned_hf_residual_payload.v1"
PROOF_SCHEMA = "snerv_lf_conditioned_hf_residual_receiver_proof.v1"
MAGIC = b"SLHR1"
HEADER_LEN_FMT = "<I"
FALSE_AUTHORITY: dict[str, bool] = {
    "score_claim": False,
    "score_claim_valid": False,
    "promotion_eligible": False,
    "ready_for_exact_eval_dispatch": False,
    "rank_or_kill_eligible": False,
    "promotable": False,
}


class SnervLfConditionedHfResidualError(ValueError):
    """Raised when an LF-conditioned HF residual payload is malformed."""


@dataclass(frozen=True)
class LfConditionedHfResidualPacket:
    """Encoded LF anchor + HF residual packet."""

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


def encode_lf_conditioned_hf_residual_payload(
    frames_b2chw: np.ndarray,
    *,
    pair_indices: Sequence[int],
    anchor_downsample: int = 2,
    residual_quant_step: float = 1.0,
) -> LfConditionedHfResidualPacket:
    """Encode selected pair frames as LF anchors plus quantized HF residuals."""

    frames = _validate_frames(frames_b2chw)
    pair_ids = [int(value) for value in pair_indices]
    if len(pair_ids) != int(frames.shape[0]):
        raise SnervLfConditionedHfResidualError(
            "pair_indices length must match frames batch; "
            f"got {len(pair_ids)} indices for {frames.shape[0]} frame pairs"
        )
    downsample = int(anchor_downsample)
    if downsample <= 0:
        raise SnervLfConditionedHfResidualError("anchor_downsample must be positive")
    step = float(residual_quant_step)
    if not math.isfinite(step) or step <= 0.0:
        raise SnervLfConditionedHfResidualError("residual_quant_step must be positive")
    h, w = int(frames.shape[-2]), int(frames.shape[-1])
    if h % downsample or w % downsample:
        raise SnervLfConditionedHfResidualError(
            "frame height/width must be divisible by anchor_downsample; "
            f"shape={tuple(frames.shape)} anchor_downsample={downsample}"
        )
    anchor = _average_pool_2d(frames, downsample)
    prediction = _nearest_upsample_2d(anchor, downsample)
    residual = frames - prediction
    quantized = np.rint(residual / step)
    if np.any(quantized < np.iinfo(np.int16).min) or np.any(
        quantized > np.iinfo(np.int16).max
    ):
        raise SnervLfConditionedHfResidualError(
            "HF residual exceeds int16 range for requested quantization step"
        )
    q16 = np.asarray(quantized, dtype="<i2")
    anchor32 = np.asarray(anchor, dtype="<f4")
    anchor_raw = anchor32.tobytes(order="C")
    residual_raw = q16.tobytes(order="C")
    raw = anchor_raw + residual_raw
    compressed = lzma.compress(raw, format=lzma.FORMAT_XZ, preset=9 | lzma.PRESET_EXTREME)
    header = {
        "schema": SCHEMA,
        "codec": "lf_anchor_float32_plus_hf_residual_int16_lzma",
        "pair_indices": pair_ids,
        "shape_b2chw": [int(v) for v in frames.shape],
        "anchor_shape_b2chw": [int(v) for v in anchor32.shape],
        "anchor_downsample": downsample,
        "anchor_dtype": "float32_le",
        "residual_dtype": "int16_le",
        "residual_quant_step": step,
        "anchor_raw_bytes": len(anchor_raw),
        "residual_raw_bytes": len(residual_raw),
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
    return LfConditionedHfResidualPacket(
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


def decode_lf_conditioned_hf_residual_payload(packet: bytes) -> np.ndarray:
    """Decode LF-conditioned HF residual bytes to ``(B,2,C,H,W)`` frames."""

    header, compressed = inspect_lf_conditioned_hf_residual_payload(packet)
    try:
        raw = lzma.decompress(compressed)
    except lzma.LZMAError as exc:
        raise SnervLfConditionedHfResidualError(
            "LF-conditioned HF residual payload decompression failed"
        ) from exc
    if len(raw) != int(header["raw_bytes"]):
        raise SnervLfConditionedHfResidualError("residual raw byte count mismatch")
    if _sha256(raw) != str(header["raw_sha256"]):
        raise SnervLfConditionedHfResidualError("residual raw sha256 mismatch")
    anchor_shape = tuple(int(v) for v in header["anchor_shape_b2chw"])
    frame_shape = tuple(int(v) for v in header["shape_b2chw"])
    anchor_bytes = int(header["anchor_raw_bytes"])
    residual_bytes = int(header["residual_raw_bytes"])
    if anchor_bytes + residual_bytes != len(raw):
        raise SnervLfConditionedHfResidualError("residual raw sections do not sum")
    anchor = np.frombuffer(raw[:anchor_bytes], dtype="<f4").copy()
    residual_q = np.frombuffer(raw[anchor_bytes:], dtype="<i2").copy()
    if anchor.size != int(np.prod(anchor_shape)):
        raise SnervLfConditionedHfResidualError("anchor section shape mismatch")
    if residual_q.size != int(np.prod(frame_shape)):
        raise SnervLfConditionedHfResidualError("residual section shape mismatch")
    anchor = anchor.reshape(anchor_shape).astype(np.float32)
    residual = residual_q.reshape(frame_shape).astype(np.float32) * float(
        header["residual_quant_step"]
    )
    prediction = _nearest_upsample_2d(anchor, int(header["anchor_downsample"]))
    if tuple(prediction.shape) != frame_shape:
        raise SnervLfConditionedHfResidualError(
            "decoded LF anchor does not upsample to residual frame shape"
        )
    return np.asarray(prediction + residual, dtype=np.float32)


def inspect_lf_conditioned_hf_residual_payload(
    packet: bytes,
) -> tuple[dict[str, Any], bytes]:
    """Return payload header and compressed body after structural validation."""

    blob = bytes(packet)
    if not blob.startswith(MAGIC):
        raise SnervLfConditionedHfResidualError("bad LF/HF residual payload magic")
    offset = len(MAGIC)
    header_len_size = struct.calcsize(HEADER_LEN_FMT)
    if len(blob) < offset + header_len_size:
        raise SnervLfConditionedHfResidualError("truncated LF/HF residual header")
    (header_len,) = struct.unpack(HEADER_LEN_FMT, blob[offset : offset + header_len_size])
    offset += header_len_size
    header_end = offset + int(header_len)
    if header_end > len(blob):
        raise SnervLfConditionedHfResidualError(
            "declared LF/HF residual header exceeds payload size"
        )
    header = json.loads(blob[offset:header_end].decode("utf-8"))
    if header.get("schema") != SCHEMA:
        raise SnervLfConditionedHfResidualError(
            f"unsupported LF/HF residual payload schema: {header.get('schema')!r}"
        )
    compressed = blob[header_end:]
    if len(compressed) != int(header["compressed_bytes"]):
        raise SnervLfConditionedHfResidualError("compressed byte count mismatch")
    if _sha256(compressed) != str(header["compressed_sha256"]):
        raise SnervLfConditionedHfResidualError("compressed sha256 mismatch")
    return dict(header), compressed


def build_lf_conditioned_hf_residual_receiver_proof(
    frames_b2chw: np.ndarray,
    *,
    pair_indices: Sequence[int],
    packet_path: str | None = None,
    source_packet_sha256: str | None = None,
    source_clip_to_uint8_range: bool = True,
    anchor_downsample: int = 2,
    residual_quant_step: float = 1.0,
    payload_path: str | None = None,
) -> tuple[dict[str, Any], bytes]:
    """Build a false-authority receiver decode proof for selected frames."""

    source = _validate_frames(frames_b2chw)
    encoded = encode_lf_conditioned_hf_residual_payload(
        source,
        pair_indices=pair_indices,
        anchor_downsample=anchor_downsample,
        residual_quant_step=residual_quant_step,
    )
    decoded = decode_lf_conditioned_hf_residual_payload(encoded.packet)
    diff = decoded - source
    abs_diff = np.abs(diff)
    decoded_stats = _array_stats(decoded)
    blockers = ["snerv_lf_conditioned_hf_residual_payload_false_authority"]
    max_allowed = float(residual_quant_step) * 0.5 + 1.0e-4
    max_abs_error = float(np.max(abs_diff)) if abs_diff.size else 0.0
    if max_abs_error > max_allowed:
        blockers.append("snerv_lf_conditioned_hf_residual_receiver_roundtrip_error")
    if float(decoded_stats.get("std") or 0.0) <= 1.0e-6:
        blockers.append("snerv_lf_conditioned_hf_residual_receiver_output_near_constant")
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
        "lf_anchor_bytes": int(encoded.header["anchor_raw_bytes"]),
        "hf_residual_bytes": int(encoded.header["residual_raw_bytes"]),
        "compressed_payload_bytes": int(encoded.header["compressed_bytes"]),
        "roundtrip_abs_error_stats": _abs_array_stats(abs_diff),
        "decoded_receiver_stats": decoded_stats,
        "closed_campaign_blockers": [
            "snerv_hf_residual_generator_receiver_payload_not_implemented"
        ],
        "blockers": blockers,
        **FALSE_AUTHORITY,
    }
    return proof, encoded.packet


def _average_pool_2d(frames: np.ndarray, factor: int) -> np.ndarray:
    b, t, c, h, w = (int(v) for v in frames.shape)
    return frames.reshape(b, t, c, h // factor, factor, w // factor, factor).mean(
        axis=(4, 6),
        dtype=np.float32,
    )


def _nearest_upsample_2d(anchor: np.ndarray, factor: int) -> np.ndarray:
    return np.repeat(np.repeat(anchor, int(factor), axis=-2), int(factor), axis=-1)


def _validate_frames(frames_b2chw: np.ndarray) -> np.ndarray:
    arr = np.asarray(frames_b2chw, dtype=np.float32)
    if arr.ndim != 5 or int(arr.shape[1]) != 2 or int(arr.shape[2]) <= 0:
        raise SnervLfConditionedHfResidualError(
            "frames must have shape (pairs,2,channels,height,width), "
            f"got {tuple(arr.shape)}"
        )
    if any(int(value) <= 0 for value in arr.shape):
        raise SnervLfConditionedHfResidualError("frame dimensions must be positive")
    if not np.all(np.isfinite(arr)):
        raise SnervLfConditionedHfResidualError("frames must be finite")
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
