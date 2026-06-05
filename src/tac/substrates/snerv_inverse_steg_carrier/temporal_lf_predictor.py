# SPDX-License-Identifier: MIT
"""Temporal LF predictor receiver payload.

This false-authority primitive stores a first-frame LF anchor plus a byte-
charged temporal correction stream for the second frame's LF plane.  It proves
that the receiver can replay the temporal LF predictor in deterministic NumPy;
it does not claim source-forward TUB authority or score authority.
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

SCHEMA = "snerv_temporal_lf_predictor_payload.v1"
PROOF_SCHEMA = "snerv_temporal_lf_predictor_receiver_proof.v1"
MAGIC = b"STLP1"
HEADER_LEN_FMT = "<I"
FALSE_AUTHORITY: dict[str, bool] = {
    "score_claim": False,
    "score_claim_valid": False,
    "promotion_eligible": False,
    "ready_for_exact_eval_dispatch": False,
    "rank_or_kill_eligible": False,
    "promotable": False,
}


class SnervTemporalLfPredictorError(ValueError):
    """Raised when a temporal LF predictor payload is malformed."""


@dataclass(frozen=True)
class TemporalLfPredictorPacket:
    """Encoded temporal LF predictor packet."""

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


def encode_temporal_lf_predictor_payload(
    frames_b2chw: np.ndarray,
    *,
    pair_indices: Sequence[int],
    lf_downsample: int = 4,
    correction_quant_step: float = 1.0,
) -> TemporalLfPredictorPacket:
    """Encode selected pairs as first LF anchor plus temporal LF correction."""

    frames = _validate_frames(frames_b2chw)
    pair_ids = [int(value) for value in pair_indices]
    if len(pair_ids) != int(frames.shape[0]):
        raise SnervTemporalLfPredictorError(
            "pair_indices length must match frames batch; "
            f"got {len(pair_ids)} indices for {frames.shape[0]} frame pairs"
        )
    downsample = int(lf_downsample)
    if downsample <= 0:
        raise SnervTemporalLfPredictorError("lf_downsample must be positive")
    step = float(correction_quant_step)
    if not math.isfinite(step) or step <= 0.0:
        raise SnervTemporalLfPredictorError(
            "correction_quant_step must be positive"
        )
    h, w = int(frames.shape[-2]), int(frames.shape[-1])
    if h % downsample or w % downsample:
        raise SnervTemporalLfPredictorError(
            "frame height/width must be divisible by lf_downsample; "
            f"shape={tuple(frames.shape)} lf_downsample={downsample}"
        )

    lf_planes = _average_pool_2d(frames, downsample)
    first_lf = np.asarray(lf_planes[:, 0], dtype="<f4")
    correction = lf_planes[:, 1] - lf_planes[:, 0]
    quantized = np.rint(correction / step)
    if np.any(quantized < np.iinfo(np.int16).min) or np.any(
        quantized > np.iinfo(np.int16).max
    ):
        raise SnervTemporalLfPredictorError(
            "temporal LF correction exceeds int16 range for requested quant step"
        )
    correction_i16 = np.asarray(quantized, dtype="<i2")
    first_raw = np.ascontiguousarray(first_lf, dtype="<f4").tobytes(order="C")
    correction_raw = np.ascontiguousarray(correction_i16, dtype="<i2").tobytes(
        order="C"
    )
    raw = first_raw + correction_raw
    compressed = lzma.compress(raw, format=lzma.FORMAT_XZ, preset=9 | lzma.PRESET_EXTREME)
    header = {
        "schema": SCHEMA,
        "codec": "first_lf_float32_plus_temporal_correction_int16_lzma",
        "predictor": "second_lf_equals_first_lf_plus_quantized_correction",
        "pair_indices": pair_ids,
        "source_shape_b2chw": [int(v) for v in frames.shape],
        "lf_shape_b2chw": [int(v) for v in lf_planes.shape],
        "first_lf_shape_bchw": [int(v) for v in first_lf.shape],
        "lf_downsample": downsample,
        "first_lf_dtype": "float32_le",
        "correction_dtype": "int16_le",
        "correction_quant_step": step,
        "first_lf_anchor_bytes": len(first_raw),
        "correction_stream_raw_bytes": len(correction_raw),
        "raw_bytes": len(raw),
        "raw_sha256": _sha256(raw),
        "compressed_bytes": len(compressed),
        "compressed_sha256": _sha256(compressed),
        "receiver_payload_implemented": True,
        "numpy_receiver_decode": True,
        "correction_stream_byte_charged": True,
        "section_native_byte_telemetry_present": True,
        **FALSE_AUTHORITY,
    }
    header_raw = _json_bytes(header)
    packet = MAGIC + struct.pack(HEADER_LEN_FMT, len(header_raw)) + header_raw + compressed
    return TemporalLfPredictorPacket(
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


def decode_temporal_lf_predictor_payload(packet: bytes) -> np.ndarray:
    """Decode a temporal LF predictor packet to ``(B,2,C,Hlf,Wlf)`` planes."""

    header, compressed = inspect_temporal_lf_predictor_payload(packet)
    try:
        raw = lzma.decompress(compressed)
    except lzma.LZMAError as exc:
        raise SnervTemporalLfPredictorError(
            "temporal LF predictor payload decompression failed"
        ) from exc
    if len(raw) != int(header["raw_bytes"]):
        raise SnervTemporalLfPredictorError("temporal LF raw byte count mismatch")
    if _sha256(raw) != str(header["raw_sha256"]):
        raise SnervTemporalLfPredictorError("temporal LF raw sha256 mismatch")
    first_bytes = int(header["first_lf_anchor_bytes"])
    correction_bytes = int(header["correction_stream_raw_bytes"])
    if first_bytes + correction_bytes != len(raw):
        raise SnervTemporalLfPredictorError("temporal LF sections do not sum")

    first_shape = tuple(int(v) for v in header["first_lf_shape_bchw"])
    lf_shape = tuple(int(v) for v in header["lf_shape_b2chw"])
    first = np.frombuffer(raw[:first_bytes], dtype="<f4").copy()
    correction_q = np.frombuffer(raw[first_bytes:], dtype="<i2").copy()
    if first.size != int(np.prod(first_shape)):
        raise SnervTemporalLfPredictorError("first LF anchor shape mismatch")
    if correction_q.size != int(np.prod(first_shape)):
        raise SnervTemporalLfPredictorError("temporal correction shape mismatch")
    first = first.reshape(first_shape).astype(np.float32)
    correction = correction_q.reshape(first_shape).astype(np.float32) * float(
        header["correction_quant_step"]
    )
    second = np.asarray(first + correction, dtype=np.float32)
    decoded = np.stack([first, second], axis=1).astype(np.float32)
    if tuple(decoded.shape) != lf_shape:
        raise SnervTemporalLfPredictorError("decoded temporal LF shape mismatch")
    return decoded


def inspect_temporal_lf_predictor_payload(
    packet: bytes,
) -> tuple[dict[str, Any], bytes]:
    """Return payload header and compressed body after structural validation."""

    blob = bytes(packet)
    if not blob.startswith(MAGIC):
        raise SnervTemporalLfPredictorError("bad temporal LF payload magic")
    offset = len(MAGIC)
    header_len_size = struct.calcsize(HEADER_LEN_FMT)
    if len(blob) < offset + header_len_size:
        raise SnervTemporalLfPredictorError("truncated temporal LF header")
    (header_len,) = struct.unpack(HEADER_LEN_FMT, blob[offset : offset + header_len_size])
    offset += header_len_size
    header_end = offset + int(header_len)
    if header_end > len(blob):
        raise SnervTemporalLfPredictorError(
            "declared temporal LF header exceeds payload size"
        )
    header = json.loads(blob[offset:header_end].decode("utf-8"))
    if header.get("schema") != SCHEMA:
        raise SnervTemporalLfPredictorError(
            f"unsupported temporal LF payload schema: {header.get('schema')!r}"
        )
    compressed = blob[header_end:]
    if len(compressed) != int(header["compressed_bytes"]):
        raise SnervTemporalLfPredictorError("compressed byte count mismatch")
    if _sha256(compressed) != str(header["compressed_sha256"]):
        raise SnervTemporalLfPredictorError("compressed sha256 mismatch")
    return dict(header), compressed


def build_temporal_lf_predictor_receiver_proof(
    frames_b2chw: np.ndarray,
    *,
    pair_indices: Sequence[int],
    packet_path: str | None = None,
    source_packet_sha256: str | None = None,
    source_clip_to_uint8_range: bool = True,
    lf_downsample: int = 4,
    correction_quant_step: float = 1.0,
    payload_path: str | None = None,
) -> tuple[dict[str, Any], bytes]:
    """Build a false-authority receiver decode proof for temporal LF payloads."""

    source = _validate_frames(frames_b2chw)
    source_lf = _average_pool_2d(source, int(lf_downsample))
    encoded = encode_temporal_lf_predictor_payload(
        source,
        pair_indices=pair_indices,
        lf_downsample=lf_downsample,
        correction_quant_step=correction_quant_step,
    )
    decoded_lf = decode_temporal_lf_predictor_payload(encoded.packet)
    abs_diff = np.abs(decoded_lf - source_lf)
    decoded_stats = _array_stats(decoded_lf)
    blockers = ["snerv_temporal_lf_predictor_payload_false_authority"]
    max_allowed = float(correction_quant_step) * 0.5 + 1.0e-4
    max_abs_error = float(np.max(abs_diff)) if abs_diff.size else 0.0
    if max_abs_error > max_allowed:
        blockers.append("snerv_temporal_lf_predictor_receiver_roundtrip_error")
    if float(decoded_stats.get("std") or 0.0) <= 1.0e-6:
        blockers.append("snerv_temporal_lf_predictor_receiver_output_near_constant")
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
        "lf_shape_b2chw": [int(v) for v in source_lf.shape],
        "payload_header": dict(encoded.header),
        "receiver_payload_implemented": True,
        "receiver_decode_proven": True,
        "numpy_receiver_decode": True,
        "correction_stream_byte_charged": True,
        "section_native_byte_telemetry_present": True,
        "first_lf_anchor_bytes": int(encoded.header["first_lf_anchor_bytes"]),
        "correction_stream_raw_bytes": int(
            encoded.header["correction_stream_raw_bytes"]
        ),
        "compressed_payload_bytes": int(encoded.header["compressed_bytes"]),
        "roundtrip_abs_error_stats": _abs_array_stats(abs_diff),
        "decoded_receiver_stats": decoded_stats,
        "closed_campaign_blockers": [
            "snerv_temporal_lf_predictor_gate_not_implemented",
            "snerv_temporal_lf_predictor_correction_stream_not_byte_charged",
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


def _validate_frames(frames_b2chw: np.ndarray) -> np.ndarray:
    arr = np.asarray(frames_b2chw, dtype=np.float32)
    if arr.ndim != 5 or int(arr.shape[1]) != 2 or int(arr.shape[2]) <= 0:
        raise SnervTemporalLfPredictorError(
            "frames must have shape (pairs,2,channels,height,width), "
            f"got {tuple(arr.shape)}"
        )
    if any(int(value) <= 0 for value in arr.shape):
        raise SnervTemporalLfPredictorError("frame dimensions must be positive")
    if not np.all(np.isfinite(arr)):
        raise SnervTemporalLfPredictorError("frames must be finite")
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
