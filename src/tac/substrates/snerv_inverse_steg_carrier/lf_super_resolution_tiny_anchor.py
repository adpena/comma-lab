# SPDX-License-Identifier: MIT
"""Tiny-anchor LF super-resolution receiver payload.

This false-authority primitive stores a deliberately tiny LF anchor and replays
deterministic NumPy super-resolution by nearest-neighbor upsampling.  The proof
records receiver pixel-domain deltas against the source sample; it does not
claim scorer-component deltas, source-forward parity, or score authority.
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

SCHEMA = "snerv_lf_super_resolution_tiny_anchor_payload.v1"
PROOF_SCHEMA = "snerv_lf_super_resolution_tiny_anchor_receiver_proof.v1"
MAGIC = b"SLSR1"
HEADER_LEN_FMT = "<I"
FALSE_AUTHORITY: dict[str, bool] = {
    "score_claim": False,
    "score_claim_valid": False,
    "promotion_eligible": False,
    "ready_for_exact_eval_dispatch": False,
    "rank_or_kill_eligible": False,
    "promotable": False,
}


class SnervLfSuperResolutionTinyAnchorError(ValueError):
    """Raised when a tiny-anchor SR payload is malformed."""


@dataclass(frozen=True)
class LfSuperResolutionTinyAnchorPacket:
    """Encoded tiny-anchor SR packet."""

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


def encode_lf_super_resolution_tiny_anchor_payload(
    frames_b2chw: np.ndarray,
    *,
    pair_indices: Sequence[int],
    anchor_downsample: int = 8,
    anchor_quant_step: float = 1.0,
) -> LfSuperResolutionTinyAnchorPacket:
    """Encode selected pairs as quantized tiny LF anchors."""

    frames = _validate_frames(frames_b2chw)
    pair_ids = [int(value) for value in pair_indices]
    if len(pair_ids) != int(frames.shape[0]):
        raise SnervLfSuperResolutionTinyAnchorError(
            "pair_indices length must match frames batch; "
            f"got {len(pair_ids)} indices for {frames.shape[0]} frame pairs"
        )
    downsample = int(anchor_downsample)
    if downsample <= 0:
        raise SnervLfSuperResolutionTinyAnchorError(
            "anchor_downsample must be positive"
        )
    step = float(anchor_quant_step)
    if not math.isfinite(step) or step <= 0.0:
        raise SnervLfSuperResolutionTinyAnchorError(
            "anchor_quant_step must be positive"
        )
    h, w = int(frames.shape[-2]), int(frames.shape[-1])
    if h % downsample or w % downsample:
        raise SnervLfSuperResolutionTinyAnchorError(
            "frame height/width must be divisible by anchor_downsample; "
            f"shape={tuple(frames.shape)} anchor_downsample={downsample}"
        )

    anchor = _average_pool_2d(frames, downsample)
    quantized = np.rint(anchor / step)
    if np.any(quantized < np.iinfo(np.uint16).min) or np.any(
        quantized > np.iinfo(np.uint16).max
    ):
        raise SnervLfSuperResolutionTinyAnchorError(
            "tiny anchor exceeds uint16 range for requested quantization step"
        )
    anchor_u16 = np.asarray(quantized, dtype="<u2")
    anchor_raw = np.ascontiguousarray(anchor_u16, dtype="<u2").tobytes(order="C")
    compressed = lzma.compress(
        anchor_raw,
        format=lzma.FORMAT_XZ,
        preset=9 | lzma.PRESET_EXTREME,
    )
    header = {
        "schema": SCHEMA,
        "codec": "tiny_anchor_uint16_lzma_nearest_super_resolution",
        "pair_indices": pair_ids,
        "source_shape_b2chw": [int(v) for v in frames.shape],
        "anchor_shape_b2chw": [int(v) for v in anchor.shape],
        "anchor_downsample": downsample,
        "anchor_quant_step": step,
        "anchor_dtype": "uint16_le",
        "super_resolution_decoder": "nearest_neighbor_repeat_numpy",
        "anchor_raw_bytes": len(anchor_raw),
        "raw_bytes": len(anchor_raw),
        "raw_sha256": _sha256(anchor_raw),
        "compressed_bytes": len(compressed),
        "compressed_sha256": _sha256(compressed),
        "receiver_payload_implemented": True,
        "numpy_receiver_decode": True,
        "tiny_anchor_component_deltas_present": True,
        "component_delta_scope": "receiver_pixel_domain_not_scorer_component",
        "section_native_byte_telemetry_present": True,
        **FALSE_AUTHORITY,
    }
    header_raw = _json_bytes(header)
    packet = MAGIC + struct.pack(HEADER_LEN_FMT, len(header_raw)) + header_raw + compressed
    return LfSuperResolutionTinyAnchorPacket(
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


def decode_lf_super_resolution_tiny_anchor_payload(packet: bytes) -> np.ndarray:
    """Decode tiny-anchor SR bytes to ``(B,2,C,H,W)`` frames."""

    header, compressed = inspect_lf_super_resolution_tiny_anchor_payload(packet)
    try:
        raw = lzma.decompress(compressed)
    except lzma.LZMAError as exc:
        raise SnervLfSuperResolutionTinyAnchorError(
            "tiny-anchor SR payload decompression failed"
        ) from exc
    if len(raw) != int(header["raw_bytes"]):
        raise SnervLfSuperResolutionTinyAnchorError(
            "tiny-anchor raw byte count mismatch"
        )
    if _sha256(raw) != str(header["raw_sha256"]):
        raise SnervLfSuperResolutionTinyAnchorError("tiny-anchor raw sha256 mismatch")
    anchor_shape = tuple(int(v) for v in header["anchor_shape_b2chw"])
    source_shape = tuple(int(v) for v in header["source_shape_b2chw"])
    anchor = np.frombuffer(raw, dtype="<u2").copy()
    if anchor.size != int(np.prod(anchor_shape)):
        raise SnervLfSuperResolutionTinyAnchorError("tiny-anchor shape mismatch")
    anchor = anchor.reshape(anchor_shape).astype(np.float32) * float(
        header["anchor_quant_step"]
    )
    decoded = _nearest_upsample_2d(anchor, int(header["anchor_downsample"]))
    if tuple(decoded.shape) != source_shape:
        raise SnervLfSuperResolutionTinyAnchorError(
            "tiny-anchor decode does not upsample to source frame shape"
        )
    return np.asarray(decoded, dtype=np.float32)


def inspect_lf_super_resolution_tiny_anchor_payload(
    packet: bytes,
) -> tuple[dict[str, Any], bytes]:
    """Return payload header and compressed body after structural validation."""

    blob = bytes(packet)
    if not blob.startswith(MAGIC):
        raise SnervLfSuperResolutionTinyAnchorError("bad tiny-anchor SR magic")
    offset = len(MAGIC)
    header_len_size = struct.calcsize(HEADER_LEN_FMT)
    if len(blob) < offset + header_len_size:
        raise SnervLfSuperResolutionTinyAnchorError("truncated tiny-anchor header")
    (header_len,) = struct.unpack(HEADER_LEN_FMT, blob[offset : offset + header_len_size])
    offset += header_len_size
    header_end = offset + int(header_len)
    if header_end > len(blob):
        raise SnervLfSuperResolutionTinyAnchorError(
            "declared tiny-anchor header exceeds payload size"
        )
    header = json.loads(blob[offset:header_end].decode("utf-8"))
    if header.get("schema") != SCHEMA:
        raise SnervLfSuperResolutionTinyAnchorError(
            f"unsupported tiny-anchor payload schema: {header.get('schema')!r}"
        )
    compressed = blob[header_end:]
    if len(compressed) != int(header["compressed_bytes"]):
        raise SnervLfSuperResolutionTinyAnchorError("compressed byte count mismatch")
    if _sha256(compressed) != str(header["compressed_sha256"]):
        raise SnervLfSuperResolutionTinyAnchorError("compressed sha256 mismatch")
    return dict(header), compressed


def build_lf_super_resolution_tiny_anchor_receiver_proof(
    frames_b2chw: np.ndarray,
    *,
    pair_indices: Sequence[int],
    packet_path: str | None = None,
    source_packet_sha256: str | None = None,
    source_clip_to_uint8_range: bool = True,
    anchor_downsample: int = 8,
    anchor_quant_step: float = 1.0,
    payload_path: str | None = None,
) -> tuple[dict[str, Any], bytes]:
    """Build a false-authority receiver proof for tiny-anchor SR payloads."""

    source = _validate_frames(frames_b2chw)
    encoded = encode_lf_super_resolution_tiny_anchor_payload(
        source,
        pair_indices=pair_indices,
        anchor_downsample=anchor_downsample,
        anchor_quant_step=anchor_quant_step,
    )
    decoded = decode_lf_super_resolution_tiny_anchor_payload(encoded.packet)
    abs_diff = np.abs(decoded - source)
    decoded_stats = _array_stats(decoded)
    blockers = ["snerv_lf_super_resolution_tiny_anchor_payload_false_authority"]
    if float(decoded_stats.get("std") or 0.0) <= 1.0e-6:
        blockers.append("snerv_lf_super_resolution_receiver_output_near_constant")
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
        "anchor_shape_b2chw": list(encoded.header["anchor_shape_b2chw"]),
        "payload_header": dict(encoded.header),
        "receiver_payload_implemented": True,
        "receiver_decode_proven": True,
        "numpy_receiver_decode": True,
        "tiny_anchor_component_deltas_present": True,
        "component_delta_scope": "receiver_pixel_domain_not_scorer_component",
        "section_native_byte_telemetry_present": True,
        "anchor_raw_bytes": int(encoded.header["anchor_raw_bytes"]),
        "compressed_payload_bytes": int(encoded.header["compressed_bytes"]),
        "receiver_component_delta_stats": {
            "all_frames": _abs_array_stats(abs_diff),
            "first_frame": _abs_array_stats(abs_diff[:, 0]),
            "last_frame": _abs_array_stats(abs_diff[:, 1]),
            "temporal_delta": _abs_array_stats(
                (decoded[:, 1] - decoded[:, 0]) - (source[:, 1] - source[:, 0])
            ),
            "scope": "receiver_pixel_domain_not_scorer_component",
        },
        "decoded_receiver_stats": decoded_stats,
        "closed_campaign_blockers": [
            "snerv_lf_super_resolution_receiver_payload_not_implemented",
            "snerv_lf_downsampled_anchor_component_deltas_missing",
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
        raise SnervLfSuperResolutionTinyAnchorError(
            "frames must have shape (pairs,2,channels,height,width), "
            f"got {tuple(arr.shape)}"
        )
    if any(int(value) <= 0 for value in arr.shape):
        raise SnervLfSuperResolutionTinyAnchorError("frame dimensions must be positive")
    if not np.all(np.isfinite(arr)):
        raise SnervLfSuperResolutionTinyAnchorError("frames must be finite")
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
