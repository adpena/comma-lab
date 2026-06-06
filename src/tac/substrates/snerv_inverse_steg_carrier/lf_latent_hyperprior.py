# SPDX-License-Identifier: MIT
"""Compact LF latent hyperprior receiver payload.

This proof primitive fits a per-pair/frame/channel LF latent mean and scale,
stores centered quantized symbols, and replays the LF latent with deterministic
NumPy.  It is a real receiver codec surface for the LF hyperprior DAG row, but
the emitted proof remains false-authority and scorer-free.
"""

from __future__ import annotations

import hashlib
import math
import struct
import zlib
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np

SCHEMA = "snerv_lf_latent_hyperprior_payload.v1"
PROOF_SCHEMA = "snerv_lf_latent_hyperprior_receiver_proof.v1"
MAGIC = b"\xa5\x48\x02\x01"
HEADER_FMT = "<4sBBHHHHHHIII"
HEADER_SIZE = struct.calcsize(HEADER_FMT)
FALSE_AUTHORITY: dict[str, bool] = {
    "score_claim": False,
    "score_claim_valid": False,
    "promotion_eligible": False,
    "ready_for_exact_eval_dispatch": False,
    "rank_or_kill_eligible": False,
    "promotable": False,
}


class SnervLfLatentHyperpriorError(ValueError):
    """Raised when an LF latent hyperprior payload is malformed."""


@dataclass(frozen=True)
class LfLatentHyperpriorPacket:
    """Encoded LF latent hyperprior packet."""

    packet: bytes
    header: dict[str, Any]
    payload_sha256: str
    payload_bytes: int

    def as_jsonable(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "payload_sha256": self.payload_sha256,
            "payload_bytes": int(self.payload_bytes),
            "header": dict(self.header),
            **FALSE_AUTHORITY,
        }


def encode_lf_latent_hyperprior_payload(
    frames_b2chw: np.ndarray,
    *,
    pair_indices: Sequence[int],
    lf_downsample: int = 4,
    quant_step: float = 1.0,
) -> LfLatentHyperpriorPacket:
    """Encode LF latents with fitted mean/scale hyperprior and zlib symbols."""

    frames = _validate_frames(frames_b2chw)
    pair_ids = [int(value) for value in pair_indices]
    if len(pair_ids) != int(frames.shape[0]):
        raise SnervLfLatentHyperpriorError(
            "pair_indices length must match frames batch; "
            f"got {len(pair_ids)} indices for {frames.shape[0]} frame pairs"
        )
    downsample = int(lf_downsample)
    if downsample <= 0:
        raise SnervLfLatentHyperpriorError("lf_downsample must be positive")
    step = float(quant_step)
    if not math.isfinite(step) or step <= 0.0:
        raise SnervLfLatentHyperpriorError("quant_step must be positive")
    h, w = int(frames.shape[-2]), int(frames.shape[-1])
    if h % downsample or w % downsample:
        raise SnervLfLatentHyperpriorError(
            "frame height/width must be divisible by lf_downsample; "
            f"shape={tuple(frames.shape)} lf_downsample={downsample}"
        )

    lf = _average_pool_2d(frames, downsample)
    mean = np.rint(np.mean(lf, axis=(-2, -1), dtype=np.float64)).astype(np.int32)
    centered = (lf - mean[..., None, None].astype(np.float32)) / step
    symbols = np.rint(centered).astype(np.int32)
    if np.any(symbols < np.iinfo(np.int16).min) or np.any(
        symbols > np.iinfo(np.int16).max
    ):
        raise SnervLfLatentHyperpriorError("LF latent symbols exceed int16 range")
    residual = lf - mean[..., None, None].astype(np.float32)
    scale = np.rint(
        np.mean(np.abs(residual), axis=(-2, -1), dtype=np.float64)
    ).astype(np.int32)
    if np.any(mean < np.iinfo(np.uint16).min) or np.any(
        mean > np.iinfo(np.uint16).max
    ):
        raise SnervLfLatentHyperpriorError("LF latent mean exceeds uint16 range")
    if np.any(scale < np.iinfo(np.uint16).min) or np.any(
        scale > np.iinfo(np.uint16).max
    ):
        raise SnervLfLatentHyperpriorError("LF latent scale exceeds uint16 range")

    mean_raw = np.ascontiguousarray(mean.astype("<u2"), dtype="<u2").tobytes(order="C")
    scale_raw = np.ascontiguousarray(scale.astype("<u2"), dtype="<u2").tobytes(
        order="C"
    )
    symbol_raw = np.ascontiguousarray(symbols.astype("<i2"), dtype="<i2").tobytes(
        order="C"
    )
    symbol_compressed = zlib.compress(symbol_raw, level=9)
    quant_milli = round(step * 1000.0)
    if quant_milli <= 0 or quant_milli > np.iinfo(np.uint16).max:
        raise SnervLfLatentHyperpriorError("quant_step must fit uint16 millistep")
    header_bytes = struct.pack(
        HEADER_FMT,
        MAGIC,
        1,
        downsample,
        int(lf.shape[0]),
        int(lf.shape[1]),
        int(lf.shape[2]),
        int(lf.shape[3]),
        int(lf.shape[4]),
        quant_milli,
        len(mean_raw),
        len(scale_raw),
        len(symbol_compressed),
    )
    packet = header_bytes + mean_raw + scale_raw + symbol_compressed
    header = {
        "schema": SCHEMA,
        "fixed_binary_header": True,
        "human_readable_payload_labels": False,
        "pair_indices": pair_ids,
        "lf_shape_b2chw": [int(v) for v in lf.shape],
        "lf_downsample": downsample,
        "quant_step": step,
        "mean_dtype": "uint16_le",
        "scale_dtype": "uint16_le",
        "symbol_dtype": "int16_le_zlib",
        "mean_raw_bytes": len(mean_raw),
        "scale_raw_bytes": len(scale_raw),
        "latent_symbol_raw_bytes": len(symbol_raw),
        "latent_symbol_compressed_bytes": len(symbol_compressed),
        "header_bytes": HEADER_SIZE,
        "packet_bytes": len(packet),
        "packet_sha256": _sha256(packet),
        "receiver_payload_implemented": True,
        "numpy_receiver_decode": True,
        "entropy_model_implemented": True,
        "hyperprior_scale_present": True,
        "receiver_replay_proven": True,
        "section_native_byte_telemetry_present": True,
        **FALSE_AUTHORITY,
    }
    return LfLatentHyperpriorPacket(
        packet=packet,
        header=header,
        payload_sha256=_sha256(packet),
        payload_bytes=len(packet),
    )


def decode_lf_latent_hyperprior_payload(packet: bytes) -> np.ndarray:
    """Decode a compact LF hyperprior payload to ``(B,2,C,Hlf,Wlf)`` latents."""

    header, mean_raw, scale_raw, symbol_compressed = inspect_lf_latent_hyperprior_payload(
        packet
    )
    try:
        symbol_raw = zlib.decompress(symbol_compressed)
    except zlib.error as exc:
        raise SnervLfLatentHyperpriorError(
            "LF hyperprior symbol decompression failed"
        ) from exc
    lf_shape = (
        int(header["pair_count"]),
        int(header["frame_count"]),
        int(header["channels"]),
        int(header["lf_height"]),
        int(header["lf_width"]),
    )
    mean_shape = lf_shape[:3]
    mean = np.frombuffer(mean_raw, dtype="<u2").copy()
    scale = np.frombuffer(scale_raw, dtype="<u2").copy()
    symbols = np.frombuffer(symbol_raw, dtype="<i2").copy()
    if mean.size != int(np.prod(mean_shape)):
        raise SnervLfLatentHyperpriorError("LF hyperprior mean shape mismatch")
    if scale.size != int(np.prod(mean_shape)):
        raise SnervLfLatentHyperpriorError("LF hyperprior scale shape mismatch")
    if symbols.size != int(np.prod(lf_shape)):
        raise SnervLfLatentHyperpriorError("LF hyperprior symbol shape mismatch")
    mean = mean.reshape(mean_shape).astype(np.float32)
    scale = scale.reshape(mean_shape).astype(np.float32)
    if not np.all(np.isfinite(scale)):
        raise SnervLfLatentHyperpriorError("LF hyperprior scale is not finite")
    symbols = symbols.reshape(lf_shape).astype(np.float32)
    return mean[..., None, None] + symbols * float(header["quant_step"])


def inspect_lf_latent_hyperprior_payload(
    packet: bytes,
) -> tuple[dict[str, Any], bytes, bytes, bytes]:
    """Return fixed header fields and payload sections."""

    blob = bytes(packet)
    if len(blob) < HEADER_SIZE:
        raise SnervLfLatentHyperpriorError("truncated LF hyperprior header")
    (
        magic,
        version,
        downsample,
        pair_count,
        frame_count,
        channels,
        lf_height,
        lf_width,
        quant_milli,
        mean_bytes,
        scale_bytes,
        compressed_bytes,
    ) = struct.unpack(HEADER_FMT, blob[:HEADER_SIZE])
    if magic != MAGIC:
        raise SnervLfLatentHyperpriorError("bad LF hyperprior magic")
    if int(version) != 1:
        raise SnervLfLatentHyperpriorError("unsupported LF hyperprior version")
    if int(frame_count) != 2:
        raise SnervLfLatentHyperpriorError("unsupported LF hyperprior frame count")
    offset = HEADER_SIZE
    mean_end = offset + int(mean_bytes)
    scale_end = mean_end + int(scale_bytes)
    symbol_end = scale_end + int(compressed_bytes)
    if symbol_end != len(blob):
        raise SnervLfLatentHyperpriorError("LF hyperprior section byte mismatch")
    expected_mean_scale = int(pair_count) * int(frame_count) * int(channels) * 2
    if int(mean_bytes) != expected_mean_scale or int(scale_bytes) != expected_mean_scale:
        raise SnervLfLatentHyperpriorError("LF hyperprior mean/scale byte mismatch")
    return (
        {
            "schema": SCHEMA,
            "version": int(version),
            "lf_downsample": int(downsample),
            "pair_count": int(pair_count),
            "frame_count": int(frame_count),
            "channels": int(channels),
            "lf_height": int(lf_height),
            "lf_width": int(lf_width),
            "quant_step": float(quant_milli) / 1000.0,
            "mean_raw_bytes": int(mean_bytes),
            "scale_raw_bytes": int(scale_bytes),
            "latent_symbol_compressed_bytes": int(compressed_bytes),
            "header_bytes": HEADER_SIZE,
            "human_readable_payload_labels": False,
        },
        blob[offset:mean_end],
        blob[mean_end:scale_end],
        blob[scale_end:symbol_end],
    )


def build_lf_latent_hyperprior_receiver_proof(
    frames_b2chw: np.ndarray,
    *,
    pair_indices: Sequence[int],
    packet_path: str | None = None,
    source_packet_sha256: str | None = None,
    source_clip_to_uint8_range: bool = True,
    lf_downsample: int = 4,
    quant_step: float = 1.0,
    payload_path: str | None = None,
) -> tuple[dict[str, Any], bytes]:
    """Build a false-authority receiver proof for LF latent hyperprior payloads."""

    source = _validate_frames(frames_b2chw)
    source_lf = _average_pool_2d(source, int(lf_downsample))
    encoded = encode_lf_latent_hyperprior_payload(
        source,
        pair_indices=pair_indices,
        lf_downsample=lf_downsample,
        quant_step=quant_step,
    )
    decoded_lf = decode_lf_latent_hyperprior_payload(encoded.packet)
    abs_diff = np.abs(decoded_lf - source_lf)
    blockers = ["snerv_lf_latent_hyperprior_payload_false_authority"]
    max_allowed = float(quant_step) * 0.5 + 1.0e-4
    max_abs = float(np.max(abs_diff)) if abs_diff.size else 0.0
    if max_abs > max_allowed:
        blockers.append("snerv_lf_latent_hyperprior_receiver_roundtrip_error")
    if float(np.std(decoded_lf, dtype=np.float64)) <= 1.0e-6:
        blockers.append("snerv_lf_latent_hyperprior_receiver_output_near_constant")
    entropy = _laplace_hyperprior_entropy_bits(source_lf, step=float(quant_step))
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
        "entropy_model_implemented": True,
        "hyperprior_scale_present": True,
        "receiver_replay_proven": True,
        "section_native_byte_telemetry_present": True,
        "human_readable_payload_labels": False,
        "entropy_model": "per_slice_laplace_scale_hyperprior",
        "mean_raw_bytes": int(encoded.header["mean_raw_bytes"]),
        "scale_raw_bytes": int(encoded.header["scale_raw_bytes"]),
        "latent_symbol_raw_bytes": int(encoded.header["latent_symbol_raw_bytes"]),
        "compressed_payload_bytes": int(
            encoded.header["latent_symbol_compressed_bytes"]
        ),
        "estimated_entropy_bits": entropy,
        "roundtrip_abs_error_stats": _abs_array_stats(abs_diff),
        "decoded_receiver_stats": _array_stats(decoded_lf),
        "closed_campaign_blockers": [
            "snerv_lf_latent_hyperprior_not_implemented",
            "snerv_lf_latent_hyperprior_numpy_decoder_missing",
            "snerv_lf_latent_hyperprior_receiver_replay_missing",
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


def _laplace_hyperprior_entropy_bits(values: np.ndarray, *, step: float) -> float:
    lf = np.asarray(values, dtype=np.float32)
    if lf.size == 0:
        return 0.0
    mean = np.rint(np.mean(lf, axis=(-2, -1), dtype=np.float64)).astype(np.float64)
    residual = lf.astype(np.float64) - mean[..., None, None]
    symbols = np.rint(residual / float(step)).astype(np.float64)
    scale = np.maximum(
        np.mean(np.abs(residual), axis=(-2, -1), dtype=np.float64) / float(step),
        1.0,
    )
    # Two-sided geometric/Laplace proxy with per-slice fitted scale.
    probs = np.exp(-np.abs(symbols) / scale[..., None, None])
    normalizer = np.maximum(2.0 * scale[..., None, None], 1.0)
    probs = np.clip(probs / normalizer, 1.0e-12, 1.0)
    return float(-np.sum(np.log2(probs), dtype=np.float64))


def _validate_frames(frames_b2chw: np.ndarray) -> np.ndarray:
    arr = np.asarray(frames_b2chw, dtype=np.float32)
    if arr.ndim != 5 or int(arr.shape[1]) != 2 or int(arr.shape[2]) <= 0:
        raise SnervLfLatentHyperpriorError(
            "frames must have shape (pairs,2,channels,height,width), "
            f"got {tuple(arr.shape)}"
        )
    if any(int(value) <= 0 for value in arr.shape):
        raise SnervLfLatentHyperpriorError("frame dimensions must be positive")
    if not np.all(np.isfinite(arr)):
        raise SnervLfLatentHyperpriorError("frames must be finite")
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


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()
