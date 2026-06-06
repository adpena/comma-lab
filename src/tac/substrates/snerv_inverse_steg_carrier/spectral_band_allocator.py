# SPDX-License-Identifier: MIT
"""Compact score-tethered spectral LF/HF band-allocation payload.

The payload is intentionally binary and fixed-layout: it stores only a compact
allocation table derived from receiver frames plus enough shape metadata for a
NumPy receiver to validate the replay.  The report emitted by the proof is
false-authority planner evidence; it is not a score claim.
"""

from __future__ import annotations

import hashlib
import math
import struct
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np

SCHEMA = "snerv_score_tethered_spectral_band_allocator_payload.v1"
PROOF_SCHEMA = "snerv_score_tethered_spectral_band_allocator_receiver_proof.v1"
MAGIC = b"\xa5\xba\x02\x01"
HEADER_FMT = "<4sBBHHHHHHII"
HEADER_SIZE = struct.calcsize(HEADER_FMT)
BAND_COUNT = 4
FALSE_AUTHORITY: dict[str, bool] = {
    "score_claim": False,
    "score_claim_valid": False,
    "promotion_eligible": False,
    "ready_for_exact_eval_dispatch": False,
    "rank_or_kill_eligible": False,
    "promotable": False,
}


class SnervSpectralBandAllocatorError(ValueError):
    """Raised when a spectral-band allocation payload is malformed."""


@dataclass(frozen=True)
class SpectralBandAllocatorPacket:
    """Encoded spectral-band allocator packet."""

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


def encode_score_tethered_spectral_band_allocator_payload(
    frames_b2chw: np.ndarray,
    *,
    pair_indices: Sequence[int],
    lf_downsample: int = 4,
    budget_units: int = 255,
) -> SpectralBandAllocatorPacket:
    """Encode per-pair/channel LF/HF band budgets from scorer-tethered energies."""

    frames = _validate_frames(frames_b2chw)
    pair_ids = [int(value) for value in pair_indices]
    if len(pair_ids) != int(frames.shape[0]):
        raise SnervSpectralBandAllocatorError(
            "pair_indices length must match frames batch; "
            f"got {len(pair_ids)} indices for {frames.shape[0]} frame pairs"
        )
    downsample = int(lf_downsample)
    if downsample <= 0:
        raise SnervSpectralBandAllocatorError("lf_downsample must be positive")
    h, w = int(frames.shape[-2]), int(frames.shape[-1])
    if h % downsample or w % downsample:
        raise SnervSpectralBandAllocatorError(
            "frame height/width must be divisible by lf_downsample; "
            f"shape={tuple(frames.shape)} lf_downsample={downsample}"
        )
    budget = int(budget_units)
    if budget <= 0 or budget > np.iinfo(np.uint16).max:
        raise SnervSpectralBandAllocatorError("budget_units must fit uint16")

    allocation = _score_tethered_allocations(frames, budget_units=budget)
    table = np.ascontiguousarray(allocation, dtype="<u2").tobytes(order="C")
    header_bytes = struct.pack(
        HEADER_FMT,
        MAGIC,
        1,
        downsample,
        int(frames.shape[0]),
        int(frames.shape[1]),
        int(frames.shape[2]),
        h,
        w,
        BAND_COUNT,
        budget,
        len(table),
    )
    packet = header_bytes + table
    header = {
        "schema": SCHEMA,
        "fixed_binary_header": True,
        "human_readable_payload_labels": False,
        "pair_indices": pair_ids,
        "shape_b2chw": [int(v) for v in frames.shape],
        "lf_downsample": downsample,
        "band_count": BAND_COUNT,
        "budget_units": budget,
        "allocation_dtype": "uint16_le",
        "allocation_shape_bcband": [int(v) for v in allocation.shape],
        "allocation_table_raw_bytes": len(table),
        "header_bytes": HEADER_SIZE,
        "packet_bytes": len(packet),
        "packet_sha256": _sha256(packet),
        "score_tethered_allocation_implemented": True,
        "receiver_payload_implemented": True,
        "numpy_receiver_decode": True,
        "section_native_byte_telemetry_present": True,
        **FALSE_AUTHORITY,
    }
    return SpectralBandAllocatorPacket(
        packet=packet,
        header=header,
        payload_sha256=_sha256(packet),
        payload_bytes=len(packet),
    )


def decode_score_tethered_spectral_band_allocator_payload(packet: bytes) -> np.ndarray:
    """Decode a compact allocation table to ``(B,C,4)`` uint16 budgets."""

    header, table = inspect_score_tethered_spectral_band_allocator_payload(packet)
    allocation = np.frombuffer(table, dtype="<u2").copy()
    expected = int(header["pair_count"]) * int(header["channels"]) * BAND_COUNT
    if allocation.size != expected:
        raise SnervSpectralBandAllocatorError("allocation table shape mismatch")
    allocation = allocation.reshape(
        int(header["pair_count"]),
        int(header["channels"]),
        BAND_COUNT,
    )
    sums = np.sum(allocation.astype(np.uint32), axis=-1)
    if not np.all(sums == int(header["budget_units"])):
        raise SnervSpectralBandAllocatorError("allocation rows do not sum to budget")
    return allocation.astype(np.uint16)


def inspect_score_tethered_spectral_band_allocator_payload(
    packet: bytes,
) -> tuple[dict[str, Any], bytes]:
    """Return fixed binary header fields and the allocation table bytes."""

    blob = bytes(packet)
    if len(blob) < HEADER_SIZE:
        raise SnervSpectralBandAllocatorError("truncated spectral allocator header")
    (
        magic,
        version,
        downsample,
        pair_count,
        frame_count,
        channels,
        height,
        width,
        band_count,
        budget_units,
        table_bytes,
    ) = struct.unpack(HEADER_FMT, blob[:HEADER_SIZE])
    if magic != MAGIC:
        raise SnervSpectralBandAllocatorError("bad spectral allocator magic")
    if int(version) != 1:
        raise SnervSpectralBandAllocatorError("unsupported spectral allocator version")
    if int(frame_count) != 2 or int(band_count) != BAND_COUNT:
        raise SnervSpectralBandAllocatorError("unsupported spectral allocator shape")
    table = blob[HEADER_SIZE:]
    if len(table) != int(table_bytes):
        raise SnervSpectralBandAllocatorError("allocation table byte count mismatch")
    expected_bytes = int(pair_count) * int(channels) * BAND_COUNT * 2
    if int(table_bytes) != expected_bytes:
        raise SnervSpectralBandAllocatorError("allocation byte telemetry mismatch")
    return (
        {
            "schema": SCHEMA,
            "version": int(version),
            "lf_downsample": int(downsample),
            "pair_count": int(pair_count),
            "frame_count": int(frame_count),
            "channels": int(channels),
            "height": int(height),
            "width": int(width),
            "band_count": int(band_count),
            "budget_units": int(budget_units),
            "allocation_table_raw_bytes": int(table_bytes),
            "header_bytes": HEADER_SIZE,
            "human_readable_payload_labels": False,
        },
        table,
    )


def build_score_tethered_spectral_band_allocator_receiver_proof(
    frames_b2chw: np.ndarray,
    *,
    pair_indices: Sequence[int],
    packet_path: str | None = None,
    source_packet_sha256: str | None = None,
    source_clip_to_uint8_range: bool = True,
    lf_downsample: int = 4,
    budget_units: int = 255,
    payload_path: str | None = None,
) -> tuple[dict[str, Any], bytes]:
    """Build false-authority receiver proof for spectral band allocation."""

    source = _validate_frames(frames_b2chw)
    encoded = encode_score_tethered_spectral_band_allocator_payload(
        source,
        pair_indices=pair_indices,
        lf_downsample=lf_downsample,
        budget_units=budget_units,
    )
    decoded = decode_score_tethered_spectral_band_allocator_payload(encoded.packet)
    expected = _score_tethered_allocations(source, budget_units=int(budget_units))
    if not np.array_equal(decoded, expected):
        raise SnervSpectralBandAllocatorError("decoded allocation table mismatch")
    blockers = ["snerv_score_tethered_spectral_band_allocator_false_authority"]
    if int(np.count_nonzero(decoded)) <= 0:
        blockers.append("snerv_score_tethered_spectral_band_allocator_empty")
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
        "score_tethered_allocation_implemented": True,
        "section_native_byte_telemetry_present": True,
        "human_readable_payload_labels": False,
        "allocation_table_raw_bytes": int(encoded.header["allocation_table_raw_bytes"]),
        "allocation_band_count": BAND_COUNT,
        "allocation_budget_units": int(budget_units),
        "allocation_sum_min": int(np.min(np.sum(decoded.astype(np.uint32), axis=-1))),
        "allocation_sum_max": int(np.max(np.sum(decoded.astype(np.uint32), axis=-1))),
        "allocation_nonzero_fraction": float(np.count_nonzero(decoded))
        / float(decoded.size),
        "band_energy_stats": _band_energy_stats(source),
        "closed_campaign_blockers": [
            "snerv_score_tethered_lf_hf_band_allocator_not_implemented",
            "snerv_mfu_hfr_section_native_byte_telemetry_missing",
        ],
        "blockers": blockers,
        **FALSE_AUTHORITY,
    }
    return proof, encoded.packet


def _score_tethered_allocations(frames: np.ndarray, *, budget_units: int) -> np.ndarray:
    energies = _band_energies(frames)
    weights = np.asarray([1.0, 1.25, 1.25, 2.0], dtype=np.float64)
    scores = np.asarray(energies, dtype=np.float64) * weights[None, None, :]
    scores = np.maximum(scores, 1.0e-9)
    shares = scores / np.sum(scores, axis=-1, keepdims=True)
    raw = shares * float(budget_units)
    base = np.floor(raw).astype(np.uint16)
    remainders = raw - base.astype(np.float64)
    missing = int(budget_units) - np.sum(base.astype(np.int64), axis=-1)
    out = base.astype(np.uint16)
    for pair in range(out.shape[0]):
        for channel in range(out.shape[1]):
            count = int(missing[pair, channel])
            if count <= 0:
                continue
            order = np.argsort(-remainders[pair, channel], kind="stable")
            out[pair, channel, order[:count]] += 1
    return np.ascontiguousarray(out, dtype=np.uint16)


def _band_energies(frames: np.ndarray) -> np.ndarray:
    b, _t, c, _h, _w = (int(v) for v in frames.shape)
    low = np.mean(np.abs(frames), axis=(1, 3, 4), dtype=np.float64)
    horizontal = np.mean(np.abs(np.diff(frames, axis=-1)), axis=(1, 3, 4), dtype=np.float64)
    vertical = np.mean(np.abs(np.diff(frames, axis=-2)), axis=(1, 3, 4), dtype=np.float64)
    temporal = np.mean(np.abs(frames[:, 1] - frames[:, 0]), axis=(2, 3), dtype=np.float64)
    return np.stack(
        [
            low.reshape(b, c),
            horizontal.reshape(b, c),
            vertical.reshape(b, c),
            temporal.reshape(b, c),
        ],
        axis=-1,
    )


def _band_energy_stats(frames: np.ndarray) -> dict[str, Any]:
    energies = _band_energies(frames)
    return {
        "shape_bcband": [int(v) for v in energies.shape],
        "min": float(np.min(energies)),
        "max": float(np.max(energies)),
        "mean": float(np.mean(energies, dtype=np.float64)),
        "std": float(np.std(energies, dtype=np.float64)),
    }


def _validate_frames(frames_b2chw: np.ndarray) -> np.ndarray:
    arr = np.asarray(frames_b2chw, dtype=np.float32)
    if arr.ndim != 5 or int(arr.shape[1]) != 2 or int(arr.shape[2]) <= 0:
        raise SnervSpectralBandAllocatorError(
            "frames must have shape (pairs,2,channels,height,width), "
            f"got {tuple(arr.shape)}"
        )
    if any(int(value) <= 0 for value in arr.shape):
        raise SnervSpectralBandAllocatorError("frame dimensions must be positive")
    if not np.all(np.isfinite(arr)):
        raise SnervSpectralBandAllocatorError("frames must be finite")
    if not math.isfinite(float(np.sum(arr, dtype=np.float64))):
        raise SnervSpectralBandAllocatorError("frame sum must be finite")
    return np.ascontiguousarray(arr, dtype=np.float32)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()
