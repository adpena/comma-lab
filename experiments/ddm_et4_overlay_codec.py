# SPDX-License-Identifier: MIT
"""ET4 counted overlay payload codec.

The archive stores two video-derived objects: the parent IX2 payload and a
Brotli-compressed sparse uint8 delta stream for frame_1.  Decode-time code is
generic and free; every byte in the overlay stream is consumed by the runtime.
"""
from __future__ import annotations

import hashlib
import io
import json
import struct
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import brotli
import numpy as np

PAYLOAD_MAGIC: Final = b"ET4OV1\0\0"
PATCH_MAGIC: Final = b"ET4PD1\0\0"
PAYLOAD_VERSION: Final = 1
PATCH_VERSION: Final = 1
CAMERA_H: Final = 874
CAMERA_W: Final = 1164
CHANNELS: Final = 3
FRAME_VALUES: Final = CAMERA_H * CAMERA_W * CHANNELS
N_PAIRS: Final = 600

_PAYLOAD_HEADER = struct.Struct("<8sB3xII")
_PATCH_HEADER = struct.Struct("<8sB3xH")
_PATCH_ROW = struct.Struct("<HI")


class ET4OverlayCodecError(ValueError):
    """Raised when an ET4 overlay payload does not close exactly."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def frame1_delta_record(pair: int, before_u8: np.ndarray, after_u8: np.ndarray) -> dict[str, Any]:
    """Build one sparse frame_1 patch record from two camera frames."""

    before = np.asarray(before_u8)
    after = np.asarray(after_u8)
    if before.dtype != np.uint8 or after.dtype != np.uint8:
        raise ET4OverlayCodecError("frame patches require uint8 input frames")
    expected = (CAMERA_H, CAMERA_W, CHANNELS)
    if before.shape != expected or after.shape != expected:
        raise ET4OverlayCodecError(f"frame shape mismatch: {before.shape} vs {after.shape}")
    if not 0 <= int(pair) < N_PAIRS:
        raise ET4OverlayCodecError(f"pair id out of range: {pair}")
    flat_before = before.reshape(-1).astype(np.int16)
    flat_after = after.reshape(-1).astype(np.int16)
    delta = flat_after - flat_before
    idx = np.flatnonzero(delta).astype("<u4", copy=False)
    vals = delta[idx].astype("<i2", copy=False)
    return {
        "pair": int(pair),
        "nnz": int(idx.size),
        "indices": idx,
        "deltas_i16": vals,
        "before_sha256": sha256_bytes(np.ascontiguousarray(before).tobytes()),
        "after_sha256": sha256_bytes(np.ascontiguousarray(after).tobytes()),
        "delta_index_sha256": sha256_bytes(np.ascontiguousarray(idx).tobytes()),
        "delta_value_sha256": sha256_bytes(np.ascontiguousarray(vals).tobytes()),
    }


def encode_patch_records(records: Sequence[Mapping[str, Any]], *, quality: int = 11) -> tuple[bytes, dict[str, Any]]:
    """Encode sparse records into one compressed counted patch stream."""

    ordered = sorted(records, key=lambda row: int(row["pair"]))
    seen: set[int] = set()
    raw = io.BytesIO()
    raw.write(_PATCH_HEADER.pack(PATCH_MAGIC, PATCH_VERSION, len(ordered)))
    total_nnz = 0
    for row in ordered:
        pair = int(row["pair"])
        if pair in seen:
            raise ET4OverlayCodecError(f"duplicate patch pair {pair}")
        if not 0 <= pair < N_PAIRS:
            raise ET4OverlayCodecError(f"patch pair out of range: {pair}")
        seen.add(pair)
        indices = np.asarray(row["indices"], dtype="<u4")
        deltas = np.asarray(row["deltas_i16"], dtype="<i2")
        if indices.ndim != 1 or deltas.ndim != 1 or indices.shape != deltas.shape:
            raise ET4OverlayCodecError(f"bad sparse arrays for pair {pair}")
        if indices.size and (int(indices.min()) < 0 or int(indices.max()) >= FRAME_VALUES):
            raise ET4OverlayCodecError(f"patch index out of frame bounds for pair {pair}")
        if indices.size and not np.all(indices[:-1] < indices[1:]):
            raise ET4OverlayCodecError(f"patch indices are not strictly sorted for pair {pair}")
        raw.write(_PATCH_ROW.pack(pair, int(indices.size)))
        raw.write(np.ascontiguousarray(indices, dtype="<u4").tobytes())
        raw.write(np.ascontiguousarray(deltas, dtype="<i2").tobytes())
        total_nnz += int(indices.size)
    raw_bytes = raw.getvalue()
    compressed = brotli.compress(raw_bytes, quality=int(quality))
    receipt = {
        "schema": "ddm_et4_overlay_patch_receipt.v1",
        "codec": "sparse_frame1_i16_delta_brotli",
        "quality": int(quality),
        "record_count": len(ordered),
        "total_nnz": total_nnz,
        "raw_bytes": len(raw_bytes),
        "compressed_bytes": len(compressed),
        "raw_sha256": sha256_bytes(raw_bytes),
        "compressed_sha256": sha256_bytes(compressed),
        "pairs": [int(row["pair"]) for row in ordered],
    }
    return compressed, receipt


def decode_patch_records(compressed: bytes) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    raw = brotli.decompress(compressed)
    if len(raw) < _PATCH_HEADER.size:
        raise ET4OverlayCodecError("patch payload is shorter than the header")
    magic, version, count = _PATCH_HEADER.unpack_from(raw, 0)
    if magic != PATCH_MAGIC or version != PATCH_VERSION:
        raise ET4OverlayCodecError("patch magic/version mismatch")
    off = _PATCH_HEADER.size
    records: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for _ in range(count):
        if off + _PATCH_ROW.size > len(raw):
            raise ET4OverlayCodecError("patch row header truncated")
        pair, nnz = _PATCH_ROW.unpack_from(raw, off)
        off += _PATCH_ROW.size
        idx_bytes = int(nnz) * 4
        val_bytes = int(nnz) * 2
        if off + idx_bytes + val_bytes > len(raw):
            raise ET4OverlayCodecError("patch row body truncated")
        indices = np.frombuffer(raw, dtype="<u4", count=int(nnz), offset=off).copy()
        off += idx_bytes
        deltas = np.frombuffer(raw, dtype="<i2", count=int(nnz), offset=off).copy()
        off += val_bytes
        if pair in records:
            raise ET4OverlayCodecError(f"duplicate decoded pair {pair}")
        if indices.size and int(indices.max()) >= FRAME_VALUES:
            raise ET4OverlayCodecError(f"decoded patch index out of bounds for pair {pair}")
        if indices.size and not np.all(indices[:-1] < indices[1:]):
            raise ET4OverlayCodecError(f"decoded patch indices not sorted for pair {pair}")
        records[int(pair)] = (indices, deltas)
    if off != len(raw):
        raise ET4OverlayCodecError("patch payload has trailing bytes")
    return records


def apply_patch_to_frame1(frame_u8: np.ndarray, patch: tuple[np.ndarray, np.ndarray] | None) -> np.ndarray:
    """Apply one decoded sparse patch to a frame_1 camera frame."""

    frame = np.asarray(frame_u8)
    if frame.dtype != np.uint8 or frame.shape != (CAMERA_H, CAMERA_W, CHANNELS):
        raise ET4OverlayCodecError(f"bad frame for overlay: {frame.dtype} {frame.shape}")
    if patch is None:
        return np.ascontiguousarray(frame)
    indices, deltas = patch
    out = frame.reshape(-1).astype(np.int16)
    values = out[indices] + deltas.astype(np.int16)
    if np.any(values < 0) or np.any(values > 255):
        raise ET4OverlayCodecError("overlay patch leaves uint8 range")
    out[indices] = values
    return np.ascontiguousarray(out.astype(np.uint8).reshape(CAMERA_H, CAMERA_W, CHANNELS))


def encode_overlay_payload(
    *,
    parent_payload: bytes,
    compressed_patch: bytes,
    metadata: Mapping[str, Any],
) -> tuple[bytes, dict[str, Any]]:
    """Build the counted ET4 single-member payload."""

    meta_bytes = _canonical_json(dict(metadata))
    header = _PAYLOAD_HEADER.pack(
        PAYLOAD_MAGIC,
        PAYLOAD_VERSION,
        len(parent_payload),
        len(compressed_patch),
    )
    payload = header + parent_payload + compressed_patch + meta_bytes
    receipt = {
        "schema": "ddm_et4_overlay_payload_receipt.v1",
        "magic": PAYLOAD_MAGIC.decode("ascii", errors="replace").rstrip("\0"),
        "parent_payload_bytes": len(parent_payload),
        "parent_payload_sha256": sha256_bytes(parent_payload),
        "compressed_patch_bytes": len(compressed_patch),
        "compressed_patch_sha256": sha256_bytes(compressed_patch),
        "metadata_bytes": len(meta_bytes),
        "metadata_sha256": sha256_bytes(meta_bytes),
        "payload_bytes": len(payload),
        "payload_sha256": sha256_bytes(payload),
    }
    return payload, receipt


def decode_overlay_payload(payload: bytes) -> tuple[bytes, bytes, dict[str, Any]]:
    if len(payload) < _PAYLOAD_HEADER.size:
        raise ET4OverlayCodecError("ET4 payload is shorter than the header")
    magic, version, parent_len, patch_len = _PAYLOAD_HEADER.unpack_from(payload, 0)
    if magic != PAYLOAD_MAGIC or version != PAYLOAD_VERSION:
        raise ET4OverlayCodecError("ET4 payload magic/version mismatch")
    off = _PAYLOAD_HEADER.size
    end_parent = off + int(parent_len)
    end_patch = end_parent + int(patch_len)
    if end_patch > len(payload):
        raise ET4OverlayCodecError("ET4 payload declared sections exceed payload length")
    parent = payload[off:end_parent]
    patch = payload[end_parent:end_patch]
    meta_raw = payload[end_patch:]
    try:
        metadata = json.loads(meta_raw.decode("utf-8")) if meta_raw else {}
    except json.JSONDecodeError as exc:
        raise ET4OverlayCodecError("ET4 metadata is not valid JSON") from exc
    return parent, patch, metadata


def load_patch_records(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    """Load per-pair patch records written as npz files by the ET4 runner."""

    records: list[dict[str, Any]] = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            records.append(
                {
                    "pair": int(data["pair"][0]),
                    "nnz": int(data["nnz"][0]),
                    "indices": np.asarray(data["indices"], dtype="<u4"),
                    "deltas_i16": np.asarray(data["deltas_i16"], dtype="<i2"),
                }
            )
    return records
