# SPDX-License-Identifier: MIT
"""Finite counted seed for sparse five-class partition events.

This module closes the finite-coder accounting hole in the G3 cell-identity
measurement.  The packet stores only video-derived constraints: strictly sorted
sites and their target/baseline cell identities.  The context model, ULEB128,
zlib decoder, deterministic tie rules, and event application are generic
receiver algorithms and therefore belong in the interpreter, not in the seed.

The packet is deliberately *not* a full partition receiver.  Its caller must
provide a baseline partition produced by a separately custodied chart/predictor.
No RGB/YUV plane values, scorer weights, margins, or inherited archive bytes are
serialized here.
"""

from __future__ import annotations

import json
import struct
import zlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Final

import numpy as np

PACKET_SCHEMA: Final = "s2_partition_event_seed.v1"
PACKET_MAGIC: Final = b"S2P1"
PACKET_VERSION: Final = 1
SEMANTIC_NAMES: Final = ("Road", "Lane", "Undrivable", "Movable", "MyCar")
_PREFIX: Final = struct.Struct("<4sIII")
_CRC: Final = struct.Struct("<I")
_HEADER_FIELDS: Final = frozenset(
    {
        "schema",
        "version",
        "codec",
        "zlib_level",
        "n_pairs",
        "height",
        "width",
        "event_count",
        "semantic_names",
        "semantic_class_ids",
        "raw_event_bytes",
        "raw_event_sha256",
        "compressed_event_bytes",
        "compressed_event_sha256",
    }
)


class PartitionSeedError(ValueError):
    """Fail-closed malformed seed, semantic detection, or receiver input."""


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise PartitionSeedError("packet header must be canonical finite JSON") from exc


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PartitionSeedError(f"{name} must be a positive integer")
    return value


def _semantic_ids(value: Sequence[int]) -> tuple[int, ...]:
    ids = tuple(int(v) for v in value)
    if len(ids) != len(SEMANTIC_NAMES) or set(ids) != set(range(len(SEMANTIC_NAMES))):
        raise PartitionSeedError("semantic_class_ids must be a permutation of five class ids")
    return ids


@dataclass(frozen=True, order=True)
class PartitionEvent:
    """One baseline-to-target argmax-cell constraint at a scorer-grid site."""

    pair: int
    row: int
    col: int
    target_class: int
    baseline_class: int


@dataclass(frozen=True)
class SemanticDetection:
    """Spatial/static signature result, independent of channel index or luma."""

    semantic_class_ids: tuple[int, ...]
    per_class: tuple[Mapping[str, float | int], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "semantic_class_ids", _semantic_ids(self.semantic_class_ids))

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": "spatial_static_signature_v1",
            "semantic_names": list(SEMANTIC_NAMES),
            "semantic_class_ids": list(self.semantic_class_ids),
            "per_class": [dict(row) for row in self.per_class],
            "luma_consulted": False,
        }


@dataclass(frozen=True)
class PartitionEventSeed:
    """Decoded finite seed and its scorer-grid geometry."""

    n_pairs: int
    height: int
    width: int
    semantic_class_ids: tuple[int, ...]
    events: tuple[PartitionEvent, ...]

    def __post_init__(self) -> None:
        n_pairs = _positive_int(self.n_pairs, "n_pairs")
        height = _positive_int(self.height, "height")
        width = _positive_int(self.width, "width")
        semantic_ids = _semantic_ids(self.semantic_class_ids)
        events = tuple(self.events)
        previous = -1
        for event in events:
            if not isinstance(event, PartitionEvent):
                raise PartitionSeedError("events must contain PartitionEvent values")
            if not 0 <= event.pair < n_pairs:
                raise PartitionSeedError("event pair is outside packet geometry")
            if not 0 <= event.row < height or not 0 <= event.col < width:
                raise PartitionSeedError("event site is outside packet geometry")
            if not 0 <= event.target_class < 5 or not 0 <= event.baseline_class < 5:
                raise PartitionSeedError("event class id is outside the five-class alphabet")
            if event.target_class == event.baseline_class:
                raise PartitionSeedError("event target must differ from its baseline class")
            site = ((event.pair * height) + event.row) * width + event.col
            if site <= previous:
                raise PartitionSeedError("events must be unique and strictly site-sorted")
            previous = site
        object.__setattr__(self, "n_pairs", n_pairs)
        object.__setattr__(self, "height", height)
        object.__setattr__(self, "width", width)
        object.__setattr__(self, "semantic_class_ids", semantic_ids)
        object.__setattr__(self, "events", events)


def detect_partition_semantics(labels: np.ndarray) -> SemanticDetection:
    """Self-detect the five semantic classes from spatial/static signatures.

    The rule intentionally never consults RGB or luma.  Lane is the smallest
    area class; Undrivable is the largest; MyCar is the lowest-centroid member
    of the remaining three; Movable is then the smaller-area member; Road is
    the survivor.  Temporal IoU is measured and returned as an audit signal.
    The input may be a read-only memmap and is consumed one pair at a time.
    """

    array = np.asarray(labels)
    if array.ndim != 3 or array.shape[0] < 2 or array.shape[1] <= 0 or array.shape[2] <= 0:
        raise PartitionSeedError("labels must be pair x height x width with at least two pairs")
    n_pairs, height, width = (int(v) for v in array.shape)
    counts = np.zeros(5, dtype=np.int64)
    row_sums = np.zeros(5, dtype=np.float64)
    intersections = np.zeros(5, dtype=np.int64)
    unions = np.zeros(5, dtype=np.int64)
    previous: np.ndarray | None = None
    rows = np.arange(height, dtype=np.float64)[:, None]
    for pair in range(n_pairs):
        current = np.asarray(array[pair])
        if current.shape != (height, width) or current.dtype.kind not in ("i", "u"):
            raise PartitionSeedError("every label plane must be an integer height x width array")
        if current.size and (int(current.min()) < 0 or int(current.max()) >= 5):
            raise PartitionSeedError("label planes must contain exactly the five-class alphabet")
        for class_id in range(5):
            mask = current == class_id
            count = int(mask.sum())
            counts[class_id] += count
            row_sums[class_id] += float((mask * rows).sum())
            if previous is not None:
                prev_mask = previous == class_id
                intersections[class_id] += int(np.count_nonzero(mask & prev_mask))
                unions[class_id] += int(np.count_nonzero(mask | prev_mask))
        previous = current
    if np.any(counts == 0):
        raise PartitionSeedError("semantic self-detection requires every class to be present")
    area = counts.astype(np.float64) / float(n_pairs * height * width)
    centroid = row_sums / counts.astype(np.float64) / float(max(height - 1, 1))
    temporal_iou = intersections.astype(np.float64) / np.maximum(unions, 1)

    lane = int(np.argmin(area))
    undrivable = int(np.argmax(area))
    if lane == undrivable:  # pragma: no cover - impossible with positive unequal extrema
        raise PartitionSeedError("spatial signature is degenerate")
    remaining = [class_id for class_id in range(5) if class_id not in (lane, undrivable)]
    my_car = max(remaining, key=lambda class_id: (centroid[class_id], temporal_iou[class_id]))
    remaining.remove(my_car)
    movable = min(remaining, key=lambda class_id: (area[class_id], temporal_iou[class_id]))
    remaining.remove(movable)
    if len(remaining) != 1:
        raise PartitionSeedError("spatial signature did not produce one Road survivor")
    road = remaining[0]
    semantic_ids = (road, lane, undrivable, movable, my_car)
    per_class = tuple(
        {
            "class_id": class_id,
            "pixel_count": int(counts[class_id]),
            "area_fraction": float(area[class_id]),
            "vertical_centroid_fraction": float(centroid[class_id]),
            "temporal_iou": float(temporal_iou[class_id]),
        }
        for class_id in range(5)
    )
    return SemanticDetection(semantic_class_ids=semantic_ids, per_class=per_class)


def _write_uleb128(value: int, output: bytearray) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PartitionSeedError("ULEB128 input must be a non-negative integer")
    while True:
        byte = value & 0x7F
        value >>= 7
        output.append(byte | (0x80 if value else 0))
        if not value:
            return


def _read_uleb128(payload: bytes, offset: int) -> tuple[int, int]:
    value = 0
    shift = 0
    for _ in range(10):
        if offset >= len(payload):
            raise PartitionSeedError("truncated ULEB128 site delta")
        byte = payload[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, offset
        shift += 7
    raise PartitionSeedError("ULEB128 site delta exceeds uint64 width")


def _encode_events(seed: PartitionEventSeed) -> bytes:
    raw = bytearray()
    previous = -1
    for event in seed.events:
        site = ((event.pair * seed.height) + event.row) * seed.width + event.col
        _write_uleb128(site - previous - 1, raw)
        raw.append(event.target_class | (event.baseline_class << 3))
        previous = site
    return bytes(raw)


def encode_partition_seed(seed: PartitionEventSeed, *, zlib_level: int = 9) -> bytes:
    """Encode a deterministic finite packet with all headers/CRC counted."""

    if isinstance(zlib_level, bool) or not isinstance(zlib_level, int) or not 0 <= zlib_level <= 9:
        raise PartitionSeedError("zlib_level must be an integer in [0,9]")
    raw = _encode_events(seed)
    body = zlib.compress(raw, level=zlib_level)
    header = {
        "schema": PACKET_SCHEMA,
        "version": PACKET_VERSION,
        "codec": "site_delta_uleb128_class6_zlib",
        "zlib_level": zlib_level,
        "n_pairs": seed.n_pairs,
        "height": seed.height,
        "width": seed.width,
        "event_count": len(seed.events),
        "semantic_names": list(SEMANTIC_NAMES),
        "semantic_class_ids": list(seed.semantic_class_ids),
        "raw_event_bytes": len(raw),
        "raw_event_sha256": sha256(raw).hexdigest(),
        "compressed_event_bytes": len(body),
        "compressed_event_sha256": sha256(body).hexdigest(),
    }
    header_bytes = _canonical_json(header)
    prefix = _PREFIX.pack(PACKET_MAGIC, PACKET_VERSION, len(header_bytes), len(body))
    checksum = _CRC.pack(zlib.crc32(header_bytes + body) & 0xFFFFFFFF)
    return prefix + header_bytes + body + checksum


def decode_partition_seed(payload: bytes) -> PartitionEventSeed:
    """Strict parse-back: reject truncation, trailing bytes, CRC, or geometry drift."""

    if not isinstance(payload, bytes) or len(payload) < _PREFIX.size + _CRC.size:
        raise PartitionSeedError("partition seed is truncated or not bytes")
    magic, version, header_size, body_size = _PREFIX.unpack_from(payload)
    if magic != PACKET_MAGIC or version != PACKET_VERSION:
        raise PartitionSeedError("partition seed magic/version mismatch")
    expected = _PREFIX.size + header_size + body_size + _CRC.size
    if len(payload) != expected:
        raise PartitionSeedError("partition seed length mismatch or trailing bytes")
    header_start = _PREFIX.size
    body_start = header_start + header_size
    body_end = body_start + body_size
    header_bytes = payload[header_start:body_start]
    body = payload[body_start:body_end]
    (stored_crc,) = _CRC.unpack(payload[body_end:])
    if stored_crc != (zlib.crc32(header_bytes + body) & 0xFFFFFFFF):
        raise PartitionSeedError("partition seed CRC mismatch")
    try:
        header = json.loads(header_bytes.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PartitionSeedError("partition seed header is not ASCII JSON") from exc
    if not isinstance(header, dict) or frozenset(header) != _HEADER_FIELDS:
        raise PartitionSeedError("partition seed header fields mismatch")
    if _canonical_json(header) != header_bytes:
        raise PartitionSeedError("partition seed header is not canonical")
    if header["schema"] != PACKET_SCHEMA or header["version"] != PACKET_VERSION:
        raise PartitionSeedError("partition seed schema/version mismatch")
    if header["codec"] != "site_delta_uleb128_class6_zlib":
        raise PartitionSeedError("partition seed codec is unsupported")
    if header["compressed_event_bytes"] != len(body):
        raise PartitionSeedError("partition seed compressed length mismatch")
    if header["compressed_event_sha256"] != sha256(body).hexdigest():
        raise PartitionSeedError("partition seed compressed SHA-256 mismatch")
    decompressor = zlib.decompressobj()
    try:
        raw = decompressor.decompress(body, int(header["raw_event_bytes"]) + 1)
        raw += decompressor.flush()
    except zlib.error as exc:
        raise PartitionSeedError("partition seed zlib body is invalid") from exc
    if not decompressor.eof or decompressor.unused_data or decompressor.unconsumed_tail:
        raise PartitionSeedError("partition seed zlib stream is incomplete or has trailing data")
    if len(raw) != header["raw_event_bytes"]:
        raise PartitionSeedError("partition seed raw length mismatch")
    if header["raw_event_sha256"] != sha256(raw).hexdigest():
        raise PartitionSeedError("partition seed raw SHA-256 mismatch")

    n_pairs = _positive_int(header["n_pairs"], "n_pairs")
    height = _positive_int(header["height"], "height")
    width = _positive_int(header["width"], "width")
    if isinstance(header["event_count"], bool) or not isinstance(header["event_count"], int):
        raise PartitionSeedError("event_count must be a non-negative integer")
    event_count = header["event_count"]
    if event_count < 0:
        raise PartitionSeedError("event_count must be a non-negative integer")
    if tuple(header["semantic_names"]) != SEMANTIC_NAMES:
        raise PartitionSeedError("semantic name order drifted")
    semantic_ids = _semantic_ids(header["semantic_class_ids"])
    events: list[PartitionEvent] = []
    offset = 0
    previous = -1
    max_sites = n_pairs * height * width
    for _ in range(event_count):
        delta, offset = _read_uleb128(raw, offset)
        if offset >= len(raw):
            raise PartitionSeedError("partition seed is missing a class transition byte")
        packed = raw[offset]
        offset += 1
        if packed & 0xC0:
            raise PartitionSeedError("partition seed class transition reserved bits are nonzero")
        target = packed & 0x07
        baseline = (packed >> 3) & 0x07
        site = previous + delta + 1
        if site <= previous or site >= max_sites:
            raise PartitionSeedError("partition seed site delta is non-monotone or out of range")
        pair, within_pair = divmod(site, height * width)
        row, col = divmod(within_pair, width)
        events.append(PartitionEvent(pair, row, col, target, baseline))
        previous = site
    if offset != len(raw):
        raise PartitionSeedError("partition seed raw event stream has trailing bytes")
    return PartitionEventSeed(
        n_pairs=n_pairs,
        height=height,
        width=width,
        semantic_class_ids=semantic_ids,
        events=tuple(events),
    )


def apply_partition_seed(baseline_labels: np.ndarray, seed: PartitionEventSeed) -> np.ndarray:
    """Apply decoded cell constraints to a caller-provided baseline partition."""

    baseline = np.asarray(baseline_labels)
    if baseline.shape != (seed.n_pairs, seed.height, seed.width):
        raise PartitionSeedError("baseline partition geometry does not match the seed")
    if baseline.dtype.kind not in ("i", "u"):
        raise PartitionSeedError("baseline partition must contain integer class ids")
    if baseline.size and (int(baseline.min()) < 0 or int(baseline.max()) >= 5):
        raise PartitionSeedError("baseline partition class ids must be in [0,4]")
    output = baseline.astype(np.uint8, copy=True)
    for event in seed.events:
        if int(output[event.pair, event.row, event.col]) != event.baseline_class:
            raise PartitionSeedError("baseline class at a seeded site does not match custody")
        output[event.pair, event.row, event.col] = event.target_class
    return output


def events_from_rows(rows: Iterable[Sequence[int | float]]) -> tuple[PartitionEvent, ...]:
    """Convert the measured G3 inventory rows without serializing margin values."""

    events = []
    for row in rows:
        values = tuple(row)
        if len(values) < 5:
            raise PartitionSeedError("G3 row must contain pair,row,col,target,baseline")
        events.append(
            PartitionEvent(
                pair=int(values[0]),
                row=int(values[1]),
                col=int(values[2]),
                target_class=int(values[3]),
                baseline_class=int(values[4]),
            )
        )
    return tuple(events)


def packet_accounting(payload: bytes) -> dict[str, Any]:
    """Return exact counted-byte anatomy after strict parse-back."""

    seed = decode_partition_seed(payload)
    _, _, header_size, body_size = _PREFIX.unpack_from(payload)
    return {
        "packet_bytes": len(payload),
        "packet_sha256": sha256(payload).hexdigest(),
        "prefix_bytes": _PREFIX.size,
        "header_bytes": header_size,
        "compressed_event_bytes": body_size,
        "crc_bytes": _CRC.size,
        "event_count": len(seed.events),
        "bytes_per_event": len(payload) / max(len(seed.events), 1),
        "counted_seed_bytes": len(payload),
        "stored_plane_value_bytes": 0,
        "generic_interpreter_algorithm": "site delta decode + cell constraint application",
    }


__all__ = [
    "PACKET_MAGIC",
    "PACKET_SCHEMA",
    "PACKET_VERSION",
    "SEMANTIC_NAMES",
    "PartitionEvent",
    "PartitionEventSeed",
    "PartitionSeedError",
    "SemanticDetection",
    "apply_partition_seed",
    "decode_partition_seed",
    "detect_partition_semantics",
    "encode_partition_seed",
    "events_from_rows",
    "packet_accounting",
]
