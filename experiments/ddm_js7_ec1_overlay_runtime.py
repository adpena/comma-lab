"""Generic counted EC1 event-overlay decoder for the CP135 runtime.

The algorithm is free runtime code.  Every video-derived event byte is carried
inside ``archive.zip`` as one typed packet appended to member ``p``.
"""

from __future__ import annotations

import dataclasses
import lzma
import struct
import zipfile
from pathlib import Path
from typing import Final

import brotli
import numpy as np

N: Final = 600
H: Final = 384
W: Final = 512
CLASSES: Final = 5
EVENT_TYPES: Final = 5
EVENT_MAGIC: Final = b"EC1PROP1"
PACKET_MAGIC: Final = b"EC1OVR1\0"
FOOTER_MAGIC: Final = b"EC1END1\0"
PACKET_HEADER: Final = struct.Struct("<8sBBHI")
FOOTER: Final = struct.Struct("<8sI")
CODER_ID: Final = {"raw": 0, "brotli_q11": 1, "lzma1_raw": 2}
CODER_NAME: Final = {value: key for key, value in CODER_ID.items()}
LZMA1_FILTERS: Final = [
    {
        "id": lzma.FILTER_LZMA1,
        "dict_size": 1 << 20,
        "lc": 3,
        "lp": 0,
        "pb": 2,
        "mode": lzma.MODE_NORMAL,
        "nice_len": 273,
        "mf": lzma.MF_BT4,
        "depth": 0,
    }
]


class EC1OverlayError(ValueError):
    """The counted overlay is malformed, non-canonical, or inapplicable."""


@dataclasses.dataclass(frozen=True, slots=True)
class Event:
    frame: int
    source_class: int
    target_class: int
    event_type: int
    indices: np.ndarray


def _get_uvarint(payload: bytes, offset: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while True:
        if offset >= len(payload) or shift > 63:
            raise EC1OverlayError("truncated or oversized event uvarint")
        byte = payload[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, offset
        shift += 7


def _decode_event(payload: bytes, offset: int) -> tuple[Event, int]:
    if offset + 13 > len(payload) or payload[offset : offset + 8] != EVENT_MAGIC:
        raise EC1OverlayError("event header differs")
    frame, source_class, target_class, event_type = struct.unpack_from("<HBBB", payload, offset + 8)
    if (
        frame >= N
        or source_class >= CLASSES
        or target_class >= CLASSES
        or source_class == target_class
        or event_type >= EVENT_TYPES
    ):
        raise EC1OverlayError("event address differs")
    cursor = offset + 13
    count, cursor = _get_uvarint(payload, cursor)
    if not count or count > H * W:
        raise EC1OverlayError("event site count differs")
    indices = np.empty(count, dtype=np.int64)
    previous = 0
    for position in range(count):
        gap, cursor = _get_uvarint(payload, cursor)
        value = gap if position == 0 else previous + gap
        if value >= H * W or (position and value <= previous):
            raise EC1OverlayError("event coordinate ordering differs")
        indices[position] = value
        previous = value
    return Event(frame, source_class, target_class, event_type, indices), cursor


def _coder_payloads(raw: bytes) -> dict[str, bytes]:
    return {
        "raw": raw,
        "brotli_q11": brotli.compress(raw, quality=11),
        "lzma1_raw": lzma.compress(raw, format=lzma.FORMAT_RAW, filters=LZMA1_FILTERS),
    }


def build_packet_candidates(event_payloads: list[bytes]) -> tuple[bytes, dict[str, bytes]]:
    """Build every retained real-coder packet for a deterministic event order."""
    if not event_payloads or len(event_payloads) > 65_535:
        raise EC1OverlayError("overlay event count must be in [1, 65535]")
    raw = b"".join(event_payloads)
    candidates = {
        name: PACKET_HEADER.pack(PACKET_MAGIC, 1, CODER_ID[name], len(event_payloads), len(raw)) + coded
        for name, coded in _coder_payloads(raw).items()
    }
    for packet in candidates.values():
        decode_packet(packet)
    return raw, candidates


def decode_packet(packet: bytes) -> tuple[list[Event], dict[str, int | str]]:
    if len(packet) < PACKET_HEADER.size:
        raise EC1OverlayError("overlay packet is truncated")
    magic, version, coder_id, count, raw_bytes = PACKET_HEADER.unpack_from(packet)
    if magic != PACKET_MAGIC or version != 1 or coder_id not in CODER_NAME or not count or not raw_bytes:
        raise EC1OverlayError("overlay packet header differs")
    coded = packet[PACKET_HEADER.size :]
    coder = CODER_NAME[coder_id]
    try:
        if coder == "raw":
            raw = coded
        elif coder == "brotli_q11":
            raw = brotli.decompress(coded)
        else:
            raw = lzma.decompress(coded, format=lzma.FORMAT_RAW, filters=LZMA1_FILTERS)
    except (brotli.error, lzma.LZMAError) as error:
        raise EC1OverlayError(f"{coder} rejected the overlay stream") from error
    if len(raw) != raw_bytes:
        raise EC1OverlayError("overlay raw-byte count differs")
    events: list[Event] = []
    cursor = 0
    for _ in range(count):
        event, cursor = _decode_event(raw, cursor)
        events.append(event)
    if cursor != len(raw):
        raise EC1OverlayError("overlay event stream has trailing bytes")
    return events, {
        "event_count": count,
        "raw_bytes": raw_bytes,
        "coded_bytes": len(coded),
        "packet_bytes": len(packet),
        "coder": coder,
    }


def append_overlay_member(base_member: bytes, packet: bytes) -> bytes:
    if not base_member:
        raise EC1OverlayError("base archive member is empty")
    decode_packet(packet)
    return base_member + packet + FOOTER.pack(FOOTER_MAGIC, len(packet))


def split_overlay_member(member: bytes) -> tuple[bytes, bytes | None]:
    if len(member) < FOOTER.size or member[-FOOTER.size : -4] != FOOTER_MAGIC:
        return member, None
    magic, packet_bytes = FOOTER.unpack_from(member, len(member) - FOOTER.size)
    if magic != FOOTER_MAGIC or not packet_bytes or packet_bytes + FOOTER.size >= len(member):
        raise EC1OverlayError("overlay footer differs")
    packet_start = len(member) - FOOTER.size - packet_bytes
    packet = member[packet_start : packet_start + packet_bytes]
    decode_packet(packet)
    return member[:packet_start], packet


def read_overlay_archive(archive_path: Path) -> tuple[bytes, bytes | None]:
    with zipfile.ZipFile(archive_path) as archive:
        if archive.namelist() != ["p"]:
            raise EC1OverlayError("overlay archive must contain exactly member p")
        return split_overlay_member(archive.read("p"))


def apply_events_inplace(tokens: object, events: list[Event]) -> dict[str, object]:
    if isinstance(tokens, np.ndarray):
        array = tokens
    elif hasattr(tokens, "numpy"):
        array = tokens.numpy()  # CPU torch tensors expose a shared NumPy view.
    else:
        raise EC1OverlayError("semantic token plane does not expose receiver bytes")
    if array.shape != (N, H, W) or array.dtype != np.uint8:
        raise EC1OverlayError("semantic token plane geometry differs")
    seen: set[tuple[int, int]] = set()
    per_frame: dict[int, int] = {}
    for event in events:
        flat = array[event.frame].reshape(-1)
        for index in event.indices.tolist():
            key = (event.frame, int(index))
            if key in seen:
                raise EC1OverlayError("two overlay events address the same semantic site")
            seen.add(key)
        if np.any(flat[event.indices] != event.source_class):
            raise EC1OverlayError("overlay source-class precondition differs")
        flat[event.indices] = event.target_class
        per_frame[event.frame] = per_frame.get(event.frame, 0) + len(event.indices)
    return {
        "event_count": len(events),
        "site_count": len(seen),
        "touched_frames": sorted(per_frame),
        "sites_per_frame": {str(frame): count for frame, count in sorted(per_frame.items())},
    }


def apply_packet_inplace(tokens: object, packet: bytes) -> dict[str, object]:
    events, packet_report = decode_packet(packet)
    return {**packet_report, **apply_events_inplace(tokens, events)}
