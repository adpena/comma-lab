# SPDX-License-Identifier: MIT
"""Receiver-checkable NR1 task-cell quotient packet.

The packet reconstructs a categorical ``(pair, y, x)`` task-token field for
the existing semantic renderer.  It is deliberately scorer-free: this module
proves a real, deterministic representation and receiver boundary, not matched
distortion or contest score.

Four paid surfaces have distinct jobs:

* ``QPARAM`` stores a learned tile dictionary;
* ``QCTX`` stores the learned default dictionary entry at every tile address;
* ``QPAIR`` stores temporal/context residual choices for every pair and tile;
* ``QEVENT`` stores sparse task-priority token corrections.

Every surface is independently real-coded and authenticated.  The parser is
strict and canonical, and the decoder records exact-once consumer use.
"""

from __future__ import annotations

import hashlib
import lzma
import struct
import zlib
from dataclasses import dataclass
from enum import IntEnum, StrEnum
from itertools import pairwise
from typing import Final

import brotli
import numpy as np

MAGIC: Final = b"NR1Q"
VERSION: Final = 1
ACTIVE_MODE: Final = 1
INACTIVE_MODE: Final = 0
CLASS_COUNT: Final = 5
SECTION_ORDER: Final = ("QPARAM", "QCTX", "QPAIR", "QEVENT")

_OUTER: Final = struct.Struct(">4sBBBBHHH32s")
_SECTION: Final = struct.Struct(">8sB3xII32s32s")
_QPARAM: Final = struct.Struct(">4sBBBB")
_QCTX: Final = struct.Struct(">4sHH")
_QPAIR: Final = struct.Struct(">4sHHH")
_QEVENT: Final = struct.Struct(">4sI")
_ZERO_SHA: Final = b"\x00" * 32


class NR1FormatError(ValueError):
    """Malformed, noncanonical, inactive-mismatched, or inert NR1 payload."""


class Coder(IntEnum):
    RAW = 0
    ZLIB_9 = 1
    LZMA1_1M = 2
    BROTLI_Q11 = 3


class Section(StrEnum):
    QPARAM = "QPARAM"
    QCTX = "QCTX"
    QPAIR = "QPAIR"
    QEVENT = "QEVENT"


@dataclass(frozen=True, slots=True)
class CodedCandidate:
    coder: Coder
    payload: bytes


@dataclass(frozen=True, slots=True)
class ParsedSection:
    name: Section
    coder: Coder
    raw: bytes
    coded: bytes
    header_start: int
    payload_start: int
    payload_end: int


@dataclass(frozen=True, slots=True)
class ParsedPacket:
    pair_count: int
    height: int
    width: int
    sections: tuple[ParsedSection, ...]


@dataclass(frozen=True, slots=True)
class ConsumptionTrace:
    qparam: int
    qctx: int
    qpair: int
    qevent: int

    def require_exact_once(self) -> None:
        if (self.qparam, self.qctx, self.qpair, self.qevent) != (1, 1, 1, 1):
            raise NR1FormatError(f"paid-surface consumption is not exact-once: {self}")


@dataclass(frozen=True, slots=True)
class DecodeResult:
    tokens: np.ndarray
    trace: ConsumptionTrace


def sha256_bytes(payload: bytes) -> bytes:
    return hashlib.sha256(payload).digest()


def coder_candidates(raw: bytes) -> tuple[CodedCandidate, ...]:
    """Return all deterministic real-coder candidates; callers retain all."""
    filters = [{"id": lzma.FILTER_LZMA1, "dict_size": 1 << 20}]
    return (
        CodedCandidate(Coder.RAW, raw),
        CodedCandidate(Coder.ZLIB_9, zlib.compress(raw, level=9)),
        CodedCandidate(
            Coder.LZMA1_1M,
            lzma.compress(raw, format=lzma.FORMAT_RAW, filters=filters),
        ),
        CodedCandidate(Coder.BROTLI_Q11, bytes(brotli.compress(raw, quality=11))),
    )


def choose_candidate(candidates: tuple[CodedCandidate, ...]) -> CodedCandidate:
    if {candidate.coder for candidate in candidates} != set(Coder):
        raise NR1FormatError("coder race must contain every declared coder exactly once")
    return min(candidates, key=lambda candidate: (len(candidate.payload), int(candidate.coder)))


def _decompress(coder: Coder, coded: bytes) -> bytes:
    if coder is Coder.RAW:
        return coded
    if coder is Coder.ZLIB_9:
        return zlib.decompress(coded)
    if coder is Coder.LZMA1_1M:
        filters = [{"id": lzma.FILTER_LZMA1, "dict_size": 1 << 20}]
        return lzma.decompress(coded, format=lzma.FORMAT_RAW, filters=filters)
    if coder is Coder.BROTLI_Q11:
        return bytes(brotli.decompress(coded))
    raise NR1FormatError(f"unknown coder {coder!r}")


def build_explicit_inactive(base_payload: bytes, pair_count: int, height: int, width: int) -> bytes:
    """Build an explicit inactive marker bound to the exact base payload."""
    _validate_shape(pair_count, height, width)
    return _OUTER.pack(
        MAGIC,
        VERSION,
        INACTIVE_MODE,
        0,
        0,
        pair_count,
        height,
        width,
        sha256_bytes(base_payload),
    )


def receive_inactive_or_base(
    payload: bytes | None,
    base_payload: bytes,
    pair_count: int,
    height: int,
    width: int,
) -> bytes:
    """Return the exact base bytes for absent or explicit-inactive NR1."""
    if payload is None:
        return base_payload
    if len(payload) != _OUTER.size:
        raise NR1FormatError("explicit inactive marker has a noncanonical length")
    magic, version, mode, count, reserved, n, h, w, base_sha = _OUTER.unpack(payload)
    if magic != MAGIC or version != VERSION or mode != INACTIVE_MODE:
        raise NR1FormatError("payload is not an explicit inactive NR1 marker")
    if count != 0 or reserved != 0 or (n, h, w) != (pair_count, height, width):
        raise NR1FormatError("explicit inactive marker metadata differs")
    if base_sha != sha256_bytes(base_payload):
        raise NR1FormatError("explicit inactive marker is bound to different base bytes")
    return base_payload


def build_packet(
    raw_sections: dict[Section, bytes],
    pair_count: int,
    height: int,
    width: int,
    *,
    selected: dict[Section, CodedCandidate] | None = None,
) -> bytes:
    """Build a canonical active packet from all four nonempty raw surfaces."""
    _validate_shape(pair_count, height, width)
    if tuple(section.value for section in raw_sections) != SECTION_ORDER:
        raise NR1FormatError("sections must be present once in canonical order")
    chosen: dict[Section, CodedCandidate] = {}
    for section in Section:
        raw = raw_sections[section]
        if not isinstance(raw, bytes) or not raw:
            raise NR1FormatError(f"active {section.value} must be nonempty bytes")
        candidate = selected[section] if selected is not None else choose_candidate(coder_candidates(raw))
        if _decompress(candidate.coder, candidate.payload) != raw:
            raise NR1FormatError(f"selected {section.value} coder does not round-trip")
        chosen[section] = candidate

    out = bytearray(
        _OUTER.pack(
            MAGIC,
            VERSION,
            ACTIVE_MODE,
            len(Section),
            0,
            pair_count,
            height,
            width,
            _ZERO_SHA,
        )
    )
    for section in Section:
        raw = raw_sections[section]
        coded = chosen[section]
        name = section.value.encode().ljust(8, b"\x00")
        out += _SECTION.pack(
            name,
            int(coded.coder),
            len(raw),
            len(coded.payload),
            sha256_bytes(raw),
            sha256_bytes(coded.payload),
        )
        out += coded.payload
    return bytes(out)


def parse_packet(payload: bytes) -> ParsedPacket:
    """Parse an active packet and require byte-for-byte canonical rebuild."""
    if len(payload) < _OUTER.size:
        raise NR1FormatError("truncated NR1 outer header")
    magic, version, mode, count, reserved, n, h, w, base_sha = _OUTER.unpack_from(payload)
    if magic != MAGIC:
        raise NR1FormatError("bad NR1 magic")
    if version != VERSION:
        raise NR1FormatError("unsupported NR1 version")
    if mode != ACTIVE_MODE:
        if mode == INACTIVE_MODE:
            raise NR1FormatError("inactive NR1 cannot be parsed as active")
        raise NR1FormatError("unknown NR1 mode")
    if count != len(Section) or reserved != 0 or base_sha != _ZERO_SHA:
        raise NR1FormatError("active NR1 outer metadata differs")
    _validate_shape(n, h, w)

    cursor = _OUTER.size
    parsed: list[ParsedSection] = []
    selected: dict[Section, CodedCandidate] = {}
    raw_sections: dict[Section, bytes] = {}
    for expected in Section:
        header_start = cursor
        if cursor + _SECTION.size > len(payload):
            raise NR1FormatError(f"truncated {expected.value} header")
        name_raw, coder_raw, raw_len, coded_len, raw_sha, coded_sha = _SECTION.unpack_from(
            payload, cursor
        )
        cursor += _SECTION.size
        try:
            name = Section(name_raw.rstrip(b"\x00").decode("ascii"))
            coder = Coder(coder_raw)
        except (UnicodeDecodeError, ValueError) as exc:
            raise NR1FormatError("unknown section or coder") from exc
        if name is not expected:
            raise NR1FormatError(f"section order differs: expected {expected.value}, got {name.value}")
        if raw_len <= 0 or coded_len <= 0:
            raise NR1FormatError(f"active {name.value} has an empty declared length")
        payload_start = cursor
        payload_end = cursor + coded_len
        if payload_end > len(payload):
            raise NR1FormatError(f"truncated {name.value} payload")
        coded = payload[payload_start:payload_end]
        cursor = payload_end
        if sha256_bytes(coded) != coded_sha:
            raise NR1FormatError(f"{name.value} coded SHA differs")
        try:
            raw = _decompress(coder, coded)
        except (brotli.error, lzma.LZMAError, zlib.error) as exc:
            raise NR1FormatError(f"{name.value} coder refusal") from exc
        if len(raw) != raw_len or sha256_bytes(raw) != raw_sha:
            raise NR1FormatError(f"{name.value} raw length or SHA differs")
        parsed.append(
            ParsedSection(name, coder, raw, coded, header_start, payload_start, payload_end)
        )
        selected[name] = CodedCandidate(coder, coded)
        raw_sections[name] = raw
    if cursor != len(payload):
        raise NR1FormatError("NR1 packet has trailing bytes")
    rebuilt = build_packet(raw_sections, n, h, w, selected=selected)
    if rebuilt != payload:
        raise NR1FormatError("NR1 packet is not canonical parse-to-repack identical")
    return ParsedPacket(n, h, w, tuple(parsed))


def physical_attribution(packet: bytes) -> dict[Section, tuple[int, int]]:
    """Attribute every physical packet byte exactly once to a logical surface."""
    parsed = parse_packet(packet)
    out: dict[Section, tuple[int, int]] = {}
    for index, section in enumerate(parsed.sections):
        start = 0 if index == 0 else section.header_start
        out[section.name] = (start, section.payload_end)
    ranges = list(out.values())
    if ranges[0][0] != 0 or ranges[-1][1] != len(packet):
        raise NR1FormatError("physical attribution does not cover the packet")
    for left, right in pairwise(ranges):
        if left[1] != right[0]:
            raise NR1FormatError("physical attribution has a gap or overlap")
    if sum(end - start for start, end in ranges) != len(packet):
        raise NR1FormatError("logical attribution does not sum to physical bytes")
    return out


def encode_raw_sections(
    tokens: np.ndarray,
    *,
    tile_height: int = 8,
    tile_width: int = 8,
    codebook_size: int = 64,
    event_indices: np.ndarray | None = None,
) -> tuple[dict[Section, bytes], np.ndarray]:
    """Fit the deterministic top-pattern/majority task-cell quotient."""
    field = np.asarray(tokens)
    if field.ndim != 3 or field.dtype != np.uint8:
        raise NR1FormatError("tokens must be a uint8 (pair,height,width) array")
    n, h, w = map(int, field.shape)
    _validate_shape(n, h, w)
    if h % tile_height or w % tile_width:
        raise NR1FormatError("tile dimensions must divide the token field")
    if not 6 <= codebook_size <= 253:
        raise NR1FormatError("codebook_size must be in [6,253]")
    if int(field.max(initial=0)) >= CLASS_COUNT:
        raise NR1FormatError("tokens contain a class outside [0,4]")

    gh, gw = h // tile_height, w // tile_width
    tiles = (
        field.reshape(n, gh, tile_height, gw, tile_width)
        .transpose(0, 1, 3, 2, 4)
        .reshape(-1, tile_height * tile_width)
    )
    packed = _pack_three_bit_labels(tiles)
    unique, inverse, counts = np.unique(
        packed,
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    order = np.lexsort((np.arange(len(counts), dtype=np.int64), -counts.astype(np.int64)))
    selected_unique: list[int] = []
    constant_packed = {
        bytes(_pack_three_bit_labels(np.full((1, tiles.shape[1]), cls, dtype=np.uint8))[0]): cls
        for cls in range(CLASS_COUNT)
    }
    for unique_index in order:
        if bytes(unique[unique_index]) in constant_packed:
            continue
        selected_unique.append(int(unique_index))
        if len(selected_unique) == codebook_size - CLASS_COUNT:
            break

    codebook = np.empty((codebook_size, tiles.shape[1]), dtype=np.uint8)
    for cls in range(CLASS_COUNT):
        codebook[cls].fill(cls)
    filled = CLASS_COUNT
    if selected_unique:
        decoded_selected = _unpack_three_bit_labels(
            unique[np.asarray(selected_unique)],
            tiles.shape[1],
        )
        codebook[filled : filled + len(decoded_selected)] = decoded_selected
        filled += len(decoded_selected)
    if filled < codebook_size:
        codebook = codebook[:filled]
    k = len(codebook)

    unique_to_code = np.full(len(unique), 255, dtype=np.uint8)
    for cls in range(CLASS_COUNT):
        key = bytes(_pack_three_bit_labels(np.full((1, tiles.shape[1]), cls, dtype=np.uint8))[0])
        hits = np.flatnonzero(np.all(unique == np.frombuffer(key, dtype=np.uint8), axis=1))
        if hits.size:
            unique_to_code[hits[0]] = cls
    for offset, unique_index in enumerate(selected_unique, start=CLASS_COUNT):
        unique_to_code[unique_index] = offset

    assignments = unique_to_code[inverse]
    fallback = assignments == 255
    if np.any(fallback):
        fallback_tiles = tiles[fallback]
        class_counts = np.stack(
            [np.count_nonzero(fallback_tiles == cls, axis=1) for cls in range(CLASS_COUNT)],
            axis=1,
        )
        assignments[fallback] = np.argmax(class_counts, axis=1).astype(np.uint8)
    assignments = assignments.reshape(n, gh, gw)

    baseline = np.empty((gh, gw), dtype=np.uint8)
    for row in range(gh):
        for col in range(gw):
            baseline[row, col] = np.bincount(
                assignments[:, row, col], minlength=k
            ).argmax()
    symbols = np.empty_like(assignments)
    previous = np.zeros((gh, gw), dtype=np.uint8)
    for pair in range(n):
        current = assignments[pair]
        use_previous = (
            np.equal(current, previous) if pair > 0 else np.zeros_like(current, dtype=bool)
        )
        use_baseline = np.equal(current, baseline) & ~use_previous
        symbols[pair] = current + 2
        symbols[pair][use_baseline] = 1
        symbols[pair][use_previous] = 0
        previous = current

    qparam = _QPARAM.pack(b"QPV1", tile_height, tile_width, k, CLASS_COUNT) + codebook.tobytes()
    qctx = _QCTX.pack(b"QCV1", gh, gw) + baseline.tobytes()
    qpair = _QPAIR.pack(b"QRV1", n, gh, gw) + symbols.tobytes()

    decoded_base = _render_assignments(codebook, assignments, n, gh, gw, tile_height, tile_width)
    event_indices_array = (
        np.empty(0, dtype=np.uint32)
        if event_indices is None
        else np.asarray(event_indices, dtype=np.uint32)
    )
    event_indices_array = np.unique(event_indices_array)
    if event_indices_array.size:
        if int(event_indices_array[-1]) >= field.size:
            raise NR1FormatError("QEVENT index is outside the token field")
        flat_source = field.reshape(-1)
        flat_base = decoded_base.reshape(-1)
        event_indices_array = event_indices_array[
            flat_source[event_indices_array] != flat_base[event_indices_array]
        ]
    if event_indices_array.size == 0:
        mismatches = np.flatnonzero(field.reshape(-1) != decoded_base.reshape(-1))
        if mismatches.size == 0:
            raise NR1FormatError("active QEVENT needs at least one live correction")
        event_indices_array = mismatches[:1].astype(np.uint32)
    qevent = _encode_events(event_indices_array, field.reshape(-1)[event_indices_array])
    raw_sections = {
        Section.QPARAM: qparam,
        Section.QCTX: qctx,
        Section.QPAIR: qpair,
        Section.QEVENT: qevent,
    }
    return raw_sections, decoded_base


def replace_qevent(
    raw_sections: dict[Section, bytes],
    source_tokens: np.ndarray,
    decoded_base: np.ndarray,
    event_indices: np.ndarray,
) -> dict[Section, bytes]:
    """Replace QEVENT after an external task-priority selector chooses debt."""
    if tuple(section.value for section in raw_sections) != SECTION_ORDER:
        raise NR1FormatError("sections must be present once in canonical order")
    source = np.asarray(source_tokens)
    base = np.asarray(decoded_base)
    if source.shape != base.shape or source.dtype != np.uint8 or base.dtype != np.uint8:
        raise NR1FormatError("source/base token fields must be aligned uint8 arrays")
    indices = np.unique(np.asarray(event_indices, dtype=np.uint32))
    if indices.size == 0 or int(indices[-1]) >= source.size:
        raise NR1FormatError("replacement QEVENT indices must be nonempty and in range")
    source_flat = source.reshape(-1)
    base_flat = base.reshape(-1)
    indices = indices[source_flat[indices] != base_flat[indices]]
    if indices.size == 0:
        raise NR1FormatError("replacement QEVENT contains no live correction")
    replaced = dict(raw_sections)
    replaced[Section.QEVENT] = _encode_events(indices, source_flat[indices])
    return replaced


def decode_packet(packet: bytes) -> DecodeResult:
    """Decode all paid surfaces exactly once into renderer-consumable tokens."""
    parsed = parse_packet(packet)
    raws = {section.name: section.raw for section in parsed.sections}
    trace = dict.fromkeys(Section, 0)

    qparam_result = _consume_qparam(raws[Section.QPARAM])
    if not isinstance(qparam_result, tuple) or len(qparam_result) != 3:
        raise NR1FormatError("QPARAM consumer did not return its declared value")
    tile_height, tile_width, codebook = qparam_result
    trace[Section.QPARAM] += 1
    qctx_result = _consume_qctx(raws[Section.QCTX], len(codebook))
    if not isinstance(qctx_result, tuple) or len(qctx_result) != 3:
        raise NR1FormatError("QCTX consumer did not return its declared value")
    gh, gw, baseline = qctx_result
    trace[Section.QCTX] += 1
    assignments = _consume_qpair(
        raws[Section.QPAIR], parsed.pair_count, gh, gw, baseline, len(codebook)
    )
    if not isinstance(assignments, np.ndarray) or assignments.shape != (parsed.pair_count, gh, gw):
        raise NR1FormatError("QPAIR consumer did not return its declared value")
    trace[Section.QPAIR] += 1
    if gh * tile_height != parsed.height or gw * tile_width != parsed.width:
        raise NR1FormatError("QPARAM/QCTX grid does not cover the declared token field")
    tokens = _render_assignments(
        codebook,
        assignments,
        parsed.pair_count,
        gh,
        gw,
        tile_height,
        tile_width,
    )
    event_count = _consume_qevent(raws[Section.QEVENT], tokens)
    if not isinstance(event_count, int) or event_count <= 0:
        raise NR1FormatError("QEVENT consumer did not acknowledge live corrections")
    trace[Section.QEVENT] += 1
    receipt = ConsumptionTrace(
        trace[Section.QPARAM],
        trace[Section.QCTX],
        trace[Section.QPAIR],
        trace[Section.QEVENT],
    )
    receipt.require_exact_once()
    tokens.setflags(write=False)
    return DecodeResult(tokens, receipt)


def _consume_qparam(raw: bytes) -> tuple[int, int, np.ndarray]:
    if len(raw) < _QPARAM.size:
        raise NR1FormatError("truncated QPARAM")
    magic, th, tw, k, classes = _QPARAM.unpack_from(raw)
    if magic != b"QPV1" or classes != CLASS_COUNT or th == 0 or tw == 0 or not 6 <= k <= 253:
        raise NR1FormatError("invalid QPARAM metadata")
    expected = _QPARAM.size + k * th * tw
    if len(raw) != expected:
        raise NR1FormatError("QPARAM length differs")
    codebook = np.frombuffer(raw, dtype=np.uint8, offset=_QPARAM.size).reshape(k, th * tw)
    if int(codebook.max(initial=0)) >= CLASS_COUNT:
        raise NR1FormatError("QPARAM codebook contains an invalid class")
    return th, tw, codebook


def _consume_qctx(raw: bytes, codebook_size: int) -> tuple[int, int, np.ndarray]:
    if len(raw) < _QCTX.size:
        raise NR1FormatError("truncated QCTX")
    magic, gh, gw = _QCTX.unpack_from(raw)
    if magic != b"QCV1" or gh == 0 or gw == 0 or len(raw) != _QCTX.size + gh * gw:
        raise NR1FormatError("invalid QCTX metadata or length")
    baseline = np.frombuffer(raw, dtype=np.uint8, offset=_QCTX.size).reshape(gh, gw)
    if int(baseline.max(initial=0)) >= codebook_size:
        raise NR1FormatError("QCTX references an unknown codebook entry")
    return gh, gw, baseline


def _consume_qpair(
    raw: bytes,
    pair_count: int,
    gh: int,
    gw: int,
    baseline: np.ndarray,
    codebook_size: int,
) -> np.ndarray:
    if len(raw) < _QPAIR.size:
        raise NR1FormatError("truncated QPAIR")
    magic, n, pair_gh, pair_gw = _QPAIR.unpack_from(raw)
    if (
        magic != b"QRV1"
        or (n, pair_gh, pair_gw) != (pair_count, gh, gw)
        or len(raw) != _QPAIR.size + n * gh * gw
    ):
        raise NR1FormatError("invalid QPAIR metadata or length")
    symbols = np.frombuffer(raw, dtype=np.uint8, offset=_QPAIR.size).reshape(n, gh, gw)
    if int(symbols.max(initial=0)) >= codebook_size + 2:
        raise NR1FormatError("QPAIR references an unknown codebook entry")
    assignments = np.empty_like(symbols)
    previous = np.zeros((gh, gw), dtype=np.uint8)
    for pair in range(n):
        current = np.where(symbols[pair] == 1, baseline, symbols[pair] - 2).astype(np.uint8)
        use_previous = symbols[pair] == 0
        if pair == 0 and np.any(use_previous):
            raise NR1FormatError("QPAIR pair zero cannot reference previous state")
        current[use_previous] = previous[use_previous]
        assignments[pair] = current
        previous = current
    return assignments


def _consume_qevent(raw: bytes, tokens: np.ndarray) -> int:
    if len(raw) < _QEVENT.size:
        raise NR1FormatError("truncated QEVENT")
    magic, count = _QEVENT.unpack_from(raw)
    if magic != b"QEV1" or count == 0:
        raise NR1FormatError("invalid or empty active QEVENT")
    cursor = _QEVENT.size
    previous = -1
    flat = tokens.reshape(-1)
    for _ in range(count):
        delta, cursor = _decode_uleb(raw, cursor)
        index = previous + 1 + delta
        if index <= previous or index >= flat.size or cursor >= len(raw):
            raise NR1FormatError("QEVENT coordinate is noncanonical or outside the field")
        value = raw[cursor]
        cursor += 1
        if value >= CLASS_COUNT or value == int(flat[index]):
            raise NR1FormatError("QEVENT value is invalid or semantically inert")
        flat[index] = value
        previous = index
    if cursor != len(raw):
        raise NR1FormatError("QEVENT has trailing bytes")
    return count


def _encode_events(indices: np.ndarray, values: np.ndarray) -> bytes:
    if len(indices) != len(values) or len(indices) == 0:
        raise NR1FormatError("QEVENT requires aligned nonempty indices and values")
    out = bytearray(_QEVENT.pack(b"QEV1", len(indices)))
    previous = -1
    for index_raw, value_raw in zip(indices, values, strict=True):
        index = int(index_raw)
        value = int(value_raw)
        if index <= previous or not 0 <= value < CLASS_COUNT:
            raise NR1FormatError("QEVENT indices must be sorted unique and values valid")
        out += _encode_uleb(index - previous - 1)
        out.append(value)
        previous = index
    return bytes(out)


def _encode_uleb(value: int) -> bytes:
    if value < 0:
        raise NR1FormatError("ULEB value must be nonnegative")
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def _decode_uleb(payload: bytes, cursor: int) -> tuple[int, int]:
    value = 0
    shift = 0
    start = cursor
    while cursor < len(payload) and shift <= 28:
        byte = payload[cursor]
        cursor += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            if _encode_uleb(value) != payload[start:cursor]:
                raise NR1FormatError("QEVENT ULEB is noncanonical")
            return value, cursor
        shift += 7
    raise NR1FormatError("truncated or oversized QEVENT ULEB")


def _render_assignments(
    codebook: np.ndarray,
    assignments: np.ndarray,
    n: int,
    gh: int,
    gw: int,
    th: int,
    tw: int,
) -> np.ndarray:
    tiles = codebook[assignments.reshape(-1)].reshape(n, gh, gw, th, tw)
    return np.ascontiguousarray(tiles.transpose(0, 1, 3, 2, 4).reshape(n, gh * th, gw * tw))


def _pack_three_bit_labels(labels: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [np.packbits((labels >> bit) & 1, axis=1, bitorder="little") for bit in range(3)],
        axis=1,
    )


def _unpack_three_bit_labels(packed: np.ndarray, label_count: int) -> np.ndarray:
    width = (label_count + 7) // 8
    out = np.zeros((len(packed), label_count), dtype=np.uint8)
    for bit in range(3):
        plane = np.unpackbits(
            packed[:, bit * width : (bit + 1) * width],
            axis=1,
            count=label_count,
            bitorder="little",
        )
        out |= plane.astype(np.uint8) << bit
    return out


def _validate_shape(pair_count: int, height: int, width: int) -> None:
    if not (1 <= pair_count <= 65535 and 1 <= height <= 65535 and 1 <= width <= 65535):
        raise NR1FormatError("pair count and dimensions must fit positive uint16")
