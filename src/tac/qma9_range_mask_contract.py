# SPDX-License-Identifier: MIT
"""Planning-time helpers for PR81/QMA9 semantic range-mask payloads.

The helpers in this module are intentionally score-agnostic. They parse and
validate charged archive bytes so experiment builders can reason about the
semantic range-mask contract without importing a public submission runtime or
touching the contest scorer.
"""
from __future__ import annotations

import hashlib
import math
import struct
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Any


ORIGINAL_VIDEO_BYTES = 37_545_489
RATE_SCORE_PER_BYTE = 25.0 / ORIGINAL_VIDEO_BYTES
QMA9_HEADER_BYTES = 20
QMA9_MAGIC = b"QMA9"
QMA9_VERTICAL_BLOCK_ESCAPE_HEADER_BYTES = 24
QMA9_VERTICAL_BLOCK_ESCAPE_MAGIC = b"QMB1"
QMA9_FIRST_ROW_SPECIALIZATION_HEADER_BYTES = 24
QMA9_FIRST_ROW_SPECIALIZATION_MAGIC = b"QMF1"
QMA9_SENTINEL = 5
QMA9_CLASS_SYMBOLS = 5
QMA9_CONTEXTS = 6**9
QMA9_SCALE_TOTAL = 65_535
QMA9_UPDATE_DELTA = 20
QMA9_FIRST_ROW_SPECIALIZATION_MODES = {
    1: "skip_static_up_gate_full_context",
    2: "skip_static_up_gate_prev_left_context",
    3: "skip_static_up_gate_left_context",
}

_ARITH_TOP = 0xFFFFFFFF
_ARITH_HALF = 0x80000000
_ARITH_FIRST_QTR = 0x40000000
_ARITH_THIRD_QTR = 0xC0000000


class QMA9ContractError(ValueError):
    """Raised when a QMA9 payload or archive violates the static contract."""


@dataclass(frozen=True)
class ZipPayload:
    """Single stored ZIP payload and custody metadata."""

    archive_path: str
    archive_bytes: int
    archive_sha256: str
    member_name: str
    member_bytes: int
    member_sha256: str
    zip_overhead_bytes: int


@dataclass(frozen=True)
class QMA9Header:
    """Parsed QMA9 semantic mask header."""

    magic: str
    frame_count: int
    width: int
    height: int
    bitstream_bytes: int
    header_bytes: int
    packed_bytes: int
    decoded_mask_bytes: int
    bitstream_sha256: str
    payload_sha256: str


@dataclass(frozen=True)
class QMA9DecodedMask:
    """Decoded QMA9 mask bytes in the on-wire raster order."""

    header: QMA9Header
    data: bytes
    sha256: str
    storage_order: str


@dataclass(frozen=True)
class QMA9VerticalBlockEscapeHeader:
    """Parsed planning-only QMA9 vertical block-copy escape header."""

    magic: str
    frame_count: int
    width: int
    height: int
    block_width: int
    bitstream_bytes: int
    header_bytes: int
    packed_bytes: int
    decoded_mask_bytes: int
    bitstream_sha256: str
    payload_sha256: str


@dataclass(frozen=True)
class QMA9FirstRowSpecializationHeader:
    """Parsed planning-only QMF1 first-row specialization header."""

    magic: str
    frame_count: int
    width: int
    height: int
    mode_id: int
    mode_name: str
    bitstream_bytes: int
    header_bytes: int
    packed_bytes: int
    decoded_mask_bytes: int
    bitstream_sha256: str
    payload_sha256: str


@dataclass(frozen=True)
class PayloadSegment:
    """Named slice inside a fixed public-style packed payload."""

    name: str
    offset: int
    size_bytes: int
    sha256: str
    codec: str


@dataclass(frozen=True)
class RateBreakEven:
    """Static rate-only arithmetic against a reference archive."""

    reference_label: str
    reference_bytes: int
    candidate_bytes: int
    delta_bytes: int
    rate_score_delta_if_components_unchanged: float
    component_worsening_budget_before_equal_score: float


@dataclass(frozen=True)
class QMA9Split:
    """Four PR81-style slices from a packed QMA9 payload container."""

    range_mask: bytes
    model: bytes
    pose: bytes
    router: bytes
    segments: tuple[PayloadSegment, ...]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_single_member_zip(path: Path, *, expected_member: str = "p") -> tuple[bytes, ZipPayload]:
    """Read a strict single-member ZIP archive and return payload plus custody.

    The public PR81 archive uses one stored member named ``p``. Builders may use
    this helper with a different expected member, but duplicate or extra members
    always fail closed.
    """

    with zipfile.ZipFile(path, "r") as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
        names = [info.filename for info in infos]
        if names != [expected_member]:
            raise QMA9ContractError(
                f"{path} must contain exactly one file member {expected_member!r}; got {names!r}"
            )
        info = infos[0]
        if info.compress_type != zipfile.ZIP_STORED:
            raise QMA9ContractError(f"{path}:{expected_member} must be ZIP_STORED")
        payload = zf.read(info)
        if len(payload) != int(info.file_size):
            raise QMA9ContractError(f"{path}:{expected_member} size mismatch")

    archive_bytes = int(path.stat().st_size)
    return payload, ZipPayload(
        archive_path=str(path),
        archive_bytes=archive_bytes,
        archive_sha256=sha256_file(path),
        member_name=expected_member,
        member_bytes=len(payload),
        member_sha256=sha256_bytes(payload),
        zip_overhead_bytes=archive_bytes - len(payload),
    )


def parse_qma9_header(payload: bytes) -> QMA9Header:
    """Parse and validate the QMA9 header at the start of ``payload``."""

    if len(payload) < QMA9_HEADER_BYTES:
        raise QMA9ContractError("QMA9 payload is shorter than its 20-byte header")
    magic, frame_count, width, height, bitstream_bytes = struct.unpack_from("<4sIIII", payload, 0)
    if magic != QMA9_MAGIC:
        raise QMA9ContractError(f"expected QMA9 magic, got {magic!r}")
    packed_bytes = QMA9_HEADER_BYTES + int(bitstream_bytes)
    if packed_bytes > len(payload):
        raise QMA9ContractError(
            f"QMA9 bitstream declares {bitstream_bytes} bytes but payload has {len(payload)} bytes"
        )
    if min(int(frame_count), int(width), int(height)) <= 0:
        raise QMA9ContractError("QMA9 dimensions must be positive")
    bitstream = payload[QMA9_HEADER_BYTES:packed_bytes]
    return QMA9Header(
        magic=magic.decode("ascii"),
        frame_count=int(frame_count),
        width=int(width),
        height=int(height),
        bitstream_bytes=int(bitstream_bytes),
        header_bytes=QMA9_HEADER_BYTES,
        packed_bytes=packed_bytes,
        decoded_mask_bytes=int(frame_count) * int(width) * int(height),
        bitstream_sha256=sha256_bytes(bitstream),
        payload_sha256=sha256_bytes(payload[:packed_bytes]),
    )


def parse_qma9_vertical_block_escape_header(payload: bytes) -> QMA9VerticalBlockEscapeHeader:
    """Parse and validate the local-only QMB1 vertical block-copy prototype."""

    if len(payload) < QMA9_VERTICAL_BLOCK_ESCAPE_HEADER_BYTES:
        raise QMA9ContractError("QMB1 payload is shorter than its 24-byte header")
    magic, frame_count, width, height, block_width, bitstream_bytes = struct.unpack_from("<4sIIIII", payload, 0)
    if magic != QMA9_VERTICAL_BLOCK_ESCAPE_MAGIC:
        raise QMA9ContractError(f"expected QMB1 magic, got {magic!r}")
    packed_bytes = QMA9_VERTICAL_BLOCK_ESCAPE_HEADER_BYTES + int(bitstream_bytes)
    if packed_bytes > len(payload):
        raise QMA9ContractError(
            f"QMB1 bitstream declares {bitstream_bytes} bytes but payload has {len(payload)} bytes"
        )
    if min(int(frame_count), int(width), int(height), int(block_width)) <= 0:
        raise QMA9ContractError("QMB1 dimensions and block width must be positive")
    bitstream = payload[QMA9_VERTICAL_BLOCK_ESCAPE_HEADER_BYTES:packed_bytes]
    return QMA9VerticalBlockEscapeHeader(
        magic=magic.decode("ascii"),
        frame_count=int(frame_count),
        width=int(width),
        height=int(height),
        block_width=int(block_width),
        bitstream_bytes=int(bitstream_bytes),
        header_bytes=QMA9_VERTICAL_BLOCK_ESCAPE_HEADER_BYTES,
        packed_bytes=packed_bytes,
        decoded_mask_bytes=int(frame_count) * int(width) * int(height),
        bitstream_sha256=sha256_bytes(bitstream),
        payload_sha256=sha256_bytes(payload[:packed_bytes]),
    )


def parse_qma9_first_row_specialization_header(payload: bytes) -> QMA9FirstRowSpecializationHeader:
    """Parse and validate the local-only QMF1 first-row specialization prototype."""

    if len(payload) < QMA9_FIRST_ROW_SPECIALIZATION_HEADER_BYTES:
        raise QMA9ContractError("QMF1 payload is shorter than its 24-byte header")
    magic, frame_count, width, height, mode_id, bitstream_bytes = struct.unpack_from("<4sIIIII", payload, 0)
    if magic != QMA9_FIRST_ROW_SPECIALIZATION_MAGIC:
        raise QMA9ContractError(f"expected QMF1 magic, got {magic!r}")
    mode_id = int(mode_id)
    mode_name = QMA9_FIRST_ROW_SPECIALIZATION_MODES.get(mode_id)
    if mode_name is None:
        raise QMA9ContractError(f"unknown QMF1 first-row specialization mode: {mode_id}")
    packed_bytes = QMA9_FIRST_ROW_SPECIALIZATION_HEADER_BYTES + int(bitstream_bytes)
    if packed_bytes > len(payload):
        raise QMA9ContractError(
            f"QMF1 bitstream declares {bitstream_bytes} bytes but payload has {len(payload)} bytes"
        )
    if min(int(frame_count), int(width), int(height)) <= 0:
        raise QMA9ContractError("QMF1 dimensions must be positive")
    bitstream = payload[QMA9_FIRST_ROW_SPECIALIZATION_HEADER_BYTES:packed_bytes]
    return QMA9FirstRowSpecializationHeader(
        magic=magic.decode("ascii"),
        frame_count=int(frame_count),
        width=int(width),
        height=int(height),
        mode_id=mode_id,
        mode_name=mode_name,
        bitstream_bytes=int(bitstream_bytes),
        header_bytes=QMA9_FIRST_ROW_SPECIALIZATION_HEADER_BYTES,
        packed_bytes=packed_bytes,
        decoded_mask_bytes=int(frame_count) * int(width) * int(height),
        bitstream_sha256=sha256_bytes(bitstream),
        payload_sha256=sha256_bytes(payload[:packed_bytes]),
    )


class _BitReader:
    def __init__(self, data: bytes):
        self._data = data
        self._pos = 0
        self._left = 0
        self._cur = 0

    def read_bit(self) -> int:
        if self._left == 0:
            self._cur = self._data[self._pos] if self._pos < len(self._data) else 0
            self._pos += 1
            self._left = 8
        bit = (self._cur >> 7) & 1
        self._cur = (self._cur << 1) & 0xFF
        self._left -= 1
        return bit

    @property
    def bits_consumed(self) -> int:
        return self._pos * 8 - self._left

    @property
    def byte_pos(self) -> int:
        return self._pos

    @property
    def bits_left_in_current_byte(self) -> int:
        return self._left


class _BitWriter:
    def __init__(self):
        self._out = bytearray()
        self._cur = 0
        self._count = 0

    def write_bit(self, bit: int) -> None:
        self._cur = ((self._cur << 1) | (1 if bit else 0)) & 0xFF
        self._count += 1
        if self._count == 8:
            self._out.append(self._cur)
            self._cur = 0
            self._count = 0

    def finish(self) -> bytes:
        if self._count:
            self._out.append((self._cur << (8 - self._count)) & 0xFF)
            self._cur = 0
            self._count = 0
        return bytes(self._out)


class _ArithmeticDecoder:
    def __init__(self, data: bytes):
        self.low = 0
        self.high = _ARITH_TOP
        self.value = 0
        self.reader = _BitReader(data)
        for _ in range(32):
            self.value = ((self.value << 1) | self.reader.read_bit()) & _ARITH_TOP

    def snapshot(self) -> dict[str, int]:
        return {
            "low": int(self.low),
            "high": int(self.high),
            "value": int(self.value),
            "bits_consumed": int(self.reader.bits_consumed),
            "byte_pos": int(self.reader.byte_pos),
            "bits_left_in_current_byte": int(self.reader.bits_left_in_current_byte),
        }

    def scaled(self, total: int) -> int:
        span = self.high - self.low + 1
        return (((self.value - self.low + 1) * total) - 1) // span

    def update(self, cum_low: int, cum_high: int, total: int) -> None:
        span = self.high - self.low + 1
        old_low = self.low
        self.high = old_low + (span * cum_high) // total - 1
        self.low = old_low + (span * cum_low) // total
        while True:
            if self.high < _ARITH_HALF:
                pass
            elif self.low >= _ARITH_HALF:
                self.value -= _ARITH_HALF
                self.low -= _ARITH_HALF
                self.high -= _ARITH_HALF
            elif self.low >= _ARITH_FIRST_QTR and self.high < _ARITH_THIRD_QTR:
                self.value -= _ARITH_FIRST_QTR
                self.low -= _ARITH_FIRST_QTR
                self.high -= _ARITH_FIRST_QTR
            else:
                break
            self.low = (self.low << 1) & _ARITH_TOP
            self.high = ((self.high << 1) | 1) & _ARITH_TOP
            self.value = ((self.value << 1) | self.reader.read_bit()) & _ARITH_TOP


class _ArithmeticEncoder:
    def __init__(self):
        self.low = 0
        self.high = _ARITH_TOP
        self.pending = 0
        self.writer = _BitWriter()

    def _emit(self, bit: int) -> None:
        self.writer.write_bit(bit)
        opposite = 0 if bit else 1
        for _ in range(self.pending):
            self.writer.write_bit(opposite)
        self.pending = 0

    def update(self, cum_low: int, cum_high: int, total: int) -> None:
        span = self.high - self.low + 1
        old_low = self.low
        self.high = old_low + (span * cum_high) // total - 1
        self.low = old_low + (span * cum_low) // total
        while True:
            if self.high < _ARITH_HALF:
                self._emit(0)
            elif self.low >= _ARITH_HALF:
                self._emit(1)
                self.low -= _ARITH_HALF
                self.high -= _ARITH_HALF
            elif self.low >= _ARITH_FIRST_QTR and self.high < _ARITH_THIRD_QTR:
                self.pending += 1
                self.low -= _ARITH_FIRST_QTR
                self.high -= _ARITH_FIRST_QTR
            else:
                break
            self.low = (self.low << 1) & _ARITH_TOP
            self.high = ((self.high << 1) | 1) & _ARITH_TOP

    def finish(self) -> bytes:
        self.pending += 1
        if self.low < _ARITH_FIRST_QTR:
            self._emit(0)
        else:
            self._emit(1)
        return self.writer.finish()


def _decode_context_digits(ctx: int) -> tuple[int, int, int, int, int, int, int, int, int]:
    if ctx < 0 or ctx >= QMA9_CONTEXTS:
        raise QMA9ContractError(f"QMA9 context out of range: {ctx}")
    digits: list[int] = []
    value = ctx
    for _ in range(9):
        digits.append(value % 6)
        value //= 6
    left2, up2, prev_down, prev_right, up_right, up_left, up, left, prev = digits
    return prev, left, up, up_left, up_right, prev_right, prev_down, up2, left2


def qma9_context_id(
    prev: int,
    left: int,
    up: int,
    up_left: int,
    up_right: int,
    prev_right: int,
    prev_down: int,
    up2: int,
    left2: int,
) -> int:
    """Return the base-6 adaptive context id used by the QMA9 mask stream."""

    ctx = int(prev)
    for value in (left, up, up_left, up_right, prev_right, prev_down, up2, left2):
        v = int(value)
        if v < 0 or v > QMA9_SENTINEL:
            raise QMA9ContractError(f"QMA9 context symbol out of range: {v}")
        ctx = ctx * 6 + v
    return ctx


@dataclass
class _ModelContext:
    prev_freq: list[int]
    left_freq: list[int]
    up_freq: list[int]
    class_freq: list[int]


class _AdaptiveModel9Binary:
    def __init__(self):
        self._contexts: dict[int, _ModelContext] = {}

    def context(self, ctx: int) -> _ModelContext:
        state = self._contexts.get(ctx)
        if state is not None:
            return state

        prev, left, up, _ul, _ur, _pr, _pd, _up2, _left2 = _decode_context_digits(ctx)
        up_freq = [1, 3]
        left_freq = [1, 4]
        prev_freq = [1, 3]
        class_freq = [1, 1, 1, 1, 1]
        if up == QMA9_SENTINEL:
            up_freq = [60_000, 1]
        if left == QMA9_SENTINEL or left == up:
            left_freq = [60_000, 1]
        if prev == QMA9_SENTINEL or prev == up or prev == left:
            prev_freq = [60_000, 1]
        for cls in range(QMA9_CLASS_SYMBOLS):
            if cls != up and cls != left and cls != prev:
                class_freq[cls] = 3
        state = _ModelContext(
            prev_freq=prev_freq,
            left_freq=left_freq,
            up_freq=up_freq,
            class_freq=class_freq,
        )
        self._contexts[ctx] = state
        return state


def _update_adaptive(freq: list[int], sym: int) -> None:
    if sum(freq) >= QMA9_SCALE_TOTAL:
        for idx, value in enumerate(freq):
            freq[idx] = max(1, (value + 1) >> 1)
    freq[sym] = min(65_535, freq[sym] + QMA9_UPDATE_DELTA)


def _decode_symbol(decoder: _ArithmeticDecoder, freq: list[int]) -> int:
    total = sum(freq)
    value = decoder.scaled(total)
    cumulative = 0
    for sym, weight in enumerate(freq):
        next_cumulative = cumulative + weight
        if value < next_cumulative:
            decoder.update(cumulative, next_cumulative, total)
            return sym
        cumulative = next_cumulative
    raise QMA9ContractError("QMA9 arithmetic decode symbol out of range")


def _decode_symbol_with_cost(decoder: _ArithmeticDecoder, freq: list[int]) -> tuple[int, float, int, int]:
    total = sum(freq)
    value = decoder.scaled(total)
    cumulative = 0
    for sym, weight in enumerate(freq):
        next_cumulative = cumulative + weight
        if value < next_cumulative:
            decoder.update(cumulative, next_cumulative, total)
            return sym, -math.log2(weight / total), total, weight
        cumulative = next_cumulative
    raise QMA9ContractError("QMA9 arithmetic decode symbol out of range")


def _encode_symbol(encoder: _ArithmeticEncoder, freq: list[int], sym: int) -> None:
    if sym < 0 or sym >= len(freq):
        raise QMA9ContractError(f"QMA9 symbol {sym} outside alphabet size {len(freq)}")
    total = sum(freq)
    cumulative = sum(freq[:sym])
    encoder.update(cumulative, cumulative + freq[sym], total)


def _neighbours(data: bytearray | bytes, frame_size: int, t: int, y: int, height: int, width: int, xcoord: int) -> tuple[int, ...]:
    base = t * frame_size + y * width
    prev_base = (t - 1) * frame_size + y * width
    prev = QMA9_SENTINEL if t == 0 else data[prev_base + xcoord]
    left = QMA9_SENTINEL if xcoord == 0 else data[base + xcoord - 1]
    up = QMA9_SENTINEL if y == 0 else data[base - width + xcoord]
    up_left = QMA9_SENTINEL if y == 0 or xcoord == 0 else data[base - width + xcoord - 1]
    up_right = QMA9_SENTINEL if y == 0 or xcoord + 1 >= width else data[base - width + xcoord + 1]
    prev_right = QMA9_SENTINEL if t == 0 or xcoord + 1 >= width else data[prev_base + xcoord + 1]
    prev_down = QMA9_SENTINEL if t == 0 or y + 1 >= height else data[(t - 1) * frame_size + (y + 1) * width + xcoord]
    up2 = QMA9_SENTINEL if y < 2 else data[base - 2 * width + xcoord]
    left2 = QMA9_SENTINEL if xcoord < 2 else data[base + xcoord - 2]
    return prev, left, up, up_left, up_right, prev_right, prev_down, up2, left2


def _validate_raw_qma9_mask(
    raw_mask: bytes | bytearray | memoryview,
    *,
    frame_count: int,
    width: int,
    height: int,
) -> bytes:
    raw = bytes(raw_mask)
    frame_count = int(frame_count)
    width = int(width)
    height = int(height)
    if min(frame_count, width, height) <= 0:
        raise QMA9ContractError("QMA9 dimensions must be positive")
    expected = frame_count * width * height
    if len(raw) != expected:
        raise QMA9ContractError(f"raw mask has {len(raw)} bytes, expected {expected}")
    bad = next((value for value in raw if value >= QMA9_CLASS_SYMBOLS), None)
    if bad is not None:
        raise QMA9ContractError(f"QMA9 raw mask class out of range: {bad}")
    return raw


def _encode_qma9_base_pixel(
    *,
    encoder: _ArithmeticEncoder,
    model: _AdaptiveModel9Binary,
    raw: bytes | bytearray,
    frame_size: int,
    t: int,
    y: int,
    width: int,
    height: int,
    xcoord: int,
) -> None:
    base = t * frame_size + y * height
    prev, left, up, up_left, up_right, prev_right, prev_down, up2, left2 = _neighbours(
        raw, frame_size, t, y, width, height, xcoord
    )
    cls = raw[base + xcoord]
    ctx = model.context(qma9_context_id(prev, left, up, up_left, up_right, prev_right, prev_down, up2, left2))
    if cls == up:
        _encode_symbol(encoder, ctx.up_freq, 1)
        _update_adaptive(ctx.up_freq, 1)
        return
    _encode_symbol(encoder, ctx.up_freq, 0)
    _update_adaptive(ctx.up_freq, 0)
    if cls == left:
        _encode_symbol(encoder, ctx.left_freq, 1)
        _update_adaptive(ctx.left_freq, 1)
        return
    _encode_symbol(encoder, ctx.left_freq, 0)
    _update_adaptive(ctx.left_freq, 0)
    if cls == prev:
        _encode_symbol(encoder, ctx.prev_freq, 1)
        _update_adaptive(ctx.prev_freq, 1)
        return
    _encode_symbol(encoder, ctx.prev_freq, 0)
    _update_adaptive(ctx.prev_freq, 0)
    _encode_symbol(encoder, ctx.class_freq, cls)
    _update_adaptive(ctx.class_freq, cls)


def _decode_qma9_base_pixel(
    *,
    decoder: _ArithmeticDecoder,
    model: _AdaptiveModel9Binary,
    out: bytearray,
    frame_size: int,
    t: int,
    y: int,
    width: int,
    height: int,
    xcoord: int,
) -> int:
    prev, left, up, up_left, up_right, prev_right, prev_down, up2, left2 = _neighbours(
        out, frame_size, t, y, width, height, xcoord
    )
    ctx_id = qma9_context_id(prev, left, up, up_left, up_right, prev_right, prev_down, up2, left2)
    ctx = model.context(ctx_id)
    bit = _decode_symbol(decoder, ctx.up_freq)
    _update_adaptive(ctx.up_freq, bit)
    if bit:
        return up
    bit = _decode_symbol(decoder, ctx.left_freq)
    _update_adaptive(ctx.left_freq, bit)
    if bit:
        return left
    bit = _decode_symbol(decoder, ctx.prev_freq)
    _update_adaptive(ctx.prev_freq, bit)
    if bit:
        return prev
    cls = _decode_symbol(decoder, ctx.class_freq)
    _update_adaptive(ctx.class_freq, cls)
    return cls


def _first_row_context_id(
    *,
    mode_id: int,
    prev: int,
    left: int,
    up: int,
    up_left: int,
    up_right: int,
    prev_right: int,
    prev_down: int,
    up2: int,
    left2: int,
) -> int:
    if int(mode_id) == 1:
        return qma9_context_id(prev, left, up, up_left, up_right, prev_right, prev_down, up2, left2)
    if int(mode_id) == 2:
        return qma9_context_id(
            prev,
            left,
            QMA9_SENTINEL,
            QMA9_SENTINEL,
            QMA9_SENTINEL,
            QMA9_SENTINEL,
            QMA9_SENTINEL,
            QMA9_SENTINEL,
            QMA9_SENTINEL,
        )
    if int(mode_id) == 3:
        return qma9_context_id(
            QMA9_SENTINEL,
            left,
            QMA9_SENTINEL,
            QMA9_SENTINEL,
            QMA9_SENTINEL,
            QMA9_SENTINEL,
            QMA9_SENTINEL,
            QMA9_SENTINEL,
            QMA9_SENTINEL,
        )
    raise QMA9ContractError(f"unknown QMF1 first-row specialization mode: {mode_id}")


def _encode_qma9_first_row_specialized_pixel(
    *,
    encoder: _ArithmeticEncoder,
    model: _AdaptiveModel9Binary,
    raw: bytes | bytearray,
    frame_size: int,
    t: int,
    y: int,
    width: int,
    height: int,
    xcoord: int,
    mode_id: int,
) -> None:
    if y != 0:
        _encode_qma9_base_pixel(
            encoder=encoder,
            model=model,
            raw=raw,
            frame_size=frame_size,
            t=t,
            y=y,
            width=width,
            height=height,
            xcoord=xcoord,
        )
        return

    base = t * frame_size + y * height
    prev, left, up, up_left, up_right, prev_right, prev_down, up2, left2 = _neighbours(
        raw, frame_size, t, y, width, height, xcoord
    )
    cls = raw[base + xcoord]
    ctx = model.context(
        _first_row_context_id(
            mode_id=mode_id,
            prev=prev,
            left=left,
            up=up,
            up_left=up_left,
            up_right=up_right,
            prev_right=prev_right,
            prev_down=prev_down,
            up2=up2,
            left2=left2,
        )
    )
    if cls == left:
        _encode_symbol(encoder, ctx.left_freq, 1)
        _update_adaptive(ctx.left_freq, 1)
        return
    _encode_symbol(encoder, ctx.left_freq, 0)
    _update_adaptive(ctx.left_freq, 0)
    if cls == prev:
        _encode_symbol(encoder, ctx.prev_freq, 1)
        _update_adaptive(ctx.prev_freq, 1)
        return
    _encode_symbol(encoder, ctx.prev_freq, 0)
    _update_adaptive(ctx.prev_freq, 0)
    _encode_symbol(encoder, ctx.class_freq, cls)
    _update_adaptive(ctx.class_freq, cls)


def _decode_qma9_first_row_specialized_pixel(
    *,
    decoder: _ArithmeticDecoder,
    model: _AdaptiveModel9Binary,
    out: bytearray,
    frame_size: int,
    t: int,
    y: int,
    width: int,
    height: int,
    xcoord: int,
    mode_id: int,
) -> int:
    if y != 0:
        return _decode_qma9_base_pixel(
            decoder=decoder,
            model=model,
            out=out,
            frame_size=frame_size,
            t=t,
            y=y,
            width=width,
            height=height,
            xcoord=xcoord,
        )

    prev, left, up, up_left, up_right, prev_right, prev_down, up2, left2 = _neighbours(
        out, frame_size, t, y, width, height, xcoord
    )
    ctx = model.context(
        _first_row_context_id(
            mode_id=mode_id,
            prev=prev,
            left=left,
            up=up,
            up_left=up_left,
            up_right=up_right,
            prev_right=prev_right,
            prev_down=prev_down,
            up2=up2,
            left2=left2,
        )
    )
    bit = _decode_symbol(decoder, ctx.left_freq)
    _update_adaptive(ctx.left_freq, bit)
    if bit:
        return left
    bit = _decode_symbol(decoder, ctx.prev_freq)
    _update_adaptive(ctx.prev_freq, bit)
    if bit:
        return prev
    cls = _decode_symbol(decoder, ctx.class_freq)
    _update_adaptive(ctx.class_freq, cls)
    return cls


def _block_copy_flag_contexts(block_count: int) -> list[list[int]]:
    if block_count <= 0:
        raise QMA9ContractError("block count must be positive")
    return [[1, 1] for _ in range(block_count)]


def decode_qma9_mask(payload: bytes) -> QMA9DecodedMask:
    """Decode a QMA9 semantic mask stream with this repo's pure-Python codec.

    The returned bytes are in QMA9 storage order:
    ``frame_count x header.width x header.height``. PR81's public inflate
    transposes the observed ``600 x 512 x 384`` stream after this decode step
    before feeding its ``384 x 512`` renderer.
    """

    header = parse_qma9_header(payload)
    bitstream = payload[QMA9_HEADER_BYTES:header.packed_bytes]
    decoder = _ArithmeticDecoder(bitstream)
    model = _AdaptiveModel9Binary()
    frame_size = header.width * header.height
    out = bytearray(header.frame_count * frame_size)
    for t in range(header.frame_count):
        for y in range(header.width):
            base = t * frame_size + y * header.height
            for xcoord in range(header.height):
                prev, left, up, up_left, up_right, prev_right, prev_down, up2, left2 = _neighbours(
                    out, frame_size, t, y, header.width, header.height, xcoord
                )
                ctx_id = qma9_context_id(prev, left, up, up_left, up_right, prev_right, prev_down, up2, left2)
                ctx = model.context(ctx_id)
                bit = _decode_symbol(decoder, ctx.up_freq)
                _update_adaptive(ctx.up_freq, bit)
                if bit:
                    cls = up
                else:
                    bit = _decode_symbol(decoder, ctx.left_freq)
                    _update_adaptive(ctx.left_freq, bit)
                    if bit:
                        cls = left
                    else:
                        bit = _decode_symbol(decoder, ctx.prev_freq)
                        _update_adaptive(ctx.prev_freq, bit)
                        if bit:
                            cls = prev
                        else:
                            cls = _decode_symbol(decoder, ctx.class_freq)
                            _update_adaptive(ctx.class_freq, cls)
                if cls < 0 or cls >= QMA9_CLASS_SYMBOLS:
                    raise QMA9ContractError(f"decoded invalid QMA9 class symbol: {cls}")
                out[base + xcoord] = cls
    data = bytes(out)
    return QMA9DecodedMask(
        header=header,
        data=data,
        sha256=sha256_bytes(data),
        storage_order="frame_major_header_width_by_header_height",
    )


def decode_qma9_prefix_frames(payload: bytes, *, frame_count: int) -> bytes:
    """Decode the first ``frame_count`` complete frames from a QMA9 payload.

    This bounded helper is for local byte screens where decoding the full
    117M-pixel PR81 mask would be unnecessary. The arithmetic/model state is
    still advanced exactly over the decoded prefix.
    """

    header = parse_qma9_header(payload)
    prefix_frames = int(frame_count)
    if prefix_frames <= 0:
        raise QMA9ContractError("prefix frame count must be positive")
    if prefix_frames > header.frame_count:
        raise QMA9ContractError(f"prefix frame count {prefix_frames} exceeds payload frames {header.frame_count}")
    bitstream = payload[QMA9_HEADER_BYTES:header.packed_bytes]
    decoder = _ArithmeticDecoder(bitstream)
    model = _AdaptiveModel9Binary()
    frame_size = header.width * header.height
    out = bytearray(prefix_frames * frame_size)
    for t in range(prefix_frames):
        for y in range(header.width):
            base = t * frame_size + y * header.height
            for xcoord in range(header.height):
                cls = _decode_qma9_base_pixel(
                    decoder=decoder,
                    model=model,
                    out=out,
                    frame_size=frame_size,
                    t=t,
                    y=y,
                    width=header.width,
                    height=header.height,
                    xcoord=xcoord,
                )
                if cls < 0 or cls >= QMA9_CLASS_SYMBOLS:
                    raise QMA9ContractError(f"decoded invalid QMA9 class symbol: {cls}")
                out[base + xcoord] = cls
    return bytes(out)


def trace_qma9_prefix(
    payload: bytes,
    *,
    max_pixels: int,
    checkpoint_pixels: Iterable[int] = (),
) -> dict[str, Any]:
    """Decode a QMA9 prefix with pure Python and emit arithmetic/model state.

    This is a forensic helper for validating the Python implementation against
    real contest payload bytes without requiring a full 117M-pixel Python pass.
    ``checkpoint_pixels`` are zero-based decoded pixel indices recorded after
    the pixel has updated the arithmetic decoder and adaptive model.
    """

    header = parse_qma9_header(payload)
    total_pixels = header.decoded_mask_bytes
    max_pixels = int(max_pixels)
    if max_pixels <= 0:
        raise QMA9ContractError("max_pixels must be positive")
    if max_pixels > total_pixels:
        raise QMA9ContractError(f"max_pixels {max_pixels} exceeds decoded mask size {total_pixels}")
    requested_checkpoints = {int(pixel) for pixel in checkpoint_pixels}
    if any(pixel < 0 or pixel >= max_pixels for pixel in requested_checkpoints):
        raise QMA9ContractError("checkpoint pixels must fall inside the decoded prefix")

    bitstream = payload[QMA9_HEADER_BYTES:header.packed_bytes]
    decoder = _ArithmeticDecoder(bitstream)
    model = _AdaptiveModel9Binary()
    frame_size = header.width * header.height
    out = bytearray(max_pixels)
    prefix_model = bytearray(max_pixels)
    stage_counts = Counter()
    stage_bits = Counter()
    predictor_counts = Counter()
    class_counts = Counter()
    context_counts = Counter()
    checkpoints: list[dict[str, Any]] = []
    decoded = 0

    for t in range(header.frame_count):
        if decoded >= max_pixels:
            break
        for y in range(header.width):
            if decoded >= max_pixels:
                break
            base = t * frame_size + y * header.height
            for xcoord in range(header.height):
                if decoded >= max_pixels:
                    break
                prev, left, up, up_left, up_right, prev_right, prev_down, up2, left2 = _neighbours(
                    prefix_model, frame_size, t, y, header.width, header.height, xcoord
                )
                ctx_id = qma9_context_id(prev, left, up, up_left, up_right, prev_right, prev_down, up2, left2)
                context_counts[ctx_id] += 1
                ctx = model.context(ctx_id)
                bit, bits, _total, _weight = _decode_symbol_with_cost(decoder, ctx.up_freq)
                stage_counts["up_gate"] += 1
                stage_bits["up_gate"] += bits
                _update_adaptive(ctx.up_freq, bit)
                if bit:
                    cls = up
                    predictor = "up"
                else:
                    bit, bits, _total, _weight = _decode_symbol_with_cost(decoder, ctx.left_freq)
                    stage_counts["left_gate"] += 1
                    stage_bits["left_gate"] += bits
                    _update_adaptive(ctx.left_freq, bit)
                    if bit:
                        cls = left
                        predictor = "left"
                    else:
                        bit, bits, _total, _weight = _decode_symbol_with_cost(decoder, ctx.prev_freq)
                        stage_counts["prev_gate"] += 1
                        stage_bits["prev_gate"] += bits
                        _update_adaptive(ctx.prev_freq, bit)
                        if bit:
                            cls = prev
                            predictor = "prev"
                        else:
                            cls, bits, _total, _weight = _decode_symbol_with_cost(decoder, ctx.class_freq)
                            stage_counts["class_fallback"] += 1
                            stage_bits["class_fallback"] += bits
                            _update_adaptive(ctx.class_freq, cls)
                            predictor = "class_fallback"
                if cls < 0 or cls >= QMA9_CLASS_SYMBOLS:
                    raise QMA9ContractError(f"decoded invalid QMA9 class symbol: {cls}")
                out[decoded] = cls
                prefix_model[base + xcoord] = cls
                predictor_counts[predictor] += 1
                class_counts[int(cls)] += 1
                if decoded in requested_checkpoints:
                    checkpoints.append(
                        {
                            "pixel_index": decoded,
                            "frame": t,
                            "row": y,
                            "col": xcoord,
                            "class": int(cls),
                            "predictor": predictor,
                            "context_id": int(ctx_id),
                            "decoder": decoder.snapshot(),
                        }
                    )
                decoded += 1

    stage_bits_dict = {key: float(value) for key, value in sorted(stage_bits.items())}
    estimated_bits = float(sum(stage_bits.values()))
    top_contexts = [
        {"context_id": int(ctx), "pixels": int(count)}
        for ctx, count in context_counts.most_common(32)
    ]
    result: dict[str, Any] = {
        "schema": "qma9_pure_python_prefix_trace_v1",
        "implementation": "src/tac/qma9_range_mask_contract.py::trace_qma9_prefix",
        "payload_sha256": header.payload_sha256,
        "bitstream_sha256": header.bitstream_sha256,
        "header": {
            "frame_count": header.frame_count,
            "width": header.width,
            "height": header.height,
            "decoded_mask_bytes": header.decoded_mask_bytes,
            "bitstream_bytes": header.bitstream_bytes,
        },
        "decoded_prefix_pixels": decoded,
        "decoded_prefix_sha256": sha256_bytes(bytes(out)),
        "estimated_model_bits": estimated_bits,
        "estimated_model_bytes": estimated_bits / 8.0,
        "stage_counts": {key: int(value) for key, value in sorted(stage_counts.items())},
        "stage_estimated_bits": stage_bits_dict,
        "predictor_counts": {key: int(value) for key, value in sorted(predictor_counts.items())},
        "class_counts": {str(key): int(value) for key, value in sorted(class_counts.items())},
        "top_contexts": top_contexts,
        "checkpoints": checkpoints,
        "decoder_state_after_prefix": decoder.snapshot(),
        "full_payload_roundtrip_verified": False,
        "full_payload_roundtrip_reason": "prefix trace only; use full decode and encode for whole-payload bitstream equality",
    }
    if decoded % frame_size == 0:
        prefix_frames = decoded // frame_size
        encoded_prefix = encode_qma9_mask(out, frame_count=prefix_frames, width=header.width, height=header.height)
        decoded_prefix = decode_qma9_mask(encoded_prefix)
        result["prefix_self_roundtrip"] = {
            "frame_count": prefix_frames,
            "encoded_payload_bytes": len(encoded_prefix),
            "encoded_payload_sha256": sha256_bytes(encoded_prefix),
            "decoded_sha256": decoded_prefix.sha256,
            "matches_prefix": decoded_prefix.data == bytes(out),
        }
    return result


def encode_qma9_mask(raw_mask: bytes | bytearray | memoryview, *, frame_count: int, width: int, height: int) -> bytes:
    """Encode class ids ``0..4`` into a deterministic QMA9 semantic mask stream."""

    frame_count = int(frame_count)
    width = int(width)
    height = int(height)
    raw = _validate_raw_qma9_mask(raw_mask, frame_count=frame_count, width=width, height=height)

    encoder = _ArithmeticEncoder()
    model = _AdaptiveModel9Binary()
    frame_size = width * height
    for t in range(frame_count):
        for y in range(width):
            for xcoord in range(height):
                _encode_qma9_base_pixel(
                    encoder=encoder,
                    model=model,
                    raw=raw,
                    frame_size=frame_size,
                    t=t,
                    y=y,
                    width=width,
                    height=height,
                    xcoord=xcoord,
                )

    bitstream = encoder.finish()
    return struct.pack("<4sIIII", QMA9_MAGIC, frame_count, width, height, len(bitstream)) + bitstream


def analyze_qma9_vertical_block_copy_opportunities(
    raw_mask: bytes | bytearray | memoryview,
    *,
    frame_count: int,
    width: int,
    height: int,
    block_width: int = 16,
) -> dict[str, int | float]:
    """Count exact row-above block copies for the local QMB1 byte screen."""

    frame_count = int(frame_count)
    width = int(width)
    height = int(height)
    block_width = int(block_width)
    raw = _validate_raw_qma9_mask(raw_mask, frame_count=frame_count, width=width, height=height)
    if block_width <= 0:
        raise QMA9ContractError("block width must be positive")
    frame_size = width * height
    block_count_per_row = (height + block_width - 1) // block_width
    eligible_blocks = 0
    copied_blocks = 0
    copied_pixels = 0
    for t in range(frame_count):
        for y in range(1, width):
            base = t * frame_size + y * height
            prev_row = base - height
            for block_index in range(block_count_per_row):
                x0 = block_index * block_width
                x1 = min(height, x0 + block_width)
                eligible_blocks += 1
                if raw[base + x0:base + x1] == raw[prev_row + x0:prev_row + x1]:
                    copied_blocks += 1
                    copied_pixels += x1 - x0
    total_pixels = frame_count * frame_size
    return {
        "block_width": block_width,
        "block_count_per_row": block_count_per_row,
        "eligible_blocks": eligible_blocks,
        "copied_blocks": copied_blocks,
        "copied_pixels": copied_pixels,
        "total_pixels": total_pixels,
        "copied_block_fraction": copied_blocks / max(1, eligible_blocks),
        "copied_pixel_fraction": copied_pixels / max(1, total_pixels),
    }


def encode_qma9_vertical_block_escape_mask(
    raw_mask: bytes | bytearray | memoryview,
    *,
    frame_count: int,
    width: int,
    height: int,
    block_width: int = 16,
) -> bytes:
    """Encode a planning-only QMB1 vertical block-copy escape variant.

    QMB1 is not a contest runtime format. It is a local prototype for measuring
    whether row-above block-copy flags can amortize QMA9's dominant up-gate
    cost before any runtime or scorer work is considered.
    """

    frame_count = int(frame_count)
    width = int(width)
    height = int(height)
    block_width = int(block_width)
    raw = _validate_raw_qma9_mask(raw_mask, frame_count=frame_count, width=width, height=height)
    if block_width <= 0:
        raise QMA9ContractError("block width must be positive")

    encoder = _ArithmeticEncoder()
    model = _AdaptiveModel9Binary()
    frame_size = width * height
    block_count_per_row = (height + block_width - 1) // block_width
    flag_contexts = _block_copy_flag_contexts(block_count_per_row)
    for t in range(frame_count):
        for y in range(width):
            base = t * frame_size + y * height
            prev_row = base - height
            for block_index in range(block_count_per_row):
                x0 = block_index * block_width
                x1 = min(height, x0 + block_width)
                copy_block = y > 0 and raw[base + x0:base + x1] == raw[prev_row + x0:prev_row + x1]
                if y > 0:
                    freq = flag_contexts[block_index]
                    flag = 1 if copy_block else 0
                    _encode_symbol(encoder, freq, flag)
                    _update_adaptive(freq, flag)
                if copy_block:
                    continue
                for xcoord in range(x0, x1):
                    _encode_qma9_base_pixel(
                        encoder=encoder,
                        model=model,
                        raw=raw,
                        frame_size=frame_size,
                        t=t,
                        y=y,
                        width=width,
                        height=height,
                        xcoord=xcoord,
                    )

    bitstream = encoder.finish()
    return struct.pack(
        "<4sIIIII",
        QMA9_VERTICAL_BLOCK_ESCAPE_MAGIC,
        frame_count,
        width,
        height,
        block_width,
        len(bitstream),
    ) + bitstream


def encode_qma9_first_row_specialization_mask(
    raw_mask: bytes | bytearray | memoryview,
    *,
    frame_count: int,
    width: int,
    height: int,
    mode_id: int,
) -> bytes:
    """Encode the local-only QMF1 first-row specialization variant.

    QMF1 preserves QMA9's base decoder for all non-first rows. On first rows,
    it removes the deterministic ``up == sentinel`` gate and optionally folds
    cold context dimensions according to ``mode_id``. It is a byte-screen
    prototype only; a contest archive would need a reviewed runtime consumer.
    """

    frame_count = int(frame_count)
    width = int(width)
    height = int(height)
    mode_id = int(mode_id)
    if mode_id not in QMA9_FIRST_ROW_SPECIALIZATION_MODES:
        raise QMA9ContractError(f"unknown QMF1 first-row specialization mode: {mode_id}")
    raw = _validate_raw_qma9_mask(raw_mask, frame_count=frame_count, width=width, height=height)

    encoder = _ArithmeticEncoder()
    model = _AdaptiveModel9Binary()
    frame_size = width * height
    for t in range(frame_count):
        for y in range(width):
            for xcoord in range(height):
                _encode_qma9_first_row_specialized_pixel(
                    encoder=encoder,
                    model=model,
                    raw=raw,
                    frame_size=frame_size,
                    t=t,
                    y=y,
                    width=width,
                    height=height,
                    xcoord=xcoord,
                    mode_id=mode_id,
                )

    bitstream = encoder.finish()
    return struct.pack(
        "<4sIIIII",
        QMA9_FIRST_ROW_SPECIALIZATION_MAGIC,
        frame_count,
        width,
        height,
        mode_id,
        len(bitstream),
    ) + bitstream


def decode_qma9_first_row_specialization_mask(payload: bytes) -> QMA9DecodedMask:
    """Decode the local-only QMF1 first-row specialization variant."""

    header = parse_qma9_first_row_specialization_header(payload)
    bitstream = payload[QMA9_FIRST_ROW_SPECIALIZATION_HEADER_BYTES:header.packed_bytes]
    decoder = _ArithmeticDecoder(bitstream)
    model = _AdaptiveModel9Binary()
    frame_size = header.width * header.height
    out = bytearray(header.frame_count * frame_size)
    for t in range(header.frame_count):
        for y in range(header.width):
            base = t * frame_size + y * header.height
            for xcoord in range(header.height):
                cls = _decode_qma9_first_row_specialized_pixel(
                    decoder=decoder,
                    model=model,
                    out=out,
                    frame_size=frame_size,
                    t=t,
                    y=y,
                    width=header.width,
                    height=header.height,
                    xcoord=xcoord,
                    mode_id=header.mode_id,
                )
                if cls < 0 or cls >= QMA9_CLASS_SYMBOLS:
                    raise QMA9ContractError(f"decoded invalid QMF1 class symbol: {cls}")
                out[base + xcoord] = cls

    data = bytes(out)
    qma9_header = QMA9Header(
        magic=header.magic,
        frame_count=header.frame_count,
        width=header.width,
        height=header.height,
        bitstream_bytes=header.bitstream_bytes,
        header_bytes=header.header_bytes,
        packed_bytes=header.packed_bytes,
        decoded_mask_bytes=header.decoded_mask_bytes,
        bitstream_sha256=header.bitstream_sha256,
        payload_sha256=header.payload_sha256,
    )
    return QMA9DecodedMask(
        header=qma9_header,
        data=data,
        sha256=sha256_bytes(data),
        storage_order="frame_major_header_width_by_header_height",
    )


def decode_qma9_vertical_block_escape_mask(payload: bytes) -> QMA9DecodedMask:
    """Decode the local-only QMB1 vertical block-copy escape variant."""

    header = parse_qma9_vertical_block_escape_header(payload)
    bitstream = payload[QMA9_VERTICAL_BLOCK_ESCAPE_HEADER_BYTES:header.packed_bytes]
    decoder = _ArithmeticDecoder(bitstream)
    model = _AdaptiveModel9Binary()
    frame_size = header.width * header.height
    block_count_per_row = (header.height + header.block_width - 1) // header.block_width
    flag_contexts = _block_copy_flag_contexts(block_count_per_row)
    out = bytearray(header.frame_count * frame_size)
    for t in range(header.frame_count):
        for y in range(header.width):
            base = t * frame_size + y * header.height
            prev_row = base - header.height
            for block_index in range(block_count_per_row):
                x0 = block_index * header.block_width
                x1 = min(header.height, x0 + header.block_width)
                copy_block = False
                if y > 0:
                    freq = flag_contexts[block_index]
                    flag = _decode_symbol(decoder, freq)
                    _update_adaptive(freq, flag)
                    copy_block = bool(flag)
                if copy_block:
                    out[base + x0:base + x1] = out[prev_row + x0:prev_row + x1]
                    continue
                for xcoord in range(x0, x1):
                    cls = _decode_qma9_base_pixel(
                        decoder=decoder,
                        model=model,
                        out=out,
                        frame_size=frame_size,
                        t=t,
                        y=y,
                        width=header.width,
                        height=header.height,
                        xcoord=xcoord,
                    )
                    if cls < 0 or cls >= QMA9_CLASS_SYMBOLS:
                        raise QMA9ContractError(f"decoded invalid QMA9 class symbol: {cls}")
                    out[base + xcoord] = cls

    data = bytes(out)
    qma9_header = QMA9Header(
        magic=header.magic,
        frame_count=header.frame_count,
        width=header.width,
        height=header.height,
        bitstream_bytes=header.bitstream_bytes,
        header_bytes=header.header_bytes,
        packed_bytes=header.packed_bytes,
        decoded_mask_bytes=header.decoded_mask_bytes,
        bitstream_sha256=header.bitstream_sha256,
        payload_sha256=header.payload_sha256,
    )
    return QMA9DecodedMask(
        header=qma9_header,
        data=data,
        sha256=sha256_bytes(data),
        storage_order="frame_major_header_width_by_header_height",
    )


def unpack_router_actions(payload: bytes, *, count: int = 600, bits_per_action: int = 3) -> tuple[int, ...]:
    """Unpack PR81 little-endian fixed-width router action ids."""

    if count < 0 or bits_per_action <= 0:
        raise QMA9ContractError("router action count and bit width must be positive")
    mask = (1 << bits_per_action) - 1
    values: list[int] = []
    acc = 0
    bits = 0
    for byte in payload:
        acc |= int(byte) << bits
        bits += 8
        while bits >= bits_per_action and len(values) < count:
            values.append(acc & mask)
            acc >>= bits_per_action
            bits -= bits_per_action
    if len(values) != count:
        raise QMA9ContractError(f"decoded {len(values)} router actions, expected {count}")
    return tuple(values)


def pack_router_actions(actions: Iterable[int], *, bits_per_action: int = 3) -> bytes:
    """Pack PR81 router action ids with deterministic little-endian bit order."""

    if bits_per_action <= 0:
        raise QMA9ContractError("router action bit width must be positive")
    max_value = (1 << bits_per_action) - 1
    out = bytearray()
    acc = 0
    bits = 0
    for action in actions:
        value = int(action)
        if value < 0 or value > max_value:
            raise QMA9ContractError(f"router action {value} outside {bits_per_action}-bit range")
        acc |= value << bits
        bits += bits_per_action
        while bits >= 8:
            out.append(acc & 0xFF)
            acc >>= 8
            bits -= 8
    if bits:
        out.append(acc & 0xFF)
    return bytes(out)


def split_qma9_pr81_payload(
    payload: bytes,
    *,
    range_mask_bytes: int,
    model_bytes: int,
    pose_bytes: int,
    router_bytes: int,
) -> QMA9Split:
    """Split a PR81-style fixed container into semantic segments."""

    specs = [
        ("range_mask.qma9", int(range_mask_bytes), "qma9_adaptive9_binary_range_mask"),
        ("split_model_reordered.br_bundle", int(model_bytes), "brotli_reordered_qzs3_model_bundle"),
        ("optimized_poses.qp1.br", int(pose_bytes), "brotli_qp1_pose_stream"),
        ("router_actions.3bit", int(router_bytes), "packed_3bit_pair_router_actions"),
    ]
    segments = slice_payload_segments(payload, specs)
    return QMA9Split(
        range_mask=payload[segments[0].offset:segments[0].offset + segments[0].size_bytes],
        model=payload[segments[1].offset:segments[1].offset + segments[1].size_bytes],
        pose=payload[segments[2].offset:segments[2].offset + segments[2].size_bytes],
        router=payload[segments[3].offset:segments[3].offset + segments[3].size_bytes],
        segments=segments,
    )


def write_stored_single_member_zip(path: Path, payload: bytes, *, member_name: str = "p") -> None:
    """Write a deterministic single-member stored ZIP archive."""

    if not member_name or member_name.startswith("/") or "/" in member_name or member_name.startswith("."):
        raise QMA9ContractError(f"unsafe single-member ZIP name: {member_name!r}")
    info = zipfile.ZipInfo(member_name)
    info.date_time = (1980, 1, 1, 0, 0, 0)
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o644 << 16
    info.create_system = 3
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(info, payload)


def slice_payload_segments(payload: bytes, specs: list[tuple[str, int, str]]) -> tuple[PayloadSegment, ...]:
    """Slice ``payload`` into named contiguous segments.

    ``specs`` contains ``(name, size_bytes, codec)`` tuples. The sizes must sum
    exactly to the payload length; this prevents silent trailing bytes or stale
    constant tables.
    """

    offset = 0
    segments: list[PayloadSegment] = []
    for name, size_bytes, codec in specs:
        size = int(size_bytes)
        if size < 0:
            raise QMA9ContractError(f"negative segment size for {name}: {size}")
        end = offset + size
        if end > len(payload):
            raise QMA9ContractError(f"segment {name} overruns payload")
        data = payload[offset:end]
        segments.append(
            PayloadSegment(
                name=str(name),
                offset=offset,
                size_bytes=size,
                sha256=sha256_bytes(data),
                codec=str(codec),
            )
        )
        offset = end
    if offset != len(payload):
        raise QMA9ContractError(f"segment specs consumed {offset} bytes but payload has {len(payload)}")
    return tuple(segments)


def rate_break_even(
    *,
    candidate_bytes: int,
    reference_bytes: int,
    reference_label: str,
) -> RateBreakEven:
    """Return static contest-rate arithmetic for a byte delta.

    This is never a score claim: it assumes the SegNet/PoseNet terms are
    unchanged and only computes the component worsening budget before the byte
    savings stop helping.
    """

    delta_bytes = int(candidate_bytes) - int(reference_bytes)
    rate_delta = delta_bytes * RATE_SCORE_PER_BYTE
    return RateBreakEven(
        reference_label=str(reference_label),
        reference_bytes=int(reference_bytes),
        candidate_bytes=int(candidate_bytes),
        delta_bytes=delta_bytes,
        rate_score_delta_if_components_unchanged=rate_delta,
        component_worsening_budget_before_equal_score=max(0.0, -rate_delta),
    )
