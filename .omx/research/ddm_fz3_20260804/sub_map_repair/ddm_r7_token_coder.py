# SPDX-License-Identifier: MIT
"""Deterministic lossless coders for the DDM TR1 int4 token lattice.

The counted object is the quantized ``[pair, row, column, channel]`` token
array.  Every codec first applies the exact temporal-mode factorization

``code = (mode_base + residual) mod levels``.

The decoder derives every probability context from the mode base and its
already-decoded prefix.  No fitted histogram, scorer tensor, checkpoint value,
or sensitivity table is hidden in this free decoder code.

This module deliberately lives under ``experiments`` pending MAIN's landing
review: ``PROGRAM.md`` does not currently allow this arm to edit ``src/tac``.
It is nevertheless a strict production-format prototype: compact framing,
integer arithmetic decode, exact SHA closure, canonical re-encode, and no
SciPy/MLX/Torch dependency.
"""

from __future__ import annotations

import hashlib
import heapq
import lzma
import math
import struct
from dataclasses import dataclass
from typing import Final

import numpy as np

try:
    import brotli
except ImportError:  # pragma: no cover - optional race arm
    brotli = None  # type: ignore[assignment]

from repair_entropy_coder_runtime_adapters import (
    ans_rans_prototype_decode,
    ans_rans_prototype_encode,
)


class DDMR7CoderError(ValueError):
    """The token frame or arithmetic stream failed closed."""


MAGIC: Final = b"DR7T"
VERSION: Final = 1
DEFAULT_CODEC: Final = "smevr"
AUTO_CODECS: Final = ("smevr", "brotli11")
CODEC_IDS: Final = {
    "kt_prev1": 1,
    "cae_inspired_identity_inter": 2,
    "smevr": 3,
    "rans_o0": 4,
    "rans_o0_on_adjacent_innovation": 5,
    "lzma1": 6,
    "brotli11": 7,
    "kt_o8_prev5_backoff": 8,
    "huffman_nibble": 9,
}
ID_CODECS: Final = {value: key for key, value in CODEC_IDS.items()}
HEADER: Final = struct.Struct("<4sBBBB4HII32s")
_SMEVR_LENGTH: Final = struct.Struct("<I")
_HUFFMAN_HEADER: Final = struct.Struct("<Q16B")
_MAX_VALUES: Final = 16_000_000
_RESCALE_AT: Final = 1 << 15

_STATE_BITS: Final = 32
_FULL_RANGE: Final = 1 << _STATE_BITS
_HALF: Final = _FULL_RANGE >> 1
_QUARTER: Final = _HALF >> 1
_THREE_QUARTERS: Final = 3 * _QUARTER


@dataclass(frozen=True, slots=True)
class TokenFrameAccounting:
    codec: str
    framed_bytes: int
    header_bytes: int
    base_bytes: int
    delta_bytes: int
    raw_token_bytes: int
    sha256: str


class _BitWriter:
    def __init__(self) -> None:
        self.output = bytearray()
        self.current = 0
        self.used = 0

    def write(self, bit: int) -> None:
        self.current = (self.current << 1) | int(bit)
        self.used += 1
        if self.used == 8:
            self.output.append(self.current)
            self.current = 0
            self.used = 0

    def finish(self) -> bytes:
        if self.used:
            self.output.append(self.current << (8 - self.used))
        return bytes(self.output)


class _BitReader:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.byte_index = 0
        self.bit_index = 0

    def read(self) -> int:
        # Arithmetic termination legitimately reads zero padding.  The outer
        # digest and canonical re-encode distinguish truncation and trailers.
        if self.byte_index >= len(self.payload):
            return 0
        value = (self.payload[self.byte_index] >> (7 - self.bit_index)) & 1
        self.bit_index += 1
        if self.bit_index == 8:
            self.bit_index = 0
            self.byte_index += 1
        return value


class _ArithmeticEncoder:
    def __init__(self) -> None:
        self.writer = _BitWriter()
        self.low = 0
        self.high = _FULL_RANGE - 1
        self.pending = 0

    def _emit(self, bit: int) -> None:
        self.writer.write(bit)
        while self.pending:
            self.writer.write(1 - bit)
            self.pending -= 1

    def encode(self, symbol: int, counts: list[int]) -> None:
        total = sum(counts)
        lower = sum(counts[:symbol])
        upper = lower + counts[symbol]
        width = self.high - self.low + 1
        self.high = self.low + (width * upper // total) - 1
        self.low += width * lower // total
        while True:
            if self.high < _HALF:
                self._emit(0)
            elif self.low >= _HALF:
                self._emit(1)
                self.low -= _HALF
                self.high -= _HALF
            elif self.low >= _QUARTER and self.high < _THREE_QUARTERS:
                self.pending += 1
                self.low -= _QUARTER
                self.high -= _QUARTER
            else:
                break
            self.low <<= 1
            self.high = (self.high << 1) | 1

    def finish(self) -> bytes:
        self.pending += 1
        self._emit(0 if self.low < _QUARTER else 1)
        return self.writer.finish()


class _ArithmeticDecoder:
    def __init__(self, payload: bytes) -> None:
        if not payload:
            raise DDMR7CoderError("arithmetic stream is empty")
        self.reader = _BitReader(payload)
        self.low = 0
        self.high = _FULL_RANGE - 1
        self.code = 0
        for _ in range(_STATE_BITS):
            self.code = (self.code << 1) | self.reader.read()

    def decode(self, counts: list[int]) -> int:
        total = sum(counts)
        width = self.high - self.low + 1
        target = ((self.code - self.low + 1) * total - 1) // width
        lower = 0
        symbol = -1
        for index, count in enumerate(counts):
            if target < lower + count:
                symbol = index
                break
            lower += count
        if symbol < 0:
            raise DDMR7CoderError("arithmetic target escaped model")
        upper = lower + counts[symbol]
        self.high = self.low + (width * upper // total) - 1
        self.low += width * lower // total
        while True:
            if self.high < _HALF:
                pass
            elif self.low >= _HALF:
                self.low -= _HALF
                self.high -= _HALF
                self.code -= _HALF
            elif self.low >= _QUARTER and self.high < _THREE_QUARTERS:
                self.low -= _QUARTER
                self.high -= _QUARTER
                self.code -= _QUARTER
            else:
                break
            self.low <<= 1
            self.high = (self.high << 1) | 1
            self.code = (self.code << 1) | self.reader.read()
        return symbol


def _row(table: dict[int, list[int]], context: int, alphabet: int) -> list[int]:
    counts = table.get(context)
    if counts is None:
        counts = [1] * alphabet
        table[context] = counts
    return counts


def _update(counts: list[int], symbol: int) -> None:
    # Integer alpha=1/2 initialization with deterministic bounded-count
    # rescaling.  This is a rescaled adaptive-KT formulation, not exact
    # unbounded KT after the first rescale.
    counts[symbol] += 2
    if sum(counts) >= _RESCALE_AT:
        counts[:] = [max(1, (value + 1) // 2) for value in counts]


def _codes(value: np.ndarray, levels: int) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.uint8 or array.ndim != 4:
        raise DDMR7CoderError("token codes must be uint8 [P,H,W,C]")
    if not (2 <= levels <= 16):
        raise DDMR7CoderError("levels must be in [2,16]")
    if array.size == 0 or array.size > _MAX_VALUES:
        raise DDMR7CoderError("token element count is outside the production bound")
    if any(int(dimension) <= 0 or int(dimension) > 65535 for dimension in array.shape):
        raise DDMR7CoderError("token dimensions are outside uint16 framing")
    if np.any(array >= levels):
        raise DDMR7CoderError("token code exceeds levels")
    return np.ascontiguousarray(array)


def factor_mode_delta(codes: np.ndarray, levels: int) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic temporal mode and modulo residual."""

    value = _codes(codes, levels)
    flat = value.reshape(value.shape[0], -1)
    base = np.empty(flat.shape[1], dtype=np.uint8)
    for index in range(flat.shape[1]):
        # argmax gives the canonical lowest-symbol tie break.
        base[index] = np.bincount(flat[:, index], minlength=levels).argmax()
    base = base.reshape(value.shape[1:])
    delta = (value.astype(np.int16) - base[None].astype(np.int16)) % levels
    return np.ascontiguousarray(base), np.ascontiguousarray(delta.astype(np.uint8))


def reconstruct_mode_delta(base: np.ndarray, delta: np.ndarray, levels: int) -> np.ndarray:
    value = (np.asarray(base, dtype=np.int16)[None] + np.asarray(delta, dtype=np.int16)) % levels
    return np.ascontiguousarray(value.astype(np.uint8))


def _semantic_digest(
    codec_id: int,
    levels: int,
    shape: tuple[int, int, int, int],
    raw: bytes,
) -> bytes:
    metadata = MAGIC + struct.pack("<BBBB4H", VERSION, codec_id, levels, 4, *shape)
    return hashlib.sha256(metadata + raw).digest()


def pack_nibbles(value: np.ndarray) -> bytes:
    flat = np.asarray(value, dtype=np.uint8).reshape(-1)
    if np.any(flat >= 16):
        raise DDMR7CoderError("nibble stream values must be below 16")
    output = np.empty((flat.size + 1) // 2, dtype=np.uint8)
    output[:] = flat[0::2] << 4
    if flat.size // 2:
        output[: flat.size // 2] |= flat[1::2]
    return output.tobytes()


def unpack_nibbles(payload: bytes, count: int) -> np.ndarray:
    if count < 0 or len(payload) != (count + 1) // 2:
        raise DDMR7CoderError("nibble payload length differs")
    packed = np.frombuffer(payload, dtype=np.uint8)
    if count % 2 and packed.size and packed[-1] & 15:
        raise DDMR7CoderError("nibble payload has nonzero canonical padding")
    output = np.empty(count, dtype=np.uint8)
    output[0::2] = packed >> 4
    output[1::2] = (packed & 15)[: count // 2]
    return output


def _raw_lzma_encode(payload: bytes) -> bytes:
    return lzma.compress(
        payload,
        format=lzma.FORMAT_RAW,
        filters=[{"id": lzma.FILTER_LZMA1, "preset": 9 | lzma.PRESET_EXTREME}],
    )


def _raw_lzma_decode(payload: bytes, *, expected_length: int) -> bytes:
    try:
        decoder = lzma.LZMADecompressor(
            format=lzma.FORMAT_RAW,
            filters=[{"id": lzma.FILTER_LZMA1, "preset": 9 | lzma.PRESET_EXTREME}],
        )
        restored = decoder.decompress(payload, max_length=expected_length + 1)
    except lzma.LZMAError as exc:
        raise DDMR7CoderError("invalid LZMA1 stream") from exc
    if len(restored) != expected_length or not decoder.eof or decoder.unused_data:
        raise DDMR7CoderError("LZMA1 stream length or termination differs")
    return restored


def _channel_views(value: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(value.transpose(3, 0, 1, 2)).reshape(value.shape[3], -1)


def _encode_kt_prev1(delta: np.ndarray, levels: int) -> bytes:
    pair_count, height, width, channels = delta.shape
    frame_values = height * width
    views = _channel_views(delta)
    rows: dict[int, list[int]] = {}
    encoder = _ArithmeticEncoder()
    for channel in range(channels):
        source = views[channel]
        for index, raw_symbol in enumerate(source):
            previous = int(source[index - frame_values]) if index >= frame_values else levels
            counts = _row(rows, channel * (levels + 1) + previous, levels)
            symbol = int(raw_symbol)
            encoder.encode(symbol, counts)
            _update(counts, symbol)
    return encoder.finish()


def _decode_kt_prev1(
    payload: bytes,
    shape: tuple[int, int, int, int],
    levels: int,
) -> np.ndarray:
    pair_count, height, width, channels = shape
    del pair_count
    frame_values = height * width
    output = np.zeros((channels, math.prod(shape[:-1])), dtype=np.uint8)
    rows: dict[int, list[int]] = {}
    decoder = _ArithmeticDecoder(payload)
    for channel in range(channels):
        target = output[channel]
        for index in range(target.size):
            previous = int(target[index - frame_values]) if index >= frame_values else levels
            counts = _row(rows, channel * (levels + 1) + previous, levels)
            symbol = decoder.decode(counts)
            target[index] = symbol
            _update(counts, symbol)
    return np.ascontiguousarray(output.reshape(channels, *shape[:-1]).transpose(1, 2, 3, 0))


def _o8_prev5_context(
    values: np.ndarray,
    index: int,
    *,
    height: int,
    width: int,
    levels: int,
) -> int:
    """Pack pp1's eight causal spatial plus five previous-frame values."""

    frame_values = height * width
    frame = index // frame_values
    cell = index % frame_values
    row, column = divmod(cell, width)
    sentinel = levels

    def current(dy: int, dx: int) -> int:
        target_row = row + dy
        target_column = column + dx
        if target_row < 0 or target_row >= height or target_column < 0 or target_column >= width:
            return sentinel
        target = index + dy * width + dx
        # All pp1 spatial offsets are raster-causal; keep the guard explicit.
        return int(values[target]) if target < index else sentinel

    def previous(dy: int, dx: int) -> int:
        target_row = row + dy
        target_column = column + dx
        if frame == 0 or target_row < 0 or target_row >= height or target_column < 0 or target_column >= width:
            return sentinel
        return int(values[index - frame_values + dy * width + dx])

    context = 0
    for value in (
        current(0, -1),
        current(-1, 0),
        current(-1, -1),
        current(-1, 1),
        current(0, -2),
        current(-2, 0),
        current(-1, -2),
        current(-1, 2),
        previous(0, 0),
        previous(0, 1),
        previous(0, -1),
        previous(1, 0),
        previous(-1, 0),
    ):
        context = context * (levels + 1) + value
    return context


def _encode_kt_o8_prev5_backoff(delta: np.ndarray, levels: int) -> bytes:
    """Actual pp1 O8+prev5 KT with deterministic singleton backoff.

    A full 13-neighbour row is activated only after its second occurrence.
    Before that, both endpoints deterministically use the already-real
    previous-frame co-located KT row.  This prevents singleton tables from
    pretending to be useful probability models while retaining the requested
    maximum context when the dump supplies evidence for it.
    """

    _pairs, height, width, channels = delta.shape
    frame_values = height * width
    views = _channel_views(delta)
    fallback: dict[int, list[int]] = {}
    first_symbol: dict[int, int] = {}
    full_rows: dict[int, list[int]] = {}
    encoder = _ArithmeticEncoder()
    context_channel_stride = (levels + 1) ** 13
    for channel in range(channels):
        source = views[channel]
        for index, raw_symbol in enumerate(source):
            previous = int(source[index - frame_values]) if index >= frame_values else levels
            fallback_row = _row(fallback, channel * (levels + 1) + previous, levels)
            context = channel * context_channel_stride + _o8_prev5_context(
                source,
                index,
                height=height,
                width=width,
                levels=levels,
            )
            counts = full_rows.get(context, fallback_row)
            symbol = int(raw_symbol)
            encoder.encode(symbol, counts)
            _update(fallback_row, symbol)
            if context in full_rows:
                _update(full_rows[context], symbol)
            elif context in first_symbol:
                row_counts = [1] * levels
                _update(row_counts, first_symbol.pop(context))
                _update(row_counts, symbol)
                full_rows[context] = row_counts
            else:
                first_symbol[context] = symbol
    return encoder.finish()


def _decode_kt_o8_prev5_backoff(
    payload: bytes,
    shape: tuple[int, int, int, int],
    levels: int,
) -> np.ndarray:
    _pairs, height, width, channels = shape
    frame_values = height * width
    output = np.zeros((channels, math.prod(shape[:-1])), dtype=np.uint8)
    fallback: dict[int, list[int]] = {}
    first_symbol: dict[int, int] = {}
    full_rows: dict[int, list[int]] = {}
    decoder = _ArithmeticDecoder(payload)
    context_channel_stride = (levels + 1) ** 13
    for channel in range(channels):
        target = output[channel]
        for index in range(target.size):
            previous = int(target[index - frame_values]) if index >= frame_values else levels
            fallback_row = _row(fallback, channel * (levels + 1) + previous, levels)
            context = channel * context_channel_stride + _o8_prev5_context(
                target,
                index,
                height=height,
                width=width,
                levels=levels,
            )
            counts = full_rows.get(context, fallback_row)
            symbol = decoder.decode(counts)
            target[index] = symbol
            _update(fallback_row, symbol)
            if context in full_rows:
                _update(full_rows[context], symbol)
            elif context in first_symbol:
                row_counts = [1] * levels
                _update(row_counts, first_symbol.pop(context))
                _update(row_counts, symbol)
                full_rows[context] = row_counts
            else:
                first_symbol[context] = symbol
    return np.ascontiguousarray(output.reshape(channels, *shape[:-1]).transpose(1, 2, 3, 0))


def _cae_context(
    partial: np.ndarray,
    index: int,
    *,
    plane: int,
    height: int,
    width: int,
    levels: int,
) -> int:
    frame_values = height * width
    cell = index % frame_values
    row, column = divmod(cell, width)
    previous = (int(partial[index - frame_values]) >> plane) & 1 if index >= frame_values else 0
    left = (int(partial[index - 1]) >> plane) & 1 if column else 0
    upper = (int(partial[index - width]) >> plane) & 1 if row else 0
    upper_left = (int(partial[index - width - 1]) >> plane) & 1 if row and column else 0
    higher = int(partial[index]) >> (plane + 1)
    # ``levels`` participates to keep the context grammar explicit even though
    # the current vehicle uses L=16.
    return ((((plane * levels + higher) * 2 + previous) * 2 + left) * 2 + upper) * 2 + upper_left


def _encode_cae_inspired_identity_inter(delta: np.ndarray, levels: int) -> bytes:
    _pairs, height, width, channels = delta.shape
    bits = (levels - 1).bit_length()
    source_views = _channel_views(delta)
    partial = np.zeros_like(source_views)
    rows: dict[int, list[int]] = {}
    encoder = _ArithmeticEncoder()
    context_stride = bits * levels * 16
    for plane in range(bits - 1, -1, -1):
        for channel in range(channels):
            source = source_views[channel]
            target = partial[channel]
            for index, raw_symbol in enumerate(source):
                context = channel * context_stride + _cae_context(
                    target,
                    index,
                    plane=plane,
                    height=height,
                    width=width,
                    levels=levels,
                )
                counts = _row(rows, context, 2)
                bit = (int(raw_symbol) >> plane) & 1
                encoder.encode(bit, counts)
                _update(counts, bit)
                target[index] |= bit << plane
    return encoder.finish()


def _decode_cae_inspired_identity_inter(
    payload: bytes,
    shape: tuple[int, int, int, int],
    levels: int,
) -> np.ndarray:
    _pairs, height, width, channels = shape
    bits = (levels - 1).bit_length()
    partial = np.zeros((channels, math.prod(shape[:-1])), dtype=np.uint8)
    rows: dict[int, list[int]] = {}
    decoder = _ArithmeticDecoder(payload)
    context_stride = bits * levels * 16
    for plane in range(bits - 1, -1, -1):
        for channel in range(channels):
            target = partial[channel]
            for index in range(target.size):
                context = channel * context_stride + _cae_context(
                    target,
                    index,
                    plane=plane,
                    height=height,
                    width=width,
                    levels=levels,
                )
                counts = _row(rows, context, 2)
                bit = decoder.decode(counts)
                _update(counts, bit)
                target[index] |= bit << plane
    if np.any(partial >= levels):
        raise DDMR7CoderError("CAE decoded symbol exceeds levels")
    return np.ascontiguousarray(partial.reshape(channels, *shape[:-1]).transpose(1, 2, 3, 0))


def _residual_to_rank(value: int, levels: int) -> int:
    # 0 is handled by the occupancy stream.  Rank nearby circular residuals
    # as +1,-1,+2,-2,...; the mapping is generic and decoder-invertible.
    signed = value if value <= levels // 2 else value - levels
    return 2 * signed - 2 if signed > 0 else -2 * signed - 1


def _rank_to_residual(rank: int, levels: int) -> int:
    signed = rank // 2 + 1 if rank % 2 == 0 else -(rank // 2 + 1)
    return signed % levels


def _encode_smevr(base: np.ndarray, delta: np.ndarray, levels: int) -> bytes:
    """Static-Mode Event/Value Renewal (SMEVR), the R7 original arm."""

    _pairs, height, width, channels = delta.shape
    frame_values = height * width
    sources = _channel_views(delta)
    bases = np.ascontiguousarray(base.transpose(2, 0, 1)).reshape(channels, -1)
    occupancy_encoder = _ArithmeticEncoder()
    value_encoder = _ArithmeticEncoder()
    occupancy_rows: dict[int, list[int]] = {}
    value_rows: dict[int, list[int]] = {}
    for channel in range(channels):
        source = sources[channel]
        base_channel = bases[channel]
        ages = np.zeros(frame_values, dtype=np.uint8)
        for index, raw_value in enumerate(source):
            cell = index % frame_values
            row, column = divmod(cell, width)
            previous_value = int(source[index - frame_values]) if index >= frame_values else 0
            previous_occupancy = int(previous_value != 0)
            left_occupancy = int(source[index - 1] != 0) if column else 0
            upper_occupancy = int(source[index - width] != 0) if row else 0
            age_bucket = min(int(ages[cell]), 3)
            context = channel
            for value, radix in (
                (int(base_channel[cell]), levels),
                (previous_occupancy, 2),
                (left_occupancy, 2),
                (upper_occupancy, 2),
                (age_bucket, 4),
            ):
                context = context * radix + value
            occupancy = int(raw_value != 0)
            counts = _row(occupancy_rows, context, 2)
            occupancy_encoder.encode(occupancy, counts)
            _update(counts, occupancy)
            if occupancy:
                value_context = (channel * levels + int(base_channel[cell])) * (levels + 1) + previous_value
                value_counts = _row(value_rows, value_context, levels - 1)
                rank = _residual_to_rank(int(raw_value), levels)
                value_encoder.encode(rank, value_counts)
                _update(value_counts, rank)
            if occupancy == previous_occupancy:
                ages[cell] = min(int(ages[cell]) + 1, 255)
            else:
                ages[cell] = 0
    occupancy_stream = occupancy_encoder.finish()
    value_stream = value_encoder.finish()
    return _SMEVR_LENGTH.pack(len(occupancy_stream)) + occupancy_stream + value_stream


def _decode_smevr(
    payload: bytes,
    base: np.ndarray,
    shape: tuple[int, int, int, int],
    levels: int,
) -> np.ndarray:
    if len(payload) <= _SMEVR_LENGTH.size:
        raise DDMR7CoderError("SMEVR stream is truncated")
    (occupancy_length,) = _SMEVR_LENGTH.unpack_from(payload)
    if occupancy_length <= 0 or _SMEVR_LENGTH.size + occupancy_length >= len(payload):
        raise DDMR7CoderError("SMEVR stream lengths differ")
    occupancy_decoder = _ArithmeticDecoder(payload[_SMEVR_LENGTH.size : _SMEVR_LENGTH.size + occupancy_length])
    value_decoder = _ArithmeticDecoder(payload[_SMEVR_LENGTH.size + occupancy_length :])
    _pairs, height, width, channels = shape
    frame_values = height * width
    output = np.zeros((channels, math.prod(shape[:-1])), dtype=np.uint8)
    bases = np.ascontiguousarray(base.transpose(2, 0, 1)).reshape(channels, -1)
    occupancy_rows: dict[int, list[int]] = {}
    value_rows: dict[int, list[int]] = {}
    for channel in range(channels):
        target = output[channel]
        base_channel = bases[channel]
        ages = np.zeros(frame_values, dtype=np.uint8)
        for index in range(target.size):
            cell = index % frame_values
            row, column = divmod(cell, width)
            previous_value = int(target[index - frame_values]) if index >= frame_values else 0
            previous_occupancy = int(previous_value != 0)
            left_occupancy = int(target[index - 1] != 0) if column else 0
            upper_occupancy = int(target[index - width] != 0) if row else 0
            age_bucket = min(int(ages[cell]), 3)
            context = channel
            for value, radix in (
                (int(base_channel[cell]), levels),
                (previous_occupancy, 2),
                (left_occupancy, 2),
                (upper_occupancy, 2),
                (age_bucket, 4),
            ):
                context = context * radix + value
            counts = _row(occupancy_rows, context, 2)
            occupancy = occupancy_decoder.decode(counts)
            _update(counts, occupancy)
            if occupancy:
                value_context = (channel * levels + int(base_channel[cell])) * (levels + 1) + previous_value
                value_counts = _row(value_rows, value_context, levels - 1)
                rank = value_decoder.decode(value_counts)
                _update(value_counts, rank)
                target[index] = _rank_to_residual(rank, levels)
            if occupancy == previous_occupancy:
                ages[cell] = min(int(ages[cell]) + 1, 255)
            else:
                ages[cell] = 0
    return np.ascontiguousarray(output.reshape(channels, *shape[:-1]).transpose(1, 2, 3, 0))


def _adjacent_innovation_nibbles(delta: np.ndarray, levels: int) -> bytes:
    flat = delta.reshape(-1)
    transformed = np.empty_like(flat)
    transformed[0] = flat[0]
    transformed[1:] = (flat[1:].astype(np.int16) - flat[:-1].astype(np.int16)) % levels
    return pack_nibbles(transformed)


def _inverse_adjacent_innovation_nibbles(
    payload: bytes,
    count: int,
    levels: int,
) -> np.ndarray:
    transformed = unpack_nibbles(payload, count)
    output = np.empty_like(transformed)
    output[0] = transformed[0]
    # The loop is deliberately integer and portable; count is bounded above.
    for index in range(1, count):
        output[index] = (int(output[index - 1]) + int(transformed[index])) % levels
    return output


def _huffman_lengths(values: np.ndarray, levels: int) -> list[int]:
    counts = np.bincount(np.asarray(values, dtype=np.uint8).reshape(-1), minlength=levels)
    heap: list[tuple[int, int, int, int | tuple[object, object]]] = []
    sequence = 0
    for symbol, count in enumerate(counts.tolist()):
        if count:
            heapq.heappush(heap, (int(count), symbol, sequence, symbol))
            sequence += 1
    if not heap:
        raise DDMR7CoderError("Huffman stream is empty")
    if len(heap) == 1:
        lengths = [0] * 16
        lengths[int(heap[0][3])] = 1
        return lengths
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        node = (left[3], right[3])
        heapq.heappush(
            heap,
            (left[0] + right[0], min(left[1], right[1]), sequence, node),
        )
        sequence += 1
    lengths = [0] * 16

    def visit(node: int | tuple[object, object], depth: int) -> None:
        if isinstance(node, int):
            lengths[node] = depth
            return
        visit(node[0], depth + 1)  # type: ignore[arg-type]
        visit(node[1], depth + 1)  # type: ignore[arg-type]

    visit(heap[0][3], 0)
    if max(lengths) > 15:
        raise DDMR7CoderError("Huffman length exceeds compact table")
    return lengths


def _canonical_huffman_codes(lengths: list[int], levels: int) -> dict[int, tuple[int, int]]:
    rows = sorted((lengths[symbol], symbol) for symbol in range(levels) if lengths[symbol])
    if not rows:
        raise DDMR7CoderError("Huffman table has no symbols")
    code = 0
    previous_length = rows[0][0]
    result: dict[int, tuple[int, int]] = {}
    for length, symbol in rows:
        if length < previous_length or length <= 0:
            raise DDMR7CoderError("Huffman lengths are invalid")
        code <<= length - previous_length
        if code >= (1 << length):
            raise DDMR7CoderError("Huffman table is oversubscribed")
        result[symbol] = (code, length)
        code += 1
        previous_length = length
    return result


def _encode_huffman_nibbles(delta: np.ndarray, levels: int) -> bytes:
    flat = np.asarray(delta, dtype=np.uint8).reshape(-1)
    lengths = _huffman_lengths(flat, levels)
    codes = _canonical_huffman_codes(lengths, levels)
    writer = _BitWriter()
    bit_count = 0
    for raw_symbol in flat:
        code, length = codes[int(raw_symbol)]
        for shift in range(length - 1, -1, -1):
            writer.write((code >> shift) & 1)
        bit_count += length
    return _HUFFMAN_HEADER.pack(bit_count, *lengths) + writer.finish()


def _decode_huffman_nibbles(
    payload: bytes,
    shape: tuple[int, int, int, int],
    levels: int,
) -> np.ndarray:
    if len(payload) < _HUFFMAN_HEADER.size:
        raise DDMR7CoderError("Huffman stream is truncated")
    unpacked = _HUFFMAN_HEADER.unpack_from(payload)
    bit_count = int(unpacked[0])
    lengths = [int(value) for value in unpacked[1:]]
    if any(lengths[symbol] for symbol in range(levels, 16)):
        raise DDMR7CoderError("Huffman table codes symbols outside levels")
    codes = _canonical_huffman_codes(lengths, levels)
    decode_map = {(length, code): symbol for symbol, (code, length) in codes.items()}
    bit_payload = payload[_HUFFMAN_HEADER.size :]
    if len(bit_payload) != (bit_count + 7) // 8:
        raise DDMR7CoderError("Huffman bit length differs")
    if bit_count % 8 and bit_payload and bit_payload[-1] & ((1 << (8 - bit_count % 8)) - 1):
        raise DDMR7CoderError("Huffman padding bits are nonzero")
    reader = _BitReader(bit_payload)
    count = math.prod(shape)
    output = np.empty(count, dtype=np.uint8)
    consumed = 0
    maximum = max(lengths)
    for index in range(count):
        code = 0
        symbol = None
        for length in range(1, maximum + 1):
            if consumed >= bit_count:
                raise DDMR7CoderError("Huffman stream ended inside a symbol")
            code = (code << 1) | reader.read()
            consumed += 1
            symbol = decode_map.get((length, code))
            if symbol is not None:
                output[index] = symbol
                break
        if symbol is None:
            raise DDMR7CoderError("Huffman code is absent from table")
    if consumed != bit_count:
        raise DDMR7CoderError("Huffman stream has trailing coded bits")
    return np.ascontiguousarray(output.reshape(shape))


def _encode_delta(codec: str, base: np.ndarray, delta: np.ndarray, levels: int) -> bytes:
    if codec == "kt_prev1":
        return _encode_kt_prev1(delta, levels)
    if codec == "kt_o8_prev5_backoff":
        return _encode_kt_o8_prev5_backoff(delta, levels)
    if codec == "cae_inspired_identity_inter":
        return _encode_cae_inspired_identity_inter(delta, levels)
    if codec == "smevr":
        return _encode_smevr(base, delta, levels)
    if codec == "rans_o0":
        return ans_rans_prototype_encode(pack_nibbles(delta))
    if codec == "rans_o0_on_adjacent_innovation":
        return ans_rans_prototype_encode(_adjacent_innovation_nibbles(delta, levels))
    if codec == "lzma1":
        return _raw_lzma_encode(pack_nibbles(delta))
    if codec == "brotli11":
        if brotli is None:
            raise DDMR7CoderError("Brotli dependency is unavailable")
        return bytes(brotli.compress(pack_nibbles(delta), quality=11))
    if codec == "huffman_nibble":
        return _encode_huffman_nibbles(delta, levels)
    raise DDMR7CoderError(f"unsupported token codec: {codec}")


def _decode_delta(
    codec: str,
    payload: bytes,
    base: np.ndarray,
    shape: tuple[int, int, int, int],
    levels: int,
) -> np.ndarray:
    count = math.prod(shape)
    if codec == "kt_prev1":
        return _decode_kt_prev1(payload, shape, levels)
    if codec == "kt_o8_prev5_backoff":
        return _decode_kt_o8_prev5_backoff(payload, shape, levels)
    if codec == "cae_inspired_identity_inter":
        return _decode_cae_inspired_identity_inter(payload, shape, levels)
    if codec == "smevr":
        return _decode_smevr(payload, base, shape, levels)
    if codec in {"rans_o0", "rans_o0_on_adjacent_innovation"}:
        fixed_header = 8 + struct.calcsize("<BQH")
        if len(payload) < fixed_header:
            raise DDMR7CoderError("rANS frame header is truncated")
        original_length = struct.unpack_from("<Q", payload, 8 + 1)[0]
        if original_length != (count + 1) // 2:
            raise DDMR7CoderError("rANS inner output length differs")
        try:
            packed = ans_rans_prototype_decode(payload)
        except ValueError as exc:
            raise DDMR7CoderError("invalid rANS frame") from exc
        flat = (
            unpack_nibbles(packed, count)
            if codec == "rans_o0"
            else _inverse_adjacent_innovation_nibbles(
                packed,
                count,
                levels,
            )
        )
        return np.ascontiguousarray(flat.reshape(shape))
    if codec == "lzma1":
        packed = _raw_lzma_decode(payload, expected_length=(count + 1) // 2)
        return np.ascontiguousarray(unpack_nibbles(packed, count).reshape(shape))
    if codec == "brotli11":
        if brotli is None:
            raise DDMR7CoderError("Brotli dependency is unavailable")
        try:
            packed = bytes(brotli.decompress(payload))
        except brotli.error as exc:
            raise DDMR7CoderError("invalid Brotli stream") from exc
        return np.ascontiguousarray(unpack_nibbles(packed, count).reshape(shape))
    if codec == "huffman_nibble":
        return _decode_huffman_nibbles(payload, shape, levels)
    raise DDMR7CoderError(f"unsupported token codec: {codec}")


def encode_token_codes(
    codes: np.ndarray,
    *,
    levels: int = 16,
    codec: str = DEFAULT_CODEC,
) -> bytes:
    """Encode one strict, self-describing, lossless R7 token frame."""

    value = _codes(codes, levels)
    if codec == "auto":
        available = [candidate for candidate in AUTO_CODECS if candidate != "brotli11" or brotli is not None]
        candidates = [encode_token_codes(value, levels=levels, codec=candidate) for candidate in available]
        return min(
            candidates,
            key=lambda frame: (len(frame), ID_CODECS[HEADER.unpack_from(frame)[2]]),
        )
    if codec not in CODEC_IDS:
        raise DDMR7CoderError(f"unsupported token codec: {codec}")
    base, delta = factor_mode_delta(value, levels)
    base_stream = _raw_lzma_encode(pack_nibbles(base))
    delta_stream = _encode_delta(codec, base, delta, levels)
    header = HEADER.pack(
        MAGIC,
        VERSION,
        CODEC_IDS[codec],
        levels,
        value.ndim,
        *value.shape,
        len(base_stream),
        len(delta_stream),
        _semantic_digest(CODEC_IDS[codec], levels, tuple(value.shape), value.tobytes()),
    )
    return header + base_stream + delta_stream


def _decode_token_codes(frame: bytes, *, canonical: bool) -> np.ndarray:
    if len(frame) < HEADER.size:
        raise DDMR7CoderError("token frame is truncated")
    (
        magic,
        version,
        codec_id,
        levels,
        rank,
        pair_count,
        height,
        width,
        channels,
        base_length,
        delta_length,
        digest,
    ) = HEADER.unpack_from(frame)
    if magic != MAGIC or version != VERSION or rank != 4:
        raise DDMR7CoderError("token frame magic/version/rank differs")
    codec = ID_CODECS.get(codec_id)
    if codec is None or not (2 <= levels <= 16):
        raise DDMR7CoderError("token frame codec/levels differs")
    shape = (pair_count, height, width, channels)
    count = math.prod(shape)
    if count <= 0 or count > _MAX_VALUES:
        raise DDMR7CoderError("token frame shape is outside bounds")
    expected = HEADER.size + base_length + delta_length
    if base_length <= 0 or delta_length <= 0 or len(frame) != expected:
        raise DDMR7CoderError("token frame lengths do not close")
    base_payload = frame[HEADER.size : HEADER.size + base_length]
    delta_payload = frame[HEADER.size + base_length :]
    base_count = height * width * channels
    base = unpack_nibbles(
        _raw_lzma_decode(base_payload, expected_length=(base_count + 1) // 2),
        base_count,
    ).reshape(
        height,
        width,
        channels,
    )
    if np.any(base >= levels):
        raise DDMR7CoderError("mode base exceeds levels")
    delta = _decode_delta(codec, delta_payload, base, shape, levels)
    if delta.shape != shape or np.any(delta >= levels):
        raise DDMR7CoderError("decoded residual differs from declared lattice")
    restored = reconstruct_mode_delta(base, delta, levels)
    expected_digest = _semantic_digest(codec_id, levels, shape, restored.tobytes())
    if expected_digest != digest:
        raise DDMR7CoderError("decoded token semantic SHA-256 differs")
    if canonical and encode_token_codes(restored, levels=levels, codec=codec) != frame:
        raise DDMR7CoderError("token frame is noncanonical or has inert bytes")
    return restored


def decode_token_codes(frame: bytes) -> np.ndarray:
    """Decode and canonically re-encode one R7 token frame."""

    try:
        return _decode_token_codes(bytes(frame), canonical=True)
    except (OverflowError, struct.error) as exc:
        raise DDMR7CoderError("token frame structure is invalid") from exc


def frame_accounting(frame: bytes) -> TokenFrameAccounting:
    if len(frame) < HEADER.size:
        raise DDMR7CoderError("token frame is truncated")
    unpacked = HEADER.unpack_from(frame)
    codec = ID_CODECS.get(unpacked[2])
    if codec is None:
        raise DDMR7CoderError("token frame codec differs")
    shape = tuple(int(value) for value in unpacked[5:9])
    return TokenFrameAccounting(
        codec=codec,
        framed_bytes=len(frame),
        header_bytes=HEADER.size,
        base_bytes=int(unpacked[9]),
        delta_bytes=int(unpacked[10]),
        raw_token_bytes=math.prod(shape),
        sha256=hashlib.sha256(frame).hexdigest(),
    )


__all__ = [
    "AUTO_CODECS",
    "CODEC_IDS",
    "DEFAULT_CODEC",
    "DDMR7CoderError",
    "TokenFrameAccounting",
    "decode_token_codes",
    "encode_token_codes",
    "factor_mode_delta",
    "frame_accounting",
    "pack_nibbles",
    "reconstruct_mode_delta",
    "unpack_nibbles",
]
