#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Deterministic Mahoney-lite logistic context mixer for arbitrary bytes.

This is a real lossless coder, not an entropy estimate.  Each bit is encoded by
an integer arithmetic coder using a probability from four causal,
decoder-derived experts:

1. global bit frequency;
2. bit-position frequency;
3. previous-byte + current-byte-prefix frequency; and
4. current run-length frequency.

The expert probabilities are converted to log-odds, combined linearly, and
mapped back through a sigmoid before arithmetic coding.  After the coded bit is
known, the weights receive the logistic-loss gradient update

    w_i <- w_i + eta * (y - p_mix) * logit(p_i).

There is no probability-domain averaging hidden behind the "logistic" name.

Portability contract
--------------------

The complete encode/decode path is stdlib-only and uses no binary floating
point.  Log and sigmoid are deterministic fixed-point approximations:

* probabilities, weights, and stored logits use Q12;
* ``ln(x)`` uses Q20 guard precision, integer range reduction, and six terms of
  ``2 * atanh((x - 1) / (x + 1))``;
* sigmoid inversion is a lookup/bisect over all positive 12-bit frequencies;
* the online learning rate is exactly 1/512.

Python's specified integer, ``struct``, and ``hashlib`` behavior therefore
defines the cross-host wire.  A decoder verifies the raw SHA-256 and then
re-encodes the decoded bytes, refusing noncanonical truncation, trailers,
alternative padding, or model evolution.

This module reads no scorer, checkpoint, or campaign state.
"""

from __future__ import annotations

import hashlib
import struct
from bisect import bisect_left
from itertools import pairwise
from typing import Final

MAGIC: Final = b"D7LM"
VERSION: Final = 1
EXPERT_COUNT: Final = 4
PROBABILITY_BITS: Final = 12
PROBABILITY_TOTAL: Final = 1 << PROBABILITY_BITS
LOGIT_BITS: Final = 12
LOGIT_SCALE: Final = 1 << LOGIT_BITS
WEIGHT_BITS: Final = 12
WEIGHT_SCALE: Final = 1 << WEIGHT_BITS
LEARNING_RATE_DENOMINATOR: Final = 512
COUNT_RESCALE_AT: Final = 1024
MAX_RAW_BYTES: Final = 64 * 1024 * 1024

_LN_GUARD_BITS: Final = 20
_LN_GUARD_SCALE: Final = 1 << _LN_GUARD_BITS
_LN2_Q20: Final = 726817
_STATE_BITS: Final = 32
_FULL_RANGE: Final = 1 << _STATE_BITS
_HALF: Final = _FULL_RANGE >> 1
_QUARTER: Final = _HALF >> 1
_THREE_QUARTERS: Final = _QUARTER * 3
_WEIGHT_LIMIT: Final = 4 * WEIGHT_SCALE
_BIAS_LIMIT: Final = 8 * LOGIT_SCALE
_HEADER: Final = struct.Struct("<4sBBBBQQ32s")


class LogisticMixError(ValueError):
    """The input, arithmetic stream, or canonical frame is invalid."""


def _as_bytes(value: bytes | bytearray | memoryview) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, (bytearray, memoryview)):
        return bytes(value)
    raise TypeError("logistic mixer input must be bytes-like")


def _round_div_signed(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    if numerator >= 0:
        return (numerator + denominator // 2) // denominator
    return -((-numerator + denominator // 2) // denominator)


def _ln_q20(value: int) -> int:
    """Return deterministic Q20 ln(value) for one positive integer."""

    if value <= 0:
        raise ValueError("integer logarithm input must be positive")
    exponent = value.bit_length() - 1
    normalized = (value << _LN_GUARD_BITS) >> exponent
    # normalized is Q20 in [1, 2).  z is in [0, 1/3), so six odd
    # atanh-series terms leave much less than one output-Q12 unit of tail error.
    z = ((normalized - _LN_GUARD_SCALE) << _LN_GUARD_BITS) // (normalized + _LN_GUARD_SCALE)
    z_squared = _round_div_signed(z * z, _LN_GUARD_SCALE)
    term = z
    series = term
    for denominator in (3, 5, 7, 9, 11):
        term = _round_div_signed(term * z_squared, _LN_GUARD_SCALE)
        series += _round_div_signed(term, denominator)
    return exponent * _LN2_Q20 + 2 * series


def _ln_q12(value: int) -> int:
    """Return deterministic Q12 ln(value), rounded from a Q20 calculation."""

    return _round_div_signed(_ln_q20(value), 1 << (_LN_GUARD_BITS - LOGIT_BITS))


def _logit_from_frequency(probability_one: int) -> int:
    if not 1 <= probability_one < PROBABILITY_TOTAL:
        raise ValueError("binary frequency must be strictly inside the total")
    return _round_div_signed(
        _ln_q20(probability_one) - _ln_q20(PROBABILITY_TOTAL - probability_one),
        1 << (_LN_GUARD_BITS - LOGIT_BITS),
    )


_LOGIT_TABLE: Final = tuple(_logit_from_frequency(probability_one) for probability_one in range(1, PROBABILITY_TOTAL))
if any(left >= right for left, right in pairwise(_LOGIT_TABLE)):
    raise RuntimeError("fixed-point logit table is not strictly increasing")


def _frequency_from_logit(logit: int) -> int:
    """Invert the fixed-point logit by deterministic nearest lookup."""

    index = bisect_left(_LOGIT_TABLE, logit)
    if index == 0:
        return 1
    if index == len(_LOGIT_TABLE):
        return PROBABILITY_TOTAL - 1
    lower = _LOGIT_TABLE[index - 1]
    upper = _LOGIT_TABLE[index]
    # Table index i corresponds to frequency i+1.  Resolve an exact midpoint
    # toward the lower frequency as part of the wire contract.
    return index if logit - lower <= upper - logit else index + 1


def _probability_from_counts(counts: list[int] | tuple[int, int]) -> int:
    zero, one = counts
    total = zero + one
    probability_one = (one * PROBABILITY_TOTAL + total // 2) // total
    return min(PROBABILITY_TOTAL - 1, max(1, probability_one))


def _update_counts(counts: list[int], bit: int) -> None:
    counts[bit] += 1
    if counts[0] + counts[1] >= COUNT_RESCALE_AT:
        counts[0] = max(1, (counts[0] + 1) // 2)
        counts[1] = max(1, (counts[1] + 1) // 2)


def _mix_logit(
    expert_logits: tuple[int, int, int, int],
    weights: list[int] | tuple[int, int, int, int],
    bias: int,
) -> int:
    """Return the true linear combination in fixed-point logit space."""

    combined = bias + _round_div_signed(
        sum(weight * expert_logit for weight, expert_logit in zip(weights, expert_logits, strict=True)),
        WEIGHT_SCALE,
    )
    return min(_LOGIT_TABLE[-1], max(_LOGIT_TABLE[0], combined))


class _Mixer:
    """Four causal experts plus an online fixed-point logistic combiner."""

    __slots__ = (
        "bias",
        "global_counts",
        "last_bit",
        "order_counts",
        "position_counts",
        "previous_byte",
        "run_counts",
        "run_length",
        "weights",
    )

    def __init__(self) -> None:
        self.global_counts = [1, 1]
        self.position_counts = [[1, 1] for _ in range(8)]
        self.order_counts: dict[int, list[int]] = {}
        self.run_counts: dict[int, list[int]] = {}
        self.weights = [WEIGHT_SCALE // EXPERT_COUNT] * EXPERT_COUNT
        self.bias = 0
        self.previous_byte = 256
        self.last_bit = 0
        self.run_length = 0

    def predict(self, bit_position: int, byte_prefix: int) -> tuple[int, tuple[int, int, int, int], int, int]:
        order_key = self.previous_byte * 2048 + bit_position * 256 + byte_prefix
        run_key = (self.last_bit * 8 + min(self.run_length, 7)) * 8 + bit_position
        rows = (
            self.global_counts,
            self.position_counts[bit_position],
            self.order_counts.get(order_key, (1, 1)),
            self.run_counts.get(run_key, (1, 1)),
        )
        expert_logits = tuple(_LOGIT_TABLE[_probability_from_counts(row) - 1] for row in rows)
        mixed_logit = _mix_logit(expert_logits, self.weights, self.bias)
        return (
            _frequency_from_logit(mixed_logit),
            expert_logits,
            order_key,
            run_key,
        )

    def update(
        self,
        *,
        bit: int,
        probability_one: int,
        expert_logits: tuple[int, int, int, int],
        bit_position: int,
        order_key: int,
        run_key: int,
    ) -> None:
        error = (PROBABILITY_TOTAL if bit else 0) - probability_one
        gradient_denominator = LOGIT_SCALE * LEARNING_RATE_DENOMINATOR
        for index, expert_logit in enumerate(expert_logits):
            delta = _round_div_signed(error * expert_logit, gradient_denominator)
            self.weights[index] = min(
                _WEIGHT_LIMIT,
                max(-_WEIGHT_LIMIT, self.weights[index] + delta),
            )
        self.bias = min(
            _BIAS_LIMIT,
            max(
                -_BIAS_LIMIT,
                self.bias + _round_div_signed(error, LEARNING_RATE_DENOMINATOR),
            ),
        )

        order = self.order_counts.get(order_key)
        if order is None:
            order = [1, 1]
            self.order_counts[order_key] = order
        run = self.run_counts.get(run_key)
        if run is None:
            run = [1, 1]
            self.run_counts[run_key] = run
        for counts in (
            self.global_counts,
            self.position_counts[bit_position],
            order,
            run,
        ):
            _update_counts(counts, bit)

        if bit == self.last_bit:
            self.run_length += 1
        else:
            self.last_bit = bit
            self.run_length = 1

    def finish_byte(self, byte: int) -> None:
        self.previous_byte = byte


class _BitWriter:
    __slots__ = ("bits", "buffer", "current")

    def __init__(self) -> None:
        self.buffer = bytearray()
        self.current = 0
        self.bits = 0

    def write(self, bit: int) -> None:
        self.current = (self.current << 1) | (bit & 1)
        self.bits += 1
        if self.bits == 8:
            self.buffer.append(self.current)
            self.current = 0
            self.bits = 0

    def finish(self) -> bytes:
        if self.bits:
            self.buffer.append(self.current << (8 - self.bits))
            self.current = 0
            self.bits = 0
        return bytes(self.buffer)


class _BitReader:
    __slots__ = ("bit_index", "byte_index", "data")

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.byte_index = 0
        self.bit_index = 0

    def read(self) -> int:
        if self.byte_index >= len(self.data):
            # Arithmetic coders conventionally zero-extend the final state.
            # Canonical re-encoding later distinguishes valid padding from a
            # truncated stream.
            return 0
        byte = self.data[self.byte_index]
        bit = (byte >> (7 - self.bit_index)) & 1
        self.bit_index += 1
        if self.bit_index == 8:
            self.bit_index = 0
            self.byte_index += 1
        return bit


class _ArithmeticEncoder:
    __slots__ = ("high", "low", "pending", "writer")

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

    def encode_bit(self, bit: int, probability_one: int) -> None:
        split = PROBABILITY_TOTAL - probability_one
        current_range = self.high - self.low + 1
        if bit:
            self.low += current_range * split // PROBABILITY_TOTAL
        else:
            self.high = self.low + current_range * split // PROBABILITY_TOTAL - 1

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
    __slots__ = ("code", "high", "low", "reader")

    def __init__(self, encoded: bytes) -> None:
        if not encoded:
            raise LogisticMixError("arithmetic stream is empty")
        self.reader = _BitReader(encoded)
        self.low = 0
        self.high = _FULL_RANGE - 1
        self.code = 0
        for _ in range(_STATE_BITS):
            self.code = (self.code << 1) | self.reader.read()

    def decode_bit(self, probability_one: int) -> int:
        if not self.low <= self.code <= self.high:
            raise LogisticMixError("arithmetic decoder state is inconsistent")
        current_range = self.high - self.low + 1
        target = ((self.code - self.low + 1) * PROBABILITY_TOTAL - 1) // current_range
        split = PROBABILITY_TOTAL - probability_one
        bit = int(target >= split)
        if bit:
            self.low += current_range * split // PROBABILITY_TOTAL
        else:
            self.high = self.low + current_range * split // PROBABILITY_TOTAL - 1

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
        return bit


def _encode_stream(raw: bytes) -> bytes:
    mixer = _Mixer()
    encoder = _ArithmeticEncoder()
    predict = mixer.predict
    update = mixer.update
    encode_bit = encoder.encode_bit
    for byte in raw:
        prefix = 0
        for bit_position in range(8):
            bit = (byte >> (7 - bit_position)) & 1
            probability_one, logits, order_key, run_key = predict(bit_position, prefix)
            encode_bit(bit, probability_one)
            update(
                bit=bit,
                probability_one=probability_one,
                expert_logits=logits,
                bit_position=bit_position,
                order_key=order_key,
                run_key=run_key,
            )
            prefix = (prefix << 1) | bit
        mixer.finish_byte(byte)
    return encoder.finish()


def _decode_stream(encoded: bytes, raw_length: int) -> bytes:
    mixer = _Mixer()
    decoder = _ArithmeticDecoder(encoded)
    restored = bytearray()
    predict = mixer.predict
    update = mixer.update
    decode_bit = decoder.decode_bit
    for _ in range(raw_length):
        prefix = 0
        for bit_position in range(8):
            probability_one, logits, order_key, run_key = predict(bit_position, prefix)
            bit = decode_bit(probability_one)
            update(
                bit=bit,
                probability_one=probability_one,
                expert_logits=logits,
                bit_position=bit_position,
                order_key=order_key,
                run_key=run_key,
            )
            prefix = (prefix << 1) | bit
        restored.append(prefix)
        mixer.finish_byte(prefix)
    return bytes(restored)


def encode_logistic_mix(
    data: bytes | bytearray | memoryview,
) -> bytes:
    """Encode arbitrary bytes into one strict canonical logistic-mix frame."""

    raw = _as_bytes(data)
    if len(raw) > MAX_RAW_BYTES:
        raise LogisticMixError(f"raw byte length exceeds format cap {MAX_RAW_BYTES}")
    encoded = _encode_stream(raw)
    return (
        _HEADER.pack(
            MAGIC,
            VERSION,
            EXPERT_COUNT,
            PROBABILITY_BITS,
            0,
            len(raw),
            len(encoded),
            hashlib.sha256(raw).digest(),
        )
        + encoded
    )


def decode_logistic_mix(
    frame: bytes | bytearray | memoryview,
) -> bytes:
    """Decode and canonically validate one logistic-mix frame."""

    payload = _as_bytes(frame)
    if len(payload) < _HEADER.size + 1:
        raise LogisticMixError("logistic-mix frame is truncated")
    (
        magic,
        version,
        expert_count,
        probability_bits,
        flags,
        raw_length,
        encoded_length,
        expected_digest,
    ) = _HEADER.unpack_from(payload)
    if (
        magic != MAGIC
        or version != VERSION
        or expert_count != EXPERT_COUNT
        or probability_bits != PROBABILITY_BITS
        or flags != 0
    ):
        raise LogisticMixError("logistic-mix frame contract differs")
    if raw_length > MAX_RAW_BYTES:
        raise LogisticMixError("declared raw byte length exceeds format cap")
    if encoded_length == 0 or len(payload) != _HEADER.size + encoded_length:
        raise LogisticMixError("logistic-mix coded length is noncanonical")
    encoded = payload[_HEADER.size :]
    try:
        raw = _decode_stream(encoded, int(raw_length))
    except (IndexError, OverflowError, ValueError) as exc:
        if isinstance(exc, LogisticMixError):
            raise
        raise LogisticMixError("logistic-mix arithmetic decode failed") from exc
    if hashlib.sha256(raw).digest() != expected_digest:
        raise LogisticMixError("logistic-mix raw digest differs")
    if _encode_stream(raw) != encoded:
        raise LogisticMixError("logistic-mix stream is noncanonical or has trailing data")
    return raw


def frame_accounting(
    frame: bytes | bytearray | memoryview,
) -> dict[str, int]:
    """Return exact counted frame/header/coded byte accounting."""

    payload = _as_bytes(frame)
    if len(payload) < _HEADER.size:
        raise LogisticMixError("logistic-mix frame is truncated")
    encoded_length = int(_HEADER.unpack_from(payload)[6])
    if encoded_length == 0 or len(payload) != _HEADER.size + encoded_length:
        raise LogisticMixError("logistic-mix coded length is noncanonical")
    return {
        "header_bytes": _HEADER.size,
        "model_parameter_bytes": 0,
        "coded_payload_bytes": encoded_length,
        "framed_bytes": len(payload),
    }


encode = encode_logistic_mix
decode = decode_logistic_mix

__all__ = [
    "EXPERT_COUNT",
    "LEARNING_RATE_DENOMINATOR",
    "PROBABILITY_BITS",
    "VERSION",
    "LogisticMixError",
    "decode",
    "decode_logistic_mix",
    "encode",
    "encode_logistic_mix",
    "frame_accounting",
]
