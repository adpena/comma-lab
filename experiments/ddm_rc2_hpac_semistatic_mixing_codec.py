"""Lossless RC2H coders for the packed IHS1 integer-model rows.

The module is both the encoder reference and the receiver implementation.  It is
standalone (stdlib + NumPy) so the experiment can copy these exact bytes into a
staged runtime.  Two video-derived objects may be carried, and both are inside the
counted stream:

* a two-pass frozen conditional probability table; and
* an int8 log-odds mixing-weight vector shared across IHS1 depths.

The logistic path is integer on the decoding decision surface.  ``stretch`` is a
deterministic fixed-point binary logarithm of the odds, and ``squash`` is its
monotone inverse.  No platform ``log``/``exp`` result enters arithmetic coding.
"""

from __future__ import annotations

import bisect
import math
import struct
from collections import defaultdict
from collections.abc import Iterable

import numpy as np

MAGIC = b"RC2H"
VERSION = 1
MODE_SEMISTATIC = 1
MODE_LOGISTIC = 2
MODE_COMBINED = 3

FAMILY_PREV_ZERO = 1
FAMILY_PREV_SIGN = 2
FAMILY_PREV_BITLEN = 3
FAMILY_EXACT_PREV = 4
FAMILY_NAMES = {
    FAMILY_PREV_ZERO: "previous_is_zero",
    FAMILY_PREV_SIGN: "previous_sign_zero",
    FAMILY_PREV_BITLEN: "previous_signed_bitlength",
    FAMILY_EXACT_PREV: "exact_previous_symbol",
}

SERIALIZER_COUNTS_ULEB = 1
SERIALIZER_PROB8 = 2
SERIALIZER_PROB12 = 3
SERIALIZER_NAMES = {
    SERIALIZER_COUNTS_ULEB: "sparse_counts_uleb128",
    SERIALIZER_PROB8: "sparse_probability_u8",
    SERIALIZER_PROB12: "sparse_probability_u12",
}

HEADER = struct.Struct("<4sBBBBIII")
IHS1_MAGIC = b"IHS1"
PROBABILITY_ONE = 4096
PROBABILITY_INITIAL = 2048
WEIGHT_SCALE = 32
EXPERT_COUNT = 8
STRETCH_SCALE = 4096


class Rc2CodecError(ValueError):
    """An RC2H stream or IHS1 layout is malformed."""


class _RangeEncoder:
    __slots__ = ("cache", "cache_size", "low", "out", "range")

    def __init__(self) -> None:
        self.low = 0
        self.range = 0xFFFFFFFF
        self.out = bytearray()
        self.cache = 0xFF
        self.cache_size = 0

    def _shift_low(self) -> None:
        if self.low < 0xFF000000 or self.low > 0xFFFFFFFF:
            carry = self.low >> 32
            if self.cache_size:
                self.out.append((self.cache + carry) & 0xFF)
            for _ in range(self.cache_size - 1):
                self.out.append((0xFF + carry) & 0xFF)
            self.cache = (self.low >> 24) & 0xFF
            self.cache_size = 0
        self.cache_size += 1
        self.low = (self.low << 8) & 0xFFFFFFFF

    def encode_bit(self, probability_zero: int, bit: int) -> None:
        probability_zero = int(probability_zero)
        if not 1 <= probability_zero < PROBABILITY_ONE:
            raise Rc2CodecError("binary probability is outside 1..4095")
        if bit:
            low, frequency = probability_zero, PROBABILITY_ONE - probability_zero
        else:
            low, frequency = 0, probability_zero
        unit = self.range // PROBABILITY_ONE
        self.low += unit * low
        self.range = unit * frequency
        while self.range < (1 << 24):
            self.range <<= 8
            self._shift_low()

    def finish(self) -> bytes:
        for _ in range(5):
            self._shift_low()
        return bytes(self.out)


class _RangeDecoder:
    __slots__ = ("buffer", "code", "position", "range")

    def __init__(self, payload: bytes) -> None:
        if len(payload) < 4:
            raise Rc2CodecError("range payload is shorter than its prefix")
        self.buffer = payload
        self.position = 0
        self.range = 0xFFFFFFFF
        self.code = 0
        for _ in range(4):
            self.code = ((self.code << 8) | self._byte()) & 0xFFFFFFFF

    def _byte(self) -> int:
        if self.position < len(self.buffer):
            value = self.buffer[self.position]
            self.position += 1
            return value
        self.position += 1
        return 0

    def decode_bit(self, probability_zero: int) -> int:
        probability_zero = int(probability_zero)
        if not 1 <= probability_zero < PROBABILITY_ONE:
            raise Rc2CodecError("binary probability is outside 1..4095")
        unit = self.range // PROBABILITY_ONE
        scaled = min(PROBABILITY_ONE - 1, self.code // unit)
        bit = int(scaled >= probability_zero)
        if bit:
            low, frequency = probability_zero, PROBABILITY_ONE - probability_zero
        else:
            low, frequency = 0, probability_zero
        self.code -= unit * low
        self.range = unit * frequency
        while self.range < (1 << 24):
            self.range <<= 8
            self.code = ((self.code << 8) | self._byte()) & 0xFFFFFFFF
        return bit


def _updated_probability(probability_zero: int, bit: int, shift: int = 5) -> int:
    if bit:
        return probability_zero - (probability_zero >> shift)
    return probability_zero + ((PROBABILITY_ONE - probability_zero) >> shift)


def _uleb(value: int) -> bytes:
    if value < 0:
        raise Rc2CodecError("ULEB128 cannot encode a negative value")
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        out.append(byte | (0x80 if value else 0))
        if not value:
            return bytes(out)


def _read_uleb(payload: bytes, offset: int) -> tuple[int, int]:
    value = 0
    shift = 0
    start = offset
    while True:
        if offset >= len(payload) or shift > 63:
            raise Rc2CodecError("truncated or oversized ULEB128")
        byte = payload[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            if payload[start:offset] != _uleb(value):
                raise Rc2CodecError("non-canonical ULEB128")
            return value, offset
        shift += 7


def _depths(prefix: bytes, row_count: int) -> np.ndarray:
    depth_bytes = (row_count + 1) // 2
    if len(prefix) != 4 + depth_bytes or not prefix.startswith(IHS1_MAGIC):
        raise Rc2CodecError("invalid IHS1 prefix")
    packed = np.frombuffer(prefix[4:], dtype=np.uint8)
    values = np.empty(depth_bytes * 2, dtype=np.uint8)
    values[0::2] = packed & 0xF
    values[1::2] = packed >> 4
    result = values[:row_count].astype(np.int64)
    if np.any(result > 8):
        raise Rc2CodecError("IHS1 depth is outside 0..8")
    return result


def split_ihs1(body: bytes, row_counts: list[int]) -> tuple[bytes, bytes, bytes, np.ndarray]:
    """Return ``prefix, packed_weights, tail, depths`` from an IHS1 body."""

    prefix_bytes = 4 + (len(row_counts) + 1) // 2
    if len(body) < prefix_bytes or not body.startswith(IHS1_MAGIC):
        raise Rc2CodecError("not an IHS1 body")
    prefix = body[:prefix_bytes]
    depths = _depths(prefix, len(row_counts))
    total_bits = sum(int(depth) * int(count) for depth, count in zip(depths.tolist(), row_counts, strict=True))
    weight_bytes = (total_bits + 7) // 8
    if prefix_bytes + weight_bytes > len(body):
        raise Rc2CodecError("IHS1 packed weights overrun the body")
    return (
        prefix,
        body[prefix_bytes : prefix_bytes + weight_bytes],
        body[prefix_bytes + weight_bytes :],
        depths,
    )


def unpack_rows(body: bytes, row_counts: list[int]) -> tuple[list[np.ndarray], np.ndarray]:
    prefix, packed, _tail, depths = split_ihs1(body, row_counts)
    del prefix
    total_bits = sum(int(depth) * int(count) for depth, count in zip(depths.tolist(), row_counts, strict=True))
    stream = np.unpackbits(np.frombuffer(packed, dtype=np.uint8), bitorder="little")[:total_bits]
    rows: list[np.ndarray] = []
    cursor = 0
    for depth, count in zip(depths.tolist(), row_counts, strict=True):
        depth, count = int(depth), int(count)
        if depth == 0:
            rows.append(np.zeros(count, dtype=np.int16))
            continue
        span = depth * count
        block = stream[cursor : cursor + span].reshape(count, depth).astype(np.int32)
        unsigned = (block * (1 << np.arange(depth, dtype=np.int32))).sum(axis=1)
        sign = 1 << (depth - 1)
        rows.append(np.where(unsigned >= sign, unsigned - (1 << depth), unsigned).astype(np.int16))
        cursor += span
    if cursor != total_bits:
        raise Rc2CodecError("IHS1 row walk did not consume all weight bits")
    return rows, depths


def pack_rows(rows: list[np.ndarray], depths: np.ndarray) -> bytes:
    if len(rows) != len(depths):
        raise Rc2CodecError("row/depth count differs")
    chunks: list[np.ndarray] = []
    for values, depth in zip(rows, depths.tolist(), strict=True):
        depth = int(depth)
        if depth == 0:
            if np.any(values):
                raise Rc2CodecError("zero-depth row contains a nonzero value")
            continue
        unsigned = np.asarray(values, dtype=np.int32) & ((1 << depth) - 1)
        block = np.empty((unsigned.size, depth), dtype=np.uint8)
        for index in range(depth):
            block[:, index] = (unsigned >> index) & 1
        chunks.append(block.reshape(-1))
    bits = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.uint8)
    if (-bits.size) % 8:
        bits = np.concatenate([bits, np.zeros((-bits.size) % 8, dtype=np.uint8)])
    return np.packbits(bits, bitorder="little").tobytes()


def _context_count(family: int, depth: int) -> int:
    if family == FAMILY_PREV_ZERO:
        return 2
    if family == FAMILY_PREV_SIGN:
        return 3
    if family == FAMILY_PREV_BITLEN:
        return 2 * depth + 1
    if family == FAMILY_EXACT_PREV:
        return 1 << depth
    raise Rc2CodecError(f"unknown semi-static family {family}")


def _context(family: int, previous: int, depth: int) -> int:
    if family == FAMILY_PREV_ZERO:
        return int(previous != 0)
    if family == FAMILY_PREV_SIGN:
        return 0 if previous < 0 else 1 if previous == 0 else 2
    if family == FAMILY_PREV_BITLEN:
        if previous == 0:
            return 0
        magnitude_bits = abs(int(previous)).bit_length()
        return magnitude_bits if previous > 0 else depth + magnitude_bits
    if family == FAMILY_EXACT_PREV:
        return int(previous) & ((1 << depth) - 1)
    raise Rc2CodecError(f"unknown semi-static family {family}")


def context_ordinal(family: int, depth: int, context: int, node: int) -> int:
    if not 1 <= depth <= 8 or not 1 <= node < (1 << depth):
        raise Rc2CodecError("semi-static key leaves its tree domain")
    contexts = _context_count(family, depth)
    if not 0 <= context < contexts:
        raise Rc2CodecError("semi-static context leaves its domain")
    offset = sum(_context_count(family, prior) * ((1 << prior) - 1) for prior in range(1, depth))
    return offset + context * ((1 << depth) - 1) + node - 1


def fit_semistatic_counts(rows: list[np.ndarray], depths: np.ndarray, family: int) -> dict[int, tuple[int, int]]:
    """Pass 1: count zero/one outcomes.  First symbol per depth is uniform."""

    counts: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    previous: dict[int, int] = {}
    for values, depth in zip(rows, depths.tolist(), strict=True):
        depth = int(depth)
        if depth == 0:
            continue
        for signed in np.asarray(values, dtype=np.int16).tolist():
            prior = previous.get(depth)
            unsigned = int(signed) & ((1 << depth) - 1)
            node = 1
            for index in range(depth - 1, -1, -1):
                bit = (unsigned >> index) & 1
                if prior is not None:
                    ordinal = context_ordinal(family, depth, _context(family, prior, depth), node)
                    counts[ordinal][bit] += 1
                node = node * 2 + bit
            previous[depth] = int(signed)
    return {key: (value[0], value[1]) for key, value in counts.items()}


def _probability_from_counts(zero: int, one: int) -> int:
    total = zero + one
    if total <= 0:
        return PROBABILITY_INITIAL
    return min(
        PROBABILITY_ONE - 1,
        max(1, (zero * PROBABILITY_ONE + total // 2) // total),
    )


def serialize_table(counts: dict[int, tuple[int, int]], serializer: int) -> bytes:
    out = bytearray(_uleb(len(counts)))
    previous = -1
    for ordinal in sorted(counts):
        zero, one = counts[ordinal]
        if zero + one <= 0:
            raise Rc2CodecError("empty semi-static cell was serialized")
        out.extend(_uleb(ordinal - previous - 1))
        if serializer == SERIALIZER_COUNTS_ULEB:
            out.extend(_uleb(zero))
            out.extend(_uleb(one))
        elif serializer == SERIALIZER_PROB8:
            probability = _probability_from_counts(zero, one)
            quantized = min(255, max(0, (probability * 256 + 2048) // 4096))
            out.append(quantized)
        elif serializer == SERIALIZER_PROB12:
            out.extend(struct.pack("<H", _probability_from_counts(zero, one)))
        else:
            raise Rc2CodecError(f"unknown table serializer {serializer}")
        previous = ordinal
    return bytes(out)


def parse_table(payload: bytes, serializer: int) -> dict[int, int]:
    count, offset = _read_uleb(payload, 0)
    table: dict[int, int] = {}
    previous = -1
    for _ in range(count):
        gap, offset = _read_uleb(payload, offset)
        ordinal = previous + gap + 1
        if ordinal <= previous:
            raise Rc2CodecError("semi-static table keys are not increasing")
        if serializer == SERIALIZER_COUNTS_ULEB:
            zero, offset = _read_uleb(payload, offset)
            one, offset = _read_uleb(payload, offset)
            probability = _probability_from_counts(zero, one)
        elif serializer == SERIALIZER_PROB8:
            if offset >= len(payload):
                raise Rc2CodecError("truncated uint8 probability table")
            quantized = payload[offset]
            offset += 1
            probability = min(4095, max(1, quantized * 16))
        elif serializer == SERIALIZER_PROB12:
            if offset + 2 > len(payload):
                raise Rc2CodecError("truncated uint12 probability table")
            probability = struct.unpack_from("<H", payload, offset)[0]
            offset += 2
            if not 1 <= probability < PROBABILITY_ONE:
                raise Rc2CodecError("uint12 table probability is outside 1..4095")
        else:
            raise Rc2CodecError(f"unknown table serializer {serializer}")
        table[ordinal] = probability
        previous = ordinal
    if offset != len(payload):
        raise Rc2CodecError("semi-static table has trailing bytes")
    return table


def _encode_semistatic_rows(
    rows: list[np.ndarray], depths: np.ndarray, family: int, table: dict[int, int]
) -> tuple[bytes, float]:
    encoder = _RangeEncoder()
    ideal_bits = 0.0
    previous: dict[int, int] = {}
    for values, depth in zip(rows, depths.tolist(), strict=True):
        depth = int(depth)
        if depth == 0:
            continue
        for signed in np.asarray(values, dtype=np.int16).tolist():
            prior = previous.get(depth)
            unsigned = int(signed) & ((1 << depth) - 1)
            node = 1
            for index in range(depth - 1, -1, -1):
                bit = (unsigned >> index) & 1
                probability = PROBABILITY_INITIAL
                if prior is not None:
                    ordinal = context_ordinal(family, depth, _context(family, prior, depth), node)
                    probability = table.get(ordinal, PROBABILITY_INITIAL)
                encoder.encode_bit(probability, bit)
                frequency = PROBABILITY_ONE - probability if bit else probability
                ideal_bits -= math.log2(frequency / PROBABILITY_ONE)
                node = node * 2 + bit
            previous[depth] = int(signed)
    return encoder.finish(), ideal_bits


def _decode_semistatic_rows(
    payload: bytes,
    row_counts: list[int],
    depths: np.ndarray,
    family: int,
    table: dict[int, int],
) -> list[np.ndarray]:
    decoder = _RangeDecoder(payload)
    previous: dict[int, int] = {}
    rows: list[np.ndarray] = []
    for count, depth in zip(row_counts, depths.tolist(), strict=True):
        count, depth = int(count), int(depth)
        values = np.zeros(count, dtype=np.int16)
        if depth == 0:
            rows.append(values)
            continue
        sign = 1 << (depth - 1)
        span = 1 << depth
        for row_index in range(count):
            prior = previous.get(depth)
            node = 1
            unsigned = 0
            for _ in range(depth):
                probability = PROBABILITY_INITIAL
                if prior is not None:
                    ordinal = context_ordinal(family, depth, _context(family, prior, depth), node)
                    probability = table.get(ordinal, PROBABILITY_INITIAL)
                bit = decoder.decode_bit(probability)
                unsigned = (unsigned << 1) | bit
                node = node * 2 + bit
            signed = unsigned - span if unsigned >= sign else unsigned
            values[row_index] = signed
            previous[depth] = signed
        rows.append(values)
    return rows


def _log2_ratio_fixed(numerator: int, denominator: int, fractional_bits: int = 12) -> int:
    """Deterministic floor(log2(numerator/denominator) * 2**fractional_bits)."""

    if numerator <= 0 or denominator <= 0:
        raise Rc2CodecError("fixed log2 requires positive integers")
    exponent = numerator.bit_length() - denominator.bit_length()
    if exponent >= 0:
        if numerator < (denominator << exponent):
            exponent -= 1
    elif (numerator << -exponent) < denominator:
        exponent -= 1
    if exponent >= 0:
        n, d = numerator, denominator << exponent
    else:
        n, d = numerator << -exponent, denominator
    fraction = 0
    for _ in range(fractional_bits):
        n *= n
        d *= d
        fraction <<= 1
        if n >= 2 * d:
            n //= 2
            fraction |= 1
    return exponent * (1 << fractional_bits) + fraction


STRETCH = tuple(
    0 if probability == PROBABILITY_INITIAL else _log2_ratio_fixed(probability, PROBABILITY_ONE - probability)
    for probability in range(1, PROBABILITY_ONE)
)


def stretch(probability_zero: int) -> int:
    if not 1 <= probability_zero < PROBABILITY_ONE:
        raise Rc2CodecError("stretch probability is outside 1..4095")
    return STRETCH[probability_zero - 1]


def squash(stretched: int) -> int:
    """Nearest monotone inverse of :func:`stretch`, tie-breaking downward."""

    index = bisect.bisect_left(STRETCH, int(stretched))
    if index <= 0:
        return 1
    if index >= len(STRETCH):
        return PROBABILITY_ONE - 1
    before, after = STRETCH[index - 1], STRETCH[index]
    return index if stretched - before <= after - stretched else index + 1


def _round_div_signed(value: int, denominator: int) -> int:
    if denominator <= 0:
        raise Rc2CodecError("rounding denominator must be positive")
    if value >= 0:
        return (value + denominator // 2) // denominator
    return -((-value + denominator // 2) // denominator)


class _AdaptiveExperts:
    """Eight causal binary predictors; every state is receiver-reproducible."""

    __slots__ = ("banks",)

    def __init__(self) -> None:
        self.banks: list[dict[object, int]] = [{} for _ in range(EXPERT_COUNT)]

    def keys(
        self,
        depth: int,
        node: int,
        bit_position: int,
        previous: int | None,
    ) -> list[object]:
        unknown = previous is None
        previous_value = 0 if previous is None else int(previous)
        zero_context = 2 if unknown else int(previous_value != 0)
        sign_context = 3 if unknown else (0 if previous_value < 0 else 1 if previous_value == 0 else 2)
        bitlen_context = 2 * depth + 1 if unknown else _context(FAMILY_PREV_BITLEN, previous_value, depth)
        exact_context = 1 << depth if unknown else _context(FAMILY_EXACT_PREV, previous_value, depth)
        return [
            (depth, node),
            (depth, zero_context, node),
            (depth, sign_context, node),
            (depth, bitlen_context, node),
            (depth, exact_context, node),
            (bit_position, node),
            bit_position,
            0,
        ]

    def predictions(self, keys: list[object]) -> list[int]:
        return [bank.get(key, PROBABILITY_INITIAL) for bank, key in zip(self.banks, keys, strict=True)]

    def update(self, keys: list[object], bit: int) -> None:
        for bank, key in zip(self.banks, keys, strict=True):
            probability = bank.get(key, PROBABILITY_INITIAL)
            bank[key] = _updated_probability(probability, bit)


def weight_group(depth: int, bit_position: int, groups: int) -> int:
    if groups == 1:
        return 0
    if groups == 2:
        return int(bit_position != 0)
    if groups == 4:
        return min(3, (4 * bit_position) // depth)
    if groups == 8:
        return bit_position
    raise Rc2CodecError("logistic mixer group count must be 1, 2, 4, or 8")


def mixed_probability(predictions: Iterable[int], weights: np.ndarray) -> int:
    prediction_list = list(predictions)
    weight_array = np.asarray(weights, dtype=np.int16).reshape(-1)
    if len(prediction_list) != EXPERT_COUNT or weight_array.size != EXPERT_COUNT:
        raise Rc2CodecError("logistic mixer requires exactly eight experts and weights")
    accumulated = sum(
        int(weight) * stretch(int(probability))
        for probability, weight in zip(prediction_list, weight_array.tolist(), strict=True)
    )
    return squash(_round_div_signed(accumulated, WEIGHT_SCALE))


def collect_logistic_events(
    rows: list[np.ndarray], depths: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Collect causal expert stretches and grouping coordinates for offline fitting."""

    experts = _AdaptiveExperts()
    previous: dict[int, int] = {}
    feature_rows: list[list[int]] = []
    outcomes: list[int] = []
    depth_rows: list[int] = []
    position_rows: list[int] = []
    node_rows: list[int] = []
    for values, depth in zip(rows, depths.tolist(), strict=True):
        depth = int(depth)
        if depth == 0:
            continue
        for signed in np.asarray(values, dtype=np.int16).tolist():
            prior = previous.get(depth)
            unsigned = int(signed) & ((1 << depth) - 1)
            node = 1
            for bit_position, index in enumerate(range(depth - 1, -1, -1)):
                bit = (unsigned >> index) & 1
                keys = experts.keys(depth, node, bit_position, prior)
                predictions = experts.predictions(keys)
                feature_rows.append([stretch(value) for value in predictions])
                outcomes.append(int(bit == 0))
                depth_rows.append(depth)
                position_rows.append(bit_position)
                node_rows.append(node)
                experts.update(keys, bit)
                node = node * 2 + bit
            previous[depth] = int(signed)
    return (
        np.asarray(feature_rows, dtype=np.int16),
        np.asarray(outcomes, dtype=np.uint8),
        np.asarray(depth_rows, dtype=np.uint8),
        np.asarray(position_rows, dtype=np.uint8),
        np.asarray(node_rows, dtype=np.uint16),
    )


def _encode_logistic_rows(rows: list[np.ndarray], depths: np.ndarray, weights: np.ndarray) -> tuple[bytes, float]:
    weights = np.asarray(weights, dtype=np.int8)
    if weights.ndim != 2 or weights.shape[1] != EXPERT_COUNT or weights.shape[0] not in (1, 2, 4, 8):
        raise Rc2CodecError("weight matrix shape must be (1|2|4|8, 8)")
    encoder = _RangeEncoder()
    experts = _AdaptiveExperts()
    previous: dict[int, int] = {}
    ideal_bits = 0.0
    for values, depth in zip(rows, depths.tolist(), strict=True):
        depth = int(depth)
        if depth == 0:
            continue
        for signed in np.asarray(values, dtype=np.int16).tolist():
            prior = previous.get(depth)
            unsigned = int(signed) & ((1 << depth) - 1)
            node = 1
            for bit_position, index in enumerate(range(depth - 1, -1, -1)):
                bit = (unsigned >> index) & 1
                keys = experts.keys(depth, node, bit_position, prior)
                predictions = experts.predictions(keys)
                group = weight_group(depth, bit_position, weights.shape[0])
                probability = mixed_probability(predictions, weights[group])
                encoder.encode_bit(probability, bit)
                frequency = PROBABILITY_ONE - probability if bit else probability
                ideal_bits -= math.log2(frequency / PROBABILITY_ONE)
                experts.update(keys, bit)
                node = node * 2 + bit
            previous[depth] = int(signed)
    return encoder.finish(), ideal_bits


def _decode_logistic_rows(
    payload: bytes, row_counts: list[int], depths: np.ndarray, weights: np.ndarray
) -> list[np.ndarray]:
    weights = np.asarray(weights, dtype=np.int8)
    if weights.ndim != 2 or weights.shape[1] != EXPERT_COUNT or weights.shape[0] not in (1, 2, 4, 8):
        raise Rc2CodecError("weight matrix shape must be (1|2|4|8, 8)")
    decoder = _RangeDecoder(payload)
    experts = _AdaptiveExperts()
    previous: dict[int, int] = {}
    rows: list[np.ndarray] = []
    for count, depth in zip(row_counts, depths.tolist(), strict=True):
        count, depth = int(count), int(depth)
        values = np.zeros(count, dtype=np.int16)
        if depth == 0:
            rows.append(values)
            continue
        sign = 1 << (depth - 1)
        span = 1 << depth
        for row_index in range(count):
            prior = previous.get(depth)
            node = 1
            unsigned = 0
            for bit_position in range(depth):
                keys = experts.keys(depth, node, bit_position, prior)
                predictions = experts.predictions(keys)
                group = weight_group(depth, bit_position, weights.shape[0])
                probability = mixed_probability(predictions, weights[group])
                bit = decoder.decode_bit(probability)
                experts.update(keys, bit)
                unsigned = (unsigned << 1) | bit
                node = node * 2 + bit
            signed = unsigned - span if unsigned >= sign else unsigned
            values[row_index] = signed
            previous[depth] = signed
        rows.append(values)
    return rows


def _build_rider(
    body: bytes,
    row_counts: list[int],
    *,
    mode: int,
    family: int,
    serializer: int,
    table_blob: bytes,
    weights_blob: bytes,
    payload: bytes,
) -> bytes:
    prefix, _weights, tail, _depths_array = split_ihs1(body, row_counts)
    header = HEADER.pack(
        MAGIC,
        VERSION,
        mode,
        family,
        serializer,
        len(table_blob),
        len(weights_blob),
        len(payload),
    )
    return header + prefix + table_blob + weights_blob + payload + tail


def encode_semistatic(
    body: bytes, row_counts: list[int], family: int, serializer: int
) -> tuple[bytes, dict[str, object]]:
    rows, depths = unpack_rows(body, row_counts)
    counts = fit_semistatic_counts(rows, depths, family)
    table_blob = serialize_table(counts, serializer)
    table = parse_table(table_blob, serializer)
    payload, ideal_bits = _encode_semistatic_rows(rows, depths, family, table)
    rider = _build_rider(
        body,
        row_counts,
        mode=MODE_SEMISTATIC,
        family=family,
        serializer=serializer,
        table_blob=table_blob,
        weights_blob=b"",
        payload=payload,
    )
    return rider, {
        "table_blob": table_blob,
        "weights_blob": b"",
        "payload": payload,
        "table_entries": len(counts),
        "ideal_bits": ideal_bits,
    }


def encode_logistic(body: bytes, row_counts: list[int], weights: np.ndarray) -> tuple[bytes, dict[str, object]]:
    rows, depths = unpack_rows(body, row_counts)
    weights_array = np.asarray(weights, dtype=np.int8)
    payload, ideal_bits = _encode_logistic_rows(rows, depths, weights_array)
    weights_blob = weights_array.tobytes(order="C")
    rider = _build_rider(
        body,
        row_counts,
        mode=MODE_LOGISTIC,
        family=0,
        serializer=0,
        table_blob=b"",
        weights_blob=weights_blob,
        payload=payload,
    )
    return rider, {
        "table_blob": b"",
        "weights_blob": weights_blob,
        "payload": payload,
        "table_entries": 0,
        "ideal_bits": ideal_bits,
    }


def parse_rider(stream: bytes, row_counts: list[int]) -> dict[str, object]:
    if len(stream) < HEADER.size:
        raise Rc2CodecError("RC2H stream is shorter than its header")
    magic, version, mode, family, serializer, table_bytes, weight_bytes, payload_bytes = HEADER.unpack_from(stream)
    if magic != MAGIC or version != VERSION or mode not in (MODE_SEMISTATIC, MODE_LOGISTIC):
        raise Rc2CodecError("unsupported RC2H header")
    prefix_bytes = 4 + (len(row_counts) + 1) // 2
    offset = HEADER.size
    prefix = stream[offset : offset + prefix_bytes]
    depths = _depths(prefix, len(row_counts))
    offset += prefix_bytes
    end_table = offset + table_bytes
    end_weights = end_table + weight_bytes
    end_payload = end_weights + payload_bytes
    if end_payload > len(stream):
        raise Rc2CodecError("RC2H component lengths overrun the stream")
    table_blob = stream[offset:end_table]
    weights_blob = stream[end_table:end_weights]
    payload = stream[end_weights:end_payload]
    tail = stream[end_payload:]
    if mode == MODE_SEMISTATIC:
        if family not in FAMILY_NAMES or serializer not in SERIALIZER_NAMES or weight_bytes:
            raise Rc2CodecError("invalid semi-static RC2H metadata")
        table = parse_table(table_blob, serializer)
        rows = _decode_semistatic_rows(payload, row_counts, depths, family, table)
    else:
        if family or serializer or table_bytes or weight_bytes not in (8, 16, 32, 64):
            raise Rc2CodecError("invalid logistic RC2H metadata")
        weights = np.frombuffer(weights_blob, dtype=np.int8).reshape(-1, EXPERT_COUNT)
        rows = _decode_logistic_rows(payload, row_counts, depths, weights)
    body = prefix + pack_rows(rows, depths) + tail
    return {
        "body": body,
        "mode": mode,
        "family": family,
        "serializer": serializer,
        "table_blob": table_blob,
        "weights_blob": weights_blob,
        "payload": payload,
        "tail": tail,
    }


def restore_hpac(stream: bytes, row_counts: list[int]) -> bytes:
    return parse_rider(stream, row_counts)["body"]  # type: ignore[return-value]


__all__ = [
    "EXPERT_COUNT",
    "FAMILY_EXACT_PREV",
    "FAMILY_NAMES",
    "FAMILY_PREV_BITLEN",
    "FAMILY_PREV_SIGN",
    "FAMILY_PREV_ZERO",
    "MAGIC",
    "MODE_LOGISTIC",
    "MODE_SEMISTATIC",
    "SERIALIZER_COUNTS_ULEB",
    "SERIALIZER_NAMES",
    "SERIALIZER_PROB8",
    "SERIALIZER_PROB12",
    "STRETCH_SCALE",
    "WEIGHT_SCALE",
    "Rc2CodecError",
    "collect_logistic_events",
    "encode_logistic",
    "encode_semistatic",
    "fit_semistatic_counts",
    "mixed_probability",
    "pack_rows",
    "parse_rider",
    "parse_table",
    "restore_hpac",
    "serialize_table",
    "split_ihs1",
    "squash",
    "stretch",
    "unpack_rows",
    "weight_group",
]
