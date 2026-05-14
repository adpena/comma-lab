# SPDX-License-Identifier: MIT
from __future__ import annotations

from bisect import bisect_right

import numpy as np


STATE_BITS = 32
FULL_RANGE = 1 << STATE_BITS
HALF = FULL_RANGE >> 1
QUARTER = HALF >> 1
THREE_QUARTERS = QUARTER * 3


def normalize_probabilities(probabilities, *, total: int = 1 << 15) -> list[int]:
    values = [float(item) for item in probabilities]
    if total <= 0:
        raise ValueError("total must be positive")
    if not values:
        raise ValueError("probabilities must be non-empty")
    if any(item < 0.0 for item in values):
        raise ValueError("probabilities must be non-negative")
    total_prob = sum(values)
    if total_prob <= 0.0:
        raise ValueError("probabilities must sum to a positive value")

    scaled = [max(1, int(round(item / total_prob * total))) for item in values]
    delta = total - sum(scaled)
    if delta > 0:
        order = sorted(range(len(values)), key=lambda idx: values[idx], reverse=True)
        for index in order[:delta]:
            scaled[index] += 1
    elif delta < 0:
        order = sorted(range(len(values)), key=lambda idx: values[idx])
        remaining = -delta
        for index in order:
            while remaining and scaled[index] > 1:
                scaled[index] -= 1
                remaining -= 1
            if remaining == 0:
                break
    if sum(scaled) != total or any(item <= 0 for item in scaled):
        raise ValueError("failed to normalize probabilities into a positive frequency table")
    return scaled


def normalize_probability_rows(probabilities, *, total: int = 1 << 15):
    values = np.asarray(probabilities, dtype=np.float64)
    squeeze = False
    if values.ndim == 1:
        values = values.reshape(1, -1)
        squeeze = True
    if values.ndim != 2:
        raise ValueError("probabilities must be a 1D or 2D array")
    if total <= 0:
        raise ValueError("total must be positive")
    if values.shape[1] == 0:
        raise ValueError("probabilities must be non-empty")
    if np.any(values < 0.0):
        raise ValueError("probabilities must be non-negative")

    total_prob = values.sum(axis=1)
    if np.any(total_prob <= 0.0):
        raise ValueError("probabilities must sum to a positive value")

    scaled = np.rint(values / total_prob[:, None] * total).astype(np.int64)
    scaled = np.maximum(scaled, 1)
    deltas = total - scaled.sum(axis=1)
    for row_index, delta in enumerate(deltas.tolist()):
        if delta > 0:
            order = np.argsort(-values[row_index], kind="stable")
            scaled[row_index, order[:delta]] += 1
        elif delta < 0:
            order = np.argsort(values[row_index], kind="stable")
            remaining = -delta
            for column_index in order.tolist():
                reducible = int(scaled[row_index, column_index] - 1)
                if reducible <= 0:
                    continue
                take = min(reducible, remaining)
                scaled[row_index, column_index] -= take
                remaining -= take
                if remaining == 0:
                    break
        if int(scaled[row_index].sum()) != total or np.any(scaled[row_index] <= 0):
            raise ValueError("failed to normalize probabilities into a positive frequency table")
    return scaled[0] if squeeze else scaled


def cumulative_frequency_rows(frequencies):
    values = np.asarray(frequencies, dtype=np.int64)
    squeeze = False
    if values.ndim == 1:
        values = values.reshape(1, -1)
        squeeze = True
    if values.ndim != 2:
        raise ValueError("frequencies must be a 1D or 2D array")
    if values.shape[1] == 0:
        raise ValueError("frequencies must be non-empty")
    if np.any(values <= 0):
        raise ValueError("frequencies must be positive")
    cumulative = np.concatenate(
        [np.zeros((values.shape[0], 1), dtype=np.int64), np.cumsum(values, axis=1, dtype=np.int64)],
        axis=1,
    )
    return cumulative[0] if squeeze else cumulative


def _coerce_int(value, *, label: str) -> int:
    try:
        coerced = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer") from exc
    if coerced != value:
        raise ValueError(f"{label} must be an integer")
    return coerced


def _validate_positive_total(total: int) -> int:
    value = _coerce_int(total, label="total")
    if value <= 0:
        raise ValueError("total must be positive")
    return value


def _validate_cumulative_table(cumulative: list[int], total: int) -> tuple[list[int], int, int]:
    total_value = _validate_positive_total(total)
    values = [_coerce_int(item, label="cumulative frequency") for item in cumulative]
    if len(values) < 2:
        raise ValueError("cumulative frequency table must contain at least two entries")
    if values[0] != 0:
        raise ValueError("cumulative frequency table must start at zero")
    if values[-1] != total_value:
        raise ValueError("cumulative frequency table must end at total")
    previous = values[0]
    for item in values[1:]:
        if item <= previous:
            raise ValueError("cumulative frequency table must be strictly increasing")
        previous = item
    return values, total_value, len(values) - 1


def _validate_symbol(symbol: int, *, symbol_count: int) -> int:
    index = _coerce_int(symbol, label="symbol")
    if index < 0 or index >= symbol_count:
        raise ValueError("symbol is outside the frequency table")
    return index


class _BitWriter:
    def __init__(self) -> None:
        self._buffer = bytearray()
        self._current = 0
        self._bits = 0

    def write(self, bit: int) -> None:
        self._current = (self._current << 1) | (bit & 1)
        self._bits += 1
        if self._bits == 8:
            self._buffer.append(self._current)
            self._current = 0
            self._bits = 0

    def finish(self) -> bytes:
        if self._bits:
            self._buffer.append(self._current << (8 - self._bits))
            self._current = 0
            self._bits = 0
        return bytes(self._buffer)


class _BitReader:
    def __init__(self, data: bytes) -> None:
        self._data = data
        self._byte_index = 0
        self._bit_index = 0

    def read(self) -> int:
        if self._byte_index >= len(self._data):
            return 0
        byte = self._data[self._byte_index]
        bit = (byte >> (7 - self._bit_index)) & 1
        self._bit_index += 1
        if self._bit_index == 8:
            self._bit_index = 0
            self._byte_index += 1
        return bit


def _cumulative(frequencies: list[int]) -> tuple[list[int], int]:
    total = sum(int(item) for item in frequencies)
    if total <= 0:
        raise ValueError("frequencies must sum to a positive total")
    cumulative = [0]
    running = 0
    for item in frequencies:
        value = int(item)
        if value <= 0:
            raise ValueError("frequencies must be positive")
        running += value
        cumulative.append(running)
    return cumulative, total


def cumulative_frequencies(frequencies: list[int]) -> tuple[list[int], int]:
    return _cumulative(frequencies)


class RangeEncoder:
    def __init__(self) -> None:
        self._writer = _BitWriter()
        self._low = 0
        self._high = FULL_RANGE - 1
        self._pending = 0

    def _emit(self, bit: int) -> None:
        self._writer.write(bit)
        while self._pending:
            self._writer.write(1 - bit)
            self._pending -= 1

    def encode(self, *, symbol: int, cumulative: list[int], total: int) -> None:
        cumulative, total, symbol_count = _validate_cumulative_table(cumulative, total)
        symbol = _validate_symbol(symbol, symbol_count=symbol_count)
        current_range = self._high - self._low + 1
        self._high = self._low + (current_range * cumulative[symbol + 1] // total) - 1
        self._low = self._low + (current_range * cumulative[symbol] // total)

        while True:
            if self._high < HALF:
                self._emit(0)
            elif self._low >= HALF:
                self._emit(1)
                self._low -= HALF
                self._high -= HALF
            elif self._low >= QUARTER and self._high < THREE_QUARTERS:
                self._pending += 1
                self._low -= QUARTER
                self._high -= QUARTER
            else:
                break
            self._low <<= 1
            self._high = (self._high << 1) | 1

    def finish(self) -> bytes:
        self._pending += 1
        self._emit(0 if self._low < QUARTER else 1)
        return self._writer.finish()


class RangeDecoder:
    def __init__(self, encoded: bytes) -> None:
        if len(encoded) == 0:
            raise ValueError("encoded range stream is empty")
        self._reader = _BitReader(encoded)
        self._low = 0
        self._high = FULL_RANGE - 1
        self._code = 0
        for _ in range(STATE_BITS):
            self._code = (self._code << 1) | self._reader.read()

    def _validate_state(self) -> None:
        if not (self._low <= self._code <= self._high):
            raise ValueError("encoded range stream is inconsistent with decoder state")

    def target(self, total: int) -> int:
        total = _validate_positive_total(total)
        self._validate_state()
        current_range = self._high - self._low + 1
        return ((self._code - self._low + 1) * total - 1) // current_range

    def update(self, *, low_count: int, high_count: int, total: int) -> None:
        low_count = _coerce_int(low_count, label="low_count")
        high_count = _coerce_int(high_count, label="high_count")
        total = _validate_positive_total(total)
        if low_count < 0 or high_count <= low_count or high_count > total:
            raise ValueError("range update interval must satisfy 0 <= low_count < high_count <= total")
        self._validate_state()
        current_range = self._high - self._low + 1
        self._high = self._low + (current_range * high_count // total) - 1
        self._low = self._low + (current_range * low_count // total)
        while True:
            if self._high < HALF:
                pass
            elif self._low >= HALF:
                self._low -= HALF
                self._high -= HALF
                self._code -= HALF
            elif self._low >= QUARTER and self._high < THREE_QUARTERS:
                self._low -= QUARTER
                self._high -= QUARTER
                self._code -= QUARTER
            else:
                break
            self._low <<= 1
            self._high = (self._high << 1) | 1
            self._code = (self._code << 1) | self._reader.read()


def encode_static_symbols(symbols, *, frequencies: list[int]) -> bytes:
    cumulative, total = _cumulative(frequencies)
    writer = _BitWriter()
    low = 0
    high = FULL_RANGE - 1
    pending = 0

    def emit(bit: int) -> None:
        nonlocal pending
        writer.write(bit)
        while pending:
            writer.write(1 - bit)
            pending -= 1

    for symbol in symbols:
        index = int(symbol)
        if index < 0 or index >= len(frequencies):
            raise ValueError("symbol is outside the frequency table")
        current_range = high - low + 1
        high = low + (current_range * cumulative[index + 1] // total) - 1
        low = low + (current_range * cumulative[index] // total)

        while True:
            if high < HALF:
                emit(0)
            elif low >= HALF:
                emit(1)
                low -= HALF
                high -= HALF
            elif low >= QUARTER and high < THREE_QUARTERS:
                pending += 1
                low -= QUARTER
                high -= QUARTER
            else:
                break
            low <<= 1
            high = (high << 1) | 1

    pending += 1
    emit(0 if low < QUARTER else 1)
    return writer.finish()


def decode_static_symbols(encoded: bytes, *, count: int, frequencies: list[int]) -> list[int]:
    if count < 0:
        raise ValueError("count must be non-negative")
    if count > 0 and len(encoded) == 0:
        raise ValueError("encoded range stream is empty")
    cumulative, total = _cumulative(frequencies)
    reader = _BitReader(encoded)
    low = 0
    high = FULL_RANGE - 1
    code = 0
    for _ in range(STATE_BITS):
        code = (code << 1) | reader.read()

    restored: list[int] = []
    for _ in range(count):
        current_range = high - low + 1
        scaled = ((code - low + 1) * total - 1) // current_range
        if scaled < 0 or scaled >= total:
            raise ValueError("encoded range stream is inconsistent with frequency table")
        symbol = bisect_right(cumulative, scaled) - 1
        if symbol < 0 or symbol + 1 >= len(cumulative):
            raise ValueError("encoded range stream is inconsistent with frequency table")
        restored.append(symbol)

        high = low + (current_range * cumulative[symbol + 1] // total) - 1
        low = low + (current_range * cumulative[symbol] // total)

        while True:
            if high < HALF:
                pass
            elif low >= HALF:
                low -= HALF
                high -= HALF
                code -= HALF
            elif low >= QUARTER and high < THREE_QUARTERS:
                low -= QUARTER
                high -= QUARTER
                code -= QUARTER
            else:
                break
            low <<= 1
            high = (high << 1) | 1
            code = (code << 1) | reader.read()
    return restored
