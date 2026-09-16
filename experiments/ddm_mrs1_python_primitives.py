"""Python inverses for the submission's arithmetic stream and causal geometry.

These implement the sealed receiver's algorithms without native libraries.
Range arithmetic and geometry products use Python integers where 64 bits do
not suffice. All geometry parameters come from the archive's counted bytes.
"""

from __future__ import annotations

import math
import struct

import numpy as np


class ArithmeticDecoder:
    """Five-symbol, 63-bit arithmetic decoder with a 31-bit frequency total."""

    TOTAL = 1 << 31
    QUARTER = 1 << 61
    HALF = 1 << 62
    TOP = (1 << 63) - 1

    def __init__(self, payload: bytes):
        if not payload:
            raise ValueError("empty arithmetic stream")
        self.payload = bytes(payload)
        self.bit_position = 0
        self.low, self.high, self.code = 0, self.TOP, 0
        for _ in range(63):
            self.code = (self.code << 1) | self._read_bit()

    def _read_bit(self):
        byte, shift = divmod(self.bit_position, 8)
        self.bit_position += 1
        return (self.payload[byte] >> (7 - shift)) & 1 if byte < len(self.payload) else 0

    def decode(self, probabilities):
        """Decode one causal group; preserve the original float32 conversion."""
        rows = np.asarray(probabilities, dtype=np.float32)
        if rows.ndim != 2 or rows.shape[1] != 5 or not len(rows):
            raise ValueError("probabilities must have shape [N, 5]")
        wide = rows.astype(np.float64)
        if not np.isfinite(wide).all() or np.any(wide <= 0) or np.any(wide > 1.00002):
            raise ValueError("invalid arithmetic probability")
        # The native decoder accumulates its five lanes from left to right.
        totals = np.zeros(len(rows), dtype=np.float64)
        for lane in range(5):
            totals += wide[:, lane]
        if np.any(totals < 0.99998) or np.any(totals > 1.00002):
            raise ValueError("probabilities do not sum to one")
        frequencies = np.maximum((wide * self.TOTAL).astype(np.int64), 1)
        winners = rows.argmax(axis=1)
        frequencies[np.arange(len(rows)), winners] += self.TOTAL - frequencies.sum(axis=1)
        if np.any(frequencies <= 0) or np.any(frequencies >= self.TOTAL):
            raise ValueError("invalid arithmetic frequencies")
        result = np.empty(len(rows), dtype=np.int32)
        low, high, code = self.low, self.high, self.code
        for index, row in enumerate(frequencies.tolist()):
            if not low <= code <= high:
                raise ValueError("arithmetic state outside interval")
            width = high - low + 1
            scaled = ((code - low + 1) * self.TOTAL - 1) // width
            before, after = 0, 0
            for symbol, frequency in enumerate(row):
                after += frequency
                if scaled < after:
                    break
                before = after
            else:
                raise ValueError("arithmetic symbol outside alphabet")
            lower, upper = (width * before) >> 31, (width * after) >> 31
            if upper <= lower:
                raise ValueError("empty arithmetic interval")
            high, low = low + upper - 1, low + lower
            while True:
                if high < self.HALF:
                    offset = 0
                elif low >= self.HALF:
                    offset = self.HALF
                elif low >= self.QUARTER and high < 3 * self.QUARTER:
                    offset = self.QUARTER
                else:
                    break
                low = (low - offset) << 1
                high = ((high - offset) << 1) | 1
                code = ((code - offset) << 1) | self._read_bit()
            result[index] = symbol
        self.low, self.high, self.code = low, high, code
        return result


HEIGHT, WIDTH, CLASSES, SLOT_WIDTH = 384, 512, 5, 64
SLOTS, FIXED_ONE = WIDTH // SLOT_WIDTH, 1 << 16
GEOMETRY_FORMAT = struct.Struct("<BHH8B6B")
ROW, COLUMN = np.indices((HEIGHT, WIDTH), dtype=np.int64)
GROUP_INDEX = COLUMN % SLOT_WIDTH + 2 * (ROW % SLOT_WIDTH)
GROUP_POSITIONS = tuple(np.flatnonzero(GROUP_INDEX.ravel() == group) for group in range(190))
MOMENTS = (np.ones_like(ROW), ROW, ROW * ROW, COLUMN, COLUMN * ROW, COLUMN * COLUMN)


def parse_geometry_config(payload):
    """Validate the sealed receiver's exact supported counted parameter domain."""
    if len(payload) != GEOMETRY_FORMAT.size:
        raise ValueError("geometry configuration must contain 19 bytes")
    values = GEOMETRY_FORMAT.unpack(payload)
    if not (values[0] < CLASSES and 0 <= values[1] < values[2] <= HEIGHT):
        raise ValueError("invalid geometry class or row band")
    if not (1 <= values[3] <= 16 and 1 <= values[4] <= 4
            and 1 <= values[5] <= 255 and 1 <= values[6] <= 255
            and 1 <= values[7] <= 2 and 1 <= values[8] <= 255
            and 1 <= values[9] <= 255 and values[10] < 64):
        raise ValueError("geometry configuration exceeds the supported domain")
    if any(a >= b for a, b in zip(values[11:-1], values[12:])):
        raise ValueError("geometry distance edges must increase")
    return values


def preceding_window(values, window):
    """Sum the preceding rows, excluding the current row, exactly as the C loop."""
    prefix = np.concatenate((np.zeros_like(values[:, :1]), np.cumsum(values, axis=1)), axis=1)
    rows = np.arange(HEIGHT)
    return prefix[:, rows] - prefix[:, np.maximum(0, rows - window)]


class CausalGeometry:
    """Integer local line fits from completed groups and the preceding frame."""

    def __init__(self, previous, config):
        self.config = parse_geometry_config(config)
        if previous is not None:
            previous = np.asarray(previous)
            if (previous.shape != (HEIGHT, WIDTH) or previous.dtype.kind not in "iu"
                    or np.any(previous >= CLASSES) or np.any(previous < 0)):
                raise ValueError("previous must be a complete class plane")
        prior = np.zeros((HEIGHT, WIDTH), bool) if previous is None else previous == self.config[0]
        self.old = np.stack([(value * prior).reshape(HEIGHT, SLOTS, SLOT_WIDTH).sum(axis=2)
                             for value in MOMENTS])
        self.current = np.zeros_like(self.old)
        prior_slots = prior.reshape(HEIGHT, SLOTS, SLOT_WIDTH)
        x = COLUMN.reshape(HEIGHT, SLOTS, SLOT_WIDTH)
        self.old_low = np.where(prior_slots, x, WIDTH).min(axis=2)
        self.old_high = np.where(prior_slots, x, -1).max(axis=2)
        preceding = np.concatenate((np.zeros((HEIGHT, SLOTS, 1), bool), prior_slots[:, :, :-1]), axis=2)
        self.ambiguous = (prior_slots & ~preceding).sum(axis=2) > 1
        self.low = np.full((HEIGHT, SLOTS), WIDTH, dtype=np.int64)
        self.high = np.full((HEIGHT, SLOTS), -1, dtype=np.int64)
        self.other = np.zeros((HEIGHT, SLOTS), dtype=np.uint64)
        self.plane = np.full((HEIGHT, WIDTH), CLASSES, dtype=np.uint8)
        self.group, self.pending = 0, False

    def _positions(self, positions):
        positions = np.asarray(positions)
        if (positions.dtype.kind not in "iu" or self.group >= len(GROUP_POSITIONS)
                or not np.array_equal(positions, GROUP_POSITIONS[self.group])):
            raise ValueError("expected the complete next causal group")
        return positions.astype(np.int64, copy=False)

    def contexts(self, positions):
        positions = self._positions(positions)
        if self.pending:
            raise ValueError("observe must follow the preceding contexts call")
        cfg = self.config
        moments = preceding_window(self.current * cfg[4] + self.old, cfg[3])
        one = np.uint64(1)
        upper = (one << (self.high % SLOT_WIDTH).astype(np.uint64)) - one
        lower = (one << ((self.low % SLOT_WIDTH) + 1).astype(np.uint64)) - one
        between = np.where(self.high > self.low + 1, upper ^ lower, np.uint64(0))
        gap = (self.other & between) != 0
        wide = np.maximum(self.high, self.old_high) - np.minimum(self.low, self.old_low) > cfg[10]
        bad = preceding_window((gap | wide | self.ambiguous)[None].astype(np.int64), cfg[3])[0] != 0
        count, sy, syy, sx, sxy, sxx = moments
        determinant, slope = count * syy - sy * sy, count * sxy - sy * sx
        valid = ((count >= cfg[5] * cfg[4]) & (determinant > 0) & ~bad
                 & (np.abs(slope) <= cfg[7] * determinant))
        center, radius = np.zeros_like(count), np.zeros_like(count)
        for y, slot in zip(*np.nonzero(valid)):
            c, sum_y, sum_yy, sum_x, sum_xy, sum_xx = map(int, moments[:, y, slot])
            d, b = c * sum_yy - sum_y * sum_y, c * sum_xy - sum_y * sum_x
            residual = d * (c * sum_xx - sum_x * sum_x) - b * b
            denominator = c * c * d
            if residual > cfg[6] * denominator:
                valid[y, slot] = False
                continue
            center[y, slot] = ((sum_x * d + b * (int(y) * c - sum_y)) * FIXED_ONE) // (c * d)
            radius[y, slot] = max(FIXED_ONE // cfg[8], math.isqrt(max(residual, 0) * cfg[9] * FIXED_ONE * FIXED_ONE // denominator))
        y, x = positions // WIDTH, positions % WIDTH
        distance = np.minimum(np.abs(x[:, None] * FIXED_ONE - center[y] + radius[y]),
                              np.abs(x[:, None] * FIXED_ONE - center[y] - radius[y]))
        quotient, remainder = np.divmod(distance, FIXED_ONE)
        rounded = quotient + ((remainder > FIXED_ONE // 2)
                              | ((remainder == FIXED_ONE // 2) & ((quotient & 1) != 0)))
        best = np.where(valid[y], rounded, np.iinfo(np.int64).max).min(axis=1)
        result = np.searchsorted(cfg[11:], best, side="left").astype(np.uint8)
        result[~valid[y].any(axis=1)] = 7
        result[(y < cfg[1]) | (y >= cfg[2])] = 8
        self.pending = True
        return result

    def observe(self, positions, symbols):
        positions, symbols = self._positions(positions), np.asarray(symbols)
        if (not self.pending or symbols.shape != positions.shape or symbols.dtype.kind not in "iu"
                or np.any(symbols < 0) or np.any(symbols >= CLASSES)):
            raise ValueError("expected the decoded symbols of the pending group")
        self.plane.ravel()[positions] = symbols
        chosen = positions[symbols == self.config[0]]
        cells = (chosen // WIDTH) * SLOTS + (chosen % WIDTH) // SLOT_WIDTH
        for index, values in enumerate(MOMENTS):
            np.add.at(self.current[index].ravel(), cells, values.ravel()[chosen])
        np.minimum.at(self.low.ravel(), cells, chosen % WIDTH)
        np.maximum.at(self.high.ravel(), cells, chosen % WIDTH)
        other = positions[symbols != self.config[0]]
        cells = (other // WIDTH) * SLOTS + (other % WIDTH) // SLOT_WIDTH
        np.bitwise_or.at(self.other.ravel(), cells, np.uint64(1) << (other % SLOT_WIDTH).astype(np.uint64))
        self.group += 1
        self.pending = False
