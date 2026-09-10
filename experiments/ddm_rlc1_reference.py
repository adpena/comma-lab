"""Independent big-integer oracle for the counted RLC1 moment geometry.

Q16 is a predeclared generic arithmetic convention, not a fitted operand.
This reference emits context bins only; it is not a payload encoder or scorer.
"""

from __future__ import annotations

import bisect
import math
import struct
from dataclasses import dataclass

import numpy as np

H, W, K, SLOT_WIDTH = 384, 512, 5, 64
SLOTS = W // SLOT_WIDTH
Q = 1 << 16
CONFIG_STRUCT = struct.Struct("<BHH8B6B")
Y, X = np.indices((H, W), dtype=np.int64)
GROUP = X % SLOT_WIDTH + 2 * (Y % SLOT_WIDTH)
POSITIONS = tuple(np.flatnonzero(GROUP.ravel() == g) for g in range(3 * SLOT_WIDTH - 2))
VALUES = (np.ones((H, W), dtype=np.int64), Y, Y * Y, X, X * Y, X * X)


@dataclass(frozen=True)
class Config:
    """All uncertain/content-selected operands arrive in the counted bytes."""

    class_id: int
    row_start: int
    row_end: int
    window: int
    prior_den: int
    min_count: int
    residual_max: int
    slope_max: int
    width_den: int
    width_mult: int
    wide_max: int
    bins: tuple[int, ...]

    @classmethod
    def from_bytes(cls, data):
        if len(data) != CONFIG_STRUCT.size:
            raise ValueError("RLC1 geometry config must have exactly 19 bytes")
        values = CONFIG_STRUCT.unpack(data)
        result = cls(*values[:11], tuple(values[11:]))
        result.validate()
        return result

    def validate(self):
        # These supported ranges are integer-width safety constraints, not defaults.
        if not 0 <= self.class_id < K or not 0 <= self.row_start < self.row_end <= H:
            raise ValueError("class or row band outside the public format")
        if not 1 <= self.window <= SLOT_WIDTH or not 1 <= self.prior_den <= 16:
            raise ValueError("window/prior denominator exceeds native integer bounds")
        if not 1 <= self.width_den <= 255 or not 1 <= self.min_count <= 255:
            raise ValueError("zero/invalid width denominator or minimum count")
        if any(not 0 <= v <= 255 for v in (self.residual_max, self.slope_max, self.width_mult, self.wide_max)):
            raise ValueError("threshold outside byte range")
        if len(self.bins) != 6 or any(not 0 <= v <= 255 for v in self.bins):
            raise ValueError("exactly six byte-valued distance edges required")
        if any(a >= b for a, b in zip(self.bins[:-1], self.bins[1:], strict=True)):
            raise ValueError("distance edges must be strictly increasing")


def round_even_q16(value):
    """Nearest integer, ties to even; value is a nonnegative Q16 distance."""
    quotient, remainder = divmod(int(value), Q)
    return quotient + (2 * remainder > Q or (2 * remainder == Q and quotient % 2 != 0))


def fitted_interval(moments, y, config):
    """Exact rational validity gates; floor-Q16 center and square-root width."""
    c, sy, syy, sx, sxy, sxx = (int(v) for v in moments)
    if c < config.min_count * config.prior_den:
        return None
    d = c * syy - sy * sy
    if d <= 0:
        return None
    n = c * sxy - sy * sx
    if abs(n) > config.slope_max * d:
        return None
    a = c * sxx - sx * sx
    r = d * a - n * n
    denominator = c * c * d
    if r > config.residual_max * denominator:
        return None
    center = ((sx * d + n * (int(y) * c - sy)) * Q) // (c * d)
    width = math.isqrt((config.width_mult * max(r, 0) * Q * Q) // denominator)
    width = max(Q // config.width_den, width)
    return center - width, center + width


def _window(values, window):
    prefix = np.concatenate((np.zeros_like(values[:, :1]), np.cumsum(values, axis=1, dtype=np.int64)), axis=1)
    rows = np.arange(H)
    return prefix[:, rows] - prefix[:, np.maximum(0, rows - window)]


class ReferenceGeometry:
    """Group-causal moments independently evaluated with Python integer algebra."""

    def __init__(self, previous, config):
        self.config = Config.from_bytes(config) if isinstance(config, bytes) else config
        self.config.validate()
        if previous is not None:
            previous = np.asarray(previous)
            if previous.shape != (H, W) or previous.dtype.kind not in "iu":
                raise ValueError("previous must be a complete integer class plane")
            if np.any(previous < 0) or np.any(previous >= K):
                raise ValueError("previous class outside the public alphabet")
        prior = np.zeros((H, W), dtype=bool) if previous is None else previous == self.config.class_id
        self.old = np.stack([(v * prior).reshape(H, SLOTS, SLOT_WIDTH).sum(axis=2) for v in VALUES])
        self.current = np.zeros_like(self.old)
        self.plane = np.full((H, W), K, dtype=np.uint8)
        prior_slots = prior.reshape(H, SLOTS, SLOT_WIDTH)
        self.old_lo = np.where(prior_slots, X.reshape(H, SLOTS, SLOT_WIDTH), W).min(axis=2)
        self.old_hi = np.where(prior_slots, X.reshape(H, SLOTS, SLOT_WIDTH), -1).max(axis=2)
        starts = prior_slots & ~np.concatenate((np.zeros((H, SLOTS, 1), dtype=bool), prior_slots[:, :, :-1]), axis=2)
        self.ambiguous_prior = starts.sum(axis=2) > 1
        self.lo = np.full((H, SLOTS), W, dtype=np.int64)
        self.hi = np.full((H, SLOTS), -1, dtype=np.int64)
        self.group = 0
        self.pending = False

    def _check_positions(self, positions):
        positions = np.asarray(positions)
        if positions.dtype.kind not in "iu" or self.group >= len(POSITIONS):
            raise ValueError("invalid group positions")
        if not np.array_equal(positions, POSITIONS[self.group]):
            raise ValueError("expected complete sorted next HPAC group")
        return positions

    def contexts(self, positions):
        positions = self._check_positions(positions)
        if self.pending:
            raise ValueError("observe required before next group")
        self.pending = True
        config = self.config
        moments = _window(self.current * config.prior_den + self.old, config.window)
        known = self.plane.reshape(H, SLOTS, SLOT_WIDTH)
        xx = X.reshape(H, SLOTS, SLOT_WIDTH)
        gap = ((known < K) & (known != config.class_id) & (xx > self.lo[:, :, None]) & (xx < self.hi[:, :, None])).any(axis=2)
        wide = np.maximum(self.hi, self.old_hi) - np.minimum(self.lo, self.old_lo) > config.wide_max
        bad = _window((gap | wide | self.ambiguous_prior)[None].astype(np.int64), config.window)[0] != 0
        yy, xx = positions // W, positions % W
        result = np.full(len(positions), 8, dtype=np.uint8)
        for y in np.unique(yy):
            if not config.row_start <= y < config.row_end:
                continue
            edges = []
            for slot in range(SLOTS):
                if not bad[y, slot]:
                    interval = fitted_interval(moments[:, y, slot], y, config)
                    if interval is not None:
                        edges.extend(interval)
            for index in np.flatnonzero(yy == y):
                if edges:
                    distance = min(abs(int(xx[index]) * Q - edge) for edge in edges)
                    result[index] = bisect.bisect_left(config.bins, round_even_q16(distance))
                else:
                    result[index] = 7
        return result

    def observe(self, positions, symbols):
        positions = self._check_positions(positions)
        symbols = np.asarray(symbols)
        if not self.pending or symbols.shape != positions.shape or symbols.dtype.kind not in "iu":
            raise ValueError("contexts and integer symbols for this group required")
        if np.any(symbols < 0) or np.any(symbols >= K):
            raise ValueError("decoded symbol outside alphabet")
        self.plane.ravel()[positions] = symbols
        chosen = positions[symbols == self.config.class_id]
        codes = (chosen // W) * SLOTS + (chosen % W) // SLOT_WIDTH
        for j, values in enumerate(VALUES):
            np.add.at(self.current[j].ravel(), codes, values.ravel()[chosen])
        np.minimum.at(self.lo.ravel(), codes, chosen % W)
        np.maximum.at(self.hi.ravel(), codes, chosen % W)
        self.group += 1
        self.pending = False
