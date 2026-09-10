"""Five generic group-causal maps extending the shipped TC3 A mixer.

No learned map tables: previous means the previous decoded pair's token plane.
The counted TC4 rider is magic4, selection-mask1, old weights40, new int8 weights.
"""

from __future__ import annotations

import numpy as np

from experiments import ddm_tc1_mixer_codec as base
from experiments.ddm_tc3_mixer import LaneMixer

H, W, K = 384, 512, 5
Y, X = np.indices((H, W))
ALL = np.arange(H * W)
LEVELS = (41, 26, 26, 10, 18)
MAGIC = b"TC4M"


def previous_map(previous):
    if previous is None:
        return np.full((H, W), 25, dtype=np.uint8)
    pad = np.pad(previous, 1, constant_values=K)
    counts = np.zeros((H, W, K), dtype=np.uint8)
    for dy in range(3):
        for dx in range(3):
            values = pad[dy : dy + H, dx : dx + W]
            counts += values[:, :, None] == np.arange(K)
    # Ties choose the smallest class; outside-image cells cast no vote.
    return (previous * K + counts.argmax(axis=2)).astype(np.uint8)


def row_runs(plane):
    known = plane < K
    reset = (~known) | (X % 64 == 0)
    reset[:, 1:] |= plane[:, 1:] != plane[:, :-1]
    start = np.maximum.accumulate(np.where(reset, X, 0), axis=1)
    length = np.where(known, X - start + 1, 0)
    left = np.full((H, W), K, dtype=np.uint8)
    left[:, 1:] = plane[:, :-1]
    run = np.zeros((H, W), dtype=np.int64)
    run[:, 1:] = length[:, :-1]
    available = (X % 64 != 0) & (left < K)
    bucket = np.searchsorted([1, 2, 4, 8, 16, 32, 63], run, side="left")
    return np.where(available, left * 8 + bucket, 40).astype(np.uint8), run, left, available


def vertical_map(plane, query_rows=None):
    transition = np.zeros((H, W), dtype=bool)
    transition[1:] = (plane[1:] < K) & (plane[:-1] < K) & (plane[1:] != plane[:-1])
    if query_rows is not None:
        # At query y, same-column symbols are decoded iff y' % 64 < y % 64.
        transition &= (Y % 64 > 0) & (query_rows > Y % 64)
    last = np.maximum.accumulate(np.where(transition, Y, -1), axis=0)
    above = np.full_like(last, -1)
    above[1:] = last[:-1]
    lo = np.maximum(above, 1)
    value = plane[lo - 1, X].astype(np.int64) * K + plane[lo, X]
    return np.where(above >= 1, value, 25).astype(np.uint8)


def context_maps(plane, temporal, lane_bins, positions=ALL, *, complete=False):
    """Online prefix path or mathematically equivalent complete-field encode path.

    complete=True is encoder-only: each map masks the exact HPAC availability.
    The receiver uses complete=False and an internally observed sentinel plane.
    Map4 reuses the bin predicted at the above cell's own earlier decode event.
    """
    plane = np.asarray(plane, dtype=np.uint8).reshape(H, W)
    run_map, run, left, available = row_runs(plane)
    if complete:
        vertical = np.full((H, W), 25, dtype=np.uint8)
        for remainder in range(1, 64):
            value = vertical_map(plane, remainder)
            vertical[remainder::64] = value[remainder::64]
    else:
        vertical = vertical_map(plane)
    above_bins = np.full((H, W), 9, dtype=np.uint8)
    above_bins[1:] = lane_bins[:-1]
    above_bins[Y % 64 == 0] = 9
    state = np.zeros((H, W), dtype=np.uint8)
    state[~available] = 8
    movable = available & (left == 3)
    state[movable] = 1 + np.searchsorted([1, 2, 4, 8, 16, 32], run[movable], side="left")
    horizon = ((Y >= 128) & (Y < 320)).astype(np.uint8) * 9 + state
    return np.stack([v.reshape(-1)[positions] for v in (run_map, vertical, temporal, above_bins, horizon)], axis=1)


def mix_rows(freq, phi, weights, original):
    result = base.mix_probabilities(freq, phi, weights, original)
    inactive = ~np.any(phi, axis=(1, 2))
    result[inactive] = original[inactive]
    return result


def unpack_rider(payload):
    if len(payload) < 46 or payload[:4] != MAGIC or not 0 < payload[4] < 32:
        raise ValueError("invalid TC4 rider")
    mask = payload[4]
    count = mask.bit_count()
    end = 45 + K * count
    if len(payload) <= end:
        raise ValueError("truncated TC4 rider")
    return payload[4:end], payload[end:]


class ContextMixer:
    """Fixed old40 weights, frame-frozen KT features, counted selected new weights."""

    def __init__(self, config):
        self.config = bytes(config)
        self.mask = config[0]
        self.selected = [j for j in range(5) if self.mask & (1 << j)]
        if self.mask >= 32 or len(config) != 41 + K * len(self.selected):
            raise ValueError("TC4 config shape mismatch")
        self.old = LaneMixer(bytes([1]) + config[1:41])
        self.weights = np.frombuffer(config[41:], dtype=np.int8).reshape(K, len(self.selected)).copy()
        self.counts = [np.zeros((K * levels, K), dtype=np.int64) for levels in LEVELS]
        self.expected = [np.zeros_like(v) for v in self.counts]
        self.tables = None
        self.codes = np.zeros((H * W, 5), dtype=np.uint8)
        self.freq = np.zeros((H * W, K), dtype=np.int64)
        self.lane_bins = np.full((H, W), 9, dtype=np.uint8)
        self.temporal = None

    @property
    def frame(self):
        return self.old.frame

    def begin_frame(self):
        self.old.begin_frame()
        self.tables = [
            base.log2_fixed(np.clip((c + 0.5) / (e / base.TOTAL + 0.5), 1 / 16, 16))
            for c, e in zip(self.counts, self.expected, strict=True)
        ]
        self.lane_bins.fill(9)
        self.temporal = None

    def features_from_codes(self, rows, codes):
        freq = base.frequencies(rows)
        winner = freq.argmax(axis=1)
        phi = np.stack([self.tables[j][winner * LEVELS[j] + codes[:, j]] for j in range(5)], axis=2)
        return phi, freq

    def coding(self, rows, positions, plane, previous):
        original = self.old.coding(rows, positions, plane, previous)
        if self.temporal is None:
            self.temporal = previous_map(previous)
        codes = context_maps(self.old.geometry.plane, self.temporal, self.lane_bins, positions)
        phi, freq = self.features_from_codes(original, codes)
        self.codes[positions], self.freq[positions] = codes, freq
        self.lane_bins.reshape(-1)[positions] = self.old.bins[positions]
        return mix_rows(freq, phi[:, :, self.selected], self.weights, original)

    def observe(self, positions, symbols):
        self.old.observe(positions, symbols)

    def update_counts(self, truth, codes, freq):
        winner = freq.argmax(axis=1)
        for j, levels in enumerate(LEVELS):
            code = winner * levels + codes[:, j]
            self.counts[j] += np.bincount(code * K + truth, minlength=levels * K * K).reshape(-1, K)
            for k in range(K):
                self.expected[j][:, k] += np.bincount(code, weights=freq[:, k], minlength=levels * K).astype(np.int64)

    def end_frame(self, plane, previous):
        self.update_counts(np.asarray(plane).reshape(-1), self.codes, self.freq)
        self.old.end_frame(plane, previous)
        self.tables = None

    def snapshot(self):
        if self.tables is not None:
            raise ValueError("checkpoint requires end_frame")
        result = {"old_" + k: v for k, v in self.old.snapshot().items()}
        result["config"] = np.frombuffer(self.config, dtype=np.uint8).copy()
        for j in range(5):
            result[f"counts{j}"], result[f"expected{j}"] = self.counts[j].copy(), self.expected[j].copy()
        return result

    def restore(self, state):
        if state["config"].tobytes() != self.config:
            raise ValueError("TC4 checkpoint config drift")
        self.old.restore({k[4:]: v for k, v in state.items() if k.startswith("old_")})
        self.counts = [state[f"counts{j}"].copy() for j in range(5)]
        self.expected = [state[f"expected{j}"].copy() for j in range(5)]
        self.tables = None
