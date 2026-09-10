"""TC3 counted lane-map mixers; generic causal receiver code, no fitted tables.

Variant A applies TC2's five weights after the unchanged 35-weight TC1 mixer.
Variant B jointly calibrates all 40 weights against the pre-TC1 probability row.
The first config byte selects A=1 or B=2; the following 40 int8s are counted.
"""

from __future__ import annotations

import numpy as np

from experiments import ddm_tc1_mixer_codec as base
from experiments.ddm_tc3_geometry import LaneGeometry, RunTrackingGeometry

K, H, W, BINS = 5, 384, 512, 9
TOTAL, Q = base.TOTAL, base.Q
MAGIC = b"TC3M"


def unpack_rider(payload):
    if len(payload) <= 45 or payload[:4] != MAGIC or payload[4] not in (1, 2, 17, 18):
        raise ValueError("invalid TC3 rider")
    version = payload[4]
    weights, stream = payload[5:45], payload[45:]
    if version & 16:
        import brotli

        stream = brotli.decompress(stream)
    if not stream:
        raise ValueError("empty TC3 stream")
    return bytes([version & 15]) + weights, stream


class LaneMixer:
    """Frame-frozen online counts plus a strictly group-causal geometry map."""

    def __init__(self, config):
        if len(config) != 41 or config[0] not in (1, 2):
            raise ValueError("TC3 needs a variant and exactly 40 counted int8s")
        self.config = bytes(config)
        self.variant = config[0]
        raw = np.frombuffer(config[1:], dtype=np.int8).copy()
        if self.variant == 1:
            self.old = base.SharedMixer(raw[:35].tobytes())
            self.weights = raw[35:].reshape(K, 1)
        else:
            self.weights = raw.reshape(K, 8)
            self.old = base.SharedMixer(self.weights[:, :7].copy().tobytes())
        self.counts = np.zeros((K * BINS, K), dtype=np.int64)
        self.expected = np.zeros_like(self.counts)
        self.base = np.empty((H * W, K), dtype=np.int64)
        self.bins = np.empty(H * W, dtype=np.uint8)
        self.seen = np.zeros(H * W, dtype=bool)
        self.geometry = None
        self.table = None
        self.pending = None

    @property
    def frame(self):
        return self.old.frame

    def begin_frame(self):
        if self.table is not None:
            raise ValueError("TC3 cannot restart an unfinished frame")
        self.old.begin_frame()
        ratio = (self.counts + 0.5) / (self.expected.astype(np.float64) / TOTAL + 0.5)
        self.table = base.log2_fixed(np.clip(ratio, 1 / 16, 16))
        self.geometry = None
        self.pending = None
        self.seen.fill(False)

    def features(self, rows, positions, plane, previous):
        positions = np.asarray(positions, dtype=np.int64)
        if self.table is None or self.pending is not None or self.seen[positions].any():
            raise ValueError("TC3 begin/coding/observe group sequence violated")
        if self.geometry is None:
            cls = LaneGeometry if self.variant == 1 else RunTrackingGeometry
            self.geometry = cls(previous)
        if self.variant == 1:
            original = self.old.coding(rows, positions, plane, previous)
            freq = base.frequencies(original)
            phi = np.empty((len(rows), K, 1), dtype=np.int16)
        else:
            original = rows
            old_phi, freq = self.old.features(rows, positions, plane, previous)
            phi = np.empty((len(rows), K, 8), dtype=np.int16)
            phi[:, :, :7] = old_phi
        bins = self.geometry.contexts(positions)
        arg = freq.argmax(axis=1)
        extra = self.table[arg * BINS + bins]
        extra[bins == 8] = 0
        phi[:, :, -1] = extra
        self.bins[positions] = bins
        self.base[positions] = freq
        self.seen[positions] = True
        self.pending = positions.copy()
        return phi, freq, original

    def coding(self, rows, positions, plane, previous):
        phi, freq, original = self.features(rows, positions, plane, previous)
        result = base.mix_probabilities(freq, phi, self.weights, original)
        if self.variant == 1:
            inactive = np.all(phi[:, :, 0] == 0, axis=1)
            result[inactive] = original[inactive]
        return result

    def observe(self, positions, symbols):
        if self.pending is None or not np.array_equal(positions, self.pending):
            raise ValueError("TC3 observed a different group")
        self.geometry.observe(positions, symbols)
        self.pending = None

    def end_frame(self, plane, previous):
        if not self.seen.all() or self.pending is not None:
            raise ValueError("incomplete TC3 frame")
        truth = np.asarray(plane, dtype=np.uint8).reshape(-1)
        np.testing.assert_array_equal(self.geometry.plane.reshape(-1), truth)
        code = self.base.argmax(axis=1) * BINS + self.bins
        self.counts += np.bincount(code * K + truth, minlength=K * BINS * K).reshape(-1, K)
        for k in range(K):
            self.expected[:, k] += np.bincount(code, weights=self.base[:, k], minlength=K * BINS).astype(np.int64)
        self.old.end_frame(plane, previous)
        self.table = None
        self.geometry = None

    def snapshot(self):
        if self.table is not None or self.pending is not None:
            raise ValueError("TC3 checkpoints require a complete frame")
        state = {"old_" + k: v for k, v in self.old.snapshot().items()}
        state.update(
            config=np.frombuffer(self.config, dtype=np.uint8).copy(),
            lane_counts=self.counts.copy(),
            lane_expected=self.expected.copy(),
        )
        return state

    def restore(self, state):
        if state["config"].tobytes() != self.config:
            raise ValueError("TC3 checkpoint weights or variant changed")
        self.old.restore({k[4:]: v for k, v in state.items() if k.startswith("old_")})
        self.counts = state["lane_counts"].copy()
        self.expected = state["lane_expected"].copy()
        self.table = None
        self.geometry = None
        self.pending = None
