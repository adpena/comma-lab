"""RLC1: inherited adaptive mixer, counted geometry configuration, Q16 native map.

Rider config contains format version, forty fitted weights and nineteen bytes
of geometry parameters. Online probability arithmetic is inherited from TC3;
only geometry becomes integer. Same-host/thread twin proof remains mandatory.
"""

from __future__ import annotations

import numpy as np

from experiments import ddm_tc1_mixer_codec as base
from experiments.ddm_rlc1_geometry import FORMAT, LaneGeometry, parse_config

K, H, W, BINS = 5, 384, 512, 9
TOTAL, Q = base.TOTAL, base.Q
MAGIC = b"RLC1"


def unpack_rider(payload):
    length = 1 + 40 + FORMAT.size
    if len(payload) <= 4 + length or payload[:4] != MAGIC or payload[4] not in (1, 17):
        raise ValueError("invalid RLC1 counted rider")
    config, stream = bytes([payload[4] & 15]) + payload[5 : 4 + length], payload[4 + length :]
    parse_config(config[41:])
    if payload[4] & 16:
        import brotli

        stream = brotli.decompress(stream)
    if not stream:
        raise ValueError("empty RLC1 stream")
    return config, stream


class LaneMixer:
    """Frame-frozen online counts plus a strictly group-causal geometry map."""

    def __init__(self, config):
        if len(config) != 41 + FORMAT.size or config[0] != 1:
            raise ValueError("TC3 needs a variant and exactly 40 counted int8s")
        self.config = bytes(config)
        self.variant = config[0]
        parse_config(config[41:])
        raw = np.frombuffer(config[1:41], dtype=np.int8).copy()
        self.old = base.SharedMixer(raw[:35].tobytes())
        self.weights = raw[35:].reshape(K, 1)
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
            self.geometry = LaneGeometry(previous, self.config[41:])
        original = self.old.coding(rows, positions, plane, previous)
        freq = base.frequencies(original)
        phi = np.empty((len(rows), K, 1), dtype=np.int16)
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
