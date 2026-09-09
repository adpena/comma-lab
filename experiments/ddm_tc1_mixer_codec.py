"""Portable TC1 counted shared mixer; only NumPy and decoded causal state.

The 35 int8 coefficients belong in the counted TC1M rider. All context maps,
fixed-point logarithms, and power tables are generic deterministic decoder code.
"""
from __future__ import annotations

import numpy as np

N, H, W, K, F = 600, 384, 512, 5, 7
TOTAL, Q, SCALE = 1 << 31, 1024, 32
LEVELS = dict(spatial2=36, spatial3=216, previous=6, run=64, rowband=12)
MAGIC = b'TC1M\x01'
Y, X = np.indices((H, W))
GROUP = (X % 64 + 2 * (Y % 64)).reshape(-1)
ALL = np.arange(H * W)


def frequencies(rows):
    values = np.asarray(rows, dtype=np.float32).astype(np.float64)
    if values.ndim != 2 or values.shape[1] != K or not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError('invalid TC1 source probabilities')
    freq = np.maximum((values * TOTAL).astype(np.int64), 1)
    winner = values.argmax(axis=1)
    freq[np.arange(len(freq)), winner] += TOTAL - freq.sum(axis=1)
    if np.any(freq <= 0) or np.any(freq >= TOTAL):
        raise ValueError('invalid TC1 frequency balance')
    return freq


def log2_fixed(value):
    value = np.asarray(value, dtype=np.float64)
    if np.any(value <= 0):
        raise ValueError('positive logarithm input required')
    mantissa, exponent = np.frexp(value)
    work = mantissa * 2.0
    code = (exponent.astype(np.int64) - 1) * (Q * 2)
    fraction = np.zeros(value.shape, dtype=np.int64)
    for _ in range(11):
        work = work * work
        bit = work >= 2.0
        work = np.where(bit, work * .5, work)
        fraction = fraction * 2 + bit
    result = (code + fraction + 1) // 2
    if np.any(result < -32768) or np.any(result > 32767):
        raise ValueError('Q10 logarithm overflow')
    return result.astype(np.int16)


def _neighbour_map(dy, dx):
    yy, xx = Y + dy, X + dx
    valid = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
    source = (np.clip(yy, 0, H - 1) * W + np.clip(xx, 0, W - 1)).reshape(-1)
    valid = valid.reshape(-1) & (GROUP[source] < GROUP)
    return source, valid


NEIGHBOURS = {offset: _neighbour_map(*offset) for offset in [(0, -i) for i in range(1, 8)] + [(-1, 0), (-1, 1)]}


def contexts(plane, previous, run, positions=ALL):
    flat = np.asarray(plane).reshape(-1)
    positions = np.asarray(positions, dtype=np.int64)
    def neighbour(offset):
        source, valid = NEIGHBOURS[offset]
        return np.where(valid[positions], flat[source[positions]], K).astype(np.int64)
    left, up, upright = neighbour((0, -1)), neighbour((-1, 0)), neighbour((-1, 1))
    spatial_run = np.zeros(len(positions), dtype=np.int64)
    active = left != K
    for distance in range(1, 8):
        value = neighbour((0, -distance))
        active &= (value != K) & (value == left)
        spatial_run += active
    coloc = (np.full(len(positions), K, dtype=np.int64) if previous is None else
             np.asarray(previous).reshape(-1)[positions].astype(np.int64))
    return dict(spatial2=left * 6 + up, spatial3=(left * 6 + up) * 6 + upright,
                previous=coloc, run=spatial_run * 8 + np.minimum(run.reshape(-1)[positions], 7),
                rowband=(positions // W) // 32)


def _power_table():
    codes = np.arange(Q * SCALE, dtype=np.int64)
    values = np.ones(len(codes), dtype=np.float64)
    radical = 2.0
    for bit in range(14, -1, -1):
        radical = float(np.sqrt(radical))
        values *= np.where((codes >> bit) & 1, radical, 1.0)
    return values


POW2 = _power_table()


def mix_probabilities(freq, phi, weights, original_rows=None):
    winner = freq.argmax(axis=1)
    exponent = np.sum(phi.astype(np.int64) * weights[winner, None, :].astype(np.int64), axis=2)
    exponent -= exponent.max(axis=1, keepdims=True)
    integer, fraction = np.divmod(exponent, Q * SCALE)
    factor = np.ldexp(POW2[fraction], integer.astype(np.int32))
    raw = freq.astype(np.float64) / TOTAL * factor
    raw /= raw.sum(axis=1, keepdims=True)
    # Keep the native C decoder's positive-input invariant even under extreme
    # allowed weights; RC64 itself applies its minimum frequency of one.
    output = np.maximum(raw, np.finfo(np.float32).tiny).astype(np.float32)
    inactive = ~np.any(weights[winner], axis=1)
    if inactive.any():
        if original_rows is None:
            raise ValueError('zero-weight banks require original HPAC float32 rows')
        output[inactive] = original_rows[inactive]
    return output


class SharedMixer:
    def __init__(self, weight_bytes):
        if len(weight_bytes) != K * F:
            raise ValueError('TC1 requires exactly 35 counted int8 weights')
        self.weights = np.frombuffer(weight_bytes, dtype=np.int8).reshape(K, F).copy()
        self.frame = 0
        self.counts = {name: np.zeros((K * levels, K), dtype=np.int64) for name, levels in LEVELS.items()}
        self.expected = {name: np.zeros((K * levels, K), dtype=np.int64) for name, levels in LEVELS.items()}
        self.run = np.zeros((H, W), dtype=np.uint8)
        self.base = np.empty((H * W, K), dtype=np.int64)
        self.seen = np.zeros(H * W, dtype=bool)
        self.tables = None

    def begin_frame(self):
        self.seen.fill(False)
        self.tables = {}
        for name in LEVELS:
            ratio = (self.counts[name].astype(np.float64) + .5) / (self.expected[name].astype(np.float64) / TOTAL + .5)
            self.tables[name] = log2_fixed(np.clip(ratio, 1 / 16, 16))

    def features(self, rows, positions, plane, previous):
        if self.tables is None:
            raise ValueError('begin_frame must precede coding')
        positions = np.asarray(positions, dtype=np.int64)
        freq = frequencies(rows)
        arg = freq.argmax(axis=1)
        context = contexts(plane, previous if self.frame else None, self.run, positions)
        phi = np.zeros((len(rows), K, F), dtype=np.int16)
        for j, (name, levels) in enumerate(LEVELS.items()):
            phi[:, :, j] = self.tables[name][arg * levels + context[name]]
        phi[:, :, 5] = log2_fixed(freq.astype(np.float64) / TOTAL)
        phi[np.arange(len(rows)), arg, 6] = Q
        self.base[positions] = freq
        self.seen[positions] = True
        return phi, freq

    def coding(self, rows, positions, plane, previous):
        phi, freq = self.features(rows, positions, plane, previous)
        return mix_probabilities(freq, phi, self.weights, rows)

    def end_frame(self, plane, previous):
        if not self.seen.all():
            raise ValueError('incomplete TC1 base-frequency plane')
        truth = np.asarray(plane, dtype=np.uint8).reshape(-1)
        if np.any(truth >= K):
            raise ValueError('decoded symbol outside TC1 alphabet')
        context = contexts(plane, previous if self.frame else None, self.run)
        arg = self.base.argmax(axis=1)
        for name, levels in LEVELS.items():
            code = arg * levels + context[name]
            self.counts[name] += np.bincount(code * K + truth, minlength=levels * K * K).reshape(levels * K, K)
            for k in range(K):
                self.expected[name][:, k] += np.bincount(code, weights=self.base[:, k], minlength=levels * K).astype(np.int64)
        self.run = (np.zeros((H, W), dtype=np.uint8) if not self.frame else
                    np.where(np.asarray(plane).reshape(H, W) == np.asarray(previous).reshape(H, W),
                             np.minimum(self.run + 1, 7), 0).astype(np.uint8))
        self.frame += 1
        self.tables = None

    def snapshot(self):
        state = dict(frame=np.array([self.frame]), run=self.run.copy(), weights=self.weights.copy())
        state.update({'counts_' + k: v.copy() for k, v in self.counts.items()})
        state.update({'expected_' + k: v.copy() for k, v in self.expected.items()})
        return state

    def restore(self, state):
        if not np.array_equal(state['weights'], self.weights):
            raise ValueError('TC1 restart changed counted weights')
        self.frame = int(state['frame'][0])
        self.run = state['run'].copy()
        self.counts = {k: state['counts_' + k].copy() for k in LEVELS}
        self.expected = {k: state['expected_' + k].copy() for k in LEVELS}
        self.tables = None


def unpack_rider(payload):
    if payload[:4] != MAGIC[:4] or payload[4:5] not in (b'\x01', b'\x11') or len(payload) <= len(MAGIC) + K * F:
        raise ValueError('invalid or truncated TC1M rider')
    end = len(MAGIC) + K * F
    stream = payload[end:]
    if payload[4] == 0x11:
        import brotli
        stream = brotli.decompress(stream)
    return payload[len(MAGIC):end], stream
