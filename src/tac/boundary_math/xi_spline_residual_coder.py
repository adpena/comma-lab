# SPDX-License-Identifier: MIT
"""ξ spline-PREDICTOR + LOSSLESS residual coder — converts the MEASURED 2182 B floor into REAL bytes.

BACKGROUND (`.omx/research/ground_frame_chart_build_and_se3_bspline_firing_20260707.md` §3, the
se3_bspline FIRST FIRING on the real n600 ξ): DIRECT spline replacement of the #257 store-nothing
ξ payload is MEASURED DEAD (derive-H probe error 53–703 px mean at every knot count vs the 0.31 px
quantization floor — the ξ sequence carries essential per-pair high-frequency ego jitter). But
spline-as-PREDICTOR + lossless integer residual has a MEASURED order-0 entropy floor of **2182 B
(M=16: 224 B knots + 1958 B residual entropy) vs the 3200 B shipped delta_ar table** — up to
~1 KB / ~0.0007 rate headroom, EXACT on the quantized grid. This module is the REAL coder that
floor was pending.

PIPELINE (encode; deterministic, $0):
  1. ``xi_ref = dequantize_xi(q, scales)`` — predict from the DECODER-VISIBLE table so encoder and
     decoder run the IDENTICAL predictor (closed loop; no drift).
  2. lift the twist SEQUENCE to an SE(3) PATH with a PREPENDED zero twist (carries ξ[0]):
     ``T_0 = I, T_j = T_{j-1}·exp(ξ_lift[j])`` — then SAMPLE ``M`` control poses at evenly spaced
     knot times (bit-parity with ``ego_xi_trajectory.fit_se3_bspline_controls``, pinned by test).
  3. knot payload = ``quantize_xi(log(ctrl))`` on the shipped grid, serialized via the REAL ξ
     payload coder (``serialize_xi_payload``, min of delta_ar/none) — the same accounting the
     firing measured (224 B at M=16).
  4. predict: dequantize knots → exp → cumulative cubic B-spline eval (Sommer 2020) at the P+1
     pair times → increments ``ξ̂[p] = log(T̂_p⁻¹·T̂_{p+1})`` → ``q̂ = round(ξ̂ / scales)``.
  5. residual ``r = q − q̂`` (int64), losslessly coded. THREE schemes are implemented and ALL
     THREE are MEASURED per encode ({zlib-9, Rice/Golomb, raw varint}); the smallest wins and its
     id is recorded in the payload (self-describing).
  6. NO-FAKE self-check: the encoder runs the FULL decode and refuses (raises) unless the decoded
     table is BIT-IDENTICAL to ``q``.

DECODE is numpy+stdlib ONLY (inflate stays numpy-portable): knots → predict (the same fp64
pipeline) → ``q = q̂ + r``. The spline eval + SE(3) exp/log are inlined here from the numpy
reference oracles (``tac.lie._se3_numpy`` is imported — pure numpy; ``tac.lie.se3_bspline`` is NOT
imported because it imports mlx at module top; the inlined eval is bit-parity-PINNED against it by
test). RULE-118: the fit happens OFFLINE at encode; only knots+residuals are COUNTED; the
predictor/decoder algorithm is FREE in inflate.py. No learned parameters anywhere.

DETERMINISM: numpy-fp64 CPU, same authority class as the shipped derive-H
(``xi_pose_coder.homographies_from_xi``) — same seed/config/inputs → same bytes; the encoder's
bit-identity self-check makes any host-side fp divergence FAIL CLOSED rather than decode wrong.
NEVER a score authority — this module only produces bytes; d_pose is measured through the frozen
CPU-torch PoseNet, and the pointer (0.19110) moves only via upstream/evaluate.py.

MEASURED VERDICT on the real n600 ξ (2026-07-07, `xi_residual_coder_curve.json` beside the
measurement script): the spline PREDICTOR is DEAD as a rate lever too — the post-spline residual
is statistically IDENTICAL to the plain first-order q-delta innovation (rice 2664 B vs 2665 B;
the ξ innovation is WHITE ego jitter, so any smooth-trend predictor removes only what a trivial
delta already removes, while its knots cost 224–349 B for nothing). The REAL win the build found
is the ENTROPY STAGE: rice on the SAME deltas the shipped delta_ar coder spends 3200 B on costs
~2700 B (−~0.0003 rate), bit-identical. That winning mode ships here as ``coder="delta_res"``
(delta predictor + best-of-three residual schemes). The 2182 B order-0 "floor" is mostly
UNREALIZABLE model-overhead mirage: the active residual channels have ~9-bit entropy with
near-unique support (~530 distinct values / 600 samples), so a transmitted-PMF coder cannot
approach the empirical order-0 number.

Wired as SELECTABLE modes: ``xi_pose_coder.serialize_xi_payload(..., coder="spline_residual")``
(coder id 2) and ``coder="delta_res"`` (coder id 3). The DEFAULT path (delta_ar) is
byte-identical to the pre-existing coder — asserted by test. If either mode is ever SELECTED for
a shipped archive, the inflate must inline this module's decode half alongside the existing
verbatim copies (bit-exact gate, as for the defaults).
"""
from __future__ import annotations

import struct
import zlib

import numpy as np

from tac.boundary_math.xi_pose_coder import (
    XiPoseCoderError,
    dequantize_xi,
    quantize_xi,
)
from tac.lie import _se3_numpy as _se3

_XI_QLEVELS_KNOTS_DEFAULT = 4096  # the shipped ξ grid (#257 sweet spot) reused for knot logs

# residual-scheme ids (recorded in the payload; self-describing)
SCHEME_VARINT = 0
SCHEME_ZLIB9 = 1
SCHEME_RICE = 2
RESIDUAL_SCHEMES = ("varint", "zlib9", "rice")
_SCHEME_IDS = {"varint": SCHEME_VARINT, "zlib9": SCHEME_ZLIB9, "rice": SCHEME_RICE}
_SCHEME_NAMES = {v: k for k, v in _SCHEME_IDS.items()}


class XiSplineResidualError(XiPoseCoderError):
    """A ξ spline-residual-coder contract violation (fail-closed; NO-FAKE)."""


# --------------------------------------------------------------------------- #
# inlined numpy spline predictor (decode-side; NO mlx import — parity-pinned by test)
# --------------------------------------------------------------------------- #
def _fit_spline_controls(xi_lift: np.ndarray, M: int) -> np.ndarray:
    """Sample ``M`` SE(3) control poses from the cumulative path of ``xi_lift`` (P+1, 6).

    Bit-parity with ``ego_xi_trajectory.fit_se3_bspline_controls`` (pinned by test): absolute pose
    at index ``j`` = ``prod exp(xi_lift[1..j])`` (row 0 dropped by design — the caller prepends a
    zero twist so ξ[0] is carried); controls sampled at ``round(linspace(0, P, M))``."""
    dxi = np.asarray(xi_lift, np.float64)
    n = int(dxi.shape[0])
    if M < 4:
        raise XiSplineResidualError("SE(3) cubic B-spline needs M >= 4 control poses.")
    if M > n:
        raise XiSplineResidualError(f"knots M={M} exceeds the {n} path poses (P+1).")
    T = np.eye(4, dtype=np.float64)
    abs_T = [T.copy()]
    for k in range(1, n):
        T = _se3.compose(T, _se3.exp_se3(dxi[k]))
        abs_T.append(T.copy())
    abs_T = np.stack(abs_T, axis=0)  # (P+1, 4, 4)
    idx = np.linspace(0, n - 1, M).round().astype(int)
    return abs_T[idx]


def _cumulative_basis(u: np.ndarray) -> np.ndarray:
    """Uniform cubic cumulative basis (B_1, B_2, B_3) — verbatim numpy oracle copy."""
    u = np.asarray(u, dtype=float)
    u2 = u * u
    u3 = u2 * u
    b1 = (5.0 + 3.0 * u - 3.0 * u2 + u3) / 6.0
    b2 = (1.0 + 3.0 * u + 3.0 * u2 - 2.0 * u3) / 6.0
    b3 = u3 / 6.0
    return np.stack([b1, b2, b3], axis=-1)


def _spline_segment(controls4: np.ndarray, u) -> np.ndarray:
    """One cumulative-B-spline segment — verbatim numpy oracle copy (se3_bspline_segment_numpy)."""
    controls4 = np.asarray(controls4, dtype=float)
    T0 = controls4[..., 0, :, :]
    T1 = controls4[..., 1, :, :]
    T2 = controls4[..., 2, :, :]
    T3 = controls4[..., 3, :, :]
    O1 = _se3.log_se3(_se3.compose(_se3.inverse(T0), T1))
    O2 = _se3.log_se3(_se3.compose(_se3.inverse(T1), T2))
    O3 = _se3.log_se3(_se3.compose(_se3.inverse(T2), T3))
    B = _cumulative_basis(u)
    T = _se3.compose(T0, _se3.exp_se3(B[..., 0:1] * O1))
    T = _se3.compose(T, _se3.exp_se3(B[..., 1:2] * O2))
    T = _se3.compose(T, _se3.exp_se3(B[..., 2:3] * O3))
    return T


def _spline_eval(controls: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Evaluate at times ``t`` (1-D) — verbatim numpy oracle copy (se3_bspline_eval_numpy)."""
    controls = np.asarray(controls, dtype=float)
    M = controls.shape[-3]
    n_seg = M - 3
    t_arr = np.atleast_1d(np.asarray(t, dtype=float))
    s = np.clip(np.floor(t_arr).astype(int), 0, n_seg - 1)
    u = np.clip(t_arr - s, 0.0, 1.0)
    out = []
    for k in range(t_arr.shape[0]):
        c4 = controls[..., s[k] : s[k] + 4, :, :]
        out.append(_spline_segment(c4, np.asarray(u[k])))
    return np.stack(out, axis=0)


def predict_q_from_knots(
    qk: np.ndarray, sk: np.ndarray, scales: np.ndarray, n_pairs: int
) -> np.ndarray:
    """The shared encoder/decoder predictor: knot payload → predicted quantized table ``q̂``.

    ``(qk int16 (M,6), sk fp32 (6,))`` knots + the ORIGINAL table's per-channel ``scales`` →
    ``q̂ (P,6) int64``. Deterministic numpy-fp64; both sides run THIS function (closed loop)."""
    P = int(n_pairs)
    logs_hat = dequantize_xi(qk, sk)                       # (M, 6) fp64
    M = int(logs_hat.shape[0])
    ctrl_hat = np.stack([_se3.exp_se3(logs_hat[i]) for i in range(M)])
    tt = np.linspace(0.0, float(M - 3), P + 1)
    T_hat = _spline_eval(ctrl_hat, tt)                     # (P+1, 4, 4)
    xi_hat = np.stack(
        [_se3.log_se3(_se3.compose(_se3.inverse(T_hat[p]), T_hat[p + 1])) for p in range(P)]
    )
    sc = np.asarray(scales, dtype=np.float64).reshape(1, -1)
    return np.round(xi_hat / sc).astype(np.int64)


# --------------------------------------------------------------------------- #
# lossless integer residual schemes (all three implemented + measured per encode)
# --------------------------------------------------------------------------- #
def _zigzag(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.int64)
    return ((v << 1) ^ (v >> 63)).astype(np.uint64)


def _unzigzag(u: np.ndarray) -> np.ndarray:
    u = np.asarray(u, dtype=np.uint64)
    return ((u >> np.uint64(1)).astype(np.int64)) ^ -(u & np.uint64(1)).astype(np.int64)


def _varint_encode(res: np.ndarray) -> bytes:
    """Column-major zigzag LEB128 varint stream of the residual matrix (P, D)."""
    out = bytearray()
    for k in range(res.shape[1]):
        for u in _zigzag(res[:, k]).tolist():
            while True:
                b = u & 0x7F
                u >>= 7
                if u:
                    out.append(b | 0x80)
                else:
                    out.append(b)
                    break
    return bytes(out)


def _varint_decode(blob: bytes, P: int, D: int) -> np.ndarray:
    vals = np.empty(P * D, dtype=np.uint64)
    pos = 0
    for i in range(P * D):
        shift = 0
        u = 0
        while True:
            b = blob[pos]
            pos += 1
            u |= (b & 0x7F) << shift
            if not (b & 0x80):
                break
            shift += 7
        vals[i] = u
    if pos != len(blob):
        raise XiSplineResidualError("varint residual stream has trailing bytes (corrupt payload)")
    return _unzigzag(vals).reshape(D, P).T  # column-major layout → (P, D)


class _BitWriter:
    __slots__ = ("acc", "nbits", "out")

    def __init__(self) -> None:
        self.acc = 0
        self.nbits = 0
        self.out = bytearray()

    def write(self, value: int, nbits: int) -> None:
        """MSB-first append of ``nbits`` of ``value``."""
        self.acc = (self.acc << nbits) | (value & ((1 << nbits) - 1))
        self.nbits += nbits
        while self.nbits >= 8:
            self.nbits -= 8
            self.out.append((self.acc >> self.nbits) & 0xFF)
        self.acc &= (1 << self.nbits) - 1

    def getvalue(self) -> bytes:
        if self.nbits:
            return bytes(self.out) + bytes([(self.acc << (8 - self.nbits)) & 0xFF])
        return bytes(self.out)


class _BitReader:
    __slots__ = ("blob", "pos")

    def __init__(self, blob: bytes) -> None:
        self.blob = blob
        self.pos = 0  # bit position

    def read(self, nbits: int) -> int:
        v = 0
        for _ in range(nbits):
            byte = self.blob[self.pos >> 3]
            v = (v << 1) | ((byte >> (7 - (self.pos & 7))) & 1)
            self.pos += 1
        return v

    def read_unary(self) -> int:
        q = 0
        while True:
            byte = self.blob[self.pos >> 3]
            bit = (byte >> (7 - (self.pos & 7))) & 1
            self.pos += 1
            if bit == 0:
                return q
            q += 1


def _rice_best_k(u: np.ndarray) -> int:
    """Per-channel Rice parameter minimizing the exact bit count (k in 0..17)."""
    n = int(u.shape[0])
    best_k, best_bits = 0, None
    for k in range(18):
        bits = int((u >> np.uint64(k)).sum()) + n + n * k  # unary q + terminators + k remainders
        if best_bits is None or bits < best_bits:
            best_k, best_bits = k, bits
    return best_k


def _rice_encode(res: np.ndarray) -> bytes:
    """Per-channel Rice/Golomb: 1 header byte (k) per channel, then the channel's bitstream.

    Value coding: ``q = u >> k`` as unary (q ones then a zero), then the k low bits MSB-first.
    Each channel's stream is byte-aligned (its own writer) so decode can be strictly sequential."""
    out = bytearray()
    for c in range(res.shape[1]):
        u = _zigzag(res[:, c])
        k = _rice_best_k(u)
        w = _BitWriter()
        for val in u.tolist():
            q = val >> k
            # unary: q ones then a 0 terminator
            while q >= 32:
                w.write((1 << 32) - 1, 32)
                q -= 32
            w.write(((1 << q) - 1) << 1, q + 1)
            if k:
                w.write(val & ((1 << k) - 1), k)
        payload = w.getvalue()
        out.append(k)
        out += struct.pack("<I", len(payload))
        out += payload
    return bytes(out)


def _rice_decode(blob: bytes, P: int, D: int) -> np.ndarray:
    res = np.empty((P, D), dtype=np.int64)
    off = 0
    for c in range(D):
        k = blob[off]
        off += 1
        (plen,) = struct.unpack_from("<I", blob, off)
        off += 4
        r = _BitReader(blob[off : off + plen])
        off += plen
        u = np.empty(P, dtype=np.uint64)
        for i in range(P):
            q = r.read_unary()
            rem = r.read(k) if k else 0
            u[i] = (q << k) | rem
        res[:, c] = _unzigzag(u)
    if off != len(blob):
        raise XiSplineResidualError("rice residual stream has trailing bytes (corrupt payload)")
    return res


def _encode_residual(res: np.ndarray, scheme: str) -> bytes:
    if scheme == "varint":
        return _varint_encode(res)
    if scheme == "zlib9":
        return zlib.compress(_varint_encode(res), 9)
    if scheme == "rice":
        return _rice_encode(res)
    raise XiSplineResidualError(f"unknown residual scheme {scheme!r} (known: {RESIDUAL_SCHEMES})")


def _decode_residual(blob: bytes, scheme_id: int, P: int, D: int) -> np.ndarray:
    if scheme_id == SCHEME_VARINT:
        return _varint_decode(blob, P, D)
    if scheme_id == SCHEME_ZLIB9:
        return _varint_decode(zlib.decompress(blob), P, D)
    if scheme_id == SCHEME_RICE:
        return _rice_decode(blob, P, D)
    raise XiSplineResidualError(f"unknown residual scheme id {scheme_id}")


def measure_residual_schemes(res: np.ndarray) -> dict[str, int]:
    """MEASURED real bytes of the residual matrix under ALL THREE schemes (the report surface)."""
    return {s: len(_encode_residual(res, s)) for s in RESIDUAL_SCHEMES}


# --------------------------------------------------------------------------- #
# the spline_residual payload BODY (nested inside the XIP2 container, coder id 2)
# --------------------------------------------------------------------------- #
def encode_spline_residual_body(
    q: np.ndarray,
    scales: np.ndarray,
    *,
    knots: int = 16,
    q_levels_knots: int = _XI_QLEVELS_KNOTS_DEFAULT,
    scheme: str | None = None,
) -> bytes:
    """Encode the quantized table ``(q int16 (P,D), scales fp32 (D,))`` as knots + residual.

    ``scheme=None`` (default) MEASURES all three residual schemes and ships the smallest.
    Body layout: ``<B scheme_id> <H M> <I klen> kblob <I rlen> rblob`` where ``kblob`` is a full
    nested ξ payload (``serialize_xi_payload`` of the quantized knot logs, min of delta_ar/none).
    NO-FAKE: runs the full decode and refuses unless it is BIT-IDENTICAL to ``q``."""
    from tac.boundary_math.xi_pose_coder import serialize_xi_payload  # lazy: avoids import cycle

    q = np.asarray(q, dtype=np.int16)
    if q.ndim != 2 or q.shape[1] != 6:
        raise XiSplineResidualError(f"q must be (P,6); got {q.shape}")
    P = int(q.shape[0])
    scales = np.asarray(scales, dtype=np.float32).reshape(-1)
    if scales.shape[0] != q.shape[1]:
        raise XiSplineResidualError(f"scales must be ({q.shape[1]},); got {scales.shape}")
    M = int(knots)
    if not (4 <= M <= P + 1):
        raise XiSplineResidualError(
            f"knots must be in [4, P+1={P + 1}] (cubic spline over the P+1 path poses); got {M}")
    # 1-2. decoder-visible table -> SE(3) path (zero twist prepended) -> sampled controls
    xi_ref = dequantize_xi(q, scales)
    xi_lift = np.concatenate([np.zeros((1, 6)), xi_ref], axis=0)
    ctrl = _fit_spline_controls(xi_lift, M)
    # 3. knot payload on the shipped grid, through the REAL ξ coder (min of delta_ar/none)
    logs = np.stack([_se3.log_se3(ctrl[i]) for i in range(M)])
    qk, sk = quantize_xi(logs, q_levels=q_levels_knots)
    kblob = min(
        (serialize_xi_payload(qk, sk, coder=c) for c in ("delta_ar", "none")), key=len)
    # 4-5. predict + residual, all three schemes measured, smallest shipped (unless pinned)
    q_hat = predict_q_from_knots(qk, sk, scales, P)
    res = q.astype(np.int64) - q_hat
    if scheme is None:
        sizes = measure_residual_schemes(res)
        scheme = min(sizes, key=lambda s: sizes[s])
    rblob = _encode_residual(res, scheme)
    body = (
        struct.pack("<BH", _SCHEME_IDS[scheme], M)
        + struct.pack("<I", len(kblob)) + kblob
        + struct.pack("<I", len(rblob)) + rblob
    )
    # 6. NO-FAKE bit-identity self-check through the REAL decode path
    q_rt, off = decode_spline_residual_body(body, 0, P, int(q.shape[1]), scales)
    if off != len(body) or not np.array_equal(q_rt, q):
        raise XiSplineResidualError(
            "spline_residual decode is NOT bit-identical to the source table (NO-FAKE refuse)")
    return body


def decode_spline_residual_body(
    blob: bytes, off: int, P: int, D: int, scales: np.ndarray
) -> tuple[np.ndarray, int]:
    """Inverse of :func:`encode_spline_residual_body` → ``(q int16 (P,D), new_off)``.

    numpy+stdlib only (inflate-portable): parse knots → run the SAME predictor → ``q = q̂ + r``."""
    from tac.boundary_math.xi_pose_coder import parse_xi_payload  # lazy: avoids import cycle

    if D != 6:
        raise XiSplineResidualError(f"spline_residual codes 6-DOF twists; got D={D}")
    scheme_id, M = struct.unpack_from("<BH", blob, off)
    off += 3
    (klen,) = struct.unpack_from("<I", blob, off)
    off += 4
    qk, sk = parse_xi_payload(blob[off : off + klen])
    off += klen
    (rlen,) = struct.unpack_from("<I", blob, off)
    off += 4
    res = _decode_residual(blob[off : off + rlen], scheme_id, P, D)
    off += rlen
    q_hat = predict_q_from_knots(qk, sk, scales, P)
    q64 = q_hat + res
    if np.abs(q64).max(initial=0) > 32767:
        raise XiSplineResidualError("decoded table exceeds int16 range (corrupt payload)")
    return q64.astype(np.int16), off


# --------------------------------------------------------------------------- #
# the delta_res payload BODY (coder id 3): trivial delta predictor + best residual scheme.
# MEASURED WINNER on the real n600 ξ (the spline predictor adds nothing; the entropy stage wins).
# --------------------------------------------------------------------------- #
def encode_delta_res_body(q: np.ndarray, *, scheme: str | None = None) -> bytes:
    """Encode ``q int16 (P,D)`` as: seed row + best-of-three-coded first-order deltas.

    Body layout: ``<B scheme_id> seed(D int16) <I rlen> rblob``. ``scheme=None`` MEASURES all
    three schemes on the delta matrix and ships the smallest. NO-FAKE: full-decode self-check."""
    q = np.asarray(q, dtype=np.int16)
    if q.ndim != 2:
        raise XiSplineResidualError(f"q must be (P,D); got {q.shape}")
    P, D = int(q.shape[0]), int(q.shape[1])
    if P < 1:
        raise XiSplineResidualError("delta_res needs at least 1 row")
    deltas = np.diff(q.astype(np.int64), axis=0)  # (P-1, D)
    if scheme is None:
        sizes = measure_residual_schemes(deltas)
        scheme = min(sizes, key=lambda s: sizes[s])
    rblob = _encode_residual(deltas, scheme)
    body = (
        struct.pack("<B", _SCHEME_IDS[scheme])
        + q[0].tobytes()
        + struct.pack("<I", len(rblob)) + rblob
    )
    q_rt, off = decode_delta_res_body(body, 0, P, D)
    if off != len(body) or not np.array_equal(q_rt, q):
        raise XiSplineResidualError(
            "delta_res decode is NOT bit-identical to the source table (NO-FAKE refuse)")
    return body


def decode_delta_res_body(blob: bytes, off: int, P: int, D: int) -> tuple[np.ndarray, int]:
    """Inverse of :func:`encode_delta_res_body` → ``(q int16 (P,D), new_off)`` (numpy+stdlib)."""
    (scheme_id,) = struct.unpack_from("<B", blob, off)
    off += 1
    seed = np.frombuffer(blob[off : off + D * 2], dtype=np.int16).astype(np.int64)
    off += D * 2
    (rlen,) = struct.unpack_from("<I", blob, off)
    off += 4
    deltas = _decode_residual(blob[off : off + rlen], scheme_id, P - 1, D) if P > 1 \
        else np.zeros((0, D), dtype=np.int64)
    off += rlen
    q64 = np.concatenate([seed[None, :], seed[None, :] + np.cumsum(deltas, axis=0)], axis=0)
    if np.abs(q64).max(initial=0) > 32767:
        raise XiSplineResidualError("decoded table exceeds int16 range (corrupt payload)")
    return q64.astype(np.int16), off


# --------------------------------------------------------------------------- #
# MEASURED report (the knot-sweep surface the measurement script drives)
# --------------------------------------------------------------------------- #
def spline_residual_rate_report(
    q: np.ndarray, scales: np.ndarray, *, knots: int = 16,
    q_levels_knots: int = _XI_QLEVELS_KNOTS_DEFAULT,
) -> dict:
    """MEASURED (never asserted) per-scheme real bytes + the total payload at one knot count.

    Returns knot bytes, all-three residual-scheme bytes, the picked scheme, the TOTAL
    ``serialize_xi_payload(coder='spline_residual')`` container bytes, and the bit-identity
    verdict (encode already fail-closes on it; re-verified here through the container)."""
    from tac.boundary_math.xi_pose_coder import parse_xi_payload, serialize_xi_payload

    q = np.asarray(q, dtype=np.int16)
    scales = np.asarray(scales, dtype=np.float32).reshape(-1)
    P = int(q.shape[0])
    xi_ref = dequantize_xi(q, scales)
    xi_lift = np.concatenate([np.zeros((1, 6)), xi_ref], axis=0)
    ctrl = _fit_spline_controls(xi_lift, int(knots))
    logs = np.stack([_se3.log_se3(ctrl[i]) for i in range(int(knots))])
    qk, sk = quantize_xi(logs, q_levels=q_levels_knots)
    kblob = min((serialize_xi_payload(qk, sk, coder=c) for c in ("delta_ar", "none")), key=len)
    res = q.astype(np.int64) - predict_q_from_knots(qk, sk, scales, P)
    sizes = measure_residual_schemes(res)
    picked = min(sizes, key=lambda s: sizes[s])
    total = serialize_xi_payload(q, scales, coder="spline_residual",
                                 spline_knots=int(knots), spline_q_levels_knots=q_levels_knots)
    q_rt, _ = parse_xi_payload(total)
    # order-0 entropy of the exact residual (the firing memo's floor metric, for comparison)
    ent_bits = 0.0
    for k in range(res.shape[1]):
        _, counts = np.unique(res[:, k], return_counts=True)
        pk = counts / counts.sum()
        ent_bits += float(P * -(pk * np.log2(pk)).sum())
    return {
        "n_pairs": P,
        "knots": int(knots),
        "knot_payload_bytes": len(kblob),
        "residual_scheme_bytes": sizes,
        "picked_scheme": picked,
        "residual_entropy_bytes_order0": ent_bits / 8.0,
        "total_payload_bytes": len(total),
        "rate_term": 25.0 * len(total) / 37_545_489.0,
        "bit_identical": bool(np.array_equal(q_rt, q)),
    }


__all__ = [
    "RESIDUAL_SCHEMES",
    "XiSplineResidualError",
    "encode_delta_res_body",
    "decode_delta_res_body",
    "encode_spline_residual_body",
    "decode_spline_residual_body",
    "measure_residual_schemes",
    "predict_q_from_knots",
    "spline_residual_rate_report",
]
