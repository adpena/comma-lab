# SPDX-License-Identifier: MIT
"""ξ (SE(3) ego-twist) pose-carrier coder — the truly-optimal store-nothing payload (#257).

FINDING-1 (#256, ``.omx/research/finding1_store_nothing_pose_rate_resolution_20260703.md``)
established that the store-nothing pose carrier shipped **52,135 B** of which **83 %
(43,200 B) was a redundant per-pair fp64 homography ``H``** that is a DETERMINISTIC FREE
function of the already-stored twist ``ξ`` (``H = K (R − t nᵀ/d) K⁻¹`` with ``(R,t) =
exp_se3(ξ)``). Every non-ξ input (K, plane normal ``n(pitch)``, camera height ``d``) is a
fixed camera-calibration constant, so **H carries NO per-pair information beyond ξ** →
storing it is pure redundancy (rule-118: the derive-H algorithm is FREE in inflate.py; only
the video-derived ξ trajectory is COUNTED).

This module is the byte-optimal ξ payload: it stores ONLY the per-pair 6-DOF ego twist,
quantized to a per-channel fixed-point grid, and (default) temporal-delta + arithmetic-codes
the smooth ego trajectory. H is DERIVED at decode (``homographies_from_xi``) — never stored.

TWO variants (both decode to the IDENTICAL quantized ξ → strict bit-parity → identical H →
identical frames → identical d_pose):

* ``coder="delta_ar"`` (default) — per-channel first-order temporal delta + a transmitted-PMF
  static arithmetic coder (``tac.lossless.range_coder``; smooth ego-motion → small deltas →
  ~0.0007 rate target at n600). The model bytes are COUNTED (self-describing payload).
* ``coder="none"`` (the ``--no-xi-coder`` fallback) — the raw int16 quantized ξ (P·6·2 B →
  ~0.005 rate at n600). The GUARANTEED-today baseline AND the strict-parity reference the
  coded path must match bit-for-bit.

QUANTIZATION (the source of truth, shared by BOTH variants):
  per-channel symmetric int16 fixed-point ``q = round(ξ / scale)``, ``scale_k =
  max_p|ξ[p,k]| / 32767`` (≥ fp16 precision; the 6 fp32 scales are stored in the payload,
  24 B). ``dequantize_xi(q, scale) = q · scale``. Deterministic + host-portable + lossless
  round-trip on the quantized grid.

NO-FAKE / rule-118 firewall:
  * COUNTED (archive.zip): the quantized ξ payload (video-derived ego trajectory) + the 6
    per-channel scales + the scalar ``pitch`` (needed to rebuild the plane geometry).
  * FREE (inflate.py): ``exp_se3`` + ``H = K(R − t nᵀ/d)K⁻¹`` + inverse-warp bilinear + R +
    the arithmetic decoder. All generic algorithm on fixed camera calibration — 0 archive bytes.

DETERMINISM / AUTHORITY: numpy-fp64 CPU is the derive-H + quantization authority (matches
``tac.boundary_math.warp_real_luma_frame0.homography_from_xi_numpy`` bit-for-bit). MLX-GPU is
NOT bit-identical cross-process, so any bit-exact proof stays CPU-locked (standing discipline).
NEVER a score authority — this module only produces bytes/frames; every d_pose is measured
through the frozen CPU-torch PoseNet.

Cross-refs: FINDING-1 memo; ``warp_real_luma_frame0`` (the warp/geometry authority);
``tac.lie._se3_numpy`` (the SE(3) exp oracle); ``tac.lossless.range_coder`` (the arithmetic
coder); the shipped inflate.py inlines VERBATIM copies of ``decode_xi_payload`` +
``homographies_from_xi`` (bit-exact-gate proven).
"""
from __future__ import annotations

import struct

import numpy as np

# Payload container magic + version. v2 = derive-H store-nothing (NO stored H block).
_XI_MAGIC = b"XIP2"
_XI_QMAX = 32767  # symmetric int16 quantization max (per-channel 15-bit precision)
_CODER_RAW = 0
_CODER_DELTA_AR = 1
_CODER_SPLINE_RESIDUAL = 2  # SE(3)-spline predictor + lossless residual (xi_spline_residual_coder)
_CODER_DELTA_RES = 3  # delta predictor + best-of-{varint,zlib9,rice} residual (MEASURED winner)

# Plane-induced ground-homography geometry constants (EON / comma2k19 road camera,
# native 1164x874). MUST match tac.boundary_math.warp_real_luma_frame0 exactly so the
# derived H is bit-identical to homography_from_xi_numpy.
_NATIVE_W, _NATIVE_H = 1164, 874
_NATIVE_FX, _NATIVE_FY = 910.0, 910.0
_NATIVE_CX, _NATIVE_CY = 582.0, 437.0
_CAMERA_HEIGHT_M = 1.22
_EPS = 1e-6


class XiPoseCoderError(ValueError):
    """A ξ-pose-coder contract violation (shape / dtype / payload corruption)."""


# --------------------------------------------------------------------------- #
# quantization (the shared source of truth for BOTH variants)
# --------------------------------------------------------------------------- #
# Default quantization precision. 4096 (12-bit) ≈ fp16+ precision — the MEASURED sweet spot
# (n24, live #205 calib): tightest d_pose to the fp64 twist (Δd_pose ~+2e-4 on d_pose~11, i.e.
# ~5e-5 in the √(10·d_pose) score term — invariant) at ~0.0021 n600 coded rate. Finer wastes rate
# on lateral ego-jitter; coarser drifts d_pose. Overridable per call; the raw fallback stores the
# SAME q so both variants stay strict-parity at any precision.
_XI_QLEVELS_DEFAULT = 4096


def quantize_xi(xi: np.ndarray, *, q_levels: int = _XI_QLEVELS_DEFAULT) -> tuple[np.ndarray, np.ndarray]:
    """``ξ (P,6) float -> (q int16 (P,6), scales fp32 (6,))``.

    Per-channel symmetric fixed-point: ``scale_k = max_p|ξ[p,k]| / q_levels`` (floored to avoid
    div-0 on an all-zero channel), ``q = clip(round(ξ / scale), ±q_levels)``. ``q_levels``
    (default ``2048`` ≈ fp16 precision) trades rate (finer → larger deltas → more coded bits)
    against warp fidelity (finer → tighter d_pose to the fp64 twist). ``q_levels ≤ 32767`` so q
    is int16. Deterministic + host-portable; the round-trip is exact on the quantized grid."""
    xi = np.asarray(xi, dtype=np.float64)
    if xi.ndim != 2 or xi.shape[1] != 6:
        raise XiPoseCoderError(f"xi must be (P,6); got {xi.shape}")
    ql = int(q_levels)
    if not (1 <= ql <= _XI_QMAX):
        raise XiPoseCoderError(f"q_levels must be in [1, {_XI_QMAX}]; got {q_levels}")
    maxabs = np.maximum(np.abs(xi).max(axis=0), _EPS)  # (6,) per-channel; floor prevents div-0
    scales = (maxabs / ql).astype(np.float32)
    q = np.round(xi / scales.astype(np.float64))
    q = np.clip(q, -ql, ql).astype(np.int16)
    return q, scales


def dequantize_xi(q: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """``(q int16 (P,6), scales fp32 (6,)) -> ξ (P,6) fp64``. The decode authority."""
    return np.asarray(q, dtype=np.float64) * np.asarray(scales, dtype=np.float64)


# --------------------------------------------------------------------------- #
# derive-H (FREE, rule-118): ξ -> plane-induced ground homography, batched
# --------------------------------------------------------------------------- #
def _intrinsics_native() -> tuple[np.ndarray, np.ndarray]:
    K = np.array(
        [[_NATIVE_FX, 0.0, _NATIVE_CX], [0.0, _NATIVE_FY, _NATIVE_CY], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return K, np.linalg.inv(K)


def homographies_from_xi(xi: np.ndarray, pitch: float) -> np.ndarray:
    """``ξ (P,6) fp64 + pitch -> H (P,3,3) fp64`` (batched, op-for-op == the per-pair
    ``warp_real_luma_frame0.homography_from_xi_numpy(ξ[p], eon(pitch))``).

    ``T = exp_se3(ξ)``: ``R = exp_so3(ω)``, ``t = J_l(ω) ρ`` (translation-first twist);
    ``H = K (R − t nᵀ/d) K⁻¹`` with ``n = [0, −cos p, −sin p]``, ``d = 1.22``. Pure numpy fp64
    (the deterministic decode authority). The shipped inflate inlines a verbatim copy."""
    xi = np.asarray(xi, dtype=np.float64)
    if xi.ndim != 2 or xi.shape[1] != 6:
        raise XiPoseCoderError(f"xi must be (P,6); got {xi.shape}")
    rho = xi[:, :3]
    omega = xi[:, 3:]
    a, b, c = omega[:, 0], omega[:, 1], omega[:, 2]
    z = np.zeros_like(a)
    # skew(omega) -> K_so3 (P,3,3): [[0,-c,b],[c,0,-a],[-b,a,0]]
    Kso = np.stack(
        [np.stack([z, -c, b], -1), np.stack([c, z, -a], -1), np.stack([-b, a, z], -1)], -2
    )
    Kso2 = Kso @ Kso
    theta2 = (omega * omega).sum(-1)
    theta = np.sqrt(np.maximum(theta2, 0.0))
    small = theta < _EPS
    th_safe = np.maximum(theta, _EPS)
    th2_safe = np.maximum(theta2, _EPS * _EPS)
    th3_safe = np.maximum(theta**3, _EPS**3)
    A = np.where(small, 1.0 - theta2 / 6.0 + theta2 * theta2 / 120.0, np.sin(theta) / th_safe)
    B = np.where(small, 0.5 - theta2 / 24.0 + theta2 * theta2 / 720.0, (1.0 - np.cos(theta)) / th2_safe)
    C = np.where(
        small,
        1.0 / 6.0 - theta2 / 120.0 + theta2 * theta2 / 5040.0,
        (theta - np.sin(theta)) / th3_safe,
    )
    eye = np.broadcast_to(np.eye(3, dtype=np.float64), Kso.shape).copy()
    R = eye + A[..., None, None] * Kso + B[..., None, None] * Kso2  # exp_so3(omega)
    V = eye + B[..., None, None] * Kso + C[..., None, None] * Kso2  # left_jacobian_so3(omega)
    t = (V @ rho[..., None])[..., 0]  # (P,3)
    K, Kinv = _intrinsics_native()
    n = np.array([0.0, -np.cos(float(pitch)), -np.sin(float(pitch))], dtype=np.float64)
    tn = t[..., :, None] * n[None, None, :]  # outer(t, n) per pair (P,3,3)
    M = R - tn / _CAMERA_HEIGHT_M
    return K[None] @ M @ Kinv[None]


# --------------------------------------------------------------------------- #
# arithmetic coder over the temporal-delta stream (reuses tac.lossless.range_coder)
# --------------------------------------------------------------------------- #
def _channel_delta_encode(col: np.ndarray) -> bytes:
    """Encode one int16 channel column ``(P,)`` as: seed + temporal-delta arithmetic stream.

    Layout: ``<i seed> <i lo> <i hi> <I model_len> brotli(counts uint32[hi-lo+1]) <I stream_len>
    stream``. A size-1 delta alphabet (all deltas equal) carries ZERO information → empty
    model+stream, deterministically reconstructed from ``lo`` at decode."""
    import brotli

    from tac.lossless.range_coder import encode_static_symbols

    col = np.asarray(col, dtype=np.int64)
    P = int(col.shape[0])
    seed = int(col[0]) if P else 0
    deltas = np.diff(col) if P > 1 else np.zeros(0, dtype=np.int64)
    lo = int(deltas.min()) if deltas.size else 0
    hi = int(deltas.max()) if deltas.size else 0
    if deltas.size and hi > lo:
        counts = np.bincount((deltas - lo).astype(np.int64), minlength=hi - lo + 1)
        counts = np.maximum(counts, 1).astype(np.uint32)  # every symbol >= 1 (decoder-safe)
        stream = encode_static_symbols((deltas - lo).astype(np.int64).tolist(), frequencies=counts.tolist())
        model = brotli.compress(counts.tobytes(), quality=11)
    else:
        stream = b""
        model = b""
    return (
        struct.pack("<iii", seed, lo, hi)
        + struct.pack("<I", len(model)) + model
        + struct.pack("<I", len(stream)) + stream
    )


def _channel_delta_decode(blob: bytes, off: int, P: int) -> tuple[np.ndarray, int]:
    """Inverse of :func:`_channel_delta_encode`. Returns ``(col int64 (P,), new_off)``."""
    import brotli

    from tac.lossless.range_coder import decode_static_symbols

    seed, lo, hi = struct.unpack_from("<iii", blob, off)
    off += 12
    (mlen,) = struct.unpack_from("<I", blob, off)
    off += 4
    counts = np.frombuffer(brotli.decompress(blob[off : off + mlen]), dtype=np.uint32) if mlen else None
    off += mlen
    (slen,) = struct.unpack_from("<I", blob, off)
    off += 4
    stream = blob[off : off + slen]
    off += slen
    col = np.empty(P, dtype=np.int64)
    if P:
        col[0] = seed
    if P > 1:
        if hi > lo and counts is not None:
            syms = np.asarray(decode_static_symbols(stream, count=P - 1, frequencies=counts.tolist()), dtype=np.int64)
            d = syms + lo
        else:
            d = np.full(P - 1, lo, dtype=np.int64)  # size-1 alphabet: every delta == lo
        col[1:] = seed + np.cumsum(d)
    return col, off


# --------------------------------------------------------------------------- #
# unified self-describing ξ payload
# --------------------------------------------------------------------------- #
def serialize_xi_payload(
    q: np.ndarray,
    scales: np.ndarray,
    *,
    coder: str = "delta_ar",
    spline_knots: int = 16,
    spline_q_levels_knots: int = _XI_QLEVELS_DEFAULT,
) -> bytes:
    """Serialize the quantized ξ payload.

    Layout: ``XIP2 | <B coder> <H P> <B D> | scales(D fp32) | body``.
    body(coder=none)     = q int16 (P,D) tobytes  (the raw ~0.005-rate fallback).
    body(coder=delta_ar) = per-channel :func:`_channel_delta_encode` (the ~0.0007 target).
    body(coder=spline_residual) = SE(3)-spline predictor knots + lossless integer residual
    (``xi_spline_residual_coder``; ``spline_knots``/``spline_q_levels_knots`` apply ONLY here).
    body(coder=delta_res) = delta predictor + best-of-{varint,zlib9,rice} residual (the MEASURED
    n600 winner: ~2.7 KB vs delta_ar's 3.2 KB on the same table, bit-identical).
    DEFAULT stays ``delta_ar`` — byte-identical to the pre-spline coder (pinned by test).
    """
    q = np.asarray(q, dtype=np.int16)
    if q.ndim != 2:
        raise XiPoseCoderError(f"q must be (P,D); got {q.shape}")
    P, D = int(q.shape[0]), int(q.shape[1])
    scales = np.asarray(scales, dtype=np.float32).reshape(-1)
    if scales.shape[0] != D:
        raise XiPoseCoderError(f"scales must be (D={D},); got {scales.shape}")
    if coder == "none":
        cid, body = _CODER_RAW, q.tobytes()
    elif coder == "delta_ar":
        cid = _CODER_DELTA_AR
        body = b"".join(_channel_delta_encode(q[:, k]) for k in range(D))
    elif coder == "spline_residual":
        from tac.boundary_math.xi_spline_residual_coder import encode_spline_residual_body

        cid = _CODER_SPLINE_RESIDUAL
        body = encode_spline_residual_body(
            q, scales, knots=spline_knots, q_levels_knots=spline_q_levels_knots)
    elif coder == "delta_res":
        from tac.boundary_math.xi_spline_residual_coder import encode_delta_res_body

        cid = _CODER_DELTA_RES
        body = encode_delta_res_body(q)
    else:
        raise XiPoseCoderError(
            f"unknown coder {coder!r} (none|delta_ar|spline_residual|delta_res)")
    return _XI_MAGIC + struct.pack("<BHB", cid, P, D) + scales.tobytes() + body


def parse_xi_payload(blob: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of :func:`serialize_xi_payload` -> ``(q int16 (P,D), scales fp32 (D,))``.

    Bit-exact on the quantized grid for BOTH coders (the coded path decodes to the IDENTICAL
    int16 q as the raw path — strict parity). The shipped inflate inlines a verbatim copy."""
    if blob[: len(_XI_MAGIC)] != _XI_MAGIC:
        raise XiPoseCoderError("bad ξ-payload magic")
    off = len(_XI_MAGIC)
    cid, P, D = struct.unpack_from("<BHB", blob, off)
    off += 4
    scales = np.frombuffer(blob[off : off + D * 4], dtype=np.float32).copy()
    off += D * 4
    if cid == _CODER_RAW:
        q = np.frombuffer(blob[off : off + P * D * 2], dtype=np.int16).reshape(P, D).copy()
    elif cid == _CODER_DELTA_AR:
        cols = []
        for _k in range(D):
            col, off = _channel_delta_decode(blob, off, P)
            cols.append(col)
        q = np.stack(cols, axis=1).astype(np.int16)
    elif cid == _CODER_SPLINE_RESIDUAL:
        from tac.boundary_math.xi_spline_residual_coder import decode_spline_residual_body

        q, off = decode_spline_residual_body(blob, off, P, D, scales)
    elif cid == _CODER_DELTA_RES:
        from tac.boundary_math.xi_spline_residual_coder import decode_delta_res_body

        q, off = decode_delta_res_body(blob, off, P, D)
    else:
        raise XiPoseCoderError(f"unknown coder id {cid}")
    return q, scales


def decode_xi_payload(blob: bytes) -> np.ndarray:
    """``ξ payload -> ξ (P,6) fp64`` (parse + dequantize). The decode-side convenience."""
    q, scales = parse_xi_payload(blob)
    return dequantize_xi(q, scales)


def xi_payload_rate_report(xi: np.ndarray, pitch: float, *, q_levels: int = _XI_QLEVELS_DEFAULT) -> dict:
    """MEASURED (not asserted) byte/rate report for both variants on a real ξ (P,6).

    Verifies the round-trip is bit-exact on the quantized grid and that BOTH variants decode
    to the IDENTICAL q (strict parity), then reports raw/coded bytes + rate contribution
    ``25·B/37_545_489``."""
    q, scales = quantize_xi(xi, q_levels=q_levels)
    raw = serialize_xi_payload(q, scales, coder="none")
    coded = serialize_xi_payload(q, scales, coder="delta_ar")
    q_raw, s_raw = parse_xi_payload(raw)
    q_cod, s_cod = parse_xi_payload(coded)
    if not (np.array_equal(q_raw, q) and np.array_equal(q_cod, q)):
        raise XiPoseCoderError("ξ payload round-trip NOT bit-exact on the quantized grid (NO-FAKE)")
    if not np.array_equal(q_raw, q_cod):
        raise XiPoseCoderError("raw vs coded ξ variants decode to DIFFERENT q (strict-parity violation)")
    denom = 37_545_489.0
    return {
        "n_pairs": int(q.shape[0]),
        "raw_bytes": len(raw),
        "coded_bytes": len(coded),
        "raw_rate": 25.0 * len(raw) / denom,
        "coded_rate": 25.0 * len(coded) / denom,
        "coded_vs_raw_ratio": len(coded) / max(1, len(raw)),
        "round_trip_bit_exact": True,
        "raw_coded_strict_parity": True,
    }


__all__ = [
    "XiPoseCoderError",
    "quantize_xi",
    "dequantize_xi",
    "homographies_from_xi",
    "serialize_xi_payload",
    "parse_xi_payload",
    "decode_xi_payload",
    "xi_payload_rate_report",
]
