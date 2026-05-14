# SPDX-License-Identifier: MIT
"""PR64 ``unified_brotli`` pose-velocity-only codec — reusable byte grammar.

This module extracts the REUSABLE pose-axis byte-grammar primitive from the
PR64 public submission
(``submissions/unified_brotli/inflate.py``, lines 281-289) into a typed,
golden-vector-backed transducer. PR64's pose codec stores ONLY the
forward-velocity component (pose dim 0, longitudinal speed). The remaining
five pose dimensions (rotation + lateral velocity + lateral/yaw rotation +
yaw rate + roll rate) are reconstructed as ZEROS at inflate time. This is
empirically defensible at PR106 r2 operating-point because the contest
videos PR64 targets are predominantly straight-line forward-velocity
trajectories.

The wire format:

* ``uint16 vel0`` — the absolute first-frame velocity quantum.
* ``int16[n-1] deltas`` — frame-to-frame velocity deltas.

Recovery formula (PR64 source, lines 285-288): ::

    vel_q[0] = vel0
    vel_q[1:] = vel0 + cumsum(deltas)
    vel = vel_q.astype(float32) / 512.0 + 20.0
    poses = concat([vel.reshape(-1, 1), zeros((n, 5))], axis=1)

This module exposes the **velocity stream codec** (encode/decode) as a
standalone primitive. The "zero-pad the other 5 dims" reconstruction is
the consumer's responsibility — this primitive only round-trips the 1D
velocity stream.

Source: ``experiments/results/public_pr_archive_kaggle_mirror/public_pr64_intake_20260505_auto/source/submissions/unified_brotli/inflate.py``
(SHA pinned via ``check_public_pr_intake_clones_pristine``-protected intake).

CLAUDE.md compliance
====================

* No scorer load — pure numpy + struct + stdlib.
* No MPS / torch import.
* No ``/tmp`` paths.
* Frozen dataclass; ``encode → decode`` is bit-exact on the
  ``pr64_unified_brotli_pose_velocity_v1`` golden vector.
* OSS-friendly: public surface is the 3 names re-exported from
  ``tac.packet_compiler``.

[empirical:src/tac/packet_compiler/golden_vectors/pr64_unified_brotli_pose_velocity_v1.json]

score_claim=false; promotion_eligible=false; ready_for_exact_eval_dispatch=false
(byte-faithful port of public PR64 wire format; downstream archive-producing
consumers must run their own contest-CUDA + contest-CPU adjudication on the
exact archive bytes that ship).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

# ── Public dataclass ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class UnifiedBrotliPoseVelocityStream:
    """Encoded pose-velocity-only payload (PR64 ``unified_brotli`` grammar).

    Attributes
    ----------
    payload:
        Raw bytes: ``uint16 vel0 || int16[n-1] deltas`` little-endian.
        Ready to brotli-compress + place in an archive member.
    n_frames:
        Frame count (PR64 uses 600 for a standard 1200-frame chunk).
    scale:
        Quantum scale (PR64 hardcodes ``1/512.0``).
    bias:
        Velocity bias (PR64 hardcodes ``20.0``).
    """

    payload: bytes
    n_frames: int
    scale: float
    bias: float


# ── Encoder / decoder ───────────────────────────────────────────────────────


def encode_unified_brotli_pose_velocity(
    velocities: np.ndarray,
    *,
    scale: float = 1.0 / 512.0,
    bias: float = 20.0,
) -> UnifiedBrotliPoseVelocityStream:
    """Encode a 1D velocity stream under PR64's unified-brotli grammar.

    Parameters
    ----------
    velocities:
        1D float array of forward-velocity values (PR64 reads pose dim 0).
        Quantised as ``round((velocities - bias) / scale)``; the result
        must fit non-negatively in uint16.
    scale, bias:
        Per-frame affine recovery: ``recovered = q * scale + bias``.
        Defaults mirror PR64 (``scale=1/512.0``, ``bias=20.0``).

    Returns
    -------
    UnifiedBrotliPoseVelocityStream
        Container with the encoded payload + recovery parameters.

    Raises
    ------
    ValueError
        On empty input, non-1D input, non-positive scale, out-of-range
        quantisation, or delta overflow of int16.
    """
    arr = np.asarray(velocities, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(
            f"velocities must be 1D; got shape {arr.shape}"
        )
    if arr.size == 0:
        raise ValueError("velocities must be non-empty")
    if not (scale > 0):
        raise ValueError(f"scale must be > 0; got {scale!r}")

    # PR64 contract: vel0 is uint16 (first frame must satisfy v0 >= bias).
    # Subsequent values may go below `bias` because PR64 uses int32 cumsum
    # over int16 deltas — the reconstruction works regardless of sign.
    q = np.round((arr - bias) / scale).astype(np.int64)
    vel0 = int(q[0])
    if vel0 < 0 or vel0 > 0xFFFF:
        raise ValueError(
            f"first-frame velocity quantum out of uint16 range [0, 65535]; "
            f"got {vel0} (v[0]={arr[0]}, bias={bias}, scale={scale})"
        )
    # The cumulative int32 reconstruction in PR64 (vel_q = int32) must not
    # overflow int32 at any cumulative step (essentially unbounded for any
    # realistic velocity stream, but we check anyway).
    int32_min = -(1 << 31)
    int32_max = (1 << 31) - 1
    if q.min() < int32_min or q.max() > int32_max:
        raise ValueError(
            f"cumulative velocity quantum out of int32 range; "
            f"got [{q.min()}, {q.max()}]"
        )

    body = bytearray()
    body += struct.pack("<H", vel0)
    if arr.size > 1:
        deltas = (q[1:] - q[:-1]).astype(np.int64)
        if deltas.min() < -32768 or deltas.max() > 32767:
            raise ValueError(
                f"velocity deltas out of int16 range "
                f"[-32768, 32767]; got [{deltas.min()}, {deltas.max()}]"
            )
        body += deltas.astype(np.int16).tobytes()

    return UnifiedBrotliPoseVelocityStream(
        payload=bytes(body),
        n_frames=int(arr.size),
        scale=float(scale),
        bias=float(bias),
    )


def decode_unified_brotli_pose_velocity(
    payload: bytes,
    *,
    n_frames: int,
    scale: float = 1.0 / 512.0,
    bias: float = 20.0,
) -> np.ndarray:
    """Decode a PR64 unified-brotli velocity payload back to float velocities.

    Parameters
    ----------
    payload:
        Bytes produced by :func:`encode_unified_brotli_pose_velocity`.
        Format: ``uint16 vel0 || int16[n_frames - 1] deltas`` LE.
    n_frames:
        Expected frame count. PR64 uses 600.
    scale, bias:
        Affine recovery parameters (must match the encoder).

    Returns
    -------
    np.ndarray
        1D float32 velocity stream of length ``n_frames``.

    Raises
    ------
    ValueError
        On truncated payload, trailing bytes, non-positive n_frames,
        non-positive scale, or negative reconstructed quantum.
    """
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise TypeError(f"payload must be bytes-like; got {type(payload)!r}")
    payload = bytes(payload)
    if n_frames <= 0:
        raise ValueError(f"n_frames must be > 0; got {n_frames}")
    if not (scale > 0):
        raise ValueError(f"scale must be > 0; got {scale!r}")

    expected_len = 2 + (n_frames - 1) * 2
    if len(payload) != expected_len:
        raise ValueError(
            f"payload length {len(payload)} != expected {expected_len} "
            f"for n_frames={n_frames}"
        )

    (vel0,) = struct.unpack_from("<H", payload, 0)
    # PR64 uses int32 cumsum (line 285 of source) — vel0 is uint16 but the
    # reconstruction tolerates negative cumulative q for subsequent frames.
    q = np.empty(n_frames, dtype=np.int64)
    q[0] = vel0
    if n_frames > 1:
        deltas = np.frombuffer(payload, dtype=np.int16, count=n_frames - 1, offset=2)
        q[1:] = vel0 + np.cumsum(deltas.astype(np.int64))
    return (q.astype(np.float64) * scale + bias).astype(np.float32)


__all__ = [
    "UnifiedBrotliPoseVelocityStream",
    "decode_unified_brotli_pose_velocity",
    "encode_unified_brotli_pose_velocity",
]
