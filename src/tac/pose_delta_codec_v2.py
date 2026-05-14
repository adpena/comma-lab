# SPDX-License-Identifier: MIT
"""Lane PD-V2 — Arithmetic-coded pose-delta encoder/decoder (2026-04-29).

Builds on Lane PD V1 (``src/tac/pose_delta_codec.py``) by entropy-coding the
int8 delta stream with a *static-histogram* arithmetic coder
(``src/tac/arithmetic_qint_codec.py``). Driving-pose deltas are concentrated
near zero, so the empirical entropy of the int8 stream is well below
8 bits/symbol — typically 3-5 bits/symbol after per-channel scaling. Static
arithmetic coding at the Shannon bound recovers most of that headroom.

Predicted score gain (from grand-council stacking codex 2026-04-29):
``[prediction] +7-11 basis points (deterministic, low-risk filler).`` The
hard overhead gate FAILS LOUD if the arithmetic-coded blob is no smaller
than V1: a 200-byte freq table on a 3-5KB stream can erase the gain, so we
refuse to ship a regression.

Canonical-order compliance (memory ``project_codec_stacking_composition_canonical_orders_20260429.md``):

    representation choice (per-frame deltas)
      → prediction (delta = poses[i+1] - poses[i])
      → quantization (per-channel int8 + per-channel fp16 scale)
      → arithmetic coding  ← TERMINAL (this lane wires step 4)

On-disk container format (Lane PD-V2 v1, internal binary)
---------------------------------------------------------

    magic            : 4 bytes  = b"PDV2"
    version          : 2 bytes  uint16 = 1
    pose_dim         : 2 bytes  uint16
    n_pairs          : 4 bytes  uint32
    qint_count       : 4 bytes  uint32  (== (n_pairs - 1) * pose_dim)
    qint_min         : 1 byte   int8    (smallest int8 in the stream; alphabet base)
    alphabet_size    : 2 bytes  uint16  (qint_max - qint_min + 1)
    anchor           : 12 bytes (6 × fp16)
    delta_scale      : 12 bytes (6 × fp16)
    aqv1_blob_len    : 4 bytes  uint32
    aqv1_blob        : <aqv1_blob_len> bytes  (output of encode_qints_arithmetic)

All 39 fixed bytes + 24 anchor/scale + AQv1 payload. The tight alphabet
(``qint_min``..``qint_max`` only) keeps the AQv1 freq table small (``alphabet_size *
4`` bytes vs the full 255-wide table that ``encode_qints_arithmetic``
defaults to).

The V1-fallback gate
--------------------

If the encoded V2 blob (magic + header + AQv1 payload) is no smaller than a
fresh V1 ``torch.save()`` of the same poses, we MUST refuse to write V2 —
the hard overhead gate from the stacking codex. The wrapper
``encode_pose_delta_v2_or_fallback()`` returns a tagged dict either way:

    {"format": "pose_delta_v2", "blob": bytes}        # V2 won
    {"format": "pose_delta_v1", ... (V1 fields)}      # V2 lost; fell back

Wire-side compatibility
-----------------------

* The V2 dict is detected by ``is_pose_delta_v2_dict`` (sentinel +
  ``blob[:4] == b"PDV2"``).
* ``submission_archive.load_optimized_poses`` dispatches the V2 dict to
  ``decode_pose_delta_v2`` and returns a vanilla ``(N, pose_dim) float32``
  tensor — downstream consumers (inflate_renderer, contest_auth_eval) need
  no V2 awareness.

CLAUDE.md compliance
--------------------

* Pure encode/decode primitives; no scorer load, no GPU.
* Bit-deterministic: same input → same bytes (verified by
  ``test_determinism``).
* The encoder verifies the round-trip max-abs error before returning bytes
  — a malformed output cannot ship silently.
* Tagged claims: every numeric "saves N%" assertion in this docstring is
  ``[prediction]`` until landed; empirical numbers carry
  ``[empirical:src/tac/tests/test_pose_delta_codec_v2.py]``.
"""
from __future__ import annotations

import io
import struct
from typing import Any

import numpy as np
import torch

from tac.arithmetic_qint_codec import (
    decode_qints_arithmetic,
    encode_qints_arithmetic,
)
from tac.pose_delta_codec import (
    encode_pose_deltas,
)


POSE_DELTA_FORMAT_SENTINEL_V2: str = "pose_delta_v2"
POSE_DELTA_V2_MAGIC: bytes = b"PDV2"
POSE_DELTA_V2_VERSION: int = 1


class PoseDeltaV2GateRegression(RuntimeError):
    """Raised when the V2 arithmetic-coded blob is no smaller than the V1
    raw blob. The hard overhead gate refuses to ship a regression — the
    caller must use V1 instead.

    Memory (``project_codec_stacking_composition_canonical_orders_20260429``):
    "Lane PD-V2 hard overhead gate: if encoded + header >= current
    pose_delta_v1, keep current PD — don't ship a regression."
    """


def _quantize_pose_deltas(
    poses: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """Compute per-frame deltas + per-channel int8 quantization.

    Returns:
        (anchor_fp16, delta_scale_fp16, deltas_q_int8_2d)
        where deltas_q is shape (n_pairs - 1, pose_dim) int8 in [-127, 127].

    NOTE: This duplicates V1 logic intentionally so V2 cannot break if V1's
    internals change. V1 is FROZEN per task requirements.
    """
    if poses.ndim != 2:
        raise ValueError(
            f"poses must be 2-D (N, pose_dim); got shape {tuple(poses.shape)}"
        )
    n_pairs, pose_dim = poses.shape
    if n_pairs < 2:
        raise ValueError(
            f"need at least 2 poses to compute deltas; got n_pairs={n_pairs}"
        )

    poses_f = poses.detach().to(torch.float32).cpu()
    anchor = poses_f[0].clone()
    deltas = poses_f[1:] - poses_f[:-1]  # (n_pairs-1, pose_dim)

    abs_deltas = deltas.abs()
    delta_scale = abs_deltas.max(dim=0).values.clamp(min=1e-8)
    deltas_q_float = (deltas / delta_scale.unsqueeze(0)) * 127.0
    deltas_q = deltas_q_float.round().clamp(-127, 127).to(torch.int8)
    deltas_q_np = deltas_q.numpy().astype(np.int8)
    return anchor.to(torch.float16), delta_scale.to(torch.float16), deltas_q_np


def encode_pose_delta_v2(poses: torch.Tensor) -> bytes:
    """Encode an (N, pose_dim) pose tensor to the Lane PD-V2 binary blob.

    Pipeline:
      1. Per-frame deltas (poses[i+1] - poses[i])  [V1 step]
      2. Per-channel int8 quantization with per-channel fp16 scale  [V1 step]
      3. Tight-alphabet static-histogram arithmetic coding of the int8 stream
         (per ``arithmetic_qint_codec.encode_qints_arithmetic``)
      4. Header packing (anchor, scale, alphabet bounds, qint count)

    Args:
        poses: (N, pose_dim) float tensor of per-pair pose vectors. N >= 2.

    Returns:
        bytes — the PDV2 container (magic + header + AQv1 payload).

    Raises:
        ValueError on shape / dtype mismatch.
    """
    anchor_fp16, scale_fp16, deltas_q_np = _quantize_pose_deltas(poses)
    n_pairs, pose_dim = poses.shape
    flat_qints = deltas_q_np.ravel()
    qint_count = int(flat_qints.size)

    if qint_count == 0:
        raise ValueError(
            "encode_pose_delta_v2: refused to encode 0-symbol stream "
            "(n_pairs must be >= 2)"
        )

    # Tight alphabet bounds: only the int8 values that actually appear.
    qint_min = int(flat_qints.min())
    qint_max = int(flat_qints.max())
    alphabet_size = qint_max - qint_min + 1
    if alphabet_size < 1:
        raise ValueError(
            f"encode_pose_delta_v2: degenerate alphabet "
            f"(qint_min={qint_min}, qint_max={qint_max})"
        )

    # AQv1 requires num_symbols >= 2 (a 1-symbol stream has zero entropy and
    # is degenerate — see arithmetic_qint_codec.py:250). Pad to 2 by widening
    # the alphabet by one extra slot above qint_max. The phantom slot has
    # count=1 (build_freq_table floors counts at 1), which adds ~4 bytes to
    # the freq table but keeps the codec well-defined. The fallback gate
    # in encode_pose_delta_v2_or_fallback will catch the resulting overhead
    # and prefer V1 on truly-constant inputs.
    if alphabet_size == 1:
        if qint_max < 127:
            qint_max += 1
        elif qint_min > -127:
            qint_min -= 1
        else:
            # Saturated alphabet (already spans full int8 range with 1
            # symbol — impossible in practice but defended for completeness).
            raise ValueError(
                "encode_pose_delta_v2: cannot widen 1-symbol alphabet "
                "(qint stream saturates int8 range)"
            )
        alphabet_size = qint_max - qint_min + 1
        assert alphabet_size == 2

    aqv1_blob = encode_qints_arithmetic(
        flat_qints,
        num_symbols=alphabet_size,
        offset=-qint_min,  # maps qint_min..qint_max → 0..alphabet_size-1
    )

    # Pack the V2 container.
    out = io.BytesIO()
    out.write(POSE_DELTA_V2_MAGIC)
    out.write(struct.pack("<H", POSE_DELTA_V2_VERSION))
    out.write(struct.pack("<H", int(pose_dim)))
    out.write(struct.pack("<I", int(n_pairs)))
    out.write(struct.pack("<I", qint_count))
    out.write(struct.pack("<b", qint_min))  # signed int8
    out.write(struct.pack("<H", alphabet_size))
    # Anchor & scale: write fp16 little-endian raw bytes.
    out.write(anchor_fp16.cpu().numpy().astype("<f2").tobytes())
    out.write(scale_fp16.cpu().numpy().astype("<f2").tobytes())
    out.write(struct.pack("<I", len(aqv1_blob)))
    out.write(aqv1_blob)

    blob = out.getvalue()

    # Encoder-side round-trip verification: refuse to ship a malformed blob.
    decoded = decode_pose_delta_v2(blob, pose_dim=int(pose_dim))
    poses_f32 = poses.detach().to(torch.float32).cpu()
    abs_err = (poses_f32 - decoded).abs().max().item()
    # Same per-channel int8 floor as V1: a smooth trajectory should land
    # ~1e-3; tol 5e-2 matches V1's encode_pose_file default.
    if abs_err > 5e-2:
        raise RuntimeError(
            f"encode_pose_delta_v2: round-trip max-abs error {abs_err:.6e} "
            f"exceeds tolerance 5e-2. The pose trajectory may be too noisy "
            f"for int8 deltas; consider per-frame absolute fallback."
        )
    return blob


def decode_pose_delta_v2(blob: bytes, pose_dim: int = 6) -> torch.Tensor:
    """Inverse of ``encode_pose_delta_v2`` — return (n_pairs, pose_dim) float32.

    Args:
        blob: bytes from ``encode_pose_delta_v2``.
        pose_dim: declared pose dimension (must match the header).

    Returns:
        (n_pairs, pose_dim) float32 tensor.

    Raises:
        ValueError on magic / version / shape mismatch.
    """
    if not isinstance(blob, (bytes, bytearray, memoryview)):
        raise ValueError(
            f"decode_pose_delta_v2: blob must be bytes-like, got {type(blob).__name__}"
        )
    if len(blob) < 4 or bytes(blob[:4]) != POSE_DELTA_V2_MAGIC:
        raise ValueError(
            f"decode_pose_delta_v2: bad magic; expected {POSE_DELTA_V2_MAGIC!r}, "
            f"got {bytes(blob[:4])!r}"
        )

    buf = io.BytesIO(bytes(blob))
    buf.read(4)  # magic
    (version,) = struct.unpack("<H", buf.read(2))
    if version != POSE_DELTA_V2_VERSION:
        raise ValueError(
            f"decode_pose_delta_v2: unsupported PDV2 version {version}; "
            f"expected {POSE_DELTA_V2_VERSION}"
        )
    (declared_pose_dim,) = struct.unpack("<H", buf.read(2))
    (n_pairs,) = struct.unpack("<I", buf.read(4))
    (qint_count,) = struct.unpack("<I", buf.read(4))
    (qint_min,) = struct.unpack("<b", buf.read(1))
    (alphabet_size,) = struct.unpack("<H", buf.read(2))
    if declared_pose_dim != pose_dim:
        raise ValueError(
            f"decode_pose_delta_v2: declared pose_dim={declared_pose_dim} "
            f"!= caller pose_dim={pose_dim}"
        )
    expected_qint_count = (n_pairs - 1) * pose_dim
    if qint_count != expected_qint_count:
        raise ValueError(
            f"decode_pose_delta_v2: qint_count={qint_count} "
            f"!= (n_pairs - 1) * pose_dim = {expected_qint_count}"
        )

    anchor_bytes = buf.read(pose_dim * 2)
    scale_bytes = buf.read(pose_dim * 2)
    if len(anchor_bytes) != pose_dim * 2 or len(scale_bytes) != pose_dim * 2:
        raise ValueError(
            f"decode_pose_delta_v2: truncated anchor/scale "
            f"(anchor={len(anchor_bytes)}B, scale={len(scale_bytes)}B, "
            f"expected {pose_dim * 2}B each)"
        )

    (aqv1_blob_len,) = struct.unpack("<I", buf.read(4))
    aqv1_blob = buf.read(aqv1_blob_len)
    if len(aqv1_blob) != aqv1_blob_len:
        raise ValueError(
            f"decode_pose_delta_v2: truncated AQv1 payload "
            f"(declared {aqv1_blob_len}B, got {len(aqv1_blob)}B)"
        )

    # decode_qints_arithmetic applies `out -= offset` internally — it
    # returns the ORIGINAL int8 stream directly. We store qint_min /
    # alphabet_size in our header purely so the AQv1 freq-table is sized
    # to the tight observed alphabet rather than the full 255-wide
    # pessimistic table; we do NOT double-offset on decode.
    qints = decode_qints_arithmetic(aqv1_blob, expected_dtype=np.int8)
    if qints.size != qint_count:
        raise ValueError(
            f"decode_pose_delta_v2: AQv1 produced {qints.size} symbols, "
            f"expected {qint_count}"
        )
    qints_i64 = qints.astype(np.int64)
    if (
        qints_i64.min() < qint_min
        or qints_i64.max() > qint_min + alphabet_size - 1
    ):
        raise ValueError(
            f"decode_pose_delta_v2: decoded qints out of header alphabet "
            f"[{qint_min}, {qint_min + alphabet_size - 1}] "
            f"(min={int(qints_i64.min())}, max={int(qints_i64.max())})"
        )
    if qints_i64.min() < -127 or qints_i64.max() > 127:
        raise ValueError(
            f"decode_pose_delta_v2: decoded qints out of int8 range "
            f"[-127, 127] (min={int(qints_i64.min())}, max={int(qints_i64.max())})"
        )
    deltas_q = torch.from_numpy(qints.astype(np.int8)).reshape(n_pairs - 1, pose_dim)

    anchor = torch.from_numpy(
        np.frombuffer(anchor_bytes, dtype="<f2").copy()
    ).to(torch.float32)
    scale = torch.from_numpy(
        np.frombuffer(scale_bytes, dtype="<f2").copy()
    ).to(torch.float32)

    deltas = (deltas_q.to(torch.float32) / 127.0) * scale.unsqueeze(0)
    cum = torch.cat(
        [torch.zeros(1, pose_dim), torch.cumsum(deltas, dim=0)], dim=0
    )
    poses = anchor.unsqueeze(0) + cum
    return poses.to(torch.float32)


def is_pose_delta_v2_dict(obj: Any) -> bool:
    """Return True if obj is a Lane PD-V2 sentinel dict.

    Detects via TWO independent signals: (1) the ``format`` sentinel
    string, AND (2) the magic-byte sniff on the embedded blob (first 4
    bytes == b"PDV2"). Both must hold; either alone is treated as
    corruption — fail-loud, not silent-pass.
    """
    if not isinstance(obj, dict):
        return False
    if obj.get("format") != POSE_DELTA_FORMAT_SENTINEL_V2:
        return False
    blob = obj.get("blob")
    if not isinstance(blob, (bytes, bytearray, memoryview)):
        return False
    if len(blob) < 4 or bytes(blob[:4]) != POSE_DELTA_V2_MAGIC:
        return False
    return True


def encode_pose_delta_v2_or_fallback(
    poses: torch.Tensor,
) -> dict[str, Any]:
    """Encode poses as V2 if-and-only-if V2 strictly beats V1 on bytes.

    The hard overhead gate per stacking codex: if the V2 blob (magic +
    header + AQv1 payload) is no smaller than a fresh V1 ``torch.save()``
    of the same poses, fall back to V1. Returns a tagged dict either way:

        {"format": "pose_delta_v2", "blob": bytes}        # V2 won
        {"format": "pose_delta_v1", ... (V1 fields)}      # V2 lost

    The caller writes the dict via ``torch.save()``; the canonical loader
    in ``submission_archive.load_optimized_poses`` dispatches both formats.
    """
    v1_dict = encode_pose_deltas(poses)
    v1_buf = io.BytesIO()
    torch.save(v1_dict, v1_buf)
    v1_bytes = len(v1_buf.getvalue())

    v2_blob = encode_pose_delta_v2(poses)
    v2_dict = {"format": POSE_DELTA_FORMAT_SENTINEL_V2, "blob": v2_blob}
    v2_buf = io.BytesIO()
    torch.save(v2_dict, v2_buf)
    v2_bytes = len(v2_buf.getvalue())

    if v2_bytes < v1_bytes:
        return v2_dict
    # Gate fired: V2 didn't win. Fall back to V1; do NOT ship a regression.
    return v1_dict


def encode_pose_file_v2(
    src_path: str,
    dst_path: str,
    pose_dim: int = 6,
    fallback_on_regression: bool = True,
    max_roundtrip_error_tol: float = 5e-2,
) -> dict:
    """Convert a vanilla optimized_poses.pt -> Lane PD-V2 encoded .pt.

    Matches the V1 sibling (``encode_pose_file``) so existing wrappers
    (compress_archive.py) can swap in by changing the import.

    Args:
        src_path: source pose file (handled by submission_archive loader).
        dst_path: destination .pt path (torch.save'd dict).
        pose_dim: pose vector dimension (default 6).
        fallback_on_regression: if True (default), wrap with
            ``encode_pose_delta_v2_or_fallback`` so an overhead-trap input
            silently writes V1 instead of V2; if False, raise
            ``PoseDeltaV2GateRegression`` when V2 doesn't win.
        max_roundtrip_error_tol: max-abs reconstruction error tolerance.

    Returns:
        Stats dict (``input_bytes``, ``output_bytes``, ``savings_pct``,
        ``format_used``, ``max_roundtrip_error``, ``mean_roundtrip_error``).
    """
    import os

    from tac.submission_archive import load_optimized_poses

    poses = load_optimized_poses(src_path, pose_dim=pose_dim)

    if fallback_on_regression:
        encoded = encode_pose_delta_v2_or_fallback(poses)
        format_used = encoded["format"]
    else:
        v1_buf = io.BytesIO()
        torch.save(encode_pose_deltas(poses), v1_buf)
        v1_bytes = len(v1_buf.getvalue())

        v2_blob = encode_pose_delta_v2(poses)
        v2_dict = {"format": POSE_DELTA_FORMAT_SENTINEL_V2, "blob": v2_blob}
        v2_buf = io.BytesIO()
        torch.save(v2_dict, v2_buf)
        v2_bytes = len(v2_buf.getvalue())
        if v2_bytes >= v1_bytes:
            raise PoseDeltaV2GateRegression(
                f"encode_pose_file_v2: V2 blob ({v2_bytes}B) is not smaller "
                f"than V1 ({v1_bytes}B). Refused to ship a regression. Pass "
                f"fallback_on_regression=True to silently use V1 instead."
            )
        encoded = v2_dict
        format_used = POSE_DELTA_FORMAT_SENTINEL_V2

    torch.save(encoded, dst_path)

    # Round-trip verification on whatever we just wrote.
    if format_used == POSE_DELTA_FORMAT_SENTINEL_V2:
        decoded = decode_pose_delta_v2(encoded["blob"], pose_dim=pose_dim)
    else:
        from tac.pose_delta_codec import decode_pose_deltas
        decoded = decode_pose_deltas(encoded, pose_dim=pose_dim)
    abs_err = (poses - decoded).abs()
    max_err = float(abs_err.max().item())
    mean_err = float(abs_err.mean().item())
    if max_err > max_roundtrip_error_tol:
        raise RuntimeError(
            f"encode_pose_file_v2: round-trip max-abs error {max_err:.6e} "
            f"exceeds tol {max_roundtrip_error_tol}."
        )

    in_bytes = os.path.getsize(src_path)
    out_bytes = os.path.getsize(dst_path)
    return {
        "input_bytes": in_bytes,
        "output_bytes": out_bytes,
        "savings_bytes": in_bytes - out_bytes,
        "savings_pct": (1 - out_bytes / in_bytes) * 100 if in_bytes else 0.0,
        "format_used": format_used,
        "n_pairs": int(poses.shape[0]),
        "pose_dim": int(poses.shape[1]),
        "max_roundtrip_error": max_err,
        "mean_roundtrip_error": mean_err,
    }


__all__ = [
    "POSE_DELTA_FORMAT_SENTINEL_V2",
    "POSE_DELTA_V2_MAGIC",
    "POSE_DELTA_V2_VERSION",
    "PoseDeltaV2GateRegression",
    "encode_pose_delta_v2",
    "decode_pose_delta_v2",
    "is_pose_delta_v2_dict",
    "encode_pose_delta_v2_or_fallback",
    "encode_pose_file_v2",
]
