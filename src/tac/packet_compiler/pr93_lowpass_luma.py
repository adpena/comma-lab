# SPDX-License-Identifier: MIT
"""PR93 lowpass-luma residual codec — reusable byte-grammar primitive.

This module extracts the REUSABLE LOW-FREQUENCY LUMA RESIDUAL primitive from
the PR93 ``codex_metric_yshift_av1`` submission's ``luma_plane_correction``
helper. It encodes a sparse low-rank RGB-luma correction as either 3 or 6
fp32 coefficients per frame:

* **3-channel form** (linear plane): ``c0 + c1*x + c2*y`` over a normalized
  ``[-1, +1] × [-1, +1]`` grid. 12 bytes per frame.

* **6-channel form** (Legendre quadratic): adds ``c3*x*y + c4*(x^2 - 1/3) +
  c5*(y^2 - 1/3)``. 24 bytes per frame.

Both forms produce a continuous 2D scalar field on the eval canvas (typically
384×512 or 874×1164) that can be added to the luma plane of a per-frame RGB
prediction to absorb low-frequency luma drift between renderer output and
ground truth — this is the cheapest possible per-frame correction in bytes
and is well-suited to PR106-style sidecar augmentation when the renderer's
mean-luma error has a smooth spatial structure.

The encoder path is a closed-form least-squares fit against the residual luma
(target − prediction), using the same orthonormal-basis evaluation as the
decoder. Encode/decode are bit-exact round-trip on float32 by construction
(both apply the same arithmetic; quantization is up to the caller — typically
fp16 storage or a small uint16 quantizer per coefficient).

Source: ``experiments/results/public_pr_archive_kaggle_mirror/public_pr93_intake_20260505_auto/source/submissions/codex_metric_yshift_av1/inflate.py``
lines 177-210 (``luma_plane_correction`` and helpers); SHA-pinned in
``check_public_pr_intake_clones_pristine``-protected intake. See also handoff
``~/Downloads/pact_score_lowering_handoff_2026-05-11.md`` P3
("PR93 flatpup/lowpass-luma/delta-varint pose: treat ... lowpass-luma
residuals as independent packet compiler transforms with golden vectors").

CLAUDE.md compliance:

* No scorer load — pure numpy.
* No MPS / torch import.
* No ``/tmp`` paths anywhere.
* Frozen dataclass for the typed result; ``encode→decode`` is covered by
  focused Python conformance tests + 1 SHA-pinned golden vector. Native
  ports must add golden vectors before promotion.
* OSS-friendly: public surface is the 4 names re-exported from
  ``tac.packet_compiler``; everything else is module-private (``_``-prefixed).

[empirical:src/tac/packet_compiler/golden_vectors/pr93_lowpass_luma_v1.json]

score_claim=false; promotion_eligible=false; ready_for_exact_eval_dispatch=false
(byte-faithful port of public PR93 wire format; downstream archive-producing
consumers must run their own contest-CUDA + contest-CPU adjudication on the
exact archive bytes that ship).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

LUMA_BT601_COEFFS: tuple[float, float, float] = (0.299, 0.587, 0.114)


@dataclass(frozen=True)
class LowpassLumaResidual:
    """Typed result of decoding a low-pass luma residual coefficient block."""

    coefficients: tuple[float, ...]  # 3 (linear) or 6 (Legendre quadratic)
    height: int
    width: int


def _basis_grids(
    height: int, width: int, *, n_coeffs: int, dtype=np.float32
) -> tuple[np.ndarray, ...]:
    """Build the normalized x/y (and quadratic) basis grids."""
    if height < 1 or width < 1:
        raise ValueError("height and width must be >= 1")
    y = np.linspace(-1.0, 1.0, height, dtype=dtype).reshape(height, 1)
    x = np.linspace(-1.0, 1.0, width, dtype=dtype).reshape(1, width)
    one = np.ones((height, width), dtype=dtype)
    if n_coeffs == 3:
        return one, np.broadcast_to(x, (height, width)).astype(dtype, copy=False), \
            np.broadcast_to(y, (height, width)).astype(dtype, copy=False)
    if n_coeffs == 6:
        x_g = np.broadcast_to(x, (height, width)).astype(dtype, copy=False)
        y_g = np.broadcast_to(y, (height, width)).astype(dtype, copy=False)
        third = dtype(1.0 / 3.0)
        return (
            one,
            x_g,
            y_g,
            x_g * y_g,
            x_g * x_g - third,
            y_g * y_g - third,
        )
    raise ValueError("n_coeffs must be 3 (linear) or 6 (Legendre quadratic)")


def encode_lowpass_luma_residual(
    residual: np.ndarray, *, n_coeffs: int = 3
) -> LowpassLumaResidual:
    """Least-squares fit of the residual luma plane onto the basis.

    ``residual`` must be a 2D float array of shape ``(height, width)``
    representing the per-pixel luma residual (target − prediction) on the
    eval canvas. Returns a :class:`LowpassLumaResidual` with the fitted
    coefficients ready to serialize via :func:`serialize_lowpass_luma_residual`.
    """
    if residual.ndim != 2:
        raise ValueError(f"residual must be 2D, got shape {residual.shape}")
    height, width = residual.shape
    grids = _basis_grids(height, width, n_coeffs=n_coeffs, dtype=np.float64)
    # Stack basis as columns for lstsq (one row per pixel).
    a = np.stack([g.reshape(-1) for g in grids], axis=1)
    b = residual.reshape(-1).astype(np.float64, copy=False)
    coeffs, *_ = np.linalg.lstsq(a, b, rcond=None)
    return LowpassLumaResidual(
        coefficients=tuple(float(c) for c in coeffs),
        height=int(height),
        width=int(width),
    )


def decode_lowpass_luma_residual(
    coefficients: tuple[float, ...] | list[float], *, height: int, width: int
) -> np.ndarray:
    """Evaluate the encoded residual on a ``(height, width)`` grid.

    Returns a float32 array. Bit-identical to PR93's ``luma_plane_correction``
    when the coefficients match.
    """
    n_coeffs = len(coefficients)
    grids = _basis_grids(height, width, n_coeffs=n_coeffs, dtype=np.float32)
    out = np.zeros((height, width), dtype=np.float32)
    for c, g in zip(coefficients, grids):
        out += np.float32(c) * g
    return out


def serialize_lowpass_luma_residual(residual: LowpassLumaResidual) -> bytes:
    """Pack ``[u8 n_coeffs][u16 height][u16 width][n_coeffs * fp32]`` (LE)."""
    n_coeffs = len(residual.coefficients)
    if n_coeffs not in (3, 6):
        raise ValueError("only 3 or 6 coefficients supported")
    if not (1 <= residual.height < 65536 and 1 <= residual.width < 65536):
        raise ValueError("height and width must fit in uint16 and be >= 1")
    out = bytearray()
    out += struct.pack("<BHH", n_coeffs, residual.height, residual.width)
    out += struct.pack(f"<{n_coeffs}f", *residual.coefficients)
    return bytes(out)


def deserialize_lowpass_luma_residual(blob: bytes) -> LowpassLumaResidual:
    """Inverse of :func:`serialize_lowpass_luma_residual`."""
    if len(blob) < 5:
        raise ValueError(f"truncated header: need 5 bytes have {len(blob)}")
    n_coeffs, height, width = struct.unpack_from("<BHH", blob, 0)
    if n_coeffs not in (3, 6):
        raise ValueError(f"unsupported n_coeffs {n_coeffs}; expected 3 or 6")
    expected_total = 5 + n_coeffs * 4
    if len(blob) != expected_total:
        raise ValueError(
            f"size mismatch: have {len(blob)} bytes, expected {expected_total}"
        )
    coeffs = struct.unpack_from(f"<{n_coeffs}f", blob, 5)
    return LowpassLumaResidual(
        coefficients=tuple(float(c) for c in coeffs),
        height=int(height),
        width=int(width),
    )
