"""Daubechies-8 (db4) wavelet residual energy for UNIWARD variance maps.

Daubechies-8 (db4) chosen for compatibility with steganalysis literature.
Alternative: db1 (Haar, faster but less smooth) or db2 for performance.
Default db4 matches Holub & Fridrich 2014 ("Universal distortion function
for steganography in an arbitrary domain", §III.B), which defines the
UNIWARD per-pixel embedding suitability as the sum of |W^h|+|W^v|+|W^d|
over the high-frequency 2-D wavelet sub-bands using an 8-tap Daubechies
filter bank.

This module replaces the broken box-filter "variance" estimate that lived
in `tac.fridrich.variance_weighted_noise`. The box filter inflates its
estimate at sharp edges (where the SegNet B2 stride-2 stem is most
sensitive — exactly where we *don't* want to add noise). The wavelet
estimator localises high-frequency *texture* energy, which is what
UNIWARD actually wants and what is genuinely undetectable by the
EfficientNet-B2 stem.

Implementation notes (Hotz: keep it fast):
  - Pure separable conv2d, no Python loops over pixels.
  - Un-decimated (stride-1) wavelet so per-pixel suitability stays at
    full (H, W) resolution — no upsampling required and the alignment
    between the suitability map and the noise field is exact.
  - Kernels are pre-registered as buffers via `_get_db4_kernels` cache
    keyed by (dtype, device) so we don't reallocate per call.
  - Reflect padding matches the box-filter version's edge behaviour and
    the upstream PyWavelets default.

The kernel coefficients are the canonical Daubechies-4 (db4) low-pass
filter — the literature uses "db4" for the wavelet with 4 vanishing
moments / 8 taps; the "Daubechies-8" name in the paper refers to the
filter LENGTH = 8. Both conventions describe the same filter.

Verification (see `tests/test_wavelet_variance.py`):
  sum(lo)         == sqrt(2)
  sum(hi)         == 0
  sum(lo**2)      == 1
  sum(hi**2)      == 1

This module exists OUTSIDE `tac.fridrich` so it can be reused by future
loss functions (e.g. boundary-weighted L∞) without circular imports.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn.functional as F


__all__ = ["db4_kernels", "wavelet_variance_map"]


# ── Daubechies-4 (db4) coefficients — 8-tap, 4 vanishing moments ────────
# Source: Daubechies, "Ten Lectures on Wavelets" (1992), Table 6.1.
# Cross-checked against PyWavelets `pywt.Wavelet('db4').dec_lo`.
_DB4_LO: Tuple[float, ...] = (
    -0.010597401784997278,
     0.032883011666982945,
     0.030841381835986965,
    -0.18703481171888114,
    -0.027983769416983849,
     0.63088076792959036,
     0.71484657055254153,
     0.23037781330885523,
)


def _qmf_high(lo: Tuple[float, ...]) -> Tuple[float, ...]:
    """Quadrature-mirror high-pass derived from the low-pass filter.

    Standard orthonormal-wavelet construction: g[n] = (-1)^n * h[N-1-n].
    Produces a high-pass filter that satisfies sum(hi) == 0 and
    sum(hi**2) == 1, which is what the Daubechies orthogonality
    conditions require.
    """
    N = len(lo)
    return tuple((-1.0) ** n * lo[N - 1 - n] for n in range(N))


_DB4_HI: Tuple[float, ...] = _qmf_high(_DB4_LO)


# Cache of pre-built kernel tensors keyed by (dtype, device) so callers
# don't pay the allocation cost on every forward pass. The kernels are
# tiny (8 floats × 4 sub-bands) so memory cost is negligible.
_KERNEL_CACHE: dict[tuple, dict[str, torch.Tensor]] = {}


def db4_kernels(
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    """Return the four db4 separable 2-D analysis kernels.

    Keys:
        'LL': low-pass on rows AND columns  (shape (1, 1, 8, 8))
        'LH': low-pass on rows, high-pass on columns
        'HL': high-pass on rows, low-pass on columns
        'HH': high-pass on rows AND columns

    All four tensors share the (1, 1, 8, 8) shape and are ready to feed
    into `F.conv2d` on a single-channel input (B, 1, H, W).
    """
    key = (str(device), str(dtype))
    if key in _KERNEL_CACHE:
        return _KERNEL_CACHE[key]

    lo = torch.tensor(_DB4_LO, dtype=dtype, device=device)
    hi = torch.tensor(_DB4_HI, dtype=dtype, device=device)

    # Outer products give the separable 2-D kernels. PyTorch conv2d
    # convention: (out_channels, in_channels, kH, kW).
    def _outer(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.outer(a, b).unsqueeze(0).unsqueeze(0)

    kernels = {
        "LL": _outer(lo, lo),
        "LH": _outer(lo, hi),
        "HL": _outer(hi, lo),
        "HH": _outer(hi, hi),
    }
    _KERNEL_CACHE[key] = kernels
    return kernels


def wavelet_variance_map(
    image: torch.Tensor,
    wavelet: str = "db4",
    eps: float = 1e-12,
) -> torch.Tensor:
    """Per-pixel UNIWARD-style wavelet residual energy (un-decimated).

    Computes |LH|^2 + |HL|^2 + |HH|^2 of a 1-level 2-D DWT applied to
    the image's luma plane. Stride-1 (un-decimated / stationary) so the
    output stays at the input's (H, W) resolution — no upsampling needed,
    no half-pixel alignment headaches.

    Args:
        image: (B, C, H, W) tensor in [0, 255]. C may be 1 (luma) or 3
            (RGB — converted to BT.601 luma internally to match the
            scorer-side numerics).
        wavelet: only 'db4' / 'daubechies8' is supported. The argument
            exists for forward compatibility (see module docstring for
            db1 / db2 alternatives).
        eps: numerical floor added before returning to keep downstream
            divisions well-behaved.

    Returns:
        (B, 1, H, W) tensor of non-negative residual energy values. Higher
        values = more textured / more UNIWARD-suitable for hiding noise.

    Raises:
        ValueError: if image is not 4-D, has unsupported channel count, or
            if `wavelet` is not 'db4'.
    """
    if image.ndim != 4:
        raise ValueError(
            f"image must be (B, C, H, W); got shape {tuple(image.shape)}"
        )
    if image.shape[1] not in (1, 3):
        raise ValueError(
            f"image must have 1 or 3 channels; got C={image.shape[1]}"
        )
    if wavelet not in ("db4", "daubechies8"):
        raise ValueError(
            f"wavelet must be 'db4' or 'daubechies8'; got {wavelet!r}"
        )

    img_f = image.float()

    # Reduce to single-channel luma (BT.601). Sharing a single suitability
    # field across RGB matches the scorer numerics: PoseNet collapses to
    # YUV6 internally, SegNet feeds the same RGB three times.
    if img_f.shape[1] == 3:
        luma = (
            0.299 * img_f[:, 0:1]
            + 0.587 * img_f[:, 1:2]
            + 0.114 * img_f[:, 2:3]
        )
    else:
        luma = img_f

    kernels = db4_kernels(device=luma.device, dtype=luma.dtype)
    K = kernels["HH"].shape[-1]  # filter length = 8

    # Reflect-pad so the output is exactly (H, W) after a stride-1 conv.
    # Filter length 8 → pad (4 left, 3 right) and (4 top, 3 bottom) to
    # produce H_out = H, W_out = W (PyTorch conv2d uses no centring).
    pad_l = K // 2
    pad_r = K - 1 - pad_l
    luma_p = F.pad(luma, (pad_l, pad_r, pad_l, pad_r), mode="reflect")

    lh = F.conv2d(luma_p, kernels["LH"])
    hl = F.conv2d(luma_p, kernels["HL"])
    hh = F.conv2d(luma_p, kernels["HH"])

    # Energy = sum of squared sub-band coefficients. Equivalent to the
    # "diversity of detail energy" used in the UNIWARD cost in §III.B.
    energy = lh.pow(2) + hl.pow(2) + hh.pow(2)
    return energy.clamp(min=0.0) + eps
