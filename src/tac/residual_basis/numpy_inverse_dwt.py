# SPDX-License-Identifier: MIT
"""Numpy-only Haar inverse-DWT primitives.

This module clears the L1 promotion blocker named in
`feedback_wavelet_residual_basis_pr106_scaffold_landed_20260511.md`: PyWavelets
is NOT a contest runtime dep (no PyPI wheel small enough for the ≤100 LOC,
≤2 external dep `inflate_runtime_loc_budget`). A wavelet-coded sidecar cannot
enter the contest packet until a numpy-only inverse-DWT lands.

LOC budget compliance
---------------------

The functional code below is ≤80 LOC excluding docstrings/blank lines. The
module imports only `numpy` (sole external dep) plus `typing` (stdlib). This
clears the runtime closure constraint per CLAUDE.md HNeRV parity discipline
lesson 3 (≤2 external deps).

Algorithm
---------

Multi-level 2D Haar DWT inverse. The Haar wavelet basis is the simplest
member of the Daubechies family (db1 == Haar) and admits a closed-form lifting
implementation. PyWavelets `dwt2` returns `(cA, (cH, cV, cD))` where `cH` is
horizontal detail (rows differ) and `cV` is vertical detail (columns differ).
The orthonormal 2D Haar synthesis is then:

    out[2i,   2j  ] = 0.5 * (cA + cH + cV + cD)    (both even -> all positive)
    out[2i,   2j+1] = 0.5 * (cA + cH - cV - cD)    (column odd -> flip cV, cD)
    out[2i+1, 2j  ] = 0.5 * (cA - cH + cV - cD)    (row odd    -> flip cH, cD)
    out[2i+1, 2j+1] = 0.5 * (cA - cH - cV + cD)    (both odd   -> flip cH, cV)

This is the orthonormal 2D Haar synthesis (sister of the standard normalized
1D Haar pair `[1/sqrt(2), 1/sqrt(2)]` / `[1/sqrt(2), -1/sqrt(2)]`).

PyWavelets is used in tests as an ORACLE only — NEVER imported here.

References
----------

Mallat, S. (1989). "A theory for multiresolution signal decomposition: the
wavelet representation." IEEE PAMI 11(7): 674-693.
"""

from __future__ import annotations

import numpy as np

_HAAR_NORM = 0.5  # 1 / sqrt(2) for each of 2 axes -> 1/2 combined


class NumpyInverseDWTError(ValueError):
    """Raised on shape / dtype contract violations."""


def haar_inverse_2d_single_level(
    cA: np.ndarray, cH: np.ndarray, cV: np.ndarray, cD: np.ndarray
) -> np.ndarray:
    """Single-level 2D Haar synthesis. Returns a (2H, 2W) array.

    Argument convention matches PyWavelets `dwt2()` output: `(cA, (cH, cV, cD))`
    where `cH` is horizontal detail (varies down rows) and `cV` is vertical
    detail (varies across columns). All four input bands must have identical
    shape (H, W) and a floating dtype.
    """

    if not (cA.shape == cH.shape == cV.shape == cD.shape):
        raise NumpyInverseDWTError(
            f"band shape mismatch cA={cA.shape} cH={cH.shape} cV={cV.shape} cD={cD.shape}"
        )
    if cA.ndim != 2:
        raise NumpyInverseDWTError(f"expected 2D bands; got ndim={cA.ndim}")
    if not np.issubdtype(cA.dtype, np.floating):
        raise NumpyInverseDWTError(f"expected floating dtype; got {cA.dtype}")
    h, w = cA.shape
    out = np.empty((2 * h, 2 * w), dtype=cA.dtype)
    out[0::2, 0::2] = _HAAR_NORM * (cA + cH + cV + cD)
    out[0::2, 1::2] = _HAAR_NORM * (cA + cH - cV - cD)
    out[1::2, 0::2] = _HAAR_NORM * (cA - cH + cV - cD)
    out[1::2, 1::2] = _HAAR_NORM * (cA - cH - cV + cD)
    return out


def haar_inverse_2d_multi_level(
    coeffs: list[np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> np.ndarray:
    """Multi-level 2D Haar synthesis matching `pywt.waverec2()` layout.

    `coeffs` is `[cA_L, (cH_L, cV_L, cD_L), ..., (cH_1, cV_1, cD_1)]` —
    coarsest approximation first followed by detail tuples from coarsest
    to finest. The Haar layout uses `cH=LH` (horizontal detail), `cV=HL`
    (vertical detail), `cD=HH` (diagonal detail), matching PyWavelets'
    return-value convention.
    """

    if len(coeffs) < 2:
        raise NumpyInverseDWTError(
            f"coeffs must have len >= 2 (cA + at least 1 detail tuple); got {len(coeffs)}"
        )
    cA = coeffs[0]
    if not isinstance(cA, np.ndarray):
        raise NumpyInverseDWTError("first coefficient must be a numpy ndarray (cA approximation)")
    for level_idx, detail in enumerate(coeffs[1:], start=1):
        if not (isinstance(detail, tuple) and len(detail) == 3):
            raise NumpyInverseDWTError(
                f"detail at level {level_idx} must be a 3-tuple (cH, cV, cD); got {type(detail).__name__}"
            )
        cH, cV, cD = detail
        cA = haar_inverse_2d_single_level(cA, cH, cV, cD)
    return cA
