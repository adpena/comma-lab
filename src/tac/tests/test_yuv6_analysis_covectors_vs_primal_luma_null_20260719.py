# SPDX-License-Identifier: MIT
"""Source-only 3x3 fixture: YUV6 analysis covectors are NOT the primal luma-null plane.

Task #570 deliverable 2 (from the #564 surprise review §4). The BT.601 map in
``upstream/frame_utils.py:60-62`` has analysis rows

    ell = ( 0.299,  0.587,  0.114)              (luma Y)
    u   = (-0.299, -0.587,  0.886) / 1.772       (U = (B - Y)/1.772 + 128)
    v   = ( 0.701, -0.587, -0.114) / 1.402       (V = (R - Y)/1.402 + 128)

Both u.(1,1,1) and v.(1,1,1) are zero, so span{u,v} = (1,1,1)^perp. But
ell.u != 0 and ell.v != 0, so span{u,v} is NOT ker(ell): the two planes differ
by a 30.27914784 deg principal angle (projector spectral distance 0.504213367).

Therefore the historical claim (frozen_scorer_exact_factorization_20260715.md:44-56
+ the tools/c2_perclass_stratum_carrier_analysis.py:17-20 docstring) that the
"chroma plane" is both span{U,V} AND the orthogonal complement of ell is FALSE.
The active Euclidean energy split in that tool is span{ell}/ker(ell) — a valid
Euclidean diagnostic, NOT a U/V analysis split.

These are exact finite-dimensional linear-algebra facts (source-derived; no
scorer forward, no data). NO score claim; the pointer (0.19108) is UNMOVED.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

# BT.601 coefficients as written in upstream/frame_utils.py:60-62.
_KYR, _KYG, _KYB = 0.299, 0.587, 0.114
_ELL = np.array([_KYR, _KYG, _KYB])
_U = np.array([-_KYR, -_KYG, 1.0 - _KYB]) / 1.772   # (B - Y)/1.772
_V = np.array([1.0 - _KYR, -_KYG, -_KYB]) / 1.402   # (R - Y)/1.402
_ONES = np.array([1.0, 1.0, 1.0])

# Values MEASURED from these exact covectors (fp64), quoted in the #564 memo §4.
_ELL_DOT_U = -0.18790406320541758
_ELL_DOT_V = -0.10553922967189727
_PRINCIPAL_ANGLE_DEG = 30.27914784
_PROJECTOR_SPECTRAL_NORM = 0.504213367


def _orthonormal_basis(cols: np.ndarray) -> np.ndarray:
    """Orthonormal column basis of the span of the given column vectors."""
    q, _ = np.linalg.qr(cols)
    return q


def _ker_ell() -> np.ndarray:
    """Orthonormal basis (columns) of the null space of the 1x3 row ell."""
    _, _, vt = np.linalg.svd(_ELL.reshape(1, 3))
    return vt[1:].T  # rows 1,2 of V^T span the null space


def _find_frame_utils() -> Path | None:
    """Locate the pinned upstream/frame_utils.py (absent in git worktrees)."""
    here = Path(__file__).resolve()
    for base in (*here.parents, Path.cwd(), Path.cwd().parent):
        cand = base / "upstream" / "frame_utils.py"
        if cand.is_file():
            return cand
    return None


def test_source_coefficients_match_frozen_frame_utils() -> None:
    """Guard against drift: the hardcoded BT.601 coefficients must appear in source."""
    src = _find_frame_utils()
    if src is None:
        pytest.skip("pinned upstream/frame_utils.py not present (e.g. git worktree)")
    text = src.read_text()
    assert "kYR, kYG, kYB = 0.299, 0.587, 0.114" in text
    assert "(B - Y) / 1.772" in text
    assert "(R - Y) / 1.402" in text


def test_u_v_span_the_gray_orthogonal_plane_not_ker_ell() -> None:
    # span{u,v} = (1,1,1)^perp: both covectors annihilate the gray axis.
    assert abs(float(_U @ _ONES)) < 1e-12
    assert abs(float(_V @ _ONES)) < 1e-12
    # ...but ell is NOT orthogonal to u or v, so span{u,v} != ker(ell).
    # (fp64 tolerance: association order differs by ~1 ULP from the memo's value.)
    assert abs(float(_ELL @ _U) - _ELL_DOT_U) < 1e-15
    assert abs(float(_ELL @ _V) - _ELL_DOT_V) < 1e-15
    assert abs(float(_ELL @ _U)) > 1e-3
    assert abs(float(_ELL @ _V)) > 1e-3


def test_full_rank_ell_u_v() -> None:
    # ell, u, v are linearly independent (rank 3): the three planes are distinct.
    assert np.linalg.matrix_rank(np.array([_ELL, _U, _V])) == 3


def test_principal_angle_and_projector_distance() -> None:
    buv = _orthonormal_basis(np.array([_U, _V]).T)
    kell = _ker_ell()
    # Nonzero principal angle between span{u,v} and ker(ell).
    s = np.clip(np.linalg.svd(buv.T @ kell, compute_uv=False), -1.0, 1.0)
    angles_deg = np.degrees(np.arccos(s))
    assert angles_deg.min() < 1e-6          # they share one direction
    assert abs(float(angles_deg.max()) - _PRINCIPAL_ANGLE_DEG) < 1e-6
    # Spectral norm of the projector difference (worst-case unit-sensitivity error).
    p_uv = buv @ buv.T
    p_k = kell @ kell.T
    top = float(np.linalg.svd(p_uv - p_k, compute_uv=False)[0])
    assert abs(top - _PROJECTOR_SPECTRAL_NORM) < 1e-6


def test_primal_luma_preserving_basis_annihilated_by_ell() -> None:
    # A primal luma-preserving displacement must satisfy ell . delta_rgb = 0.
    d1 = np.array([1.0, -_KYR / _KYG, 0.0])
    d2 = np.array([0.0, -_KYB / _KYG, 1.0])
    assert abs(float(_ELL @ d1)) < 1e-12
    assert abs(float(_ELL @ d2)) < 1e-12
    # This primal basis is DISTINCT from the analysis covectors u, v.
    assert np.linalg.matrix_rank(np.array([d1, d2, _U])) == 3
