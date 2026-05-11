"""Tests for `tac.residual_basis.numpy_inverse_dwt`.

PyWavelets is used as the round-trip ORACLE only. The module-under-test
imports ONLY numpy + stdlib.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import pywt  # ORACLE only — never imported by the module-under-test.

from tac.residual_basis.numpy_inverse_dwt import (
    NumpyInverseDWTError,
    haar_inverse_2d_multi_level,
    haar_inverse_2d_single_level,
)


# ---------------------------------------------------------------------------
# Module isolation: confirm numpy_inverse_dwt does NOT import pywt.
# ---------------------------------------------------------------------------


def test_module_does_not_import_pywavelets() -> None:
    """Per HNeRV parity discipline lesson 3 (≤2 ext deps; PyWavelets forbidden)."""

    src = (
        __import__("tac.residual_basis.numpy_inverse_dwt", fromlist=["__file__"]).__file__
    )
    text = open(src, encoding="utf-8").read()
    # Allow the substring "pywt" inside docstrings/comments referring to the oracle,
    # but NOT as an actual import.
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("import pywt") or stripped.startswith("from pywt"):
            raise AssertionError(f"numpy_inverse_dwt imports pywt: {stripped!r}")


def test_module_loc_within_budget() -> None:
    """Functional LOC <= 80 per HNeRV parity discipline lesson 3."""
    import ast

    src_path = (
        __import__("tac.residual_basis.numpy_inverse_dwt", fromlist=["__file__"]).__file__
    )
    src = open(src_path, encoding="utf-8").read()
    tree = ast.parse(src)
    lines = src.splitlines()
    docstring_lines: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            body = getattr(node, "body", [])
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                for ln in range(body[0].lineno, body[0].end_lineno + 1):
                    docstring_lines.add(ln)
    functional = 0
    for idx, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if idx in docstring_lines:
            continue
        functional += 1
    assert functional <= 80, f"functional LOC {functional} exceeds budget 80"


# ---------------------------------------------------------------------------
# Single-level 2D Haar synthesis: oracle parity.
# ---------------------------------------------------------------------------


def test_single_level_round_trip_against_pywt_haar_small() -> None:
    """`haar_inverse_2d_single_level()` matches pywt 'haar' synthesis byte-for-byte."""

    rng = np.random.default_rng(seed=0xC0FFEE)
    img = rng.standard_normal((8, 8)).astype(np.float64)
    cA, (cH, cV, cD) = pywt.dwt2(img, "haar")
    ours = haar_inverse_2d_single_level(cA, cH, cV, cD)
    theirs = pywt.idwt2((cA, (cH, cV, cD)), "haar")
    np.testing.assert_allclose(ours, theirs, atol=1e-12, rtol=1e-12)
    # And vs original.
    np.testing.assert_allclose(ours, img, atol=1e-12, rtol=1e-12)


def test_single_level_round_trip_non_square() -> None:
    """Non-square shapes round-trip correctly."""

    rng = np.random.default_rng(seed=42)
    img = rng.standard_normal((6, 10)).astype(np.float64)
    cA, (cH, cV, cD) = pywt.dwt2(img, "haar")
    ours = haar_inverse_2d_single_level(cA, cH, cV, cD)
    np.testing.assert_allclose(ours, img, atol=1e-12, rtol=1e-12)


def test_single_level_round_trip_float32() -> None:
    """Float32 inputs round-trip within float32 precision."""

    rng = np.random.default_rng(seed=7)
    img = rng.standard_normal((4, 4)).astype(np.float32)
    cA, (cH, cV, cD) = pywt.dwt2(img.astype(np.float64), "haar")
    cA = cA.astype(np.float32)
    cH = cH.astype(np.float32)
    cV = cV.astype(np.float32)
    cD = cD.astype(np.float32)
    ours = haar_inverse_2d_single_level(cA, cH, cV, cD)
    assert ours.dtype == np.float32
    np.testing.assert_allclose(ours, img, atol=1e-5, rtol=1e-5)


def test_single_level_zero_input_zero_output() -> None:
    """All-zero bands -> all-zero reconstruction."""

    z = np.zeros((4, 4), dtype=np.float64)
    out = haar_inverse_2d_single_level(z, z, z, z)
    assert np.all(out == 0)
    assert out.shape == (8, 8)


def test_single_level_shape_mismatch_raises() -> None:
    """Mismatched band shapes raise."""

    a = np.zeros((4, 4), dtype=np.float64)
    b = np.zeros((4, 5), dtype=np.float64)
    with pytest.raises(NumpyInverseDWTError, match="band shape mismatch"):
        haar_inverse_2d_single_level(a, b, a, a)


def test_single_level_wrong_ndim_raises() -> None:
    """1D bands raise."""

    a = np.zeros(8, dtype=np.float64)
    with pytest.raises(NumpyInverseDWTError, match="expected 2D bands"):
        haar_inverse_2d_single_level(a, a, a, a)


def test_single_level_integer_dtype_raises() -> None:
    """Integer bands raise."""

    a = np.zeros((4, 4), dtype=np.int32)
    with pytest.raises(NumpyInverseDWTError, match="expected floating dtype"):
        haar_inverse_2d_single_level(a, a, a, a)


def test_single_level_output_shape_doubles_each_axis() -> None:
    """Output shape is exactly (2H, 2W)."""

    z = np.zeros((3, 7), dtype=np.float64)
    out = haar_inverse_2d_single_level(z, z, z, z)
    assert out.shape == (6, 14)


def test_single_level_known_impulse_top_left() -> None:
    """LL impulse + zero detail -> uniform reconstruction (LL only)."""

    ll = np.array([[2.0]])
    z = np.array([[0.0]])
    out = haar_inverse_2d_single_level(ll, z, z, z)
    # Per the Haar normalization 0.5 * (2 + 0 + 0 + 0) = 1.0 at all 4 output pixels.
    expected = np.full((2, 2), 1.0)
    np.testing.assert_allclose(out, expected, atol=1e-15)


# ---------------------------------------------------------------------------
# Multi-level synthesis: pywt.waverec2 parity at levels 1, 2, 3.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("level", [1, 2, 3])
def test_multi_level_round_trip_against_pywt(level: int) -> None:
    """N-level Haar round-trip vs pywt at multiple depths."""

    side = 2 ** (level + 2)  # 8, 16, 32 — divisible enough at every level
    rng = np.random.default_rng(seed=0xDEADBEEF + level)
    img = rng.standard_normal((side, side)).astype(np.float64)
    coeffs = pywt.wavedec2(img, "haar", level=level)
    ours = haar_inverse_2d_multi_level(list(coeffs))
    theirs = pywt.waverec2(coeffs, "haar")
    np.testing.assert_allclose(ours, theirs, atol=1e-11, rtol=1e-11)
    np.testing.assert_allclose(ours, img, atol=1e-11, rtol=1e-11)


def test_multi_level_pr106_camera_resolution_subblock() -> None:
    """At a PR106-camera-resolution-divisible subblock, round-trip holds."""

    # 874 not power-of-2 — for now test at 512x768 (divisible by 8 = 2^3).
    rng = np.random.default_rng(seed=2026)
    img = rng.standard_normal((512, 768)).astype(np.float64)
    coeffs = pywt.wavedec2(img, "haar", level=3)
    ours = haar_inverse_2d_multi_level(list(coeffs))
    np.testing.assert_allclose(ours, img, atol=1e-10, rtol=1e-10)


def test_multi_level_too_few_coeffs_raises() -> None:
    """Single-element coeffs (no detail tuples) raises."""

    cA = np.zeros((4, 4), dtype=np.float64)
    with pytest.raises(NumpyInverseDWTError, match="coeffs must have len >= 2"):
        haar_inverse_2d_multi_level([cA])


def test_multi_level_bad_detail_shape_raises() -> None:
    """Detail must be a 3-tuple; 2-tuple raises with the level index."""

    cA = np.zeros((4, 4), dtype=np.float64)
    z = np.zeros((4, 4), dtype=np.float64)
    with pytest.raises(NumpyInverseDWTError, match="must be a 3-tuple"):
        haar_inverse_2d_multi_level([cA, (z, z)])  # type: ignore[list-item]


def test_multi_level_first_coeff_must_be_ndarray() -> None:
    """First coeff (cA) must be ndarray, not tuple."""

    z = np.zeros((4, 4), dtype=np.float64)
    with pytest.raises(NumpyInverseDWTError, match="first coefficient must be a numpy ndarray"):
        haar_inverse_2d_multi_level([(z, z, z), (z, z, z)])  # type: ignore[list-item]


def test_multi_level_zero_round_trip() -> None:
    """All-zero coeffs at all levels -> all-zero reconstruction.

    Per pywt.wavedec2 layout: cA_L has the smallest shape; detail tuples grow
    by 2x per level. For a 2-level decomposition of a 16x16 input the layout is
    `cA: (4, 4)`, detail level 2: `(4, 4)`, detail level 1: `(8, 8)`.
    """

    z4 = np.zeros((4, 4), dtype=np.float64)
    z8 = np.zeros((8, 8), dtype=np.float64)
    out = haar_inverse_2d_multi_level([z4, (z4, z4, z4), (z8, z8, z8)])
    assert out.shape == (16, 16)
    assert np.all(out == 0)


def test_multi_level_preserves_dtype_float32() -> None:
    """float32 multi-level round-trip preserves dtype."""

    rng = np.random.default_rng(seed=1)
    img = rng.standard_normal((16, 16)).astype(np.float32)
    coeffs = pywt.wavedec2(img.astype(np.float64), "haar", level=2)
    coeffs32 = [coeffs[0].astype(np.float32)] + [
        tuple(c.astype(np.float32) for c in det) for det in coeffs[1:]
    ]
    ours = haar_inverse_2d_multi_level(coeffs32)
    assert ours.dtype == np.float32


def test_multi_level_smoke_recovers_known_constant() -> None:
    """Constant signal multi-level round-trip = constant signal."""

    img = np.full((8, 8), 5.0, dtype=np.float64)
    coeffs = pywt.wavedec2(img, "haar", level=2)
    ours = haar_inverse_2d_multi_level(list(coeffs))
    np.testing.assert_allclose(ours, img, atol=1e-12, rtol=1e-12)


# ---------------------------------------------------------------------------
# Promotion-status invariants (research-only by construction).
# ---------------------------------------------------------------------------


def test_promotion_status_invariants_documented() -> None:
    """The numpy_inverse_dwt module is a primitive — no result dataclass to
    promote. The promotion-status invariants (score_claim=False etc.) are
    enforced by the consumer modules (cool_chic_residual, c3_residual etc.)
    that USE this primitive. Here we just verify the module docstring names
    the constraint."""

    import tac.residual_basis.numpy_inverse_dwt as m

    docstring = m.__doc__ or ""
    assert "contest runtime dep" in docstring.lower(), (
        "module docstring must name the runtime-dep blocker it clears"
    )
    assert "loc budget" in docstring.lower(), (
        "module docstring must declare LOC budget compliance"
    )


# ---------------------------------------------------------------------------
# Numerical stability spot-checks.
# ---------------------------------------------------------------------------


def test_extreme_magnitudes_round_trip() -> None:
    """Very large + very small magnitudes still round-trip."""

    img = np.array([[1e10, -1e10, 1e-10, -1e-10]] * 4, dtype=np.float64)
    cA, (cH, cV, cD) = pywt.dwt2(img, "haar")
    ours = haar_inverse_2d_single_level(cA, cH, cV, cD)
    np.testing.assert_allclose(ours, img, atol=1e-5, rtol=1e-15)


def test_negative_values_round_trip() -> None:
    """Negative signal values round-trip correctly."""

    rng = np.random.default_rng(seed=9)
    img = -rng.standard_normal((8, 8)).astype(np.float64) - 5.0
    cA, (cH, cV, cD) = pywt.dwt2(img, "haar")
    ours = haar_inverse_2d_single_level(cA, cH, cV, cD)
    np.testing.assert_allclose(ours, img, atol=1e-12, rtol=1e-12)


def test_haar_norm_is_one_half() -> None:
    """The internal `_HAAR_NORM` constant is exactly 0.5 (not 1/sqrt(2)).

    The orthonormal 2D synthesis uses 1/sqrt(2) per 1D axis -> 1/2 combined.
    """

    from tac.residual_basis.numpy_inverse_dwt import _HAAR_NORM

    assert _HAAR_NORM == 0.5
    # Sanity: that's also 1/sqrt(2) * 1/sqrt(2).
    assert math.isclose(_HAAR_NORM, (1.0 / math.sqrt(2.0)) ** 2, rel_tol=1e-15)
