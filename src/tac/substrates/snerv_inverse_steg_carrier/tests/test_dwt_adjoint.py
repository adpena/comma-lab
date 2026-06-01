# SPDX-License-Identifier: MIT
"""NO-FAKE tests for the SNeRV orthonormal DWT + the EXACT G3 adjoint.

Slot EEE Class 2 discipline: every test verifies ACTUAL transform behaviour
(perfect reconstruction, the orthonormal adjoint dot-product identity, the
pad-to-square squareness), NOT constants. If the synthesis stopped being the exact
adjoint, ``test_adjoint_dot_product_identity_native_dims`` FAILS.
"""

from __future__ import annotations

import numpy as np
import pytest

from tac.substrates.snerv_inverse_steg_carrier.dwt import (
    DEFAULT_WAVELET,
    SnervDwtError,
    WaveletPyramid,
    dwt2_multilevel,
    dwt2_native_synthesis_adjoint,
    idwt2_multilevel,
    lf_coeff_count,
    synthesis_adjoint_residual,
)


def _flatten_coeffs(coeffs: list) -> np.ndarray:
    parts = [np.asarray(coeffs[0], dtype=np.float64).ravel()]
    for lh, hl, hh in coeffs[1:]:
        parts.append(np.asarray(lh, dtype=np.float64).ravel())
        parts.append(np.asarray(hl, dtype=np.float64).ravel())
        parts.append(np.asarray(hh, dtype=np.float64).ravel())
    return np.concatenate(parts)


def _random_like_coeffs(coeffs: list, rng: np.random.Generator) -> list:
    out: list = [rng.standard_normal(np.asarray(coeffs[0]).shape)]
    for lh, hl, hh in coeffs[1:]:
        out.append(
            (
                rng.standard_normal(np.asarray(lh).shape),
                rng.standard_normal(np.asarray(hl).shape),
                rng.standard_normal(np.asarray(hh).shape),
            )
        )
    return out


@pytest.mark.parametrize("hw", [(64, 96), (384, 512), (874, 1164), (100, 150)])
def test_perfect_reconstruction(hw):
    """idwt2(dwt2(x)) == x to ~1e-14 (orthonormal perfect reconstruction)."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal(hw)
    pyr = dwt2_multilevel(x, levels=3)
    recon = idwt2_multilevel(pyr)
    assert recon.shape == hw  # cropped back to native
    assert np.abs(recon - x).max() < 1e-12


@pytest.mark.parametrize("hw", [(64, 96), (384, 512), (874, 1164), (100, 150)])
def test_adjoint_dot_product_identity_native_dims(hw):
    """The G3 claim: <S c, g> == <c, S^T g> at NATIVE dims (rel-residual ~0).

    This is the load-bearing exactness number. ``S`` is padded-canvas synthesis
    cropped back to native frame size, so ``S^T`` must zero-embed the native
    cotangent before DWT analysis. NO-FAKE: this is the LIVE dot-product test, not
    an asserted constant.
    """
    res = synthesis_adjoint_residual(hw, levels=3, seed=1)
    assert res < 1e-12, f"adjoint rel-residual {res} not floating-point zero at {hw}"


def test_native_crop_adjoint_matches_explicit_dot_product_odd_dims():
    """NO-FAKE: odd native dims use crop-aware zero-embed adjoint, not prose."""
    rng = np.random.default_rng(11)
    hw = (65, 97)
    levels = 3
    template = dwt2_multilevel(np.zeros(hw), levels=levels, wavelet=DEFAULT_WAVELET)
    coeffs = _random_like_coeffs(template.coeffs, rng)
    native_pixels = idwt2_multilevel(
        WaveletPyramid(
            coeffs=coeffs,
            levels=levels,
            wavelet=DEFAULT_WAVELET,
            orig_hw=hw,
            padded_hw=template.padded_hw,
        )
    )
    pixel_cotangent = rng.standard_normal(hw)
    adjoint_coeffs = dwt2_native_synthesis_adjoint(
        pixel_cotangent, levels=levels, wavelet=DEFAULT_WAVELET
    ).coeffs

    lhs = float(np.dot(native_pixels.ravel(), pixel_cotangent.ravel()))
    rhs = float(np.dot(_flatten_coeffs(coeffs), _flatten_coeffs(adjoint_coeffs)))
    assert abs(lhs - rhs) / (abs(lhs) + 1e-30) < 1e-12


def test_reflect_padded_analysis_is_not_native_crop_adjoint_on_odd_dims():
    """Regression guard for the prior G3 overclaim on odd native dimensions."""
    rng = np.random.default_rng(12)
    pixel_cotangent = rng.standard_normal((65, 97))
    native = dwt2_native_synthesis_adjoint(pixel_cotangent, levels=3).coeffs
    reflect = dwt2_multilevel(pixel_cotangent, levels=3).coeffs
    native_vec = _flatten_coeffs(native)
    reflect_vec = _flatten_coeffs(reflect)
    rel_diff = np.linalg.norm(native_vec - reflect_vec) / (np.linalg.norm(native_vec) + 1e-30)
    assert rel_diff > 1e-3


def test_pad_to_square_makes_transform_square():
    """On the padded canvas the coeff count equals padded H*W (square transform)."""
    rng = np.random.default_rng(2)
    x = rng.standard_normal((874, 1164))
    pyr = dwt2_multilevel(x, levels=3)
    ph, pw = pyr.padded_hw
    # square iff total coeff count == padded pixel count
    assert pyr.total_coeff_count() == ph * pw
    # padded dims are multiples of 2**levels
    assert ph % 8 == 0 and pw % 8 == 0
    # padding only grows (never shrinks) the native dims
    assert ph >= 874 and pw >= 1164


def test_native_already_square_no_padding():
    """When native dims are multiples of 2**levels, no padding is applied."""
    pyr = dwt2_multilevel(np.zeros((384, 512)), levels=3)
    assert pyr.padded_hw == (384, 512)


def test_lf_is_super_small_fraction_of_pixels():
    """SNeRV super-small-rate-by-design: LF is ~1/4**levels of the pixels.

    At level 3 the LF approximation holds ~1.5-1.6% of the pixels; at level 4
    ~0.4%. This is the structural rate lever (the rest is GENERATED).
    """
    n3 = lf_coeff_count((874, 1164), 3)
    n4 = lf_coeff_count((874, 1164), 4)
    frac3 = n3 / (874 * 1164)
    frac4 = n4 / (874 * 1164)
    assert 0.013 < frac3 < 0.020, f"level-3 LF frac {frac3} out of expected band"
    assert 0.002 < frac4 < 0.006, f"level-4 LF frac {frac4} out of expected band"
    assert n4 < n3  # coarser stores fewer


def test_lf_block_changes_when_input_changes_not_a_constant():
    """NO-FAKE: the LF coefficients ACTUALLY depend on the input (not a stub)."""
    rng = np.random.default_rng(3)
    a = dwt2_multilevel(rng.standard_normal((64, 96)), levels=3).lf
    b = dwt2_multilevel(rng.standard_normal((64, 96)), levels=3).lf
    assert a.shape == b.shape
    assert not np.allclose(a, b)  # different inputs -> different LF


def test_levels_validation():
    with pytest.raises(SnervDwtError):
        dwt2_multilevel(np.zeros((8, 8)), levels=0)
    with pytest.raises(SnervDwtError):
        dwt2_multilevel(np.zeros((4,)), levels=1)  # not 2D


def test_idwt_uses_synthesis_not_identity():
    """NO-FAKE: synthesis ACTUALLY inverts a non-trivial pyramid (not pass-through)."""
    rng = np.random.default_rng(4)
    x = rng.standard_normal((32, 48)) * 10.0
    pyr = dwt2_multilevel(x, levels=2)
    # zero the detail -> reconstruction must DIFFER from x (proves detail matters)
    coeffs_no_detail = [pyr.lf] + [
        (np.zeros_like(lh), np.zeros_like(hl), np.zeros_like(hh))
        for lh, hl, hh in pyr.details
    ]
    blurred = idwt2_multilevel(
        WaveletPyramid(
            coeffs=coeffs_no_detail, levels=2, wavelet=pyr.wavelet,
            orig_hw=pyr.orig_hw, padded_hw=pyr.padded_hw,
        )
    )
    full = idwt2_multilevel(pyr)
    # full reconstruction ~= x; detail-zeroed reconstruction differs (LF-only blur)
    assert np.abs(full - x).max() < 1e-10
    assert np.abs(blurred - x).max() > 1e-3  # detail carried real signal
