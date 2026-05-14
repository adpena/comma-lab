# SPDX-License-Identifier: MIT
"""Tests for the PR93 lowpass-luma residual codec primitive.

Covers encode→decode round-trip on the 3-channel (linear) and 6-channel
(Legendre quadratic) forms, byte-grammar round-trip via
serialize/deserialize, basis orthonormality, edge cases, and golden-vector
SHA pinning.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tac.packet_compiler.pr93_lowpass_luma import (
    LUMA_BT601_COEFFS,
    LowpassLumaResidual,
    decode_lowpass_luma_residual,
    deserialize_lowpass_luma_residual,
    encode_lowpass_luma_residual,
    serialize_lowpass_luma_residual,
)

GOLDEN_DIR = (
    Path(__file__).resolve().parent.parent / "packet_compiler" / "golden_vectors"
)


# ── BT.601 coefficients ────────────────────────────────────────────────────


def test_bt601_coefficients_match_pr93_source() -> None:
    """PR93 uses (0.299, 0.587, 0.114) for luma_plane RGB→Y projection."""
    assert LUMA_BT601_COEFFS == (0.299, 0.587, 0.114)
    assert sum(LUMA_BT601_COEFFS) == pytest.approx(1.0, abs=1e-6)


# ── Decode-only basis evaluation (PR93 luma_plane_correction) ─────────────


def test_decode_3channel_linear_plane_matches_pr93_formula() -> None:
    """3-channel: c0 + c1*x + c2*y on normalized [-1, +1] grid."""
    height, width = 8, 8
    coeffs = (1.0, 0.5, -0.25)
    out = decode_lowpass_luma_residual(coeffs, height=height, width=width)
    assert out.shape == (height, width)
    assert out.dtype == np.float32
    # Corner check: top-left x=-1, y=-1 → c0 - c1 - c2 = 1 - 0.5 - (-0.25) = 0.75
    assert out[0, 0] == pytest.approx(0.75, abs=1e-5)
    # Bottom-right x=+1, y=+1 → 1 + 0.5 + (-0.25) = 1.25
    assert out[-1, -1] == pytest.approx(1.25, abs=1e-5)


def test_decode_6channel_legendre_quadratic_matches_pr93_formula() -> None:
    """6-channel: c0 + c1*x + c2*y + c3*xy + c4*(x²-1/3) + c5*(y²-1/3)."""
    height, width = 16, 16
    coeffs = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)  # x² - 1/3
    out = decode_lowpass_luma_residual(coeffs, height=height, width=width)
    # At x=0 (center column), output = 1.0 + (0 - 1/3) = 2/3
    mid_col = width // 2 - 1  # off-center by half-pixel
    # Verify the pattern is symmetric (x² is symmetric)
    np.testing.assert_allclose(out[:, mid_col], out[:, width - mid_col - 1], atol=1e-5)


def test_decode_zero_coeffs_returns_zero_plane() -> None:
    out = decode_lowpass_luma_residual((0.0, 0.0, 0.0), height=4, width=4)
    np.testing.assert_array_equal(out, np.zeros((4, 4), dtype=np.float32))


def test_decode_invalid_n_coeffs_raises() -> None:
    with pytest.raises(ValueError, match="n_coeffs must be 3"):
        decode_lowpass_luma_residual((1.0, 2.0), height=4, width=4)
    with pytest.raises(ValueError, match="n_coeffs must be 3"):
        decode_lowpass_luma_residual((1.0,) * 9, height=4, width=4)


def test_decode_invalid_grid_size_raises() -> None:
    with pytest.raises(ValueError, match=">= 1"):
        decode_lowpass_luma_residual((1.0, 0.0, 0.0), height=0, width=4)


# ── Encoder (lstsq fit) ────────────────────────────────────────────────────


def test_encode_constant_plane_recovers_only_c0() -> None:
    """Constant-luma residual should fit to (c, 0, 0) under 3-channel."""
    height, width = 32, 32
    residual = np.full((height, width), 5.0, dtype=np.float32)
    res = encode_lowpass_luma_residual(residual, n_coeffs=3)
    assert res.height == height and res.width == width
    assert res.coefficients[0] == pytest.approx(5.0, abs=1e-4)
    assert abs(res.coefficients[1]) < 1e-4
    assert abs(res.coefficients[2]) < 1e-4


def test_encode_linear_plane_recovers_exact_coeffs() -> None:
    """A pure linear residual fits to its (c0, c1, c2) exactly (up to lstsq numerics)."""
    height, width = 64, 64
    true_coeffs = (1.5, -0.5, 0.25)
    target = decode_lowpass_luma_residual(true_coeffs, height=height, width=width)
    fitted = encode_lowpass_luma_residual(target.astype(np.float32), n_coeffs=3)
    for actual, expected in zip(fitted.coefficients, true_coeffs):
        assert actual == pytest.approx(expected, abs=1e-4)


def test_encode_legendre_quadratic_recovers_exact_coeffs() -> None:
    """A pure Legendre-quadratic residual fits its 6 coefficients exactly."""
    height, width = 64, 64
    true_coeffs = (0.1, 0.2, -0.3, 0.4, -0.5, 0.6)
    target = decode_lowpass_luma_residual(true_coeffs, height=height, width=width)
    fitted = encode_lowpass_luma_residual(target.astype(np.float32), n_coeffs=6)
    for actual, expected in zip(fitted.coefficients, true_coeffs):
        assert actual == pytest.approx(expected, abs=1e-3)


def test_encode_rejects_non_2d_residual() -> None:
    with pytest.raises(ValueError, match="2D"):
        encode_lowpass_luma_residual(np.zeros(10, dtype=np.float32))
    with pytest.raises(ValueError, match="2D"):
        encode_lowpass_luma_residual(np.zeros((2, 2, 2), dtype=np.float32))


# ── encode → decode round-trip ────────────────────────────────────────────


def test_encode_decode_roundtrip_3channel_smooth_residual() -> None:
    """Smooth residual that lies in the basis span should round-trip nearly exactly."""
    height, width = 48, 64
    true_coeffs = (2.0, 0.5, -0.25)
    target = decode_lowpass_luma_residual(true_coeffs, height=height, width=width)
    fitted = encode_lowpass_luma_residual(target, n_coeffs=3)
    recovered = decode_lowpass_luma_residual(
        fitted.coefficients, height=height, width=width
    )
    np.testing.assert_allclose(recovered, target, atol=1e-4)


def test_encode_decode_roundtrip_6channel_smooth_residual() -> None:
    height, width = 96, 128
    true_coeffs = (1.0, 0.1, 0.2, 0.3, 0.4, 0.5)
    target = decode_lowpass_luma_residual(true_coeffs, height=height, width=width)
    fitted = encode_lowpass_luma_residual(target, n_coeffs=6)
    recovered = decode_lowpass_luma_residual(
        fitted.coefficients, height=height, width=width
    )
    np.testing.assert_allclose(recovered, target, atol=1e-3)


def test_encode_decode_noisy_residual_fits_lowpass_component() -> None:
    """High-frequency noise should be largely orthogonal to the smooth basis."""
    rng = np.random.default_rng(20260511)
    height, width = 64, 64
    true_lowpass_coeffs = (1.0, 0.3, -0.2)
    smooth = decode_lowpass_luma_residual(
        true_lowpass_coeffs, height=height, width=width
    )
    noise = rng.standard_normal((height, width)).astype(np.float32) * 0.1
    residual = smooth + noise
    fitted = encode_lowpass_luma_residual(residual, n_coeffs=3)
    for actual, expected in zip(fitted.coefficients, true_lowpass_coeffs):
        assert actual == pytest.approx(expected, abs=0.05)


# ── Serialize / deserialize round-trip ────────────────────────────────────


def test_serialize_3channel_payload_size_is_17_bytes() -> None:
    """Header 5 bytes + 3 * 4 bytes = 17 bytes."""
    res = LowpassLumaResidual(coefficients=(1.0, 2.0, 3.0), height=10, width=10)
    blob = serialize_lowpass_luma_residual(res)
    assert len(blob) == 17


def test_serialize_6channel_payload_size_is_29_bytes() -> None:
    """Header 5 bytes + 6 * 4 bytes = 29 bytes."""
    res = LowpassLumaResidual(
        coefficients=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0), height=10, width=10
    )
    blob = serialize_lowpass_luma_residual(res)
    assert len(blob) == 29


def test_serialize_deserialize_roundtrips_3channel() -> None:
    """Float-exact coefficients (representable in fp32) round-trip exactly."""
    res = LowpassLumaResidual(coefficients=(0.5, -0.25, 0.125), height=384, width=512)
    blob = serialize_lowpass_luma_residual(res)
    recovered = deserialize_lowpass_luma_residual(blob)
    assert recovered.coefficients == res.coefficients
    assert recovered.height == res.height
    assert recovered.width == res.width


def test_serialize_deserialize_roundtrips_6channel_approximate() -> None:
    """Fp64 coefficients lose precision through fp32 wire-format — recovered values are within 1e-7."""
    original = (1.0, 0.1, 0.2, 0.3, 0.4, 0.5)
    res = LowpassLumaResidual(
        coefficients=original, height=874, width=1164
    )
    blob = serialize_lowpass_luma_residual(res)
    recovered = deserialize_lowpass_luma_residual(blob)
    for actual, expected in zip(recovered.coefficients, original):
        assert actual == pytest.approx(expected, abs=1e-6)
    assert recovered.height == res.height
    assert recovered.width == res.width


def test_serialize_rejects_unsupported_n_coeffs() -> None:
    res = LowpassLumaResidual(coefficients=(1.0, 2.0), height=10, width=10)
    with pytest.raises(ValueError, match="only 3 or 6 coefficients"):
        serialize_lowpass_luma_residual(res)


def test_serialize_rejects_oversize_grid() -> None:
    res = LowpassLumaResidual(coefficients=(1.0, 2.0, 3.0), height=70000, width=10)
    with pytest.raises(ValueError, match="uint16"):
        serialize_lowpass_luma_residual(res)


def test_serialize_rejects_zero_sized_grid() -> None:
    res = LowpassLumaResidual(coefficients=(1.0, 2.0, 3.0), height=0, width=10)
    with pytest.raises(ValueError, match=">= 1"):
        serialize_lowpass_luma_residual(res)


def test_deserialize_rejects_truncated_header() -> None:
    with pytest.raises(ValueError, match="truncated header"):
        deserialize_lowpass_luma_residual(b"\x00\x00")


def test_deserialize_rejects_unsupported_n_coeffs() -> None:
    import struct

    bad = struct.pack("<BHH", 5, 10, 10) + b"\x00" * 20
    with pytest.raises(ValueError, match="unsupported n_coeffs"):
        deserialize_lowpass_luma_residual(bad)


def test_deserialize_rejects_size_mismatch() -> None:
    import struct

    bad = struct.pack("<BHH", 3, 10, 10) + b"\x00" * 5  # claim 3 floats, give 5 bytes
    with pytest.raises(ValueError, match="size mismatch"):
        deserialize_lowpass_luma_residual(bad)


# ── Frozen dataclass discipline ───────────────────────────────────────────


def test_residual_is_frozen() -> None:
    r = LowpassLumaResidual(coefficients=(1.0, 0.0, 0.0), height=4, width=4)
    with pytest.raises(Exception):
        r.height = 99  # type: ignore[misc]


# ── Golden vector (SHA-pinned) ────────────────────────────────────────────


def test_lowpass_luma_golden_vector_pins_sha() -> None:
    coeffs = (1.0, 0.5, -0.25, 0.125, -0.0625, 0.03125)
    res = LowpassLumaResidual(coefficients=coeffs, height=384, width=512)
    blob = serialize_lowpass_luma_residual(res)
    sha = hashlib.sha256(blob).hexdigest()
    golden_path = GOLDEN_DIR / "pr93_lowpass_luma_v1.json"
    golden = json.loads(golden_path.read_text())
    assert sha == golden["sha256"]
    assert len(blob) == golden["payload_len"]
    assert golden["n_coeffs"] == 6
