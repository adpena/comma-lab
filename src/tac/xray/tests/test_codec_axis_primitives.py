"""Tests for codec-axis xray primitives: F5/F6/F10."""

from __future__ import annotations

import math

import pytest
import torch

from tac.xray.base import XRayPrimitive
from tac.xray.vq_codebook_coverage import (
    VQCodebookCoverage,
    VQCoverageReport,
)
from tac.xray.wavelet_hf_energy import (
    HFEnergyReport,
    WaveletHFEnergy,
)
from tac.xray.yuv6_sublattice_geometry import (
    BT601_R,
    BT601_U,
    BT601_V,
    LUMA_SUBLATTICE_INDICES,
    YUV6SublatticeGeometry,
    YUV6SublatticeReport,
)


# ── F5 VQCodebookCoverage ────────────────────────────────────────────────


def test_vq_protocol():
    assert isinstance(VQCodebookCoverage(), XRayPrimitive)


def test_vq_name():
    assert VQCodebookCoverage().name == "vq_codebook_coverage"


def test_vq_hooks():
    h = VQCodebookCoverage().wire_in_hooks
    assert "bit_allocator" in h


def test_vq_report_rejects_non_positive_codebook():
    with pytest.raises(ValueError, match="codebook_size"):
        VQCoverageReport(
            codebook_size=0,
            patch_dim=16,
            n_patches_evaluated=10,
            coverage_fraction=0.5,
            mean_quantization_distance=0.1,
            codebook_index_entropy_bits=5.0,
            codebook_byte_budget_lower_bound=100,
        )


def test_vq_report_rejects_coverage_out_of_range():
    with pytest.raises(ValueError, match="coverage_fraction"):
        VQCoverageReport(
            codebook_size=4,
            patch_dim=16,
            n_patches_evaluated=10,
            coverage_fraction=1.5,
            mean_quantization_distance=0.1,
            codebook_index_entropy_bits=2.0,
            codebook_byte_budget_lower_bound=64,
        )


def test_vq_compute_identical_codebook_full_coverage():
    """When the patches are themselves codebook entries, coverage is 1.0
    and the nearest-distance is 0."""
    torch.manual_seed(0)
    codebook = torch.randn(8, 16)
    # Use a subset of codebook as patches.
    patches = codebook[:4].clone()
    result = VQCodebookCoverage().compute(
        target=patches, codebook=codebook, coverage_tolerance=1e-6
    )
    r = result.primitive_value
    assert r.coverage_fraction == 1.0
    assert r.mean_quantization_distance < 1e-5


def test_vq_compute_high_tolerance_full_coverage():
    """With a huge tolerance, any random patches are 'covered'."""
    torch.manual_seed(1)
    codebook = torch.randn(8, 16)
    patches = torch.randn(20, 16)
    result = VQCodebookCoverage().compute(
        target=patches, codebook=codebook, coverage_tolerance=1e6
    )
    assert result.primitive_value.coverage_fraction == 1.0


def test_vq_compute_codebook_byte_budget():
    """Byte budget = K * patch_dim * bytes_per_entry."""
    torch.manual_seed(2)
    codebook = torch.randn(4, 16)
    patches = torch.randn(5, 16)
    result = VQCodebookCoverage().compute(
        target=patches,
        codebook=codebook,
        bytes_per_codebook_entry=2,
    )
    r = result.primitive_value
    assert r.codebook_byte_budget_lower_bound == 4 * 16 * 2


def test_vq_compute_refuses_dim_mismatch():
    patches = torch.randn(5, 10)
    codebook = torch.randn(4, 16)
    with pytest.raises(ValueError, match="patch_dim"):
        VQCodebookCoverage().compute(target=patches, codebook=codebook)


def test_vq_compute_refuses_nondim2_target():
    patches = torch.randn(5, 10, 2)
    codebook = torch.randn(4, 16)
    with pytest.raises(ValueError, match="target must be 2-D"):
        VQCodebookCoverage().compute(target=patches, codebook=codebook)


def test_vq_compute_refuses_nondim2_codebook():
    patches = torch.randn(5, 10)
    codebook = torch.randn(4, 10, 2)
    with pytest.raises(ValueError, match="codebook must be 2-D"):
        VQCodebookCoverage().compute(target=patches, codebook=codebook)


def test_vq_compute_entropy_uniform_at_log2k():
    """When patches map uniformly across K=4 entries, entropy = log2(4) = 2."""
    torch.manual_seed(3)
    codebook = torch.randn(4, 16)
    # Construct 4 patches each near one codebook entry.
    patches = codebook.clone()
    result = VQCodebookCoverage().compute(
        target=patches, codebook=codebook, coverage_tolerance=1e-3
    )
    # Uniform assignment over 4 -> entropy = log2(4) = 2.0.
    assert result.primitive_value.codebook_index_entropy_bits == pytest.approx(
        2.0, abs=0.01
    )


def test_vq_compute_returns_envelope():
    torch.manual_seed(4)
    codebook = torch.randn(4, 8)
    patches = torch.randn(10, 8)
    result = VQCodebookCoverage().compute(target=patches, codebook=codebook)
    assert result.primitive_name == "vq_codebook_coverage"
    assert result.evidence_grade == "council-deliberation"


# ── F6 WaveletHFEnergy ──────────────────────────────────────────────────


def test_wavelet_protocol():
    assert isinstance(WaveletHFEnergy(), XRayPrimitive)


def test_wavelet_name():
    assert WaveletHFEnergy().name == "wavelet_hf_energy"


def test_wavelet_hooks():
    h = WaveletHFEnergy().wire_in_hooks
    assert "bit_allocator" in h
    assert "sensitivity_map" in h


def test_wavelet_constant_frame_zero_hf():
    """A constant frame has ZERO HF energy."""
    x = torch.full((1, 1, 32, 32), 5.0)
    result = WaveletHFEnergy().compute(target=x, n_levels=2)
    r = result.primitive_value
    assert r.total_hf_energy == pytest.approx(0.0, abs=1e-6)


def test_wavelet_random_frame_nonzero_hf():
    torch.manual_seed(0xDEAD)
    x = torch.randn(1, 1, 32, 32)
    result = WaveletHFEnergy().compute(target=x, n_levels=2)
    r = result.primitive_value
    assert r.total_hf_energy > 0


def test_wavelet_2d_input_accepted():
    x = torch.randn(16, 16)
    result = WaveletHFEnergy().compute(target=x, n_levels=1)
    assert result.primitive_value.n_levels >= 1


def test_wavelet_3d_input_accepted():
    x = torch.randn(1, 16, 16)
    result = WaveletHFEnergy().compute(target=x, n_levels=1)
    assert result.primitive_value.n_levels >= 1


def test_wavelet_rejects_5d():
    x = torch.randn(1, 1, 1, 16, 16)
    with pytest.raises(ValueError, match="must be 2-D, 3-D, or 4-D"):
        WaveletHFEnergy().compute(target=x)


def test_wavelet_rejects_zero_levels():
    x = torch.randn(16, 16)
    with pytest.raises(ValueError, match="n_levels"):
        WaveletHFEnergy().compute(target=x, n_levels=0)


def test_wavelet_fraction_above_threshold_in_range():
    torch.manual_seed(0xDEAD)
    x = torch.randn(1, 1, 32, 32)
    result = WaveletHFEnergy().compute(
        target=x, coefficient_threshold=0.1
    )
    assert 0.0 <= result.primitive_value.fraction_above_threshold <= 1.0


def test_wavelet_returns_envelope():
    x = torch.randn(8, 8)
    result = WaveletHFEnergy().compute(target=x)
    assert result.primitive_name == "wavelet_hf_energy"
    assert result.evidence_grade == "council-deliberation"


def test_wavelet_hf_fraction_bounded_above_one():
    """Edge case: numerical pathological frames cannot exceed 1.0."""
    x = torch.randn(1, 1, 64, 64) * 100
    result = WaveletHFEnergy().compute(target=x)
    assert result.primitive_value.hf_energy_fraction <= 1.0


def test_wavelet_records_chosen_basis():
    x = torch.randn(8, 8)
    result = WaveletHFEnergy().compute(target=x, wavelet="db8")
    # We don't implement db8 natively but we record the requested wavelet.
    assert result.primitive_value.wavelet == "db8"


# ── F10 YUV6SublatticeGeometry ──────────────────────────────────────────


def test_yuv6_protocol():
    assert isinstance(YUV6SublatticeGeometry(), XRayPrimitive)


def test_yuv6_name():
    assert YUV6SublatticeGeometry().name == "yuv6_sublattice_geometry"


def test_yuv6_hooks():
    h = YUV6SublatticeGeometry().wire_in_hooks
    assert "sensitivity_map" in h
    assert "bit_allocator" in h


def test_yuv6_constants_pinned_to_bt601():
    """BT.601 coefficients match upstream/frame_utils.py."""
    assert BT601_R == (0.299, 0.587, 0.114)
    assert BT601_U == (-0.168736, -0.331264, 0.5)
    assert BT601_V == (0.5, -0.418688, -0.081312)


def test_yuv6_sublattice_indices_cover_four_quadrants():
    """The 4 luma sublattices correspond to the 4 (y%2, x%2) indices."""
    assert LUMA_SUBLATTICE_INDICES == ((0, 0), (1, 0), (0, 1), (1, 1))


def test_yuv6_compute_rgb_3d_input():
    """Accept (3, H, W) input shape."""
    x = torch.rand(3, 64, 64)
    result = YUV6SublatticeGeometry().compute(target=x)
    r = result.primitive_value
    assert r.rgb_shape == (3, 64, 64)
    assert r.yuv6_shape == (6, 32, 32)


def test_yuv6_compute_rgb_4d_batch_uses_first():
    x = torch.rand(2, 3, 64, 64)
    result = YUV6SublatticeGeometry().compute(target=x)
    assert result.primitive_value.rgb_shape == (3, 64, 64)


def test_yuv6_compute_refuses_non_rgb_channels():
    x = torch.rand(4, 64, 64)
    with pytest.raises(ValueError, match="first dim must be 3"):
        YUV6SublatticeGeometry().compute(target=x)


def test_yuv6_compute_refuses_wrong_dim():
    x = torch.rand(64, 64)  # 2-D
    with pytest.raises(ValueError, match="must be"):
        YUV6SublatticeGeometry().compute(target=x)


def test_yuv6_compute_odd_dims_cropped():
    """Odd H/W is cropped to even-divisible size."""
    x = torch.rand(3, 65, 67)
    result = YUV6SublatticeGeometry().compute(target=x)
    # Cropped to (3, 64, 66) -> yuv6 shape (6, 32, 33).
    assert result.primitive_value.rgb_shape == (3, 64, 66)


def test_yuv6_compute_sublattice_fractions_sum_to_one():
    torch.manual_seed(0)
    x = torch.rand(3, 64, 64)
    result = YUV6SublatticeGeometry().compute(target=x)
    total = sum(result.primitive_value.per_sublattice_fraction)
    assert total == pytest.approx(1.0, abs=1e-6)


def test_yuv6_project_to_sublattice_returns_half_resolution():
    x = torch.rand(3, 64, 64)
    primitive = YUV6SublatticeGeometry()
    for idx in range(4):
        sublattice = primitive.project_to_sublattice(x, idx)
        assert sublattice.shape == (32, 32)


def test_yuv6_project_to_sublattice_rejects_invalid_idx():
    x = torch.rand(3, 64, 64)
    primitive = YUV6SublatticeGeometry()
    with pytest.raises(ValueError, match="sublattice_idx"):
        primitive.project_to_sublattice(x, 4)


def test_yuv6_compute_returns_envelope():
    x = torch.rand(3, 32, 32)
    result = YUV6SublatticeGeometry().compute(target=x)
    assert result.primitive_name == "yuv6_sublattice_geometry"
    assert result.evidence_grade == "structural-code-contract"
    assert "bilinear_resize_nullspace" in result.composes_with


def test_yuv6_metadata_records_bt601():
    x = torch.rand(3, 32, 32)
    result = YUV6SublatticeGeometry().compute(target=x)
    assert result.metadata["bt601_y_coefficients"] == BT601_R


def test_yuv6_zero_luma_falls_back_to_uniform_fraction():
    """When R+G+B = 0, sublattice_fractions defaults to (0.25, 0.25, 0.25, 0.25)."""
    x = torch.zeros(3, 32, 32)
    result = YUV6SublatticeGeometry().compute(target=x)
    assert result.primitive_value.per_sublattice_fraction == (0.25, 0.25, 0.25, 0.25)


# ── Compose F5 + F6 ─────────────────────────────────────────────────────


def test_compose_vq_with_wavelet():
    """Composition of VQ + wavelet should union their hook sets."""
    vq = VQCodebookCoverage()
    wavelet = WaveletHFEnergy()
    composed = vq.compose_with(wavelet)
    assert "bit_allocator" in composed.wire_in_hooks
    assert "sensitivity_map" in composed.wire_in_hooks
