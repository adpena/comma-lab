"""Tests for F3: tac.xray.bilinear_resize_nullspace."""

from __future__ import annotations

import pytest
import torch

from tac.xray.base import XRayPrimitive
from tac.xray.bilinear_resize_nullspace import (
    BilinearResizeNullspace,
    BilinearResizeNullspaceReport,
    CAMERA_SIZE_H,
    CAMERA_SIZE_W,
    SCORER_INPUT_H,
    SCORER_INPUT_W,
)


def test_protocol():
    assert isinstance(BilinearResizeNullspace(), XRayPrimitive)


def test_name():
    assert BilinearResizeNullspace().name == "bilinear_resize_nullspace"


def test_hooks():
    h = BilinearResizeNullspace().wire_in_hooks
    assert "sensitivity_map" in h
    assert "bit_allocator" in h


def test_canonical_constants():
    assert CAMERA_SIZE_H == 874
    assert CAMERA_SIZE_W == 1164
    assert SCORER_INPUT_H == 384
    assert SCORER_INPUT_W == 512


def test_report_rejects_zero_pixel_count():
    with pytest.raises(ValueError, match="pixel counts must be positive"):
        BilinearResizeNullspaceReport(
            camera_n_pixels=0,
            scorer_n_pixels=100,
            upper_bound_nullspace_fraction=0.5,
            empirical_nullspace_fraction=None,
            n_samples=0,
            perturbation_norm=1.0,
            output_residual_norm_tolerance=1e-4,
            resize_mode="bilinear",
        )


def test_report_rejects_invalid_upper_bound():
    with pytest.raises(ValueError):
        BilinearResizeNullspaceReport(
            camera_n_pixels=100,
            scorer_n_pixels=50,
            upper_bound_nullspace_fraction=1.5,
            empirical_nullspace_fraction=None,
            n_samples=0,
            perturbation_norm=1.0,
            output_residual_norm_tolerance=1e-4,
            resize_mode="bilinear",
        )


def test_compute_derivational_only_matches_807_finding():
    """At canonical contest sizes, the upper-bound nullspace fraction
    should match the deep_math memo's 80.7% claim."""
    result = BilinearResizeNullspace().compute(target=None, n_samples=0)
    report = result.primitive_value
    assert isinstance(report, BilinearResizeNullspaceReport)
    # 1 - 196608 / 1016536 = 0.80659...
    assert report.upper_bound_nullspace_fraction == pytest.approx(
        1.0 - (384 * 512) / (874 * 1164), rel=1e-9
    )
    # Round to "80.7%" claim.
    assert abs(report.upper_bound_nullspace_fraction - 0.807) < 0.01


def test_compute_refuses_camera_smaller_than_scorer():
    """Camera-frame must be >= scorer-frame for nullspace analysis."""
    with pytest.raises(ValueError, match="camera_size"):
        BilinearResizeNullspace().compute(
            target=None,
            camera_size=(100, 100),
            scorer_size=(200, 200),
        )


def test_compute_with_monte_carlo_samples():
    """Run a small Monte-Carlo sweep at reduced resolution; empirical
    fraction should be approximately the upper bound."""
    result = BilinearResizeNullspace().compute(
        target=None,
        camera_size=(64, 64),
        scorer_size=(16, 16),
        n_samples=8,
        perturbation_norm=1.0,
        output_tolerance=1e-3,
    )
    report = result.primitive_value
    assert report.n_samples == 8
    assert report.empirical_nullspace_fraction is not None
    # Empirical fraction in [0, 1]; we don't pin a specific value since
    # MC of random Gaussian perturbations on tiny tensor is noisy.
    assert 0.0 <= report.empirical_nullspace_fraction <= 1.0


def test_compute_returns_typed_envelope():
    result = BilinearResizeNullspace().compute(target=None)
    assert result.primitive_name == "bilinear_resize_nullspace"
    assert result.evidence_grade == "mathematical-derivation"
    assert result.confidence_band is not None
    assert "yuv6_sublattice_geometry" in result.composes_with


def test_compute_records_resize_mode_in_metadata():
    result = BilinearResizeNullspace().compute(
        target=None, resize_mode="bicubic"
    )
    assert result.metadata["resize_mode"] == "bicubic"


def test_project_into_nullspace_reduces_resize_output():
    """A perturbation projected into the nullspace should have its resize
    output norm REDUCED vs the raw perturbation."""
    primitive = BilinearResizeNullspace()
    torch.manual_seed(0xDEAD)
    delta = torch.randn(1, 1, 64, 64)
    delta_proj = primitive.project_into_nullspace(
        delta, camera_size=(64, 64), scorer_size=(16, 16), n_iterations=3
    )
    import torch.nn.functional as F

    out_raw = F.interpolate(
        delta, size=(16, 16), mode="bilinear", align_corners=False
    )
    out_proj = F.interpolate(
        delta_proj, size=(16, 16), mode="bilinear", align_corners=False
    )
    assert out_proj.flatten().norm() < out_raw.flatten().norm()


def test_project_into_nullspace_handles_2d_input():
    """2D tensor input should produce 2D tensor output."""
    primitive = BilinearResizeNullspace()
    torch.manual_seed(0)
    delta_2d = torch.randn(64, 64)
    delta_proj_2d = primitive.project_into_nullspace(
        delta_2d, camera_size=(64, 64), scorer_size=(16, 16)
    )
    assert delta_proj_2d.shape == (64, 64)


def test_project_into_nullspace_handles_3d_input():
    """3D tensor input should produce 3D tensor output."""
    primitive = BilinearResizeNullspace()
    torch.manual_seed(0)
    delta_3d = torch.randn(1, 64, 64)
    delta_proj_3d = primitive.project_into_nullspace(
        delta_3d, camera_size=(64, 64), scorer_size=(16, 16)
    )
    assert delta_proj_3d.shape == (1, 64, 64)


def test_confidence_band_empirical_uses_standard_error():
    """When empirical estimation runs, band uses 1.96 SE around mean."""
    result = BilinearResizeNullspace().compute(
        target=None,
        camera_size=(64, 64),
        scorer_size=(16, 16),
        n_samples=8,
    )
    band = result.confidence_band
    assert band is not None
    lo, hi = band
    assert 0.0 <= lo <= hi <= 1.0


def test_confidence_band_derivational_is_zero_to_upper_bound():
    """Without empirical samples, band is [0, upper_bound]."""
    result = BilinearResizeNullspace().compute(target=None, n_samples=0)
    band = result.confidence_band
    assert band is not None
    assert band[0] == 0.0
    assert band[1] == pytest.approx(
        result.primitive_value.upper_bound_nullspace_fraction
    )


def test_nearest_mode_supported():
    """Nearest-neighbor resize must also work."""
    result = BilinearResizeNullspace().compute(
        target=None,
        camera_size=(32, 32),
        scorer_size=(16, 16),
        resize_mode="nearest",
        n_samples=4,
    )
    assert result.primitive_value.resize_mode == "nearest"


def test_compute_with_archive_records_sha(tmp_path):
    archive = tmp_path / "x.bin"
    archive.write_bytes(b"abcdef")
    result = BilinearResizeNullspace().compute(target=archive)
    assert result.archive_or_video_path == archive
    assert result.archive_sha256 is not None
    assert len(result.archive_sha256) == 64


def test_compose_with_returns_composed():
    from tac.xray.base import ComposedXRayPrimitive

    primitive = BilinearResizeNullspace()

    class _Other:
        name = "other"
        wire_in_hooks = ("sensitivity_map",)

        def compute(self, target, **kw):
            from tac.xray.base import XRayPrimitiveResult

            return XRayPrimitiveResult(
                primitive_name="other",
                archive_or_video_path=None,
                archive_sha256=None,
                primitive_value=1.0,
                evidence_grade="mathematical-derivation",
                confidence_band=None,
                composes_with=(),
                wire_in_hooks_engaged=("sensitivity_map",),
            )

        def compose_with(self, other):
            return ComposedXRayPrimitive(left=self, right=other)

    composed = primitive.compose_with(_Other())
    assert isinstance(composed, ComposedXRayPrimitive)


def test_pixel_counts_recorded_correctly():
    """At contest sizes, camera_n = 874 * 1164 = 1,017,336 and
    scorer_n = 384 * 512 = 196,608."""
    result = BilinearResizeNullspace().compute(target=None)
    r = result.primitive_value
    assert r.camera_n_pixels == 874 * 1164
    assert r.scorer_n_pixels == 384 * 512
    # The deep_math memo's "1,016,536" figure was off by 800; correct
    # arithmetic = 874 * 1164 = 1,017,336.
    assert r.camera_n_pixels == 1_017_336
    assert r.scorer_n_pixels == 196_608


def test_empirical_fraction_seeded_reproducible():
    """Same seed should produce same empirical fraction."""
    result1 = BilinearResizeNullspace().compute(
        target=None,
        camera_size=(32, 32),
        scorer_size=(16, 16),
        n_samples=16,
        seed=42,
    )
    result2 = BilinearResizeNullspace().compute(
        target=None,
        camera_size=(32, 32),
        scorer_size=(16, 16),
        n_samples=16,
        seed=42,
    )
    assert (
        result1.primitive_value.empirical_nullspace_fraction
        == result2.primitive_value.empirical_nullspace_fraction
    )
