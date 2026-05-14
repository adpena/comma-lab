# SPDX-License-Identifier: MIT
"""Tests for the 4 non-HNeRV residual basis scaffolds.

Covers:
* `tac.residual_basis.cool_chic_residual` (hierarchical pyramid signal)
* `tac.residual_basis.c3_residual` (conditional residual signal)
* `tac.residual_basis.siren_residual` (frequency-domain signature)
* `tac.residual_basis.coordinate_mlp_residual` (family-agnostic Laplacian)

Each scaffold gets coverage of:
* main entry-point round-trip on a synthetic gradient + noise frame
* shape / dtype / parameter contract enforcement
* promotion-status invariant refusal (score_claim must stay False)
* file-load / smoke contract
"""

from __future__ import annotations


import numpy as np
import pytest

from tac.residual_basis.c3_residual import (
    C3ResidualError,
    C3ResidualResult,
    compute_c3_residual_stats,
    compute_conditional_residual,
)
from tac.residual_basis.cool_chic_residual import (
    CoolChicPyramidLevelStats,
    CoolChicResidualError,
    CoolChicResidualResult,
    compute_cool_chic_residual_stats,
    compute_pyramid_residual,
)
from tac.residual_basis.coordinate_mlp_residual import (
    CoordinateMlpResidualError,
    CoordinateMlpResidualResult,
    compute_coordinate_mlp_residual_stats,
    compute_finite_difference_laplacian,
)
from tac.residual_basis.siren_residual import (
    SirenResidualError,
    SirenResidualResult,
    compute_radial_frequency_buckets,
    compute_siren_residual_stats,
)

H_SMALL = 32
W_SMALL = 48


def _synthetic_frames(n_frames: int = 4, h: int = H_SMALL, w: int = W_SMALL) -> np.ndarray:
    """Build a small synthetic (T, H, W, 3) uint8 RGB stream."""
    rng = np.random.default_rng(seed=0x5EED)
    base = np.linspace(0, 255, w, dtype=np.float64)[None, :, None]  # gradient
    base = np.broadcast_to(base, (h, w, 3))  # type: ignore[assignment]
    frames = np.empty((n_frames, h, w, 3), dtype=np.uint8)
    for t in range(n_frames):
        noisy = base + rng.normal(0, 8, size=(h, w, 3))
        frames[t] = np.clip(noisy, 0, 255).astype(np.uint8)
    return frames


# ---------------------------------------------------------------------------
# Cool-Chic scaffold
# ---------------------------------------------------------------------------


def test_cool_chic_returns_typed_result_with_frozen_promotion_status() -> None:
    frames = _synthetic_frames()
    result = compute_cool_chic_residual_stats(frames, pyramid_levels=3)
    assert isinstance(result, CoolChicResidualResult)
    assert result.score_claim is False
    assert result.promotion_eligible is False
    assert result.ready_for_exact_eval_dispatch is False
    assert result.evidence_grade == "research_signal"
    assert result.schema.startswith("cool_chic_residual_pr106_scaffold")
    assert result.n_frames == frames.shape[0]
    assert len(result.per_level_stats) == 3


def test_cool_chic_per_level_stats_shape_halves() -> None:
    frames = _synthetic_frames(h=16, w=24)
    result = compute_cool_chic_residual_stats(frames, pyramid_levels=3)
    # Level 0: full resolution. Level 1: ~half. Level 2: ~quarter.
    h0 = result.per_level_stats[0].height
    h1 = result.per_level_stats[1].height
    h2 = result.per_level_stats[2].height
    assert h0 == 16
    assert h1 == 8
    assert h2 == 4


def test_cool_chic_zero_input_zero_stats() -> None:
    frames = np.zeros((2, 8, 8, 3), dtype=np.uint8)
    result = compute_cool_chic_residual_stats(frames, pyramid_levels=2)
    for stats in result.per_level_stats:
        assert stats.abs_mean == 0.0
        # 100% sparse (every value < epsilon)
        assert stats.sparsity_fraction == pytest.approx(1.0)


def test_cool_chic_bad_shape_raises() -> None:
    with pytest.raises(CoolChicResidualError, match="expected"):
        compute_cool_chic_residual_stats(np.zeros((4, 4, 4), dtype=np.uint8))


def test_cool_chic_bad_dtype_raises() -> None:
    with pytest.raises(CoolChicResidualError, match="expected dtype"):
        compute_cool_chic_residual_stats(np.zeros((2, 8, 8, 3), dtype=np.int32))


def test_cool_chic_pyramid_helper_returns_correct_levels() -> None:
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    pyramid = compute_pyramid_residual(frame, levels=3)
    assert len(pyramid) == 3
    assert pyramid[0].shape == (3, 16, 16)
    assert pyramid[1].shape == (3, 8, 8)
    assert pyramid[2].shape == (3, 4, 4)


def test_cool_chic_promotion_tampering_refused() -> None:
    """Direct dataclass construction can't bypass frozen promotion status."""
    valid_stat = CoolChicPyramidLevelStats(
        level=0,
        height=8,
        width=8,
        n_coefficients=192,
        abs_mean=1.0,
        abs_std=0.5,
        abs_max=2.0,
        sparsity_fraction=0.1,
        entropy_bits=4.0,
    )
    # We cannot assign score_claim=True because it's init=False (always False).
    result = CoolChicResidualResult(
        pyramid_levels=1,
        n_frames=1,
        n_channels=3,
        height=8,
        width=8,
        per_level_stats=(valid_stat,),
    )
    assert result.score_claim is False
    # Cannot mutate frozen dataclass.
    with pytest.raises((AttributeError, TypeError)):
        result.score_claim = True  # type: ignore[misc]


# ---------------------------------------------------------------------------
# C3 scaffold
# ---------------------------------------------------------------------------


def test_c3_returns_typed_result_frame_delta_mode() -> None:
    frames = _synthetic_frames()
    result = compute_c3_residual_stats(frames, conditioning_mode="frame_delta")
    assert isinstance(result, C3ResidualResult)
    assert result.score_claim is False
    assert result.promotion_eligible is False
    assert result.stats.conditioning_mode == "frame_delta"


def test_c3_mean_baseline_mode() -> None:
    frames = _synthetic_frames()
    result = compute_c3_residual_stats(frames, conditioning_mode="mean_baseline")
    assert result.stats.conditioning_mode == "mean_baseline"


def test_c3_frame_delta_requires_at_least_2_frames() -> None:
    frames = _synthetic_frames(n_frames=1)
    with pytest.raises(C3ResidualError, match="requires >= 2 frames"):
        compute_c3_residual_stats(frames, conditioning_mode="frame_delta")


def test_c3_unknown_conditioning_mode_raises() -> None:
    frames = _synthetic_frames()
    with pytest.raises(C3ResidualError, match="unknown conditioning_mode"):
        compute_c3_residual_stats(frames, conditioning_mode="not_real")


def test_c3_zero_input_zero_residual() -> None:
    frames = np.zeros((4, 8, 8, 3), dtype=np.uint8)
    result = compute_c3_residual_stats(frames, conditioning_mode="frame_delta")
    assert result.stats.abs_mean == 0.0
    assert result.stats.sparsity_fraction == pytest.approx(1.0)


def test_c3_conditional_residual_helper() -> None:
    frames = np.zeros((3, 4, 4, 3), dtype=np.uint8)
    frames[1] = 10  # delta = 10 at t=1
    frames[2] = 5  # delta = -5 at t=2
    residual = compute_conditional_residual(frames, conditioning_mode="frame_delta")
    assert residual.shape == (2, 4, 4, 3)  # T-1 deltas
    assert residual[0].mean() == 10.0
    assert residual[1].mean() == -5.0


def test_c3_promotion_invariants() -> None:
    frames = _synthetic_frames()
    result = compute_c3_residual_stats(frames)
    assert result.evidence_grade == "research_signal"
    assert result.schema.startswith("c3_residual_pr106_scaffold")
    with pytest.raises((AttributeError, TypeError)):
        result.score_claim = True  # type: ignore[misc]


# ---------------------------------------------------------------------------
# SIREN scaffold
# ---------------------------------------------------------------------------


def test_siren_returns_typed_result_with_3_bands() -> None:
    frames = _synthetic_frames()
    result = compute_siren_residual_stats(frames)
    assert isinstance(result, SirenResidualResult)
    assert result.score_claim is False
    assert len(result.per_band_stats) == 3
    band_names = [s.band_name for s in result.per_band_stats]
    assert band_names == ["low", "mid", "high"]


def test_siren_energy_fractions_sum_to_approximately_one() -> None:
    frames = _synthetic_frames()
    result = compute_siren_residual_stats(frames)
    total = sum(s.energy_fraction for s in result.per_band_stats)
    assert 0.95 <= total <= 1.05


def test_siren_radial_band_helper_returns_three_bands() -> None:
    frame = _synthetic_frames(n_frames=1)[0]
    buckets = compute_radial_frequency_buckets(frame)
    assert set(buckets.keys()) == {"low", "mid", "high"}
    for arr in buckets.values():
        assert arr.shape == frame.shape[:2]


def test_siren_bad_shape_raises() -> None:
    with pytest.raises(SirenResidualError, match="expected"):
        compute_siren_residual_stats(np.zeros((4, 4), dtype=np.uint8))


def test_siren_low_freq_dominant_for_smooth_signal() -> None:
    """A smooth gradient should have most energy in the LOW frequency band."""
    h, w = 32, 32
    rng = np.random.default_rng(seed=0)
    # Pure low-frequency gradient (no high-frequency content).
    gradient = np.linspace(0, 255, w, dtype=np.float64)[None, :, None]
    smooth = np.broadcast_to(gradient, (h, w, 3)).astype(np.uint8)
    frames = np.stack([smooth] * 2)
    result = compute_siren_residual_stats(frames)
    by_band = {s.band_name: s.energy_fraction for s in result.per_band_stats}
    # Low band should dominate (>50%) for a smooth gradient.
    assert by_band["low"] > 0.5


def test_siren_promotion_invariants() -> None:
    frames = _synthetic_frames()
    result = compute_siren_residual_stats(frames)
    assert result.evidence_grade == "research_signal"
    with pytest.raises((AttributeError, TypeError)):
        result.score_claim = True  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Coordinate-MLP (family-agnostic Laplacian) scaffold
# ---------------------------------------------------------------------------


def test_coordinate_mlp_returns_typed_result() -> None:
    frames = _synthetic_frames()
    result = compute_coordinate_mlp_residual_stats(frames)
    assert isinstance(result, CoordinateMlpResidualResult)
    assert result.score_claim is False
    assert result.promotion_eligible is False
    assert result.stats.smoothness_fraction >= 0.0
    assert result.stats.smoothness_fraction <= 1.0


def test_coordinate_mlp_zero_input_perfectly_smooth() -> None:
    frames = np.zeros((2, 8, 8, 3), dtype=np.uint8)
    result = compute_coordinate_mlp_residual_stats(frames)
    # Laplacian of zero is zero -> 100% smooth.
    assert result.stats.smoothness_fraction == pytest.approx(1.0)
    assert result.stats.abs_mean == 0.0


def test_coordinate_mlp_constant_input_perfectly_smooth() -> None:
    """Constant input has zero Laplacian -> 100% smooth."""
    frames = np.full((2, 8, 8, 3), 127, dtype=np.uint8)
    result = compute_coordinate_mlp_residual_stats(frames)
    assert result.stats.smoothness_fraction == pytest.approx(1.0)


def test_coordinate_mlp_laplacian_helper_shape() -> None:
    frame = _synthetic_frames(n_frames=1)[0]
    lap = compute_finite_difference_laplacian(frame)
    assert lap.shape == (3, frame.shape[0], frame.shape[1])
    assert lap.dtype == np.float64


def test_coordinate_mlp_laplacian_impulse_response() -> None:
    """Single-pixel impulse has known Laplacian (5-point stencil): -4 at center,
    +1 at each of 4 neighbors. The boundary replication then sweeps the +1
    over the corner-adjacent edge pixels but the interior 3x3 around the
    impulse is exactly the 5-point stencil."""

    frame = np.zeros((5, 5, 3), dtype=np.float64)
    frame[2, 2] = 1.0  # center impulse, single channel won't trip — broadcast
    lap = compute_finite_difference_laplacian(frame.astype(np.uint8))
    # Center 5-point stencil for an impulse: f(x-1,y) + f(x+1,y) + f(x,y-1) + f(x,y+1) - 4*f(x,y).
    # impulse at (2,2): 0 + 0 + 0 + 0 - 4*1 = -4 (truncated to int8 in conversion, but float64 internally).
    # Note: frame.astype(np.uint8) truncates 1.0 -> 1, so center=1.
    assert lap[0, 2, 2] == -4.0  # interior pixel
    assert lap[0, 1, 2] == 1.0  # neighbor
    assert lap[0, 3, 2] == 1.0  # neighbor
    assert lap[0, 2, 1] == 1.0  # neighbor
    assert lap[0, 2, 3] == 1.0  # neighbor


def test_coordinate_mlp_bad_shape_raises() -> None:
    with pytest.raises(CoordinateMlpResidualError, match="expected"):
        compute_coordinate_mlp_residual_stats(np.zeros((4, 4), dtype=np.uint8))


def test_coordinate_mlp_promotion_invariants() -> None:
    frames = _synthetic_frames()
    result = compute_coordinate_mlp_residual_stats(frames)
    assert result.evidence_grade == "research_signal"
    assert result.schema.startswith("coordinate_mlp_residual_pr106_scaffold")
    with pytest.raises((AttributeError, TypeError)):
        result.score_claim = True  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Cross-scaffold smoke: all 4 work on the same input
# ---------------------------------------------------------------------------


def test_all_four_scaffolds_run_on_same_input() -> None:
    frames = _synthetic_frames()
    cc_res = compute_cool_chic_residual_stats(frames, pyramid_levels=2)
    c3_res = compute_c3_residual_stats(frames, conditioning_mode="frame_delta")
    siren_res = compute_siren_residual_stats(frames)
    coord_res = compute_coordinate_mlp_residual_stats(frames)
    # All four return non-empty results with frozen promotion-status.
    assert cc_res.score_claim is False
    assert c3_res.score_claim is False
    assert siren_res.score_claim is False
    assert coord_res.score_claim is False


def test_all_four_scaffolds_have_research_signal_evidence_grade() -> None:
    frames = _synthetic_frames()
    for fn in (
        compute_cool_chic_residual_stats,
        compute_c3_residual_stats,
        compute_siren_residual_stats,
        compute_coordinate_mlp_residual_stats,
    ):
        result = fn(frames)  # type: ignore[arg-type]
        assert result.evidence_grade == "research_signal"


def test_all_four_scaffolds_refuse_negative_n_frames() -> None:
    """0-frame input is shape-rejected by all 4 scaffolds."""
    empty = np.zeros((0, 8, 8, 3), dtype=np.uint8)
    for fn, err in (
        (compute_cool_chic_residual_stats, CoolChicResidualError),
        (compute_c3_residual_stats, C3ResidualError),
        (compute_siren_residual_stats, SirenResidualError),
        (compute_coordinate_mlp_residual_stats, CoordinateMlpResidualError),
    ):
        with pytest.raises(err):
            fn(empty)  # type: ignore[arg-type]
