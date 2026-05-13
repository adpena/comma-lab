"""Tests for `tac.residual_basis.saliency_masked_residual`.

Per W's DEFERRED reactivation criterion #2 + N D2 council verdict + Catalog
#123 (`check_no_weight_domain_saliency_on_score_gradient_substrate`) +
CLAUDE.md "Bugs must be permanently fixed AND self-protected against".
"""

from __future__ import annotations


import pytest
import torch

from tac.residual_basis.hinton_distilled_scorer_surrogate import (
    ScorerSurrogateConfig,
    load_pretrained_distilled_scorer_pair,
)
from tac.residual_basis.saliency_masked_residual import (
    DEFAULT_SALIENCY_PERCENTILE,
    DEFAULT_SALIENCY_THRESHOLD,
    SaliencyMaskedResidualError,
    SaliencyMaskingConfig,
    compose_saliency_with_threshold_mask,
    compute_score_aware_saliency,
    mask_residual_by_saliency,
)


# ---------------------------------------------------------------------------
# SaliencyMaskingConfig
# ---------------------------------------------------------------------------


def test_saliency_masking_config_council_canonical():
    config = SaliencyMaskingConfig.council_canonical()
    assert config.threshold == DEFAULT_SALIENCY_THRESHOLD
    assert config.percentile == DEFAULT_SALIENCY_PERCENTILE
    assert config.minimum_kept_fraction == 0.01
    assert config.per_channel_aggregation == "max"


def test_saliency_masking_config_rejects_invalid_threshold():
    for bad in (-1.0, float("nan"), float("inf")):
        with pytest.raises(SaliencyMaskedResidualError, match="threshold"):
            SaliencyMaskingConfig(
                threshold=bad,
                percentile=None,
                minimum_kept_fraction=0.01,
                per_channel_aggregation="max",
            )


def test_saliency_masking_config_rejects_invalid_percentile():
    for bad in (-0.1, 1.0, 1.5, float("nan")):
        with pytest.raises(SaliencyMaskedResidualError, match="percentile"):
            SaliencyMaskingConfig(
                threshold=0.0,
                percentile=bad,
                minimum_kept_fraction=0.01,
                per_channel_aggregation="max",
            )


def test_saliency_masking_config_rejects_invalid_min_kept_fraction():
    for bad in (-0.1, 0.0, 1.5):
        with pytest.raises(SaliencyMaskedResidualError, match="minimum_kept_fraction"):
            SaliencyMaskingConfig(
                threshold=0.0,
                percentile=None,
                minimum_kept_fraction=bad,
                per_channel_aggregation="max",
            )


def test_saliency_masking_config_rejects_invalid_aggregation():
    with pytest.raises(SaliencyMaskedResidualError, match="per_channel_aggregation"):
        SaliencyMaskingConfig(
            threshold=0.0,
            percentile=None,
            minimum_kept_fraction=0.01,
            per_channel_aggregation="median",  # not allowed
        )


def test_saliency_masking_config_is_frozen():
    config = SaliencyMaskingConfig.council_canonical()
    with pytest.raises(Exception):
        config.threshold = 1.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# compute_score_aware_saliency
# ---------------------------------------------------------------------------


def _make_input_pair(B=2, H=64, W=96):
    return torch.rand(B, 3, H, W) * 200.0 + 28.0


def test_compute_saliency_returns_non_negative_per_pixel_map():
    """Saliency is a gradient-norm — must be non-negative."""
    config = ScorerSurrogateConfig.council_canonical()
    seg, pose = load_pretrained_distilled_scorer_pair(config=config)
    decoded = _make_input_pair()
    gt = _make_input_pair()
    saliency = compute_score_aware_saliency(
        decoded, gt,
        distilled_segnet=seg, distilled_posenet=pose,
        eval_roundtrip=False,
    )
    assert saliency.shape == (decoded.shape[0], decoded.shape[2], decoded.shape[3])
    assert (saliency >= 0).all(), "saliency must be non-negative"
    assert torch.isfinite(saliency).all()


def test_compute_saliency_is_score_aware_not_weight_domain():
    """Per Catalog #123: saliency is computed via INPUT-SPACE gradient.

    The function returns the gradient norm AT INPUT pixels — NOT a function
    of weight magnitudes. This is what makes it "score-aware" rather than
    "weight-domain" (the Track 4 v1 anti-pattern).
    """
    config = ScorerSurrogateConfig.council_canonical()
    seg, pose = load_pretrained_distilled_scorer_pair(config=config)
    decoded = _make_input_pair()
    gt = _make_input_pair()
    sal_a = compute_score_aware_saliency(
        decoded.clone(), gt.clone(),
        distilled_segnet=seg, distilled_posenet=pose,
        eval_roundtrip=False,
    )
    # Different decoded → different saliency (score-aware: depends on input).
    decoded2 = _make_input_pair()
    sal_b = compute_score_aware_saliency(
        decoded2, gt.clone(),
        distilled_segnet=seg, distilled_posenet=pose,
        eval_roundtrip=False,
    )
    # Same input → same saliency (deterministic).
    sal_a_again = compute_score_aware_saliency(
        decoded.clone(), gt.clone(),
        distilled_segnet=seg, distilled_posenet=pose,
        eval_roundtrip=False,
    )
    assert torch.allclose(sal_a, sal_a_again, atol=1e-6), (
        "same input must produce same saliency"
    )
    # Different inputs likely → different saliency
    assert not torch.allclose(sal_a, sal_b, atol=1e-6), (
        "different inputs should produce different saliency (score-aware)"
    )


def test_compute_saliency_rejects_odd_batch():
    config = ScorerSurrogateConfig.council_canonical()
    seg, pose = load_pretrained_distilled_scorer_pair(config=config)
    decoded = _make_input_pair(B=3)
    gt = _make_input_pair(B=3)
    with pytest.raises(SaliencyMaskedResidualError, match="even"):
        compute_score_aware_saliency(
            decoded, gt,
            distilled_segnet=seg, distilled_posenet=pose,
        )


def test_compute_saliency_rejects_shape_mismatch():
    config = ScorerSurrogateConfig.council_canonical()
    seg, pose = load_pretrained_distilled_scorer_pair(config=config)
    decoded = _make_input_pair(B=2, H=64, W=96)
    gt = _make_input_pair(B=2, H=128, W=128)
    with pytest.raises(SaliencyMaskedResidualError, match="shape mismatch"):
        compute_score_aware_saliency(
            decoded, gt,
            distilled_segnet=seg, distilled_posenet=pose,
        )


# ---------------------------------------------------------------------------
# mask_residual_by_saliency
# ---------------------------------------------------------------------------


def test_mask_residual_by_saliency_zeros_low_saliency_pixels():
    config = SaliencyMaskingConfig.council_canonical()
    # Use a simple synthetic saliency: half pixels are high, half are low.
    B, H, W = 2, 16, 24
    saliency = torch.zeros(B, H, W)
    saliency[:, :H // 2, :] = 1.0  # top half high; bottom half zero
    residual = torch.randn(B, H, W, 3)
    masked, diag = mask_residual_by_saliency(residual, saliency, config=config)
    assert masked.shape == residual.shape
    # Top half should be unchanged; bottom half zeroed.
    assert torch.allclose(masked[:, :H // 2, :, :], residual[:, :H // 2, :, :])
    assert torch.all(masked[:, H // 2:, :, :] == 0)
    # kept_fraction reflects 50% (top half).
    assert abs(diag["saliency_kept_fraction_overall"] - 0.5) < 0.01


def test_mask_residual_by_saliency_supports_bchw_layout():
    """Both (B, H, W, C) and (B, C, H, W) layouts are accepted."""
    config = SaliencyMaskingConfig.council_canonical()
    B, H, W = 2, 16, 24
    saliency = torch.rand(B, H, W)
    residual_bchw = torch.randn(B, 3, H, W)
    masked_bchw, _ = mask_residual_by_saliency(residual_bchw, saliency, config=config)
    assert masked_bchw.shape == (B, 3, H, W)
    residual_bhwc = torch.randn(B, H, W, 3)
    masked_bhwc, _ = mask_residual_by_saliency(residual_bhwc, saliency, config=config)
    assert masked_bhwc.shape == (B, H, W, 3)


def test_mask_residual_by_saliency_threshold_mode():
    config = SaliencyMaskingConfig(
        threshold=0.5,
        percentile=None,
        minimum_kept_fraction=0.01,
        per_channel_aggregation="max",
    )
    B, H, W = 2, 16, 24
    saliency = torch.full((B, H, W), 0.7)
    saliency[:, :H // 4, :] = 0.3  # low saliency in top quarter
    residual = torch.randn(B, H, W, 3)
    masked, diag = mask_residual_by_saliency(residual, saliency, config=config)
    # Top quarter should be zeroed (saliency 0.3 < 0.5).
    assert torch.all(masked[:, :H // 4, :, :] == 0)
    assert torch.allclose(masked[:, H // 4:, :, :], residual[:, H // 4:, :, :])
    expected_kept = 1.0 - 0.25
    assert abs(diag["saliency_kept_fraction_overall"] - expected_kept) < 0.01


def test_mask_residual_by_saliency_refuses_too_aggressive_mask():
    """When mask would zero > (1 - min_kept_fraction) of pixels, it refuses."""
    config = SaliencyMaskingConfig(
        threshold=0.99,
        percentile=None,
        minimum_kept_fraction=0.5,  # demand 50% kept
        per_channel_aggregation="max",
    )
    B, H, W = 2, 16, 24
    saliency = torch.full((B, H, W), 0.1)  # ALL below threshold
    residual = torch.randn(B, H, W, 3)
    with pytest.raises(SaliencyMaskedResidualError, match="kept_fraction"):
        mask_residual_by_saliency(residual, saliency, config=config)


def test_mask_residual_by_saliency_rejects_shape_mismatch():
    config = SaliencyMaskingConfig.council_canonical()
    saliency = torch.rand(2, 16, 24)
    residual = torch.randn(2, 32, 32, 3)  # different spatial
    with pytest.raises(SaliencyMaskedResidualError, match="shape"):
        mask_residual_by_saliency(residual, saliency, config=config)


def test_mask_residual_rejects_invalid_residual_dim():
    config = SaliencyMaskingConfig.council_canonical()
    saliency = torch.rand(2, 16, 24)
    residual = torch.randn(2, 3, 16, 24, 5)  # 5-dim
    with pytest.raises(SaliencyMaskedResidualError, match="4-dim"):
        mask_residual_by_saliency(residual, saliency, config=config)


def test_mask_residual_rejects_non_rgb_channel_dim():
    config = SaliencyMaskingConfig.council_canonical()
    saliency = torch.rand(2, 16, 24)
    residual = torch.randn(2, 16, 24, 5)  # 5 channels (not 3)
    with pytest.raises(SaliencyMaskedResidualError, match="RGB channel dim"):
        mask_residual_by_saliency(residual, saliency, config=config)


def test_mask_residual_rejects_non_config():
    saliency = torch.rand(2, 16, 24)
    residual = torch.randn(2, 16, 24, 3)
    with pytest.raises(SaliencyMaskedResidualError, match="SaliencyMaskingConfig"):
        mask_residual_by_saliency(residual, saliency, config="not_a_config")


# ---------------------------------------------------------------------------
# compose_saliency_with_threshold_mask (composition with sparse PacketIR)
# ---------------------------------------------------------------------------


def test_compose_saliency_with_threshold_two_masks_compose_multiplicatively():
    """High-saliency AND high-magnitude pixels are kept; others zeroed."""
    config = SaliencyMaskingConfig(
        threshold=0.5,
        percentile=None,
        minimum_kept_fraction=0.01,
        per_channel_aggregation="max",
    )
    B, H, W = 2, 16, 24
    saliency = torch.full((B, H, W), 0.7)  # all above threshold
    residual = torch.randn(B, H, W, 3)
    composed, diag = compose_saliency_with_threshold_mask(
        residual, saliency,
        saliency_config=config,
        magnitude_threshold=0.5,
    )
    # Saliency mask passes everything; magnitude mask zeros |x| < 0.5.
    expected = residual.clone()
    expected[expected.abs() < 0.5] = 0
    assert torch.allclose(composed, expected)


def test_compose_saliency_rejects_invalid_magnitude_threshold():
    config = SaliencyMaskingConfig.council_canonical()
    B, H, W = 2, 16, 24
    saliency = torch.rand(B, H, W)
    residual = torch.randn(B, H, W, 3)
    for bad in (-1.0, float("nan"), float("inf")):
        with pytest.raises(SaliencyMaskedResidualError, match="magnitude_threshold"):
            compose_saliency_with_threshold_mask(
                residual, saliency,
                saliency_config=config,
                magnitude_threshold=bad,
            )


def test_compose_saliency_diagnostics_include_both_kept_fractions():
    config = SaliencyMaskingConfig(
        threshold=0.0,
        percentile=None,
        minimum_kept_fraction=0.01,
        per_channel_aggregation="max",
    )
    B, H, W = 2, 16, 24
    saliency = torch.full((B, H, W), 1.0)
    residual = torch.randn(B, H, W, 3)
    _, diag = compose_saliency_with_threshold_mask(
        residual, saliency,
        saliency_config=config,
        magnitude_threshold=0.5,
    )
    assert "saliency_kept_fraction_overall" in diag
    assert "magnitude_n_zeroed_coefficients" in diag
    assert "magnitude_n_total_coefficients" in diag
    assert "composed_kept_fraction" in diag
    assert diag["magnitude_threshold"] == 0.5
