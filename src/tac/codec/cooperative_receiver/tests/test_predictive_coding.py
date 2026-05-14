# SPDX-License-Identifier: MIT
"""Tests for the Rao-Ballard predictive-coding-hierarchy primitive."""

from __future__ import annotations

import pytest
import torch

from tac.codec.cooperative_receiver.predictive_coding import (
    PredictiveCodingOutput,
    PredictiveCodingWeights,
    predictive_coding_residual_term,
)


def test_default_weights_match_time_traveler_default() -> None:
    """Default delta_predict matches the in-tree time-traveler default."""
    w = PredictiveCodingWeights()
    assert w.delta_predict == 0.1
    assert w.residual_floor == 0.0


def test_weights_reject_negative_delta_predict() -> None:
    with pytest.raises(ValueError, match="delta_predict"):
        PredictiveCodingWeights(delta_predict=-0.5)


def test_weights_reject_negative_residual_floor() -> None:
    with pytest.raises(ValueError, match="residual_floor"):
        PredictiveCodingWeights(residual_floor=-0.01)


def test_residual_term_returns_typed_output() -> None:
    """Primitive returns the canonical typed dataclass."""
    out = predictive_coding_residual_term(torch.randn(8, 8))
    assert isinstance(out, PredictiveCodingOutput)
    assert out.scaled_term.dim() == 0
    assert out.unscaled_residual_l2.dim() == 0


def test_residual_term_constant_residual_matches_closed_form() -> None:
    """For residual = 0.5 everywhere, mean(residual^2) = 0.25 exactly."""
    residual = torch.ones(4, 4) * 0.5
    out = predictive_coding_residual_term(
        residual, weights=PredictiveCodingWeights(delta_predict=1.0)
    )
    assert torch.isclose(out.unscaled_residual_l2, torch.tensor(0.25), atol=1e-7)
    # delta_predict=1 => scaled term equals unscaled.
    assert torch.isclose(out.scaled_term, out.unscaled_residual_l2, atol=1e-7)


def test_residual_term_zero_residual_yields_zero_loss() -> None:
    """Zero residual => zero predictive-coding loss."""
    out = predictive_coding_residual_term(torch.zeros(4, 4))
    assert out.unscaled_residual_l2.item() == 0.0
    assert out.scaled_term.item() == 0.0


def test_residual_term_gradient_flows_into_residual() -> None:
    """The scaled term contributes gradient back to the residual input."""
    residual = torch.randn(4, 4, requires_grad=True)
    out = predictive_coding_residual_term(
        residual, weights=PredictiveCodingWeights(delta_predict=1.0)
    )
    out.scaled_term.backward()
    assert residual.grad is not None
    assert residual.grad.abs().sum().item() > 0


def test_residual_term_default_delta_predict_is_one_tenth_of_unscaled() -> None:
    """Default delta_predict=0.1 => scaled = 0.1 * unscaled (closed form)."""
    residual = torch.ones(2, 2) * 0.5
    out = predictive_coding_residual_term(residual)
    assert torch.isclose(out.scaled_term, 0.1 * out.unscaled_residual_l2, atol=1e-7)


def test_residual_term_zero_delta_predict_yields_zero_loss() -> None:
    """delta_predict=0 disables the primitive contribution."""
    residual = torch.randn(8, 8, requires_grad=True)
    out = predictive_coding_residual_term(
        residual, weights=PredictiveCodingWeights(delta_predict=0.0)
    )
    assert out.scaled_term.item() == 0.0


def test_residual_term_supports_arbitrary_residual_shape() -> None:
    """Residual can be any rank; mean is over ALL dimensions."""
    for shape in [(3,), (2, 3), (1, 2, 3, 4), (2, 5, 7, 7, 3)]:
        residual = torch.ones(shape) * 2.0
        out = predictive_coding_residual_term(
            residual, weights=PredictiveCodingWeights(delta_predict=1.0)
        )
        # mean(2^2) = 4 regardless of shape.
        assert torch.isclose(
            out.unscaled_residual_l2, torch.tensor(4.0), atol=1e-6
        ), f"shape={shape}"


def test_residual_term_residual_floor_clamps_small_magnitudes() -> None:
    """When residual_floor > 0, |residual| < floor is bumped up to floor."""
    residual = torch.tensor([[0.001, 0.001], [0.001, 0.001]])
    out = predictive_coding_residual_term(
        residual,
        weights=PredictiveCodingWeights(delta_predict=1.0, residual_floor=0.5),
    )
    # All |residual| = 0.001 < 0.5; clamped to 0.5; mean(0.5^2) = 0.25.
    assert torch.isclose(
        out.unscaled_residual_l2, torch.tensor(0.25), atol=1e-6
    )


def test_residual_term_residual_floor_zero_is_default_no_op() -> None:
    """residual_floor=0 (default) is exactly the unclamped formula."""
    residual = torch.tensor([[0.1, -0.1], [0.2, -0.2]])
    out = predictive_coding_residual_term(
        residual, weights=PredictiveCodingWeights(delta_predict=1.0)
    )
    # mean([0.01, 0.01, 0.04, 0.04]) = 0.025
    assert torch.isclose(
        out.unscaled_residual_l2, torch.tensor(0.025), atol=1e-7
    )


def test_residual_term_negative_residual_squares_correctly() -> None:
    """Sign is irrelevant under squaring; |residual|^2 = residual^2."""
    residual_pos = torch.ones(4) * 0.7
    residual_neg = torch.ones(4) * -0.7
    out_pos = predictive_coding_residual_term(residual_pos)
    out_neg = predictive_coding_residual_term(residual_neg)
    assert torch.isclose(
        out_pos.unscaled_residual_l2, out_neg.unscaled_residual_l2, atol=1e-7
    )


def test_residual_term_unscaled_l2_carries_gradient() -> None:
    """unscaled_residual_l2 stays attached to graph for diagnostic logging."""
    residual = torch.randn(4, requires_grad=True)
    out = predictive_coding_residual_term(residual)
    assert out.unscaled_residual_l2.requires_grad


def test_residual_term_scales_linearly_with_delta_predict() -> None:
    """scaled = delta_predict * unscaled (linearity in the weight)."""
    residual = torch.ones(2, 2) * 0.5
    out_one_tenth = predictive_coding_residual_term(
        residual, weights=PredictiveCodingWeights(delta_predict=0.1)
    )
    out_one_half = predictive_coding_residual_term(
        residual, weights=PredictiveCodingWeights(delta_predict=0.5)
    )
    # 0.5 / 0.1 = 5x.
    assert torch.isclose(
        out_one_half.scaled_term, 5.0 * out_one_tenth.scaled_term, atol=1e-7
    )


def test_residual_term_default_weights_resolve_to_pcw_defaults() -> None:
    """Passing weights=None resolves to PredictiveCodingWeights() defaults."""
    residual = torch.ones(4) * 0.5
    out_none = predictive_coding_residual_term(residual)
    out_explicit = predictive_coding_residual_term(
        residual, weights=PredictiveCodingWeights()
    )
    assert torch.isclose(out_none.scaled_term, out_explicit.scaled_term, atol=1e-7)
    assert torch.isclose(
        out_none.unscaled_residual_l2,
        out_explicit.unscaled_residual_l2,
        atol=1e-7,
    )
