# SPDX-License-Identifier: MIT
"""Tests for ``tac.freezing.lottery_ticket_extraction``."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from tac.freezing.lottery_ticket_extraction import (
    LotteryTicketMask,
    extract_lottery_ticket,
)


def test_returns_lottery_ticket_mask():
    """Helper returns a :class:`LotteryTicketMask`."""
    m = nn.Linear(8, 4)
    out = extract_lottery_ticket(m, keep_fraction=0.5)
    assert isinstance(out, LotteryTicketMask)


def test_keep_all_fraction_one():
    """``keep_fraction=1.0`` keeps every parameter."""
    m = nn.Linear(8, 4)
    total = sum(p.numel() for p in m.parameters() if p.numel() > 0)
    out = extract_lottery_ticket(m, keep_fraction=1.0)
    assert out.total_parameters == total
    assert out.kept_parameters == total


def test_keep_half_fraction():
    """``keep_fraction=0.5`` keeps approximately half the parameters."""
    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(64, 32), nn.Linear(32, 16))
    out = extract_lottery_ticket(m, keep_fraction=0.5)
    total = out.total_parameters
    # Boundary effects allow tolerance; expect ~ total/2.
    assert abs(out.kept_parameters - total // 2) <= max(2, int(0.02 * total))


def test_keep_fraction_below_zero_rejected():
    """``keep_fraction <= 0`` raises."""
    m = nn.Linear(4, 2)
    with pytest.raises(ValueError):
        extract_lottery_ticket(m, keep_fraction=0.0)


def test_keep_fraction_above_one_rejected():
    """``keep_fraction > 1.0`` raises."""
    m = nn.Linear(4, 2)
    with pytest.raises(ValueError):
        extract_lottery_ticket(m, keep_fraction=1.5)


def test_masks_match_parameter_shapes():
    """Each mask has the same shape as the parameter it masks."""
    m = nn.Sequential(nn.Linear(8, 4), nn.Linear(4, 2))
    out = extract_lottery_ticket(m, keep_fraction=0.5)
    for name, param in m.named_parameters():
        assert name in out.masks
        assert out.masks[name].shape == param.shape


def test_masks_are_boolean():
    """Masks are boolean tensors."""
    m = nn.Linear(8, 4)
    out = extract_lottery_ticket(m, keep_fraction=0.5)
    for mask in out.masks.values():
        assert mask.dtype == torch.bool


def test_largest_magnitude_weights_are_kept():
    """The kept positions correspond to the largest-magnitude weights."""
    m = nn.Linear(4, 4, bias=False)
    with torch.no_grad():
        # Construct a known-weight pattern: [[1, 2, 3, 4], [5, 6, 7, 8], ...].
        m.weight.copy_(torch.arange(16, dtype=torch.float).reshape(4, 4))
    # Keep 4 of 16 = the top 4 magnitudes (12, 13, 14, 15).
    out = extract_lottery_ticket(m, keep_fraction=0.25)
    mask = out.masks["weight"]
    # The bottom-right 2x2 has weights 10, 11, 12, 13, 14, 15.
    # Top-4 magnitudes are 12, 13, 14, 15.
    expected = torch.tensor(
        [
            [False, False, False, False],
            [False, False, False, False],
            [False, False, False, False],
            [True, True, True, True],
        ]
    )
    assert torch.equal(mask, expected)


def test_apply_mask_zeros_pruned_weights():
    """User can apply the mask to zero the pruned weights."""
    m = nn.Linear(4, 4, bias=False)
    with torch.no_grad():
        m.weight.copy_(torch.arange(16, dtype=torch.float).reshape(4, 4))
    out = extract_lottery_ticket(m, keep_fraction=0.5)
    with torch.no_grad():
        for name, p in m.named_parameters():
            p.mul_(out.masks[name].to(p.dtype))
    # Half the entries are now zero; remainder is preserved.
    nonzero_count = int((m.weight != 0).sum().item())
    assert nonzero_count == 8


def test_empty_model_handled():
    """A model with no parameters returns an empty-mask report."""

    class _Empty(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    out = extract_lottery_ticket(_Empty(), keep_fraction=0.5)
    assert out.masks == {}
    assert out.total_parameters == 0
    assert out.kept_parameters == 0


def test_keep_fraction_stored_in_report():
    """``keep_fraction`` round-trips into the report."""
    m = nn.Linear(8, 4)
    out = extract_lottery_ticket(m, keep_fraction=0.7)
    assert out.keep_fraction == pytest.approx(0.7)
