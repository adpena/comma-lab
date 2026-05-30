# SPDX-License-Identifier: MIT
"""Tests for canonical ALASKA color-separation pattern.

Per CLAUDE.md "NO FAKE IMPLEMENTATIONS" non-negotiable + Slot EEE
substantive-distinctness rule: every test verifies BEHAVIOR not just
constants. Jaccard < 1.0 across alternatives is the canonical
substantive-distinctness gate.
"""

from __future__ import annotations

import pytest

from tac.composition.alaska_inverse_steganalysis_patterns import (
    ColorBranchStrategy,
    ColorBranchSliceStrategy,
    SRNET_BRANCH_ORDER,
    YUV6_CHANNEL_LAYOUT,
    branch_to_yuv6_channel_slice,
)


def test_srnet_branch_order_matches_yousfi_upstream() -> None:
    """``SRNET_BRANCH_ORDER`` must be 1:1 with Yousfi's ``models.py:34``.

    Upstream verbatim: ``for branch in ['YCrCb', 'CrCb', 'Y', 'Cr', 'Cb']:``
    """
    assert SRNET_BRANCH_ORDER == ("YCrCb", "CrCb", "Y", "Cr", "Cb")


def test_color_branch_strategy_enum_5_values() -> None:
    """5 enum values, 1:1 with Yousfi's 5 branches."""
    assert {b.value for b in ColorBranchStrategy} == {
        "YCrCb",
        "CrCb",
        "Y",
        "Cr",
        "Cb",
    }


def test_yuv6_channel_layout_substantively_distinct() -> None:
    """Each named slice MUST produce a structurally distinct channel-subset.

    Slot EEE substantive-distinctness gate: if any 2 named slices produce
    identical channel-indices, the enum value is structurally padded
    (canonical Catalog #308 alternative-probe-methodology bug class).
    """
    slices = list(YUV6_CHANNEL_LAYOUT.items())
    n_distinct = len({tuple(v) for _, v in slices})
    n_total = len(slices)
    # ALL 11 named slices MUST be distinct.
    assert n_distinct == n_total, (
        f"YUV6_CHANNEL_LAYOUT has duplicate slices: "
        f"{n_distinct} distinct / {n_total} total"
    )


def test_yuv6_channel_layout_11_canonical_slices() -> None:
    """11 named slices per the canonical taxonomy."""
    expected = {
        "Y0",
        "Y1",
        "Y2",
        "Y3",
        "U",
        "V",
        "Y_only",
        "UV_only",
        "YUV6_full",
        "Y0_UV",
        "Y123_UV",
    }
    assert set(YUV6_CHANNEL_LAYOUT.keys()) == expected


def test_yuv6_channel_layout_full_contains_all_six() -> None:
    """``YUV6_full`` MUST contain all 6 channel indices."""
    assert tuple(YUV6_CHANNEL_LAYOUT["YUV6_full"]) == (0, 1, 2, 3, 4, 5)


def test_yuv6_channel_layout_y_only_excludes_uv() -> None:
    """``Y_only`` MUST contain Y0..Y3 and exclude U/V."""
    assert tuple(YUV6_CHANNEL_LAYOUT["Y_only"]) == (0, 1, 2, 3)


def test_yuv6_channel_layout_uv_only_excludes_y() -> None:
    """``UV_only`` MUST contain U/V and exclude Y0..Y3."""
    assert tuple(YUV6_CHANNEL_LAYOUT["UV_only"]) == (4, 5)


def test_branch_to_yuv6_channel_slice_enum() -> None:
    """Helper accepts :class:`ColorBranchSliceStrategy` directly."""
    assert branch_to_yuv6_channel_slice(ColorBranchSliceStrategy.Y0_UV) == (0, 4, 5)
    assert branch_to_yuv6_channel_slice(ColorBranchSliceStrategy.YUV6_FULL) == (
        0,
        1,
        2,
        3,
        4,
        5,
    )


def test_branch_to_yuv6_channel_slice_str() -> None:
    """Helper accepts plain str for ergonomics."""
    assert branch_to_yuv6_channel_slice("Y_only") == (0, 1, 2, 3)


def test_branch_to_yuv6_channel_slice_raises_on_unknown() -> None:
    """Unknown branch name raises KeyError with valid-list in message."""
    with pytest.raises(KeyError, match="not in canonical YUV6 layout"):
        branch_to_yuv6_channel_slice("BOGUS_BRANCH")
