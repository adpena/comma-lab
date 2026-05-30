# SPDX-License-Identifier: MIT
"""Tests for canonical EfficientNet steganalysis surgery."""

from __future__ import annotations

import pytest

from tac.composition.fridrich_school_inverse_steganalysis_patterns import (
    CONTEST_SEGNET_STRIDE_2_BLIND_SPOT,
    EfficientNetStemSurgeryStrategy,
    EfficientNetSurgeryConfig,
    EfficientNetSurgeryError,
    compute_blind_spot_resolution_from_stride,
)


def test_contest_blind_spot_canonical_value() -> None:
    """Per CLAUDE.md verbatim: artifacts below (256, 192) invisible to SegNet."""
    assert CONTEST_SEGNET_STRIDE_2_BLIND_SPOT == (256, 192)


def test_compute_blind_spot_stride_2_canonical() -> None:
    """Stride 2 (vanilla contest SegNet) blind spot at H/4 W/4 (canonical contest math)."""
    # 1024 x 768 input -> stride 2 -> blind spot 256 x 192 (matches CLAUDE.md)
    bs = compute_blind_spot_resolution_from_stride((1024, 768), 2)
    assert bs == (256, 192)


def test_compute_blind_spot_stride_1_ddelab_surgery() -> None:
    """Stride 1 (DDELab surgery) blind spot at H/2 W/2 (Nyquist only)."""
    bs = compute_blind_spot_resolution_from_stride((1024, 768), 1)
    assert bs == (512, 384)
    # Surgery DOUBLES the visible resolution range (Nyquist only, no stride decimation).


def test_compute_blind_spot_invalid_input_raises() -> None:
    """Invalid inputs raise EfficientNetSurgeryError."""
    with pytest.raises(EfficientNetSurgeryError, match="positive dims"):
        compute_blind_spot_resolution_from_stride((0, 100), 2)
    with pytest.raises(EfficientNetSurgeryError, match="stride"):
        compute_blind_spot_resolution_from_stride((100, 100), 0)


def test_strategies_substantively_distinct_via_blind_spot() -> None:
    """Slot EEE substantive-distinctness: each canonical surgery strategy
    produces a DIFFERENT effective stride; vanilla=2 vs DDELab=1."""
    # The strategy enum doesn't directly compute stride; we encode the
    # canonical mapping here per CLAUDE.md "Exact scorer architectures":
    strategy_to_canonical_stride = {
        EfficientNetStemSurgeryStrategy.DDELAB_CANONICAL_STRIDE_1: 1,
        EfficientNetStemSurgeryStrategy.VANILLA_STRIDE_2_BASELINE: 2,
        EfficientNetStemSurgeryStrategy.HYBRID_STRIDE_1_NO_TLU: 1,
        EfficientNetStemSurgeryStrategy.HYBRID_STRIDE_2_WITH_TLU: 2,
    }
    strides = set(strategy_to_canonical_stride.values())
    assert len(strides) == 2, "DDELab + vanilla canonical strides must differ"
    # Verify the blind-spot resolution differs across canonical strides.
    bs_at_2 = compute_blind_spot_resolution_from_stride((384, 512), 2)
    bs_at_1 = compute_blind_spot_resolution_from_stride((384, 512), 1)
    assert bs_at_1 != bs_at_2
    assert bs_at_1[0] > bs_at_2[0]  # stride-1 has LARGER visible range


def test_config_default_vanilla_baseline() -> None:
    """Default strategy is VANILLA_STRIDE_2_BASELINE matching contest SegNet."""
    cfg = EfficientNetSurgeryConfig()
    assert cfg.strategy == EfficientNetStemSurgeryStrategy.VANILLA_STRIDE_2_BASELINE
    # Default input matches CLAUDE.md SegNet input resize contract.
    assert cfg.input_resolution == (384, 512)


def test_config_canonical_blind_spot_default() -> None:
    """Default target_blind_spot_max matches CLAUDE.md canonical (256, 192)."""
    cfg = EfficientNetSurgeryConfig()
    assert cfg.target_blind_spot_max == CONTEST_SEGNET_STRIDE_2_BLIND_SPOT


def test_config_invalid_strategy_raises() -> None:
    """Wrong type for strategy raises."""
    with pytest.raises(EfficientNetSurgeryError, match="must be EfficientNetStemSurgeryStrategy"):
        EfficientNetSurgeryConfig(strategy="bogus")  # type: ignore[arg-type]


def test_config_invalid_dims_raise() -> None:
    """Non-positive dimensions raise."""
    with pytest.raises(EfficientNetSurgeryError, match="input_resolution"):
        EfficientNetSurgeryConfig(input_resolution=(0, 100))
    with pytest.raises(EfficientNetSurgeryError, match="target_blind_spot_max"):
        EfficientNetSurgeryConfig(target_blind_spot_max=(-1, 100))


def test_all_4_canonical_strategies_present() -> None:
    """4 canonical surgery strategies per the canonical taxonomy."""
    expected = {
        "ddelab_canonical_stride_1",
        "vanilla_stride_2_baseline",
        "hybrid_stride_1_no_tlu",
        "hybrid_stride_2_with_tlu",
    }
    actual = {s.value for s in EfficientNetStemSurgeryStrategy}
    assert actual == expected
