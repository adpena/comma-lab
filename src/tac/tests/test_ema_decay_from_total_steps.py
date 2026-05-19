# SPDX-License-Identifier: MIT
"""Tests for ``tac.training.EMA.decay_from_total_steps`` canonical helper.

Per ARBITRARINESS-EXTINCTION audit 2026-05-18 commit 2d042f7e6 TOP-4
(``ema_decay_0.997_hardcoded_all_substrate_trainers``). Verifies the
closed-form Polyak-averaging window formula recovers the Quantizr PR101
empirical anchor ``decay=0.997`` at ``total_steps=1666`` and behaves
correctly across canonical training lengths.
"""
from __future__ import annotations

import pytest

from tac.training import EMA


# ---------------------------------------------------------------------------
# Canonical formula correctness
# ---------------------------------------------------------------------------


def test_quantizr_anchor_recovered_at_1666_steps() -> None:
    """Per Quantizr PR101 empirical anchor: decay=0.997 at ~1666 steps.

    Derivation: target_window = 0.2 * 1666 = 333.2 steps.
    decay = 1 - 1/333.2 = 0.99700 (4-decimal precision).
    """
    decay = EMA.decay_from_total_steps(1666)
    assert round(decay, 4) == 0.997


def test_decay_formula_matches_closed_form_at_canonical_values() -> None:
    """Direct verification: ``decay = 1 - 1/(0.2 * total_steps)``."""
    for total_steps in [100, 1000, 5000, 12000, 100000]:
        decay = EMA.decay_from_total_steps(total_steps)
        expected_raw = 1.0 - 1.0 / (0.2 * total_steps)
        # Apply the canonical [0.99, 0.9999] clamp.
        expected = max(0.99, min(0.9999, expected_raw))
        assert decay == pytest.approx(expected, rel=1e-12)


def test_long_training_12000_steps_yields_99958() -> None:
    """12000-step Modal A100 typical training -> decay ~ 0.99958."""
    decay = EMA.decay_from_total_steps(12000)
    assert round(decay, 5) == 0.99958


# ---------------------------------------------------------------------------
# Monotonicity invariant
# ---------------------------------------------------------------------------


def test_decay_monotone_increasing_in_total_steps() -> None:
    """As training length grows, EMA window widens -> decay -> 1.

    Strictly monotone within the [0.99, 0.9999] unclamped range.
    """
    steps_grid = [100, 500, 1000, 1666, 5000, 10000, 50000]
    decays = [EMA.decay_from_total_steps(s) for s in steps_grid]
    for d_lo, d_hi in zip(decays, decays[1:]):
        assert d_lo <= d_hi


# ---------------------------------------------------------------------------
# Clamping behavior
# ---------------------------------------------------------------------------


def test_short_training_below_clamp_returns_99_floor() -> None:
    """Very short training (raw decay < 0.99) is clamped to 0.99."""
    # target_window = 0.2 * 1 = 0.2 (clamped to 1.0 by max()).
    # raw decay = 1 - 1/1.0 = 0.0 -> clamped up to 0.99.
    decay = EMA.decay_from_total_steps(1)
    assert decay == 0.99


def test_huge_training_above_clamp_returns_9999_ceiling() -> None:
    """Very long training (raw decay > 0.9999) is clamped to 0.9999."""
    # 1e9 steps -> target_window = 2e8 -> decay = 1 - 5e-9 ~ 0.99999999.
    decay = EMA.decay_from_total_steps(1_000_000_000)
    assert decay == 0.9999


def test_canonical_range_99_to_9999_invariant() -> None:
    """Every returned decay must lie in [0.99, 0.9999]."""
    for steps in [1, 100, 1666, 12000, 100000, 1_000_000, 1_000_000_000]:
        decay = EMA.decay_from_total_steps(steps)
        assert 0.99 <= decay <= 0.9999


# ---------------------------------------------------------------------------
# target_window_fraction parameter
# ---------------------------------------------------------------------------


def test_target_window_fraction_0_3_yields_different_decay_than_0_1() -> None:
    """Different target_window_fraction values yield different decays.

    Wider window (larger fraction) -> larger target_window -> decay closer
    to 1.
    """
    decay_narrow = EMA.decay_from_total_steps(
        10000, target_window_fraction=0.1
    )
    decay_wide = EMA.decay_from_total_steps(
        10000, target_window_fraction=0.3
    )
    assert decay_narrow < decay_wide


def test_target_window_fraction_default_is_polyak_canonical_0_2() -> None:
    """The default ``0.2`` matches Polyak's canonical "last 20 percent" rule."""
    explicit_02 = EMA.decay_from_total_steps(
        1666, target_window_fraction=0.2
    )
    default = EMA.decay_from_total_steps(1666)
    assert explicit_02 == default


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------


def test_total_steps_zero_raises_value_error() -> None:
    """Zero steps is undefined."""
    with pytest.raises(ValueError, match="total_steps must be positive"):
        EMA.decay_from_total_steps(0)


def test_total_steps_negative_raises_value_error() -> None:
    """Negative steps is undefined."""
    with pytest.raises(ValueError, match="total_steps must be positive"):
        EMA.decay_from_total_steps(-100)


def test_target_window_fraction_zero_raises_value_error() -> None:
    """Zero fraction is undefined (degenerate window)."""
    with pytest.raises(ValueError, match="target_window_fraction must lie in"):
        EMA.decay_from_total_steps(1000, target_window_fraction=0.0)


def test_target_window_fraction_above_one_raises_value_error() -> None:
    """Fraction > 1 means EMA window exceeds training length."""
    with pytest.raises(ValueError, match="target_window_fraction must lie in"):
        EMA.decay_from_total_steps(1000, target_window_fraction=1.5)


def test_target_window_fraction_negative_raises_value_error() -> None:
    """Negative fraction is undefined."""
    with pytest.raises(ValueError, match="target_window_fraction must lie in"):
        EMA.decay_from_total_steps(1000, target_window_fraction=-0.2)


def test_target_window_fraction_one_is_accepted_as_edge_case() -> None:
    """``target_window_fraction=1.0`` means EMA window equals training length."""
    decay = EMA.decay_from_total_steps(1000, target_window_fraction=1.0)
    # target_window = 1.0 * 1000 = 1000. decay = 1 - 1/1000 = 0.999.
    assert decay == pytest.approx(0.999, abs=1e-12)


# ---------------------------------------------------------------------------
# Classmethod surface
# ---------------------------------------------------------------------------


def test_decay_from_total_steps_is_classmethod_no_instance_required() -> None:
    """Callable without instantiating the EMA class."""
    decay = EMA.decay_from_total_steps(1666)
    assert isinstance(decay, float)


def test_existing_ema_init_signature_preserved() -> None:
    """ADDITIVE extension: original EMA.__init__ contract unchanged."""
    import inspect

    sig = inspect.signature(EMA.__init__)
    params = list(sig.parameters.keys())
    assert params == ["self", "model", "decay"]
    # default decay remains 0.997 (the Quantizr empirical anchor)
    assert sig.parameters["decay"].default == 0.997
