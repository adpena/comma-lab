# SPDX-License-Identifier: MIT
"""Tests for Row #7 — Prechelt 1998 early stopping."""
from __future__ import annotations

import pytest

from tac.formula_extinctions.early_stopping_prechelt_slope import (
    EarlyStoppingInput,
    canonical_early_stopping_patience,
)


def test_decreasing_loss_no_stop():
    """Decreasing val loss -> continue + counter resets."""
    hist = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.12]
    r = canonical_early_stopping_patience(
        EarlyStoppingInput(val_loss_history=hist, window_size=10, patience_count=3)
    )
    stop_now, counter = r.solved_value
    assert stop_now is False
    assert counter == 0


def test_flat_loss_patience_increments():
    """Flat val loss -> patience counter increments."""
    hist = [0.1] * 12
    r = canonical_early_stopping_patience(
        EarlyStoppingInput(val_loss_history=hist, window_size=10, patience_count=3)
    )
    stop_now, counter = r.solved_value
    assert stop_now is False
    assert counter == 1


def test_patience_threshold_triggers_stop():
    """After K consecutive flat windows, stop_now=True."""
    hist = [0.1] * 12
    r = canonical_early_stopping_patience(
        EarlyStoppingInput(
            val_loss_history=hist, window_size=10, patience_count=3, patience_counter=2,
        )
    )
    stop_now, counter = r.solved_value
    assert stop_now is True
    assert counter == 3


def test_history_too_short_continues():
    """history shorter than window+1 -> never stop."""
    hist = [0.1, 0.09, 0.08]
    r = canonical_early_stopping_patience(
        EarlyStoppingInput(val_loss_history=hist, window_size=10, patience_count=3)
    )
    stop_now, _ = r.solved_value
    assert stop_now is False
    assert r.intermediate_values["reason"] == "history_too_short"


def test_invalid_inputs_raise():
    """Bad params raise."""
    with pytest.raises(ValueError, match="window_size"):
        EarlyStoppingInput(val_loss_history=[0.1] * 12, window_size=1)
    with pytest.raises(ValueError, match="patience_count"):
        EarlyStoppingInput(val_loss_history=[0.1] * 12, patience_count=0)
    with pytest.raises(ValueError, match="slope_epsilon"):
        EarlyStoppingInput(val_loss_history=[0.1] * 12, slope_epsilon=0.0)


def test_citation_prechelt():
    """Prechelt 1998 citation."""
    r = canonical_early_stopping_patience(
        EarlyStoppingInput(val_loss_history=[0.1] * 12)
    )
    assert "Prechelt 1998" in r.literature_citation
